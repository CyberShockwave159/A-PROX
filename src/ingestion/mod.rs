pub mod parser;
pub mod watcher;

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
use walkdir::WalkDir;

use crate::config::IngestionConfig;
use crate::db::VectorStore;
use crate::rag::RagEngine;

/// Metadata about a single indexed file
#[derive(Debug, Clone)]
pub struct IndexedFileInfo {
    pub path: String,
    pub collection: String,
    pub chunks_count: usize,
    pub content_hash: String,
    pub last_indexed: u64,
}

/// Results from an ingestion run
#[derive(Debug, Clone)]
pub struct IngestionResult {
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub files_errors: usize,
    pub chunks_total: usize,
}

/// Manages directory-based file ingestion into the RAG vector store
pub struct DirectoryIngestor {
    config: IngestionConfig,
    vector_store: Arc<VectorStore>,
    rag_engine: Arc<RagEngine>,
    http_client: reqwest::Client,
    upstream_base_url: String,
    upstream_api_key: String,
    upstream_model: String,
    active_directories: Arc<Mutex<HashSet<String>>>,
    stats: Arc<Mutex<IngestionResult>>,
}

impl DirectoryIngestor {
    pub fn new(
        config: IngestionConfig,
        vector_store: Arc<VectorStore>,
        rag_engine: Arc<RagEngine>,
        upstream_base_url: String,
        upstream_api_key: String,
        upstream_model: String,
    ) -> Self {
        Self {
            config,
            vector_store,
            rag_engine,
            http_client: reqwest::Client::new(),
            upstream_base_url,
            upstream_api_key,
            upstream_model,
            active_directories: Arc::new(Mutex::new(HashSet::new())),
            stats: Arc::new(Mutex::new(IngestionResult {
                files_indexed: 0,
                files_skipped: 0,
                files_errors: 0,
                chunks_total: 0,
            })),
        }
    }

    /// Run initial ingestion across all configured directories
    pub async fn run_initial_ingestion(&self) -> IngestionResult {
        let mut result = IngestionResult {
            files_indexed: 0,
            files_skipped: 0,
            files_errors: 0,
            chunks_total: 0,
        };

        let directories = self.config.directories.clone();
        if directories.is_empty() {
            tracing::info!("No ingestion directories configured");
            return result;
        }

        let allowed_exts: HashSet<String> = self
            .config
            .file_extensions
            .iter()
            .map(|e| e.to_lowercase())
            .collect();

        let excluded: HashSet<String> = self
            .config
            .excluded_patterns
            .iter()
            .map(|p| p.to_lowercase())
            .collect();

        let max_bytes = self.config.max_file_size_kb * 1024;

        for dir_str in &directories {
            let dir = expand_tilde(dir_str);
            if !dir.exists() {
                tracing::warn!("Ingestion directory does not exist: {}", dir_str);
                continue;
            }

            tracing::info!("Scanning directory for ingestion: {}", dir_str);

            let walker = WalkDir::new(&dir)
                .follow_links(true)
                .max_depth(20);

            for entry in walker {
                let entry = match entry {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::debug!("WalkDir error: {}", e);
                        continue;
                    }
                };

                let path = entry.path();

                if !path.is_file() {
                    continue;
                }

                // Check extension
                let ext = match path.extension() {
                    Some(e) => e.to_string_lossy().to_lowercase(),
                    None => continue,
                };

                if !allowed_exts.contains(&ext) {
                    continue;
                }

                // Check excluded patterns
                let filename = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_lowercase())
                    .unwrap_or_default();

                let is_excluded = excluded.iter().any(|pattern| {
                    glob_pattern_matches(pattern, &filename) || glob_pattern_matches(pattern, &ext)
                });
                if is_excluded {
                    continue;
                }

                // Check file size
                let metadata = match fs::metadata(path) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if metadata.len() as usize > max_bytes {
                    tracing::debug!(
                        "Skipping oversized file ({} bytes): {}",
                        metadata.len(),
                        path.display()
                    );
                    result.files_skipped += 1;
                    continue;
                }

                result = match self.ingest_single_file(
                    path,
                    &self.get_collection_for_path(path),
                )
                .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!("Error ingesting {}: {}", path.display(), e);
                        let mut stats = self.stats.lock().await;
                        stats.files_errors += 1;
                        stats.clone()
                    }
                };
            }

            // Track active directory for watcher
            let mut dirs = self.active_directories.lock().await;
            dirs.insert(dir_str.clone());
        }

        tracing::info!(
            "Initial ingestion complete: {} indexed, {} skipped, {} errors, {} total chunks",
            result.files_indexed,
            result.files_skipped,
            result.files_errors,
            result.chunks_total
        );

        result
    }

    /// Ingest a single file into the RAG store
    async fn ingest_single_file(
        &self,
        path: &Path,
        collection: &str,
    ) -> anyhow::Result<IngestionResult> {
        let content_hash = compute_file_hash(path)?;
        let content = self.read_file_content(path).await?;

        // Check if already indexed with same hash
        if self.is_fresh_index(path, &content_hash)? {
            let mut stats = self.stats.lock().await;
            stats.files_skipped += 1;
            return Ok(stats.clone());
        }

        tracing::info!("Ingesting file: {} (collection: {})", path.display(), collection);

        let chunks_count = self
            .rag_engine
            .ingest_document(collection, &path.display().to_string(), &content)?;

        self.record_indexed_file(path, &content_hash, chunks_count as i64, collection)?;

        let mut stats = self.stats.lock().await;
        stats.files_indexed += 1;
        stats.chunks_total += chunks_count;
        Ok(stats.clone())
    }

    /// Get collection name for a file path (uses directory basename for per-directory collections)
    fn get_collection_for_path(&self, path: &Path) -> String {
        // For PDFs, use the configured collection directly
        if path.extension().map(|e| e.to_string_lossy().to_lowercase())
            == Some("pdf".to_string())
        {
            return self.config.collection.clone();
        }

        // For other files, use parent dir basename as collection
        if let Some(parent) = path.parent() {
            if let Some(name) = parent.file_name() {
                return name.to_string_lossy().to_lowercase().replace([' ', '-', '_'], "_");
            }
        }
        self.config.collection.clone()
    }

    /// Check if a file is already indexed with the same hash
    fn is_fresh_index(&self, path: &Path, content_hash: &str) -> anyhow::Result<bool> {
        let conn = self.vector_store.connection();
        let mut stmt = conn.prepare(
            "SELECT last_indexed, chunks_count FROM ingestion_sources
             WHERE source_path = ?1 AND content_hash = ?2",
        )?;
        match stmt.query_row([path.display().to_string(), content_hash.to_string()], |row| {
            Ok((row.get::<_, u64>(0)?, row.get::<_, i64>(1)?))
        }) {
            Ok((_timestamp, _chunks)) => Ok(true),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(false),
            Err(e) => Err(e).context("Error checking ingestion history"),
        }
    }

    /// Record a file as indexed
    fn record_indexed_file(
        &self,
        path: &Path,
        content_hash: &str,
        chunks_count: i64,
        collection: &str,
    ) -> anyhow::Result<()> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let conn = self.vector_store.connection();
        conn.execute(
            "INSERT OR REPLACE INTO ingestion_sources (source_path, content_hash, last_indexed, chunks_count, collection)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            [
                path.display().to_string(),
                content_hash.to_string(),
                now.to_string(),
                chunks_count.to_string(),
                collection.to_string(),
            ],
        )?;
        Ok(())
    }

    /// Read file content based on file type
    async fn read_file_content(&self, path: &Path) -> anyhow::Result<String> {
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        let is_pdf = ext == "pdf";
        let is_html = ext == "html" || ext == "htm";

        // If PDF and upstream enabled, send to llama.cpp for conversion
        if is_pdf && self.config.pdf_enabled {
            return self.read_pdf_content(path).await;
        }

        // HTML: use existing readability scraper
        if is_html {
            return self.read_html_content(path).await;
        }

        // All other text-based files: raw read
        let content = fs::read_to_string(path)?;
        Ok(content.trim().to_string())
    }

    /// Read a PDF by sending it to the upstream llama.cpp multimodal endpoint
    async fn read_pdf_content(&self, path: &Path) -> anyhow::Result<String> {
        let file_bytes = fs::read(path)?;
        let encoded = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &file_bytes);

        let model = &self.upstream_model;
        let resp = self
            .http_client
            .post(format!("{}/v1/chat/completions", self.upstream_base_url))
            .header(
                "Authorization",
                format!("Bearer {}", self.upstream_api_key),
            )
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text content from this PDF. Return only the text content, preserving paragraphs and structure. Do not include markdown formatting."},
                        {"type": "image_url", "image_url": {"url": format!("data:application/pdf;base64,{}", encoded)}}
                    ]
                }],
                "max_tokens": 4096,
                "temperature": 0.1
            }))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send PDF to upstream: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            tracing::warn!(
                "PDF conversion failed for {}: {} - {}",
                path.display(),
                status,
                body
            );
            // Fall back to raw file reading if upstream fails
            return fs::read_to_string(path)
                .map(|c| c.trim().to_string())
                .with_context(|| format!("Fallback read failed for PDF: {}", path.display()));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse upstream PDF response: {}", e))?;

        let content = json
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| anyhow::anyhow!("No content in upstream PDF response"))?;

        Ok(content.to_string())
    }

    /// Read HTML content using the existing readability extractor
    async fn read_html_content(&self, path: &Path) -> anyhow::Result<String> {
        let html = fs::read_to_string(path)?;
        let cleaned = crate::search::readability::clean_html(&html);
        Ok(format!("Title: {}\n\n{}", cleaned.title, cleaned.content))
    }

    /// Re-index all configured directories (clears and rebuilds)
    pub async fn reindex_all(&self) -> IngestionResult {
        tracing::info!("Reindexing all directories...");

        // Clear existing ingestion records
        {
            let conn = self.vector_store.connection();
            conn.execute("DELETE FROM ingestion_sources", [])
                .ok();
        }

        // Clear stats
        {
            let mut stats = self.stats.lock().await;
            *stats = IngestionResult {
                files_indexed: 0,
                files_skipped: 0,
                files_errors: 0,
                chunks_total: 0,
            };
        }

        self.run_initial_ingestion().await
    }

    /// Add a new directory to watch and ingest
    pub async fn add_directory(&self, dir_path: &str) -> IngestionResult {
        tracing::info!("Adding directory to watch: {}", dir_path);

        let mut dirs = self.active_directories.lock().await;
        if dirs.contains(dir_path) {
            return IngestionResult {
                files_indexed: 0,
                files_skipped: 0,
                files_errors: 0,
                chunks_total: 0,
            };
        }
        dirs.insert(dir_path.to_string());

        let mut result = self.run_initial_ingestion().await;
        result.files_indexed = 0; // run_initial_ingestion counts from 0, add our new dir
        result
    }

    /// Get current ingestion stats
    pub async fn get_stats(&self) -> IngestionResult {
        self.stats.lock().await.clone()
    }

    /// List all indexed files
    pub fn list_indexed_files(&self) -> anyhow::Result<Vec<IndexedFileInfo>> {
        let conn = self.vector_store.connection();
        let mut stmt = conn.prepare(
            "SELECT source_path, collection, chunks_count, content_hash, last_indexed
             FROM ingestion_sources ORDER BY last_indexed DESC",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(IndexedFileInfo {
                path: row.get(0)?,
                collection: row.get(1)?,
                chunks_count: row.get::<_, i64>(2)? as usize,
                content_hash: row.get(3)?,
                last_indexed: row.get(4)?,
            })
        })?;

        let mut files = Vec::new();
        for file in rows {
            match file {
                Ok(f) => files.push(f),
                Err(e) => {
                    tracing::error!("Error reading indexed file: {}", e);
                }
            }
        }

        Ok(files)
    }
}

/// Compute SHA-256 hash of a file
fn compute_file_hash(path: &Path) -> anyhow::Result<String> {
    let mut hasher = Sha256::new();
    let mut file = fs::File::open(path)?;
    io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Expand ~ to user's home directory
pub fn expand_tilde(path_str: &str) -> PathBuf {
    if path_str.starts_with("~/") || path_str == "~" {
        if let Some(home) = dirs::home_dir() {
            return home.join(&path_str[2..]);
        }
    }
    PathBuf::from(path_str)
}

/// Simple glob pattern matching (supports * wildcards)
fn glob_pattern_matches(pattern: &str, text: &str) -> bool {
    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            let prefix = parts[0];
            let suffix = parts[1];
            text.starts_with(prefix) && text.ends_with(suffix)
        } else {
            false
        }
    } else {
        text == pattern
    }
}
