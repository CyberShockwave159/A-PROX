use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};

use super::parser::DirectoryIngestor;

/// Passive file watcher that monitors directories for changes
/// and triggers ingestion on file modifications.
///
/// The watcher is designed to be non-aggressive:
/// - It coalesces rapid changes (batching window: 2 seconds)
/// - It uses a debounce interval from config (default: 30s between full scans)
/// - It skips files already indexed with matching hash
pub struct DirectoryWatcher {
    ingestor: Arc<DirectoryIngestor>,
    watch_interval: Duration,
}

impl DirectoryWatcher {
    pub fn new(
        ingestor: Arc<DirectoryIngestor>,
        watch_interval_secs: u64,
    ) -> Self {
        Self {
            ingestor,
            watch_interval: Duration::from_secs(watch_interval_secs),
        }
    }

    /// Start the watcher in a background tokio task
    pub fn spawn(self: Arc<Self>, directories: &[String]) {
        if directories.is_empty() {
            return;
        }

        tokio::spawn(async move {
            let mut periodic_interval = tokio::time::interval(self.watch_interval);
            periodic_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

            loop {
                periodic_interval.tick().await;

                // Full re-scan of all active directories
                let result = self.ingestor.run_initial_ingestion().await;
                tracing::info!(
                    "Periodic watcher scan: {} indexed, {} skipped, {} errors, {} chunks",
                    result.files_indexed,
                    result.files_skipped,
                    result.files_errors,
                    result.chunks_total
                );
            }
        });
    }

    /// Set up notify-based file system watchers on all active directories
    pub fn watch_directories(
        &self,
        _directories: &[String],
    ) -> anyhow::Result<()> {
        // Get the shared active_directories from ingestor
        let shared_dirs = self.ingestor.active_directories.clone();
        let ingestor = Arc::clone(&self.ingestor);

        tokio::spawn(async move {
            let mut dir_strings: Vec<String> = Vec::new();
            {
                let dirs = shared_dirs.lock().await;
                for dir_str in dirs.iter() {
                    dir_strings.push(dir_str.clone());
                }
            }

            if dir_strings.is_empty() {
                return;
            }

            // Set up notify watchers
            for dir_str in &dir_strings {
                let dir = super::expand_tilde(dir_str);
                if !dir.exists() {
                    tracing::warn!("Watch directory does not exist: {}", dir_str);
                    continue;
                }

                match Self::setup_notify_watcher(dir.clone(), Arc::clone(&ingestor)) {
                    Ok(_watcher) => {
                        tracing::info!("Set up file watcher for: {}", dir_str);
                    }
                    Err(e) => {
                        tracing::error!(
                            "Failed to set up file watcher for {}: {}",
                            dir_str,
                            e
                        );
                    }
                }
            }
        });

        Ok(())
    }

    /// Set up a notify watcher for a single directory
    fn setup_notify_watcher(
        dir: std::path::PathBuf,
        ingestor: Arc<DirectoryIngestor>,
    ) -> anyhow::Result<RecommendedWatcher> {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<std::path::PathBuf>(100);
        
        let mut watcher =
            RecommendedWatcher::new(
                move |res: Result<Event, notify::Error>| {
                    let tx = tx.clone();
                    tokio::spawn(async move {
                        match res {
                            Ok(event) => {
                                if event.kind.is_create()
                                    || event.kind.is_modify()
                                {
                                    for path in &event.paths {
                                        if path.is_file() {
                                            let _ = tx.send(path.clone()).await;
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::error!("Watch error: {}", e);
                            }
                        }
                    });
                },
                Config::default(),
            )?;

        watcher.watch(&dir, RecursiveMode::NonRecursive)?;

        // Process events from channel
        tokio::spawn(async move {
            let mut debounce = tokio::time::interval(Duration::from_millis(500));
            debounce.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

            let mut pending_paths: Vec<std::path::PathBuf> = Vec::new();

            loop {
                tokio::select! {
                    _ = debounce.tick() => {
                        if !pending_paths.is_empty() {
                            let paths = std::mem::take(&mut pending_paths);
                            for path in paths {
                                let collection = get_collection_for_path(&path);
                                match ingestor.ingest_single_file(&path, &collection).await {
                                    Ok(result) => {
                                        tracing::debug!(
                                            "Watch ingestion for {}: indexed {} chunks",
                                            path.display(),
                                            result.chunks_total
                                        );
                                    }
                                    Err(e) => {
                                        tracing::debug!(
                                            "Watch ingestion skipped {}: {}",
                                            path.display(),
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Some(path) = rx.recv() => {
                        if let Some(ext) = path.extension() {
                            let ext_str = ext.to_string_lossy().to_lowercase();
                            let allowed_exts = ["txt", "md", "json", "yaml", "yml", "toml", "html", "htm", "log", "csv", "xml", "pdf"];
                            if allowed_exts.contains(&ext_str.as_str()) {
                                pending_paths.push(path);
                            }
                        }
                    }
                }
            }
        });

        Ok(watcher)
    }
}

/// Get collection name for a path (same logic as DirectoryIngestor)
fn get_collection_for_path(path: &Path) -> String {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase());
    let is_pdf = ext.as_deref() == Some("pdf");

    if is_pdf {
        return "auto-indexed".to_string();
    }

    if let Some(parent) = path.parent() {
        if let Some(name) = parent.file_name() {
            return name.to_string_lossy().to_lowercase().replace([' ', '-', '_'], "_");
        }
    }
    "auto-indexed".to_string()
}
