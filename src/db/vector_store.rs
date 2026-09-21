use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use rusqlite::{params, Connection};
use super::schema::{INIT_SQL, init_vec_table_sql};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub chunk_id: i64,
    pub collection: String,
    pub source_uri: String,
    pub chunk_index: usize,
    pub content: String,
    pub score: f32, // Higher is better (fused RRF or cosine similarity)
}

pub struct VectorStore {
    conn: Mutex<Connection>,
    dimension: usize,
}

impl VectorStore {
    pub fn new<P: AsRef<Path>>(
        db_path: P,
        dimension: usize,
        mmap_size_mb: usize,
        cache_size_mb: usize,
    ) -> anyhow::Result<Self> {
        // Register sqlite-vec extension auto-init once globally
        static VEC_INIT: std::sync::Once = std::sync::Once::new();
        VEC_INIT.call_once(|| {
            unsafe {
                rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                    sqlite_vec::sqlite3_vec_init as *const (),
                )));
            }
        });

        let path = db_path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)?;

        // Apply hardware-tuned PRAGMAs (exploiting large DDR4 RAM and SSD)
        conn.execute_batch(&format!(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA temp_store = MEMORY;
            PRAGMA mmap_size = {};
            PRAGMA cache_size = -{};
            "#,
            mmap_size_mb * 1024 * 1024,
            cache_size_mb * 1024
        ))?;

        // Initialize core schema
        conn.execute_batch(INIT_SQL)?;

        // Initialize vec virtual table
        let vec_sql = init_vec_table_sql(dimension);
        conn.execute_batch(&vec_sql)?;

        tracing::info!(
            "VectorStore initialized at {:?} (dimension: {}, mmap: {}MB, cache: {}MB)",
            path,
            dimension,
            mmap_size_mb,
            cache_size_mb
        );

        Ok(Self {
            conn: Mutex::new(conn),
            dimension,
        })
    }

    /// Ingests a chunk with its embedding and updates FTS and vector indexes
    pub fn insert_chunk(
        &self,
        collection: &str,
        source_uri: &str,
        chunk_index: usize,
        content: &str,
        token_count: usize,
        embedding: &[f32],
    ) -> anyhow::Result<i64> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64;

        // 1. Insert metadata
        tx.execute(
            "INSERT INTO document_chunks (collection, source_uri, chunk_index, content, token_count, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![collection, source_uri, chunk_index as i64, content, token_count as i64, now],
        )?;
        let chunk_id = tx.last_insert_rowid();

        // 2. Insert into FTS5 index
        tx.execute(
            "INSERT INTO document_chunks_fts (rowid, content) VALUES (?1, ?2)",
            params![chunk_id, content],
        )?;

        // 3. Insert into vec0 table
        let bytes = f32_slice_to_bytes(embedding);
        tx.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
            params![chunk_id, bytes],
        )?;

        tx.commit()?;
        Ok(chunk_id)
    }

    /// Hybrid Search: Combines vector similarity with FTS5 lexical match using RRF
    pub fn hybrid_search(
        &self,
        query_text: &str,
        query_vector: &[f32],
        collection: Option<&str>,
        limit: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let conn = self.conn.lock().unwrap();

        // 1. Vector Search
        let bytes = f32_slice_to_bytes(query_vector);
        let mut vec_stmt = conn.prepare(
            "SELECT chunk_id, distance
             FROM vec_chunks
             WHERE embedding MATCH ?1
             ORDER BY distance
             LIMIT ?2",
        )?;

        let vec_rows = vec_stmt.query_map(params![bytes, (limit * 2) as i64], |row| {
            let chunk_id: i64 = row.get(0)?;
            let distance: f32 = row.get(1)?;
            Ok((chunk_id, distance))
        })?;

        let mut vec_ranks: HashMap<i64, usize> = HashMap::new();
        for (rank, item) in vec_rows.enumerate() {
            if let Ok((chunk_id, _)) = item {
                vec_ranks.insert(chunk_id, rank + 1);
            }
        }

        // 2. Lexical FTS5 Search
        let sanitized_query = sanitize_fts_query(query_text);
        let mut fts_ranks: HashMap<i64, usize> = HashMap::new();

        if !sanitized_query.is_empty() {
            if let Ok(mut fts_stmt) = conn.prepare(
                "SELECT rowid, rank
                 FROM document_chunks_fts
                 WHERE document_chunks_fts MATCH ?1
                 ORDER BY rank
                 LIMIT ?2",
            ) {
                if let Ok(fts_rows) = fts_stmt.query_map(params![sanitized_query, (limit * 2) as i64], |row| {
                    let rowid: i64 = row.get(0)?;
                    let rank_val: f32 = row.get(1)?;
                    Ok((rowid, rank_val))
                }) {
                    for (rank, item) in fts_rows.enumerate() {
                        if let Ok((rowid, _)) = item {
                            fts_ranks.insert(rowid, rank + 1);
                        }
                    }
                }
            }
        }

        // 3. Reciprocal Rank Fusion (RRF)
        // RRF(d) = sum(1 / (60 + rank))
        let k = 60.0f32;
        let mut rrf_scores: HashMap<i64, f32> = HashMap::new();

        for (&chunk_id, &rank) in &vec_ranks {
            let score = 1.0 / (k + rank as f32);
            *rrf_scores.entry(chunk_id).or_default() += score;
        }

        for (&chunk_id, &rank) in &fts_ranks {
            let score = 1.0 / (k + rank as f32);
            *rrf_scores.entry(chunk_id).or_default() += score;
        }

        let mut ranked_chunks: Vec<(i64, f32)> = rrf_scores.into_iter().collect();
        ranked_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked_chunks.truncate(limit);

        // 4. Fetch chunk contents and metadata
        let mut results = Vec::new();
        let mut fetch_stmt = conn.prepare(
            "SELECT id, collection, source_uri, chunk_index, content
             FROM document_chunks
             WHERE id = ?1",
        )?;

        for (chunk_id, score) in ranked_chunks {
            let res = fetch_stmt.query_row(params![chunk_id], |row| {
                Ok(SearchResult {
                    chunk_id: row.get(0)?,
                    collection: row.get(1)?,
                    source_uri: row.get(2)?,
                    chunk_index: row.get::<_, i64>(3)? as usize,
                    content: row.get(4)?,
                    score,
                })
            });

            if let Ok(item) = res {
                if let Some(col) = collection {
                    if item.collection != col {
                        continue;
                    }
                }
                results.push(item);
            }
        }

        Ok(results)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the SQLite connection for direct queries
    pub fn connection(&self) -> std::sync::MutexGuard<'_, rusqlite::Connection> {
        self.conn.lock().unwrap()
    }

    pub fn stats(&self) -> anyhow::Result<(i64, std::collections::HashMap<String, i64>)> {
        let conn = self.conn.lock().unwrap();
        let total: i64 = conn.query_row(
            "SELECT COUNT(*) FROM document_chunks",
            [],
            |row| row.get(0),
        )?;

        let mut collections: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        let mut stmt = conn.prepare(
            "SELECT collection, COUNT(*) FROM document_chunks GROUP BY collection"
        )?;
        let rows = stmt.query_map([], |row| {
            let col: String = row.get(0)?;
            let cnt: i64 = row.get(1)?;
            Ok((col, cnt))
        })?;
        for r in rows {
            if let Ok((col, cnt)) = r {
                collections.insert(col, cnt);
            }
        }

        Ok((total, collections))
    }
}

fn f32_slice_to_bytes(slice: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<f32>(),
        )
    }
}

fn sanitize_fts_query(input: &str) -> String {
    let alphanumeric_words: Vec<&str> = input
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() > 1)
        .collect();

    alphanumeric_words.join(" OR ")
}
