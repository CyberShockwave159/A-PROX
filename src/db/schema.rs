pub const INIT_SQL: &str = r#"
-- Base metadata table for document chunks
CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    created_at INTEGER NOT NULL
);

-- Full Text Search table using FTS5 with porter stemmer
CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
    content,
    tokenize = 'porter unicode61'
);

-- Request audit & cache log
CREATE TABLE IF NOT EXISTS request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    total_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    created_at INTEGER NOT NULL
);

-- Tracking for directory ingestion sources
CREATE TABLE IF NOT EXISTS ingestion_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    last_indexed INTEGER NOT NULL,
    chunks_count INTEGER NOT NULL,
    collection TEXT NOT NULL
);
"#;

pub fn init_vec_table_sql(dimension: usize) -> String {
    format!(
        r#"
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding float[{}] distance_metric=cosine
);
"#,
        dimension
    )
}
