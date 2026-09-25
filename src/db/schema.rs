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

-- Async request queue & result cache
CREATE TABLE IF NOT EXISTS async_requests (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    result TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    route_decision TEXT,
    tokens_received INTEGER DEFAULT 0,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_async_requests_status ON async_requests(status);
CREATE INDEX IF NOT EXISTS idx_async_requests_expires ON async_requests(expires_at);
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
