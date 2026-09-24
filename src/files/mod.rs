pub mod store;

pub use store::{content_type_for, FileStore};

use std::path::PathBuf;

/// A text file produced by the `write_file` tool and served by `GET /files/{name}`.
#[derive(Debug, Clone)]
pub struct GeneratedFile {
    /// Stable logical id shared across `mode=append` chunks, e.g. `file_1789..._0`.
    pub id: String,
    /// User-facing file name (sanitized, owns an extension), e.g. `story.txt`.
    pub name: String,
    /// Name used as the `/files/{name}` key and on-disk file, `{id}_{name}`.
    pub servable_name: String,
    /// Publicly reachable URL of the served file. Always a served URL, used in
    /// LLM-facing contexts (tool results) — never a base64 blob.
    pub public_url: String,
    /// Base64 data-URL of the file, present only when
    /// `[file_generation].inline_data_url` is enabled. Emitted to the CLIENT in
    /// the `file_url` payload only; never placed in LLM context.
    pub data_url: Option<String>,
    /// On-disk path of the stored file.
    pub file_path: PathBuf,
    /// Best-effort MIME type from the extension.
    pub mime: &'static str,
    /// Total length of the stored file in bytes.
    pub size: u64,
}