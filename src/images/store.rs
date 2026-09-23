use std::path::{Path, PathBuf};

/// On-disk store for generated images, served by `GET /images/{name}`.
pub struct ImageStore {
    dir: PathBuf,
}

impl ImageStore {
    pub fn new(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Persist a generated PNG and return the on-disk path.
    pub fn save_png(&self, id: &str, bytes: &[u8]) -> anyhow::Result<PathBuf> {
        let safe_id = id
            .replace('/', "_")
            .replace('\\', "_")
            .replace("..", "_")
            .replace(' ', "_");
        let path = self.dir.join(format!("{safe_id}.png"));
        std::fs::write(&path, bytes)?;
        Ok(path)
    }

    /// Resolve a stored filename to a safe path for serving. Returns None for
    /// anything that could escape the store directory (path traversal).
    pub fn resolve(&self, name: &str) -> Option<PathBuf> {
        if name.is_empty()
            || name.contains('/')
            || name.contains('\\')
            || name.contains("..")
            || name.starts_with('.')
        {
            return None;
        }
        let path = self.dir.join(name);
        if path.is_file() {
            Some(path)
        } else {
            None
        }
    }

    /// Read a stored image by filename (extension-prefixed name, e.g. `x.png`).
    pub fn read(&self, name: &str) -> Option<Vec<u8>> {
        self.resolve(name)
            .and_then(|p| std::fs::read(p).ok())
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// Best-effort content-type from a file name ending.
pub fn content_type_for(name: &str) -> &'static str {
    let lower = name.to_lowercase();
    if lower.ends_with(".png") {
        "image/png"
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg"
    } else if lower.ends_with(".webp") {
        "image/webp"
    } else if lower.ends_with(".gif") {
        "image/gif"
    } else {
        "application/octet-stream"
    }
}