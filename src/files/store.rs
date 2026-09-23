use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

/// On-disk store for generated text files, served by `GET /files/{name}`.
pub struct FileStore {
    dir: PathBuf,
}

impl FileStore {
    pub fn new(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Persist a stored file body under `stored_name`, creating (or appending to)
    /// the on-disk file. Returns the resolved path.
    pub fn save(&self, stored_name: &str, bytes: &[u8], append: bool) -> anyhow::Result<PathBuf> {
        let safe = stored_name
            .replace('/', "_")
            .replace('\\', "_")
            .replace("..", "_")
            .replace(' ', "_");
        let path = self.dir.join(safe);
        if append {
            let mut f = OpenOptions::new().create(true).append(true).open(&path)?;
            f.write_all(bytes)?;
        } else {
            std::fs::write(&path, bytes)?;
        }
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

    /// Read a stored file by its servable name (e.g. `file_1789..._0_story.txt`).
    pub fn read(&self, name: &str) -> Option<Vec<u8>> {
        self.resolve(name).and_then(|p| std::fs::read(p).ok())
    }

    /// Current byte length of a stored file, if it exists.
    pub fn size(&self, name: &str) -> Option<u64> {
        self.resolve(name)
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_creates_and_appends() {
        let dir = std::env::temp_dir().join(format!("a-prox-filestore-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = FileStore::new(&dir).unwrap();
        let name = "test.md";

        store.save(name, b"one\n", false).unwrap();
        assert_eq!(store.size(name).unwrap(), 4);

        store.save(name, b"two\n", true).unwrap();
        assert_eq!(store.size(name).unwrap(), 8);
        assert_eq!(String::from_utf8(store.read(name).unwrap()).unwrap(), "one\ntwo\n");

        // overwrite replaces, does not grow
        store.save(name, b"new", false).unwrap();
        assert_eq!(store.size(name).unwrap(), 3);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_rejects_traversal() {
        let dir = std::env::temp_dir().join(format!("a-prox-filestore-safe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = FileStore::new(&dir).unwrap();
        store.save("ok.md", b"x", false).unwrap();
        assert!(store.resolve("ok.md").is_some());
        assert!(store.resolve("../ok.md").is_none());
        assert!(store.resolve("a/b.md").is_none());
        assert!(store.resolve(".hidden").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn content_types() {
        assert_eq!(content_type_for("a.md"), "text/markdown");
        assert_eq!(content_type_for("notes.txt"), "text/plain");
        assert_eq!(content_type_for("fib.py"), "text/x-python");
        assert_eq!(content_type_for("data.json"), "application/json");
        assert_eq!(content_type_for("rows.csv"), "text/csv");
        assert_eq!(content_type_for("weird.xyz"), "application/octet-stream");
    }
}

/// Best-effort content-type from a file name extension.
pub fn content_type_for(name: &str) -> &'static str {
    let lower = name.to_lowercase();
    if lower.ends_with(".md") || lower.ends_with(".markdown") {
        "text/markdown"
    } else if lower.ends_with(".txt") || lower.ends_with(".text") || lower.ends_with(".log") {
        "text/plain"
    } else if lower.ends_with(".json") {
        "application/json"
    } else if lower.ends_with(".csv") {
        "text/csv"
    } else if lower.ends_with(".yml") || lower.ends_with(".yaml") {
        "application/yaml"
    } else if lower.ends_with(".toml") {
        "application/toml"
    } else if lower.ends_with(".xml") {
        "application/xml"
    } else if lower.ends_with(".py") {
        "text/x-python"
    } else if lower.ends_with(".js") {
        "text/javascript"
    } else if lower.ends_with(".ts") {
        "text/typescript"
    } else if lower.ends_with(".rs") {
        "text/x-rust"
    } else if lower.ends_with(".go") {
        "text/x-go"
    } else if lower.ends_with(".sql") {
        "application/sql"
    } else if lower.ends_with(".svg") {
        "image/svg+xml"
    } else {
        "application/octet-stream"
    }
}