use std::fs;
use std::io::Write;
use std::net::TcpListener;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use tokio::time;
use tokio::time::Instant;

use super::config::generate_settings;

pub struct SearXNGManager {
    child: std::sync::Mutex<Option<Child>>,
}

impl SearXNGManager {
    pub fn new() -> Self {
        Self {
            child: std::sync::Mutex::new(None),
        }
    }

    /// Start the SearXNG subprocess. Returns Ok(()) if SearXNG is ready or already running.
    pub async fn start(&self, config: &crate::config::SearXNGConfig, http_client: &reqwest::Client) -> anyhow::Result<()> {
        let port = config.port;

        // Check if SearXNG is already installed
        let install_dir = Self::expand_tilde(&config.install_dir);
        let venv_python = Path::new(&install_dir).join(".venv").join("bin").join("python");
        let settings_path = Path::new(&install_dir).join("settings.yml");

        if !venv_python.exists() {
            tracing::info!(
                "SearXNG not installed at {}. Run: ./scripts/download_searxng.sh",
                install_dir
            );
            return Ok(());
        }

        // Generate or update settings if needed
        let needs_settings = !settings_path.exists() || Self::settings_port_mismatch(&settings_path, port);
        if needs_settings {
            let settings_yaml = generate_settings(config, port);
            let mut f = fs::File::create(&settings_path)?;
            f.write_all(settings_yaml.as_bytes())?;
            tracing::info!("Generated SearXNG settings at {}", settings_path.display());
        }

        // Check if another process already occupies the port
        if TcpListener::bind(("127.0.0.1", port)).is_err() {
            tracing::warn!("Port {} already in use. Assuming SearXNG is running externally.", port);
            return Ok(());
        }

        // Start SearXNG subprocess
        tracing::info!("Starting SearXNG on port {}...", port);
        let settings_path = Path::new(&install_dir).join("settings.yml");
        let child = Command::new(&venv_python)
            .args(["-c", "from searx.webapp import run; run()"])
            .env("SEARXNG_SETTINGS_PATH", &settings_path)
            .current_dir(install_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;

        {
            let mut guard = self.child.lock().unwrap();
            *guard = Some(child);
        }

        // Wait for SearXNG health endpoint
        let timeout = Duration::from_secs(30);
        let poll_interval = Duration::from_millis(500);
        let start = Instant::now();

        loop {
            if start.elapsed() > timeout {
                tracing::warn!(
                    "SearXNG failed to become ready within {}s. Using fallback search.",
                    timeout.as_secs()
                );
                self.stop();
                return Ok(());
            }

            // Check if process exited prematurely
            {
                let mut guard = self.child.lock().unwrap();
                if let Some(ref mut ch) = *guard {
                    if ch.try_wait().ok().flatten().is_some() {
                        tracing::warn!("SearXNG process exited prematurely.");
                        return Err(anyhow::anyhow!("SearXNG exited"));
                    }
                }
            }

            // Health check
            let health_url = format!("http://127.0.0.1:{}/search?q=test&format=json", port);
            match http_client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    tracing::info!("SearXNG is ready on port {}", port);
                    return Ok(());
                }
                Ok(resp) => {
                    tracing::debug!("SearXNG health check returned {}: {}", resp.status(), health_url);
                }
                Err(e) => {
                    tracing::debug!("SearXNG health check error: {}", e);
                }
            }

            time::sleep(poll_interval).await;
        }
    }

    /// Stop the SearXNG subprocess gracefully.
    /// Returns true if the SearXNG subprocess is currently running
    pub fn is_running(&self) -> bool {
        let mut guard = self.child.lock().unwrap();
        if let Some(ref mut child) = guard.as_mut() {
            child.try_wait().unwrap_or(None).is_none()
        } else {
            false
        }
    }

    pub fn stop(&self) {
        if let Some(ref mut child) = *self.child.lock().unwrap() {
            tracing::info!("Sending SIGTERM to SearXNG...");
            let _ = child.kill();

            let start = Instant::now();
            let timeout = Duration::from_secs(5);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => {
                        tracing::info!("SearXNG stopped gracefully.");
                        break;
                    }
                    Ok(None) => {
                        if start.elapsed() > timeout {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    Err(_) => break,
                }
            }

            // Force kill if still alive
            {
                let mut guard = self.child.lock().unwrap();
                if let Some(ref mut ch) = *guard {
                    match ch.try_wait() {
                        Ok(None) => {
                            tracing::warn!("SearXNG did not stop in time, sending SIGKILL...");
                            let _ = ch.kill();
                            let _ = ch.wait();
                        }
                        _ => {}
                    }
                    *guard = None;
                }
            }
        }
    }

    fn settings_port_mismatch(path: &Path, port: u16) -> bool {
        if let Ok(content) = fs::read_to_string(path) {
            !content.contains(&format!("port: {}", port))
        } else {
            true
        }
    }

    fn expand_tilde(path_str: &str) -> String {
        if let Some(stripped) = path_str.strip_prefix("~/") {
            if let Some(home) = dirs::home_dir() {
                return home.join(stripped).to_string_lossy().into_owned();
            }
        }
        path_str.to_string()
    }
}

impl Drop for SearXNGManager {
    fn drop(&mut self) {
        self.stop();
    }
}
