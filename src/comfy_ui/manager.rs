use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{json, Value};

use crate::config::ComfyUiConfig;

/// Manages the ComfyUI backend used for image generation. ComfyUI is left
/// running/loaded after a job (models auto-unload); it is only spawned the
/// first time it is needed and never stopped between requests.
pub struct ComfyUIManager {
    child: Mutex<Option<Child>>,
    url: String,
}

impl ComfyUIManager {
    pub fn new(url: String) -> Self {
        Self {
            child: Mutex::new(None),
            url: url.trim_end_matches('/').to_string(),
        }
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub async fn is_up(&self, client: &reqwest::Client) -> bool {
        match client.get(format!("{}/system_stats", self.url)).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// Ensure ComfyUI is reachable, spawning the managed subprocess if needed.
    /// Tolerates an externally-running instance. Never fails the caller unless
    /// the subprocess could not be spawned AND nothing is listening.
    pub async fn ensure_running(&self, cfg: &ComfyUiConfig, client: &reqwest::Client) -> anyhow::Result<()> {
        if self.is_up(client).await {
            return Ok(());
        }

        let python = cfg.python.trim();
        let workdir = cfg.workdir.trim();
        if python.is_empty() || workdir.is_empty() {
            anyhow::bail!(
                "ComfyUI is not reachable at {} and no spawn config is set (comfy_ui.python/workdir)",
                self.url
            );
        }

        tracing::info!("Starting managed ComfyUI from {}...", workdir);
        let mut cmd = Command::new(python);
        cmd.args(&cfg.args);
        cmd.current_dir(workdir);
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::null());

        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                // Port may be occupied by an external instance that is mid-boot.
                if self.is_up(client).await {
                    return Ok(());
                }
                return Err(anyhow::anyhow!("failed to spawn ComfyUI: {e}"));
            }
        };

        {
            let mut guard = self.child.lock().unwrap();
            *guard = Some(child);
        }

        let timeout = Duration::from_secs(cfg.health_timeout_s.max(10));
        let start = std::time::Instant::now();
        loop {
            {
                let mut guard = self.child.lock().unwrap();
                if let Some(ref mut ch) = guard.as_mut() {
                    if ch.try_wait().ok().flatten().is_some() {
                        tracing::warn!("ComfyUI process exited prematurely while starting");
                        *guard = None;
                        break;
                    }
                }
            }
            if self.is_up(client).await {
                tracing::info!("ComfyUI is ready at {}", self.url);
                return Ok(());
            }
            if start.elapsed() > timeout {
                tracing::warn!("ComfyUI did not become ready within {}s", timeout.as_secs());
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        anyhow::bail!("ComfyUI failed to become ready at {}", self.url)
    }

    /// Submit a workflow graph. Returns (prompt_id, number).
    pub async fn submit(&self, client: &reqwest::Client, graph: &Value) -> anyhow::Result<(String, u64)> {
        let body = json!({
            "prompt": graph,
            "client_id": "a-prox",
        });
        let resp = client
            .post(format!("{}/prompt", self.url))
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("ComfyUI /prompt request failed: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("ComfyUI /prompt returned {}: {}", status, text.chars().take(400).collect::<String>());
        }
        let data: Value = resp.json().await.map_err(|e| anyhow::anyhow!("bad /prompt response: {e}"))?;
        let prompt_id = data
            .get("prompt_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing prompt_id in /prompt response"))?
            .to_string();
        let number = data.get("number").and_then(|v| v.as_u64()).unwrap_or(0);
        Ok((prompt_id, number))
    }

    /// Poll /history/{id} until the outputs contain node "9" (SaveImageAdvanced)
    /// or the timeout elapses. Returns the full outputs map.
    pub async fn poll_outputs(
        &self,
        client: &reqwest::Client,
        prompt_id: &str,
        poll_interval_ms: u64,
        timeout_s: u64,
    ) -> anyhow::Result<Value> {
        let start = std::time::Instant::now();
        loop {
            let url = format!("{}/history/{}", self.url, prompt_id);
            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    if let Ok(data) = resp.json::<Value>().await {
                        if let Some(entry) = data.get(prompt_id) {
                            if let Some(outputs) = entry.get("outputs") {
                                if outputs.get("9").is_some() {
                                    return Ok(outputs.clone());
                                }
                            }
                        }
                    }
                }
                Ok(resp) => {
                    tracing::debug!("ComfyUI /history poll returned {}", resp.status());
                }
                Err(e) => {
                    tracing::debug!("ComfyUI /history poll error: {e}");
                }
            }

            if start.elapsed() > Duration::from_secs(timeout_s.max(10)) {
                anyhow::bail!(
                    "ComfyUI generation timed out after {}s (prompt {})",
                    timeout_s,
                    prompt_id
                );
            }
            tokio::time::sleep(Duration::from_millis(poll_interval_ms.max(100))).await;
        }
    }

    /// Fetch generated image bytes via /view.
    pub async fn fetch_view(
        &self,
        client: &reqwest::Client,
        filename: &str,
        subfolder: &str,
        image_type: &str,
    ) -> anyhow::Result<Vec<u8>> {
        let url = format!(
            "{}/view?filename={}&subfolder={}&type={}",
            self.url,
            url_escape(filename),
            url_escape(subfolder),
            url_escape(image_type)
        );
        let resp = client
            .get(&url)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("ComfyUI /view request failed: {e}"))?;
        if !resp.status().is_success() {
            anyhow::bail!("ComfyUI /view returned {}", resp.status());
        }
        resp.bytes().await.map(|b| b.to_vec()).map_err(|e| anyhow::anyhow!("failed reading /view body: {e}"))
    }

    /// Upload an image to ComfyUI's input directory. Returns the stored name.
    pub async fn upload_image(
        &self,
        client: &reqwest::Client,
        bytes: Vec<u8>,
        filename: &str,
    ) -> anyhow::Result<String> {
        use reqwest::multipart::{Form, Part};

        let mime = if filename.to_lowercase().ends_with(".png") {
            "image/png"
        } else if filename.to_lowercase().ends_with(".jpg") || filename.to_lowercase().ends_with(".jpeg") {
            "image/jpeg"
        } else if filename.to_lowercase().ends_with(".webp") {
            "image/webp"
        } else {
            "image/png"
        };

        let part = Part::bytes(bytes)
            .file_name(filename.to_string())
            .mime_str(mime)
            .map_err(|e| anyhow::anyhow!("bad mime: {e}"))?;
        let form = Form::new()
            .text("type", "input")
            .part("image", part);

        let resp = client
            .post(format!("{}/upload/image", self.url))
            .multipart(form)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("ComfyUI /upload/image failed: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("ComfyUI /upload/image returned {}: {}", status, text.chars().take(300).collect::<String>());
        }
        let data: Value = resp.json().await.map_err(|e| anyhow::anyhow!("bad upload response: {e}"))?;
        data.get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("missing 'name' in /upload/image response"))
    }

    pub fn has_child(&self) -> bool {
        let mut guard = self.child.lock().unwrap();
        if let Some(ref mut ch) = guard.as_mut() {
            ch.try_wait().unwrap_or(None).is_none()
        } else {
            false
        }
    }
}

impl Drop for ComfyUIManager {
    fn drop(&mut self) {
        if let Some(ref mut child) = *self.child.lock().unwrap() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn url_escape(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}