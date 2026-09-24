use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::config::LlamaServerConfig;

/// Owns the llama.cpp server child process that A-PROX starts, stops around
/// image-generation jobs, and restarts afterwards.
///
/// The launch flags mirror the previously-manual invocation exactly, minus
/// `--load-mode mlock` (see the image-gen implementation plan, decision #1).
pub struct LlamaServerManager {
    child: Mutex<Option<Child>>,
    base_url: Mutex<String>,
}

impl LlamaServerManager {
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
            base_url: Mutex::new(String::new()),
        }
    }

    pub fn url(&self) -> String {
        self.base_url.lock().unwrap().clone()
    }

    /// True when a subprocess slot is held and the process has not exited.
    pub fn has_child(&self) -> bool {
        let mut guard = self.child.lock().unwrap();
        if let Some(ref mut ch) = guard.as_mut() {
            ch.try_wait().unwrap_or(None).is_none()
        } else {
            false
        }
    }

    /// Health probe against the configured base URL.
    pub async fn is_up(&self, client: &reqwest::Client, base_url: &str) -> bool {
        let url = format!("{}/health", base_url.trim_end_matches('/'));
        match client.get(&url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// Ensure llama.cpp is serving. If it is already healthy, returns Ok(false)
    /// (nothing spawned). Otherwise spawns the subprocess and waits for health.
    pub async fn start(
        &self,
        cfg: &LlamaServerConfig,
        client: &reqwest::Client,
    ) -> anyhow::Result<bool> {
        let base_url = format!("http://{}:{}", cfg.host, cfg.port);
        *self.base_url.lock().unwrap() = base_url.clone();

        if self.is_up(client, &base_url).await {
            tracing::info!("llama.cpp already serving at {base_url}; using external instance");
            return Ok(false);
        }

        self.stop();

        let model_path = cfg.model_path.trim();
        if model_path.is_empty() {
            anyhow::bail!("llama_server.model_path not configured");
        }
        let executable = cfg.executable.trim();
        if executable.is_empty() {
            anyhow::bail!("llama_server.executable not configured");
        }

        tracing::info!("Starting managed llama.cpp server on port {}", cfg.port);

        let cmd = build_args(cfg);
        let child = Command::new(executable)
            .args(&cmd)
            .stdin(Stdio::null())
            // Forward llama.cpp's stdout/stderr to the A-PROX console so the
            // user sees what the backend is doing (model load progress, etc.).
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| anyhow::anyhow!("failed to spawn llama-server: {e}"))?;

        {
            let mut guard = self.child.lock().unwrap();
            *guard = Some(child);
        }

        // Wait for the health endpoint.
        let timeout = Duration::from_secs(cfg.health_timeout_s.max(5));
        let poll = Duration::from_millis(500);
        let start = Instant::now();
        loop {
            if start.elapsed() > timeout {
                tracing::warn!("llama.cpp did not become healthy within {}s", timeout.as_secs());
                self.stop();
                return Ok(false);
            }

            {
                let mut guard = self.child.lock().unwrap();
                if let Some(ref mut ch) = guard.as_mut() {
                    if ch.try_wait().ok().flatten().is_some() {
                        tracing::warn!("llama.cpp process exited prematurely while starting");
                        *guard = None;
                        return Err(anyhow::anyhow!("llama-server exited during startup"));
                    }
                }
            }

            if self.is_up(client, &base_url).await {
                tracing::info!("llama.cpp is healthy on port {}", cfg.port);
                return Ok(true);
            }

            tokio::time::sleep(poll).await;
        }
    }

    /// Graceful stop: SIGTERM, wait up to `stop_grace_s`, then SIGKILL.
    pub fn stop(&self) {
        let mut guard = self.child.lock().unwrap();
        if let Some(ref mut child) = guard.as_mut() {
            let pid = child.id();
            tracing::info!("Sending SIGTERM to llama.cpp (pid {pid})...");
            // Child::kill() sends SIGKILL; use libc for a graceful SIGTERM.
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }

            let start = Instant::now();
            loop {
                match child.try_wait() {
                    Ok(Some(_status)) => {
                        tracing::info!("llama.cpp stopped gracefully.");
                        break;
                    }
                    Ok(None) => {
                        if start.elapsed() > Duration::from_secs(30) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    Err(_) => break,
                }
            }

            match child.try_wait() {
                Ok(None) => {
                    tracing::warn!("llama.cpp did not exit after SIGTERM; sending SIGKILL");
                    let _ = child.kill();
                    let _ = child.wait();
                }
                _ => {}
            }
            *guard = None;
        }
    }
}

impl Default for LlamaServerManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for LlamaServerManager {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Build the llama-server argument vector from config. Boolean flags are added
/// only when enabled; `--load-mode` is added only when not "none".
fn build_args(cfg: &LlamaServerConfig) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();

    args.push("--model".to_string());
    args.push(cfg.model_path.clone());

    if !cfg.mmproj_path.trim().is_empty() {
        args.push("--mmproj".to_string());
        args.push(cfg.mmproj_path.clone());
    }

    args.push("--ctx-size".to_string());
    args.push(cfg.ctx_size.to_string());

    if cfg.no_mmproj_offload {
        args.push("--no-mmproj-offload".to_string());
    }

    args.push("--flash-attn".to_string());
    args.push(cfg.flash_attn.clone());

    args.push("--n-cpu-moe".to_string());
    args.push(cfg.n_cpu_moe.to_string());

    args.push("--cache-type-k".to_string());
    args.push(cfg.cache_type_k.clone());
    args.push("--cache-type-v".to_string());
    args.push(cfg.cache_type_v.clone());

    if cfg.reasoning_preserve {
        args.push("--reasoning-preserve".to_string());
    }
    if cfg.kv_unified {
        args.push("--kv-unified".to_string());
    }

    args.push("--threads".to_string());
    args.push(cfg.threads.to_string());
    args.push("--batch-size".to_string());
    args.push(cfg.batch_size.to_string());
    args.push("--ubatch-size".to_string());
    args.push(cfg.ubatch_size.to_string());
    args.push("--image-min-tokens".to_string());
    args.push(cfg.image_min_tokens.to_string());
    args.push("--image-max-tokens".to_string());
    args.push(cfg.image_max_tokens.to_string());

    if !cfg.cors_origins.trim().is_empty() {
        args.push("--cors-origins".to_string());
        args.push(cfg.cors_origins.clone());
    }

    args.push("--host".to_string());
    args.push(cfg.host.clone());
    args.push("--port".to_string());
    args.push(cfg.port.to_string());

    if !cfg.api_key.trim().is_empty() {
        args.push("--api-key".to_string());
        args.push(cfg.api_key.clone());
    }

    let load_mode = cfg.load_mode.trim();
    if !load_mode.is_empty() && load_mode != "none" {
        args.push("--load-mode".to_string());
        args.push(load_mode.to_string());
    }

    args
}