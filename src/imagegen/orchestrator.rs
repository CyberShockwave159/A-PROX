use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde_json::Value;

use crate::comfy_ui::{ComfyUIManager, WorkflowKind, WorkflowParams, WorkflowTemplate};
use crate::config::{ComfyUiConfig, ImageGenerationConfig, LlamaServerConfig};
use crate::images::ImageStore;
use crate::llama_server::LlamaServerManager;
use crate::imagegen::ratio::compute_dimensions;

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Deterministic-looking but per-job-random seed for the KSampler (ComfyUI's
/// schema rejects -1). Mixes wall-clock nanos with a counter; never zero.
fn random_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xBADC0FFEE0DDF00D);
    let ctr = ID_COUNTER.fetch_add(1, Ordering::SeqCst).wrapping_mul(2654435761);
    let mut s = nanos ^ ctr ^ (nanos >> 17);
    if s == 0 {
        s = 0x9E3779B97F4A7C15;
    }
    s
}

/// A successfully generated image, persisted and publicly addressable.
pub struct GeneratedImage {
    pub id: String,
    pub public_url: String,
    pub file_path: std::path::PathBuf,
    pub png_bytes: Vec<u8>,
    pub kind: WorkflowKind,
}

/// Everything needed to run one image job.
pub struct GenerateRequest {
    pub kind: WorkflowKind,
    pub prompt: String,
    pub negative_prompt: String,
    pub wh_ratio: Option<String>,
    /// Original attached image dims, used when the model asks to follow them.
    pub reference_dims: Option<(u32, u32)>,
    /// The attached image bytes (uploaded to ComfyUI for i2i), if any.
    pub reference_image: Option<Vec<u8>>,
}

/// Owns llama.cpp lifecycle + ComfyUI orchestration for image jobs.
///
/// The `generate` flow intentionally holds the caller's concurrency permit for
/// its whole duration (llama.cpp stop → ComfyUI job → llama.cpp restart), so at
/// most one image job runs at a time, mirroring the guardrails semaphore.
pub struct ImageGenService {
    llama: Arc<LlamaServerManager>,
    comfy: Arc<ComfyUIManager>,
    store: Arc<ImageStore>,
    http_client: reqwest::Client,
    llama_cfg: LlamaServerConfig,
    comfy_cfg: ComfyUiConfig,
    img_cfg: ImageGenerationConfig,
    public_base: String,
}

impl ImageGenService {
    pub fn new(
        llama: Arc<LlamaServerManager>,
        comfy: Arc<ComfyUIManager>,
        store: Arc<ImageStore>,
        http_client: reqwest::Client,
        llama_cfg: LlamaServerConfig,
        comfy_cfg: ComfyUiConfig,
        img_cfg: ImageGenerationConfig,
        server_port: u16,
    ) -> Self {
        let public_base = if !img_cfg.public_base_url.trim().is_empty() {
            img_cfg.public_base_url.trim_end_matches('/').to_string()
        } else {
            format!("http://127.0.0.1:{server_port}")
        };
        Self {
            llama,
            comfy,
            store,
            http_client,
            llama_cfg,
            comfy_cfg,
            img_cfg,
            public_base,
        }
    }

    pub async fn generate(&self, req: GenerateRequest) -> anyhow::Result<GeneratedImage> {
        if !self.img_cfg.enabled {
            anyhow::bail!("image generation is disabled ([image_generation] enabled = false)");
        }

        // Ensure ComfyUI is reachable before we stop llama (which frees VRAM).
        self.comfy.ensure_running(&self.comfy_cfg, &self.http_client).await?;

        tracing::info!("Image job: stopping llama.cpp to free VRAM for ComfyUI...");
        self.llama.stop();

        let start = Instant::now();
        let outcome: anyhow::Result<GeneratedImage> = async {
            let (w, h) = compute_dimensions(
                req.wh_ratio.as_deref(),
                req.reference_dims,
                self.img_cfg.mp,
                self.img_cfg.multiple,
                self.img_cfg.max_side,
            );

            let workflow_path = match req.kind {
                WorkflowKind::TextToImage => &self.img_cfg.t2i_workflow,
                WorkflowKind::ImageToImage => &self.img_cfg.i2i_workflow,
            };
            let mut graph = WorkflowTemplate::load(workflow_path)?;

            let id = format!(
                "gen_{}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0),
                ID_COUNTER.fetch_add(1, Ordering::SeqCst)
            );

            // Upload the reference image before submitting (i2i).
            let uploaded = match (&req.kind, &req.reference_image) {
                (WorkflowKind::ImageToImage, Some(bytes)) => {
                    let name = format!("a-prox-{}.jpg", id);
                    Some(self.comfy.upload_image(&self.http_client, bytes.clone(), &name).await?)
                }
                _ => None,
            };

            let params = WorkflowParams {
                prompt: req.prompt.clone(),
                negative_prompt: req.negative_prompt.clone(),
                seed: random_seed(),
                width: w,
                height: h,
                filename_prefix: id.clone(),
                image_filename: uploaded,
                unet: self.img_cfg.unet.clone(),
                clip: self.img_cfg.clip.clone(),
                vae: self.img_cfg.vae.clone(),
            };
            WorkflowTemplate::apply(&mut graph, req.kind, &params);

            let (prompt_id, _number) = self.comfy.submit(&self.http_client, &graph).await?;
            tracing::info!(
                "ComfyUI executing {w}x{h} {} job: {prompt_id}",
                match req.kind {
                    WorkflowKind::TextToImage => "t2i",
                    WorkflowKind::ImageToImage => "i2i",
                }
            );

            let outputs = self
                .comfy
                .poll_outputs(
                    &self.http_client,
                    &prompt_id,
                    self.img_cfg.poll_interval_ms,
                    self.img_cfg.generation_timeout_s,
                )
                .await?;

            let image_meta = extract_output_image(&outputs)
                .ok_or_else(|| anyhow::anyhow!("comfy response had no generated image in outputs"))?;

            let bytes = self
                .comfy
                .fetch_view(
                    &self.http_client,
                    &image_meta.filename,
                    &image_meta.subfolder,
                    &image_meta.image_type,
                )
                .await?;

            let file_path = self.store.save_png(&id, &bytes)?;
            tracing::info!("Image saved to {}", file_path.display());

            Ok(GeneratedImage {
                public_url: format!("{}/images/{}.png", self.public_base, id),
                id,
                file_path,
                png_bytes: bytes,
                kind: req.kind,
            })
        }
        .await;

        // Always bring llama.cpp back, regardless of job outcome.
        let restart_attempt = self.llama.start(&self.llama_cfg, &self.http_client).await;
        if let Err(e) = &restart_attempt {
            tracing::error!("failed to restart llama.cpp after image job: {e}");
        }
        tracing::info!(
            "Image job finished in {:.1}s; llama.cpp up: {}",
            start.elapsed().as_secs_f64(),
            restart_attempt.unwrap_or(true)
        );

        outcome
    }
}

struct OutputImage {
    filename: String,
    subfolder: String,
    image_type: String,
}

/// Pull the first image entry from workflow node "9" (SaveImageAdvanced) output.
fn extract_output_image(outputs: &Value) -> Option<OutputImage> {
    let images = outputs.get("9")?.get("images")?.as_array()?;
    let first = images.first()?;
    Some(OutputImage {
        filename: first.get("filename")?.as_str()?.to_string(),
        subfolder: first.get("subfolder").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        image_type: first.get("type").and_then(|v| v.as_str()).unwrap_or("output").to_string(),
    })
}