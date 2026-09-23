use serde_json::{json, Value};

/// Which ComfyUI workflow template to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowKind {
    TextToImage,
    ImageToImage,
}

/// Runtime values baked into a workflow template before submission.
pub struct WorkflowParams {
    pub prompt: String,
    pub negative_prompt: String,
    /// ComfyUI KSampler seed. Must be >= 0 (the workflow schema rejects -1),
    /// so A-PROX supplies a per-job random value instead.
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub filename_prefix: String,
    /// Uploaded reference image filename (image-to-image only).
    pub image_filename: Option<String>,
    pub unet: String,
    pub clip: String,
    pub vae: String,
}

/// Loads a workflow API JSON document.
pub struct WorkflowTemplate;

impl WorkflowTemplate {
    pub fn load(path: &str) -> anyhow::Result<Value> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read workflow template {path}: {e}"))?;
        let template: Value = serde_json::from_str(&content)
            .map_err(|e| anyhow::anyhow!("invalid workflow template {path}: {e}"))?;
        if !template.is_object() {
            anyhow::bail!("workflow template {path} must be a JSON object of nodes");
        }
        Ok(template)
    }

    /// Inject runtime values into a workflow graph by mutating the node inputs.
    /// Node ids follow the verified t2i/i2i templates:
    /// 1 = UnetLoaderGGUF, 2 = CLIPLoader, 5 = KSampler, 6 = VAELoader,
    /// 8 = TextEncodeQwenImage21, 9 = SaveImageAdvanced, 26 = EmptyLatentImage
    /// (t2i), 11/32 = LoadImage + ResizeImageMaskNode (i2i).
    pub fn apply(template: &mut Value, kind: WorkflowKind, p: &WorkflowParams) {
        set_input(template, "1", "unet_name", json!(p.unet));
        set_input(template, "2", "clip_name", json!(p.clip));
        set_input(template, "2", "type", json!("qwen_image"));
        set_input(template, "6", "vae_name", json!(p.vae));
        set_input(template, "5", "seed", json!(p.seed));
        set_input(template, "8", "prompt", json!(p.prompt));
        set_input(template, "8", "negative_prompt", json!(p.negative_prompt));
        set_input(template, "9", "filename_prefix", json!(p.filename_prefix));
        set_input(template, "9", "format", json!("png"));

        match kind {
            WorkflowKind::TextToImage => {
                set_input(template, "26", "width", json!(p.width));
                set_input(template, "26", "height", json!(p.height));
                set_input(template, "26", "batch_size", json!(1));
            }
            WorkflowKind::ImageToImage => {
                if let Some(name) = &p.image_filename {
                    set_input(template, "11", "image", json!(name));
                }
                set_input(template, "32", "resize_type.width", json!(p.width));
                set_input(template, "32", "resize_type.height", json!(p.height));
            }
        }
    }
}

fn set_input(template: &mut Value, node_id: &str, key: &str, value: Value) {
    if let Some(node) = template.get_mut(node_id) {
        if let Some(inputs) = node.get_mut("inputs") {
            if let Some(obj) = inputs.as_object_mut() {
                obj.insert(key.to_string(), value);
            }
        }
    }
}