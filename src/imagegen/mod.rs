#[cfg(test)]
mod tests;

pub mod orchestrator;
pub mod ratio;

pub use orchestrator::{GeneratedImage, GenerateRequest, ImageGenService};
pub use ratio::compute_dimensions;

use crate::comfy_ui::WorkflowKind;
use crate::context::ChatMessage;
use crate::tools::ExtractedToolCall;

/// Splendid image-generator system prompt (text-to-image).
/// The model must emit its revision as a single JSON object — nothing else.
pub const T2I_SYSTEM_PROMPT_EMBEDDED: &str = include_str!("../../prompts/t-iprompt.txt");

/// Splendid image-generator system prompt (image-to-image).
pub const I2I_SYSTEM_PROMPT_EMBEDDED: &str = include_str!("../../prompts/i-iprompt.txt");

/// Directive appended after the verbatim prompt file so the harness can detect a
/// valid tool call even when llama.cpp's structured-output block misformats.
pub const HARNESS_DIRECTIVE: &str =
    "\n\nYou MUST emit your JSON as the arguments of a call to the image_generate tool. \
     Produce exactly ONE bare JSON object, on a SINGLE line, with no surrounding prose, \
     no markdown fences, and no text before or after it. If you cannot, emit only the \
     JSON object alone.";

/// Prefix keywords that (with a post/image-phrase) suggest a text-to-image request.
pub const T2I_KEYWORDS: &[&str] = &[
    "generate an image",
    "generate a picture",
    "generate a photo",
    "create an image",
    "create a picture",
    "create a photo",
    "draw me",
    "draw a",
    "make an image",
    "make a picture",
    "make a photo",
    "render a",
    "render an",
    "design a logo",
    "generate a logo",
    "make a logo",
    "produce an image",
    "produce a picture",
    "illustrate a",
    "create a rendering",
    "generate a visual",
    "create a visual",
    "make me a",
    "imagine a",
    "generate an artwork",
    "create an artwork",
    "make an artwork",
    "make a painting",
    "create a painting",
    "make a drawing",
    "make a sketch",
    "paint this",
    "turn this sketch",
    "turn this drawing",
    "turn this doodle",
];

/// Keywords that suggest editing/transforming the attached image.
pub const I2I_KEYWORDS: &[&str] = &[
    "edit",
    "redesign",
    "restyle",
    "recolor",
    "recolour",
    "transform",
    "turn this",
    "turn it",
    "convert this",
    "change this",
    "change the",
    "add snow",
    "add rain",
    "change the background",
    "remove the background",
    "make it a painting",
    "make it an illustration",
    "style the photo",
    "colorize",
    "colourise",
    "colorize this",
    "enhance this photo",
    "fix this photo",
    "upscale this image",
    "combine these",
    "merge these",
    "blend these",
    "in this photo",
    "in this picture",
    "in this image",
];

impl ChatMessage {
    /// True when the message content contains an `image_url` part.
    pub fn has_image_parts(&self) -> bool {
        match &self.content {
            Some(serde_json::Value::Array(parts)) => parts.iter().any(|p| {
                p.get("type").and_then(|t| t.as_str()) == Some("image_url")
            }),
            _ => false,
        }
    }

    /// Extract (mime, base64 payload) from the first image_url part, if any.
    pub fn extract_image_part(&self) -> Option<(&str, &str)> {
        match &self.content {
            Some(serde_json::Value::Array(parts)) => parts.iter().find_map(|p| {
                let url = p.get("image_url")?.get("url")?.as_str()?;
                split_data_url(url)
            }),
            _ => None,
        }
    }
}

/// Split `data:<mime>;base64,<data>` into (mime, payload).
fn split_data_url(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, payload) = rest.split_once(',')?;
    let mime = meta
        .strip_suffix(";base64")
        .or_else(|| meta.strip_suffix(",base64"))
        .unwrap_or("application/octet-stream");
    Some((mime, payload))
}

/// Decode + decode the first attached image into bytes and its pixel dimensions.
/// Returns None when no valid (data-URL or decodable) image is attached.
pub fn extract_reference_image(msg: &ChatMessage) -> Option<(Vec<u8>, (u32, u32))> {
    use base64::engine::general_purpose::STANDARD as B64;
    use base64::Engine;

    let (_mime, payload) = msg.extract_image_part()?;
    let bytes = B64.decode(payload).ok()?;
    let dims = image_dimensions(&bytes)?;
    Some((bytes, dims))
}

/// Decode image bytes into (width, height) using the `image` crate.
pub fn image_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    let img = image::load_from_memory_with_format(bytes, guess_format(bytes)?).ok()?;
    Some((img.width(), img.height()))
}

fn guess_format(bytes: &[u8]) -> Option<image::ImageFormat> {
    use image::ImageFormat;
    if bytes.starts_with(b"\x89PNG") {
        Some(ImageFormat::Png)
    } else if bytes.starts_with(b"\xFF\xD8") {
        Some(ImageFormat::Jpeg)
    } else if bytes.starts_with(b"RIFF") {
        Some(ImageFormat::WebP)
    } else {
        None
    }
}

/// Build a data-URL for injecting the reference image back into the agent loop.
pub fn to_data_url(bytes: &[u8]) -> String {
    use base64::engine::general_purpose::STANDARD as B64;
    use base64::Engine;

    let mime = match guess_format(bytes) {
        Some(image::ImageFormat::Jpeg) => "image/jpeg",
        Some(image::ImageFormat::WebP) => "image/webp",
        _ => "image/png",
    };
    format!("data:{};base64,{}", mime, B64.encode(bytes))
}

/// Load the workflow-specific image-generator prompt file, falling back to the
/// embedded copy if the path is missing at runtime.
pub fn load_prompt_file(kind: WorkflowKind, config_path: &str) -> String {
    let path = config_path.trim();
    if !path.is_empty() {
        if let Ok(contents) = std::fs::read_to_string(path) {
            return contents;
        }
        tracing::warn!("prompt file {path} unreadable; using embedded prompt");
    }
    match kind {
        WorkflowKind::TextToImage => T2I_SYSTEM_PROMPT_EMBEDDED.to_string(),
        WorkflowKind::ImageToImage => I2I_SYSTEM_PROMPT_EMBEDDED.to_string(),
    }
}

/// Detect whether the trailing user turn (image parts + verb keywords) is an
/// image-generation request. Returns the workflow kind when confident.
pub fn detect_image_request(messages: &[ChatMessage]) -> Option<WorkflowKind> {
    let latest = messages.iter().rev().find(|m| m.role == "user")?;
    let text = latest.content_as_str().to_lowercase();

    if latest.has_image_parts() {
        // Attached image + any editing/generation phrase → i2i (never bare t2i).
        if contains_any(&text, I2I_KEYWORDS) || contains_any(&text, T2I_KEYWORDS) {
            return Some(WorkflowKind::ImageToImage);
        }
        return None;
    }

    if contains_any(&text, T2I_KEYWORDS) {
        return Some(WorkflowKind::TextToImage);
    }
    None
}

/// Cheap whole-request signal used by the router to force the agentic loop for
/// image requests regardless of the 0.70 intent threshold: attached image parts
/// or any t2i/i2i verb phrase in the latest user turn.
pub fn is_image_request(messages: &[ChatMessage]) -> bool {
    detect_image_request(messages).is_some()
}

fn contains_any(text: &str, keywords: &[&str]) -> bool {
    keywords.iter().any(|k| text.contains(k))
}

/// Fallback harness: parse the model's `image_generate` JSON out of response
/// content when no structured tool call was emitted (bare single-line JSON, with
/// optional surrounding prose). Returns a synthetic tool call when found.
pub fn try_parse_image_generate_json(content: &str) -> Option<ExtractedToolCall> {
    let trimmed = content.trim();

    let candidate = |v: &serde_json::Value| -> Option<ExtractedToolCall> {
        let obj = v.as_object()?;
        if !obj.contains_key("rewritten_prompt") {
            return None;
        }
        Some(ExtractedToolCall {
            id: "call_img_bare".to_string(),
            name: "image_generate".to_string(),
            arguments: v.clone(),
        })
    };

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(call) = candidate(&v) {
            return Some(call);
        }
    }

    // The model wrapped the JSON in prose or fences: take the outermost {...} span.
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end <= start {
        return None;
    }
    let slice = &trimmed[start..=end];
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(slice) {
        if let Some(call) = candidate(&v) {
            return Some(call);
        }
    }
    None
}

/// The text a CLAN-AI2 user sends describing an image edit.
pub fn describe_image_request(kind: WorkflowKind) -> &'static str {
    match kind {
        WorkflowKind::TextToImage => "an image request (text-to-image)",
        WorkflowKind::ImageToImage => "an image request with an attached reference photo (image-to-image)",
    }
}

/// Dummy reference for the router intent mapping.
pub fn image_generation_intent_label() -> &'static str {
    "image_generation"
}