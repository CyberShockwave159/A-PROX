use super::*;

fn sys_msg(text: &str) -> ChatMessage {
    ChatMessage {
        role: "system".to_string(),
        content: Some(serde_json::Value::String(text.to_string())),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn user_msg(text: &str) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: Some(serde_json::Value::String(text.to_string())),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn user_msg_with_image(text: &str, data_url: &str) -> ChatMessage {
    use serde_json::json;
    ChatMessage {
        role: "user".to_string(),
        content: Some(json!([
            { "type": "text", "text": text },
            { "type": "image_url", "image_url": { "url": data_url } }
        ])),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn tiny_png() -> Vec<u8> {
    use base64::engine::general_purpose::STANDARD as B64;
    use base64::Engine;
    // 1x1 red PNG
    B64
        .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
        .unwrap()
}

#[test]
fn split_data_url_works() {
    let (mime, payload) = split_data_url("data:image/png;base64,AAAA").unwrap();
    assert_eq!(mime, "image/png");
    assert_eq!(payload, "AAAA");
    assert!(split_data_url("http://x/y.png").is_none());
}

#[test]
fn detect_t2i() {
    let msgs = vec![sys_msg("system"), user_msg("generate an image of a cat astronaut")];
    assert_eq!(detect_image_request(&msgs), Some(WorkflowKind::TextToImage));
}

#[test]
fn detect_t2i_with_image_is_i2i() {
    let data = format!("data:image/png;base64,{}", {
        use base64::engine::general_purpose::STANDARD as B64;
        use base64::Engine;
        B64.encode(tiny_png())
    });
    let msgs = vec![sys_msg("s"), user_msg_with_image("make a painting of a dragon", &data)];
    assert_eq!(detect_image_request(&msgs), Some(WorkflowKind::ImageToImage));
}

#[test]
fn detect_plain_question_is_none() {
    let msgs = vec![sys_msg("s"), user_msg("what is the capital of France?")];
    assert_eq!(detect_image_request(&msgs), None);
}

#[test]
fn extract_reference_image_dimensions() {
    let data = format!("data:image/png;base64,{}", {
        use base64::engine::general_purpose::STANDARD as B64;
        use base64::Engine;
        B64.encode(tiny_png())
    });
    let msg = user_msg_with_image("edit this", &data);
    let (bytes, dims) = extract_reference_image(&msg).unwrap();
    assert_eq!(dims.0 as u64, 1);
    assert_eq!(dims.1 as u64, 1);
    assert!(!bytes.is_empty());
}

#[test]
fn has_image_parts_is_true_for_data_image() {
    let data = format!("data:image/png;base64,{}", {
        use base64::engine::general_purpose::STANDARD as B64;
        use base64::Engine;
        B64.encode(tiny_png())
    });
    let msg = user_msg_with_image("go", &data);
    assert!(msg.has_image_parts());
}

#[test]
fn to_data_url_roundtrips_format() {
    let url = to_data_url(&tiny_png());
    assert!(url.starts_with("data:image/png;base64,"));
}

#[test]
fn image_served_url_joins_base_and_id() {
    assert_eq!(image_served_url("http://host:8000", "gen_1"), "http://host:8000/images/gen_1.png");
    assert_eq!(image_served_url("https://example.com/", "gen_2"), "https://example.com/images/gen_2.png");
}

#[test]
fn compute_dims_are_stable() {
    for (r, ref_dims) in [
        (None, None),
        (Some("16:9"), None),
        (Some("2:3"), None),
        (Some("1:1"), Some((1200, 900))),
    ] {
        let (w, h) = compute_dimensions(r, ref_dims, 2.0, 16, 4096);
        assert_eq!(w % 16, 0, "{w:?}");
        assert_eq!(h % 16, 0, "{h:?}");
        assert!(w >= 1 && h >= 1);
        assert!(w <= 4096 && h <= 4096);
    }
}

#[test]
fn prompt_files_load() {
    use crate::imagegen::load_prompt_file;
    let t2i = load_prompt_file(WorkflowKind::TextToImage, "");
    let i2i = load_prompt_file(WorkflowKind::ImageToImage, "");
    assert!(!t2i.is_empty());
    assert!(!i2i.is_empty());
    assert!(!T2I_SYSTEM_PROMPT_EMBEDDED.is_empty());
    assert!(!I2I_SYSTEM_PROMPT_EMBEDDED.is_empty());
}