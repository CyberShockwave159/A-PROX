use crate::context::ChatMessage;
use crate::tools::ExtractedToolCall;

/// Directive merged into the leading system message when a file-write request is
/// detected (mirrors the image harness). Kept inline so the model knows to call
/// `write_file` and to split oversized content via `mode=append`.
pub const FILE_DIRECTIVE: &str =
    "\n\nWhen the user wants text saved to a file (scripts, notes, stories, summaries, markdown, \
     code, CSV, etc.), you MUST call the write_file tool with 'filename' (a name ending in a \
     suitable extension) and 'content' (the complete file body). Do NOT paste the file content \
     into your visible reply instead of calling the tool — if write_file is not called, no file \
     will be created. If the content is too large for one call, split it into chunks: first chunk \
     with mode=\"overwrite\", the rest with mode=\"append\" (same filename). After the final \
     chunk, keep your visible reply brief: say what the file is and give its URL.";

/// Phrases that gate the `write_file` tool; checked against the latest user turn.
pub const FILE_KEYWORDS: &[&str] = &[
    "write a file",
    "write to a file",
    "write me a file",
    "create a file",
    "create a text file",
    "create a markdown file",
    "write a text file",
    "write a markdown file",
    "write an md file",
    "create an md file",
    "write a python script",
    "create a python script",
    "write a script",
    "create a script",
    "write a json file",
    "create a json file",
    "write a csv file",
    "write a notes file",
    "create a notes file",
    "write this to a file",
    "save this to a file",
    "save this as a file",
    "write this to",
    "save as a .",
    "output to a file",
    "make a notes file",
    "as a markdown file",
    "as a text file",
    "as a json file",
    "as a python script",
    "as a .md file",
    "as a .txt file",
];

/// Writing verbs that, combined with a file cue, signal a file-write request.
const FILE_VERBS: &[&str] = &[
    "write", "create", "save", "generate", "output", "save this", "write this",
    "turn this into", "convert this to", "put this in", "dump this to", "append",
];

/// File-ish cues used in the two-tier signal check (verb + cue).
const FILE_CUES: &[&str] = &[
    "to a file", "into a file", "to a text file", "to a markdown file", "to a json file",
    "to a file called", "in a file", "on file", "a file named", "as a file", "file called",
    "file named", "same file", "write_file", "using write_file", "use write_file",
    "call write_file", "call the write_file", "markdown file", "text file", "json file",
    "csv file", "notes file", "md file", "python script", "shell script", "bash script",
    "javascript file", "todo list file", "script file", "a .md", "a .txt", "a .py",
    "a .json", "a .csv",
];

/// Cheap whole-request signal used by the router to force the agentic loop for
/// file-write requests regardless of the intent threshold.
pub fn is_file_request(messages: &[ChatMessage]) -> bool {
    let latest = messages.iter().rev().find(|m| m.role == "user");
    let Some(latest) = latest else { return false };
    let text = latest.content_as_str().to_lowercase();
    if FILE_KEYWORDS.iter().any(|k| text.contains(k)) {
        return true;
    }
    // Two-tier signal: a file-writing verb AND a file cue in the same request,
    // e.g. "write a short python script to a file" (intervening adjectives).
    let verb = FILE_VERBS.iter().any(|v| text.contains(v));
    let cue = FILE_CUES.iter().any(|c| text.contains(c));
    verb && cue
}

/// Sanitize a user/model-supplied filename into a safe base name with an
/// extension. Returns `file.txt` when nothing usable survives.
pub fn clean_filename(raw: &str) -> String {
    let raw = raw.trim();
    let base = raw.rsplit(['/', '\\']).next().unwrap_or(raw).trim();
    let mut out = String::new();
    for (i, c) in base.chars().enumerate() {
        if i >= 60 {
            break;
        }
        if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+') {
            out.push(c);
        } else if c.is_whitespace() {
            out.push('_');
        }
    }
    let out = out.trim_start_matches('.').replace("..", "_");
    let out = if out.is_empty() { "file".to_string() } else { out };
    if out.contains('.') {
        out
    } else {
        format!("{out}.txt")
    }
}

/// Extension of a cleaned filename (lowercase, no dot), or "" if none.
pub fn extension_of(name: &str) -> String {
    let lower = name.to_lowercase();
    match lower.rsplit_once('.') {
        Some((_, ext)) if !ext.is_empty() && !ext.contains([' ', '/', '\\']) => ext.to_string(),
        _ => String::new(),
    }
}

const DEFAULT_DENIED_EXTS: &[&str] = &[
    "exe", "sh", "bat", "com", "cmd", "dll", "so", "sys", "html", "htm", "php", "jar", "lnk",
    "scr", "swf",
];

/// True when the cleaned filename's extension is on the deny list (executables,
/// script launchers, documents that could run active content).
pub fn is_denied_extension(filename: &str, deny_exts: &[String]) -> bool {
    let ext = extension_of(filename);
    let denied: Vec<String> = if deny_exts.is_empty() {
        DEFAULT_DENIED_EXTS.iter().map(|e| e.to_string()).collect()
    } else {
        deny_exts
            .iter()
            .map(|e| e.trim().trim_start_matches('.').to_lowercase())
            .collect()
    };
    denied.iter().any(|d| ext == *d)
}

/// Fallback harness: parse the model's `write_file` JSON (filename + content)
/// out of response content when no structured tool call was emitted.
pub fn try_parse_write_file_json(content: &str) -> Option<ExtractedToolCall> {
    let trimmed = content.trim();

    let candidate = |v: &serde_json::Value| -> Option<ExtractedToolCall> {
        let obj = v.as_object()?;
        if !obj.contains_key("filename") || !obj.contains_key("content") {
            return None;
        }
        Some(ExtractedToolCall {
            id: "call_file_bare".to_string(),
            name: "write_file".to_string(),
            arguments: v.clone(),
        })
    };

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(call) = candidate(&v) {
            return Some(call);
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(serde_json::json!(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            roleplay: None,
        }
    }

    #[test]
    fn detects_file_requests() {
        let yes = [
            "Please write a python script that reads a csv",
            "Create a markdown file with my meeting notes",
            "save this to a file",
            "write this to a file called poem.txt",
        ];
        for q in yes {
            assert!(is_file_request(&[msg("user", q)]), "missed: {q}");
        }
        let no = [
            "summarize the meeting",
            "what is 2+2?",
            "save this to my knowledge base",
            "explain this python script to me",
        ];
        for q in no {
            assert!(!is_file_request(&[msg("user", q)]), "false positive: {q}");
        }
    }

    #[test]
    fn detects_files_with_intervening_words() {
        let yes = [
            "Please write a short python script to a file that prints fibonacci",
            "Can you write a nice markdown file of my meeting notes?",
            "generate a todo list file please",
            "write a small bash script into a file",
        ];
        for q in yes {
            assert!(is_file_request(&[msg("user", q)]), "missed: {q}");
        }
    }

    #[test]
    fn cleans_filenames() {
        assert_eq!(clean_filename("story.txt"), "story.txt");
        assert_eq!(clean_filename("../evil/..//notes .md"), "notes_.md");
        assert_eq!(clean_filename("no extension"), "no_extension.txt");
        assert_eq!(clean_filename(&("a".repeat(200) + ".py")).len() < 80, true);
        assert_eq!(clean_filename(""), "file.txt");
        assert_eq!(clean_filename(".hidden"), "hidden.txt");
    }

    #[test]
    fn denies_dangerous_extensions() {
        assert!(is_denied_extension("evil.sh", &[]));
        assert!(is_denied_extension("evil.html", &[]));
        assert!(!is_denied_extension("video.md", &[]));
        assert!(!is_denied_extension("main.py", &[]));
        assert!(is_denied_extension("a.exe", &["exe".into()]));
    }

    #[test]
    fn parses_bare_file_json() {
        let call = try_parse_write_file_json(
            r#"Sure! Here is the script: {"filename":"solver.py","content":"print('hi')"}"#,
        )
        .unwrap();
        assert_eq!(call.name, "write_file");
        assert_eq!(call.arguments["filename"], "solver.py");
        assert_eq!(call.arguments["content"], "print('hi')");
        assert!(try_parse_write_file_json("no tools here").is_none());
    }
}