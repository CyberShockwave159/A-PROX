use std::path::Path;
use tokenizers::Tokenizer;

pub struct FastTokenizer {
    inner: Option<Tokenizer>,
}

impl FastTokenizer {
    pub fn new<P: AsRef<Path>>(model_path: Option<P>) -> Self {
        let inner = model_path.and_then(|p| {
            if p.as_ref().exists() {
                Tokenizer::from_file(p).ok()
            } else {
                None
            }
        });

        Self { inner }
    }

    /// Counts tokens accurately using the loaded tokenizer if available,
    /// or uses a tuned BPE-approximator (approx 3.7 chars per token for code/prose).
    pub fn count_tokens(&self, text: &str) -> usize {
        if let Some(ref tokenizer) = self.inner {
            if let Ok(encoding) = tokenizer.encode(text, false) {
                return encoding.get_ids().len();
            }
        }

        // Fast fallback heuristic tuned for multilingual/code BPE:
        // ~3.5-3.8 characters per token in English + whitespace handling
        let char_count = text.chars().count();
        let word_count = text.split_whitespace().count();
        ((char_count * 2 + word_count * 5) / 7).max(1)
    }

    /// Truncates text to fit within a given token budget
    pub fn truncate_tokens(&self, text: &str, max_tokens: usize) -> String {
        if let Some(ref tokenizer) = self.inner {
            if let Ok(encoding) = tokenizer.encode(text, false) {
                let ids = encoding.get_ids();
                if ids.len() <= max_tokens {
                    return text.to_string();
                }
                // Slice token ids and decode
                let slice = &ids[..max_tokens];
                if let Ok(decoded) = tokenizer.decode(slice, true) {
                    return decoded;
                }
            }
        }

        // Fallback truncation by character estimation
        let est_chars = max_tokens * 3;
        if text.len() <= est_chars {
            text.to_string()
        } else {
            let mut end = est_chars;
            while !text.is_char_boundary(end) && end > 0 {
                end -= 1;
            }
            format!("{}... [truncated]", &text[..end])
        }
    }
}
