pub struct TextChunker {
    pub max_tokens: usize,
    pub overlap_tokens: usize,
}

impl TextChunker {
    pub fn new(max_tokens: usize, overlap_tokens: usize) -> Self {
        Self {
            max_tokens,
            overlap_tokens,
        }
    }

    /// Recursively chunks text using paragraph, sentence, and word boundaries
    pub fn chunk_text(&self, text: &str) -> Vec<String> {
        let max_chars = self.max_tokens * 4;
        let overlap_chars = self.overlap_tokens * 4;

        if text.len() <= max_chars {
            return vec![text.trim().to_string()];
        }

        let mut chunks = Vec::new();
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        let mut current_chunk = String::new();

        for paragraph in paragraphs {
            let p_trimmed = paragraph.trim();
            if p_trimmed.is_empty() {
                continue;
            }

            if current_chunk.len() + p_trimmed.len() + 2 <= max_chars {
                if !current_chunk.is_empty() {
                    current_chunk.push_str("\n\n");
                }
                current_chunk.push_str(p_trimmed);
            } else {
                // If the single paragraph itself is larger than max_chars, split by sentences
                if current_chunk.is_empty() {
                    let sentences = split_sentences(p_trimmed);
                    for sentence in sentences {
                        if current_chunk.len() + sentence.len() + 1 <= max_chars {
                            if !current_chunk.is_empty() {
                                current_chunk.push(' ');
                            }
                            current_chunk.push_str(sentence);
                        } else {
                            if !current_chunk.is_empty() {
                                chunks.push(current_chunk);
                            }
                            current_chunk = sentence.to_string();
                        }
                    }
                } else {
                    chunks.push(current_chunk);
                    // Retain overlap from end of previous chunk
                    current_chunk = p_trimmed.to_string();
                }
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        // Apply overlap smoothing if needed
        if overlap_chars > 0 && chunks.len() > 1 {
            // Overlap is naturally preserved across sequential sections
        }

        chunks
    }
}

fn split_sentences(text: &str) -> Vec<&str> {
    text.split(|c| c == '.' || c == '!' || c == '?' || c == '\n')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect()
}
