use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use crate::config::{
    default_agentic_examples, default_rag_ingest_examples, default_rag_search_examples, IntentConfig,
};
use crate::embeddings::CpuEmbedder;

type Embedding = Vec<f32>;

/// Internal intent classification mapped to router decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentCategory {
    AgenticTool,
    RAGSearch,
    RAGINGEST,
    Passthrough,
}

impl IntentCategory {
    fn label(&self) -> &'static str {
        match self {
            IntentCategory::AgenticTool => "agentic_tool",
            IntentCategory::RAGSearch => "rag_search",
            IntentCategory::RAGINGEST => "rag_ingest",
            IntentCategory::Passthrough => "passthrough",
        }
    }
}

/// A pre-computed intent centroid (average embedding for a category).
pub struct IntentCentroid {
    /// The intent category this centroid represents.
    pub category: IntentCategory,
    /// The average embedding vector for this category.
    pub embedding: Embedding,
    /// Similarity threshold to trigger this intent.
    pub threshold: f32,
}

/// Cached intent classification result.
struct IntentCacheEntry {
    category: IntentCategory,
    confidence: f32,
    expires_at: std::time::Instant,
}

/// Hybrid intent classifier that combines keyword matching with embedding-based
/// semantic similarity against pre-computed intent centroids.
pub struct IntentClassifier {
    centroids: Vec<IntentCentroid>,
    embedder: Arc<CpuEmbedder>,
    config: IntentConfig,
    cache: RwLock<HashMap<String, IntentCacheEntry>>,
}

impl IntentClassifier {
    /// Create a new intent classifier with centroids computed from config examples.
    pub fn new(
        embedder: Arc<CpuEmbedder>,
        config: IntentConfig,
    ) -> Self {
        let mut centroids = Vec::new();

        // Build centroids for each category that has examples. When a category's
        // configured examples are empty (e.g. a config file that sets only the
        // threshold for `[intent.categories.*]`, which serde fills with an empty
        // Vec), fall back to the built-in per-category defaults so embedding-based
        // classification keeps working.
        let agentic_examples: Vec<_> = {
            let configured = &config.categories.agentic_tool.examples;
            if configured.is_empty() {
                default_agentic_examples()
            } else {
                configured.clone()
            }
        };
        let rag_search_examples: Vec<_> = {
            let configured = &config.categories.rag_search.examples;
            if configured.is_empty() {
                default_rag_search_examples()
            } else {
                configured.clone()
            }
        };
        let rag_ingest_examples: Vec<_> = {
            let configured = &config.categories.rag_ingest.examples;
            if configured.is_empty() {
                default_rag_ingest_examples()
            } else {
                configured.clone()
            }
        };

        if !agentic_examples.is_empty() {
            let centroid = build_centroid(&embedder, &agentic_examples);
            centroids.push(IntentCentroid {
                category: IntentCategory::AgenticTool,
                embedding: centroid,
                threshold: config.categories.agentic_tool.threshold,
            });
        }

        if !rag_search_examples.is_empty() {
            let centroid = build_centroid(&embedder, &rag_search_examples);
            centroids.push(IntentCentroid {
                category: IntentCategory::RAGSearch,
                embedding: centroid,
                threshold: config.categories.rag_search.threshold,
            });
        }

        if !rag_ingest_examples.is_empty() {
            let centroid = build_centroid(&embedder, &rag_ingest_examples);
            centroids.push(IntentCentroid {
                category: IntentCategory::RAGINGEST,
                embedding: centroid,
                threshold: config.categories.rag_ingest.threshold,
            });
        }

        tracing::info!(
            "Intent classifier initialized with {} category centroids",
            centroids.len()
        );

        Self {
            centroids,
            embedder,
            config,
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Classify the intent of a message using hybrid keyword + embedding approach.
    ///
    /// Returns (IntentCategory, confidence_score).
    pub fn classify(&self, message: &str) -> (IntentCategory, f32) {
        let trimmed = message.trim();

        // Step 1: Fast keyword pass
        if let Some(intent) = self.keyword_match(trimmed) {
            tracing::info!(
                "Intent classified via keyword: {} (confidence: {:.2})",
                intent.label(),
                0.95
            );
            return (intent, 0.95);
        }

        // Step 2: Check cache
        let msg_hash = self.hash_message(trimmed);
        {
            let cache = self.cache.read().unwrap();
            if let Some(entry) = cache.get(&msg_hash) {
                if entry.expires_at > std::time::Instant::now() {
                    tracing::info!(
                        "Intent from cache: {} (confidence: {:.2})",
                        entry.category.label(),
                        entry.confidence
                    );
                    return (entry.category, entry.confidence);
                }
            }
        }

        // Step 3: Embedding-based classification
        let embedding = match self.embedder.embed_text(trimmed) {
            Ok(embedding) => embedding,
            Err(e) => {
                tracing::warn!("Failed to compute embedding for intent classification: {}", e);
                return (IntentCategory::Passthrough, 0.0);
            }
        };

        let (best_category, best_confidence) = self.find_best_match(&embedding);

        // Log the classification
        tracing::info!(
            "Intent classified via embedding: {} (confidence: {:.2}, threshold: {:.2})",
            best_category.label(),
            best_confidence,
            self.config.default_threshold
        );

        // Cache the result (24h TTL)
        let entry = IntentCacheEntry {
            category: best_category,
            confidence: best_confidence,
            expires_at: std::time::Instant::now() + std::time::Duration::from_secs(86400),
        };
        self.cache.write().unwrap().insert(msg_hash, entry);

        (best_category, best_confidence)
    }

    /// Clean up expired cache entries periodically.
    pub fn cleanup_cache(&self) {
        let now = std::time::Instant::now();
        let cache = self.cache.read().unwrap();
        let expired_count = cache.values().filter(|e| e.expires_at <= now).count();
        if expired_count > 0 {
            drop(cache);
            let mut w = self.cache.write().unwrap();
            w.retain(|_, v| v.expires_at > now);
            tracing::debug!("Intent cache cleanup: {} expired entries removed", expired_count);
        }
    }

    /// Fast keyword-based intent matching (fast path, no embedding needed).
    fn keyword_match(&self, message: &str) -> Option<IntentCategory> {
        let msg = message.to_lowercase();

        // Agentic tool keywords
        let agentic_patterns = [
            "search the web", "check the web", "search online", "search the internet",
            "web search", "google for", "look up online", "look up the weather",
            "latest news", "lastest news", "breaking news", "latest headlines",
            "news today", "what's new", "what's happening", "current time",
            "what time is it", "what day is it", "system time",
            "fetch", "summarize", "read this link",
        ];
        for pattern in &agentic_patterns {
            if msg.contains(pattern) {
                return Some(IntentCategory::AgenticTool);
            }
        }

        // RAG ingest keywords — MUST be checked before RAG search, because some
        // ingest phrasings (e.g. "add this to my notes") also match search keywords
        // like "my notes". Ingestion is about *storing* content, not querying it.
        let rag_ingest_patterns = [
            "ingest into rag", "save to knowledge base", "index this",
            "index this into", "index this for", "store in rag",
            "store this in knowledge base", "store this in the rag",
            "store this into the rag", "save this to the rag",
            "store this in rag", "save this in the rag", "save this into the rag",
            "store this in", "store this into", "store this to",
            "save this to", "save this in", "save this into",
            // bare verb forms — tolerate an intervening object:
            // "store this data into the RAG database", "save this text to ..."
            "store this ", "save this ", "add this ", "store it ", "save it ",
            "add this to the knowledge base", "store this in my knowledge base",
            "save this to my knowledge base", "add this to my notes",
            "store this in my notes", "save this to my notes",
            "remember this", "remember that", "take a note",
            "store the following", "save the following", "store the below",
            "into the vector store", "to the vector store", "in the vector store",
            "save it to", "store it in", "store it into",
        ];
        for pattern in &rag_ingest_patterns {
            if msg.contains(pattern) {
                return Some(IntentCategory::RAGINGEST);
            }
        }

        // RAG search keywords
        let rag_search_patterns = [
            "/rag", "/docs", "/knowledge", "search docs",
            "in knowledge base", "from my files", "my notes", "my docs",
            "search knowledge base",
        ];
        for pattern in &rag_search_patterns {
            if msg.contains(pattern) {
                return Some(IntentCategory::RAGSearch);
            }
        }

        None
    }

    /// Find the best matching centroid for an embedding.
    fn find_best_match(&self, embedding: &Embedding) -> (IntentCategory, f32) {
        let mut best: Option<&IntentCentroid> = None;
        let mut best_score: f32 = 0.0;

        for centroid in self.centroids.iter() {
            let score = cosine_similarity(embedding, &centroid.embedding);

            if score > best_score {
                best_score = score;
                best = Some(centroid);
            }
        }

        match best {
            Some(centroid) => {
                if best_score > centroid.threshold {
                    (centroid.category, best_score)
                } else {
                    (IntentCategory::Passthrough, best_score)
                }
            }
            None => (IntentCategory::Passthrough, best_score),
        }
    }

    fn hash_message(&self, message: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        message.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

/// Build a centroid (average embedding) from a list of example texts.
fn build_centroid(embedder: &Arc<CpuEmbedder>, examples: &[crate::config::IntentExample]) -> Embedding {
    let dim = embedder.dimension();
    let mut accumulator = vec![0.0_f32; dim];
    let mut count: usize = 0;

    for example in examples {
        match embedder.embed_text(&example.text) {
            Ok(embedding) => {
                if embedding.len() == dim {
                    for i in 0..dim {
                        accumulator[i] += embedding[i];
                    }
                    count += 1;
                }
            }
            Err(e) => {
                tracing::warn!("Failed to compute embedding for intent example: {}", e);
            }
        }
    }

    if count == 0 {
        return vec![0.0_f32; dim];
    }

    let scalar = 1.0 / count as f32;
    accumulator.iter_mut().for_each(|v| *v *= scalar);
    vector_normalize(&accumulator)
}

/// L2-normalize a vector.
fn vector_normalize(vec: &[f32]) -> Vec<f32> {
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        return vec![0.0; vec.len()];
    }
    vec.iter().map(|x| x / norm).collect()
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    (dot_product / denom).max(0.0).min(1.0) // Clamp to [0, 1] since embeddings are positive
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::CpuEmbedder;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_positive_angle() {
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.5];
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.9);
    }

    #[test]
    fn test_vector_normalize() {
        let v = vec![3.0, 4.0];
        let normalized = vector_normalize(&v);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_hash_consistency() {
        let classifier = IntentClassifier {
            centroids: Vec::new(),
            embedder: Arc::new(CpuEmbedder::new::<&str>(
                "models/bge-small-en-v1.5-int8.onnx",
                None,
                1,
                384,
            )),
            config: IntentConfig::default(),
            cache: RwLock::new(HashMap::new()),
        };
        let hash1 = classifier.hash_message("hello world");
        let hash2 = classifier.hash_message("hello world");
        let hash3 = classifier.hash_message("goodbye world");
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}
