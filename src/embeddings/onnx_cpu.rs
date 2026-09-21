use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use std::sync::Mutex;

pub struct CpuEmbedder {
    session: Option<Arc<Mutex<Session>>>,
    tokenizer: Option<Tokenizer>,
    dimension: usize,
}

impl CpuEmbedder {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        tokenizer_path: Option<P>,
        threads: usize,
        dimension: usize,
    ) -> Self {
        let model_path = model_path.as_ref();
        let session = if model_path.exists() {
            tracing::info!("Initializing CPU ONNX Session from: {:?}", model_path);
            let res: Result<Session, ort::Error> = (|| {
                let mut builder = Session::builder()?;
                builder = builder.with_intra_threads(threads)?;
                builder = builder.with_optimization_level(GraphOptimizationLevel::Level3)?;
                builder.commit_from_file(model_path)
            })();

            match res {
                Ok(s) => {
                    tracing::info!("Successfully loaded CPU ONNX Embedding session");
                    Some(Arc::new(Mutex::new(s)))
                }
                Err(e) => {
                    tracing::warn!("Failed to load ONNX model at {:?}: {}. Falling back to heuristic embedding.", model_path, e);
                    None
                }
            }
        } else {
            tracing::warn!("ONNX model file not found at {:?}. Using heuristic embedding fallback.", model_path);
            None
        };

        let tokenizer = tokenizer_path.and_then(|p| {
            let p = p.as_ref();
            if p.exists() {
                Tokenizer::from_file(p).ok()
            } else {
                None
            }
        });

        Self {
            session,
            tokenizer,
            dimension,
        }
    }

    /// Generates a normalized embedding vector for the input text
    pub fn embed_text(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        if let (Some(session), Some(tokenizer)) = (&self.session, &self.tokenizer) {
            let encoding = tokenizer.encode(text, true)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;

            let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
            let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
            let seq_len = input_ids.len();

            let input_ids_tensor = Tensor::from_array((vec![1usize, seq_len], input_ids))?;
            let attention_mask_tensor = Tensor::from_array((vec![1usize, seq_len], attention_mask))?;
            let token_type_ids: Vec<i64> = vec![0; seq_len];
            let token_type_ids_tensor = Tensor::from_array((vec![1usize, seq_len], token_type_ids))?;

            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ];

            let mut session_guard = session.lock().map_err(|e| anyhow::anyhow!("Session lock poisoned: {e}"))?;
            let outputs = session_guard.run(inputs)?;
            let (_shape, slice) = outputs[0].try_extract_tensor::<f32>()?;

            // Mean pooling across sequence tokens
            let mut pooled = vec![0.0f32; self.dimension];
            let mut sum_mask = 0.0f32;

            for token_idx in 0..seq_len {
                let mask_val = encoding.get_attention_mask()[token_idx] as f32;
                if mask_val > 0.0 {
                    sum_mask += mask_val;
                    let offset = token_idx * self.dimension;
                    for dim_idx in 0..self.dimension {
                        if offset + dim_idx < slice.len() {
                            pooled[dim_idx] += slice[offset + dim_idx];
                        }
                    }
                }
            }

            if sum_mask > 0.0 {
                for v in &mut pooled {
                    *v /= sum_mask;
                }
            }

            // L2 normalize
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for v in &mut pooled {
                *v /= norm;
            }

            Ok(pooled)
        } else {
            // Deterministic CPU heuristic embedding fallback
            Ok(self.heuristic_embedding(text))
        }
    }

    /// Fast, deterministic hash-based embedding fallback for testing/initial boot
    fn heuristic_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dimension];
        let bytes = text.as_bytes();

        for (i, &b) in bytes.iter().enumerate() {
            let idx = (i * 31 + b as usize) % self.dimension;
            embedding[idx] += ((b as f32) / 255.0) - 0.5;
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for v in &mut embedding {
            *v /= norm;
        }

        embedding
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}
