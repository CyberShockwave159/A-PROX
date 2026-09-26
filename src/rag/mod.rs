pub mod chunker;

use std::sync::Arc;
use crate::context::FastTokenizer;
use crate::db::{SearchResult, VectorStore};
use crate::embeddings::CpuEmbedder;
pub use chunker::TextChunker;

pub struct RagEngine {
    vector_store: Arc<VectorStore>,
    embedder: Arc<CpuEmbedder>,
    tokenizer: Arc<FastTokenizer>,
    chunker: TextChunker,
}

impl RagEngine {
    pub fn new(
        vector_store: Arc<VectorStore>,
        embedder: Arc<CpuEmbedder>,
        tokenizer: Arc<FastTokenizer>,
    ) -> Self {
        Self {
            vector_store,
            embedder,
            tokenizer,
            chunker: TextChunker::new(512, 64),
        }
    }

    /// Ingests a raw text document: splits into chunks, computes embeddings, and stores in SQLite
    ///
    /// Idempotent by `source_uri`: any chunks previously stored for the same
    /// `(collection, source_uri)` are removed first, so re-ingesting a rewritten
    /// document replaces it instead of accumulating duplicates.
    pub fn ingest_document(
        &self,
        collection: &str,
        source_uri: &str,
        content: &str,
    ) -> anyhow::Result<usize> {
        let removed = self.vector_store.delete_document(collection, source_uri)?;
        if removed > 0 {
            tracing::info!(
                "Replacing {} existing chunk(s) for source {:?} (collection: {})",
                removed,
                source_uri,
                collection
            );
        }

        let chunks = self.chunker.chunk_text(content);
        let total_chunks = chunks.len();

        tracing::info!(
            "Ingesting document {:?} (collection: {}) into {} chunks",
            source_uri,
            collection,
            total_chunks
        );

        for (idx, chunk) in chunks.iter().enumerate() {
            let embedding = self.embedder.embed_text(chunk)?;
            let token_count = self.tokenizer.count_tokens(chunk);

            self.vector_store.insert_chunk(
                collection,
                source_uri,
                idx,
                chunk,
                token_count,
                &embedding,
            )?;
        }

        Ok(total_chunks)
    }

    /// Performs hybrid search across vector and lexical indices
    pub fn query_rag(
        &self,
        query: &str,
        collection: Option<&str>,
        limit: usize,
    ) -> anyhow::Result<Vec<SearchResult>> {
        let query_vector = self.embedder.embed_text(query)?;
        self.vector_store.hybrid_search(query, &query_vector, collection, limit)
    }

    /// Returns database stats (total chunks and collection counts)
    pub fn stats(&self) -> anyhow::Result<(i64, std::collections::HashMap<String, i64>)> {
        self.vector_store.stats()
    }

    /// Formats search results into a clean context block for prompt injection
    pub fn format_rag_context(&self, results: &[SearchResult]) -> String {
        if results.is_empty() {
            return String::new();
        }

        let mut out = String::from("### Relevant Retrieved Context:\n");
        for (i, r) in results.iter().enumerate() {
            out.push_str(&format!(
                "\n[Citation {} | Source: {} | Score: {:.3}]\n{}\n",
                i + 1,
                r.source_uri,
                r.score,
                r.content.trim()
            ));
        }
        out.push_str("\n### End Retrieved Context\n");
        out
    }
}
