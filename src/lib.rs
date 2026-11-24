#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod bench;
pub mod bm25;
pub mod chunker;
pub mod embed;
pub mod fusion;
pub mod store;
pub mod tokenizer;

use serde::{Deserialize, Serialize};

/// A single indexed source code chunk with structural metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeChunk {
  pub chunk_id: u32,
  pub path: String,
  pub language: String,
  pub symbol: String,
  pub start_line: usize,
  pub end_line: usize,
  pub content: String,
}

/// Retrieval mode selectable per query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
  Hybrid,
  Bm25,
  Dense,
}

/// Compute device for Candle tensor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeDevice {
  Cpu,
  Metal,
}

/// Dense embedding backend strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EmbedderConfig {
  /// Built-in zero-download subword + char n-gram + bilingual anchor encoder projected via Candle.
  HashNgram { dim: usize },
  /// Offline local HuggingFace BERT/BGE/MiniLM model (`config.json` + `model.safetensors` + `tokenizer.json`).
  Bert { model_dir: String },
}

impl Default for EmbedderConfig {
  fn default() -> Self {
    Self::HashNgram {
      dim: embed::DEFAULT_HASH_DIM,
    }
  }
}

/// Options controlling query execution and result filtering.
#[derive(Debug, Clone)]
pub struct SearchOptions {
  pub mode: SearchMode,
  pub top_k: usize,
  pub rrf_k: f32,
  pub bm25_weight: f32,
  pub dense_weight: f32,
  pub language: Option<String>,
  pub path_prefix: Option<String>,
  pub snippet_lines: usize,
}

impl Default for SearchOptions {
  fn default() -> Self {
    Self {
      mode: SearchMode::Hybrid,
      top_k: 10,
      rrf_k: 30.0,
      bm25_weight: 1.0,
      dense_weight: 1.2,
      language: None,
      path_prefix: None,
      snippet_lines: 6,
    }
  }
}

/// A single ranked search result with full score provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
  pub rank: usize,
  pub chunk_id: u32,
  pub path: String,
  pub language: String,
  pub symbol: String,
  pub start_line: usize,
  pub end_line: usize,
  pub score: f32,
  pub bm25_rank: Option<usize>,
  pub bm25_score: Option<f32>,
  pub dense_rank: Option<usize>,
  pub dense_score: Option<f32>,
  pub snippet: String,
}

/// Summary metrics emitted after building or refreshing an index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSummary {
  pub root: String,
  pub index_dir: String,
  pub files_indexed: usize,
  pub files_skipped: usize,
  pub chunks_indexed: usize,
  pub embedding_dim: usize,
  pub embedder_label: String,
  pub device_label: String,
  pub index_bytes: u64,
  pub elapsed_ms: f64,
}
