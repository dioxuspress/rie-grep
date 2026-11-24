use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::chunker::ChunkConfig;
use crate::store::SearchStore;
use crate::{ComputeDevice, EmbedderConfig, IndexSummary, SearchMode, SearchOptions};

/// A labeled benchmark query with expected path substrings (e.g. `crates/modules/cases`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalQuery {
  pub query: String,
  pub expected_paths: Vec<String>,
  #[serde(default)]
  pub category: Option<String>,
}

/// Aggregated latency & retrieval quality report for one search mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeBenchmark {
  pub mode: SearchMode,
  pub queries: usize,
  pub p50_ms: f64,
  pub p95_ms: f64,
  pub p99_ms: f64,
  pub mean_ms: f64,
  pub mrr_at_k: f64,
  pub recall_at_1: f64,
  pub recall_at_5: f64,
  pub recall_at_10: f64,
}

/// Complete reproducible benchmark output comparing BM25, Dense (Candle), and Hybrid (RRF).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
  pub index: IndexSummary,
  pub top_k: usize,
  pub modes: Vec<ModeBenchmark>,
}

/// Loads labeled queries from a JSON file (`Vec<EvalQuery>`).
pub fn load_eval_queries(path: &Path) -> Result<Vec<EvalQuery>> {
  let raw = std::fs::read_to_string(path)
    .with_context(|| format!("failed to read benchmark queries {}", path.display()))?;
  serde_json::from_str(&raw).context("failed to parse eval queries JSON")
}

/// Default self-contained latency queries when no external query file is supplied.
pub fn default_smoke_queries() -> Vec<EvalQuery> {
  vec![
    EvalQuery {
      query: "reciprocal rank fusion".into(),
      expected_paths: vec!["fusion".into()],
      category: Some("exact".into()),
    },
    EvalQuery {
      query: "cosine similarity matmul candle".into(),
      expected_paths: vec!["embed".into()],
      category: Some("exact".into()),
    },
    EvalQuery {
      query: "tantivy bm25 schema".into(),
      expected_paths: vec!["bm25".into()],
      category: Some("exact".into()),
    },
    EvalQuery {
      query: "热重载 检索 索引".into(),
      expected_paths: vec!["bm25".into(), "store".into(), "search".into()],
      category: Some("bilingual".into()),
    },
  ]
}

/// Runs a full build + warm-up + 3-mode comparison benchmark on `root`.
pub fn run_benchmark(
  root: &Path,
  index_dir: &Path,
  embedder_config: &EmbedderConfig,
  device: ComputeDevice,
  queries: &[EvalQuery],
  top_k: usize,
) -> Result<BenchmarkReport> {
  let (store, index) = SearchStore::build(
    root,
    index_dir,
    &ChunkConfig::default(),
    embedder_config,
    device,
  )?;

  let mut modes = Vec::with_capacity(3);
  for mode in [SearchMode::Bm25, SearchMode::Dense, SearchMode::Hybrid] {
    modes.push(evaluate_mode(&store, mode, queries, top_k)?);
  }

  Ok(BenchmarkReport {
    index,
    top_k,
    modes,
  })
}

fn evaluate_mode(
  store: &SearchStore,
  mode: SearchMode,
  queries: &[EvalQuery],
  top_k: usize,
) -> Result<ModeBenchmark> {
  let opts = SearchOptions {
    mode,
    top_k: top_k.max(10),
    ..SearchOptions::default()
  };

  // One warm-up call so page cache & Candle kernels are hot.
  if let Some(first) = queries.first() {
    let _ = store.search(&first.query, &opts)?;
  }

  let mut latencies = Vec::with_capacity(queries.len());
  let mut mrr_sum = 0.0f64;
  let mut r1_hits = 0usize;
  let mut r5_hits = 0usize;
  let mut r10_hits = 0usize;

  for q in queries {
    let started = Instant::now();
    let hits = store.search(&q.query, &opts)?;
    latencies.push(started.elapsed().as_secs_f64() * 1000.0);

    let first_match_rank = hits
      .iter()
      .position(|hit| {
        q.expected_paths
          .iter()
          .any(|needle| hit.path.contains(needle))
      })
      .map(|idx| idx + 1);

    if let Some(rank) = first_match_rank {
      mrr_sum += 1.0 / (rank as f64);
      if rank <= 1 {
        r1_hits += 1;
      }
      if rank <= 5 {
        r5_hits += 1;
      }
      if rank <= 10 {
        r10_hits += 1;
      }
    }
  }

  latencies.sort_by(f64::total_cmp);
  let n = queries.len().max(1) as f64;
  let mean_ms = latencies.iter().sum::<f64>() / n;

  Ok(ModeBenchmark {
    mode,
    queries: queries.len(),
    p50_ms: percentile(&latencies, 0.50),
    p95_ms: percentile(&latencies, 0.95),
    p99_ms: percentile(&latencies, 0.99),
    mean_ms,
    mrr_at_k: mrr_sum / n,
    recall_at_1: r1_hits as f64 / n,
    recall_at_5: r5_hits as f64 / n,
    recall_at_10: r10_hits as f64 / n,
  })
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
  if sorted.is_empty() {
    return 0.0;
  }
  let idx = ((sorted.len().saturating_sub(1)) as f64 * p).round() as usize;
  sorted[idx.min(sorted.len() - 1)]
}

/// Formats a [`BenchmarkReport`] as a clean Markdown summary table.
pub fn format_markdown_report(report: &BenchmarkReport) -> String {
  let idx = &report.index;
  let index_mb = idx.index_bytes as f64 / (1024.0 * 1024.0);
  let chunks_per_sec = if idx.elapsed_ms > 0.0 {
    idx.chunks_indexed as f64 / (idx.elapsed_ms / 1000.0)
  } else {
    0.0
  };

  let mut out = String::new();
  out.push_str(&format!(
    "### rie-grep Benchmark Report\n\n\
     - **Root**: `{}`\n\
     - **Indexed**: {} files ({} skipped) → **{} chunks** in **{:.1} ms** ({:.0} chunks/s)\n\
     - **Index Size**: **{:.2} MiB** | **Embedder**: `{}` | **Device**: `{}`\n\n\
     | Mode | Queries | P50 (ms) | P95 (ms) | P99 (ms) | Recall@1 | Recall@5 | Recall@10 | MRR@10 |\n\
     | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n",
    idx.root,
    idx.files_indexed,
    idx.files_skipped,
    idx.chunks_indexed,
    idx.elapsed_ms,
    chunks_per_sec,
    index_mb,
    idx.embedder_label,
    idx.device_label
  ));

  for m in &report.modes {
    let name = match m.mode {
      SearchMode::Bm25 => "BM25 (Tantivy)",
      SearchMode::Dense => "Dense (Candle)",
      SearchMode::Hybrid => "**Hybrid (RRF)**",
    };
    out.push_str(&format!(
      "| {name} | {} | {:.2} | {:.2} | {:.2} | {:.1}% | {:.1}% | {:.1}% | {:.3} |\n",
      m.queries,
      m.p50_ms,
      m.p95_ms,
      m.p99_ms,
      m.recall_at_1 * 100.0,
      m.recall_at_5 * 100.0,
      m.recall_at_10 * 100.0,
      m.mrr_at_k
    ));
  }
  out
}
