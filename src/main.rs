#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

use rie_grep::bench::{
  default_smoke_queries, format_markdown_report, load_eval_queries, run_benchmark,
};
use rie_grep::chunker::ChunkConfig;
use rie_grep::embed::DEFAULT_HASH_DIM;
use rie_grep::store::{SearchStore, default_index_dir};
use rie_grep::{ComputeDevice, EmbedderConfig, SearchHit, SearchMode, SearchOptions};

#[derive(Parser)]
#[command(
  name = "rie-grep",
  version,
  about = "Single-binary local AI hybrid code search engine (Tantivy BM25 + Candle vectors + RRF)"
)]
struct Cli {
  #[command(subcommand)]
  command: Command,
}

#[derive(Subcommand)]
enum Command {
  /// Scan a codebase and build the hybrid Tantivy + Candle index.
  Index {
    /// Target workspace directory to index.
    #[arg(default_value = ".")]
    path: PathBuf,
    /// Custom index output directory (defaults to `<path>/.rie-grep`).
    #[arg(long)]
    index_dir: Option<PathBuf>,
    /// Local HuggingFace BERT/BGE/MiniLM model directory (optional; defaults to zero-download HashNgram).
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Embedding vector dimension for HashNgram mode.
    #[arg(long, default_value_t = DEFAULT_HASH_DIM)]
    dim: usize,
    /// Compute device for Candle tensor ops (`cpu` or `metal`).
    #[arg(long, value_enum, default_value_t = CliDevice::Cpu)]
    device: CliDevice,
    /// Maximum lines per code chunk.
    #[arg(long, default_value_t = 48)]
    chunk_lines: usize,
    /// Emit machine-readable JSON summary.
    #[arg(long)]
    json: bool,
  },

  /// Search an indexed codebase using Hybrid (RRF), BM25, or Dense vector recall.
  Search {
    /// Natural-language or identifier query string.
    query: String,
    /// Workspace root containing `.rie-grep` (auto-indexes if `--auto-index` is set).
    #[arg(short = 'p', long, default_value = ".")]
    path: PathBuf,
    /// Custom index directory (overrides `<path>/.rie-grep`).
    #[arg(long)]
    index_dir: Option<PathBuf>,
    /// Retrieval mode: hybrid (RRF), bm25, or dense.
    #[arg(short = 'm', long, value_enum, default_value_t = CliMode::Hybrid)]
    mode: CliMode,
    /// Maximum number of results to display.
    #[arg(short = 'k', long, default_value_t = 8)]
    top_k: usize,
    /// Filter by language (e.g. rust, rs, md, ts, py).
    #[arg(short = 'l', long)]
    lang: Option<String>,
    /// Filter by relative path prefix (e.g. `crates/core`).
    #[arg(long)]
    prefix: Option<String>,
    /// Maximum snippet lines printed per result.
    #[arg(short = 'c', long, default_value_t = 6)]
    context: usize,
    /// Compute device (`cpu` or `metal`).
    #[arg(long, value_enum, default_value_t = CliDevice::Cpu)]
    device: CliDevice,
    /// Automatically build the index if `.rie-grep` does not exist yet.
    #[arg(long)]
    auto_index: bool,
    /// Emit machine-readable JSON output.
    #[arg(long)]
    json: bool,
  },

  /// Run reproducible indexing + BM25 vs Dense vs Hybrid latency & recall benchmarks.
  Bench {
    /// Target workspace directory to benchmark.
    #[arg(default_value = ".")]
    path: PathBuf,
    /// Optional JSON file of labeled evaluation queries.
    #[arg(long)]
    queries: Option<PathBuf>,
    /// Optional local BERT/BGE/MiniLM model directory.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Custom index output directory (defaults to `<path>/.rie-grep`).
    #[arg(long)]
    index_dir: Option<PathBuf>,
    /// Compute device (`cpu` or `metal`).
    #[arg(long, value_enum, default_value_t = CliDevice::Cpu)]
    device: CliDevice,
    /// Evaluation cutoff K (minimum 10 for MRR@10).
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    /// Emit raw JSON report instead of Markdown table.
    #[arg(long)]
    json: bool,
  },

  /// Inspect index statistics, language breakdown, and disk footprint.
  Stats {
    /// Workspace root containing `.rie-grep`.
    #[arg(default_value = ".")]
    path: PathBuf,
    /// Custom index directory.
    #[arg(long)]
    index_dir: Option<PathBuf>,
    /// Emit JSON stats.
    #[arg(long)]
    json: bool,
  },
}

#[derive(Clone, Copy, ValueEnum)]
enum CliMode {
  Hybrid,
  Bm25,
  Dense,
}

impl From<CliMode> for SearchMode {
  fn from(value: CliMode) -> Self {
    match value {
      CliMode::Hybrid => Self::Hybrid,
      CliMode::Bm25 => Self::Bm25,
      CliMode::Dense => Self::Dense,
    }
  }
}

#[derive(Clone, Copy, ValueEnum)]
enum CliDevice {
  Cpu,
  Metal,
}

impl From<CliDevice> for ComputeDevice {
  fn from(value: CliDevice) -> Self {
    match value {
      CliDevice::Cpu => Self::Cpu,
      CliDevice::Metal => Self::Metal,
    }
  }
}

fn main() -> Result<()> {
  let cli = Cli::parse();
  match cli.command {
    Command::Index {
      path,
      index_dir,
      model_dir,
      dim,
      device,
      chunk_lines,
      json,
    } => {
      let idx_dir = index_dir.unwrap_or_else(|| default_index_dir(&path));
      let embedder_cfg = match model_dir {
        Some(dir) => EmbedderConfig::Bert {
          model_dir: dir.to_string_lossy().to_string(),
        },
        None => EmbedderConfig::HashNgram { dim },
      };
      let chunk_cfg = ChunkConfig {
        max_chunk_lines: chunk_lines,
        ..ChunkConfig::default()
      };
      let (_store, summary) =
        SearchStore::build(&path, &idx_dir, &chunk_cfg, &embedder_cfg, device.into())?;

      if json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
      } else {
        let mb = summary.index_bytes as f64 / (1024.0 * 1024.0);
        println!(
          "Indexed {} files ({} skipped) -> {} chunks in {:.1} ms",
          summary.files_indexed, summary.files_skipped, summary.chunks_indexed, summary.elapsed_ms
        );
        println!(
          "Index: {} ({:.2} MiB) | Embedder: {} | Device: {}",
          summary.index_dir, mb, summary.embedder_label, summary.device_label
        );
      }
    }

    Command::Search {
      query,
      path,
      index_dir,
      mode,
      top_k,
      lang,
      prefix,
      context,
      device,
      auto_index,
      json,
    } => {
      let idx_dir = index_dir.unwrap_or_else(|| default_index_dir(&path));
      let store = open_or_auto_index(&path, &idx_dir, device.into(), auto_index)?;
      let opts = SearchOptions {
        mode: mode.into(),
        top_k,
        language: lang,
        path_prefix: prefix,
        snippet_lines: context,
        ..SearchOptions::default()
      };

      let started = Instant::now();
      let hits = store.search(&query, &opts)?;
      let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

      if json {
        let payload = serde_json::json!({
          "query": query,
          "mode": opts.mode,
          "device": store.device_label,
          "elapsed_ms": elapsed_ms,
          "hits": hits,
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
      } else {
        print_terminal_hits(&query, opts.mode, store.device_label, elapsed_ms, &hits);
      }
    }

    Command::Bench {
      path,
      queries,
      model_dir,
      index_dir,
      device,
      top_k,
      json,
    } => {
      let idx_dir = index_dir.unwrap_or_else(|| default_index_dir(&path));
      let eval_queries = match queries {
        Some(q_path) => load_eval_queries(&q_path)?,
        None => default_smoke_queries(),
      };
      let embedder_cfg = match model_dir {
        Some(dir) => EmbedderConfig::Bert {
          model_dir: dir.to_string_lossy().to_string(),
        },
        None => EmbedderConfig::default(),
      };
      let report = run_benchmark(
        &path,
        &idx_dir,
        &embedder_cfg,
        device.into(),
        &eval_queries,
        top_k,
      )?;
      if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
      } else {
        print!("{}", format_markdown_report(&report));
      }
    }

    Command::Stats {
      path,
      index_dir,
      json,
    } => {
      let idx_dir = index_dir.unwrap_or_else(|| default_index_dir(&path));
      let stats = SearchStore::read_stats(&idx_dir)?;
      if json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
      } else {
        let to_mb = |b: u64| b as f64 / (1024.0 * 1024.0);
        println!("Index dir : {}", stats.index_dir);
        println!("Root      : {}", stats.manifest.root);
        println!(
          "Chunks    : {} across {} files ({} skipped)",
          stats.manifest.chunks_indexed, stats.manifest.files_indexed, stats.manifest.files_skipped
        );
        println!(
          "Embedder  : {} ({}d)",
          stats.manifest.embedder_label, stats.manifest.embedding_dim
        );
        println!(
          "Disk size : {:.2} MiB total (tantivy {:.2} MiB, vectors {:.2} MiB, chunks {:.2} MiB)",
          to_mb(stats.index_bytes),
          to_mb(stats.tantivy_bytes),
          to_mb(stats.vectors_bytes),
          to_mb(stats.chunks_bytes)
        );
        println!("Languages :");
        for (lang, count) in &stats.manifest.languages {
          println!("  - {:<12} {:>6} chunks", lang, count);
        }
      }
    }
  }
  Ok(())
}

fn open_or_auto_index(
  root: &Path,
  index_dir: &Path,
  device: ComputeDevice,
  auto_index: bool,
) -> Result<SearchStore> {
  if !index_dir.join("manifest.json").exists() && auto_index {
    let (store, _summary) = SearchStore::build(
      root,
      index_dir,
      &ChunkConfig::default(),
      &EmbedderConfig::default(),
      device,
    )?;
    return Ok(store);
  }
  SearchStore::open(index_dir, device)
}

fn print_terminal_hits(
  query: &str,
  mode: SearchMode,
  device: &str,
  elapsed_ms: f64,
  hits: &[SearchHit],
) {
  let color = std::io::stdout().is_terminal();
  let (bold, cyan, dim, yellow, reset) = if color {
    ("\x1b[1m", "\x1b[36m", "\x1b[2m", "\x1b[33m", "\x1b[0m")
  } else {
    ("", "", "", "", "")
  };

  println!(
    "{bold}rie-grep{reset} {dim}[mode={mode:?} device={device} {elapsed_ms:.2} ms]{reset} query={yellow}\"{query}\"{reset} ({} hits)\n",
    hits.len()
  );

  if hits.is_empty() {
    println!("No matching chunks found.");
    return;
  }

  for hit in hits {
    let bm25_tag = match (hit.bm25_rank, hit.bm25_score) {
      (Some(r), Some(s)) => format!("bm25=#{r}({s:.2})"),
      _ => "bm25=-".into(),
    };
    let dense_tag = match (hit.dense_rank, hit.dense_score) {
      (Some(r), Some(s)) => format!("dense=#{r}({s:.3})"),
      _ => "dense=-".into(),
    };

    println!(
      "{bold}#{}{reset} {cyan}{}:{}-{}{reset}  {yellow}{}{reset}  {dim}[score={:.4} {bm25_tag} {dense_tag}]{reset}",
      hit.rank, hit.path, hit.start_line, hit.end_line, hit.symbol, hit.score
    );
    for line in hit.snippet.lines() {
      println!("   {dim}|{reset} {line}");
    }
    println!();
  }
}
