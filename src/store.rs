use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::bm25::Bm25Index;
use crate::chunker::{ChunkConfig, walk_and_chunk};
use crate::embed::{DenseEmbedder, VectorStore, resolve_device};
use crate::fusion::fuse_rankings;
use crate::{
  CodeChunk, ComputeDevice, EmbedderConfig, IndexSummary, SearchHit, SearchMode, SearchOptions,
};

/// Default hidden directory stored inside an indexed workspace.
pub const DEFAULT_INDEX_SUBDIR: &str = ".rie-grep";

/// Persistent manifest stored at `<index_dir>/manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexManifest {
  pub version: u32,
  pub root: String,
  pub files_indexed: usize,
  pub files_skipped: usize,
  pub chunks_indexed: usize,
  pub embedding_dim: usize,
  pub embedder: EmbedderConfig,
  pub embedder_label: String,
  pub languages: BTreeMap<String, usize>,
}

/// Statistics returned by `rie-grep stats`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
  pub index_dir: String,
  pub manifest: IndexManifest,
  pub index_bytes: u64,
  pub tantivy_bytes: u64,
  pub vectors_bytes: u64,
  pub chunks_bytes: u64,
}

/// Loaded hybrid code search engine owning Tantivy BM25 + Candle VectorStore + chunk metadata.
pub struct SearchStore {
  pub index_dir: PathBuf,
  pub manifest: IndexManifest,
  pub chunks: Vec<CodeChunk>,
  pub bm25: Bm25Index,
  pub vectors: VectorStore,
  pub embedder: DenseEmbedder,
  pub device_label: &'static str,
}

impl SearchStore {
  /// Walks `root`, builds Tantivy BM25 + Candle dense vectors, and saves all artifacts under `index_dir`.
  pub fn build(
    root: &Path,
    index_dir: &Path,
    chunk_config: &ChunkConfig,
    embedder_config: &EmbedderConfig,
    device_choice: ComputeDevice,
  ) -> Result<(Self, IndexSummary)> {
    let started = Instant::now();
    let canonical_root = root
      .canonicalize()
      .with_context(|| format!("failed to canonicalize {}", root.display()))?;
    std::fs::create_dir_all(index_dir)
      .with_context(|| format!("failed to create index dir {}", index_dir.display()))?;

    let walk = walk_and_chunk(&canonical_root, chunk_config)?;
    let (device, device_label) = resolve_device(device_choice)?;
    let embedder = DenseEmbedder::from_config(embedder_config, device)?;

    let bm25_dir = index_dir.join("tantivy");
    let bm25 = Bm25Index::build(&bm25_dir, &walk.chunks)?;

    let matrix = embedder.encode_chunks(&walk.chunks)?;
    let vectors = VectorStore::from_tensor(matrix)?;
    vectors.save(&index_dir.join("vectors.bin"))?;

    save_chunks_jsonl(&index_dir.join("chunks.jsonl"), &walk.chunks)?;

    let mut languages: BTreeMap<String, usize> = BTreeMap::new();
    for chunk in &walk.chunks {
      *languages.entry(chunk.language.clone()).or_insert(0) += 1;
    }

    let manifest = IndexManifest {
      version: 1,
      root: canonical_root.to_string_lossy().to_string(),
      files_indexed: walk.files_indexed,
      files_skipped: walk.files_skipped,
      chunks_indexed: walk.chunks.len(),
      embedding_dim: embedder.dim(),
      embedder: embedder_config.clone(),
      embedder_label: embedder.label(),
      languages,
    };
    save_manifest(&index_dir.join("manifest.json"), &manifest)?;

    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let index_bytes = dir_size_bytes(index_dir);
    let summary = IndexSummary {
      root: manifest.root.clone(),
      index_dir: index_dir.to_string_lossy().to_string(),
      files_indexed: manifest.files_indexed,
      files_skipped: manifest.files_skipped,
      chunks_indexed: manifest.chunks_indexed,
      embedding_dim: manifest.embedding_dim,
      embedder_label: manifest.embedder_label.clone(),
      device_label: device_label.to_string(),
      index_bytes,
      elapsed_ms,
    };

    let store = Self {
      index_dir: index_dir.to_path_buf(),
      manifest,
      chunks: walk.chunks,
      bm25,
      vectors,
      embedder,
      device_label,
    };
    Ok((store, summary))
  }

  /// Opens an existing index from `index_dir` onto `device_choice`.
  pub fn open(index_dir: &Path, device_choice: ComputeDevice) -> Result<Self> {
    let manifest = load_manifest(&index_dir.join("manifest.json"))?;
    let chunks = load_chunks_jsonl(&index_dir.join("chunks.jsonl"))?;
    let bm25 = Bm25Index::open(&index_dir.join("tantivy"))?;
    let (device, device_label) = resolve_device(device_choice)?;
    let vectors = VectorStore::load(&index_dir.join("vectors.bin"), &device)?;
    let embedder = DenseEmbedder::from_config(&manifest.embedder, device)?;

    Ok(Self {
      index_dir: index_dir.to_path_buf(),
      manifest,
      chunks,
      bm25,
      vectors,
      embedder,
      device_label,
    })
  }

  /// Executes a BM25, dense, or RRF hybrid query against the loaded store.
  pub fn search(&self, query: &str, options: &SearchOptions) -> Result<Vec<SearchHit>> {
    // Over-fetch candidates before post-filtering and RRF fusion.
    let recall_depth = (options.top_k.saturating_mul(6)).clamp(50, 500);

    let bm25_hits = if matches!(options.mode, SearchMode::Hybrid | SearchMode::Bm25) {
      self.bm25.search(query, recall_depth)?
    } else {
      Vec::new()
    };

    let dense_hits = if matches!(options.mode, SearchMode::Hybrid | SearchMode::Dense) {
      let q_vec = self.embedder.encode_query(query)?;
      self.vectors.search(&q_vec, recall_depth)?
    } else {
      Vec::new()
    };

    Ok(fuse_rankings(
      &self.chunks,
      &bm25_hits,
      &dense_hits,
      query,
      options,
    ))
  }

  /// Computes on-disk footprint breakdown for an index directory.
  pub fn read_stats(index_dir: &Path) -> Result<IndexStats> {
    let manifest = load_manifest(&index_dir.join("manifest.json"))?;
    let tantivy_bytes = dir_size_bytes(&index_dir.join("tantivy"));
    let vectors_bytes = file_size_bytes(&index_dir.join("vectors.bin"));
    let chunks_bytes = file_size_bytes(&index_dir.join("chunks.jsonl"));
    let index_bytes = dir_size_bytes(index_dir);
    Ok(IndexStats {
      index_dir: index_dir.to_string_lossy().to_string(),
      manifest,
      index_bytes,
      tantivy_bytes,
      vectors_bytes,
      chunks_bytes,
    })
  }
}

/// Resolves the default `.rie-grep` directory for a repository root.
pub fn default_index_dir(root: &Path) -> PathBuf {
  root.join(DEFAULT_INDEX_SUBDIR)
}

fn save_manifest(path: &Path, manifest: &IndexManifest) -> Result<()> {
  let payload = serde_json::to_string_pretty(manifest).context("serialize manifest")?;
  std::fs::write(path, payload)
    .with_context(|| format!("failed to write manifest {}", path.display()))
}

fn load_manifest(path: &Path) -> Result<IndexManifest> {
  let raw = std::fs::read_to_string(path).with_context(|| {
    format!(
      "index manifest not found at {} (run `rie-grep index` first)",
      path.display()
    )
  })?;
  serde_json::from_str(&raw).context("failed to parse manifest.json")
}

fn save_chunks_jsonl(path: &Path, chunks: &[CodeChunk]) -> Result<()> {
  let file = File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
  let mut writer = BufWriter::new(file);
  for chunk in chunks {
    serde_json::to_writer(&mut writer, chunk).context("serialize CodeChunk")?;
    writer.write_all(b"\n")?;
  }
  writer.flush()?;
  Ok(())
}

fn load_chunks_jsonl(path: &Path) -> Result<Vec<CodeChunk>> {
  let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
  let reader = BufReader::new(file);
  let mut chunks = Vec::new();
  for line in reader.lines() {
    let line = line?;
    if line.trim().is_empty() {
      continue;
    }
    let chunk: CodeChunk = serde_json::from_str(&line).context("parse CodeChunk jsonl")?;
    chunks.push(chunk);
  }
  Ok(chunks)
}

fn file_size_bytes(path: &Path) -> u64 {
  std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn dir_size_bytes(path: &Path) -> u64 {
  if !path.exists() {
    return 0;
  }
  let mut total = 0u64;
  let mut stack = vec![path.to_path_buf()];
  while let Some(dir) = stack.pop() {
    let Ok(entries) = std::fs::read_dir(&dir) else {
      continue;
    };
    for entry in entries.flatten() {
      let Ok(meta) = entry.metadata() else {
        continue;
      };
      if meta.is_dir() {
        stack.push(entry.path());
      } else {
        total = total.saturating_add(meta.len());
      }
    }
  }
  total
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::time::{SystemTime, UNIX_EPOCH};

  fn temp_workspace(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap()
      .as_nanos();
    let dir = std::env::temp_dir().join(format!("rie_grep_store_{tag}_{nanos}"));
    std::fs::create_dir_all(dir.join("src")).unwrap();
    dir
  }

  #[test]
  fn end_to_end_index_reopen_and_hybrid_search() {
    let ws = temp_workspace("e2e");
    std::fs::write(
      ws.join("src/auth.rs"),
      "\
pub struct JwtValidator {
  secret: String,
}

impl JwtValidator {
  pub fn verify_bearer_token(&self, token: &str) -> bool {
    token.starts_with(&self.secret)
  }
}
",
    )
    .unwrap();
    std::fs::write(
      ws.join("src/reload.rs"),
      "\
pub async fn watch_sighup_signal() {
  // Triggers zero-downtime hot reload of cached markdown files
}
",
    )
    .unwrap();

    let idx_dir = default_index_dir(&ws);
    let (_store, summary) = SearchStore::build(
      &ws,
      &idx_dir,
      &ChunkConfig::default(),
      &EmbedderConfig::default(),
      ComputeDevice::Cpu,
    )
    .unwrap();
    assert_eq!(summary.files_indexed, 2);
    assert!(summary.chunks_indexed >= 2);

    let reopened = SearchStore::open(&idx_dir, ComputeDevice::Cpu).unwrap();

    // 1. Exact English identifier query.
    let hits = reopened
      .search("verify bearer token", &SearchOptions::default())
      .unwrap();
    assert_eq!(hits[0].path, "src/auth.rs");

    // 2. Chinese semantic query against English Rust source.
    let zh_hits = reopened
      .search("热重载 信号", &SearchOptions::default())
      .unwrap();
    assert_eq!(zh_hits[0].path, "src/reload.rs");

    let stats = SearchStore::read_stats(&idx_dir).unwrap();
    assert!(stats.index_bytes > 0);
    assert_eq!(stats.manifest.chunks_indexed, summary.chunks_indexed);

    let _ = std::fs::remove_dir_all(&ws);
  }
}
