use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use half::f16;
use rayon::prelude::*;
use tokenizers::Tokenizer;

use crate::tokenizer::tokenize_code;
use crate::{CodeChunk, ComputeDevice, EmbedderConfig};

/// Default dense dimension for the built-in zero-download HashNgram encoder.
pub const DEFAULT_HASH_DIM: usize = 384;

const VECTOR_MAGIC: &[u8; 8] = b"RIEV0001";
const VECTOR_DTYPE_F16: u8 = 1;
const BERT_BATCH_SIZE: usize = 32;
const BERT_MAX_TOKENS: usize = 256;

/// Resolves a Candle compute [`Device`] from CLI/config choice.
pub fn resolve_device(requested: ComputeDevice) -> Result<(Device, &'static str)> {
  match requested {
    ComputeDevice::Cpu => Ok((Device::Cpu, "cpu")),
    ComputeDevice::Metal => {
      #[cfg(feature = "metal")]
      {
        let dev = Device::new_metal(0).context("failed to initialize Apple Metal device 0")?;
        Ok((dev, "metal"))
      }
      #[cfg(not(feature = "metal"))]
      {
        bail!(
          "Metal backend was requested, but this binary was compiled without `--features metal`"
        )
      }
    }
  }
}

/// Dense embedding encoder backed by Candle tensors (CPU or Metal GPU).
pub enum DenseEmbedder {
  HashNgram { dim: usize, device: Device },
  Bert(Box<BertEmbedder>),
}

/// Local offline Transformer encoder (`all-MiniLM` / `bge-small` / BERT safetensors).
pub struct BertEmbedder {
  model: BertModel,
  tokenizer: Tokenizer,
  device: Device,
  dim: usize,
  pad_token_id: u32,
  max_len: usize,
}

impl DenseEmbedder {
  /// Constructs an embedder from [`EmbedderConfig`] on `device`.
  pub fn from_config(config: &EmbedderConfig, device: Device) -> Result<Self> {
    match config {
      EmbedderConfig::HashNgram { dim } => {
        if *dim == 0 || *dim > 4096 {
          bail!("invalid HashNgram embedding dimension: {dim}");
        }
        Ok(Self::HashNgram { dim: *dim, device })
      }
      EmbedderConfig::Bert { model_dir } => {
        let bert = BertEmbedder::load(Path::new(model_dir), device)?;
        Ok(Self::Bert(Box::new(bert)))
      }
    }
  }

  /// Returns output vector dimensionality.
  pub fn dim(&self) -> usize {
    match self {
      Self::HashNgram { dim, .. } => *dim,
      Self::Bert(bert) => bert.dim,
    }
  }

  /// Human-readable label recorded in the index manifest.
  pub fn label(&self) -> String {
    match self {
      Self::HashNgram { dim, .. } => format!("hash-ngram-{dim}d (candle)"),
      Self::Bert(bert) => format!("bert-{}d (candle)", bert.dim),
    }
  }

  /// Encodes all chunks into a single L2-normalized `[N, D]` f32 Candle tensor.
  pub fn encode_chunks(&self, chunks: &[CodeChunk]) -> Result<Tensor> {
    let dim = self.dim();
    if chunks.is_empty() {
      let dev = self.device();
      return Tensor::zeros((0, dim), DType::F32, dev).context("failed to build empty tensor");
    }

    match self {
      Self::HashNgram { dim, device } => {
        let per_chunk: Vec<Vec<f32>> = chunks
          .par_iter()
          .map(|chunk| hash_encode_chunk(chunk, *dim))
          .collect();
        let mut flat = Vec::with_capacity(chunks.len() * (*dim));
        for vec in per_chunk {
          flat.extend_from_slice(&vec);
        }
        let raw = Tensor::from_vec(flat, (chunks.len(), *dim), device)
          .context("failed to upload chunk features to Candle tensor")?;
        l2_normalize_rows(&raw)
      }
      Self::Bert(bert) => {
        let texts: Vec<String> = chunks
          .iter()
          .map(|c| format!("{} {} {}", c.path, c.symbol, c.content))
          .collect();
        bert.encode_texts(&texts)
      }
    }
  }

  /// Encodes a single query string into an L2-normalized `[1, D]` f32 Candle tensor.
  pub fn encode_query(&self, query: &str) -> Result<Tensor> {
    match self {
      Self::HashNgram { dim, device } => {
        let features = hash_encode_query(query, *dim);
        let raw = Tensor::from_vec(features, (1, *dim), device)
          .context("failed to upload query features to Candle tensor")?;
        l2_normalize_rows(&raw)
      }
      Self::Bert(bert) => bert.encode_texts(&[query.to_string()]),
    }
  }

  fn device(&self) -> &Device {
    match self {
      Self::HashNgram { device, .. } => device,
      Self::Bert(bert) => &bert.device,
    }
  }
}

impl BertEmbedder {
  /// Loads a local HuggingFace BERT/BGE/MiniLM directory (`config.json`, `tokenizer.json`,
  /// `model.safetensors`) with zero network access.
  pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
    let config_path = model_dir.join("config.json");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let weights_path = model_dir.join("model.safetensors");

    let config_str = std::fs::read_to_string(&config_path)
      .with_context(|| format!("missing BERT config at {}", config_path.display()))?;
    let config: BertConfig =
      serde_json::from_str(&config_str).context("failed to parse BERT config.json")?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
      .map_err(|e| anyhow::anyhow!("failed to load tokenizer {}: {e}", tokenizer_path.display()))?;

    if !weights_path.exists() {
      bail!("missing safetensors weights at {}", weights_path.display());
    }

    // SAFETY: `weights_path` is an immutable local safetensors file loaded read-only
    // during `BertModel::load`, which materializes tensors into Candle storage.
    let vb = unsafe {
      VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
        .context("failed to mmap model.safetensors")?
    };
    let model = BertModel::load(vb, &config).context("failed to load BertModel weights")?;

    let max_len = config.max_position_embeddings.clamp(16, BERT_MAX_TOKENS);
    Ok(Self {
      model,
      tokenizer,
      device,
      dim: config.hidden_size,
      pad_token_id: config.pad_token_id as u32,
      max_len,
    })
  }

  /// Encodes a slice of texts with attention-masked mean pooling + L2 normalization.
  pub fn encode_texts(&self, texts: &[String]) -> Result<Tensor> {
    if texts.is_empty() {
      return Tensor::zeros((0, self.dim), DType::F32, &self.device)
        .context("failed to build empty BERT embedding tensor");
    }

    let mut batch_tensors = Vec::new();
    for batch in texts.chunks(BERT_BATCH_SIZE) {
      let mut encoded_ids: Vec<Vec<u32>> = Vec::with_capacity(batch.len());
      let mut encoded_masks: Vec<Vec<u32>> = Vec::with_capacity(batch.len());
      let mut max_seq = 1usize;

      for text in batch {
        let encoding = self
          .tokenizer
          .encode(text.as_str(), true)
          .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let mut ids: Vec<u32> = encoding
          .get_ids()
          .iter()
          .copied()
          .take(self.max_len)
          .collect();
        let mut mask: Vec<u32> = encoding
          .get_attention_mask()
          .iter()
          .copied()
          .take(self.max_len)
          .collect();
        if ids.is_empty() {
          ids.push(self.pad_token_id);
          mask.push(1);
        }
        max_seq = max_seq.max(ids.len());
        encoded_ids.push(ids);
        encoded_masks.push(mask);
      }

      let bsz = batch.len();
      let mut flat_ids = Vec::with_capacity(bsz * max_seq);
      let mut flat_masks = Vec::with_capacity(bsz * max_seq);
      for (mut ids, mut mask) in encoded_ids.into_iter().zip(encoded_masks) {
        ids.resize(max_seq, self.pad_token_id);
        mask.resize(max_seq, 0);
        flat_ids.extend_from_slice(&ids);
        flat_masks.extend_from_slice(&mask);
      }

      let input_ids =
        Tensor::from_vec(flat_ids, (bsz, max_seq), &self.device).context("input_ids tensor")?;
      let token_type_ids = input_ids.zeros_like().context("token_type_ids tensor")?;
      let mask_u32 = Tensor::from_vec(flat_masks, (bsz, max_seq), &self.device)
        .context("attention_mask tensor")?;

      let hidden = self
        .model
        .forward(&input_ids, &token_type_ids, Some(&mask_u32))
        .context("BERT forward pass failed")?;

      let mask_f32 = mask_u32
        .to_dtype(DType::F32)
        .context("mask to f32")?
        .unsqueeze(2)
        .context("mask unsqueeze")?;
      let masked = hidden
        .broadcast_mul(&mask_f32)
        .context("mask broadcast_mul")?;
      let summed = masked.sum(1).context("mean-pool sum")?;
      let denom = mask_f32
        .sum(1)
        .context("mask count sum")?
        .clamp(1.0_f64, f64::INFINITY)
        .context("mask clamp")?;
      let pooled = summed.broadcast_div(&denom).context("mean-pool div")?;
      let normed = l2_normalize_rows(&pooled)?;
      batch_tensors.push(normed);
    }

    if batch_tensors.len() == 1 {
      return Ok(batch_tensors.remove(0));
    }
    let refs: Vec<&Tensor> = batch_tensors.iter().collect();
    Tensor::cat(&refs, 0).context("failed to concatenate BERT batch tensors")
  }
}

/// Compact f16 on-disk vector store loaded directly into a Candle `[N, D]` tensor for
/// single-shot batched cosine similarity search (`Q @ D^T`).
pub struct VectorStore {
  matrix: Tensor,
  num_chunks: usize,
  dim: usize,
}

impl VectorStore {
  /// Wraps an already L2-normalized `[N, D]` f32 Candle tensor.
  pub fn from_tensor(matrix: Tensor) -> Result<Self> {
    let (num_chunks, dim) = matrix.dims2().context("vector matrix must be 2D [N, D]")?;
    Ok(Self {
      matrix,
      num_chunks,
      dim,
    })
  }

  /// Persists the `[N, D]` vector matrix in compact little-endian f16 format (`RIEV0001`).
  pub fn save(&self, path: &Path) -> Result<()> {
    let flat = self
      .matrix
      .flatten_all()
      .context("flatten matrix")?
      .to_vec1::<f32>()
      .context("read matrix back to host f32")?;

    let file =
      File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(VECTOR_MAGIC)?;
    writer.write_all(&(self.num_chunks as u32).to_le_bytes())?;
    writer.write_all(&(self.dim as u32).to_le_bytes())?;
    writer.write_all(&[VECTOR_DTYPE_F16])?;

    for value in flat {
      let half_bits = f16::from_f32(value).to_le_bytes();
      writer.write_all(&half_bits)?;
    }
    writer.flush()?;
    Ok(())
  }

  /// Loads `vectors.bin` onto `device` as an `[N, D]` f32 Candle tensor.
  pub fn load(path: &Path, device: &Device) -> Result<Self> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 8];
    reader
      .read_exact(&mut magic)
      .context("failed to read vector magic header")?;
    if &magic != VECTOR_MAGIC {
      bail!("invalid vector store magic in {}", path.display());
    }

    let mut u32_buf = [0u8; 4];
    reader.read_exact(&mut u32_buf)?;
    let num_chunks = u32::from_le_bytes(u32_buf) as usize;
    reader.read_exact(&mut u32_buf)?;
    let dim = u32::from_le_bytes(u32_buf) as usize;

    let mut dtype_buf = [0u8; 1];
    reader.read_exact(&mut dtype_buf)?;
    if dtype_buf[0] != VECTOR_DTYPE_F16 {
      bail!("unsupported vector dtype tag {}", dtype_buf[0]);
    }

    let total = num_chunks
      .checked_mul(dim)
      .context("vector matrix dimensions overflow")?;
    let mut raw_bytes = vec![0u8; total * 2];
    reader
      .read_exact(&mut raw_bytes)
      .context("truncated vector payload")?;

    let mut floats = Vec::with_capacity(total);
    for &pair in raw_bytes.as_chunks::<2>().0 {
      floats.push(f16::from_le_bytes(pair).to_f32());
    }

    let matrix = Tensor::from_vec(floats, (num_chunks, dim), device)
      .context("failed to load vector matrix onto Candle device")?;
    Ok(Self {
      matrix,
      num_chunks,
      dim,
    })
  }

  pub fn num_chunks(&self) -> usize {
    self.num_chunks
  }

  pub fn dim(&self) -> usize {
    self.dim
  }

  /// Computes exact cosine similarity between an L2-normalized `[1, D]` query vector
  /// and all `[N, D]` chunk vectors in one Candle `matmul`, returning top `limit` `(chunk_id, score)`.
  pub fn search(&self, query_vec: &Tensor, limit: usize) -> Result<Vec<(u32, f32)>> {
    if self.num_chunks == 0 || limit == 0 {
      return Ok(Vec::new());
    }
    let scores_tensor = self
      .matrix
      .matmul(&query_vec.t().context("query transpose")?)
      .context("Candle matmul cosine similarity failed")?
      .squeeze(1)
      .context("squeeze similarity scores")?;
    let scores = scores_tensor
      .to_vec1::<f32>()
      .context("read similarity scores to host")?;

    let mut scored: Vec<(u32, f32)> = scores
      .into_iter()
      .enumerate()
      .filter(|(_, s)| s.is_finite() && *s > 0.0)
      .map(|(idx, s)| (idx as u32, s))
      .collect();

    let take = limit.min(scored.len());
    if take == 0 {
      return Ok(Vec::new());
    }
    if take < scored.len() {
      scored.select_nth_unstable_by(take, |a, b| b.1.total_cmp(&a.1));
      scored.truncate(take);
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok(scored)
  }
}

fn l2_normalize_rows(tensor: &Tensor) -> Result<Tensor> {
  let sq = tensor.sqr().context("l2 sqr")?;
  let sum = sq.sum_keepdim(1).context("l2 sum")?;
  let norm = sum
    .sqrt()
    .context("l2 sqrt")?
    .clamp(1e-12_f64, f64::INFINITY)
    .context("l2 clamp")?;
  tensor.broadcast_div(&norm).context("l2 broadcast_div")
}

/// Signed random projection of subword tokens, character n-grams, and bilingual code concepts.
fn hash_encode_chunk(chunk: &CodeChunk, dim: usize) -> Vec<f32> {
  let mut acc = vec![0.0f32; dim];
  accumulate_text_features(&chunk.symbol, 3.0, &mut acc);
  accumulate_text_features(&chunk.path, 2.0, &mut acc);
  accumulate_text_features(&chunk.language, 1.2, &mut acc);
  accumulate_text_features(&chunk.content, 1.0, &mut acc);
  acc
}

fn hash_encode_query(query: &str, dim: usize) -> Vec<f32> {
  let mut acc = vec![0.0f32; dim];
  accumulate_text_features(query, 1.0, &mut acc);
  acc
}

fn accumulate_text_features(text: &str, field_weight: f32, acc: &mut [f32]) {
  let tokens = tokenize_code(text);
  let mut prev_tok: Option<&str> = None;

  for tok in &tokens {
    if is_stopword(tok) {
      prev_tok = None;
      continue;
    }

    // 1. Exact subword token feature.
    add_hashed(acc, b"tok:", tok.as_bytes(), field_weight * 1.4);

    // 2. Character 3-grams and 4-grams for morphological & typo resilience.
    let bytes = tok.as_bytes();
    if bytes.is_ascii() && bytes.len() >= 4 && bytes.len() <= 24 {
      let mut padded = Vec::with_capacity(bytes.len() + 2);
      padded.push(b'^');
      padded.extend_from_slice(bytes);
      padded.push(b'$');
      for gram in padded.windows(3) {
        add_hashed(acc, b"g3:", gram, field_weight * 0.45);
      }
      if padded.len() >= 4 {
        for gram in padded.windows(4) {
          add_hashed(acc, b"g4:", gram, field_weight * 0.35);
        }
      }
    }

    // 3. Adjacent token bigram captures multi-word phrases like `hot reload` or `kv cache`.
    if let Some(prev) = prev_tok {
      let pair = format!("{prev}_{tok}");
      add_hashed(acc, b"bi:", pair.as_bytes(), field_weight * 0.9);
    }

    // 4. Lightweight bilingual programming concept anchors so Chinese & English queries align
    //    into the same dense neighborhood even with zero model download.
    for concept in match_concept_anchors(tok) {
      add_hashed(acc, b"cpt:", concept.as_bytes(), field_weight * 1.6);
    }

    prev_tok = Some(tok.as_str());
  }
}

fn add_hashed(acc: &mut [f32], prefix: &[u8], payload: &[u8], weight: f32) {
  let h = fnv1a_64(prefix, payload);
  let idx = (h as usize) % acc.len();
  let sign = if (h >> 63) == 0 { 1.0f32 } else { -1.0f32 };
  acc[idx] += sign * weight;
}

fn fnv1a_64(prefix: &[u8], payload: &[u8]) -> u64 {
  let mut hash: u64 = 0xcbf29ce484222325;
  for &b in prefix.iter().chain(payload.iter()) {
    hash ^= u64::from(b);
    hash = hash.wrapping_mul(0x100000001b3);
  }
  hash
}

fn is_stopword(tok: &str) -> bool {
  matches!(
    tok,
    "the"
      | "and"
      | "for"
      | "with"
      | "from"
      | "that"
      | "this"
      | "let"
      | "mut"
      | "pub"
      | "use"
      | "self"
      | "return"
      | "true"
      | "false"
      | "none"
      | "some"
  )
}

/// Maps high-frequency Chinese & English systems/AI engineering terms to shared concept buckets.
fn match_concept_anchors(tok: &str) -> &'static [&'static str] {
  match tok {
    "热重" | "重载" | "reload" | "sighup" | "refresh" | "invalidate" | "hot" => &["hot_reload"],
    "鉴权" | "认证" | "权限" | "登录" | "auth" | "jwt" | "bearer" | "rbac" | "oauth" | "token"
    | "claims" | "verify" => &["auth_jwt"],
    "网关" | "代理" | "转发" | "gateway" | "proxy" | "pingora" | "sni" | "upstream" | "tls" => {
      &["gateway_proxy"]
    }
    "搜索" | "检索" | "索引" | "search" | "tantivy" | "bm25" | "index" | "query" | "rrf"
    | "ranking" => &["search_index"],
    "向量" | "嵌入" | "语义" | "embed" | "embedding" | "embeddings" | "vector" | "cosine"
    | "semantic" | "candle" | "tensor" => &["dense_vector"],
    "缓存" | "cache" | "kv" | "lru" | "ttl" | "memo" => &["caching"],
    "显存" | "内存" | "量化" | "vram" | "memory" | "quant" | "quantize" | "nf4" | "int4"
    | "fp16" | "bf16" => &["memory_quant"],
    "路由" | "route" | "router" | "axum" | "endpoint" | "handler" | "middleware" => {
      &["http_routing"]
    }
    "数据" | "迁移" | "数据库" | "database" | "postgres" | "sql" | "seaorm" | "migration"
    | "entity" => &["database_orm"],
    "静态" | "资源" | "封面" | "static" | "assets" | "servedir" | "cover" | "svg" | "og" => {
      &["static_assets"]
    }
    "安全" | "防护" | "限流" | "csrf" | "csp" | "xss" | "rate" | "limit" | "sanitize" => {
      &["security_guard"]
    }
    "支付" | "订单" | "课程" | "payment" | "order" | "course" | "checkout" | "stripe"
    | "alipay" => &["commerce_course"],
    "论坛" | "评论" | "讨论" | "forum" | "comment" | "topic" | "reply" | "thread" => {
      &["forum_comments"]
    }
    "日志" | "追踪" | "监控" | "tracing" | "telemetry" | "metrics" | "span" | "instrument" => {
      &["observability"]
    }
    "解析" | "渲染" | "文档" | "markdown" | "mdx" | "pulldown" | "parse" | "render" | "toc" => {
      &["markdown_render"]
    }
    "案例" | "橱窗" | "case" | "cases" | "showcase" | "favorite" => &["cases_showcase"],
    _ => &[],
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use safetensors::tensor::{Dtype as SafeDtype, TensorView};
  use std::collections::BTreeMap;
  use std::time::{SystemTime, UNIX_EPOCH};

  fn temp_dir(tag: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap()
      .as_nanos();
    std::env::temp_dir().join(format!("rie_grep_embed_{tag}_{nanos}"))
  }

  #[test]
  fn hash_embedder_aligns_bilingual_queries_and_roundtrips_f16() {
    let chunks = vec![
      CodeChunk {
        chunk_id: 0,
        path: "crates/app/src/reload.rs".into(),
        language: "rust".into(),
        symbol: "async fn handle_sighup_reload".into(),
        start_line: 1,
        end_line: 12,
        content: "pub async fn handle_sighup_reload() { invalidate_cache(); }".into(),
      },
      CodeChunk {
        chunk_id: 1,
        path: "crates/modules/payment/src/checkout.rs".into(),
        language: "rust".into(),
        symbol: "async fn create_order".into(),
        start_line: 1,
        end_line: 12,
        content: "pub async fn create_order(amount: u64) -> u64 { amount }".into(),
      },
    ];

    let embedder =
      DenseEmbedder::from_config(&EmbedderConfig::HashNgram { dim: 256 }, Device::Cpu).unwrap();
    let matrix = embedder.encode_chunks(&chunks).unwrap();
    let store = VectorStore::from_tensor(matrix).unwrap();

    let dir = temp_dir("roundtrip");
    std::fs::create_dir_all(&dir).unwrap();
    let vec_path = dir.join("vectors.bin");
    store.save(&vec_path).unwrap();

    let reloaded = VectorStore::load(&vec_path, &Device::Cpu).unwrap();
    assert_eq!(reloaded.num_chunks(), 2);
    assert_eq!(reloaded.dim(), 256);

    // Chinese query with zero overlapping ASCII words still recalls SIGHUP reload chunk first.
    let q = embedder.encode_query("热重载").unwrap();
    let hits = reloaded.search(&q, 2).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].0, 0);

    let _ = std::fs::remove_dir_all(&dir);
  }

  #[test]
  fn bert_embedder_runs_offline_safetensors_forward_pass() {
    let dir = temp_dir("tiny_bert");
    std::fs::create_dir_all(&dir).unwrap();

    // 1. Minimal valid BERT config (1 layer, hidden=16, 2 heads, vocab=16).
    let hidden = 16usize;
    let intermediate = 32usize;
    let vocab_size = 16usize;
    let max_pos = 32usize;
    let config_json = serde_json::json!({
      "vocab_size": vocab_size,
      "hidden_size": hidden,
      "num_hidden_layers": 1,
      "num_attention_heads": 2,
      "intermediate_size": intermediate,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.0,
      "max_position_embeddings": max_pos,
      "type_vocab_size": 2,
      "initializer_range": 0.02,
      "layer_norm_eps": 1e-12,
      "pad_token_id": 0,
      "model_type": "bert"
    });
    std::fs::write(
      dir.join("config.json"),
      serde_json::to_string_pretty(&config_json).unwrap(),
    )
    .unwrap();

    // 2. Minimal HuggingFace WordLevel tokenizer.json (pure JSON, zero extra deps).
    let tokenizer_json = serde_json::json!({
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": { "type": "Lowercase" },
      "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": {
          "[PAD]": 0,
          "[UNK]": 1,
          "rust": 2,
          "search": 3,
          "tantivy": 4,
          "candle": 5,
          "vector": 6,
          "bm25": 7
        },
        "unk_token": "[UNK]"
      }
    });
    std::fs::write(
      dir.join("tokenizer.json"),
      serde_json::to_string_pretty(&tokenizer_json).unwrap(),
    )
    .unwrap();

    // 3. Deterministic safetensors weights matching candle-transformers BertModel layout.
    let mut buffers: BTreeMap<String, (Vec<usize>, Vec<u8>)> = BTreeMap::new();
    let mut add_tensor = |name: &str, shape: Vec<usize>, fill: f32| {
      let count: usize = shape.iter().product();
      let mut bytes = Vec::with_capacity(count * 4);
      for i in 0..count {
        let v = fill + ((i % 7) as f32 - 3.0) * 0.01;
        bytes.extend_from_slice(&v.to_le_bytes());
      }
      buffers.insert(name.to_string(), (shape, bytes));
    };

    add_tensor(
      "embeddings.word_embeddings.weight",
      vec![vocab_size, hidden],
      0.05,
    );
    add_tensor(
      "embeddings.position_embeddings.weight",
      vec![max_pos, hidden],
      0.01,
    );
    add_tensor(
      "embeddings.token_type_embeddings.weight",
      vec![2, hidden],
      0.0,
    );
    add_tensor("embeddings.LayerNorm.weight", vec![hidden], 1.0);
    add_tensor("embeddings.LayerNorm.bias", vec![hidden], 0.0);

    let p = "encoder.layer.0";
    add_tensor(
      &format!("{p}.attention.self.query.weight"),
      vec![hidden, hidden],
      0.02,
    );
    add_tensor(&format!("{p}.attention.self.query.bias"), vec![hidden], 0.0);
    add_tensor(
      &format!("{p}.attention.self.key.weight"),
      vec![hidden, hidden],
      0.02,
    );
    add_tensor(&format!("{p}.attention.self.key.bias"), vec![hidden], 0.0);
    add_tensor(
      &format!("{p}.attention.self.value.weight"),
      vec![hidden, hidden],
      0.02,
    );
    add_tensor(&format!("{p}.attention.self.value.bias"), vec![hidden], 0.0);
    add_tensor(
      &format!("{p}.attention.output.dense.weight"),
      vec![hidden, hidden],
      0.02,
    );
    add_tensor(
      &format!("{p}.attention.output.dense.bias"),
      vec![hidden],
      0.0,
    );
    add_tensor(
      &format!("{p}.attention.output.LayerNorm.weight"),
      vec![hidden],
      1.0,
    );
    add_tensor(
      &format!("{p}.attention.output.LayerNorm.bias"),
      vec![hidden],
      0.0,
    );
    add_tensor(
      &format!("{p}.intermediate.dense.weight"),
      vec![intermediate, hidden],
      0.02,
    );
    add_tensor(
      &format!("{p}.intermediate.dense.bias"),
      vec![intermediate],
      0.0,
    );
    add_tensor(
      &format!("{p}.output.dense.weight"),
      vec![hidden, intermediate],
      0.02,
    );
    add_tensor(&format!("{p}.output.dense.bias"), vec![hidden], 0.0);
    add_tensor(&format!("{p}.output.LayerNorm.weight"), vec![hidden], 1.0);
    add_tensor(&format!("{p}.output.LayerNorm.bias"), vec![hidden], 0.0);

    let views: Vec<(String, TensorView<'_>)> = buffers
      .iter()
      .map(|(name, (shape, bytes))| {
        (
          name.clone(),
          TensorView::new(SafeDtype::F32, shape.clone(), bytes).unwrap(),
        )
      })
      .collect();
    safetensors::serialize_to_file(views, &None, &dir.join("model.safetensors")).unwrap();

    // 4. Load through DenseEmbedder and run a real forward pass.
    let cfg = EmbedderConfig::Bert {
      model_dir: dir.to_string_lossy().to_string(),
    };
    let embedder = DenseEmbedder::from_config(&cfg, Device::Cpu).unwrap();
    assert_eq!(embedder.dim(), hidden);

    let q_tensor = embedder.encode_query("rust tantivy candle vector").unwrap();
    assert_eq!(q_tensor.dims2().unwrap(), (1, hidden));
    let vals = q_tensor.squeeze(0).unwrap().to_vec1::<f32>().unwrap();
    let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "expected unit norm, got {norm}");

    let _ = std::fs::remove_dir_all(&dir);
  }
}
