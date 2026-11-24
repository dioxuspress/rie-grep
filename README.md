# rie-grep

> **Single-binary local AI hybrid code search engine in pure Rust** — combining **Tantivy 0.27 BM25** exact identifier recall, **Candle 0.11 (CPU / Apple Metal)** dense vector search, and **Reciprocal Rank Fusion (RRF)** in one 10 MiB offline binary.

Part of the [Rust in Everything](https://rustineverything.app/case/rie-grep) open-source showcase ecosystem under [`dioxuspress`](https://github.com/dioxuspress).

## Why `rie-grep`?

- **Beyond literal regex**: `ripgrep` excels at exact patterns, while `rie-grep` answers intent queries like `"CONTENT_DIR cases static overlay ServeDir"` or Chinese cross-lingual queries like `"热重载 缓存 刷新"` against an English Rust codebase.
- **Zero Python, zero external vector DB, zero cloud API**: No Docker Compose, no Qdrant/Milvus server, no source code leaving your machine.
- **Dual dense backends (100% offline)**:
  - **Default `HashNgram-384d` (zero-download)**: Subword + character 3/4-gram + adjacent bigram + bilingual systems concept projection, L2-normalized and queried via a single Candle `Tensor::matmul` (`Q @ D^T`), stored on disk as compact `f16` (`vectors.bin`).
  - **Local BERT / BGE / MiniLM (`--model-dir`)**: Drop any local HuggingFace directory (`config.json` + `tokenizer.json` + `model.safetensors`) to run real Transformer attention-masked mean pooling via `candle-transformers` on CPU or Apple Metal.

## Quick Start

```bash
# Build & install (CPU default — fastest for < 50k chunks)
cargo install --path .

# Or build with Apple Silicon Metal support
cargo build --release --features metal

# 1. Index a codebase (honors .gitignore, skips binaries)
rie-grep index /path/to/repo

# 2. Hybrid search (Tantivy BM25 + Candle Dense + RRF)
rie-grep search "SIGHUP zero downtime reload" -p /path/to/repo

# Cross-lingual search + language filter + JSON output for Agents/MCP
rie-grep search "热重载 缓存 刷新" -p /path/to/repo --lang rs --top-k 5 --json

# Compare retrieval modes side by side
rie-grep search "jwt bearer claims" -p /path/to/repo --mode bm25
rie-grep search "jwt bearer claims" -p /path/to/repo --mode dense
rie-grep search "jwt bearer claims" -p /path/to/repo --mode hybrid

# 3. Inspect index breakdown & disk footprint
rie-grep stats /path/to/repo

# 4. Run reproducible latency + Recall/MRR benchmark
rie-grep bench /path/to/repo --queries benches/rie_queries.json
```

### Offline Transformer Mode (`--model-dir`)

```bash
rie-grep index /path/to/repo \
  --model-dir /path/to/bge-small-en-v1.5 \
  --device metal
```

No files are downloaded at runtime; `rie-grep` reads `config.json`, `tokenizer.json`, and memory-maps `model.safetensors` strictly from disk.

## Reproducible Benchmark

Measured on `dioxuspress/dioxuspress` (`508` source/doc files → `6,318` symbol-aware chunks, `10 MiB` stripped release binary with `--features metal`, Apple Silicon, using [`benches/rie_queries.json`](benches/rie_queries.json) containing 8 English + 8 Chinese queries):

- **Full Index Build**: `415.9 ms` on CPU (`15,190 chunks/s`), `497.0 ms` on Metal (`12,712 chunks/s`)
- **On-Disk Footprint**: `13.66 MiB` total (`tantivy/` `3.35 MiB` + `vectors.bin` f16 `4.63 MiB` + `chunks.jsonl` `5.69 MiB`)

| Mode | Device | P50 (ms) | P95 (ms) | P99 (ms) | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BM25 (Tantivy) | CPU | `0.12` | `0.24` | `0.25` | `18.8%` | `43.8%` | `50.0%` | `0.299` |
| Dense (Candle) | CPU | `0.28` | `0.30` | `0.30` | `37.5%` | `93.8%` | `93.8%` | `0.593` |
| **Hybrid (RRF)** | **CPU** | **`0.37`** | **`0.45`** | **`0.49`** | `18.8%` | **`68.8%`** | **`87.5%`** | **`0.392`** |
| Dense (Candle) | Metal | `0.70` | `0.74` | `0.75` | `37.5%` | `93.8%` | `93.8%` | `0.593` |
| Hybrid (RRF) | Metal | `0.75` | `0.85` | `0.94` | `18.8%` | `68.8%` | `87.5%` | `0.392` |

> **Hardware trade-off note**: At ~6k chunks × 384d, CPU BLAS (`gemm`) finishes the single `[1, 384] @ [384, 6318]` cosine matmul in `0.28 ms`, faster than Metal's `~0.42 ms` command-buffer dispatch overhead. Keep `--device cpu` as default for everyday repos, and switch to `--device metal` when encoding with local BERT/BGE models or indexing >50k chunks.

## Architecture

- [`src/chunker.rs`](src/chunker.rs): `.gitignore`-aware parallel traversal (`ignore` + `rayon`) with declaration boundary detection (`fn`/`struct`/`impl`/`trait`/`def`/`class`/`func`/`#`) and bounded sliding windows (`48` lines, `8`-line overlap).
- [`src/tokenizer.rs`](src/tokenizer.rs): `CodeTokenizer` splitting `camelCase`, `snake_case`, acronyms (`HTTPSRequest`), alpha-digit boundaries, and CJK bigrams while emitting canonical joined identifiers so `verify_jwt_token` and `verifyJwtToken` match symmetrically.
- [`src/bm25.rs`](src/bm25.rs): Persistent `tantivy 0.27` `MmapDirectory` index with `symbol (3.5x)` / `path (2.5x)` / `body (1.0x)` field boosts.
- [`src/embed.rs`](src/embed.rs): `DenseEmbedder` (`HashNgram` + offline `BertEmbedder`) and `VectorStore` (`RIEV0001` compact `f16` binary format + single-shot Candle `matmul` cosine retrieval).
- [`src/fusion.rs`](src/fusion.rs): Reciprocal Rank Fusion (`w / (k + rank)`) with full score provenance, language/path filtering, and query-dense snippet extraction.
- [`src/bench.rs`](src/bench.rs): Built-in benchmark harness reporting index throughput, P50/P95/P99 latency, Recall@1/5/10, and MRR@10.

## License

MIT
