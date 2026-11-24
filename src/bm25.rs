use std::path::Path;

use anyhow::{Context, Result};
use tantivy::collector::TopDocs;
use tantivy::directory::MmapDirectory;
use tantivy::query::QueryParser;
use tantivy::schema::{
  FAST, Field, IndexRecordOption, STORED, Schema, TextFieldIndexing, TextOptions, Value,
};
use tantivy::tokenizer::TextAnalyzer;
use tantivy::{Index, IndexReader, ReloadPolicy, TantivyDocument, doc};

use crate::CodeChunk;
use crate::tokenizer::{CODE_TOKENIZER_NAME, CodeTokenizer};

/// Field handles for the Tantivy code index schema.
#[derive(Clone)]
pub struct Bm25Fields {
  pub chunk_id: Field,
  pub path: Field,
  pub language: Field,
  pub symbol: Field,
  pub body: Field,
}

/// Persistent Tantivy BM25 index over code chunks.
pub struct Bm25Index {
  index: Index,
  reader: IndexReader,
  fields: Bm25Fields,
}

impl Bm25Index {
  /// Builds (or replaces) the Tantivy index at `dir` from `chunks`.
  pub fn build(dir: &Path, chunks: &[CodeChunk]) -> Result<Self> {
    if dir.exists() {
      std::fs::remove_dir_all(dir)
        .with_context(|| format!("failed to clear previous BM25 dir {}", dir.display()))?;
    }
    std::fs::create_dir_all(dir)
      .with_context(|| format!("failed to create BM25 dir {}", dir.display()))?;

    let (schema, fields) = build_schema();
    let mmap = MmapDirectory::open(dir)
      .with_context(|| format!("failed to open MmapDirectory {}", dir.display()))?;
    let index = Index::create(mmap, schema, tantivy::IndexSettings::default())
      .context("failed to create Tantivy index")?;
    register_code_tokenizer(&index);

    let mut writer = index
      .writer(50_000_000)
      .context("failed to create Tantivy IndexWriter")?;
    for chunk in chunks {
      writer
        .add_document(doc!(
          fields.chunk_id => u64::from(chunk.chunk_id),
          fields.path => chunk.path.as_str(),
          fields.language => chunk.language.as_str(),
          fields.symbol => chunk.symbol.as_str(),
          fields.body => chunk.content.as_str(),
        ))
        .context("failed to add chunk document to Tantivy")?;
    }
    writer.commit().context("failed to commit Tantivy index")?;

    let reader = index
      .reader_builder()
      .reload_policy(ReloadPolicy::Manual)
      .try_into()
      .context("failed to open Tantivy IndexReader")?;

    Ok(Self {
      index,
      reader,
      fields,
    })
  }

  /// Opens an existing Tantivy index from `dir`.
  pub fn open(dir: &Path) -> Result<Self> {
    let mmap = MmapDirectory::open(dir)
      .with_context(|| format!("failed to open BM25 index at {}", dir.display()))?;
    let index = Index::open(mmap).context("failed to load Tantivy index")?;
    register_code_tokenizer(&index);

    let schema = index.schema();
    let fields = Bm25Fields {
      chunk_id: schema
        .get_field("chunk_id")
        .context("missing chunk_id field")?,
      path: schema.get_field("path").context("missing path field")?,
      language: schema
        .get_field("language")
        .context("missing language field")?,
      symbol: schema.get_field("symbol").context("missing symbol field")?,
      body: schema.get_field("body").context("missing body field")?,
    };

    let reader = index
      .reader_builder()
      .reload_policy(ReloadPolicy::Manual)
      .try_into()
      .context("failed to open Tantivy IndexReader")?;

    Ok(Self {
      index,
      reader,
      fields,
    })
  }

  /// Runs a BM25 query returning `(chunk_id, bm25_score)` pairs sorted descending.
  pub fn search(&self, query_text: &str, limit: usize) -> Result<Vec<(u32, f32)>> {
    let trimmed = query_text.trim();
    if trimmed.is_empty() || limit == 0 {
      return Ok(Vec::new());
    }

    let query_terms = crate::tokenizer::tokenize_code(trimmed);
    if query_terms.is_empty() {
      return Ok(Vec::new());
    }
    let normalized_query = query_terms.join(" ");

    let mut parser = QueryParser::for_index(
      &self.index,
      vec![self.fields.symbol, self.fields.path, self.fields.body],
    );
    parser.set_field_boost(self.fields.symbol, 3.5);
    parser.set_field_boost(self.fields.path, 2.5);
    parser.set_field_boost(self.fields.body, 1.0);

    let (query, _errors) = parser.parse_query_lenient(&normalized_query);
    let searcher = self.reader.searcher();
    let top_docs = searcher
      .search(&query, &TopDocs::with_limit(limit).order_by_score())
      .context("BM25 search failed")?;

    let mut hits = Vec::with_capacity(top_docs.len());
    for (score, doc_addr) in top_docs {
      let doc: TantivyDocument = searcher
        .doc(doc_addr)
        .context("failed to fetch Tantivy document")?;
      if let Some(id) = doc.get_first(self.fields.chunk_id).and_then(|v| v.as_u64()) {
        hits.push((id as u32, score));
      }
    }

    Ok(hits)
  }
}

fn build_schema() -> (Schema, Bm25Fields) {
  let mut builder = Schema::builder();
  let code_indexing = TextFieldIndexing::default()
    .set_tokenizer(CODE_TOKENIZER_NAME)
    .set_index_option(IndexRecordOption::WithFreqsAndPositions);
  let code_text = TextOptions::default().set_indexing_options(code_indexing);

  let chunk_id = builder.add_u64_field("chunk_id", STORED | FAST);
  let path = builder.add_text_field("path", code_text.clone());
  let language = builder.add_text_field("language", code_text.clone());
  let symbol = builder.add_text_field("symbol", code_text.clone());
  let body = builder.add_text_field("body", code_text);

  let schema = builder.build();
  let fields = Bm25Fields {
    chunk_id,
    path,
    language,
    symbol,
    body,
  };
  (schema, fields)
}

fn register_code_tokenizer(index: &Index) {
  let analyzer = TextAnalyzer::builder(CodeTokenizer).build();
  index.tokenizers().register(CODE_TOKENIZER_NAME, analyzer);
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::time::{SystemTime, UNIX_EPOCH};

  fn temp_dir(tag: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap()
      .as_nanos();
    std::env::temp_dir().join(format!("rie_grep_bm25_{tag}_{nanos}"))
  }

  #[test]
  fn ranks_exact_symbol_above_body_mention() {
    let dir = temp_dir("rank");
    let chunks = vec![
      CodeChunk {
        chunk_id: 0,
        path: "src/auth.rs".into(),
        language: "rust".into(),
        symbol: "fn verify_jwt_token".into(),
        start_line: 1,
        end_line: 10,
        content: "pub fn verify_jwt_token(raw: &str) -> bool { !raw.is_empty() }".into(),
      },
      CodeChunk {
        chunk_id: 1,
        path: "src/notes.md".into(),
        language: "markdown".into(),
        symbol: "Architecture Notes".into(),
        start_line: 1,
        end_line: 5,
        content: "We call jwt validation inside middleware.".into(),
      },
    ];

    let idx = Bm25Index::build(&dir, &chunks).unwrap();
    let hits = idx.search("jwt token", 5).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].0, 0);

    let reopened = Bm25Index::open(&dir).unwrap();
    let hits2 = reopened.search("verifyJwtToken", 5).unwrap();
    assert_eq!(hits2[0].0, 0);

    let _ = std::fs::remove_dir_all(&dir);
  }
}
