use std::collections::HashMap;

use crate::{CodeChunk, SearchHit, SearchMode, SearchOptions};

#[derive(Default)]
struct ScoreEntry {
  bm25_rank: Option<usize>,
  bm25_score: Option<f32>,
  dense_rank: Option<usize>,
  dense_score: Option<f32>,
  fused_score: f32,
}

/// Fuses BM25 and dense vector rankings using Reciprocal Rank Fusion (RRF), applies
/// language/path filters, and extracts focused code snippets.
pub fn fuse_rankings(
  chunks: &[CodeChunk],
  bm25_hits: &[(u32, f32)],
  dense_hits: &[(u32, f32)],
  query: &str,
  options: &SearchOptions,
) -> Vec<SearchHit> {
  let mut table: HashMap<u32, ScoreEntry> = HashMap::new();
  let rrf_k = options.rrf_k.max(1.0);

  if matches!(options.mode, SearchMode::Hybrid | SearchMode::Bm25) {
    for (idx, &(chunk_id, score)) in bm25_hits.iter().enumerate() {
      let rank = idx + 1;
      let entry = table.entry(chunk_id).or_default();
      entry.bm25_rank = Some(rank);
      entry.bm25_score = Some(score);
      entry.fused_score += match options.mode {
        SearchMode::Bm25 => score,
        SearchMode::Hybrid => options.bm25_weight / (rrf_k + rank as f32),
        SearchMode::Dense => 0.0,
      };
    }
  }

  if matches!(options.mode, SearchMode::Hybrid | SearchMode::Dense) {
    for (idx, &(chunk_id, score)) in dense_hits.iter().enumerate() {
      let rank = idx + 1;
      let entry = table.entry(chunk_id).or_default();
      entry.dense_rank = Some(rank);
      entry.dense_score = Some(score);
      entry.fused_score += match options.mode {
        SearchMode::Dense => score,
        SearchMode::Hybrid => options.dense_weight / (rrf_k + rank as f32),
        SearchMode::Bm25 => 0.0,
      };
    }
  }

  let lang_filter = options.language.as_deref().map(|s| s.to_ascii_lowercase());
  let prefix_filter = options.path_prefix.as_deref();

  let mut candidates: Vec<(u32, ScoreEntry)> = table
    .into_iter()
    .filter(|(chunk_id, _)| {
      let Some(chunk) = chunks.get(*chunk_id as usize) else {
        return false;
      };
      if let Some(ref lang) = lang_filter
        && !language_matches(&chunk.language, lang)
      {
        return false;
      }
      if let Some(prefix) = prefix_filter
        && !chunk.path.starts_with(prefix)
      {
        return false;
      }
      true
    })
    .collect();

  candidates.sort_by(|(id_a, a), (id_b, b)| {
    b.fused_score
      .total_cmp(&a.fused_score)
      .then_with(|| id_a.cmp(id_b))
  });
  candidates.truncate(options.top_k);

  let query_terms = crate::tokenizer::tokenize_code(query);

  candidates
    .into_iter()
    .enumerate()
    .filter_map(|(idx, (chunk_id, entry))| {
      let chunk = chunks.get(chunk_id as usize)?;
      Some(SearchHit {
        rank: idx + 1,
        chunk_id,
        path: chunk.path.clone(),
        language: chunk.language.clone(),
        symbol: chunk.symbol.clone(),
        start_line: chunk.start_line,
        end_line: chunk.end_line,
        score: entry.fused_score,
        bm25_rank: entry.bm25_rank,
        bm25_score: entry.bm25_score,
        dense_rank: entry.dense_rank,
        dense_score: entry.dense_score,
        snippet: extract_best_snippet(&chunk.content, &query_terms, options.snippet_lines),
      })
    })
    .collect()
}

fn language_matches(chunk_lang: &str, filter: &str) -> bool {
  match filter {
    "rs" => chunk_lang == "rust",
    "py" => chunk_lang == "python",
    "ts" => chunk_lang == "typescript",
    "js" => chunk_lang == "javascript",
    "md" => chunk_lang == "markdown",
    "sh" => chunk_lang == "shell",
    other => chunk_lang == other,
  }
}

/// Picks the most query-dense contiguous line window inside a chunk for clean terminal display.
fn extract_best_snippet(content: &str, query_terms: &[String], max_lines: usize) -> String {
  let lines: Vec<&str> = content.lines().collect();
  if lines.len() <= max_lines || max_lines == 0 {
    return content.to_string();
  }

  let mut best_line = 0usize;
  let mut best_hits = 0usize;

  for (idx, line) in lines.iter().enumerate() {
    let lower = line.to_ascii_lowercase();
    let hits = query_terms
      .iter()
      .filter(|t| t.len() >= 2 && lower.contains(t.as_str()))
      .count();
    if hits > best_hits {
      best_hits = hits;
      best_line = idx;
    }
  }

  let half = max_lines / 2;
  let start = best_line
    .saturating_sub(half)
    .min(lines.len().saturating_sub(max_lines));
  let end = (start + max_lines).min(lines.len());
  lines[start..end].join("\n")
}

#[cfg(test)]
mod tests {
  use super::*;

  fn dummy_chunks() -> Vec<CodeChunk> {
    (0..3)
      .map(|i| CodeChunk {
        chunk_id: i,
        path: format!("src/mod_{i}.rs"),
        language: "rust".into(),
        symbol: format!("fn handler_{i}"),
        start_line: 1,
        end_line: 4,
        content: format!("pub fn handler_{i}() {{ /* body {i} */ }}"),
      })
      .collect()
  }

  #[test]
  fn rrf_rewards_consensus_across_bm25_and_dense() {
    let chunks = dummy_chunks();
    // Chunk 1 appears #2 in BM25 and #1 in Dense -> beats single-retriever #1s.
    let bm25 = vec![(0, 12.0), (1, 9.5)];
    let dense = vec![(1, 0.88), (2, 0.72)];

    let hits = fuse_rankings(&chunks, &bm25, &dense, "handler", &SearchOptions::default());
    assert_eq!(hits[0].chunk_id, 1);
    assert_eq!(hits[0].bm25_rank, Some(2));
    assert_eq!(hits[0].dense_rank, Some(1));
  }

  #[test]
  fn supports_language_alias_and_path_prefix_filtering() {
    let mut chunks = dummy_chunks();
    chunks[0].language = "markdown".into();
    chunks[0].path = "docs/guide.md".into();

    let bm25 = vec![(0, 10.0), (1, 8.0), (2, 7.0)];
    let opts = SearchOptions {
      mode: SearchMode::Bm25,
      language: Some("rs".into()),
      path_prefix: Some("src/mod_2".into()),
      ..SearchOptions::default()
    };

    let hits = fuse_rankings(&chunks, &bm25, &[], "handler", &opts);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].chunk_id, 2);
  }
}
