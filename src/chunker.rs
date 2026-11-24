use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use rayon::prelude::*;

use crate::CodeChunk;

/// Default maximum file size to index (1 MiB).
pub const DEFAULT_MAX_FILE_BYTES: u64 = 1024 * 1024;
/// Default maximum lines per code chunk.
pub const DEFAULT_MAX_CHUNK_LINES: usize = 48;
/// Default overlap lines when splitting long blocks.
pub const DEFAULT_OVERLAP_LINES: usize = 8;

/// Configuration for repository walking and source chunking.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
  pub max_file_bytes: u64,
  pub max_chunk_lines: usize,
  pub overlap_lines: usize,
  pub include_hidden: bool,
}

impl Default for ChunkConfig {
  fn default() -> Self {
    Self {
      max_file_bytes: DEFAULT_MAX_FILE_BYTES,
      max_chunk_lines: DEFAULT_MAX_CHUNK_LINES,
      overlap_lines: DEFAULT_OVERLAP_LINES,
      include_hidden: false,
    }
  }
}

/// Output of scanning a workspace into ordered code chunks.
#[derive(Debug, Clone)]
pub struct WalkResult {
  pub chunks: Vec<CodeChunk>,
  pub files_indexed: usize,
  pub files_skipped: usize,
}

/// Walks `root` honoring `.gitignore`, filters non-text/binary files, and produces
/// deterministic, symbol-aware [`CodeChunk`] entries.
pub fn walk_and_chunk(root: &Path, config: &ChunkConfig) -> Result<WalkResult> {
  let canonical_root = root
    .canonicalize()
    .with_context(|| format!("failed to resolve root {}", root.display()))?;

  let mut builder = WalkBuilder::new(&canonical_root);
  builder
    .hidden(!config.include_hidden)
    .git_ignore(true)
    .git_global(true)
    .git_exclude(true)
    .filter_entry(|entry| {
      let name = entry.file_name().to_string_lossy();
      !matches!(
        name.as_ref(),
        ".git" | ".rie-grep" | "target" | "node_modules" | "dist" | ".next" | "__pycache__"
      )
    });

  let mut candidate_files: Vec<PathBuf> = Vec::new();
  let mut skipped = 0usize;

  for entry in builder.build() {
    let Ok(entry) = entry else {
      skipped += 1;
      continue;
    };
    if !entry.file_type().is_some_and(|ft| ft.is_file()) {
      continue;
    }
    let path = entry.into_path();
    let Ok(meta) = std::fs::metadata(&path) else {
      skipped += 1;
      continue;
    };
    if meta.len() == 0 || meta.len() > config.max_file_bytes {
      skipped += 1;
      continue;
    }
    if detect_language(&path).is_none() {
      skipped += 1;
      continue;
    }
    candidate_files.push(path);
  }

  candidate_files.sort();

  let per_file: Vec<Option<Vec<RawChunk>>> = candidate_files
    .par_iter()
    .map(|path| {
      let bytes = std::fs::read(path).ok()?;
      // Skip binary blobs containing NUL bytes in the leading window.
      let probe_len = bytes.len().min(4096);
      if bytes[..probe_len].contains(&0) {
        return None;
      }
      let text = String::from_utf8(bytes).ok()?;
      let rel = path
        .strip_prefix(&canonical_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
      let language = detect_language(path)?;
      Some(chunk_source_raw(
        &rel,
        language,
        &text,
        config.max_chunk_lines,
        config.overlap_lines,
      ))
    })
    .collect();

  let mut chunks = Vec::new();
  let mut files_indexed = 0usize;
  let mut next_id = 0u32;

  for item in per_file {
    match item {
      Some(file_chunks) if !file_chunks.is_empty() => {
        files_indexed += 1;
        for raw in file_chunks {
          chunks.push(CodeChunk {
            chunk_id: next_id,
            path: raw.path,
            language: raw.language,
            symbol: raw.symbol,
            start_line: raw.start_line,
            end_line: raw.end_line,
            content: raw.content,
          });
          next_id = next_id.saturating_add(1);
        }
      }
      _ => {
        skipped += 1;
      }
    }
  }

  Ok(WalkResult {
    chunks,
    files_indexed,
    files_skipped: skipped,
  })
}

struct RawChunk {
  path: String,
  language: String,
  symbol: String,
  start_line: usize,
  end_line: usize,
  content: String,
}

/// Splits a single text file into structural chunks using declaration boundaries
/// and bounded sliding windows for long bodies.
pub fn chunk_source(
  rel_path: &str,
  language: &str,
  text: &str,
  max_lines: usize,
  overlap_lines: usize,
) -> Vec<CodeChunk> {
  chunk_source_raw(rel_path, language, text, max_lines, overlap_lines)
    .into_iter()
    .enumerate()
    .map(|(idx, raw)| CodeChunk {
      chunk_id: idx as u32,
      path: raw.path,
      language: raw.language,
      symbol: raw.symbol,
      start_line: raw.start_line,
      end_line: raw.end_line,
      content: raw.content,
    })
    .collect()
}

fn chunk_source_raw(
  rel_path: &str,
  language: &str,
  text: &str,
  max_lines: usize,
  overlap_lines: usize,
) -> Vec<RawChunk> {
  let max_lines = max_lines.max(8);
  let overlap_lines = overlap_lines.min(max_lines / 2);
  let lines: Vec<&str> = text.lines().collect();
  if lines.is_empty() {
    return Vec::new();
  }

  // Locate structural anchor lines (function/type/heading declarations).
  let mut anchors: Vec<(usize, String)> = Vec::new();
  for (idx, line) in lines.iter().enumerate() {
    if let Some(sym) = extract_symbol_declaration(language, line) {
      anchors.push((idx, sym));
    }
  }

  let mut sections: Vec<(usize, usize, String)> = Vec::new();
  if anchors.is_empty() {
    sections.push((0, lines.len(), file_stem_symbol(rel_path)));
  } else {
    if anchors[0].0 > 0 {
      sections.push((
        0,
        anchors[0].0,
        format!("{} (header)", file_stem_symbol(rel_path)),
      ));
    }
    for w in 0..anchors.len() {
      let start = anchors[w].0;
      let end = anchors.get(w + 1).map_or(lines.len(), |next| next.0);
      sections.push((start, end, anchors[w].1.clone()));
    }
  }

  let mut chunks = Vec::new();
  for (sec_start, sec_end, symbol) in sections {
    let mut cursor = sec_start;
    while cursor < sec_end {
      let end = (cursor + max_lines).min(sec_end);
      let body = lines[cursor..end].join("\n");
      if !body.trim().is_empty() {
        chunks.push(RawChunk {
          path: rel_path.to_string(),
          language: language.to_string(),
          symbol: symbol.clone(),
          start_line: cursor + 1,
          end_line: end,
          content: body,
        });
      }
      if end >= sec_end {
        break;
      }
      let step = max_lines.saturating_sub(overlap_lines).max(1);
      cursor += step;
    }
  }

  chunks
}

/// Detects supported source/config/doc languages by extension or filename.
pub fn detect_language(path: &Path) -> Option<&'static str> {
  if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
    return match ext.to_ascii_lowercase().as_str() {
      "rs" => Some("rust"),
      "py" => Some("python"),
      "ts" | "tsx" => Some("typescript"),
      "js" | "jsx" | "mjs" | "cjs" => Some("javascript"),
      "go" => Some("go"),
      "c" | "h" => Some("c"),
      "cc" | "cpp" | "cxx" | "hpp" => Some("cpp"),
      "java" => Some("java"),
      "kt" => Some("kotlin"),
      "swift" => Some("swift"),
      "sol" => Some("solidity"),
      "sql" => Some("sql"),
      "sh" | "bash" | "zsh" => Some("shell"),
      "toml" => Some("toml"),
      "yaml" | "yml" => Some("yaml"),
      "json" => Some("json"),
      "md" | "mdx" => Some("markdown"),
      "css" | "scss" => Some("css"),
      "html" | "htm" => Some("html"),
      _ => None,
    };
  }

  match path.file_name().and_then(|n| n.to_str()) {
    Some("Dockerfile" | "Makefile" | "Justfile") => Some("shell"),
    _ => None,
  }
}

fn file_stem_symbol(rel_path: &str) -> String {
  Path::new(rel_path)
    .file_name()
    .and_then(|n| n.to_str())
    .unwrap_or(rel_path)
    .to_string()
}

/// Extracts a concise symbol label when a line starts a top-level declaration or heading.
fn extract_symbol_declaration(language: &str, line: &str) -> Option<String> {
  let trimmed = line.trim_start();
  let indent = line.len() - trimmed.len();

  match language {
    "rust" => {
      if indent > 4 {
        return None;
      }
      let stripped = strip_rust_visibility(trimmed);
      for kw in [
        "async fn ",
        "const fn ",
        "unsafe fn ",
        "fn ",
        "struct ",
        "enum ",
        "trait ",
        "mod ",
      ] {
        if let Some(rest) = stripped.strip_prefix(kw) {
          let name = take_ident(rest);
          if !name.is_empty() {
            return Some(format!("{}{}", kw.trim_end(), format_symbol_suffix(name)));
          }
        }
      }
      if let Some(rest) = stripped.strip_prefix("impl") {
        let rest = rest.trim_start();
        if rest.starts_with('<') || rest.starts_with(|c: char| c.is_ascii_alphabetic()) {
          let header = rest.split('{').next().unwrap_or(rest).trim();
          let compact: String = header.chars().take(48).collect();
          if !compact.is_empty() {
            return Some(format!("impl {compact}"));
          }
        }
      }
      None
    }
    "python" => {
      if indent > 4 {
        return None;
      }
      for kw in ["async def ", "def ", "class "] {
        if let Some(rest) = trimmed.strip_prefix(kw) {
          let name = take_ident(rest);
          if !name.is_empty() {
            return Some(format!("{} {name}", kw.trim()));
          }
        }
      }
      None
    }
    "typescript" | "javascript" => {
      if indent > 4 {
        return None;
      }
      let stripped = trimmed.strip_prefix("export ").unwrap_or(trimmed);
      let stripped = stripped.strip_prefix("default ").unwrap_or(stripped);
      for kw in [
        "async function ",
        "function ",
        "class ",
        "interface ",
        "type ",
      ] {
        if let Some(rest) = stripped.strip_prefix(kw) {
          let name = take_ident(rest);
          if !name.is_empty() {
            return Some(format!("{} {name}", kw.trim()));
          }
        }
      }
      None
    }
    "go" => {
      if indent > 0 {
        return None;
      }
      if let Some(rest) = trimmed.strip_prefix("func ") {
        let header = rest.split('{').next().unwrap_or(rest).trim();
        let compact: String = header.chars().take(48).collect();
        if !compact.is_empty() {
          return Some(format!("func {compact}"));
        }
      }
      if let Some(rest) = trimmed.strip_prefix("type ") {
        let name = take_ident(rest);
        if !name.is_empty() {
          return Some(format!("type {name}"));
        }
      }
      None
    }
    "markdown" => {
      if trimmed.starts_with('#') {
        let heading = trimmed.trim_start_matches('#').trim();
        if !heading.is_empty() {
          return Some(heading.chars().take(64).collect());
        }
      }
      None
    }
    _ => None,
  }
}

fn strip_rust_visibility(s: &str) -> &str {
  if let Some(rest) = s.strip_prefix("pub(crate) ") {
    rest.trim_start()
  } else if let Some(rest) = s.strip_prefix("pub(super) ") {
    rest.trim_start()
  } else if let Some(rest) = s.strip_prefix("pub ") {
    rest.trim_start()
  } else {
    s
  }
}

fn take_ident(s: &str) -> &str {
  let end = s
    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
    .unwrap_or(s.len());
  &s[..end]
}

fn format_symbol_suffix(name: &str) -> String {
  format!(" {name}")
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn chunks_rust_file_along_symbol_boundaries() {
    let sample = "\
use std::io;

pub struct GatewayConfig {
  pub port: u16,
}

pub async fn reload_signal() -> io::Result<()> {
  Ok(())
}
";
    let chunks = chunk_source("src/gateway.rs", "rust", sample, 48, 8);
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].symbol, "gateway.rs (header)");
    assert_eq!(chunks[1].symbol, "struct GatewayConfig");
    assert_eq!(chunks[2].symbol, "async fn reload_signal");
    assert_eq!(chunks[2].start_line, 7);
  }

  #[test]
  fn splits_long_function_with_overlap() {
    let mut lines = vec!["pub fn huge_fn() {".to_string()];
    for i in 0..80 {
      lines.push(format!("  let v{i} = {i};"));
    }
    lines.push("}".to_string());
    let src = lines.join("\n");
    let chunks = chunk_source("src/big.rs", "rust", &src, 32, 8);
    assert!(chunks.len() >= 3);
    assert_eq!(chunks[0].start_line, 1);
    assert_eq!(chunks[0].end_line, 32);
    assert_eq!(chunks[1].start_line, 25);
  }
}
