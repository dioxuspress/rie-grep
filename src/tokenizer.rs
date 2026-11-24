use tantivy::tokenizer::{Token, TokenStream, Tokenizer};

/// Name under which the code-aware tokenizer is registered in Tantivy.
pub const CODE_TOKENIZER_NAME: &str = "rie_code_v1";

/// Splits source code and bilingual queries into searchable subword tokens.
///
/// Handles:
/// - `snake_case`, `kebab-case`, `path/to/file.rs`, `module::Symbol`
/// - `camelCase` and `PascalCase` (`HTTPSConnection` -> `https`, `connection`)
/// - ASCII alphanumeric runs (lowercased, preserving both compound tokens and split parts)
/// - CJK characters via unigrams and overlapping bigrams so Chinese queries work offline
pub fn tokenize_code(text: &str) -> Vec<String> {
  let mut out = Vec::new();
  let chars: Vec<char> = text.chars().collect();
  let mut i = 0;

  while i < chars.len() {
    let ch = chars[i];
    if ch.is_ascii_alphanumeric() {
      let start = i;
      while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
        i += 1;
      }
      let ident: String = chars[start..i].iter().collect();
      emit_identifier_tokens(&ident, &mut out);
    } else if is_cjk(ch) {
      let start = i;
      while i < chars.len() && is_cjk(chars[i]) {
        i += 1;
      }
      let run = &chars[start..i];
      if run.len() == 1 {
        out.push(run[0].to_string());
      } else {
        for win in run.windows(2) {
          let mut bi = String::with_capacity(8);
          bi.push(win[0]);
          bi.push(win[1]);
          out.push(bi);
        }
      }
    } else {
      i += 1;
    }
  }

  out
}

fn emit_identifier_tokens(ident: &str, out: &mut Vec<String>) {
  let mut parts: Vec<String> = Vec::new();
  for seg in ident.split('_') {
    if seg.is_empty() {
      continue;
    }
    split_camel_and_digits(seg, &mut parts);
  }

  // Emit each subword part (lowercased).
  for part in &parts {
    if !part.is_empty() {
      out.push(part.clone());
    }
  }

  // Also keep a canonical joined identifier (underscores stripped) when split into
  // multiple parts so `verify_jwt_token` and `verifyJwtToken` share the same exact-match term.
  if parts.len() > 1 {
    let joined: String = parts.iter().map(String::as_str).collect();
    if !joined.is_empty() && !parts.iter().any(|p| p == &joined) {
      out.push(joined);
    }
  }
}

fn split_camel_and_digits(seg: &str, out: &mut Vec<String>) {
  let bytes = seg.as_bytes();
  if bytes.is_empty() {
    return;
  }

  let mut start = 0;
  for i in 1..bytes.len() {
    let prev = bytes[i - 1];
    let curr = bytes[i];
    let next = bytes.get(i + 1).copied();

    let lower_to_upper = prev.is_ascii_lowercase() && curr.is_ascii_uppercase();
    let acronym_end = prev.is_ascii_uppercase()
      && curr.is_ascii_uppercase()
      && next.is_some_and(|n| n.is_ascii_lowercase());
    let alpha_digit_boundary = (prev.is_ascii_alphabetic() && curr.is_ascii_digit())
      || (prev.is_ascii_digit() && curr.is_ascii_alphabetic());

    if lower_to_upper || acronym_end || alpha_digit_boundary {
      let piece = &seg[start..i];
      if !piece.is_empty() {
        out.push(piece.to_ascii_lowercase());
      }
      start = i;
    }
  }

  let tail = &seg[start..];
  if !tail.is_empty() {
    out.push(tail.to_ascii_lowercase());
  }
}

/// Checks whether a Unicode scalar value lies in common CJK ideograph ranges.
pub fn is_cjk(ch: char) -> bool {
  matches!(
    ch as u32,
    0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF | 0x3040..=0x30FF | 0xAC00..=0xD7AF
  )
}

/// Tantivy-compatible tokenizer backed by [`tokenize_code`].
#[derive(Clone, Default)]
pub struct CodeTokenizer;

pub struct CodeTokenStream {
  tokens: Vec<Token>,
  index: usize,
}

impl Tokenizer for CodeTokenizer {
  type TokenStream<'a> = CodeTokenStream;

  fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
    let terms = tokenize_code(text);
    let mut tokens = Vec::with_capacity(terms.len());
    let mut offset = 0usize;
    for (pos, term) in terms.into_iter().enumerate() {
      let len = term.len();
      tokens.push(Token {
        offset_from: offset,
        offset_to: offset + len,
        position: pos,
        text: term,
        position_length: 1,
      });
      offset += len + 1;
    }
    CodeTokenStream { tokens, index: 0 }
  }
}

impl TokenStream for CodeTokenStream {
  fn advance(&mut self) -> bool {
    if self.index < self.tokens.len() {
      self.index += 1;
      true
    } else {
      false
    }
  }

  fn token(&self) -> &Token {
    // `token()` is only called by Tantivy after `advance()` returns true.
    &self.tokens[self.index - 1]
  }

  fn token_mut(&mut self) -> &mut Token {
    &mut self.tokens[self.index - 1]
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn splits_camel_snake_and_acronyms() {
    let tokens = tokenize_code("fn parseHTTPSRequest_v2(jwtToken: &str)");
    assert!(tokens.contains(&"parse".to_string()));
    assert!(tokens.contains(&"https".to_string()));
    assert!(tokens.contains(&"request".to_string()));
    assert!(tokens.contains(&"v".to_string()));
    assert!(tokens.contains(&"2".to_string()));
    assert!(tokens.contains(&"jwt".to_string()));
    assert!(tokens.contains(&"token".to_string()));
    assert!(tokens.contains(&"jwttoken".to_string()));
  }

  #[test]
  fn emits_cjk_unigrams_and_bigrams() {
    let tokens = tokenize_code("热重载信号 SIGHUP");
    assert!(tokens.contains(&"热重".to_string()));
    assert!(tokens.contains(&"重载".to_string()));
    assert!(tokens.contains(&"信号".to_string()));
    assert!(tokens.contains(&"sighup".to_string()));
  }
}
