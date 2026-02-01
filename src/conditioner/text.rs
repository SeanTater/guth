//! Text tokenization and conditioning for TTS.

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn_nn::{Embedding, EmbeddingConfig};
use sentencepiece::SentencePieceProcessor;

/// SentencePiece-based text tokenizer for the TTS model.
///
/// Handles text preprocessing (normalization, capitalization, punctuation) and
/// tokenization using a trained SentencePiece model.
#[derive(Debug)]
pub struct TextTokenizer {
    sp: SentencePieceProcessor,
}

impl TextTokenizer {
    /// Load a SentencePiece model from disk.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let sp = SentencePieceProcessor::open(path)?;
        Ok(Self { sp })
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let pieces = self.sp.encode(text)?;
        Ok(pieces.into_iter().map(|piece| piece.id).collect())
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(self.sp.decode_piece_ids(ids)?)
    }

    /// Prepare raw text for TTS generation.
    ///
    /// Applies normalization rules:
    /// - Trims whitespace and collapses multiple spaces
    /// - Capitalizes the first character
    /// - Adds trailing punctuation if missing
    /// - Pads short prompts for better prosody
    ///
    /// # Returns
    ///
    /// A tuple of (prepared_text, frames_after_eos) where `frames_after_eos` is a
    /// heuristic for how many frames to generate after end-of-sequence detection.
    pub fn prepare_text_prompt(text: &str) -> Result<(String, usize)> {
        let mut text = text.trim().to_string();
        if text.is_empty() {
            anyhow::bail!("Text prompt cannot be empty");
        }

        while text.contains("  ") {
            text = text.replace("  ", " ");
        }
        text = text.replace(['\n', '\r'], " ");

        let number_of_words = text.split_whitespace().count();
        let frames_after_eos_guess = if number_of_words <= 4 { 3 } else { 1 };

        if !text
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false)
        {
            let mut chars = text.chars();
            if let Some(first) = chars.next() {
                let mut updated = first.to_uppercase().to_string();
                updated.push_str(chars.as_str());
                text = updated;
            }
        }

        if text
            .chars()
            .last()
            .map(|c| c.is_alphanumeric())
            .unwrap_or(false)
        {
            text.push('.');
        }

        if text.split_whitespace().count() < 5 {
            text = "        ".to_string() + &text;
        }

        Ok((text, frames_after_eos_guess))
    }

    /// Split long text into sentence-like chunks with a token budget.
    pub fn split_into_best_sentences(&self, text: &str, max_tokens: usize) -> Result<Vec<String>> {
        let (text, _) = Self::prepare_text_prompt(text)?;
        let text = text.trim();

        let list_of_tokens = self.encode(text)?;
        let mut end_of_sentence_tokens = self.encode(".!...?")?;
        if !end_of_sentence_tokens.is_empty() {
            end_of_sentence_tokens.remove(0);
        }

        let mut end_of_sentences_indices = vec![0usize];
        let mut previous_was_end_of_sentence_token = false;

        for (token_idx, token) in list_of_tokens.iter().enumerate() {
            if end_of_sentence_tokens.contains(token) {
                previous_was_end_of_sentence_token = true;
            } else {
                if previous_was_end_of_sentence_token {
                    end_of_sentences_indices.push(token_idx);
                }
                previous_was_end_of_sentence_token = false;
            }
        }
        end_of_sentences_indices.push(list_of_tokens.len());

        let mut nb_tokens_and_sentences = Vec::new();
        for window in end_of_sentences_indices.windows(2) {
            let start = window[0];
            let end = window[1];
            let sentence = self.decode(&list_of_tokens[start..end])?;
            nb_tokens_and_sentences.push((end - start, sentence));
        }

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_nb_tokens = 0usize;

        for (nb_tokens, sentence) in nb_tokens_and_sentences {
            if current_chunk.is_empty() {
                current_chunk = sentence;
                current_nb_tokens = nb_tokens;
                continue;
            }

            if current_nb_tokens + nb_tokens > max_tokens {
                chunks.push(current_chunk.trim().to_string());
                current_chunk = sentence;
                current_nb_tokens = nb_tokens;
            } else {
                current_chunk.push(' ');
                current_chunk.push_str(&sentence);
                current_nb_tokens += nb_tokens;
            }
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }

        Ok(chunks)
    }
}

/// Lookup-table conditioner that embeds token IDs for the transformer.
#[derive(Debug)]
pub struct LutConditioner<B: Backend> {
    pub tokenizer: Option<TextTokenizer>,
    pub embed: Embedding<B>,
    pub dim: usize,
    pub output_dim: usize,
}

impl<B: Backend> LutConditioner<B> {
    /// Create a new LUT conditioner and load its tokenizer.
    pub fn new(
        n_bins: usize,
        tokenizer_path: impl AsRef<std::path::Path>,
        dim: usize,
        output_dim: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let tokenizer = Some(TextTokenizer::open(tokenizer_path)?);
        let embed = EmbeddingConfig::new(n_bins + 1, dim).init::<B>(device);
        Ok(Self {
            tokenizer,
            embed,
            dim,
            output_dim,
        })
    }

    /// Embed token IDs into model space.
    pub fn forward_tokens(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed.forward(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::TextTokenizer;
    use serde::Deserialize;
    use std::fs;
    use std::path::PathBuf;

    #[derive(Debug, Deserialize)]
    /// Tokenizer fixture with model path and cases.
    struct TokenizerFixture {
        model_file: String,
        cases: Vec<TokenizerCase>,
        split_cases: Vec<SplitCase>,
    }

    #[derive(Debug, Deserialize)]
    /// Expected behavior for a single text prompt.
    struct TokenizerCase {
        text: String,
        prepared: String,
        frames_after_eos: usize,
        encoded: Vec<u32>,
        decoded: String,
    }

    #[derive(Debug, Deserialize)]
    /// Expected sentence splitting behavior for a prompt.
    struct SplitCase {
        text: String,
        max_tokens: usize,
        chunks: Vec<String>,
    }

    /// Resolve a fixture file path under `tests/fixtures`.
    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    /// Load the tokenizer fixture JSON.
    fn load_fixture() -> TokenizerFixture {
        let data =
            fs::read_to_string(fixture_path("tokenizer_fixture.json")).expect("fixture read");
        serde_json::from_str(&data).expect("fixture parse")
    }

    #[test]
    fn prepare_text_prompt_basic() {
        let (text, frames) =
            TextTokenizer::prepare_text_prompt("hello world").expect("valid text prompt");
        assert!(text.starts_with("        Hello"));
        assert!(text.ends_with('.'));
        assert_eq!(frames, 3);
    }

    #[test]
    fn prepare_text_prompt_rejects_empty() {
        let err = TextTokenizer::prepare_text_prompt("   \n").unwrap_err();
        assert!(err.to_string().contains("Text prompt cannot be empty"));
    }

    #[test]
    fn split_long_text_respects_max_tokens() {
        let fixture = load_fixture();
        let model_path = fixture_path(&fixture.model_file);
        let tokenizer = TextTokenizer::open(model_path).expect("open tokenizer");

        let long_text = "This is a sentence.".repeat(40);
        let max_tokens = 24;
        let chunks = tokenizer
            .split_into_best_sentences(&long_text, max_tokens)
            .expect("split");

        assert!(chunks.len() > 1);
        for chunk in chunks {
            let tokens = tokenizer.encode(&chunk).expect("encode chunk");
            assert!(tokens.len() <= max_tokens);
        }
    }

    #[test]
    fn tokenizer_matches_fixture() {
        let fixture = load_fixture();
        let model_path = fixture_path(&fixture.model_file);
        let tokenizer = TextTokenizer::open(model_path).expect("open tokenizer");

        for case in fixture.cases {
            let (prepared, frames) =
                TextTokenizer::prepare_text_prompt(&case.text).expect("prepare");
            assert_eq!(prepared, case.prepared);
            assert_eq!(frames, case.frames_after_eos);

            let encoded = tokenizer.encode(&prepared).expect("encode");
            assert_eq!(encoded, case.encoded);

            let decoded = tokenizer.decode(&encoded).expect("decode");
            assert_eq!(decoded, case.decoded);
        }

        for case in fixture.split_cases {
            let chunks = tokenizer
                .split_into_best_sentences(&case.text, case.max_tokens)
                .expect("split");
            assert_eq!(chunks, case.chunks);
        }
    }
}
