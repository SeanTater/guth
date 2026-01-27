use anyhow::Result;
use sentencepiece::SentencePieceProcessor;

#[derive(Debug)]
pub struct TextTokenizer {
    sp: SentencePieceProcessor,
}

impl TextTokenizer {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let sp = SentencePieceProcessor::open(path)?;
        Ok(Self { sp })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let pieces = self.sp.encode(text)?;
        Ok(pieces.into_iter().map(|piece| piece.id).collect())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(self.sp.decode_piece_ids(ids)?)
    }

    pub fn prepare_text_prompt(text: &str) -> Result<(String, usize)> {
        let mut text = text.trim().to_string();
        if text.is_empty() {
            anyhow::bail!("Text prompt cannot be empty");
        }

        while text.contains("  ") {
            text = text.replace("  ", " ");
        }
        text = text.replace('\n', " ").replace('\r', " ");

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

#[cfg(test)]
mod tests {
    use super::TextTokenizer;

    #[test]
    fn prepare_text_prompt_basic() {
        let (text, frames) = TextTokenizer::prepare_text_prompt("hello world")
            .expect("valid text prompt");
        assert!(text.starts_with("        Hello"));
        assert!(text.ends_with('.'));
        assert_eq!(frames, 3);
    }
}
