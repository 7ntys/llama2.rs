use std::{io::Error, usize};

use crate::config::Config;

pub struct Tokenizer {
    pub(crate) max_token_length: usize,
    pub(crate) words: Vec<Vec<u8>>,
    pub(crate) scores: Vec<f32>,
    pub(crate) byte_pieces: [u8; 256],
}

impl Tokenizer {
    pub fn new(file_path: &str, cfg: &Config) -> Result<Self, Error> {
        let buffer = std::fs::read(file_path)?;

        // Read max_token_length
        let val_bytes: [u8; 4] = buffer[0..4].try_into().unwrap();
        let max_token_length = i32::from_le_bytes(val_bytes) as usize;

        let mut words: Vec<Vec<u8>> = vec![];
        let mut scores: Vec<f32> = vec![];

        let mut buffer_index = 4;
        for _ in 0..cfg.vocab_size {
            // Read scores:
            let score_bytes = buffer[buffer_index..buffer_index + 4].try_into().unwrap();
            let scores_f32 = f32::from_le_bytes(score_bytes);
            buffer_index += 4;

            // Read word length:
            let word_length_bytes = buffer[buffer_index..buffer_index + 4].try_into().unwrap();
            buffer_index += 4;
            let word_length = i32::from_le_bytes(word_length_bytes) as usize;

            // Read from buffer_index to word_length:
            let word_bytes = buffer[buffer_index..buffer_index + word_length]
                .try_into()
                .unwrap();
            buffer_index += word_length;

            // record :
            words.push(word_bytes);
            scores.push(scores_f32);
        }
        let mut byte_pieces = [0u8; 256];

        for i in 0..256 {
            byte_pieces[i] = i as u8;
        }

        Ok(Self {
            max_token_length,
            words,
            scores,
            byte_pieces,
        })
    }

    pub fn decode(&self, prev_token: usize, token: usize) -> &[u8] {
        let word = &self.words[token];
        if prev_token == 1 && word[0] == b' ' {
            return &word[1..];
        }
        // Means special token :
        if token >= 3 && token <= 258 {
            let idx = token - 3;
            return &self.byte_pieces[idx..idx + 1];
        }

        return &word;
    }
}
