use std::{collections::HashMap, io::Error, usize};

use crate::config::Config;

pub struct Tokenizer {
    pub(crate) max_token_length: usize,
    pub(crate) words: Vec<Vec<u8>>,
    pub(crate) scores: Vec<f32>,
    pub(crate) byte_pieces: [u8; 256],
    pub(crate) token_number: HashMap<Vec<u8>, usize>,
}

impl Tokenizer {
    pub fn new(file_path: &str, cfg: &Config) -> Result<Self, Error> {
        let buffer = std::fs::read(file_path)?;

        // Read max_token_length
        let val_bytes: [u8; 4] = buffer[0..4].try_into().unwrap();
        let max_token_length = i32::from_le_bytes(val_bytes) as usize;

        let mut words: Vec<Vec<u8>> = vec![];
        let mut scores: Vec<f32> = vec![];
        let mut token_number: HashMap<Vec<u8>, usize> = HashMap::new();

        let mut buffer_index = 4;
        for i in 0..cfg.vocab_size {
            // Read scores:
            let score_bytes = buffer[buffer_index..buffer_index + 4].try_into().unwrap();
            let scores_f32 = f32::from_le_bytes(score_bytes);
            buffer_index += 4;

            // Read word length:
            let word_length_bytes = buffer[buffer_index..buffer_index + 4].try_into().unwrap();
            buffer_index += 4;
            let word_length = i32::from_le_bytes(word_length_bytes) as usize;

            // Read from buffer_index to word_length:
            let word_bytes: Vec<u8> = buffer[buffer_index..buffer_index + word_length]
                .try_into()
                .unwrap();
            let word_copy: Vec<u8> = word_bytes.clone();
            buffer_index += word_length;

            // record :
            words.push(word_bytes);
            scores.push(scores_f32);
            token_number.insert(word_copy, i);
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
            token_number,
        })
    }

    pub fn encode(&self, prompt: &str) -> Vec<usize> {
        let mut tokens: Vec<usize> = vec![1];
        if prompt.is_empty() {
            return tokens;
        }
        let mut buffer = [0u8; 4];
        let space_token_bytes = ' '.encode_utf8(&mut buffer).as_bytes();
        let space_token = self
            .token_number
            .get(space_token_bytes)
            .expect("no space in the vocab");
        tokens.push(*space_token);

        // Process all chars in prompt :
        for char in prompt.chars() {
            let token_bytes = char.encode_utf8(&mut buffer).as_bytes();
            let token = self.token_number.get(token_bytes);

            if let Some(t) = token {
                tokens.push(*t);
            } else {
                for byte in token_bytes {
                    let val = *byte as usize + 3;
                    tokens.push(val);
                }
            }
        }

        // Merge tokens by affinity :
        let mut found = true;
        let mut buf: Vec<u8> = vec![];
        while found {
            found = false;

            let mut idx_to_replace: Option<usize> = None;
            let mut new_token_number: Option<usize> = None;
            let mut curr_score: Option<f32> = None;

            for i in 0..tokens.len() - 1 {
                let a = tokens[i];
                let b = tokens[i + 1];

                let word_a: &Vec<u8> = &self.words[a];
                let word_b: &Vec<u8> = &self.words[b];

                buf.clear();
                buf.extend_from_slice(word_a);
                buf.extend_from_slice(word_b);

                if let Some(t) = self.token_number.get(&buf) {
                    let score = self.scores[*t];

                    match curr_score {
                        Some(s) => {
                            if s < score {
                                new_token_number = Some(*t);
                                idx_to_replace = Some(i);
                                curr_score = Some(score);
                                found = true
                            }
                        }
                        None => {
                            new_token_number = Some(*t);
                            idx_to_replace = Some(i);
                            curr_score = Some(score);
                            found = true
                        }
                    }
                }
            }

            // Did we find anything ?
            if found
                && let Some(i) = idx_to_replace
                && let Some(val) = new_token_number
            {
                tokens[i] = val;
                tokens.remove(i + 1);
            }
        }

        return tokens;
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
