use std::io::Error;

use crate::{config::Config, kernels::*, runstate::RunState, weights::Weights};

pub struct Transformer {
    cfg: Config,
    weights: Weights,
    state: RunState,
}

impl Transformer {
    pub fn new(cfg: Config, weights: Weights, state: RunState) -> Self {
        Self {
            cfg,
            weights,
            state,
        }
    }

    pub fn build_transformer(file_path: &str) -> Result<Self, Error> {
        let cfg = Config::new(&file_path)?;
        let weights = Weights::new(&file_path, &cfg)?;
        let state = RunState::new(&cfg);

        Ok(Self {
            cfg,
            weights,
            state,
        })
    }

    pub fn forward(&mut self, token: usize, pos: usize) {
        self.retrieve_embeddings(token);

        for l in 0..self.cfg.n_layers {
            self.normalize_before_attention(l);
            self.fill_attention(l, pos);
            self.rope(l, pos);
            self.perform_attention(l, pos);
        }
        todo!()
    }

    fn perform_attention(&mut self, l: usize, pos: usize) {
        // Iterate over n_heads:
        let head_size_float = self.cfg.head_size as f32;
        for h in 0..self.cfg.n_heads {
            for t in 0..=pos {
                let start_cache = l * self.cfg.seq_len * self.cfg.kv_dim
                    + t * self.cfg.kv_dim
                    + h * self.cfg.head_size;
                let score = dot_product(
                    &self.state.q[h * self.cfg.head_size..(h + 1) * self.cfg.head_size],
                    &self.state.key_cache[start_cache..start_cache + self.cfg.head_size],
                );
                let normalized_score = score / head_size_float.sqrt();

                self.state.attention_scores[h * self.cfg.seq_len + t] = normalized_score;
            }

            softmax(
                &mut self.state.attention_scores[h * self.cfg.seq_len..(h + 1) * self.cfg.seq_len],
                pos + 1,
            );

            // Set scratch matrix to 0
            self.state.scratch[h * self.cfg.head_size..(h + 1) * self.cfg.head_size].fill(0.0);

            for t in 0..=pos {
                let start_cache = l * self.cfg.seq_len * self.cfg.kv_dim
                    + t * self.cfg.kv_dim
                    + h * self.cfg.head_size;
                for i in 0..self.cfg.head_size {
                    self.state.scratch[h * self.cfg.head_size + i] += self.state.attention_scores
                        [h * self.cfg.seq_len + t]
                        * self.state.value_cache[start_cache + i];
                }
            }
        }
    }

    fn rope(&mut self, l: usize, pos: usize) {
        let start_cache = l * self.cfg.seq_len * self.cfg.kv_dim + pos * self.cfg.kv_dim;
        for i in (0..self.cfg.transformer_dimension).step_by(2) {
            let head_dim = i % self.cfg.head_size;
            let x: f32 = head_dim as f32 / self.cfg.head_size as f32;
            let freq = 1.0 / (10_000.0_f32.powf(x));
            let angle = pos as f32 * freq;

            let cos = angle.cos();
            let sin = angle.sin();

            Transformer::rotate_pair(i, cos, sin, &mut self.state.q);

            if i < self.cfg.kv_dim {
                Transformer::rotate_pair(
                    i,
                    cos,
                    sin,
                    &mut self.state.key_cache[start_cache..start_cache + self.cfg.kv_dim],
                );
            }
        }
    }

    fn rotate_pair(i: usize, cos: f32, sin: f32, v: &mut [f32]) {
        let a = v[i];
        let b = v[i + 1];

        v[i] = a * cos - sin * b;
        v[i + 1] = a * sin + cos * b;
    }

    fn fill_attention(&mut self, l: usize, pos: usize) {
        // Fill in Q, K, V
        // weights.attention_query is in the form (n_layers * transformer_dim * transformer_dim) => (6 * 288 * 288)
        let start = l * self.cfg.transformer_dimension * self.cfg.transformer_dimension;
        let end = (l + 1) * self.cfg.transformer_dimension * self.cfg.transformer_dimension;
        matmul(
            &self.state.scratch,
            &self.weights.attention_query[start..end],
            &mut self.state.q,
            self.cfg.transformer_dimension,
            self.cfg.transformer_dimension,
        );

        // state.key_cache is in the form (n_layers * seq_len * kv_dim) => (6 * 256 * 288 )
        // with kv_dim = (n_kv_heads * head_size) => (6 * 48)
        // with head_size = (transformer_dim / n_heads) => (288 / 6)
        let start_cache = l * self.cfg.seq_len * self.cfg.kv_dim + pos * self.cfg.kv_dim;
        let end_cache = l * self.cfg.seq_len * self.cfg.kv_dim + (pos + 1) * self.cfg.kv_dim;

        // weights.attention_key is the form (n_layers * kv_dim * transformer_dim) => (6 * 288 * 288)
        let start_kv = l * self.cfg.kv_dim * self.cfg.transformer_dimension;
        let end_kv = (l + 1) * self.cfg.kv_dim * self.cfg.transformer_dimension;
        matmul(
            &self.state.scratch,
            &self.weights.attention_key[start_kv..end_kv],
            &mut self.state.key_cache[start_cache..end_cache],
            self.cfg.transformer_dimension,
            self.cfg.kv_dim,
        );

        matmul(
            &self.state.scratch,
            &self.weights.attention_value[start_kv..end_kv],
            &mut self.state.value_cache[start_cache..end_cache],
            self.cfg.transformer_dimension,
            self.cfg.kv_dim,
        );
    }

    fn normalize_before_attention(&mut self, l: usize) {
        let start = l * self.cfg.transformer_dimension;
        let end = (l + 1) * self.cfg.transformer_dimension;

        rmsnorm(
            &self.state.x,
            &self.weights.attention_norm[start..end],
            &mut self.state.scratch,
            self.cfg.transformer_dimension,
        );
    }

    fn retrieve_embeddings(&mut self, token: usize) {
        let start = token * self.cfg.transformer_dimension;
        let end = (token + 1) * self.cfg.transformer_dimension;
        self.state
            .x
            .copy_from_slice(&self.weights.token_embeddings[start..end]);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn verify_retrieve_embedings() {
        todo!()
    }
}
