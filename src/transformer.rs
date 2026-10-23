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
            self.attention_block(l, pos);
        }
        todo!()
    }

    fn attention_block(&mut self, l: usize, pos: usize) {
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
        let kv_dim = (self.cfg.transformer_dimension * self.cfg.n_kv_heads) / self.cfg.n_heads;
        let start_cache = l * self.cfg.seq_len * kv_dim + pos * kv_dim;
        let end_cache = l * self.cfg.seq_len * kv_dim + (pos + 1) * kv_dim;

        // weights.attention_key is the form (n_layers * kv_dim * transformer_dim) => (6 * 288 * 288)
        let start_kv = l * kv_dim * self.cfg.transformer_dimension;
        let end_kv = (l + 1) * kv_dim * self.cfg.transformer_dimension;
        matmul(
            &self.state.scratch,
            &self.weights.attention_key[start_kv..end_kv],
            &mut self.state.key_cache[start_cache..end_cache],
            self.cfg.transformer_dimension,
            kv_dim,
        );

        matmul(
            &self.state.scratch,
            &self.weights.attention_value[start_kv..end_kv],
            &mut self.state.value_cache[start_cache..end_cache],
            self.cfg.transformer_dimension,
            kv_dim,
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
