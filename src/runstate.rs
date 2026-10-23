use std::vec;

use crate::config::Config;

pub struct RunState {
    pub(crate) x: Vec<f32>,
    pub(crate) scratch: Vec<f32>,
    pub(crate) attention_out: Vec<f32>,
    pub(crate) hb: Vec<f32>,
    pub(crate) hb2: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) att: Vec<f32>,
    pub(crate) logits: Vec<f32>,
    pub(crate) key_cache: Vec<f32>,
    pub(crate) value_cache: Vec<f32>,
}

impl RunState {
    pub fn new(cfg: &Config) -> Self {
        let x: Vec<f32> = vec![0.0; cfg.transformer_dimension];
        let scratch: Vec<f32> = vec![0.0; cfg.transformer_dimension];
        let attention_out: Vec<f32> = vec![0.0; cfg.transformer_dimension];

        let hb: Vec<f32> = vec![0.0; cfg.hidden_dimension];
        let hb2: Vec<f32> = vec![0.0; cfg.hidden_dimension];

        let q: Vec<f32> = vec![0.0; cfg.transformer_dimension];
        let att: Vec<f32> = vec![0.0; cfg.n_heads * cfg.seq_len];

        let logits: Vec<f32> = vec![0.0; cfg.vocab_size];
        let kv_dim = (cfg.transformer_dimension * cfg.n_kv_heads) / cfg.n_heads;

        let key_cache: Vec<f32> = vec![0.0; cfg.n_layers * cfg.seq_len * kv_dim];
        let value_cache: Vec<f32> = vec![0.0; cfg.n_layers * cfg.seq_len * kv_dim];

        Self {
            x,
            scratch,
            attention_out,
            hb,
            hb2,
            q,
            att,
            logits,
            key_cache,
            value_cache,
        }
    }
}
