use std::{
    fs::{self, File},
    io::{Error, Read},
};

use crate::{config::Config, weights};

pub struct Weights {
    token_embeddings: Vec<f32>,
    attention_norm: Vec<f32>,
    attention_query: Vec<f32>,
    attention_key: Vec<f32>,
    attention_value: Vec<f32>,
    attention_output: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_down: Vec<f32>,
    ffn_up: Vec<f32>,
    final_norm: Vec<f32>,
    lm_head: Option<Vec<f32>>,
}

impl Weights {
    pub fn new(file_path: &str, cfg: Config) -> Result<Self, Error> {
        todo!()
    }

    fn get_float_vector_from_binary(
        buffer: &Vec<u8>,
        start_float: usize,
        size_float: usize,
    ) -> Result<Vec<f32>, Error> {
        let start_byte = start_float * 4;
        let size_byte = size_float * 4;
        let bytes = &buffer[start_byte..start_byte + size_byte];

        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn carve_binary(buffer: Vec<u8>, cfg: &Config) -> Result<Self, Error> {
        // Goal: Carve the huge binary file into dedicated var in Weights, according to Config declared size

        // Skip header:
        let header_size = 7;
        let mut index = header_size;

        // Compute useful stuff:
        let head_size = cfg.transformer_dimension / cfg.n_heads;
        let kv_dim = cfg.transformer_dimension * cfg.n_kv_heads / cfg.n_heads;

        // Process token_embeddings
        let embedding_size = cfg.vocab_size * cfg.transformer_dimension;
        let token_embeddings =
            Weights::get_float_vector_from_binary(&buffer, index, embedding_size)?;
        index += embedding_size;

        // Process attention_norm
        let attention_norm_size = cfg.n_layers * cfg.transformer_dimension;
        let attention_norm =
            Weights::get_float_vector_from_binary(&buffer, index, attention_norm_size)?;
        index += attention_norm_size;

        // Process attention_query
        let attention_query_size =
            cfg.n_layers * cfg.transformer_dimension * cfg.n_heads * head_size;
        let attention_query =
            Weights::get_float_vector_from_binary(&buffer, index, attention_query_size)?;
        index += attention_query_size;

        // Process attention_key
        let attention_key_size = cfg.n_layers * cfg.transformer_dimension * kv_dim;
        let attention_key =
            Weights::get_float_vector_from_binary(&buffer, index, attention_key_size)?;
        index += attention_key_size;

        // Process attention_value
        let attention_value_size = cfg.n_layers * cfg.transformer_dimension * kv_dim;
        let attention_value =
            Weights::get_float_vector_from_binary(&buffer, index, attention_value_size)?;
        index += attention_value_size;

        // Process attention_output
        let attention_output_size =
            cfg.n_layers * cfg.n_heads * head_size * cfg.transformer_dimension;
        let attention_output =
            Weights::get_float_vector_from_binary(&buffer, index, attention_output_size)?;
        index += attention_output_size;

        // Process ffn_norm
        let ffn_norm_size = cfg.n_layers * cfg.transformer_dimension;
        let ffn_norm = Weights::get_float_vector_from_binary(&buffer, index, ffn_norm_size)?;
        index += ffn_norm_size;

        // Process ffn_gate
        let ffn_gate_size = cfg.n_layers * cfg.transformer_dimension * cfg.hidden_dimension;
        let ffn_gate = Weights::get_float_vector_from_binary(&buffer, index, ffn_gate_size)?;
        index += ffn_gate_size;

        // Process ffn_down
        let ffn_down_size = cfg.n_layers * cfg.transformer_dimension * cfg.hidden_dimension;
        let ffn_down = Weights::get_float_vector_from_binary(&buffer, index, ffn_down_size)?;
        index += ffn_down_size;

        // Process ffn_up
        let ffn_up_size = cfg.n_layers * cfg.transformer_dimension * cfg.hidden_dimension;
        let ffn_up = Weights::get_float_vector_from_binary(&buffer, index, ffn_up_size)?;
        index += ffn_up_size;

        // Process final_norm
        let final_norm_size = cfg.transformer_dimension;
        let final_norm = Weights::get_float_vector_from_binary(&buffer, index, final_norm_size)?;
        index += final_norm_size;

        // Skip freq_cis_real (useless)
        index += cfg.seq_len * head_size / 2;

        // Skip freq_cis_imag
        index += cfg.seq_len * head_size / 2;

        // Process lm_head (if not shared)
        let lm_head = if !cfg.shared_weights {
            let lm_head_size = cfg.vocab_size * cfg.transformer_dimension;
            index += lm_head_size;
            Some(Weights::get_float_vector_from_binary(
                &buffer,
                index,
                lm_head_size,
            )?)
        } else {
            None
        };

        Ok(Self {
            token_embeddings,
            attention_norm,
            attention_query,
            attention_key,
            attention_value,
            attention_output,
            ffn_norm,
            ffn_gate,
            ffn_down,
            ffn_up,
            final_norm,
            lm_head,
        })
    }

    fn load_binary(file_path: &str) -> Result<Vec<u8>, Error> {
        // Goal: Load every bytes from binary:

        let mut f = File::open(file_path)?;
        let metadata = fs::metadata(file_path)?;

        let mut buffer: Vec<u8> = vec![0; metadata.len() as usize];
        f.read(&mut buffer)?;
        Ok(buffer)
    }
}
