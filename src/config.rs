use std::{
    fs::File,
    io::{Error, Read},
};

#[derive(Debug)]
pub struct Config {
    pub(crate) transformer_dimension: usize,
    pub(crate) hidden_dimension: usize,
    pub(crate) n_layers: usize,
    pub(crate) n_heads: usize,
    pub(crate) n_kv_heads: usize,
    pub(crate) vocab_size: usize,
    pub(crate) seq_len: usize,
    pub(crate) shared_weights: bool,
}

impl Config {
    fn get_bytes_from_binary(file_path: &str) -> Result<[u8; 28], Error> {
        let mut f = File::open(file_path)?;
        let mut buffer = [0; 28];

        f.read_exact(&mut buffer)?;
        return Ok(buffer);
    }

    fn decode(buffer: [u8; 28]) -> Self {
        let mut values: Vec<i32> = vec![];
        for i in (0..28).step_by(4) {
            let val_bytes: [u8; 4] = buffer[i..i + 4].try_into().unwrap();
            let val = i32::from_le_bytes(val_bytes);
            values.push(val);
        }

        let vocab_size = values[5].abs();
        let shared_weights = values[5] > 0;

        Config {
            transformer_dimension: values[0] as usize,
            hidden_dimension: values[1] as usize,
            n_layers: values[2] as usize,
            n_heads: values[3] as usize,
            n_kv_heads: values[4] as usize,
            vocab_size: vocab_size as usize,
            seq_len: values[6] as usize,
            shared_weights,
        }
    }

    pub fn new(file_path: &str) -> Result<Self, Error> {
        let buffer = Config::get_bytes_from_binary(file_path)?;

        Ok(Config::decode(buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_shared_weights() {
        let values: Vec<i32> = vec![288, 768, 6, 6, 6, -32000, 256];

        let mut bytes: Vec<_> = vec![];

        for i in (0..28).step_by(4) {
            let val = i32::to_le_bytes(values[i / 4]);
            bytes.extend_from_slice(&val);
        }

        let buffer: [u8; 28] = bytes.try_into().unwrap();
        let cfg = Config::decode(buffer);

        assert_eq!(cfg.shared_weights, false)
    }
}
