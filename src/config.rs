use std::{
    fs::File,
    io::{Error, Read},
};

#[derive(Debug)]
pub struct Config {
    transformer_dimension: i32,
    hidden_dimension: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
    shared_weights: bool,
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
            transformer_dimension: values[0],
            hidden_dimension: values[1],
            n_layers: values[2],
            n_heads: values[3],
            n_kv_heads: values[4],
            vocab_size,
            seq_len: values[6],
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
