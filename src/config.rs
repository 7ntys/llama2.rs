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
}

impl Config {
    pub fn new(file_path: &str) -> Result<Self, Error> {
        let mut f = File::open(file_path)?;
        let mut buffer = [0; 28];

        f.read_exact(&mut buffer)?;
        let mut values: Vec<i32> = vec![];
        for i in (0..28).step_by(4) {
            let val_bytes: [u8; 4] = buffer[i..i + 4].try_into().unwrap();
            let val = i32::from_le_bytes(val_bytes);
            values.push(val);
        }

        Ok(Config {
            transformer_dimension: values[0],
            hidden_dimension: values[1],
            n_layers: values[2],
            n_heads: values[3],
            n_kv_heads: values[4],
            vocab_size: values[5],
            seq_len: values[6],
        })
    }
}
