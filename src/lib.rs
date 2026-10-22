use std::io::Error;

use crate::{config::Config, weights::Weights};

mod config;
mod weights;

pub fn entry_point() -> Result<(), Error> {
    let file_path = "reference/llama2.c/stories15M.bin";
    let cfg = Config::new(&file_path)?;
    println!("{cfg:#?}");
    let weights = Weights::new(&file_path, &cfg);
    Ok(())
}
