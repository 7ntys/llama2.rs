use std::io::Error;

use crate::{config::Config, transformer::Transformer, weights::Weights};

mod config;
mod kernels;
mod runstate;
mod transformer;
mod weights;

pub fn entry_point() -> Result<(), Error> {
    let file_path = "reference/llama2.c/stories15M.bin";
    let transformer = Transformer::build_transformer(file_path)?;
    Ok(())
}
