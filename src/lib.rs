use std::io::Error;

use crate::{config::Config, sampler::argmax, transformer::Transformer, weights::Weights};

mod config;
mod kernels;
mod runstate;
mod sampler;
mod transformer;
mod weights;

pub fn entry_point() -> Result<(), Error> {
    let file_path = "reference/llama2.c/stories15M.bin";
    let mut transformer = Transformer::build_transformer(file_path)?;
    let logits = transformer.forward(1, 0);

    // Compare first 5 logits
    for i in 0..6 {
        println!("{}", logits[i]);
    }

    let next = argmax(logits);

    println!("{next}");

    Ok(())
}
