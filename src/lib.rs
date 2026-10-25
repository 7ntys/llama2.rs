use std::io::{Error, Write};

use crate::{
    config::Config, sampler::argmax, tokenizer::Tokenizer, transformer::Transformer,
    weights::Weights,
};

mod config;
mod kernels;
mod runstate;
mod sampler;
mod tokenizer;
mod transformer;
mod weights;

pub fn entry_point() -> Result<(), Error> {
    let weight_path = "reference/llama2.c/stories15M.bin";
    let tokenizer_path = "reference/llama2.c/tokenizer.bin";
    let mut transformer = Transformer::build_transformer(weight_path)?;
    let tokenizer = Tokenizer::new(tokenizer_path, &transformer.cfg)?;

    // Create a story :
    let mut token = 1;
    let mut pos = 0;
    let mut output = std::io::stdout();

    // hard kv cache limit
    while pos < transformer.cfg.seq_len {
        let logits = transformer.forward(token, pos);
        let next = argmax(logits);

        if next == 1 {
            break;
        }

        let value = tokenizer.decode(token, next);
        output.write_all(value)?;
        output.flush()?;

        pos += 1;
        token = next;
    }

    Ok(())
}
