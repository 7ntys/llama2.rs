use std::{
    io::{Error, Write},
    time::Instant,
};

use crate::{
    config::Config, sampler::Sampler, tokenizer::Tokenizer, transformer::Transformer,
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
    let mut sampler = Sampler::build_sampler(1.0, 42, 0.9);

    let mut timer: Option<Instant> = None;

    // Create a story :
    let mut token = 1;
    let mut pos = 0;
    let mut output = std::io::stdout();

    // Test encoding prompt:
    let prompt_tokens = tokenizer.encode("Hello my name is");

    // hard kv cache limit
    while pos < transformer.cfg.seq_len {
        let logits = transformer.forward(token, pos);
        let next: usize;

        if pos < prompt_tokens.len() - 1 {
            next = prompt_tokens[pos + 1];
        } else {
            next = sampler.sample(logits);
        }

        if next == 1 {
            break;
        }

        let value = tokenizer.decode(token, next);
        output.write_all(value)?;
        output.flush()?;

        pos += 1;
        token = next;
        if timer.is_none() {
            timer = Some(Instant::now());
        }
    }

    if let Some(t) = timer {
        let elapsed = t.elapsed().as_secs_f64();

        let token_per_second = pos as f64 / elapsed;
        eprintln!("tok/s : {token_per_second}");
    }

    Ok(())
}
