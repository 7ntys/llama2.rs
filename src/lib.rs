use std::io::Error;

use crate::config::Config;

mod config;
mod weights;

pub fn entry_point() -> Result<(), Error> {
    let cfg = Config::new("reference/llama2.c/stories15M.bin")?;
    println!("{cfg:#?}");
    Ok(())
}
