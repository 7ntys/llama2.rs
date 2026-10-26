use clap::Parser;
use llama_rust::entry_point;

#[derive(Parser)]
struct Args {
    /// Weight file path
    weight_path: String,

    /// Tokenizer file path
    #[arg(short = 'z', long, default_value = "reference/llama2.c/tokenizer.bin")]
    tokenizer_path: String,

    /// Temperature : (0.0 -> 2.0) => lowest means lower originality but less errors
    #[arg(short, long, default_value_t = 0.0)]
    temperature: f32,

    /// Top_p : (0.0 -> 1.0) => pick only the top_p best choices
    #[arg(short = 'p', long, default_value_t = 0.0)]
    top_p: f32,

    /// Seed : for random purposes
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Your prompt
    #[arg(short = 'i', long, default_value = "")]
    prompt: String,
}

fn main() {
    let args = Args::parse();
    entry_point(
        args.weight_path,
        args.tokenizer_path,
        args.temperature,
        args.seed,
        args.top_p,
        args.prompt,
    )
    .unwrap()
}
