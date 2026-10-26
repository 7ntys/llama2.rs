# llama-rust

**Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) inference engine (`run.c`), ported to Rust from scratch. The repo is also a lab: use it to port llama2.c to your own language.**

There's no ML framework and no tensor library here, just flat `Vec<f32>` buffers, a for-loop matmul, and the Llama 2 architecture written out by hand. The output matches the C reference byte for byte.

```
$ cargo run --release -- reference/llama2.c/stories15M.bin -t 0
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
Lily wanted to play with the ball, but it was too high up in the sky. ...
```

## Results

All numbers below come from an Apple M5 running `stories15M.bin` on a single thread.

**Correctness.** Each run was compared with `diff` against `./run` from llama2.c, using the same flags:

| Run | vs `run.c` |
|---|---|
| `-t 0` (greedy), no prompt | identical (full story, 222 tokens) |
| `-t 0 -i "Hello my name is"` | identical |
| `-t 1 -p 0.9 -s 42 -i "Hello my name is"` (temperature + top-p) | identical |

The one difference left: `run.c` prints a trailing newline at the end, and this port doesn't yet.

**Speed.** Tokens per second, median of 3 runs, measured on 2026-10-25:

| | tok/s |
|---|---|
| `run.c` built with `-O3` | ~220 |
| this port (`--release`) | ~215 |

## Quickstart

Clone the repo, the C reference (it includes `tokenizer.bin`) and the 15M-parameter TinyStories model:

```bash
git clone https://github.com/7ntys/llama-rust.git
cd llama-rust
git clone --depth 1 https://github.com/karpathy/llama2.c.git reference/llama2.c
curl -L -o reference/llama2.c/stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

Generate a story:

```bash
cargo run --release -- reference/llama2.c/stories15M.bin -t 0
cargo run --release -- reference/llama2.c/stories15M.bin -t 1 -p 0.9 -s 42 -i "Tom and his dog"
```

Build and run the C reference to compare:

```bash
cd reference/llama2.c && make run && ./run stories15M.bin -t 0
```

### Flags

The flags are the same as `run.c`'s.

| Flag | Meaning | Default here |
|---|---|---|
| `<checkpoint>` | model file | required |
| `-t`, `--temperature` | `0` = greedy (argmax) | `0` |
| `-p`, `--top-p` | nucleus sampling; `0` or `1` = off | `0` |
| `-s`, `--seed` | RNG seed (same xorshift as `run.c`) | `42` |
| `-i`, `--prompt` | text to start from | none |
| `-z`, `--tokenizer-path` | tokenizer file | `reference/llama2.c/tokenizer.bin` |

The defaults are deterministic, which isn't the case in `run.c` (`-t 1 -p 0.9`, seed taken from the clock). When comparing the two, pass every flag explicitly on both sides.

## How it works

```
token ─► embedding (288)
      ─► 6 × [ RMSNorm → Q, K, V → RoPE → attention (6 heads, KV cache) → Wo → + residual
               RMSNorm → gate, up (768) → SwiGLU → down → + residual ]
      ─► RMSNorm ─► classifier (32,000 logits) ─► sampler ─► next token ─► decode ─► text
```

| File | Role | In `run.c` |
|---|---|---|
| `src/config.rs` | the 7-integer header of the checkpoint | `Config`, `read_checkpoint` |
| `src/weights.rs` | slicing the float array into tensors | `memory_map_weights` |
| `src/runstate.rs` | scratch buffers and KV cache | `RunState` |
| `src/kernels.rs` | `rmsnorm`, `softmax`, `matmul`, `silu` | `rmsnorm`, `softmax`, `matmul` |
| `src/transformer.rs` | `forward`: one token through the model | `forward` |
| `src/tokenizer.rs` | BPE `encode`, `decode` | `Tokenizer`, `encode`, `decode` |
| `src/sampler.rs` | argmax, temperature, top-p, xorshift RNG | `Sampler`, `sample` |
| `src/lib.rs` | the generation loop | `generate` |
| `src/main.rs` | command line (clap) | `main` |

The only dependency is [clap](https://crates.io/crates/clap), used for the command line. The inference engine itself uses only the standard library.

## Port it yourself: the lab

You can use this repo to port `run.c` to **your** language (Go, Python, Zig, Swift, TypeScript, …) from scratch. An AI assistant acts as your tutor: it explains and reviews, but **you write every line**.

| File | What it is |
|---|---|
| `lab/run_c_walkthrough.html` | A section-by-section guide to `run.c`: the 12 steps of `forward`, tiny worked examples, interactive widgets (RoPE, KV cache index, temperature), interview questions. Open it in a browser. Tips that only apply to Rust are marked **Rust note**. |
| `CLAUDE.md` | The tutor's instructions. Claude Code loads it automatically. With another assistant, point it to this file (or copy it to `AGENTS.md`). |
| `LEARNER.example.md` | Your profile: the languages you know, your background, how you learn. Copy it to `LEARNER.md` (gitignored), and the tutor adapts to it. |

Getting started:

1. Follow the Quickstart to get the C reference and the model.
2. Run `cp LEARNER.example.md LEARNER.md` and fill it in, including where your port lives (e.g. `ports/go/`).
3. Open your assistant in the repo and say "let's start".

**No peeking.** Don't read existing ports in your target language until yours works. If you're porting to Rust, don't open `src/`.

### Checkpoints

Each step has a value you can check. All values are for `stories15M.bin` and were checked against `run.c`.

| # | Checkpoint | Expected |
|---|---|---|
| 1 | Config header | `dim 288, hidden_dim 768, n_layers 6, n_heads 6, n_kv_heads 6, vocab_size 32000, seq_len 256` |
| 2 | Weights | the last tensor ends exactly at the end of the file (60,816,028 bytes); the classifier is shared with the embedding |
| 3 | `rmsnorm` | `x = [3, 4]`, `w = [1, 2]` → `[0.849, 2.263]` |
| 4 | `forward(token = 1, pos = 0)` | `logits[0..6] = -6.7908044, 0.8281139, -6.7904391, -6.7904944, -6.7904906, -6.7906294`; argmax = `9038` (`" Once"`) |
| 5 | Greedy, first 12 tokens | `9038 2501 263 931 29892 727 471 263 2217 7826 4257 365` |
| 6 | Greedy story | identical to `./run stories15M.bin -t 0`; it ends at position 222 with token `1` |
| 7 | `tokenizer.bin` | `max_token_length = 27`; token `9038` = `" Once"`; the last entry ends exactly at byte 433,869 |
| 8 | `encode("Once upon")` | `[1, 9038, 2501]` |
| 9 | xorshift RNG, seed 42 | `random_f32` → `0.3390852, 0.7822558, 0.7901370` |
| 10 | Sampling | identical to `./run stories15M.bin -t 1 -p 0.9 -s 42 -i "Hello my name is"` |

To get any intermediate value out of the C, write a small probe in `reference/llama2.c/`. The `#define TESTING` line turns off `run.c`'s own `main`:

```c
#define TESTING
#include "run.c"

int main(void) {
    Transformer t;
    build_transformer(&t, "stories15M.bin");
    float *logits = forward(&t, 1, 0);
    for (int i = 0; i < 6; i++) printf("%d %.7f\n", i, logits[i]);
    free_transformer(&t);
    return 0;
}
```

```bash
cc -O2 -o probe probe.c -lm && ./probe
```

**Sampling at high temperature can diverge for a reason that isn't a bug in your port.** clang may fuse the multiply-add in `matmul` into a single FMA instruction, which rounds differently in the last bits. At `-t 5`, that is enough to change which token gets drawn. To compare in that case, build the reference with `cc -O3 -ffp-contract=off -o run run.c -lm`.

## Status

- [x] Checkpoint loading, kernels, `forward` with KV cache
- [x] Greedy, temperature, top-p, xorshift RNG identical to `run.c`
- [x] BPE `encode` and `decode`, prompts
- [x] Command line
- [ ] Trailing newline, `-n` (max steps)
- [ ] Chat mode (needs a chat model: Llama 2 7B Chat is about 27 GB in fp32)
- [ ] Multithreaded matmul
- [ ] int8 quantization, with `runq.c` as the reference
- [ ] `stories42M` and `stories110M`
- [ ] A serving layer: paged KV cache, continuous batching, prefix caching

## Credits

- [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy, MIT license. `lab/run_c_walkthrough.html` quotes excerpts of `run.c`, which are covered by that license (see `lab/LLAMA2C_LICENSE`).
- The `stories*` checkpoints were trained by Andrej Karpathy on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (Eldan & Li, 2023).
