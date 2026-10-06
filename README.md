# Tokens to Transformers in Mojo

**Build a GPT-2 from scratch — tokenizer to training loop — in a systems language for AI.**

This book takes you from raw bytes to a working language model. You'll build a BPE
tokenizer, an attention mechanism, a GPT-2 model, and a training loop — all in
[Mojo](https://www.modular.com/mojo), all from first principles, with no framework
underneath you. Every chapter ends with code you can run.

The book is grounded in two open-source libraries: **[mbpe](https://github.com/ratulb/mbpe)**
(a production BPE tokenizer, tiktoken-compatible) and
**[Tenmo](https://github.com/ratulb/tenmo)** (a tensor library and neural network
framework in Mojo). Nothing in the book is pseudo-code — every snippet is drawn from
those two repos at a pinned revision, and the test commands reproduce them.

---

## Read the book

**Chapter 1 — How GPT Sees Your Text** · [read →](https://ratulb.github.io/tokens-to-transformers/ch01/)

> Eleven characters go in, two integers come out. Build a BPE tokenizer by hand, watch
> where the toy breaks, and close the gaps with a production engine that matches
> tiktoken byte for byte.

| # | Chapter | Status |
|---|---------|--------|
| 1 | How GPT Sees Your Text | ✅ Published |
| 2 | Attention from Scratch | 🚧 In progress |
| 3 | Assembling GPT-2 | 📝 Planned |
| 4 | Pretraining on TinyStories | 📝 Planned |
| 5 | Fine-tuning for Classification | 📝 Planned |
| 6 | The BERT Encoder Side | 📝 Planned |

Chapter 1 is free to read. The rest of the book is in progress — check back, or watch
this repo to follow along.

---

## What the book covers

- **Tokenization.** Byte-pair encoding from first principles: how a vocabulary is
  learned, how text is encoded, why byte-level vocabularies matter, and how one
  generic engine serves GPT-2, GPT-4, and GPT-4o through three different fronts.
- **Attention.** Scaled dot-product attention, causal masking, multi-head attention —
  built by hand, then optimized.
- **The GPT-2 model.** Layer norm, GELU, feedforward blocks, residual connections,
  the full decoder stack.
- **Pretraining.** Loss calculation, the training loop, decoding strategies,
  loading pretrained weights.
- **Fine-tuning.** Classification heads, instruction tuning, and the encoder side
  with BERT.

The through-line: everything is built in Mojo, from SIMD kernels up. No PyTorch, no
hidden BLAS calls, no framework between you and the metal.

---

## Why Mojo

Mojo is a systems language for AI that combines Python-level ergonomics with
C-level performance. The book uses it because it lets you see the whole stack —
memory layout, vectorization, dispatch — without leaving the language. If you've
ever wanted to understand what a tensor library actually does, this is the path.

You don't need prior Mojo experience to read the book. You do need to be comfortable
with Python-level code and have seen a neural network once.

---

## Code

Every chapter is backed by runnable code. Pin the revisions before you start:

```bash
git clone https://github.com/ratulb/simple_bpe   # the toy tokenizer
git clone https://github.com/ratulb/mbpe         # the production tokenizer
git clone https://github.com/ratulb/tenmo        # the tensor library


