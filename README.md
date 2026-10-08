# Tokens to Transformers in Mojo

**Build GPT-2 from scratch, from tokenizer to training loop, in Mojo.**

This book takes you from raw bytes to a working language model. You will build a
BPE tokenizer, an attention mechanism, a GPT-2 model and a training loop, all in
[Mojo](https://www.modular.com/mojo), starting from first principles. Every chapter
ends with code you can run.

The book is code first. Each chapter shows the code, runs it, and explains it.

It is built on three open-source repositories, listed under [Code](#code). Nothing in
the book is pseudo-code. Every snippet comes from those repositories at a pinned
revision, and the test commands in each chapter reproduce them.

---

## Who this book is for

The book goes from tokenization to a working GPT-2 and a BERT sentiment classifier.
It cannot also teach the background those models rest on, so it assumes you already
have it and spends its pages on code. You should have:

- **A working idea of what a neural network is.** Layers, weights, a loss function,
  and training by gradient descent. You do not need to derive backpropagation.
- **know what embeddings are.** An embedding is a table of learned vectors, one per token ID.
  The book uses embeddings from Chapter 1 onward and does not teach them.
- **Read or heard about transformers.** You have seen an explanation of attention and
  the transformer architecture, for example in a video or an article. If you have read
  *Build a Large Language Model (From Scratch)* by Sebastian Raschka, even better:
  this book follows a similar path, and builds the pieces in Mojo.
- **Comfort reading Python.** Mojo syntax is close to Python. You do not need to know
  Mojo already. New Mojo constructs are explained where they first appear.

You do not need to know anything about tokenization. Chapter 1 starts from nothing.
Systems topics such as memory layout and SIMD are introduced where the code needs
them.

When a chapter relies on a concept it does not teach, it gives a short definition and
a pointer to a fuller explanation, then continues. If you get lost, follow the
pointer, then come back.

If you need a refresher first, these are good places to start:

- Sebastian Raschka, *Build a Large Language Model (From Scratch)*
- The neural network and transformer videos by 3Blue1Brown
- Andrej Karpathy's *Neural Networks: Zero to Hero* video series
- The [Mojo manual](https://docs.modular.com/mojo/manual/), if you want to read the
  language documentation alongside the code

---

## Read the book

**Chapter 1: How GPT Sees Your Text** · [read →](https://ratulb.github.io/tokens-to-transformers/ch01/)

> Eleven characters go in, two integers come out. Build a BPE tokenizer by hand, see
> where this small version breaks, then read how `mbpe`, a production-grade tokenizer
> that matches tiktoken byte for byte, fixes each problem.

| # | Chapter | Status |
|---|---------|--------|
| 1 | How GPT Sees Your Text | ✅ Published |
| 2 | Attention from Scratch | 🚧 In progress |
| 3 | Assembling GPT-2 | 📝 Planned |
| 4 | Pretraining on TinyStories | 📝 Planned |
| 5 | Fine-tuning for Classification | 📝 Planned |
| 6 | The BERT Encoder Side | 📝 Planned |

Chapter 1 is free to read. The rest of the book is in progress. Watch this repository
to follow along.

---

## What the book covers

- **Tokenization.** Byte-pair encoding from first principles: how a vocabulary is
  learned, how text is encoded, why byte-level vocabularies matter, and how one
  tokenizer implementation serves GPT-2, GPT-4 and GPT-4o with three different
  pre-tokenizers.
- **Attention.** Scaled dot-product attention, causal masking and multi-head
  attention, built by hand and then optimized.
- **The GPT-2 model.** Layer norm, GELU, feedforward blocks, residual connections and
  the full decoder stack.
- **Pretraining.** Loss calculation, the training loop, decoding strategies and
  loading pretrained weights.
- **Fine-tuning.** Classification heads, and the encoder side with BERT, fine-tuned
  for IMDB sentiment classification.

Everything is written in Mojo, from SIMD kernels up, and every layer is code you can read.
 
---

## Why Mojo

Mojo is a language from Modular with Python-like syntax that compiles to native code.
The book uses it because you can follow the whole stack, including memory layout,
vectorization and dispatch, without leaving the language. If you want to understand
what a tensor library does, you can read one here from top to bottom.

You do not need prior Mojo experience. See [Who this book is for](#who-this-book-is-for)
for what you do need.

---

## Code

Every chapter is backed by runnable code in three repositories:

| Repository | What it is |
|---|---|
| [`simple_bpe`](https://github.com/ratulb/simple_bpe) | A small BPE tokenizer for learning. Chapter 1 builds it first. |
| [`mbpe`](https://github.com/ratulb/mbpe) | A production-grade BPE tokenizer, compatible with tiktoken. |
| [`tenmo`](https://github.com/ratulb/tenmo) | A tensor library and neural network framework in Mojo. |

Clone them:

```bash
git clone https://github.com/ratulb/simple_bpe
git clone https://github.com/ratulb/mbpe
git clone https://github.com/ratulb/tenmo
```

Each chapter lists the exact revision of each repository it was checked against, and
the commands that run its code. Check out those revisions before you follow along, so
that the code matches the text. Unless a chapter says otherwise, the code is tested
with Mojo 1.1.0.

---

## Feedback

Found an error, or a place where the book assumes something it should have explained?
Open an issue on this repository.
