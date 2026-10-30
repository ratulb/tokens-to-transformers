# Tokens to Transformers in Mojo

**Build GPT-2 from scratch, from tokenizer to training loop, in Mojo.**

This book takes you from raw bytes to a working language model. You will build a BPE tokenizer, an attention mechanism, a GPT-2 model, and a training loop, all in [Mojo](https://www.modular.com/mojo), starting from first principles. Every chapter ends with code you can run. Nothing is pseudo-code.

The book is code first. Each chapter shows the code, runs it, and explains it.

It is built on three open-source repositories, listed under [Code](#code): a small learning tokenizer, used once in Chapter 1 and left behind, and the two production libraries — [`mbpe`](https://github.com/ratulb/mbpe) and [`tenmo`](https://github.com/ratulb/tenmo) — that everything else stands on. Every snippet comes from those repositories at a pinned revision, and the test commands in each chapter reproduce them.

---

## Why this book exists

Every developer who wants to understand transformers deeply hits the same wall: the tutorials use PyTorch, so the tensor operations, the memory layout, and the autograd graph stay hidden. You learn the architecture, but not the machine underneath it.

This book removes the framework. You write every layer of the model yourself — tokenization, attention, the transformer blocks, the training loop — in a language you can read from top to bottom. By the end you have a working GPT-2 you built yourself, and you understand every layer because you wrote every layer.

That is only possible because of the language it is written in. **Mojo** is the first language built on MLIR, created by Chris Lattner — the creator of LLVM, Clang, Swift, and MLIR. Mojo reached 1.0 in August 2026 and is now at 1.2.0; the compiler is open source under Apache 2.0 with LLVM exceptions.

Mojo exists to solve a problem you will hit the moment you try to deploy anything. Today the AI stack is fragmented: you train in Python and PyTorch, the framework compiles to C++, you get weights, and you run inference in yet another language — while writing kernels in CUDA that only run on one vendor's hardware. Mojo's stated goal is a single system language for a heterogeneous era: one codebase across Nvidia, AMD, and Apple Silicon GPUs, AWS Trainium, Google TPUs, and Qualcomm accelerators. Building GPT-2 in Mojo is how you understand both the model and the language that wants to unify the stack around it.

---

## What you will build

- **A BPE tokenizer from first principles.** How a vocabulary is learned, how text is encoded, why byte-level vocabularies matter, and how one tokenizer serves GPT-2, GPT-4, and GPT-4o with three different pre-tokenizers.
- **Embeddings.** From token IDs to vectors: batching windows, the lookup table, what training changes, and the positions that tell identical tokens apart.
- **Attention.** Scaled dot-product attention, causal masking, and multi-head attention, built by hand.
- **The GPT-2 model.** Layer norm, GELU, feedforward blocks, residual connections, and the full decoder stack.
- **Pretraining.** Loss calculation, the training loop, decoding strategies, and loading pretrained weights.
- **Fine-tuning.** Classification heads on GPT-2, instruction fine-tuning, and a planned BERT encoder side for sentiment classification.

Everything is written in Mojo, from SIMD kernels up, and every layer is code you can read.

---

## Who this book is for

The book goes from tokenization to a working GPT-2, then to classification and instruction fine-tuning; a BERT encoder chapter is planned. It cannot also teach the background those models rest on, so it assumes you already have it and spends its pages on code. You should have:

- **A working idea of what a neural network is.** Layers, weights, a loss function, and training by gradient descent. You do not need to derive backpropagation.
- **Know what embeddings are.** An embedding is a table of learned vectors, one per token ID. Chapter 2 teaches how the table works — lookup, training, positions — but it assumes the idea itself is familiar.
- **Read or heard about transformers.** You have seen an explanation of attention and the transformer architecture, for example in a video or an article. If you have read *Build a Large Language Model (From Scratch)* by Sebastian Raschka, even better: this book follows a similar path, and builds the pieces in Mojo.
- **Comfort reading Python.** Mojo syntax is close to Python. You do not need to know Mojo already. New Mojo constructs are explained where they first appear.

You do not need to know anything about tokenization. Chapter 1 starts from nothing. Systems topics such as memory layout and SIMD are introduced where the code needs them.

When a chapter relies on a concept it does not teach, it gives a short definition and a pointer to a fuller explanation, then continues. If you get lost, follow the pointer, then come back.

If you need a refresher first, these are good places to start:

- Sebastian Raschka, *Build a Large Language Model (From Scratch)*
- The neural network and transformer videos by 3Blue1Brown
- Andrej Karpathy's *Neural Networks: Zero to Hero* video series
- The [Mojo manual](https://docs.modular.com/mojo/manual/), if you want to read the language documentation alongside the code

---

## Read the book

**Chapter 1: From Text to Token IDs** · [read →](https://ratulb.github.io/tokens-to-transformers/ch01/)

> Eleven characters in, two integers come out. Read and run a small BPE tokenizer, see where this small version breaks, then read how `mbpe`, a production-grade tokenizer that matches tiktoken byte for byte, fixes each problem.

**Chapter 2: From Token IDs to Vectors**

> A flat list of IDs in, vectors out. Batch the story into windows, look up each ID's row, add positions — then check every shape and count against GPT-2's configuration.

**Chapter 3: Attention from Scratch**

> Eight windows in, eight windows out — but every vector now sees the others. Score, weight, mix, plus the mask that hides the future and twelve heads that divide the width.

**Chapter 4: Assembling GPT-2**

> One attention layer is not a language model. Layer norm, GELU, residuals, twelve blocks, a tied head — then OpenAI's weights, checked against the real GPT-2, ending in speech.

**Chapter 5: Pretraining on TinyStories**

> The loss that scores wrongness, the step downhill by hand, and a 6.9M model trained on a storybook with its own tokenizer — attention, neighbours, gradients, then samplers and the cache. The domain gap is the evidence.

| # | Chapter | Status |
|---|---|---|
| 1 | From Text to Token IDs | ✅ Published |
| 2 | From Token IDs to Vectors | ✅ Complete |
| 3 | Attention from Scratch | ✅ Complete |
| 4 | Assembling GPT-2 | ✅ Complete |
| 5 | Pretraining on TinyStories | ✅ Complete |
| 6 | Fine-tuning for Classification | ✅ Complete |
| 7 | Instruction Fine-tuning | 📝 Planned |
| 8 | The BERT Encoder Side | 📝 Planned |

Chapters 1 to 6 are complete and chapter 1 is free to read. Chapters 7 and 8 are planned. Watch this repository to follow along.

---

## A note on scope

The book teaches the *model* from first principles: the tokenizer, the embeddings, the attention mechanism, the transformer blocks, the training loop, and the fine-tuning procedure. You write every one of those.

It does not teach how to build a tensor library, an autograd engine, SIMD kernels, or GPU support. Those are the job of [`tenmo`](https://github.com/ratulb/tenmo), a companion library the author wrote and maintains. The book *uses* Tenmo's tensor operations and autograd as its substrate, in the same way other from-scratch LLM books use NumPy or PyTorch as theirs. Building Tenmo is a separate undertaking — a different book, not this one.

This is the same principle every from-scratch LLM book follows: Raschka's uses NumPy, Karpathy's uses PyTorch. The reader builds the model, not the numerical substrate. The difference here is that the substrate is also written in Mojo, by the same author, so you can inspect it end to end if you choose — but you are not asked to rebuild it.

---

## Why Mojo

Mojo is a language from Modular with Python-like syntax that compiles to native code. The book uses it because you can follow the whole stack, including memory layout, vectorization, and dispatch, without leaving the language. If you want to understand what a tensor library does, you can read one here from top to bottom.

You do not need prior Mojo experience. See [Who this book is for](#who-this-book-is-for) for what you do need.

---

## Code

Every chapter is backed by runnable code in three repositories. `simple_bpe` appears once, in Chapter 1, as a teaching artifact. `mbpe` and `tenmo` are the two production libraries everything else stands on:

| Repository | What it is |
|---|---|
| [`simple_bpe`](https://github.com/ratulb/simple_bpe) | A small BPE tokenizer for learning. Chapter 1 builds it, breaks it on purpose, and leaves it behind. |
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
with Mojo 1.2.0.

---

## Feedback

Found an error, or a place where the book assumes something it should have explained?
Open an issue on this repository.
