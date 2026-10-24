---
layout: default
title: "Tokens to Transformers in Mojo"
description: "Build a GPT-2 from scratch in Mojo — tokenizer to training loop."
---

<p class="kicker">A book in progress · written in public</p>

# Tokens to Transformers in Mojo

<p class="lede">Build a GPT-2 from scratch — tokenizer to training loop — in a systems language for AI. From raw bytes to a working language model, all in <a href="https://docs.modular.com/mojo/">Mojo</a>, all from first principles, with no framework underneath you. Every chapter ends with code you can run. The book follows one thread from start to finish: text becomes token IDs, token IDs become vectors, vectors become contextual representations, and the model learns to predict the next token.</p>

[Start reading → Chapter 1]({{ '/ch01/' | relative_url }}){:.cta}

## Contents {#contents}

{% include chapter_list.html %}

### Coming next

- **Chapter 6 — Fine-tuning for Classification** <span class="status">in progress</span>
- **Chapter 7 — Instruction Fine-tuning** <span class="status">planned</span>
- **Chapter 8 — The BERT Encoder Side** <span class="status">planned</span>

## How this book works

Nothing here is pseudo-code. The book uses three open-source repositories. Chapter 1 builds [simple_bpe](https://github.com/ratulb/simple_bpe), a small tokenizer for learning, and leaves it behind. Everything the book carries forward stands on two production libraries — [mbpe](https://github.com/ratulb/mbpe), a production BPE tokenizer, and [Tenmo](https://github.com/ratulb/tenmo), a tensor library and neural network framework in Mojo. Every snippet is drawn from those repos at a pinned revision, and the test commands reproduce them. Each chapter names its revisions up front, in a "Code for this chapter" note where the building starts.
