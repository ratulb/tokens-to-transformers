---
layout: default
title: "Tokens to Transformers in Mojo"
description: "Build a GPT-2 from scratch in Mojo — tokenizer to training loop."
---

<p class="kicker">A book in progress · written in public</p>

# Tokens to Transformers in Mojo

<p class="lede">Build a GPT-2 from scratch — tokenizer to training loop — in a systems language for AI. From raw bytes to a working language model, all in Mojo, all from first principles, with no framework underneath you. Every chapter ends with code you can run.</p>

[Start reading → Chapter 1]({{ '/ch01/' | relative_url }}){:.cta}

## Contents {#contents}

{% include chapter_list.html %}

### Coming next

- **Chapter 2 — Attention from Scratch** <span class="status">in progress</span>
- **Chapter 3 — Assembling GPT-2** <span class="status">planned</span>
- **Chapter 4 — Pretraining on TinyStories** <span class="status">planned</span>
- **Chapter 5 — Fine-tuning for Classification** <span class="status">planned</span>
- **Chapter 6 — The BERT Encoder Side** <span class="status">planned</span>

## How this book works

Nothing here is pseudo-code. The book is grounded in two open-source libraries — [mbpe](https://github.com/ratulb/mbpe), a production BPE tokenizer, and [Tenmo](https://github.com/ratulb/tenmo), a tensor library and neural network framework in Mojo. Every snippet is drawn from those repos at a pinned revision, and the test commands reproduce them. Each chapter names its revisions up front, in a "Code for this chapter" note where the building starts.
