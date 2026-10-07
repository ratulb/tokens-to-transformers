---
layout: chapter
title: "How GPT Sees Your Text"
book: "Tokens to Transformers in Mojo"
chapter: 1
permalink: /ch01/
description: >
  In this chapter we build a simple BPE tokenizer in Mojo, observe its shortcomings and then explore a production    ready tokenizer that trains, encodes/decodes and matches tiktoken byte for byte.
  
---

> **What this chapter covers**
>
> - What a token is, and why models use tokens instead of raw text
> - How BPE builds a vocabulary by merging frequent pairs
> - A toy BPE in Mojo — and its four gaps
> - How a production tokenizer closes those gaps
> - How GPT-2, GPT-4, and GPT-4o share one BPE engine
> - What makes the implementation fast
> - How to verify correctness — and extend the engine with a new tokenizer


---


**Part I — Learn BPE.** A small character-level tokenizer, built from scratch. It works, but not suitable for production deployment. The whole point of this part is to understand what a BPE tokenizer has to accomplish and how it could do it. It has got shortcomings. In Part II, we explore `mbpe` - which circumvent these issues.

---

## 1. The problem

Run this:

```python
import mbpe
tok = mbpe.get_encoding("gpt2")
tok.encode("hello world")
# [31373, 995]
```

Eleven characters in, two integers out. Decode those two integers and you get the eleven characters back.

That round-trip is the whole job. A language model never sees text. It sees IDs. On every request the tokenizer sits on both ends of the model:

```text
"hello world" -> [tokenizer] -> [31373, 995] -> model -> [31373, 995] -> [tokenizer] -> "hello world"
```

Everything between the two arrows operates on IDs. Embeddings are looked up by ID. Attention mixes IDs. The output projection scores IDs. The model has no idea how the text was split. That decision is made beforehand, offline, when we train the tokenizer on a corpus and freeze the mapping. ID `31373` means `"hello"` for that model, because the weights were trained against that mapping.

So the question the tokenizer has to answer is a practical one: how do you map an unbounded stream of text — new words, typos, code, other languages, emoji — onto a fixed table of integers, deterministically and reversibly?

A token is whatever piece got an ID. It need not be a word. In some vocabularies you will see entries like these:

```text
" world"   "ing"   "'s"
```

Note the leading space in `" world"`. Whitespace isn't stripped and added back later. It is part of the token. That is why `"hello world"` and `"hello  world"` tokenize differently. The tokenizer encodes exact bytes, not words.

Scripts like Devanagari or emoji make the point sharper. Under GPT-2's vocabulary, `नमस्ते` and `😀` are not single tokens — they fall back to several byte-level pieces each, because GPT-2 never learned them whole. A different vocabulary trained on more Hindi or more emoji might grant them single IDs. Whether a string is "one token" is a property of the vocabulary, not of the script. Try it yourself once `mbpe` is installed (§14 lists the exact commands): encode an Assamese line under `gpt2`, `cl100k` and `o200k` and compare tokens per character across the three. That ratio is the tax a tokenizer levies on a language — and it is why tokenizer choice matters far beyond English.

## 2. What should a token be?

Every tokenizer picks a fixed vocabulary and rewrites all input as entries from that table. There are three options. The first two fail, which shows why the third wins.

**Words.** Intuitive, short sequences, embeddings carry meaning directly. But no fixed table covers English plus names, typos, identifiers like `resurrect_db_connection()`, URLs, and every other language. Whatever misses becomes `<UNK>`, and the information is gone:

```text
"the cat sat on zxqv" -> [the] [cat] [sat] [on] [<UNK>]
```

Worse, morphology is hidden. `run`, `runs`, `running`, `runner` get four unrelated IDs. The model can learn the relationship from context, but the representation doesn't help.

**Characters or bytes.** Coverage becomes perfect — 256 byte values represent anything. No `<UNK>` ever. But sequences explode, and attention costs grow quadratically:

```text
50 tokens  ->  2,500 pairwise positions
100 tokens -> 10,000 pairwise positions
```

The model also wastes capacity relearning that `t-h-e` spells "the".

**Subwords.** The compromise everyone ships. Keep pieces frequent enough to earn their own ID, fall back to bytes for everything else:

```text
running  -> run + ning
lower    -> low + er
```

Common words stay compact. Rare words decompose into smaller pieces. Unknown input decomposes all the way to bytes. No `<UNK>`, sequences stay short, and `un` in `unwanted`, `unhappy`, `unclear` is one unit the model can learn once.

| Representation | Vocabulary | Sequence length | Unseen words |
| -------------- | ---------- | --------------- | ------------ |
| Words          | Large      | Short           | Poor         |
| Characters     | Small      | Long            | Excellent    |
| Subwords       | Moderate   | Moderate        | Excellent*   |

`*` with a byte-level fallback.

> Subwords win because they turn an open vocabulary into a closed one without destroying structure.

One more angle before we build, because it hooks directly into Chapter 3. Vocabulary size is not free: every entry owns up to two rows — one in the embedding matrix, one in the output projection (GPT-2 ties the two, so one; most larger models keep both). A 50k vocabulary at 768 dimensions is ~38M parameters per side; 200k at the same width is ~150M. Bigger tables mean shorter sequences but fatter matrices — and every ID you add is an ID the model must learn to use. Keep that trade in mind when §6 picks a target size and §12 counts lines in a `.tiktoken` file. The tokenizer spends parameters before the transformer sees a single flop.

Which pieces get an ID? BPE answers with one rule:

> Find the most frequent adjacent pair. Merge it. Repeat.

> **Code for this chapter.** Everything below is adapted from two repos — `simple_bpe` (the toy) and `mbpe` (the engine). Pin them before building:
>
> ```bash
> git clone https://github.com/ratulb/simple_bpe
> git clone https://github.com/ratulb/mbpe
> cd simple_bpe && git checkout 13031914  # Jul 2026
> cd ../mbpe && git checkout 8cef990a    # Sep 2026 (splits, loads, benchmarks verified here)
> ```
>
> Every "run" below refers to those two checkouts. Toy, from its repo root with Mojo 1.1.0: `pixi install`, then `pixi run mojo main.mojo` (6 tests). Engine: copy flags from its `scripts/run_tests.sh` (`-I .` matters; see Pitfalls). Snippets are trimmed for print; the repo plus its tests win. Assumes: readable Mojo, dict/list costs, one transformer diagram seen; no tokenization background.

## 3. Building BPE by hand

We build the toy now. Character-level, no regex, merge rules stored in order. It trains, encodes, decodes, saves and loads. The full source is `simple_bpe/tokenizer.mojo`. What follows is that file with the scaffolding removed.

**Base vocabulary.** Collect every character in the corpus. Add `<UNK>` at 0 for anything unseen later.

```mojo
# Build the alphabet from all unique characters
var alphabet: List[String] = []
for word in word_freqs.keys():
    for letter in word.codepoints():
        var char_str = chr(Int(letter))
        if char_str not in alphabet:
            alphabet.append(char_str)
sort(alphabet)

self.vocab = List[String](capacity=vocab_size)
self.vocab.append(String("<UNK>"))
self.stoi = Dict[String, Int]()
self.stoi["<UNK>"] = 0
for i, char in enumerate(alphabet):
    self.vocab.append(char)
    self.stoi[char] = i + 1
```

For `"hello world"` the alphabet is eight characters plus `<UNK>`. Small on purpose — you can hold the whole vocabulary in your head while the merge loop runs.

**The merge loop.** Count every adjacent pair, weighted by word frequency. Merge the winner everywhere. Record the rule. Repeat.

```mojo
def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^
```

```mojo
while len(self.vocab) < vocab_size:
    var pair_freqs = _compute_pair_freqs(splits, word_freqs)
    if len(pair_freqs) == 0:
        break  # every word is a single token, nothing left to merge
    var best_pair: Tuple[String, String] = ("", "")
    var max_freq = -1
    for pair_freq in pair_freqs.items():
        var pair = pair_freq.key
        var freq = pair_freq.value
        if max_freq == -1 or max_freq < freq:
            best_pair = pair
            max_freq = freq
    _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
    var joined = best_pair[0].copy() + best_pair[1].copy()
    self.merges[best_pair] = joined
    self.vocab.append(joined)
    self.stoi[joined] = len(self.vocab) - 1
```

`_merge_pair` walks every word and splices the winner in with list slicing:

```mojo
if split[i] == a and split[i + 1] == b:
    split = ([e for e in split[:i]] + [a + b] + [e for e in split[i + 2:]])
else:
    i += 1
```

Two notes on this code. First, holding `i` after a merge is harmless but does no real work here: the fresh token `a+b` can never equal `a`, so the same rule cannot refire at the same position. The branch is kept because it mirrors the standard formulation and costs nothing. Second, the splice itself rebuilds the word's list on every occurrence — quadratic work per word in the worst case. This is slower but easier to follow; §11 removes the cost without changing the result.

The trace below needs one label first: run the toy itself on the single word `aaabdaaabac` and there is almost nothing to learn — the base is `<UNK>` at 0 plus the sorted alphabet `a, b, c, d` at 1–4, merges take IDs 5, 6, 7 onward, and a one-word corpus collapses greedily toward the whole word. Correct for the toy, useless as a teaching trace: most counts sit at 1 and the ending is foreordained. So what follows uses the production engine instead: byte-level base of 256, ties broken by first-seen order, on `mbpe`'s `GPT2Tokenizer` (`tests/test_tokenizer.mojo`, `test_wikipedia_example`, the canonical Wikipedia byte-pair example). Same algorithm as the toy, correct base for the example. Start with bytes:

```text
a a a b d a a a b a c
```

**Round 1 — count.** Every adjacent pair across the word:

| pair | count |
| ---- | ----- |
| `aa` | 4 |
| `ab` | 2 |
| `bd`, `da`, `ba`, `ac` | 1 each |

`aa` wins outright. Merge it to token 256 and rewrite:

```text
256 a b d 256 a b a c        (256 = "aa")
```

**Round 2 — count again.** `mbpe` keeps its pair table across rounds and updates only what the merge touched; the toy recounts from scratch each round (simpler, slower — §6 adopts the incremental version). Either way the counts here read: (256, a) and (a, b) tied at 2 each. `mbpe` breaks the tie toward the incumbent, the pair inserted earliest when the corpus was first walked — (a, b) predates anything containing 256, so it wins. Token 257 = `"ab"`:

```text
256 257 d 256 257 a c
```

The toy as written uses a strict `<` scan over its freshly rebuilt dict, so on this round it would keep the first maximum it encounters in iteration order rather than the incumbent rule above. Same counts, possibly different winner, different IDs downstream. That is precisely why the tie-break rule lives with the engine in §6 instead of the toy: IDs are baked into model weights, so the rule must be stated, pinned by test (`test_train_tie_breaking_is_deterministic`), and reproducible across runs — not left to dict iteration luck.

**Round 3.** (256, 257) occurs twice and nothing else clears 1. Token 258 = `"aaab"`:

```text
258 d 258 a c     i.e. IDs [258, 100, 258, 97, 99]
```

Count pairs, merge the winner, repeat, newest merge takes the next ID. That loop is training, and `mbpe` pins this exact run: `train(["aaabdaaabac"], 259)` then `encode("aaabdaaabac")` returns `[258, 100, 258, 97, 99]`.

**Encoding.** Split into words, break each word into characters, replay the merges in the order they were learned. Our toy splitter does two things: it turns every space into a `Ġ` marker (U+0120) prefixed on the next word, so `"hello world"` becomes `["hello", "Ġworld"]`, and it splits periods off words. A cheap stand-in for GPT-2's regex — §5 replaces it with the real one. This is the only place `Ġ` is introduced, and decoding is its exact inverse. (Where does `Ġ` itself come from? It is OpenAI's `bytes_to_unicode` table showing through: byte `0x20`, the space, is remapped to the visible character U+0120 so BPE never has to stare at raw whitespace. The toy borrows the same marker rather than inventing its own.)

```mojo
def encode(self, text: String) raises -> List[Int]:
    var tokens = self._tokenize(text)
    var ids = List[Int](capacity=len(tokens))
    for token in tokens:
        ids.append(self.stoi.get(token, 0))
    return ids^
```

The `.get(token, 0)` is the `<UNK>` fallback. The `_tokenize` loop itself is O(rules x tokens) — it scans every rule against every word. We fix the speed in §11 without changing the result.

**Decoding.** Look up each ID, join, turn `Ġ` back into a space:

```mojo
def decode(self, ids: List[Int]) raises -> String:
    if len(ids) == 0:
        return String("")
    var raw = StringSlice("").join([self.vocab[i] for i in ids])
    return String(StringSlice(raw).replace("Ġ", " "))
```

Train it on the four-sentence Hugging Face corpus and it round-trips:

```mojo
var tok = BPETokenizer()
tok.train(corpus, 50)
tok.decode(tok.encode("This is not a token."))
# "This is not a token."
```

It works, but it has four gaps, covered in §§5–12.

## 4. Where the toy breaks

1. **No real pre-tokenization.** Our splitter does two things: it turns spaces into `Ġ` markers and splits periods off words. That already stops merges from crossing spaces — `"the cat"` never becomes one token here. What it gets wrong is everything else a real pre-tokenizer handles: punctuation glued to words (`"world!"` keeps its bang), contractions (`"don't"` never splits), digits running into letters. GPT-2's regex puts letters, digits, punctuation and whitespace in separate lanes with an optional leading space, and merges stay inside their lane. Without those lanes the toy learns pieces that don't generalize.

2. **Character-level base.** Anything not in training becomes `<UNK>`. Send an emoji the corpus never saw and you get `[0]`. Production tokenizers start from 256 bytes, so every input decomposes to something. No `<UNK>`, ever.

3. **Merge-list encoding.** We store ordered pairs and scan every rule against every word. `tiktoken`-style encoders instead replay merges in rank order — at each step the lowest-rank applicable pair wins, which is exactly the order training discovered them — backed by a flat `bytes → rank` table rather than a rule scan. Same merges, far fewer wasted comparisons; a vocabulary hit on a whole piece resolves in essentially one lookup instead of a sweep over all rules. §7 works the replay logic; §12 shows how the rank table is recovered from a bare vocabulary file.

4. **A private file format.** Our save/load is JSON with our own layout. The ecosystem speaks `.tiktoken`: one line per token, base64 bytes plus rank. If you can't read and write that, you can't share vocabularies with anyone.

> The toy teaches the concept. Closing those four gaps — byte-level vocab, regex pre-tokenization, rank-based encoding, and the file format — is what separates a teaching implementation from a production one.

---

**Part II — Build the real thing (§§5–12).** What it takes to close those gaps without losing correctness, speed, or compatibility with the files everyone already ships.

---

## 5. Split before you merge

Gap 1 from §4 was "no real pre-tokenization." The pre-tokenizer is the component in front of BPE that decides what BPE is allowed to see — where the boundaries fall, which pairs may merge, how spaces, punctuation and numbers are treated. It is not part of BPE itself.

If you feed a whole string to BPE as one unit, it happily merges across word boundaries and learns `"the cat"` as a token. The fix is to split first and let BPE work inside each piece.

GPT-2's rule is five cases: contractions (`'s`, `'t`, `'ll` …) split off; letter runs, digit runs, punctuation runs, each with an optional leading space; whitespace for the rest. `"Hello, world!"` becomes `["Hello", ",", " world", "!"]`. Note `" world"` keeps its space. That is why GPT-2 has separate tokens for `"world"` and `" world"`.

The split rule decides the vocabulary. GPT-2 and GPT-4 run the same BPE on the same idea of a corpus and learn incompatible vocabularies, because their split rules differ. When someone says "the GPT-2 tokenizer" they mean three things bound together: the GPT-2 pre-tokenizer, the GPT-2 vocabulary, the BPE algorithm. Change any one and the IDs change.

In `mbpe` this is a compile-time trait. The listing below shows the two splitting entry points; the rest of the trait (byte mapping, name, special tokens) has defaults or required constants spelled out in §14:

```mojo
trait PreTokenizer(Movable & Defaultable & Deinitable & Writable):
    comptime byte_map: ByteMapping
    def split[...](self, text: StringSpan[origin]) raises -> List[StringSpan[origin]]: ...
    def count_words[...](self, text: StringSpan[origin], mut counts: WordCounts) raises: ...
    # ... byte_to_id / id_to_byte (identity defaults), name(), special_tokens() (empty default)
```

`split` returns zero-copy views — no per-word allocation, the encode hot path — and the trait's default `count_words` delegates to it. Each production pre-tokenizer overrides `count_words` with a fused loop that never materializes words at all — the training path. The engine is `BPETokenizer[PT]`, generic over the trait. GPT-2, GPT-4 and GPT-4o are three instantiations of one engine; the engine itself never branches on which one it is. §14 puts that claim to the test by adding a fourth.

## 6. Training by counting and merging

Training is five steps plus a loop condition. Steps 1–2 run once to set up the table; steps 3–5 repeat until the vocabulary hits its target size or no pair has a positive count.

1. **Split** the corpus, count each distinct word.
2. **Break** each word into base units — bytes, not characters. Initial vocab is 256 entries, so everything is representable.
3. **Count** every adjacent pair, weighted by word frequency.
4. **Merge** the winner, assign the next ID.
5. **Update** counts incrementally — only the pairs each occurrence touches, not a full recount.

On `low low low lower lowest` the weighting is what makes it work — and this is where it differs from the §3 trace. There, most counts sat at 1 with a clear winner each round; here, counts decide. `low` occurs three times, so the opening table looks like this:

| pair | count |
| ---- | ----- |
| `(l, o)` | 3 + 1 + 1 = 5 |
| `(o, w)` | 3 + 1 + 1 = 5 |
| `(w, e)` | 1 + 1 = 2 |
| `(e, r)`, `(e, s)`, `(s, t)` | 1 each |

`(l, o)` and `(o, w)` tie at 5; the tie goes to `(l, o)` since `l` precedes `o` precedes `w` in every word and was inserted first. Token 256 = `"lo"`. The new pair `(lo, w)` inherits the full 5 and merges next to `"low"`. Then `(low, e)` at 2 merges to `"lowe"`. The only signal is frequency: pair counts weighted by how often each word showed up.

Two details carry forward. First, ties go to the incumbent — the pair inserted earliest when the corpus was first walked — so the order is reproducible and the ranks that downstream weights depend on don't shuffle between runs. The toy's fresh-recount loop can't promise this on its own; the engine's incremental table plus its first-seen rule can, and the test suite pins it. Second, step 5 is what makes training affordable: each merge touches affected words only, via a `where_dict` from pair to word list, instead of rescanning the corpus. That removes the per-merge rescan — the dominant cost on large corpora. Picking the best pair each round still scans the distinct-pair table, so "near-linear" describes scaling with corpus size in practice on real text, not a formal bound. A heap or lazy priority queue would tighten it further; the repo notes that as future work. §11 returns to what was measured versus what is claimed.

## 7. Encode replays — it never counts

Training counts pairs and decides the merges; encoding replays them. Given a new string:

1. Split it with the same pre-tokenizer.
2. Replay the merges in rank order — at every step, the lowest-rank applicable pair wins.

No counting. The merge list is a script; encoding executes it. This also closes gap 3 from §4: this section fixes the *order* (rank replay instead of scanning rules in storage order), and §12 supplies the *table*.

Replay `low`-family rules on `lowering`, a word the corpus never contained: `l o w e r i n g` -> `lo w e r i n g` -> `low e r i n g` -> `lowe r i n g`. The learned prefix fires on unseen input because the prefix is shared. That generalization is why a bounded vocabulary covers unbounded text.

`(lo, w)` cannot fire before `(l, o)` creates `lo`. Rank order is correctness, not convention: every merge's parents have lower rank than the merge itself. That invariant — `bytes[merged] == bytes[left] + bytes[right]` — is checked by test (`test_tiktoken_merge_consistency`), and §12 uses it to recover merge rules from a bare vocabulary file that never stored them.

**The round-trip guarantee.** For byte-level BPE over raw bytes, `decode(encode(s)) == s` holds for every input — that is the point of starting from 256 bytes. Two things people mistake for exceptions are not. *Distribution shift* (a vocabulary trained on news meeting medical text) changes token *quality* — more, smaller pieces — never round-trip correctness; the bytes still come back exactly. What can break the round trip in practice lives outside BPE proper: *special tokens* like `<|endoftext|>`, which the encoder may recognize or treat as literal text depending on `encode` versus `encode_ordinary` and the `allowed_special` setting (§12), and *invalid UTF-8*, where the byte sequence survives but the host `String` may not render it. The bytes come back exactly — specials and string validity are the only asterisks, and both are handled explicitly where they arise.

## 8. Whole Unicode, exactly

Byte-level buys coverage, but bytes aren't characters. `é` is two bytes, `😀` four. The pre-tokenizer must reason about codepoints (is this a letter?) while the tokenizer preserves bytes (round-trip exactly). `mbpe` does both: it reads bytes as codepoints for classification, never re-encodes.

Classification has to be right for all 1.1M codepoints. Hand-written if-chains rot — one missed boundary in a Unicode update and `U+1F6D5` silently misclassifies. `mbpe` generates the table from Python's `regex` module (Unicode 16.0) into a 3219-entry step function over the codepoint space, binary-searched at runtime, with an ASCII fast path for the hot region. The generator is the test: re-running it reproduces the table byte for byte.

Two quirks fall out of the byte scheme, and both live in the pre-tokenizer, not the engine. `bytes_to_unicode` explains both the table and the `Ġ` you met in §3: raw bytes 0x00–0xFF include control characters and whitespace that would confuse BPE bookkeeping, so OpenAI remaps every byte to a visible Unicode character — printables mostly to themselves, space (`0x20`) to `Ġ` (U+0120), other awkward bytes to higher codepoints. The mapping is bijective, so nothing is lost; the algorithm just never has to stare at a raw newline.

**Quirk 1 — which byte gets which rank.** Start with a checkable fact: all three shipped files share one base-256 order. Rank 0 is `!` (0x21), byte 0x00 sits at rank 188, `A` at rank 32 — identical in `gpt2.tiktoken`, `cl100k.tiktoken` and `o200k.tiktoken` (read the first lines and spot-check with base64). That shared order is OpenAI's `bytes_to_unicode` permutation: printables mostly to themselves, awkward bytes remapped, bijective throughout. So "shuffled" has no visible source in the files — every file looks the same at the bottom. The difference lives in fresh training:

| Encoding | Base-256 order in the shipped file | Fresh `train()` base assignment | `ByteMapping` enum |
| -------- | ---------------------------------- | ------------------------------- | ------------------ |
| GPT-2 / GPT-4 | shared order (rank 0 = `!`, 0x00 → 188) | identity: rank = byte value | `SEQUENTIAL` |
| GPT-4o | shared order (same table) | 256-entry table reproducing the shared order | `SHUFFLED` |

`SEQUENTIAL` means a freshly trained tokenizer numbers byte 0x00 as rank 0; `SHUFFLED` means it numbers bytes through the table, so a freshly trained vocabulary already matches the file convention. The enum names the training convention.

What does this change on load? Less than you might expect. `load_tiktoken` rebuilds `byte_to_rank` from the file's own rank assignments, and merge recovery (`_recover_merges` via `_bpe`) works purely from file bytes plus file ranks — checked: `o200k.tiktoken` loaded under a `SEQUENTIAL` pre-tokenizer encodes `hello world`, control bytes and `café` identically to the `SHUFFLED` pre-tokenizer, and round-trips byte-exact either way. The mapping does not fail on load, because the file carries the ranks and the loader trusts them.

It fails in two places the loader cannot fix. First, fresh `train()` builds base IDs through `id_to_byte` — identity or table — so training an o200k-style vocabulary under `SEQUENTIAL` bakes the wrong base assignment into every merged ID downstream. Second, and the one you will actually meet: the pre-tokenizer travels with its split rules. Load a file under the wrong *family* pre-tokenizer and the regexes diverge (`"1234"`, `iPhone` in §9) even where the base bytes agree — right bytes, wrong pieces, wrong IDs, round-trip still green. So the check is two parts, mapping first: does this pre-tokenizer's `ByteMapping` match the vocabulary it is about to serve, and does its split family match the file? The file declares neither; the type does.

## 9. Three families, one algorithm

The byte table is shared across families; the split rules differ. GPT-2 and the GPT-4 family run the same BPE over different pieces and learn incompatible vocabularies — and GPT-4o's pre-tokenizer differs from GPT-4's yet again. The split happens before BPE runs, so identical bytes take different paths to different IDs — and the round trip still comes back green either way.

The cells below come from `mbpe`'s own matchers, not copied from docs. Reproduce them with a few lines calling `split` on each pre-tokenizer (`GPT2Pretokenizer`, `GPT4Pretokenizer[SEQUENTIAL]`, `GPT4Pretokenizer[SHUFFLED]`):

| Input | GPT-2 | cl100k | o200k |
| ----- | ----- | ------ | ----- |
| `"1234"` | `["1234"]` | `["123", "4"]` | `["123", "4"]` |
| `"$hello"` | `["$", "hello"]` | `["$hello"]` | `["$hello"]` |
| `"foo/bar"` | `["foo", "/", "bar"]` | `["foo", "/bar"]` | `["foo", "/bar"]` |
| `"iPhone"` | `["iPhone"]` | `["iPhone"]` | `["i", "Phone"]` |
| `"abcDEF"` | `["abcDEF"]` | `["abcDEF"]` | `["abc", "DEF"]` |
| `"ABCdef"` | `["ABCdef"]` | `["ABCdef"]` | `["ABCdef"]` |

Two patterns to read off the table. First, the GPT-4 family shares three behaviors GPT-2 lacks: case-folded contractions, digit runs capped at 3, and letter runs that absorb one leading non-letter — that last rule is what keeps `"$hello"` whole and turns `"foo/bar"` into `"foo", "/bar"` identically on cl100k and o200k. Second, o200k's distinctive addition is only the case-transition rule: `iPhone` and `abcDEF` split where the other two keep them whole, while all-caps-led `ABCdef` stays whole under all three — the split bites on lower-to-upper transitions and single-lowercase leads, not on every case boundary. One more probe: o200k's punctuation pattern also names `/` explicitly, but the runs don't show it biting — `"a/"` gives `["a", "/"]`, `"//"` gives `["//"]`, and `"a//b"` gives `["a", "//", "b"]`, identically on all three pre-tokenizers. The `/` difference between GPT-2 and the GPT-4 family comes entirely from the letter-rule prefix. The same BPE with different splits produces incompatible IDs. Hence three pre-tokenizers — and one engine behind them that never branches on which one it is.

## 10. One engine, many pre-tokenizers

Part II in one declaration:

```mojo
struct BPETokenizer[PT: PreTokenizer = GPT2Pretokenizer](
    Sized & Movable & Writable & Tokenizer
):
    var pt: Self.PT
    var merges: List[MergeRule]
    var lookup_table: MergeLookup
    # ...
```

`PT` is compile-time. `BPETokenizer[GPT2Pretokenizer]` and `BPETokenizer[GPT4Pretokenizer[SHUFFLED]]` are different types, different codegen, pre-tokenizer logic inlined at each call site, dead branch eliminated. You can't swap pre-tokenizers at runtime — and you shouldn't, because the pre-tokenizer determines the vocabulary and the vocabulary is baked into weights. Compile-time checking prevents mismatches.

`GPT4Pretokenizer` is itself generic over `ByteMapping`: `SEQUENTIAL` gives cl100k, `SHUFFLED` gives o200k, with a `comptime if` picking the rule set per specialization. Three shipped tokenizers, one engine, no `if tokenizer == GPT2` anywhere in the hot path.

`Tokenizer` (encode/decode) and `PreTokenizer` (split/count/map) are different traits. The first is what a tokenizer *is*. The second is what its pre-tokenizer *does*. `BPETokenizer[PT]` takes the second as a parameter and implements the first.

One term needs defining before §12: *special tokens* such as `<|endoftext|>` are control IDs that sit outside the mergeable vocabulary. They mark boundaries (end of document, fill-in-the-middle slots) rather than text content, they are declared per family on the pre-tokenizer, and the file format deliberately does not carry them — §12 explains why.

## 11. How it earns its speed

> Reading guide. This section and §13 go deeper into systems performance — cache behavior, allocation counts, ABI crossings — than the rest of the chapter. On a first read, take the one-line version of each layer and move on; the tables in §13 still land. Come back when you want the mechanism.

Six layers, each removing a specific cost. Read each one as an answer to "where does the naive version waste its time?" — waste first, fix second.

1. **Views, not copies.** Naively, splitting `"hello world"` allocates two `String`s plus a list — N heap objects for N words, and a million-line corpus pays a million times. `split` returns `StringSpan` views into the input instead: one list allocation, zero copies. Decode mirrors it: build one `String` of the right total length, then `unsafe_memcpy` each token's bytes into it, instead of concatenating strings one at a time and regrowing the buffer N times.
2. **Flat byte arena.** Naively, each token's bytes live in their own allocation, so decoding chases N pointers to N unrelated addresses. `mbpe` stores every token's bytes back-to-back in one buffer plus an `(offset, length)` table per token. Lookup by ID still gathers from scattered offsets, but the per-token allocation is gone: one arena, one index table, memcpys into a single output. Creating a merged token is two memcpys plus one span append.
3. **Lookup cache.** Naively, every encode position hashes a pair `(a, b)` and probes a table — a hundred words means roughly a hundred hashes per encode. `mbpe` keeps a flat array for IDs under 1000 (every byte plus the first ~744 merges) indexed by `(id1 << 10) | id2` — a shift and an OR, then a load, no hash. Rarer high-ID pairs fall through to a real dict. The win is skipping the hash, not cache fit: at 4 bytes per entry the 1024×1024 table is ~4 MB, larger than L1 or L2 on most machines — the frequent byte-adjacent pairs are the ones that hit.
4. **Incremental counts.** Naively, each of V merges rescans W words of length L: O(V × W × L), too slow to use on a real corpus. `mbpe` updates only the pairs each occurrence touches (`(a,b)`, `(prev,a)`, `(b,next)` destroyed; `(prev,merged)`, `(merged,next)` created) and finds affected words through a `where_dict` instead of scanning for them. The rescan is gone; the per-round best-pair scan over distinct pairs remains, so treat the speedup as large and measured on real corpora rather than asymptotically tight.
5. **Two encoders.** Naively, one algorithm serves all word lengths and loses somewhere: a scan is quadratic on long URLs, a heap's setup dominates on short words. Under 32 tokens `mbpe` linearly scans; at 32+ it switches to a heap over a linked list, O(n log n). The 32 is `comptime SCAN_LIMIT` in `bpe/tokenizer.mojo`, set from measuring the crossover on real corpora per the code comment. There is no published crossover curve — only the constant and its comment — so treat 32 as a measured tunable rather than a derived one. Flip the one number, rerun `benchmarks/run.sh` on your hardware, watch the long-word columns move. That reproducibility is the claim.

6. **Specialization.** Compile-time specialization inlines the pre-tokenizer into encode, so the hot path carries no dispatch.

## 12. Files on disk and the Python face

Closing gap 4 means reading and writing the files everyone already ships. A `.tiktoken` file is one token per line: `base64(bytes) rank`. `gpt2.tiktoken` has 50,256 lines for a 50,257 vocabulary — `<|endoftext|>` isn't in the file. Neither are any specials. The file carries only mergeable tokens. Specials are type-level: `GPT2Pretokenizer.special_tokens()` declares them, load re-registers them. Custom specials must be re-registered by hand after every load. That asymmetry is the format's most common gotcha, and it's deliberate — it keeps the file deterministic and diffable.

The file carries no merge list either. Merges are recovered from ranks using the §7 invariant: for each token from 256 up, run the rank-restricted mini-encoder on its bytes; if it yields exactly two pieces, those are the parents. Ranks *are* the history. Shuffle them and recovery yields a wrong engine. Deterministic training (§6 tie-break) is what keeps them right.

Most users meet this through Python. `python-binding/mbpe.mojo` compiles three concrete types to `_mbpe.so` (one method set each). `mbpe/__init__.py` wraps them in `GPT2Tokenizer` / `GPT4Tokenizer` / `GPT4oTokenizer` plus `get_encoding`, matching `tiktoken`'s surface including the `allowed_special` three-way split. `encode` handles specials, `encode_ordinary` doesn't, subset mode splits in Python around allowed specials. Everything else delegates.

```python
tok = mbpe.GPT2Tokenizer()
tok.train(["the cat sat on the mat"], vocab_size=300)
tok.save_tiktoken("mine.tiktoken")
```

Round-trips are exact across save/load. Compatibility is verified by byte-for-byte tests on real corpora.

---

**Part III — Prove it (§§13–14).** Measured speed with the caveats stated alongside, then the live proof that the engine is generic. Takeaways and references follow.

---

## 13. Benchmarks

Mechanism so far; now measurement. Same files, same API, one core.

Corpus: *Alice in Wonderland* (~150–170 KB), repeated/concatenated to 5 MB so per-call overhead amortizes. Metric: millions of tokens/sec, higher better, best-of-3 runs on one core (single-threaded throughout) — best-of-3 reports the peak; medians and spread are in `benchmarks/results/`, and short-string workloads will show smaller margins where call overhead dominates. Same shipped `.tiktoken` files for every implementation. Environment, regenerated per run: AMD EPYC 9B45, 4 cores, 15 Gi RAM, Debian 12; Python 3.14.7, Rust 1.98.1, tiktoken 0.14.0. One version note: this benchmark run used Mojo 1.0.0, while the chapter's snippets are pinned and verified on Mojo 1.1.0 — rerun `benchmarks/run.sh` to refresh the numbers; the rig, not the table, is the durable artifact.

Encode:

| Encoding | Mojo native | Py bindings | tiktoken (Py) | tiktoken-rs |
|---|---|---|---|---|
| gpt2 | **17** | 14.3 | 6.2 | 5.5 |
| cl100k | **13.7** | 11.4 | 5 | 4.4 |
| o200k | **10** | 9.1 | 7.1 | 7.4 |

Decode:

| Encoding | Mojo native | Py bindings | tiktoken (Py) | tiktoken-rs |
|---|---|---|---|---|
| gpt2 | **205.3** | 94.6 | 43.9 | 80 |
| cl100k | **222.5** | 90.4 | 45.6 | 87.2 |
| o200k | **213.3** | 90.7 | 48.5 | 89 |

Native is fastest in every row, and the bindings beat `tiktoken` (Py) in every row too — same API, same files, import swap. The margins differ by column, so here they are separately. Native vs `tiktoken` (Py): encode 2.74x, 2.74x, 1.41x (gpt2, cl100k, o200k) and decode 4.68x, 4.88x, 4.40x. Bindings vs `tiktoken` (Py): encode 2.31x, 2.28x, 1.28x and decode 2.16x, 1.98x, 1.87x. So native's margins run 1.4–2.7x on encode and 4.4–4.9x on decode; the bindings' run ~1.3–2.3x and ~1.9–2.2x. The 4–5x figure belongs to native alone.

What the numbers suggest. The native decode margin exceeds its encode margin, which is *consistent with* the arena removing per-token allocation on the decode path — but without an ablation run that link is unproven. The encode margin is identical on gpt2 and cl100k (2.74x native) and lower on o200k (1.41x): the narrowing comes mostly from the `mbpe` side slowing on o200k's larger vocabulary and costlier splits (17 → 10 native), while the baseline's own per-token cost moves the other way (`tiktoken` is actually faster on o200k than cl100k, 7.1 vs 5) — so no story in which "every implementation does more work" survives the table; read it as margin arithmetic, not mechanism. The bindings trail native by 10–20% on encode but 2.2–2.5x on decode — building a Python `str` copies every byte across the ABI, while a list of ints barely notices.

Three caveats before you quote this table. First, the 5 MB corpus is repetition of a small book, which flatters caches and skews merge-path behavior toward Alice's diction; a code sample or multilingual mix — where o200k's regex differences bite hardest — is the obvious next measurement, and the rig makes it cheap to add. Second, `tiktoken-rs` trailing Python `tiktoken` on gpt2/cl100k encode surprises people; all four ran single-threaded on the same box. The likely cause is overhead in the Rust port's split-and-match path on these vocabularies — unconfirmed, since profiling the Rust side is out of scope here. Third, training has no chart here — only an encode/decode rig plus a separate training harness (`benchmarks/bm_train.mojo`, with a vocab-size sweep in `bm_training.mojo`). Until training numbers land against a baseline like minbpe or HF `tokenizers`, read §6/§11 training claims as mechanism (rescans removed) rather than measured speedup.

For a Python user the comparison that counts is bindings vs `tiktoken` (Py): roughly 2x faster everywhere measured, no code change. And speed without identical IDs would be a different tokenizer — the suite checks byte-for-byte equality on real corpora, so the table compares the same work.

## 14. One more tokenizer, built live

§§5–10 argued the engine is generic. The payoff: a character-level pre-tokenizer, thirty lines, no engine changes. First, this pre-tokenizer splits per codepoint, so merges can only ever join bytes *within* one codepoint — training exhausts its pairs quickly on multibyte text and the learned vocabulary stays shallow. That makes it a deliberately weak fourth tokenizer and a strong architecture test: if even this pre-tokenizer trains, encodes, decodes, saves and loads untouched, the genericity claim holds. A whitespace or code-aware splitter would be the more convincing production fourth; the point here is the plug, not the pre-tokenizer. Second, the trait sketch in §5 omitted defaults for brevity — `count_words` defaults to delegating to `split`, and `name()` / `special_tokens()` have defaults this pre-tokenizer partially overrides — so the struct below compiles with only what it must define.

```mojo
struct CharPretokenizer(PreTokenizer):
    comptime byte_map: ByteMapping = ByteMapping.SEQUENTIAL

    def __init__(out self):
        pass

    @staticmethod
    def name() -> String:
        return String("char")

    def split[mut: Bool, //, origin: Origin[mut=mut]](
        self, text: StringSpan[origin]
    ) raises -> List[StringSpan[origin]]:
        var result = List[StringSpan[origin]]()
        var n = text.byte_length()
        if n == 0:
            return result^
        var span = text.as_bytes()
        var pos = 0
        while pos < n:
            var end = pos + utf8_codepoint_byte_length(span[pos])
            if end > n:
                end = n
            result.append(StringSpan(unsafe_from_utf8=span[pos:end]))
            pos = end
        return result^

    def write_to[T: Writer](self, mut writer: T):
        writer.write(String("CharPretokenizer"))
```

Use it:

```mojo
var tok = BPETokenizer[CharPretokenizer]()
```

Train, encode, decode, save, load — all present, all working. The tests round-trip multibyte text (`café`, `中文`, `😀`, Assamese `বন্ধ ঠাইৰ ভয়`) including characters training never saw, with no `<UNK>`. Bytes are the fallback. That is §2's promise, exercised.

Adding a tokenizer requires only writing a struct and naming it as a parameter.

For context: most tokenizer libraries already offer pluggable pre-tokenizers — Hugging Face `tokenizers` lets you compose normalizers, pre-tokenizers and models at runtime with training and serialization to match. The distinction here is narrower: `mbpe` specializes the whole pipeline at compile time for the pre-tokenizer in use, so the split loop, byte mapping and merge tables inline into the hot path with no dispatch left. Runtime composition allows mixing components at runtime; compile-time specialization inlines the loop. A new family here is a new pre-tokenizer — thirty lines, one trait, full train/encode/decode/save/load from the moment it compiles. To see the pattern carry weight beyond a demo, the natural next pre-tokenizer is a whitespace or code-aware splitter; the scaffolding for it is exactly what this section exercised.

From Python, the same engine is one import away — this is also the setup for the §1 exercise on tokens per character:

```python
import mbpe  # pip install mbpe, or build _mbpe.so per scripts/run_tests.sh
text = open("para.txt").read()  # or paste a paragraph literal here
for enc in ["gpt2", "cl100k", "o200k"]:
    tok = mbpe.get_encoding(enc)
    ids = tok.encode(text)
    print(enc, len(ids), ids)
```

## Pitfalls

Things that will bite you when you extend this code.

**Stale `_mbpe.so`.** Python imports the compiled library, not source. Edit Mojo, re-run Python, nothing changes until you rebuild. The test runner rebuilds from scratch for this reason.

**Wrong pre-tokenizer for the file.** Loads fine, round-trips fine, IDs wrong for the model. Two things must match the vocabulary being served — the `ByteMapping` (matters at fresh-train time, §8) and the split family (matters on every encode, §9). Check both first; round-trip green proves byte handling, not family agreement.

**Specials vanish across save/load.** By contract. Built-ins come back via the trait. Customs need re-registration.

**`encode` vs `encode_ordinary`.** The first sees specials, the second treats them as text. Seven tokens where you expected one `"<|endoftext|>"` is correct behavior for the second.

**Data dir mismatch.** Mojo reads `MBPE_DATA_DIR` then `./data/`. Python uses `importlib.resources` with a dev fallback. Set both or keep repo layout.

**Forgetting `-I .`.** `from bpe.tokenizer import ...` needs it. `main.mojo` doesn't. Copy from `scripts/run_tests.sh`.

## Takeaways

A model never sees text. BPE finds frequent pairs, merges them, repeats; encoding replays in rank order and never counts; splitting comes first and decides the vocabulary. Subwords with a byte fallback give bounded tables, short sequences, no `<UNK>`. One generic engine plus a compile-time pre-tokenizer gives three OpenAI tokenizers and a thirty-line fourth. Speed comes from six accumulated wins — views, arena, lookup cache, incremental counts, two encoders, specialization. Compatibility is verified by byte-for-byte tests on real corpora.

Eleven characters in, two integers out. Now you know why those two integers are what they are — and what it costs to get them right.

**Try it.** Three exercises that close the remaining gaps with your own hands: (1) give the toy a flat rank table — replace the rule scan in `_tokenize` with a string-keyed rank table that repeatedly merges the lowest-rank adjacent pair, and confirm identical IDs on the HF corpus (this is gap 3, built; derive each rule's rank from its insertion order in `self.merges`, which matches its vocab index past the base); (2) measure the tax — encode one Assamese paragraph under `gpt2`, `cl100k`, `o200k` with the §14 snippet and report tokens per character alongside the byte-fallback fraction; (3) write the whitespace pre-tokenizer — a `PreTokenizer` that splits on whitespace runs only, train it to 1000 merges on `benchmarks/corpus.txt` (the Alice text shipped in the repo, ~1 MB), and compare its vocabulary against the character pre-tokenizer's. Each takes a few hours.

## Further reading

Sennrich, Haddow and Birch (2016), *Neural Machine Translation of Rare Words with Subword Units* — the subword paper; short and still the best statement of the size-versus-coverage trade. Gage (1994), *A New Algorithm for Data Compression* — BPE's origin as compression, useful precisely because it was never about language. Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2) — the byte-level vocab plus regex pre-tokenizer as shipped. Karpathy's minbpe ([github](https://github.com/karpathy/minbpe)) and his tokenizer video — the closest pedagogical comparable; Python-first, with a regex variant that matches cl100k, where this chapter takes on the full production-compat burden instead. Raschka, *Build a Large Language Model (From Scratch)*, Chapter 2 — his tokenizer chapter, the spine this chapter mirrors; his book continues through pretraining and instruction tuning without BERT, while ours diverges first at the handoff (we build the `tiktoken` equivalent rather than importing it) and again at the end (we add the encoder side). OpenAI `tiktoken` — the format authority and the baseline every number here is measured against. Warren, *Hacker's Delight* — the bit-manipulation idioms behind `mbpe`'s byte-level matchers in `bpe/pretokenizer.mojo` (the SWAR-style word-at-a-time tests live there, not in this chapter). Unicode UCD — ground truth for the class table in `bpe/unicode_tables.mojo`. Code: `github.com/ratulb/simple_bpe` for the toy, `github.com/ratulb/mbpe` for the engine.

## What's next

Chapter 2 builds attention from scratch — the operation that turns these IDs into context. Chapter 3 assembles a GPT-2 model around it. Chapter 4 pretrains that model on TinyStories. Chapter 5 fine-tunes it for classification. Chapter 6 builds the BERT encoder side — with one caveat carried from this chapter: BERT's canonical tokenizer is WordPiece, not BPE, so Chapter 6 either ports a WordPiece pre-tokenizer onto the same engine or states exactly where the BPE stack stops and WordPiece takes over. All in Mojo: bytes in, IDs out, and everything after operating on IDs. The tokenizer you just built produces Chapter 2's input.
