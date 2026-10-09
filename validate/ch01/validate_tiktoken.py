"""Chapter 1 validation, tiktoken side. Run from the repo root with:
    pixi run validate
(which runs the Mojo half first, then this file).

Re-encodes the same inputs with the reference tiktoken library, diffs the
full ID lists against validate/ch01/out/mbpe_*.txt, and prints the token /
mismatch tables that appear in Chapter 1, section 1.8. Exits nonzero on any
mismatch.
"""
import sys

import tiktoken

VERDICT = "assets/data/the-verdict.txt"
OUT = "validate/ch01/out/"
RASCHKA = (
    "Hello, do you like tea? <|endoftext|> "
    "In the sunlit terraces of someunknownPlace."
)
CELLS = ["don't", "hello  world", "WÉÉ", "a。b"]
PAIRS = [("gpt2", "gpt2"), ("cl100k", "cl100k_base"), ("o200k", "o200k_base")]


def read_ids(path):
    with open(path) as f:
        return [int(line) for line in f]


def main():
    failures = 0
    with open(VERDICT, encoding="utf-8") as f:
        verdict = f.read()

    print(f"{'vocabulary':<10} {'tokens':>7} {'mismatches':>10}")
    for mbpe_name, tk_name in PAIRS:
        enc = tiktoken.get_encoding(tk_name)
        kw = {"allowed_special": {"<|endoftext|>"}}
        for label, text in [("raschka", RASCHKA), ("verdict", verdict)]:
            mine = enc.encode(text, **kw)
            theirs = read_ids(f"{OUT}mbpe_{label}_{mbpe_name}.txt")
            bad = sum(1 for a, b in zip(mine, theirs) if a != b)
            bad += abs(len(mine) - len(theirs))
            if label == "verdict":
                print(f"{tk_name:<10} {len(mine):>7} {bad:>10}")
            if bad:
                failures += 1
                print(f"MISMATCH {tk_name} {label}: {bad} differing ids")

    print("\nsplit cells (tiktoken pieces):")
    for mbpe_name, tk_name in PAIRS:
        enc = tiktoken.get_encoding(tk_name)
        for cell in CELLS:
            ids = enc.encode(cell)
            pieces = [enc.decode([i]) for i in ids]
            print(f"{tk_name:<12} {cell!r:<16} {ids} {pieces}")

    if failures:
        sys.exit(1)
    print("\nvalidation OK: mbpe and tiktoken agree ID-for-ID")


if __name__ == "__main__":
    main()
