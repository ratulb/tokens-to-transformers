"""Chapter 1 validation, Mojo side.

Run from the repo root with `pixi run validate`.

Encodes the Raschka sentence and the-verdict.txt with mbpe (gpt2, cl100k,
o200k) and writes the full ID lists to validate/ch01/out/. The Python half
(validate/ch01/validate_tiktoken.py) diffs them against tiktoken.
"""
from bpe.tokenizer import Tokenizers


def dump(path: String, ids: List[Int]) raises:
    var out = String()
    for id in ids:
        out += String(id) + "\n"
    with open(path, "w") as f:
        f.write(out)


def main() raises:
    var verdict_path = "assets/data/the-verdict.txt"
    var out_dir = "validate/ch01/out/"
    var raschka = (
        "Hello, do you like tea? <|endoftext|> "
        "In the sunlit terraces of someunknownPlace."
    )

    var corpus = ""
    with open(verdict_path, "r") as f:
        corpus = f.read()

    var gpt2 = Tokenizers.get[Tokenizers.gpt2]()
    var cl100k = Tokenizers.get[Tokenizers.cl100k]()
    var o200k = Tokenizers.get[Tokenizers.o200k]()

    dump(out_dir + "mbpe_raschka_gpt2.txt", gpt2.encode(raschka))
    dump(out_dir + "mbpe_raschka_cl100k.txt", cl100k.encode(raschka))
    dump(out_dir + "mbpe_raschka_o200k.txt", o200k.encode(raschka))
    dump(out_dir + "mbpe_verdict_gpt2.txt", gpt2.encode(corpus))
    dump(out_dir + "mbpe_verdict_cl100k.txt", cl100k.encode(corpus))
    dump(out_dir + "mbpe_verdict_o200k.txt", o200k.encode(corpus))

    print("raschka gpt2:", gpt2.encode(raschka))
    print("verdict gpt2 tokens:", len(gpt2.encode(corpus)))
    print("verdict cl100k tokens:", len(cl100k.encode(corpus)))
    print("verdict o200k tokens:", len(o200k.encode(corpus)))
