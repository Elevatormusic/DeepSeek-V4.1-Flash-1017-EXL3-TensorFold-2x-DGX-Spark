#!/usr/bin/env python3
"""Build the prefill test prompt bodies (no salt line; the harness adds a unique first line).

A  : exactly what tests/v41_suite.py speed_prefill() builds (repeated sentence).
B  : random natural text: unique shuffled sentences (Python language reference + recipe docs).
C  : random token ids from the normal vocabulary, decoded to text.
All bodies for one size get the same local token count as the A body (within 1-2 tokens).
"""
import glob
import json
import random
import re
import sys

from tokenizers import Tokenizer

HERE = sys.path[0]
REC = sys.argv[1]            # path to the recipe repo clone
OUT = sys.argv[2]
tok = Tokenizer.from_file(HERE + "/tokenizer.json")


def ntok(s):
    return len(tok.encode(s, add_special_tokens=False).ids)


FILLER = ("The quick brown fox jumps over the lazy dog while the archivist catalogues another "
          "ledger of tide tables and harbor fees. ")


def suite_prompt(target):                      # copy of speed_prefill()
    return FILLER * max(1, target // 24) + "\nReply with OK."


# ---------------------------------------------------------------- corpus for B
def sentences():
    import pydoc_data.topics as T
    srcs = list(T.topics.values())
    for p in glob.glob(REC + "/**/*.md", recursive=True):
        try:
            srcs.append(open(p, encoding="utf-8").read())
        except Exception:
            pass
    import ast, os, sysconfig
    lib = sysconfig.get_paths()["stdlib"]
    for root, dirs, files in os.walk(lib):
        dirs[:] = [d for d in dirs if d not in ("test", "tests", "site-packages", "__pycache__", "idlelib", "lib2to3", "turtledemo")]
        for f in files:
            if f.endswith(".py"):
                try:
                    tree = ast.parse(open(os.path.join(root, f), encoding="utf-8").read())
                except Exception:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        d = ast.get_docstring(node)
                        if d:
                            srcs.append(d)
    out, seen = [], set()
    for text in srcs:
        for para in re.split(r"\n\s*\n", text):
            if re.search(r"(^|\n) {4,}\S|>>>|::|\|\s|```|\\\\|\{|\}|=|#", para):
                continue
            para = " ".join(para.split())
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", para):
                w = s.split()
                if not (7 <= len(w) <= 55) or not s.endswith("."):
                    continue
                letters = sum(c.isalpha() or c == " " for c in s)
                if letters / len(s) < 0.9 or s in seen:
                    continue
                seen.add(s)
                out.append(s)
    return out


CURSOR = [0]


def build_b(pool, target):
    """Take the next unused sentences of one shuffled pool (so no sentence is used in two prompts)."""
    parts = []
    i = CURSOR[0]
    est = 0
    while est < target * 0.97:
        parts.append(pool[i]); i += 1
        est += len(pool[i - 1]) / 3.7 + 1
    while True:
        text = " ".join(parts)
        n = ntok(text)
        if n >= target:
            break
        parts.append(pool[i]); i += 1
    while ntok(" ".join(parts)) > target:
        parts.pop(); i -= 1
    text = " ".join(parts)
    # fill the last few tokens with words of the next sentence
    extra = []
    for w in pool[i].split():
        if ntok(text + " " + " ".join(extra + [w])) > target:
            break
        extra.append(w)
    if extra:
        text = text + " " + " ".join(extra)
    CURSOR[0] = i + 1
    return text


# ---------------------------------------------------------------- vocab for C
def eligible_ids():
    ids = []
    for i in range(128000):                    # normal vocabulary; 128000 and up are special / added tokens
        s = tok.decode([i], skip_special_tokens=False)
        if s and "�" not in s and all((c.isprintable() or c == " ") for c in s):
            ids.append(i)
    return ids


def build_c(ids, seed, target):
    rng = random.Random(seed)
    big = [rng.choice(ids) for _ in range(int(target * 1.3))]
    k = target
    for _ in range(20):
        text = tok.decode(big[:k], skip_special_tokens=False)
        n = ntok(text)
        if n == target:
            return text
        k += target - n
    # fine steps
    for dk in range(-6, 7):
        text = tok.decode(big[:k + dk], skip_special_tokens=False)
        if ntok(text) == target:
            return text
    return text


a32 = suite_prompt(32000)
a8 = suite_prompt(8000)
n32, n8 = ntok(a32), ntok(a8)
print("A tokens (local, text only): 8K", n8, "32K", n32)

sents = sentences()
print("B sentence pool:", len(sents), "unique sentences,", sum(len(s) for s in sents), "chars")
ids = eligible_ids()
print("C eligible vocabulary pieces:", len(ids), "of 128000")

res = {"A_exact_8k": a8, "A_exact_32k": a32, "local_tokens": {"A8": n8, "A32": n32}, "B32": [], "C32": [], "B8": [], "C8": []}
POOL = sents[:]
random.Random(1234).shuffle(POOL)
res["W8"] = build_b(POOL, n8)
res["W32"] = build_b(POOL, n32)
for r in range(3):
    res["B32"].append(build_b(POOL, n32))
    res["C32"].append(build_c(ids, 200 + r, n32))
    res["B8"].append(build_b(POOL, n8))
    res["C8"].append(build_c(ids, 400 + r, n8))
res["meta"] = {"b_pool_sentences": len(sents), "c_vocab_pieces": len(ids)}
for k in ("B32", "C32", "B8", "C8"):
    print(k, [ntok(x) for x in res[k]])
print("W8", ntok(res["W8"]), "W32", ntok(res["W32"]))
# overlap check for B: no sentence appears twice inside one prompt, and sets differ between replicates
for k in ("B32",):
    sets = [set(re.split(r"(?<=[.!?])\s+", x)) for x in res[k]]
    print("B replicate overlap (sentences):", [len(sets[i] & sets[j]) for i in range(3) for j in range(i + 1, 3)])
json.dump(res, open(OUT, "w"), ensure_ascii=False)
print("wrote", OUT)
