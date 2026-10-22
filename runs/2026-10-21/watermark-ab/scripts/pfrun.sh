#!/usr/bin/env bash
# pfrun.sh OUTDIR TAG: one pass of the stall test (same steps as runs/2026-10-18/scripts/pf.sh):
# one 8K warm prompt, 8K twice, then 32K, 64K and 128K once each. Random-word prompts (new seed each time): no prefix cache hit.
# 124 chunks of 2,048 tokens per pass. Exit 1 if a step failed.
D=$(dirname "$(readlink -f "$0")")
out=${1:?outdir}; tag=${2:?tag}
KB=$HOME/deploy/tensorfold-v41-20261017/tf-lane/tools/dsv41/kit_bench.py
rc=0
run() { python3 -I -u "$D/v41/src127.py" "$KB" --base http://172.17.0.1:8888 --model DeepSeek-v4.1-Flash-1017-TF "$@" >> "$out/pf-$tag.log" 2>&1 || rc=1; }
run prefill --lengths 8192 --reps 1 --out "$out/pf-$tag-warm.json"
run prefill --lengths 8192 --reps 2 --out "$out/pf-$tag-8k.json"
run prefill --lengths 32768,65536,131072 --reps 1 --out "$out/pf-$tag-long.json"
exit $rc
