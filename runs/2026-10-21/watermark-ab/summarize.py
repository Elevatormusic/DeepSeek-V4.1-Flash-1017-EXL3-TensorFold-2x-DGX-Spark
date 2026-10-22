#!/usr/bin/env python3
"""Make the summary table of the watermark A/B test from the summary.json file of each run.

  python3 -I summarize.py > summary-table.txt

Each <run>/summary.json comes from scripts/analyze.py (rank-0 chunk times, 1 s memory samples, /proc/vmstat deltas).
MemAvailable is in MiB. A slow chunk is a prefill chunk over 1.3 s. Standard library only.
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ["B", "B2", "B3", "V1", "V1r", "V1c", "V2", "V2r", "V3"]
GROUPS = [("B (B, B2)", ["B", "B2"]), ("V1 (V1, V1r)", ["V1", "V1r"]), ("V2 (V2r only for the long run)", ["V2", "V2r"]), ("V3", ["V3"])]
PARTIAL = ["B3", "V1c"]   # third window: stopped early by the head memory rule, no long run


def load(name):
    with open(os.path.join(HERE, name, "summary.json")) as f:
        return json.load(f)


def get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def med(values):
    return round(statistics.median(values)) if values else None


data = {n: load(n) for n in RUNS}
print("Per run (the stall-test low of B3 and V1c is the low of the whole partial run; MiB for MemAvailable; head compaction = compact_stall/compact_fail/allocstall)")
print("run | idle h/w | stall-test low h | long-run low h/w | chunks n, median, p95, max, slow | long chunk max, slow | tok/s 8K/32K/64K/128K | head compaction stall test | head compaction long | errors")
for n in RUNS:
    s = data[n]
    st = s["stall_test"]
    ch = get(st, "both", "rank0", default=None) or get(st, "pf1", "rank0", default={})
    lg = s.get("long")
    tps = s["prefill_tps"]
    t = "/".join(str(med(tps.get(k, []))) for k in ("8192", "32768", "65536", "131072"))
    vh = st.get("vm_head", {}) or {}
    lh = lg["vm_head"] if lg else {}
    err = "stopped by the head memory rule" if s.get("trip") else f"{lg['errors']} errors, {lg['elapsed_s']} s" if lg else "n/a"
    print(f"{n} | {s['idle_loaded']['head']['median_avail_mib']}/{s['idle_loaded']['worker']['median_avail_mib']}"
          f" | {get(st, 'head_mem', 'min_avail_mib') or get(s, 'all', 'head_mem', 'min_avail_mib')}"
          f" | {get(lg, 'head_mem', 'min_avail_mib')}/{get(lg, 'worker_mem', 'min_avail_mib')}"
          f" | {ch.get('n')}, {ch.get('median')}, {ch.get('p95')}, {ch.get('max')}, {ch.get('slow13')}"
          f" | {get(lg, 'chunks_rank0', 'max')}, {get(lg, 'chunks_rank0', 'slow13')}"
          f" | {t} | {vh.get('compact_stall')}/{vh.get('compact_fail')}/{vh.get('allocstall')}"
          f" | {lh.get('compact_stall')}/{lh.get('compact_fail')}/{lh.get('allocstall')} | {err}")

print()
print("Per setting (mean of the runs; gain against the mean of B and B2)")
base_idle = statistics.mean(data[n]["idle_loaded"]["head"]["median_avail_mib"] for n in ("B", "B2"))
base_low = statistics.mean(data[n]["long"]["head_mem"]["min_avail_mib"] for n in ("B", "B2"))
for label, names in GROUPS:
    idle = statistics.mean(data[n]["idle_loaded"]["head"]["median_avail_mib"] for n in names)
    lows = [data[n]["long"]["head_mem"]["min_avail_mib"] for n in names if data[n].get("long")]
    low = statistics.mean(lows) if lows else None
    slow = [get(data[n], "stall_test", "both", "rank0", "slow13") for n in names]
    chunks = [get(data[n], "stall_test", "both", "rank0", "n") for n in names]
    mx = max(get(data[n], "stall_test", "both", "rank0", "max") for n in names)
    print(f"{label}: idle head {idle:.0f} ({idle - base_idle:+.0f}); lowest head in long run "
          f"{'n/a' if low is None else f'{low:.0f} ({low - base_low:+.0f})'}; slow chunks {slow} of {chunks}; max chunk {mx} s")

print()
print("Third window (B3 and V1c): the head memory rule (1.25 GiB) stopped both runs early. They are not in the means above.")
for n in PARTIAL:
    s = data[n]
    st = s["stall_test"]
    ch = get(st, "both", "rank0", default=None) or get(st, "pf1", "rank0", default={})
    print(f"{n}: idle head {s['idle_loaded']['head']['median_avail_mib']}; lowest head {get(s, 'all', 'head_mem', 'min_avail_mib')}; "
          f"chunks {ch.get('n')}, slow {ch.get('slow13')}, max {ch.get('max')} s; stop: {s['trip']}")
