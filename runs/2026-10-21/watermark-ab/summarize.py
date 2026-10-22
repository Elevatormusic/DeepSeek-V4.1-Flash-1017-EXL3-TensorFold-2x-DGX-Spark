#!/usr/bin/env python3
"""Make the summary table of the watermark A/B test from the files of each run.

  python3 -I summarize.py > summary-table.txt

Each <run>/summary.json comes from scripts/analyze.py (rank-0 chunk times, 1 s memory samples, /proc/vmstat deltas).
MemAvailable is in MiB. A slow chunk is a prefill chunk over 1.3 s. Standard library only.
The runs B3 and V1c ran after midnight UTC, so they are in ../../2026-10-22/watermark-ab/.

The formula check uses the identity of the kernel function si_mem_available():
  MemAvailable = MemFree - totalreserve + X
  X = (page cache - min(page cache / 2, sum of low watermarks)) + (reclaimable - min(reclaimable / 2, sum of low watermarks))
Per sample, X = MemAvailable - MemFree + totalreserve. A lower low watermark lowers a cut only when the cut is
capped by it. If X does not change between two settings, no cut is capped, and the gain is the drop of totalreserve.
"""
import csv
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
LATER = os.path.join(HERE, "..", "..", "2026-10-22", "watermark-ab")   # third window, after midnight UTC
RUNS = ["B", "B2", "B3", "V1", "V1r", "V1c", "V2", "V2r", "V3"]
GROUPS = [("B (B, B2)", ["B", "B2"]), ("V1 (V1, V1r)", ["V1", "V1r"]), ("V2 (V2r only for the long run)", ["V2", "V2r"]), ("V3", ["V3"])]
PARTIAL = ["B3", "V1c"]   # third window: stopped early by the head memory rule, no long run


def rundir(name):
    return os.path.join(LATER if name in PARTIAL else HERE, name)


def load(name):
    with open(os.path.join(rundir(name), "summary.json")) as f:
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
print("Per setting (mean of the complete runs; gain against the mean of B and B2)")
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


def zone_sums(name):
    text = open(os.path.join(rundir(name), "zone-head.txt")).read()
    return {k: int(v) for k, v in re.findall(r"(\w+)_MiB=(\d+)", text)}


def idle_x(name):
    """Mean of X = MemAvailable - MemFree + totalreserve over the idle window of the head (MiB)."""
    reserve = zone_sums(name)["totalreserve"]
    win = None
    with open(os.path.join(rundir(name), "windows.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] == "idle":
                win = (float(p[1]), float(p[2]))
    xs, avail, free, cached = [], [], [], []
    with open(os.path.join(rundir(name), "mem-head.csv")) as f:
        for r in csv.DictReader(f):
            if win[0] <= float(r["epoch"]) <= win[1]:
                a, fr = float(r["MemAvailable"]) / 1024, float(r["MemFree"]) / 1024
                xs.append(a - fr + reserve)
                avail.append(a)
                free.append(fr)
                cached.append(float(r["Cached"]) / 1024)
    return reserve, statistics.mean(xs), statistics.mean(avail), statistics.mean(free), statistics.mean(cached)


print()
print("Formula check at the engine-loaded idle state of the head (MiB): X = MemAvailable - MemFree + totalreserve")
rows = {n: idle_x(n) for n in ("B", "B2", "V1", "V1r", "V2r", "V3", "B3", "V1c")}
for n, (res, x, a, fr, ca) in rows.items():
    print(f"{n}: totalreserve {res}, MemAvailable {a:.0f}, MemFree {fr:.0f}, Cached {ca:.0f}, X {x:.0f}")
res_b = statistics.mean(rows[n][0] for n in ("B", "B2"))
x_b = statistics.mean(rows[n][1] for n in ("B", "B2"))
print(f"B mean: totalreserve {res_b:.0f}, X {x_b:.0f}")
print("Gain at the same MemFree = drop of totalreserve + change of X (the cuts):")
for label, names in (("V1", ("V1", "V1r")), ("V2", ("V2r",)), ("V3", ("V3",))):
    res = statistics.mean(rows[n][0] for n in names)
    x = statistics.mean(rows[n][1] for n in names)
    drop = res_b - res
    print(f"{label}: totalreserve drop {drop:.0f}, change of X {x - x_b:+.0f}, gain {drop + x - x_b:.0f}")
print("Full-cut bound (both low-watermark cuts capped at their sum, a page cache of 3.2 GiB or more): "
      "V1 +934, V2 +1,024, V3 +1,958")
