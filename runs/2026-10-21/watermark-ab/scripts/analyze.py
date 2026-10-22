#!/usr/bin/env python3
"""Analysis for the watermark A/B test. Standard library only.

  analyze.py quick RANK0_LOG START END     one line: chunk count, slow chunks (> 1.3 s), max, median in the window
  analyze.py setting SETDIR                 summary.json and a text summary of one setting
  analyze.py report LOGDIR NAME...          one text block per setting (reads each SETDIR/summary.json)
"""
import csv
import datetime
import json
import os
import re
import statistics
import sys

CHUNK = re.compile(r"^(\S+)\s.*rank (\d) chunk at (\d+): ([\d.]+)s")
SLOW = 1.3


def ts_epoch(s):
    m = re.match(r"(\d{4}-\d\d-\d\d)T(\d\d:\d\d:\d\d)(\.\d+)?Z", s)
    if not m:
        return None
    dt = datetime.datetime.strptime(m.group(1) + " " + m.group(2), "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp() + (float(m.group(3)) if m.group(3) else 0.0)


def chunks(path, rank):
    out = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = CHUNK.match(line)
                if m and int(m.group(2)) == rank:
                    out.append((ts_epoch(m.group(1)), int(m.group(3)), float(m.group(4))))
    except OSError:
        pass
    return out


def cstats(rows):
    v = sorted(s for _, _, s in rows)
    if not v:
        return {"n": 0}
    return {"n": len(v), "median": round(statistics.median(v), 3), "mean": round(sum(v) / len(v), 3),
            "p95": round(v[min(len(v) - 1, int(0.95 * len(v)))], 3), "max": round(v[-1], 3),
            "slow13": sum(1 for x in v if x > SLOW), "slow20": sum(1 for x in v if x > 2.0),
            "slow_sec": round(sum(x for x in v if x > SLOW), 2)}


def window(rows, t0, t1):
    return [r for r in rows if r[0] is not None and t0 <= r[0] <= t1]


def load_windows(setdir):
    w = {}
    try:
        for line in open(os.path.join(setdir, "windows.tsv")):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                w[p[0]] = (float(p[1]), float(p[2]), p[3] if len(p) > 3 else "")
    except OSError:
        pass
    return w


def load_csv(path):
    rows = []
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({k: float(v) for k, v in r.items() if v not in ("", None)})
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def inwin(rows, t0, t1):
    return [r for r in rows if t0 <= r["epoch"] <= t1]


def mem_stats(rows, t0, t1):
    w = inwin(rows, t0, t1)
    if not w:
        return {}
    lo = min(w, key=lambda r: r["MemAvailable"])
    lf = min(w, key=lambda r: r["MemFree"])
    return {"min_avail_mib": round(lo["MemAvailable"] / 1024), "min_avail_at": datetime.datetime.fromtimestamp(lo["epoch"], datetime.timezone.utc).strftime("%H:%M:%S"),
            "min_free_mib": round(lf["MemFree"] / 1024), "median_avail_mib": round(statistics.median(r["MemAvailable"] for r in w) / 1024),
            "samples": len(w)}


COUNTERS = ["compact_stall", "compact_fail", "compact_success", "compact_daemon_wake", "allocstall", "pgscan_direct", "pgsteal_direct",
            "pgscan_kswapd", "pgsteal_kswapd", "pageoutrun", "pswpin", "pswpout", "pgmajfault", "pgmigrate_success",
            "compact_migrate_scanned", "compact_free_scanned"]


def deltas(rows, t0, t1):
    w = inwin(rows, t0, t1)
    if len(w) < 2:
        return {}
    a, b = w[0], w[-1]
    d = {k: int(b.get(k, 0) - a.get(k, 0)) for k in COUNTERS}
    d["kswapd_cpu_s"] = round((b.get("kswapd_ticks", 0) - a.get("kswapd_ticks", 0)) / 100.0, 2)
    d["kcompactd_cpu_s"] = round((b.get("kcompactd_ticks", 0) - a.get("kcompactd_ticks", 0)) / 100.0, 2)
    d["seconds"] = round(b["epoch"] - a["epoch"])
    return d


def read(path):
    try:
        return open(path, errors="replace").read()
    except OSError:
        return ""


def prefill_tps(setdir):
    by = {}
    for p in (1, 2):
        for part in ("warm", "8k", "long"):
            f = os.path.join(setdir, f"pf-p{p}-{part}.json")
            try:
                for r in json.load(open(f))["result"]:
                    by.setdefault(int(r["length"]), []).append(round(r["prefill_tps"]))
            except (OSError, ValueError, KeyError):
                pass
    return by


def cmd_quick(path, t0, t1):
    st = cstats(window(chunks(path, 0), float(t0), float(t1)))
    if not st.get("n"):
        print("n=0 slow=0 max=0 median=0")
    else:
        print(f"n={st['n']} slow={st['slow13']} max={st['max']} median={st['median']}")


def cmd_setting(setdir):
    w = load_windows(setdir)
    head = load_csv(os.path.join(setdir, "mem-head.csv"))
    work = load_csv(os.path.join(setdir, "mem-worker.csv"))
    r0 = chunks(os.path.join(setdir, "rank-r0.log"), 0)
    r1 = chunks(os.path.join(setdir, "rank-r1.log"), 1)
    s = {"name": os.path.basename(setdir.rstrip("/")), "windows": {k: [v[0], v[1], v[2]] for k, v in w.items()}}
    s["zone_head"] = read(os.path.join(setdir, "zone-head.txt")).strip()
    s["zone_worker"] = read(os.path.join(setdir, "zone-worker.txt")).strip()
    if "idle0" in w:
        s["idle_unloaded"] = {"head": mem_stats(head, *w["idle0"][:2]), "worker": mem_stats(work, *w["idle0"][:2])}
    if "idle" in w:
        s["idle_loaded"] = {"head": mem_stats(head, *w["idle"][:2]), "worker": mem_stats(work, *w["idle"][:2])}
    pf = {}
    for p in ("pf1", "pf2"):
        if p in w:
            pf[p] = {"rank0": cstats(window(r0, *w[p][:2])), "rank1": cstats(window(r1, *w[p][:2])), "rc": w[p][2]}
    if "pf1" in w and "pf2" in w:
        t0, t1 = w["pf1"][0], w["pf2"][1]
        both0 = window(r0, w["pf1"][0], w["pf1"][1]) + window(r0, w["pf2"][0], w["pf2"][1])
        both1 = window(r1, w["pf1"][0], w["pf1"][1]) + window(r1, w["pf2"][0], w["pf2"][1])
        pf["both"] = {"rank0": cstats(both0), "rank1": cstats(both1)}
        pf["head_mem"] = mem_stats(head, t0, t1)
        pf["worker_mem"] = mem_stats(work, t0, t1)
        pf["vm_head"] = deltas(head, t0, t1)
        pf["vm_worker"] = deltas(work, t0, t1)
    s["stall_test"] = pf
    s["prefill_tps"] = prefill_tps(setdir)
    if "long" in w:
        t0, t1 = w["long"][:2]
        lg = read(os.path.join(setdir, "long4.log"))
        m = re.search(r"elapsed (\d+)s errors=(\d+)", lg)
        s["long"] = {"head_mem": mem_stats(head, t0, t1), "worker_mem": mem_stats(work, t0, t1),
                     "vm_head": deltas(head, t0, t1), "vm_worker": deltas(work, t0, t1),
                     "chunks_rank0": cstats(window(r0, t0, t1)), "chunks_rank1": cstats(window(r1, t0, t1)),
                     "elapsed_s": int(m.group(1)) if m else None, "errors": int(m.group(2)) if m else None,
                     "rc": w["long"][2], "streams": [l.strip() for l in lg.splitlines() if l.strip().startswith("long ")]}
    if "all" in w:
        t0, t1 = w["all"][:2]
        s["all"] = {"head_mem": mem_stats(head, t0, t1), "worker_mem": mem_stats(work, t0, t1),
                    "vm_head": deltas(head, t0, t1), "vm_worker": deltas(work, t0, t1)}
    errs = {}
    for tag in ("rank-r0.log", "rank-r1.log"):
        t = read(os.path.join(setdir, tag))
        errs[tag] = {"WARNING": len(re.findall(r"WARNING", t)), "Traceback": len(re.findall(r"Traceback", t)),
                     "OutOfMemory": len(re.findall(r"OutOfMemory|out of memory", t)),
                     "admission_waits": len(re.findall(r"admission waits", t)), "late_capture": len(re.findall(r"captured while serving", t))}
    s["rank_log_counts"] = errs
    s["earlyoom_head"] = read(os.path.join(setdir, "earlyoom-head.txt")).strip()
    s["earlyoom_worker"] = read(os.path.join(setdir, "earlyoom-worker.txt")).strip()
    s["trip"] = read(os.path.join(setdir, "trip.txt")).strip()
    s["notes"] = read(os.path.join(setdir, "notes.txt")).strip()
    s["pf_errors"] = len(re.findall(r"Traceback|Error|error", read(os.path.join(setdir, "pf-p1.log")) + read(os.path.join(setdir, "pf-p2.log"))))
    json.dump(s, open(os.path.join(setdir, "summary.json"), "w"), indent=1)
    print(json.dumps(s, indent=1)[:6000])


def cmd_report(logdir, names):
    for n in names:
        p = os.path.join(logdir, n, "summary.json")
        if not os.path.exists(p):
            print(f"== {n}: no summary")
            continue
        s = json.load(open(p))
        print(f"== {n}")
        print(json.dumps({k: v for k, v in s.items() if k not in ("windows",)}, indent=1))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "quick":
        cmd_quick(*sys.argv[2:5])
    elif c == "setting":
        cmd_setting(sys.argv[2])
    elif c == "report":
        cmd_report(sys.argv[2], sys.argv[3:])
