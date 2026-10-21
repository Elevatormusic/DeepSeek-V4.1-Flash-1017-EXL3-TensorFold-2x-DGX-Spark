#!/usr/bin/env python3
"""Computes what the kernel hides from MemAvailable, from the raw reads in this folder (head-raw.txt, worker-raw.txt).

The method is the kernel's si_mem_available() (mm/show_mem.c):
  available = MemFree - totalreserve + (page cache - min(page cache / 2, sum of low)) + (reclaimable slab - min(slab / 2, sum of low))
  totalreserve = for each zone: min(max(lowmem_reserve) + high watermark, managed pages)
  page cache = Active(file) + Inactive(file), sum of low = the low watermarks of all zones
The result is checked against the MemAvailable line of the same read. Run: python3 -I compute_reserve.py
"""
import re
import sys
from pathlib import Path

GIB = 1024 ** 3


def parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    page = int(re.search(r"^pagesize (\d+)", text, re.M).group(1))
    meminfo = {k: int(v) * 1024 for k, v in re.findall(r"^(\w+(?:\(\w+\))?):\s+(\d+) kB", text, re.M)}
    zones, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"Node \d+, zone\s+(\w+)", line)
        if m:
            cur = zones.setdefault(m.group(1), {})
            continue
        m = re.match(r"\s+(min|low|high|managed|present)\s+(\d+)", line)
        if m and cur is not None:
            cur[m.group(1)] = int(m.group(2))
        m = re.match(r"\s+protection: \(([\d, ]+)\)", line)
        if m and cur is not None:
            cur["protection"] = [int(x) for x in m.group(1).split(",")]
    sysctl = dict(re.findall(r"^vm\.(\w+) (\d+)$", text, re.M))
    return {"page": page, "meminfo": meminfo, "zones": zones, "sysctl": sysctl}


def compute(d: dict) -> dict:
    page, mi, zones = d["page"], d["meminfo"], d["zones"]
    reserve = 0
    for z in zones.values():
        managed = z.get("managed", 0)
        top = min(max(z.get("protection", [0])) + z["high"], managed)
        reserve += top
    low = sum(z["low"] for z in zones.values())
    high = sum(z["high"] for z in zones.values())
    cache = (mi["Active(file)"] + mi["Inactive(file)"]) // page
    slab = mi["SReclaimable"] // page
    free = mi["MemFree"] // page
    avail = free - reserve + (cache - min(cache // 2, low)) + (slab - min(slab // 2, low))
    return {"reserve": reserve * page, "low": low * page, "high": high * page, "calc": max(avail, 0) * page,
            "cache_cut": min(cache // 2, low) * page, "slab_cut": min(slab // 2, low) * page,
            "real": mi["MemAvailable"], "lowmem": max(zones["DMA"]["protection"]) * page}


def default_estimate(d: dict) -> dict:
    """The same totals for the kernel defaults (min_free_kbytes 67584, watermark_scale_factor 10), by the kernel formula.
    An estimate: nothing was changed to measure it."""
    page, z = d["page"], d["zones"]["Normal"]
    managed = z["managed"]
    min_n = 67584 * 1024 // page * managed // sum(x.get("managed", 0) for x in d["zones"].values())
    gap = max(min_n // 4, managed * 10 // 10000)
    low_n, high_n = min_n + gap, min_n + 2 * gap
    lowmem = max(d["zones"]["DMA"]["protection"])
    return {"reserve": (high_n + lowmem + d["zones"]["DMA"]["high"]) * page, "low": low_n * page}


def boost_pages(d: dict) -> int:
    """Watermark boost of the Normal zone: its high watermark minus the high watermark that the two sysctl values give.
    The kernel adds a boost after a fragmentation event, and /proc/zoneinfo prints the boosted values."""
    page, zones = d["page"], d["zones"]
    z = zones["Normal"]
    managed_all = sum(x.get("managed", 0) for x in zones.values())
    min_n = int(d["sysctl"]["min_free_kbytes"]) * 1024 // page * z["managed"] // managed_all
    gap = max(min_n // 4, z["managed"] * int(d["sysctl"]["watermark_scale_factor"]) // 10000)
    return z["high"] - (min_n + 2 * gap)


def main() -> int:
    here = Path(__file__).resolve().parent
    for node in ("head", "worker"):
        d = parse(here / f"{node}-raw.txt")
        r = compute(d)
        e = default_estimate(d)
        boost = boost_pages(d)
        print(f"[{node}]")
        print(f"  min_free_kbytes {d['sysctl']['min_free_kbytes']}, watermark_scale_factor {d['sysctl']['watermark_scale_factor']}")
        print(f"  sum of zone low watermarks:  {r['low'] / GIB:.2f} GiB")
        print(f"  sum of zone high watermarks: {r['high'] / GIB:.2f} GiB")
        print(f"  lowmem reserve of zone DMA against Normal: {r['lowmem'] / GIB:.2f} GiB")
        print(f"  totalreserve (subtracted from MemFree):    {r['reserve'] / GIB:.2f} GiB")
        print(f"  page cache cut (min(cache/2, sum of low)): {r['low'] / GIB:.2f} GiB when the page cache is over {2 * r['low'] / GIB:.2f} GiB")
        print(f"  reclaimable slab cut (min(slab/2, low)):   {r['slab_cut'] / GIB:.2f} GiB")
        print(f"  hidden in all:                             {(r['reserve'] + r['cache_cut'] + r['slab_cut']) / GIB:.2f} GiB")
        print(f"  MemAvailable by the formula {r['calc'] // 1024} kB, in /proc/meminfo {r['real'] // 1024} kB, "
              f"difference {(r['calc'] - r['real']) // 1024} kB")
        print(f"  watermark boost in the Normal zone (approx.): {boost * d['page'] / GIB:.2f} GiB")
        print(f"  kernel defaults, by the formula (estimate): totalreserve {e['reserve'] / GIB:.2f} GiB, low sum {e['low'] / GIB:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
