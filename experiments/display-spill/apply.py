#!/usr/bin/env python3
"""Install display-safe SSD spill on top of the display-KV source transforms.

This feature depends on the display-KV transform. A composer must validate the
union of pristine hashes once, apply display first, then apply this transform.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import subprocess

REVISION = "07aa02585f64cfb04c60a96de6554c0d4708ed46"
FAMILY = Path("src/tensorfold/families/deepseek_v41/cuda")
ENGINE = FAMILY / "engine.py"
MULTI = FAMILY / "multi.py"
APP = FAMILY / "app.py"
HTTP = Path("src/tensorfold/cuda/http.py")
FILES = {
    ENGINE: "019f616166d3ed72f117461e4978df37d36c85c28cef9c7f42a51b5a3824fee5",
    MULTI: "0e19663c2b2f994ef7c160d21fa444624af012c3f756a9265fae5a082c24213d",
    APP: "426b9463b5a254d96895a8c38f2f7cb5cc6c949bc8b607326e09af3a28d1de33",
    HTTP: "bf8cb5e428c372358ff1e8676f3f92a6f354f1dc35999adfd8487cf6ff1aa1cb",
}
HERE = Path(__file__).resolve().parent


def _replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{name}: expected one unmodified anchor, found {count}")
    return text.replace(old, new)


def transform(relative: str | Path, text: str) -> str:
    """Transform one file after the display transform has been applied."""
    relative = Path(relative)
    if relative not in FILES:
        raise ValueError(f"unsupported source file: {relative}")
    name = relative.name
    if "SpillStore" in text or "_flush_spill" in text:
        raise ValueError(f"{relative}: spill tier appears to be installed already")
    if name == "engine.py":
        # The display-only transform refuses a nonzero TF_DS_SPILL_GIB. This image has the spill tier, so that guard goes.
        return _replace_once(text,
            '            if float(os.environ.get("TF_DS_SPILL_GIB") or 0) != 0:\n'
            '                raise ValueError("display-only KV requires TF_DS_SPILL_GIB=0")\n', '', name)
    if name == "multi.py":
        text = _replace_once(text, '        self.m, self.cap = m, int(cap)\n        self.cap = (display_kv.pool_capacity(m.cfg, engine.display_bytes, max_rows=0)\n                    if engine.display_bytes else int(cap))\n        self.pool = m.new_pool(slots, self.cap, display_bytes=engine.display_bytes)\n', '        self.m, self.cap = m, int(cap)\n        self.spill = None\n        self.spill_jobs: list = []\n        self.spill_nonce = os.urandom(4).hex() if engine.rank == 0 else None\n        spill_cfg, spill_problem, spill_wanted = None, None, False\n        if KEEP and float(os.environ.get("TF_DS_SPILL_GIB") or 0) > 0:\n            spill_wanted = True\n            try:                                         # a failure here is shared with the other rank below\n                from tensorfold.cuda.spill import SpillConfig, preallocate\n\n                spill_cfg = SpillConfig(\n                    root=os.path.expanduser(os.environ.get("TF_DS_SPILL_DIR") or "~/.cache/tensorfold"),\n                    gib=float(os.environ.get("TF_DS_SPILL_GIB")),\n                    min_tokens=int(os.environ.get("TF_DS_SPILL_MIN") or 4096),\n                    min_free_gib=float(os.environ.get("TF_DS_SPILL_MIN_FREE_GIB") or 50.0),\n                    keep_builds=int(os.environ.get("TF_DS_SPILL_KEEP_BUILDS") or 1),\n                    writers=int(os.environ.get("TF_SPILL_WRITERS") or 2),\n                    direct=os.environ.get("TF_SPILL_DIRECT", "1") != "0",\n                    flush_s=float(os.environ.get("TF_SPILL_FLUSH_S") or 60),\n                )\n                preallocate(spill_cfg)\n            except Exception as exc:                     # noqa: BLE001\n                spill_cfg, spill_problem = None, exc\n        self.cap = (display_kv.pool_capacity(m.cfg, engine.display_bytes, max_rows=0)\n                    if engine.display_bytes else int(cap))\n        self.pool = m.new_pool(slots, self.cap, display_bytes=engine.display_bytes)\n', name)
        text = _replace_once(text, '        self.spill = None                                # TF_DS_SPILL_GIB: kept prompts on this rank\'s disk\n        self.spill_nonce = os.urandom(4).hex() if engine.rank == 0 else None   # names this run\'s spilled prompts\n        if KEEP:\n            from . import spill as sp\n\n            if sp.SPILL_GIB > 0:\n                try:\n                    ratios = {i: m.cfg.compress_ratios[i] for i in set(self.pool.comp) | set(self.pool.index_k)}\n                    ident = sp.engine_id(getattr(engine, "model_dir", "."), world=engine.world, pool=self.pool,\n                                         dpool=self.dpool)\n                    self.spill = sp.Spill(sp.SPILL_DIR, engine=ident, rank=engine.rank, pool=self.pool, ratios=ratios,\n                                          budget=int(sp.SPILL_GIB * 2**30), max_age=sp.SPILL_MAX_AGE_H * 3600.0)\n                    self.spill.recycle()\n                    if engine.rank == 0:\n                        st = self.spill.stats()\n                        print(f"[tensorfold] spilled kept prompts: {self.spill.dir} ({st[\'manifests\']} on disk, "\n                              f"{st[\'bytes\'] / 2**30:.2f} of {sp.SPILL_GIB:g} GiB)", flush=True)\n                except Exception as exc:                 # noqa: BLE001  (no disk: kept prompts stay in the window only)\n                    print(f"[tensorfold] rank {engine.rank}: spilled kept prompts off ({type(exc).__name__}: {exc})",\n                          flush=True)\n                    self.spill = None\n', '        if spill_wanted:\n            # Every rank must reach the same collectives. A failure on one rank (a disk, pinned memory or a source\n            # read) is shared first, so that no rank waits in the gather of SpillStore for a rank that skipped it.\n            problem, parts = spill_problem, None\n            try:\n                from tensorfold.cuda.spill import SpillStore, build_id, release_pinned, weights_id\n\n                if spill_cfg is None:\n                    raise RuntimeError("the spill settings or buffers failed on this rank")\n                planes = []\n                for kind, planes_dict in (("comp", self.pool.comp), ("index_k", self.pool.index_k)):\n                    for i, x in sorted(planes_dict.items()):\n                        for j, t in enumerate(x if isinstance(x, tuple) else (x,)):\n                            planes.append((kind, i, j, tuple(t.shape[1:]), str(t.dtype)))\n                build = build_id(getattr(engine, "model_dir", "."))\n                identity = ("TF_DS_KV", "TF_DS_REPLAY", "TF_DS_REPLAY_FLOOR", "TF_DS_PREFILL_CHUNK", "TF_DS_KERNELS",\n                            "TF_DS_ENGRAM", "TF_DS_VISION_EXTRA")\n                signature = "|".join([\n                    "dsv41-multi", build,\n                    os.environ.get("TF_DS_KV", "native"),\n                    f"world{engine.world}",\n                    f"display{getattr(engine, \'display_bytes\', 0)}",\n                    f"replay{int(self._replay())}",\n                    repr(planes),\n                    f"tokens{self.pool.tokens.dtype}ring{self.pool.ring_size}x{len(self.pool.ring)}",\n                    repr([tuple(t.shape[1:]) + (str(t.dtype),) for t in self.pool.ring[:1]]),\n                    repr([(i, tuple(t.shape[1:]), str(t.dtype))\n                          for i, pair in sorted(self.pool.comp_raw.items()) for t in pair]),\n                    repr([(tuple(t.shape), str(t.dtype)) for t in self.dpool.views[0].rings]\n                         if self.dpool is not None else None),\n                    repr([(k, os.environ.get(k)) for k in identity]),\n                ])\n                spanned = []\n                if getattr(self.pool, "display_arena", None) is not None:\n                    for kind, planes_dict in (("comp", self.pool.comp), ("index_k", self.pool.index_k)):\n                        for i, x in planes_dict.items():\n                            for j in range(len(x) if isinstance(x, tuple) else 1):\n                                spanned.append(f"{kind}.{i}.{j}")\n                engram = os.environ.get("TF_DS_ENGRAM")          # the Engram tables are weights too\n                parts = (signature, weights_id(getattr(engine, "model_dir", "."), *([engram] if engram else [])),\n                         spanned)\n            except Exception as exc:                     # noqa: BLE001\n                problem = exc\n            if any(not row[0] for row in self._gather_ints([0 if problem is not None else 1])):\n                print(f"[tensorfold] rank {engine.rank}: spilled kept prompts off on every rank "\n                      f"({type(problem).__name__ + \': \' + str(problem) if problem is not None else \'another rank failed\'})",\n                      flush=True)\n                try:\n                    from tensorfold.cuda.spill import release_pinned\n\n                    release_pinned()                     # the staging that was pinned for a tier that is off\n                except Exception:                        # noqa: BLE001\n                    pass\n            else:\n                try:\n                    self.spill = SpillStore(spill_cfg, rank=engine.rank, world=engine.world, device="cuda",\n                                            signature=parts[0],\n                                            model_id=os.path.basename(str(getattr(engine, "model_dir", "."))),\n                                            share=self._share, gather=self._gather_ints,\n                                            weights=parts[1],\n                                            classes=["tensorfold.families.deepseek_v41.cuda.multi:Kept"],\n                                            spanned=parts[2])\n                except Exception as exc:                 # noqa: BLE001  (no disk: kept prompts stay in the window only)\n                    problem, self.spill = exc, None\n                if any(not row[0] for row in self._gather_ints([0 if self.spill is None else 1])):\n                    print(f"[tensorfold] rank {engine.rank}: spilled kept prompts off ("\n                          f"{type(problem).__name__ + \': \' + str(problem) if problem is not None else \'another rank could not start the tier\'})", flush=True)\n                    if self.spill is not None:\n                        self.spill.close()               # its threads and buffers go: the tier is off on every rank\n                    else:\n                        release_pinned()\n                    self.spill = None                    # all ranks or none: a rank without the tier skips its collectives\n                elif engine.rank == 0:\n                    self.spill.announce()\n                    st = self.spill.info()\n                    print(f"[tensorfold] spilled kept prompts: {self.spill.dir} ({st[\'entries\']} on disk, "\n                          f"{st[\'stored_bytes\'] / 2**30:.2f} of {spill_cfg.gib:g} GiB)", flush=True)\n', name)
        text = _replace_once(text, '    def _forget(self, k: Kept, give: bool = True) -> None:\n        self.kept.pop(k.eid, None)\n', '    def _release_rows(self, base: int, keep: int = 0) -> None:\n        """Wait for pending writes that read rows past ``keep`` of the extent at ``base``."""\n\n        if self.spill is None or not self.spill_jobs:\n            return\n        self.spill_jobs = [j for j in self.spill_jobs if not j[0].copied.is_set()]\n        for job, jbase, jtop in self.spill_jobs:\n            if jbase == base and jtop > keep:\n                self.spill.release(job)\n\n    def _spill_obj(self, k: Kept):\n        """A shallow copy of ``k`` whose fields the spill tier can encode (host as an array, snaps string-keyed)."""\n\n        obj = Kept.__new__(Kept)\n        obj.eid, obj.base, obj.size, obj.top = k.eid, k.base, k.size, k.top\n        obj.host = k.host.view()\n        obj.snaps = {str(b): v for b, v in k.snaps.items()}\n        obj.replay, obj.keys, obj.tick = k.replay, k.keys, k.tick\n        return obj\n\n    def _spill_rows(self, k: Kept):\n        """The pool rows of ``k``\'s prompt, named for the spill tier (``comp.<layer>.<part>``)."""\n\n        rows = []\n        for kind, planes in (("comp", self.pool.comp), ("index_k", self.pool.index_k)):\n            for i, x in sorted(planes.items()):\n                r = self.m.cfg.compress_ratios[i]\n                a, b = k.base // r, (k.base + k.top) // r\n                for j, t in enumerate(x if isinstance(x, tuple) else (x,)):\n                    rows.append((f"{kind}.{i}.{j}", t[a:b]))\n        return rows\n\n    def _dst_for(self, base: int):\n        """Place a stored row into this rank\'s pool at ``base`` (display-backed rows go through the tier\'s kernel)."""\n\n        def dst(name, shape, dt):\n            if name.startswith("rows.comp.") or name.startswith("rows.index_k."):\n                kind, layer, part = name[5:].split(".")\n                planes = self.pool.comp if kind == "comp" else self.pool.index_k\n                x = planes[int(layer)]\n                t = x[int(part)] if isinstance(x, tuple) else x\n                r = self.m.cfg.compress_ratios[int(layer)]\n                a = base // r\n                return t[a:a + shape[0]]\n            return None\n        return dst\n\n    def _share(self, values: list[int] | None) -> list[int]:\n        """Rank 0\'s ``values`` on every rank (the spill tier\'s index agreement)."""\n\n        if self.e.world < 2:\n            return list(values or [])\n        n = torch.tensor([len(values) if values is not None else 0], dtype=torch.int64, device="cuda")\n        length = max(row[0] for row in self.m.comm.gather(n).tolist())\n        if length == 0:\n            return []\n        if values is None:\n            values = [0] * length\n        t = torch.tensor(values, dtype=torch.int64, device="cuda")\n        if t.numel() < length:\n            t = torch.cat([t, torch.zeros(length - t.numel(), dtype=torch.int64, device="cuda")])\n        return self.m.comm.gather(t).tolist()[0][:length]\n\n    def _gather_ints(self, values: list[int]) -> list[list[int]]:\n        """Every rank\'s ``values`` (the spill tier\'s agreement)."""\n\n        if self.e.world < 2:\n            return [list(values)]\n        t = torch.tensor(values, dtype=torch.int64, device="cuda")\n        return self.m.comm.gather(t).tolist()\n\n    def _forget(self, k: Kept, give: bool = True) -> None:\n        self._release_rows(k.base, 0)\n        self.kept.pop(k.eid, None)\n', name)
        text = _replace_once(text, '        if self.spill is not None:                       # write-through: its rows past what its lineage holds\n            k.disk = self.spill.persist(f"{self.spill_nonce or \'run\'}-{k.eid:06d}", base=k.base, top=k.top,\n                                        replay=k.replay, snaps=k.snaps, host=k.host,\n                                        lineage=getattr(s, "lineage", None), keys=k.keys)\n', '        if self.spill is not None:                       # write-through: its rows and snapshots to disk\n            ids, failed = (k.keys if k.keys is not None else range(len(k.host))), None   # (the view is made below)\n            try:                                         # a failure here is shared by save() before its exchange\n                if k.keys is None:\n                    ids = k.host.view()\n                spill_obj, spill_rows = self._spill_obj(k), self._spill_rows(k)\n                spill_tokens = k.host.view() if k.keys is not None else None\n            except Exception as exc:                     # noqa: BLE001\n                spill_obj, spill_rows, spill_tokens, failed = None, (), None, exc\n            job = self.spill.save(ids, spill_obj, spill_rows,\n                                  skip=("disk", "hits"), prepare_failure=failed, tokens=spill_tokens)\n            if job is not None:\n                k.disk = job\n                self.spill_jobs.append((job, k.base, k.top))\n', name)
        text = _replace_once(text, '        if self.spill is not None and s.keys is not None:   # a spilled boundary past every kept prompt\'s\n            far = self.spill.best(s.keys, len(s.prompt), window=self.m.cfg.window, replay=self._replay(),\n                                  above=int(reuse[1]) if reuse else 0)\n            if far is not None:\n                reuse = ["disk", far[0], far[1]]\n', '        if self.spill is not None and s.keys is not None:   # a stored prompt longer than any kept one\n            far = self.spill.find(s.keys, grid=0, whole_needs_head=True,\n                                  usable=lambda e: bool(e.fields.get("replay")) == self._replay())\n            if far is not None and (not reuse or far.n > int(reuse[1])):\n                reuse = ["disk", far.key, far.n]\n', name)
        text = _replace_once(text, '            disk = bool(reuse) and reuse[0] == "disk"\n            if disk and not self._all(self.spill is not None and self.spill.check(str(reuse[1]), int(reuse[2]))):\n                if e.rank == 0:\n                    print(f"[tensorfold] spilled prompt {reuse[1]}: not on every rank, the prompt fills fresh",\n                          flush=True)\n                reuse, disk = None, False\n', '            disk = bool(reuse) and reuse[0] == "disk"\n', name)
        text = _replace_once(text, '            if disk:\n                name, at, t0 = str(reuse[1]), int(reuse[2]), time.perf_counter()\n                try:\n                    snaps, host, lineage = self.spill.load(name, at, base)\n                    loaded = at in snaps\n                except Exception as exc:                 # noqa: BLE001  (every rank agrees on the outcome below)\n                    print(f"[tensorfold] rank {e.rank}: spilled prompt {name} did not load ({type(exc).__name__}: "\n                          f"{exc})", flush=True)\n                    loaded = False\n                if self._all(loaded):                    # else the rows written so far get filled over, fresh\n                    cut, how = at, "disk"\n                    self._restore(index, snaps[cut])\n                    slot.sc.host = HostIds(host)\n                    s.snaps, s.lineage = snaps, lineage\n                    self.keep_stats[how] = self.keep_stats.get(how, 0) + 1\n                    if e.rank == 0:\n                        print(f"[tensorfold] spilled prompt {name}: continued at {cut} of {len(s.prompt)} (disk, "\n                              f"{time.perf_counter() - t0:.2f}s)", flush=True)\n                elif e.rank == 0:\n                    print(f"[tensorfold] spilled prompt {name}: a rank could not read it, the prompt fills fresh",\n                          flush=True)\n', '            if disk:\n                key, at, t0 = tuple(reuse[1]), int(reuse[2]), time.perf_counter()\n                try:\n                    lj = self.spill.load_async(key, self._dst_for(base))\n                    obj, _rows, _meta = self.spill.finish_load(lj)\n                    loaded = obj is not None and any(int(b) == at for b in obj.snaps)   # a snapshot at the cut must exist\n                except Exception as exc:                 # noqa: BLE001  (every rank agrees on the outcome below)\n                    print(f"[tensorfold] rank {e.rank}: spilled prompt {key} did not load ({type(exc).__name__}: "\n                          f"{exc})", flush=True)\n                    loaded = False\n                if self._all(loaded):                    # else the rows written so far get filled over, fresh\n                    cut, how = at, "disk"\n                    snaps = {int(b): v for b, v in obj.snaps.items()}\n                    self._restore(index, snaps[cut])\n                    slot.sc.host = HostIds(obj.host)\n                    s.snaps, s.lineage = snaps, None\n                    self.keep_stats[how] = self.keep_stats.get(how, 0) + 1\n                    if e.rank == 0:\n                        print(f"[tensorfold] spilled prompt {key}: continued at {cut} of {len(s.prompt)} (disk, "\n                              f"{time.perf_counter() - t0:.2f}s)", flush=True)\n                else:\n                    self.spill.forget(key)\n                    if e.rank == 0:\n                        print(f"[tensorfold] spilled prompt {key}: a rank could not read it, the prompt fills fresh",\n                              flush=True)\n', name)
        text = _replace_once(text, '        sp = self.spill\n        spilled = sp is not None and all(k.disk is not None for k in kept_now if k.top >= sp.min_top)\n        for eid in drops:\n            if int(eid) in self.kept:\n                k = self.kept[int(eid)]\n                if spilled and k.disk is not None:\n                    sp.forget(k.disk["name"])\n                self._forget(k)\n        while len(self.kept) > KEEP_ENTRIES:\n            self._forget(min(self.kept.values(), key=lambda k: k.tick))\n        if sp is not None and kept_now:\n            sp.recycle(pinned={k.disk["name"] for k in self.kept.values() if k.disk is not None})\n', '        sp = self.spill\n        spilled = sp is not None and all(k.disk is not None for k in kept_now if k.top >= sp.cfg.min_tokens)\n        for eid in drops:\n            if int(eid) in self.kept:\n                k = self.kept[int(eid)]\n                if spilled and k.disk is not None:\n                    sp.forget(k.disk.entry.key)\n                self._forget(k)\n        while len(self.kept) > KEEP_ENTRIES:\n            self._forget(min(self.kept.values(), key=lambda k: k.tick))\n', name)
        text = _replace_once(text, '            if any(b - a >= trial.size(need) for a, b in trial.gaps):\n                self.extents.give(k.base + size, k.size - size)\n', '            if any(b - a >= trial.size(need) for a, b in trial.gaps):\n                self._release_rows(k.base, m)\n                self.extents.give(k.base + size, k.size - size)\n', name)
        text = _replace_once(text, '        if src == dst:\n            return\n        c = self.m.cfg\n', '        if src == dst:\n            return\n        self._release_rows(src, 0)\n        c = self.m.cfg\n', name)
        text = _replace_once(text, '            if src is not None:\n                self.extents.give(src.base, src.size)\n', '            if src is not None:\n                self._release_rows(src.base, 0)\n                self.extents.give(src.base, src.size)\n', name)
        text = _replace_once(text, "                if self.spill is not None:               # its spilled rows to the boundary are this prompt's too\n                    s.lineage = self.spill.lineage(src.disk, cut)\n", '                s.lineage = None                         # the GLM tier stores full rows: no lineage to inherit\n', name)
        # Rank 1 never serves HTTP: its loop gets no SIGTERM handler from the server, so a plain SIGTERM (docker stop)
        # would end it without a drain. Rank 0 gets its handler from cuda/http.py serve().
        text = _replace_once(text,
            "    def follow(self, link: Link) -> None:\n"
            "        \"\"\"A follower rank: rank 0's steps, in order, forever (or until the link or the lane breaks).\"\"\"\n"
            "\n"
            "        while True:\n",
            "    def follow(self, link: Link) -> None:\n"
            "        \"\"\"A follower rank: rank 0's steps until SIGTERM or until the link closes, then its queued spill writes are drained.\"\"\"\n"
            "\n"
            "        import signal\n"
            "\n"
            "        def _stop(signum, frame):\n"
            "            raise KeyboardInterrupt                  # (a plain SIGTERM would skip the drain below)\n"
            "\n"
            "        try:\n"
            "            signal.signal(signal.SIGTERM, _stop)\n"
            "        except ValueError:                           # not the main thread: no handler\n"
            "            pass\n"
            "        try:\n"
            "            self._follow_steps(link)\n"
            "        except KeyboardInterrupt:\n"
            "            pass                                     # SIGTERM from docker stop\n"
            "        finally:                                     # also when rank 0 closed the link first (stop.sh stops rank 0 first)\n"
            "            if self.spill is not None:\n"
            "                try:\n"
            "                    signal.signal(signal.SIGTERM, signal.SIG_IGN)   # a later docker stop must not cut the drain\n"
            "                except ValueError:\n"
            "                    pass\n"
            "                self.spill.shutdown_drain()\n"
            "\n"
            "    def _follow_steps(self, link: Link) -> None:\n"
            "        \"\"\"Rank 0's steps, in order, forever (or until the link or the lane breaks).\"\"\"\n"
            "\n"
            "        while True:\n", name)
        return text
    if name == "app.py":
        text = _replace_once(text, '        template.render = render_effort\n', '        template.render = render_effort\n        self.on_exit = self._flush_spill\n', name)
        text = _replace_once(text, '    @staticmethod\n    def _response_format(', '    def _flush_spill(self) -> None:\n        """Wait for queued spill writes before the process exits (docker stop sends SIGTERM)."""\n\n        multi = getattr(self.engine, "multi", None)\n        spill = getattr(multi, "spill", None)\n        if spill is None:\n            return\n        import signal\n\n        try:\n            signal.signal(signal.SIGTERM, signal.SIG_IGN)   # a second SIGTERM must not cut the drain\n        except ValueError:\n            pass\n        spill.shutdown_drain()      # waits at most the smaller of TF_SPILL_FLUSH_S and 55 s\n\n    @staticmethod\n    def _response_format(', name)
        return text
    text = _replace_once(text, '    except KeyboardInterrupt:\n        pass\n    finally:\n        server.server_close()\n', '    except KeyboardInterrupt:\n        hook = getattr(app, "on_exit", None)\n        if hook is not None:\n            try:\n                hook()\n            except Exception as exc:\n                print(f"[tensorfold] shutdown hook failed: {exc!r}", flush=True)\n    finally:\n        server.server_close()\n', name)
    return text


def overlays() -> dict[Path, str]:
    return {Path("src/tensorfold/cuda/spill.py"): (HERE / "overlay/spill_tier.py").read_text()}


def install(checkout: Path) -> None:
    checkout = Path(checkout).resolve()
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"checkout HEAD must be {REVISION}")
    additions = overlays()
    for path in additions:
        if (checkout / path).exists() or (checkout / path).is_symlink():
            raise ValueError(f"{path}: overlay already exists")
    source = {}
    for path, expected in FILES.items():
        raw = (checkout / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"{path}: source fingerprint differs from pinned revision "
                             f"(expected sha256 {expected}, found sha256 {hashlib.sha256(raw).hexdigest()})")
        source[path] = raw.decode()
    edits = {path: transform(path, text) for path, text in source.items()}
    for path, text in (edits | additions).items():
        ast.parse(text, filename=str(path))
    for path, text in (edits | additions).items():
        (checkout / path).write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    install(parser.parse_args().checkout)


if __name__ == "__main__":
    main()
