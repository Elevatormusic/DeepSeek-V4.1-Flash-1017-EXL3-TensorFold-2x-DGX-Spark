"""Checks of the spill transform on the pinned source. Set TF_SPILL_SOURCE to a pristine pinned checkout.

The spill transform runs after the display transform, so each check applies both to the pristine file.
"""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import textwrap
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("TF_SPILL_SOURCE", ""))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


display = load(ROOT.parent / "display-kv" / "apply.py", "display_apply_for_spill_integration")
spill = load(ROOT / "apply.py", "spill_apply_for_spill_integration")


@unittest.skipUnless(os.environ.get("TF_SPILL_SOURCE"), "set TF_SPILL_SOURCE to a pristine pinned checkout")
class SpillIntegrationTests(unittest.TestCase):
    def changed(self, relative) -> str:
        relative = Path(relative)
        text = (SOURCE / relative).read_text()
        if relative in display.FILES:
            text = display.transform(relative, text)
        return spill.transform(relative, text)

    def method_of(self, tree, class_name, *names):
        """The named methods of ``class_name``, without decorators and annotations."""
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        methods = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({method.name for method in methods}, set(names))
        for method in methods:
            method.decorator_list, method.returns = [], None
            for arg in [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]:
                arg.annotation = None
        return methods

    def test_shutdown_drain_hook_is_installed(self):
        changed = self.changed(spill.APP)
        self.assertIn("self.on_exit = self._flush_spill", changed)
        self.assertIn("spill.shutdown_drain()", changed)
        changed = self.changed(spill.HTTP)
        self.assertIn('hook = getattr(app, "on_exit", None)', changed)

    def test_every_argument_of_save_is_made_before_the_call_inside_the_guard(self):
        text = self.changed(spill.MULTI)
        call = text.index("job = self.spill.save(")
        guarded = text[text.rindex("try:", 0, call):call]
        for needle in ("self._spill_obj(k)", "self._spill_rows(k)", "k.host.view()"):
            self.assertIn(needle, guarded)                                # a failure joins the save agreement
        self.assertNotIn("k.host.view()", text[call:text.index(chr(10), text.index("tokens=", call))])
        # Every view of the ids is inside the guard too: a rank that fails there still reaches the exchange of save().
        start = text.rindex("write-through: its rows and snapshots to disk", 0, call)
        first_try = text.index("try:", start)
        for position in [i for i in range(start, call) if text.startswith("k.host.view()", i)]:
            self.assertGreater(position, first_try)
        self.assertIn("range(len(k.host))", text[start:call])      # (a placeholder with the right length, not a view)

    def test_the_composed_engine_accepts_a_nonzero_spill_size(self):
        guard = "display-only KV requires TF_DS_SPILL_GIB=0"
        pristine = (SOURCE / spill.ENGINE).read_text()
        displayed = display.transform(spill.ENGINE, pristine)
        self.assertIn(guard, displayed)                      # the display-only image keeps its guard
        composed = spill.transform(spill.ENGINE, displayed)
        self.assertNotIn(guard, composed)
        self.assertIn("display KV requires --parallel greater than 1 before weights load", composed)
        self.assertIn("display KV requires TF_DS_KV=native packed storage", composed)
        ast.parse(composed)

    def init_block(self):
        """The spill start-up block of MultiDecoder.__init__ as a function of (self, engine, spill_cfg, spill_problem)."""
        text = self.changed(spill.MULTI)
        start = text.index("        if spill_wanted:\n")
        end_marker = "{st['stored_bytes'] / 2**30:.2f} of {spill_cfg.gib:g} GiB)\", flush=True)\n"
        block = text[start:text.index(end_marker, start) + len(end_marker)]
        namespace = {"os": os}
        exec(compile("def init(self, engine, spill_cfg, spill_problem, spill_wanted=True):\n" + block, "init", "exec"),
             namespace)
        return block, namespace["init"]

    def fake_decoder(self, gather, draft=True):
        class Tensor:
            def __init__(self, shape, dtype="bf16"):
                self.shape, self.dtype = shape, dtype

        pool = types.SimpleNamespace(comp={0: Tensor((4, 8))}, index_k={0: (Tensor((4, 2)),)}, tokens=Tensor((4,), "i32"),
                                     ring=[Tensor((4, 16))], ring_size=128, comp_raw={0: (Tensor((4, 3)),)})
        dpool = types.SimpleNamespace(views=[types.SimpleNamespace(rings=[Tensor((128, 64))])]) if draft else None
        return types.SimpleNamespace(pool=pool, dpool=dpool, _replay=lambda: True, _share=None, _gather_ints=gather,
                                     spill=None)

    def run_init(self, rows, *, store_error=None, build_error=None, draft=True, cfg_problem=None, weights=None):
        """Run the block on a fake decoder; ``rows`` is what the other ranks answer in each agreement."""
        _, init = self.init_block()
        calls, created = [], []
        self.stores, self.released = [], []

        def gather(values):
            calls.append(list(values))
            return [list(values)] + [list(row) for row in rows.pop(0)] if rows else [list(values)]

        outer = self

        class Store:
            def __init__(self, *args, **kwargs):
                if store_error:
                    raise store_error
                created.append(kwargs)
                self.dir = "/x"
                self.closed = False
                outer.stores.append(self)

            announced = False

            def close(self):
                self.closed = True

            def info(self):
                return {"entries": 0, "stored_bytes": 0}

            def announce(self):
                self.announced = True

        def build_id(model_dir):
            if build_error:
                raise build_error
            return "build|driver=1"

        module = types.ModuleType("tensorfold.cuda.spill")
        module.SpillStore, module.build_id = Store, build_id
        module.release_pinned = lambda: self.released.append(1)
        module.weights_id = (lambda *dirs: weights.append(dirs) or "w") if weights is not None else (lambda *dirs: "w")
        engine = types.SimpleNamespace(world=2, rank=0, display_bytes=1 << 20, model_dir="/m")
        decoder = self.fake_decoder(gather, draft)
        with mock.patch.dict("sys.modules", {"tensorfold": types.ModuleType("tensorfold"),
                                             "tensorfold.cuda": types.ModuleType("tensorfold.cuda"),
                                             "tensorfold.cuda.spill": module}):
            init(decoder, engine, None if cfg_problem else types.SimpleNamespace(gib=1), cfg_problem)
        return decoder, calls, created

    def test_the_spill_signature_covers_the_draft_pool_and_the_identity_settings(self):
        with mock.patch.dict(os.environ, {"TF_DS_PREFILL_CHUNK": "512"}):
            decoder, calls, created = self.run_init([[[1]], [[1]]])
        signature = created[0]["signature"]
        self.assertIn("(128, 64)", signature)                       # the draft rings
        self.assertIn("('TF_DS_PREFILL_CHUNK', '512')", signature)
        self.assertIn("ring128x1", signature)
        self.assertIn("|driver=1", signature)                       # a new driver starts a new spill namespace
        self.assertIsNotNone(decoder.spill)
        self.assertEqual(calls, [[1], [1]])
        with mock.patch.dict(os.environ, {"TF_DS_PREFILL_CHUNK": "512"}):
            _, _, again = self.run_init([[[1]], [[1]]], draft=False)
        self.assertNotEqual(again[0]["signature"], signature)

    def test_a_failure_on_one_rank_turns_the_tier_off_on_every_rank_before_any_spill_collective(self):
        # This rank fails while it prepares: it still joins the agreement and never builds the store.
        decoder, calls, created = self.run_init([[[1]]], build_error=OSError("no source"))
        self.assertEqual(calls, [[0]])
        self.assertEqual(created, [])
        self.assertIsNone(decoder.spill)
        # Another rank fails: this rank sees a 0 in the agreement and does not build the store either.
        decoder, calls, created = self.run_init([[[0]]])
        self.assertEqual(calls, [[1]])
        self.assertEqual(created, [])
        self.assertIsNone(decoder.spill)

    def test_a_failure_in_the_settings_or_pinned_buffers_joins_the_agreement(self):
        # SpillConfig or preallocate() failed on this rank before the block: it still shares its 0.
        decoder, calls, created = self.run_init([[[1]]], cfg_problem=MemoryError("pinned"))
        self.assertEqual(calls, [[0]])
        self.assertEqual(created, [])
        self.assertIsNone(decoder.spill)

    def test_the_engram_tables_are_part_of_the_weights_fingerprint(self):
        seen = []
        with mock.patch.dict(os.environ, {"TF_DS_ENGRAM": "/engram"}):
            self.run_init([[[1]], [[1]]], weights=seen)
        self.assertEqual(seen, [("/m", "/engram")])
        seen.clear()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TF_DS_ENGRAM", None)
            self.run_init([[[1]], [[1]]], weights=seen)
        self.assertEqual(seen, [("/m",)])

    def test_the_tier_construction_reads_the_settings_inside_the_agreement(self):
        text = self.changed(spill.MULTI)
        self.assertIn("spill_cfg, spill_problem, spill_wanted = None, None, False", text)
        self.assertNotIn("\n                preallocate(spill_cfg)\n        self.cap", text)
        start = text.index("spill_cfg, spill_problem, spill_wanted")
        self.assertLess(start, text.index("preallocate(spill_cfg)"))
        self.assertLess(text.index("preallocate(spill_cfg)"), text.index("if spill_wanted:"))

    def test_a_restored_prompt_without_a_snapshot_at_the_cut_falls_back_to_a_fresh_fill(self):
        text = self.changed(spill.MULTI)
        start = text.index("            if disk:\n                key, at, t0")
        end = text.index("            elif how != \"fresh\":", start)
        block = textwrap.dedent(text[start:end])

        def run(snaps):
            seen = []

            class Spill:
                def load_async(self, key, dst):
                    return key

                def finish_load(self, job):
                    return types.SimpleNamespace(snaps=snaps, host=[1, 2]), [], {}

                def forget(self, key):
                    seen.append(("forget", key))

            class Self:
                spill = Spill()
                keep_stats = {}

                def _dst_for(self, base):
                    return None

                def _all(self, value):
                    seen.append(("all", bool(value)))
                    return bool(value)

                def _restore(self, index, snap):
                    seen.append(("restore", snap))

            namespace = {"time": __import__("time"), "HostIds": lambda ids: ids, "e": types.SimpleNamespace(rank=1),
                         "reuse": ["disk", (1, 2), 100], "disk": True, "base": 0, "index": 0,
                         "slot": types.SimpleNamespace(sc=types.SimpleNamespace()),
                         "s": types.SimpleNamespace(prompt=[0] * 200, snaps=None, lineage=None)}
            body = "def f(self, how):\n    cut = 0\n" + textwrap.indent(block, "    ") + "    return cut\n"
            exec(compile(body, "load", "exec"), namespace)
            cut = namespace["f"](Self(), "fresh")
            return cut, seen

        cut, seen = run({"32": "a", "64": "b"})                           # snapshots at 32 and 64, none at 100
        self.assertEqual(cut, 0)
        self.assertIn(("all", False), seen)
        self.assertIn(("forget", (1, 2)), seen)
        self.assertNotIn("restore", [item[0] for item in seen])
        cut, seen = run({"32": "a", "100": "b"})                          # a snapshot at the cut: the restore runs
        self.assertEqual(cut, 100)
        self.assertIn(("restore", "b"), seen)

    def test_a_store_that_fails_to_start_on_one_rank_is_dropped_on_every_rank(self):
        decoder, calls, created = self.run_init([[[1]], [[1]]], store_error=OSError("disk"))
        self.assertEqual(calls, [[1], [0]])
        self.assertIsNone(decoder.spill)
        decoder, calls, created = self.run_init([[[1]], [[0]]])          # this rank is fine, another is not
        self.assertEqual(calls, [[1], [1]])
        self.assertIsNone(decoder.spill)
        self.assertTrue(self.stores[0].closed)                           # its threads and buffers go with it
        decoder, calls, created = self.run_init([[[1]], [[1]]], store_error=OSError("disk"))
        self.assertEqual(self.released, [1])                             # a rank without a store lets go of the staging
        decoder, calls, created = self.run_init([[[0]]])                 # a peer fails in the first agreement
        self.assertEqual(self.released, [1])

    def test_the_start_line_comes_after_every_rank_agreed_and_a_peer_failure_is_reported(self):
        import contextlib
        import io

        decoder, calls, created = self.run_init([[[1]], [[1]]])
        self.assertTrue(decoder.spill.announced)                         # all ranks agreed: the start line is printed
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            decoder, calls, created = self.run_init([[[1]], [[0]]])      # this rank is fine, another is not
        self.assertIsNone(decoder.spill)
        self.assertIn("another rank could not start the tier", out.getvalue())

    def test_the_spill_config_has_no_highwater_and_ids_are_not_copied_to_a_list(self):
        changed = self.changed(spill.MULTI)
        self.assertNotIn("highwater", changed)
        self.assertNotIn("tokens=list(k.host.view())", changed)

    def test_shutdown_drain_waits_for_queued_writes(self):
        tree = ast.parse(self.changed(spill.APP))
        [method] = self.method_of(tree, "DsApp", "_flush_spill")
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "_flush_spill", "exec"),
             namespace)

        class Spill:
            def __init__(self):
                self.calls = 0

            def shutdown_drain(self):
                self.calls += 1
                return True

        class App:
            _flush_spill = namespace["_flush_spill"]

        app = App()
        app.engine = types.SimpleNamespace(multi=types.SimpleNamespace(spill=Spill()))
        app._flush_spill()
        self.assertEqual(app.engine.multi.spill.calls, 1)
        app.engine.multi = None
        app._flush_spill()

    def test_rank_zero_gets_its_sigterm_handler_from_the_pinned_server(self):
        """The drain hook of rank 0 runs in the KeyboardInterrupt branch, so SIGTERM must raise it. The pinned
        revision does that in serve(); if a new pin drops it, this test fails and the hook needs its own handler."""
        pinned = (SOURCE / spill.HTTP).read_text()
        self.assertIn("signal.signal(signal.SIGTERM, _terminate)", pinned)
        self.assertIn("raise KeyboardInterrupt", pinned)
        changed = self.changed(spill.HTTP)
        self.assertLess(changed.index("signal.signal(signal.SIGTERM"), changed.index('getattr(app, "on_exit"'))

    def follower(self, drain):
        tree = ast.parse(self.changed(spill.MULTI))
        methods = self.method_of(tree, "MultiDecoder", "follow", "_follow_steps")
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), "follow", "exec"), namespace)

        class Decoder:
            follow = namespace["follow"]
            _follow_steps = namespace["_follow_steps"]

        decoder = Decoder()
        decoder.spill = drain
        return decoder

    def test_follower_rank_drains_its_spill_writes_on_sigterm(self):
        import signal
        import threading

        if os.name == "nt":
            self.skipTest("needs POSIX signals")

        class Spill:
            drained = 0

            def shutdown_drain(self):
                Spill.drained += 1

        decoder = self.follower(Spill())

        class Link:
            def receive(self):
                threading.Event().wait(10)       # a follower waits here for rank 0's next step

        previous = signal.getsignal(signal.SIGTERM)
        timer = threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGTERM))
        try:
            timer.start()
            decoder._follow_steps = lambda link: Link().receive()
            decoder.follow(Link())
        finally:
            timer.cancel()
            signal.signal(signal.SIGTERM, previous)
        self.assertEqual(Spill.drained, 1)

    def test_follower_drains_its_spill_writes_when_rank_0_closes_the_link_first(self):
        import signal

        class Spill:
            drained = 0

            def shutdown_drain(self):
                Spill.drained += 1

        previous = signal.getsignal(signal.SIGTERM)
        try:
            decoder = self.follower(Spill())

            def broken(link):
                raise RuntimeError("rank 0's step link closed")      # stop.sh stops rank 0 first

            decoder._follow_steps = broken
            with self.assertRaises(RuntimeError):
                decoder.follow(object())
        finally:
            signal.signal(signal.SIGTERM, previous)
        self.assertEqual(Spill.drained, 1)

    def test_a_sigterm_during_the_drain_does_not_cut_it(self):
        import signal
        import time

        if os.name == "nt":
            self.skipTest("needs POSIX signals")

        class Spill:
            finished = False

            def shutdown_drain(self):
                os.kill(os.getpid(), signal.SIGTERM)          # docker stop arrives while the writes are queued
                time.sleep(0.2)
                Spill.finished = True

        previous = signal.getsignal(signal.SIGTERM)
        try:
            decoder = self.follower(Spill())

            def broken(link):
                raise RuntimeError("rank 0's step link closed")      # stop.sh stops rank 0 first

            decoder._follow_steps = broken
            with self.assertRaises(RuntimeError):
                decoder.follow(object())
        finally:
            signal.signal(signal.SIGTERM, previous)
        self.assertTrue(Spill.finished)

    def test_the_exit_hook_of_rank_zero_ignores_a_second_sigterm(self):
        import signal

        if os.name == "nt":
            self.skipTest("needs POSIX signals")
        tree = ast.parse(self.changed(spill.APP))
        [method] = self.method_of(tree, "DsApp", "_flush_spill")
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "_flush_spill", "exec"),
             namespace)
        seen = []

        class Spill:
            def shutdown_drain(self):
                seen.append(signal.getsignal(signal.SIGTERM))

        class App:
            _flush_spill = namespace["_flush_spill"]

        app = App()
        app.engine = types.SimpleNamespace(multi=types.SimpleNamespace(spill=Spill()))
        previous = signal.getsignal(signal.SIGTERM)
        try:
            app._flush_spill()
        finally:
            signal.signal(signal.SIGTERM, previous)
        self.assertEqual(seen, [signal.SIG_IGN])

    def test_follower_without_a_spill_tier_stops_cleanly_and_other_errors_pass_through(self):
        import signal

        previous = signal.getsignal(signal.SIGTERM)
        try:
            decoder = self.follower(None)

            def steps(link):
                raise KeyboardInterrupt

            decoder._follow_steps = steps
            decoder.follow(object())
            self.assertTrue(callable(signal.getsignal(signal.SIGTERM)))

            def broken(link):
                raise RuntimeError("rank 0's step link closed")

            decoder._follow_steps = broken
            with self.assertRaises(RuntimeError):
                decoder.follow(object())
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_drift_error_names_the_expected_and_the_found_hash(self):
        install_root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, install_root, True)
        for relative in spill.FILES:
            target = install_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((SOURCE / relative).read_text())
        changed = install_root / spill.MULTI
        changed.write_text(changed.read_text() + "# local patch" + chr(10))
        found = spill.hashlib.sha256(changed.read_text().encode()).hexdigest()
        with mock.patch.object(spill.subprocess, "check_output", return_value=spill.REVISION + chr(10)):
            with self.assertRaises(ValueError) as raised:
                spill.install(install_root)
        message = str(raised.exception)
        self.assertIn(str(spill.MULTI), message)
        self.assertIn(spill.FILES[spill.MULTI], message)
        self.assertIn(found, message)


if __name__ == "__main__":
    unittest.main()
