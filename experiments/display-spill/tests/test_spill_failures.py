"""Fault tests for the optional spill tier. Rank agreement uses CPU threads."""
import errno
import os
import tempfile
import threading
import unittest
from unittest import mock

from test_spill_tier import load


class SpillFailureTests(unittest.TestCase):
    def setUp(self):
        self.sp = load()
        self.sp.CHUNK = 8192                     # small CPU staging buffers for these small files

    def store(self, root, rank=0):
        cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, writers=1, direct=False)
        store = self.sp.SpillStore(cfg, rank=rank, device="cpu", quiet=True)
        self.addCleanup(store.close)
        return store

    def pair(self, root):
        stores = [self.store(root, rank) for rank in range(2)]
        barrier = threading.Barrier(2, timeout=3)
        rows = [None, None]

        def gather(rank, values):
            rows[rank] = list(values)
            barrier.wait()
            result = [list(row) for row in rows]
            barrier.wait()
            return result

        def connect():
            for rank, store in enumerate(stores):
                store.world = 2
                store.gather = lambda values, rank=rank: gather(rank, values)
                store.share = lambda values, rank=rank: gather(rank, values or [])[0]

        return stores, connect

    def run_pair(self, stores, action):
        results = [None, None]

        def call(rank):
            try:
                results[rank] = action(stores[rank], rank)
            except Exception as exc:
                results[rank] = exc

        threads = [threading.Thread(target=call, args=(rank,), daemon=True) for rank in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(6)
        self.assertFalse(any(thread.is_alive() for thread in threads), "a rank did not finish the agreement")
        self.assertFalse(any(isinstance(result, threading.BrokenBarrierError) for result in results), results)
        return results

    @unittest.skipUnless(hasattr(os, "mkfifo"), "needs POSIX FIFOs")
    def test_scan_rejects_a_fifo_or_link_swapped_after_lstat(self):
        for linked in (False, True):
            with self.subTest(linked=linked), tempfile.TemporaryDirectory() as root:
                store = self.store(root)
                job = store.save([1, 2])
                self.assertTrue(store.drain(3))
                path = job.path
                store.index.clear()
                fifo = os.path.join(root, "pipe") if linked else path
                real_lstat = os.lstat
                swapped = []

                def lstat(target, *args, **kwargs):
                    status = real_lstat(target, *args, **kwargs)
                    if target == path and not swapped:
                        swapped.append(True)
                        os.remove(path)
                        os.mkfifo(fifo)
                        if linked:
                            os.symlink(fifo, path)
                    return status

                errors = []

                def scan():
                    try:
                        store._scan()
                    except Exception as exc:
                        errors.append(exc)

                with mock.patch.object(self.sp.os, "lstat", lstat):
                    thread = threading.Thread(target=scan, daemon=True)
                    thread.start()
                    thread.join(1)
                    blocked = thread.is_alive()
                    if blocked:                           # release a plain open in the unfixed source
                        fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                        os.close(fd)
                        thread.join(2)
                self.assertFalse(blocked, "the startup scan blocked on a substituted FIFO")
                self.assertEqual(errors, [])
                self.assertEqual(store.index, {})
                self.assertFalse(os.path.lexists(path))
                if linked:
                    self.assertTrue(os.path.exists(fifo))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "needs POSIX FIFOs")
    def test_partial_creation_rejects_existing_fifo_link_and_file(self):
        for kind in ("fifo", "link", "file"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                store = self.store(root)
                path = os.path.join(store.dir, "entry.partial")
                target = os.path.join(root, "target")
                with open(target, "wb") as output:
                    output.write(b"keep")
                if kind == "fifo":
                    os.mkfifo(path)
                elif kind == "link":
                    os.symlink(target, path)
                else:
                    with open(path, "wb") as output:
                        output.write(b"keep")
                results = []

                def create():
                    try:
                        fd, _ = store._open_write(path)
                        os.close(fd)
                        results.append(None)
                    except OSError as exc:
                        results.append(exc)

                thread = threading.Thread(target=create, daemon=True)
                thread.start()
                thread.join(1)
                blocked = thread.is_alive()
                if blocked:
                    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                    thread.join(2)
                    os.close(fd)
                self.assertFalse(blocked, "partial creation blocked on a FIFO")
                self.assertIsInstance(results[0], OSError)
                with open(target, "rb") as source:
                    self.assertEqual(source.read(), b"keep")
                if kind == "file":
                    with open(path, "rb") as source:
                        self.assertEqual(source.read(), b"keep")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "needs POSIX FIFOs")
    def test_partial_creation_checks_and_closes_a_nonregular_descriptor(self):
        with tempfile.TemporaryDirectory() as root:
            store = self.store(root)
            path = os.path.join(root, "pipe")
            os.mkfifo(path)
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            with mock.patch.object(self.sp.os, "open", return_value=fd):
                with self.assertRaises(OSError):
                    store._open_write(os.path.join(store.dir, "entry.partial"))
            with self.assertRaises(OSError) as caught:
                os.fstat(fd)
            self.assertEqual(caught.exception.errno, errno.EBADF)

    def test_a_runtime_partial_that_cannot_be_removed_holds_capacity(self):
        with tempfile.TemporaryDirectory() as root:
            store = self.store(root)
            real_remove = os.remove

            def remove(path):
                if path.endswith(".partial"):
                    raise PermissionError("cannot remove partial")
                return real_remove(path)

            def write(fd, data):
                os.write(fd, data[:128])
                raise OSError("write stopped")

            with mock.patch.object(self.sp.os, "remove", remove):
                with mock.patch.object(store, "_write_all", side_effect=write):
                    job = store.save([1, 2])
                    self.assertTrue(store.drain(3))
                self.assertFalse(job.ok)
                self.assertEqual(store.stuck, {job.partial: 128})
                self.assertEqual(store._retry_stuck(), 128)
                store.cap = store._total() + job.nbytes
                self.assertIsNone(store.save([3, 4], keep=frozenset(store.index)))

    def test_a_lookup_memory_failure_on_either_rank_skips_the_restore_on_both(self):
        for failing_rank in range(2):
            with self.subTest(rank=failing_rank), tempfile.TemporaryDirectory() as root:
                stores, connect = self.pair(root)
                for store in stores:
                    store.save([1, 2])
                    self.assertTrue(store.drain(3))
                connect()
                class NoMemory(dict):
                    def values(self):
                        raise MemoryError("lookup list")

                original = stores[failing_rank].index
                stores[failing_rank].index = NoMemory(original)
                try:
                    results = self.run_pair(stores, lambda store, rank: store.find([1, 2, 3]))
                finally:
                    stores[failing_rank].index = original
                self.assertEqual(results, [None, None])
                results = self.run_pair(stores, lambda store, rank: store.find([1, 2, 3]))
                self.assertTrue(all(result is not None and not isinstance(result, Exception) for result in results))
                self.assertEqual(results[0].key, results[1].key)

    def test_a_reconcile_apply_failure_disables_both_ranks_after_failed_unlink(self):
        with tempfile.TemporaryDirectory() as root:
            stores, connect = self.pair(root)
            for store in stores:
                store.save([1, 2])
                store.save([3, 4])
                self.assertTrue(store.drain(3))
            stores[1].forget(self.sp.ids_key([3, 4]))
            victim = stores[0].index[self.sp.ids_key([3, 4])]
            path = os.path.join(stores[0].dir, victim.name + ".safetensors")
            real_remove = os.remove

            class NoMemory(dict):
                def __setitem__(self, key, value):
                    raise MemoryError("stuck index")

            def remove(target):
                if target == path:
                    raise PermissionError("cannot remove victim")
                return real_remove(target)

            stores[0].stuck = NoMemory()
            connect()
            with mock.patch.object(self.sp.os, "remove", remove):
                results = self.run_pair(stores, lambda store, rank: store._reconcile())
            self.assertIsInstance(results[0], MemoryError)
            self.assertIsInstance(results[1], RuntimeError)
            for store in stores:
                self.assertTrue(store._disabled)
                self.assertEqual(store.index, {})
                self.assertIsNone(store.find([1, 2, 3]))
                self.assertIsNone(store.save([5, 6]))

    def test_a_save_queue_failure_rolls_back_both_ranks_before_any_write(self):
        for failing_rank in range(2):
            for append_first in (False, True):
                with self.subTest(rank=failing_rank, append_first=append_first), tempfile.TemporaryDirectory() as root:
                    stores, connect = self.pair(root)
                    for store in stores:
                        store.save([1, 2])
                        self.assertTrue(store.drain(3))
                    connect()
                    failing = stores[failing_rank]
                    clocks = [store.clock for store in stores]
                    real_put = failing.jobs.put
                    queued = []

                    def put(job):
                        queued.append(job)
                        if append_first:
                            real_put(job)
                        raise MemoryError("queue append")

                    with mock.patch.object(failing.jobs, "put", side_effect=put), \
                            mock.patch.object(stores[0], "_write", wraps=stores[0]._write) as write0, \
                            mock.patch.object(stores[1], "_write", wraps=stores[1]._write) as write1:
                        results = self.run_pair(stores, lambda store, rank: store.save([3, 4]))
                        self.assertEqual(results, [None, None])
                        self.assertTrue(all(store.drain(3) for store in stores))
                        write0.assert_not_called()
                        write1.assert_not_called()
                    for store in stores:
                        self.assertTrue(store.has([1, 2]))
                        self.assertFalse(store.has([3, 4]))
                        self.assertEqual(store.pending, {})
                        self.assertEqual(store.active, set())
                        self.assertFalse(store._disabled)
                    self.assertEqual([store.clock for store in stores], clocks)
                    self.assertTrue(queued[0].done.is_set())
                    self.assertTrue(queued[0].copied.is_set())
                    self.assertIsNone(queued[0].items)
                    results = self.run_pair(stores, lambda store, rank: store.save([3, 4]))
                    self.assertTrue(all(not isinstance(result, Exception) and result is not None for result in results))
                    self.assertTrue(all(store.drain(3) for store in stores))
                    self.assertTrue(all(result.ok for result in results))

    def test_a_failed_final_gather_cancels_the_local_job_and_propagates(self):
        for stage in ("final", "rollback"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as root:
                store = self.store(root)
                queued, exchanges = [], []
                real_put = store.jobs.put

                def put(job):
                    queued.append(job)
                    real_put(job)

                def gather(values):
                    if len(values) == 5:
                        return [list(values), list(values)]
                    if stage == "rollback" and not exchanges:
                        exchanges.append("commit")
                        return [list(values), [1]]          # the peer could not commit
                    raise RuntimeError(stage + " gather failed")

                store.world, store.gather = 2, gather
                threads = list(store._threads)
                with mock.patch.object(store.jobs, "put", side_effect=put), \
                        mock.patch.object(store, "_write", wraps=store._write) as write:
                    with self.assertRaisesRegex(RuntimeError, stage + " gather failed"):
                        store.save([1, 2])
                    write.assert_not_called()
                self.assertEqual(len(queued), 1)
                job = queued[0]
                self.assertTrue(job.decided.is_set())
                self.assertTrue(job.copied.is_set())
                self.assertTrue(job.done.is_set())
                self.assertTrue(job.cancelled)
                self.assertIsNone(job.items)
                self.assertEqual(store.index, {})
                self.assertEqual(store.pending, {})
                self.assertEqual(store.active, set())
                self.assertTrue(store._disabled)
                self.assertFalse(os.path.exists(job.path))
                store.close()
                self.assertFalse(any(thread.is_alive() for thread in threads))


if __name__ == "__main__":
    unittest.main()
