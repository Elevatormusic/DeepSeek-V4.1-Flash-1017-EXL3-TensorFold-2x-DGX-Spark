"""CPU regressions for exclusive creation and unsupported direct I/O."""
import errno
import os
import tempfile
import unittest
from unittest import mock

try:
    import fcntl
except ImportError as exc:
    raise unittest.SkipTest(f"needs POSIX descriptor flags: {exc}")

from test_spill_tier import load


@unittest.skipUnless(hasattr(os, "O_DIRECT"), "needs Linux direct I/O flags")
class SpillDirectTests(unittest.TestCase):
    def setUp(self):
        self.sp = load()
        self.sp.CHUNK = 8192

    def store(self, root):
        cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, writers=1, direct=False)
        store = self.sp.SpillStore(cfg, device="cpu", quiet=True)
        self.addCleanup(store.close)
        return store

    def test_unsupported_direct_creation_cannot_break_buffered_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            store = self.store(root)
            store.direct = True
            real_open, real_fcntl = os.open, fcntl.fcntl
            creates, unsafe = [], []
            rejected_fds = []

            def open_file(path, flags, mode=0o777, **kwargs):
                if str(path).endswith(".partial") and flags & os.O_CREAT:
                    creates.append(path)
                    if flags & os.O_DIRECT:              # reproduce a filesystem that creates before EINVAL
                        fd = real_open(path, flags & ~os.O_DIRECT, mode, **kwargs)
                        os.write(fd, b"left by failed direct open")
                        os.close(fd)
                        unsafe.append(path)
                        raise OSError(errno.EINVAL, "direct open unsupported")
                return real_open(path, flags, mode, **kwargs)

            def set_flags(fd, operation, argument=0):
                if operation == fcntl.F_SETFL and argument & os.O_DIRECT:
                    rejected_fds.append((fd, os.fstat(fd).st_ino))
                    raise OSError(errno.EINVAL, "direct mode unsupported")
                return real_fcntl(fd, operation, argument)

            payload = self.sp.torch.arange(64, dtype=self.sp.torch.uint8)
            with mock.patch.object(self.sp.os, "open", side_effect=open_file), \
                    mock.patch.object(fcntl, "fcntl", side_effect=set_flags):
                job = store.save([1, 2], rows=[("payload", payload)])
                self.assertTrue(store.drain(3))
            self.assertTrue(job.ok)
            self.assertEqual(len(creates), 1)
            self.assertEqual(unsafe, [])
            self.assertEqual(len(rejected_fds), 1)
            self.assertFalse(store.direct)
            self.assertFalse(os.path.exists(job.partial))
            self.assertEqual(store.stuck, {})
            _, rows, _ = store.load(self.sp.ids_key([1, 2]))
            self.assertTrue(self.sp.torch.equal(rows["payload"], payload))

    def test_a_direct_mode_error_cleans_or_charges_the_created_partial(self):
        for removable in (True, False):
            with self.subTest(removable=removable), tempfile.TemporaryDirectory() as root:
                store = self.store(root)
                store.direct = True
                real_open, real_remove, real_fcntl = os.open, os.remove, fcntl.fcntl

                def open_file(path, flags, mode=0o777, **kwargs):
                    return real_open(path, flags & ~os.O_DIRECT, mode, **kwargs)

                def set_flags(fd, operation, argument=0):
                    if operation == fcntl.F_SETFL:
                        os.write(fd, b"p" * 128)
                        raise OSError(errno.EIO, "mode setup failed")
                    return real_fcntl(fd, operation, argument)

                def remove(path):
                    if str(path).endswith(".partial") and not removable:
                        raise PermissionError("cannot remove partial")
                    return real_remove(path)

                with mock.patch.object(self.sp.os, "open", side_effect=open_file), \
                        mock.patch.object(self.sp.os, "remove", side_effect=remove), \
                        mock.patch.object(fcntl, "fcntl", side_effect=set_flags):
                    job = store.save([1, 2])
                    self.assertTrue(store.drain(3))
                    self.assertFalse(job.ok)
                    if removable:
                        self.assertFalse(os.path.exists(job.partial))
                        self.assertEqual(store.stuck, {})
                    else:
                        self.assertEqual(store.stuck, {job.partial: 128})
                        self.assertEqual(store._retry_stuck(), 128)
                        store.cap = store._total() + job.nbytes
                        self.assertIsNone(store.save([3, 4], keep=frozenset(store.index)))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "needs POSIX FIFOs")
    def test_buffered_fallback_does_not_touch_a_fifo_or_link_replacement(self):
        for kind in ("fifo", "link"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                store = self.store(root)
                store.direct = True
                path = os.path.join(store.dir, "owned.partial")
                target = os.path.join(root, "keep")
                with open(target, "wb") as output:
                    output.write(b"keep")
                fd = store._open_write(path)
                self.addCleanup(os.close, fd)
                real_fcntl = fcntl.fcntl

                def set_flags(opened, operation, argument=0):
                    if operation == fcntl.F_SETFL:
                        self.assertEqual(opened, fd)
                        os.unlink(path)
                        if kind == "fifo":
                            os.mkfifo(path)
                        else:
                            os.symlink(target, path)
                        raise OSError(errno.EINVAL, "direct mode unsupported")
                    return real_fcntl(opened, operation, argument)

                with mock.patch.object(fcntl, "fcntl", side_effect=set_flags), \
                        mock.patch.object(self.sp.os, "open", side_effect=AssertionError("path reopened")), \
                        mock.patch.object(self.sp.os, "remove", side_effect=AssertionError("path removed")):
                    self.assertFalse(store._direct_write(fd))
                    self.assertEqual(os.write(fd, b"owned bytes"), len(b"owned bytes"))
                self.assertTrue(os.path.lexists(path))
                if kind == "link":
                    self.assertTrue(os.path.islink(path))
                else:
                    self.assertTrue(self.sp.stat.S_ISFIFO(os.lstat(path).st_mode))
                with open(target, "rb") as source:
                    self.assertEqual(source.read(), b"keep")

    @unittest.skipUnless(hasattr(os, "mkfifo"), "needs POSIX FIFOs")
    def test_failed_exclusive_creation_does_not_unlink_an_existing_fifo_link_or_file(self):
        for kind in ("fifo", "link", "file"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                store = self.store(root)
                original = store._open_write
                target = os.path.join(root, "keep")
                with open(target, "wb") as output:
                    output.write(b"keep")

                def create(path):
                    if kind == "fifo":
                        os.mkfifo(path)
                    elif kind == "link":
                        os.symlink(target, path)
                    else:
                        with open(path, "wb") as output:
                            output.write(b"keep")
                    return original(path)

                with mock.patch.object(store, "_open_write", side_effect=create), \
                        mock.patch.object(self.sp.os, "remove", wraps=os.remove) as remove:
                    job = store.save([1, 2])
                    self.assertTrue(store.drain(3))
                    self.assertFalse(job.ok)
                    self.assertFalse(any(call.args[0] == job.partial for call in remove.call_args_list))
                self.assertTrue(os.path.lexists(job.partial))
                with open(target, "rb") as source:
                    self.assertEqual(source.read(), b"keep")
                if kind == "file":
                    with open(job.partial, "rb") as source:
                        self.assertEqual(source.read(), b"keep")

    def test_an_evicted_partial_is_charged_when_removal_fails_after_close(self):
        with tempfile.TemporaryDirectory() as root:
            store = self.store(root)
            real_remove = os.remove
            original = store._open_write

            def evict(path):
                opened = original(path)
                store.index.clear()
                return opened

            def remove(path):
                if str(path).endswith(".partial"):
                    raise PermissionError("cannot remove evicted partial")
                return real_remove(path)

            with mock.patch.object(store, "_open_write", side_effect=evict), \
                    mock.patch.object(self.sp.os, "remove", side_effect=remove):
                job = store.save([1, 2])
                self.assertTrue(store.drain(3))
            self.assertFalse(job.ok)
            self.assertEqual(store.stuck, {job.partial: os.lstat(job.partial).st_size})
            self.assertGreater(store.stuck[job.partial], 0)


if __name__ == "__main__":
    unittest.main()
