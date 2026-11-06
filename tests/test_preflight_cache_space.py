"""Check cache-space limits with the real local and worker shell paths."""
import unittest

from test_memory_checks import ENV_BASE, Tree, needs_bash

GIB = 1073741824


@needs_bash
class PreflightCacheSpace(unittest.TestCase):
    def check_space(self, free, warm, output=None, status=0, mkdir_fails=False):
        tree = Tree(self, "", ("common.sh", "preflight.sh"))
        (tree.dir / ".env").write_text(ENV_BASE.format(d=tree.dir), encoding="utf-8")
        if warm:
            (tree.dir / "cache" / "tf-ds-rank").mkdir(parents=True)
        for name in ("ping", "ss", "docker", "systemctl", "ibv_devinfo"):
            tree.stub(name, "exit 0\n" if name == "ping" else "exit 1\n")
        if mkdir_fails:
            tree.stub("mkdir", "exit 1\n")
        tree.stub("df", 'printf "%s\\n" "$*" >> "$DF_LOG"\n'
                  'case "$1" in\n'
                  '  -B1) printf "%s" "$DF_OUTPUT" ;;\n'
                  '  -BG) printf "Avail\\n%sG\\n" "$DF_ROUNDED" ;;\n'
                  '  *) exit 98 ;;\n'
                  'esac\nexit "$DF_STATUS"\n')
        log = tree.dir / "df-args"
        result = tree.run(["bash", "scripts/preflight.sh"], {
            "DF_LOG": str(log), "DF_OUTPUT": output if output is not None else f"Avail\n{free}\n",
            "DF_ROUNDED": str((free + GIB - 1) // GIB), "DF_STATUS": str(status),
        })
        lines = [line for line in result.stdout.splitlines()
                 if "free for the cache" in line or "free in " in line or "cannot read free space" in line]
        self.assertEqual(len(lines), 2, result.stdout + result.stderr)
        return result, lines, log.read_text().splitlines() if log.exists() else []

    def assert_cache_state(self, free, warm, passed, **kwargs):
        result, lines, calls = self.check_space(free, warm, **kwargs)
        for line in lines:
            self.assertTrue(line.startswith("[ ok ]" if passed else "[FAIL]"), line)
        if not passed:
            self.assertNotEqual(result.returncode, 0)
        return lines, calls

    def test_one_byte_below_each_limit_fails(self):
        for warm, need in ((True, 5), (False, 110)):
            with self.subTest(warm=warm):
                self.assert_cache_state(need * GIB - 1, warm, False)

    def test_exact_limit_and_one_byte_above_pass(self):
        for warm, need in ((True, 5), (False, 110)):
            for extra in (0, 1):
                with self.subTest(warm=warm, extra=extra):
                    self.assert_cache_state(need * GIB + extra, warm, True)

    def test_rounded_up_fraction_does_not_pass(self):
        for warm, need in ((True, 5), (False, 110)):
            with self.subTest(warm=warm):
                self.assert_cache_state((need - 1) * GIB + GIB // 3, warm, False)

    def test_display_is_decimal_and_df_reads_bytes(self):
        lines, calls = self.assert_cache_state(5 * GIB, True, True)
        self.assertTrue(all("5.0 GiB free for the cache" in line for line in lines), lines)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call.startswith("-B1 --output=avail ") for call in calls), calls)

    def test_zero_space_fails(self):
        self.assert_cache_state(0, True, False)

    def test_failed_query_cannot_pass_with_valid_stdout(self):
        lines, _ = self.assert_cache_state(200 * GIB, True, False, status=1)
        self.assertTrue(all("cannot read free space" in line for line in lines), lines)

    def test_invalid_query_output_fails(self):
        for output in ("", "Avail\n", "Avail\n-1\n", "Avail\n5G\n", "Avail\n1 2\n",
                       "Avail\n214748364800\nextra\n"):
            with self.subTest(output=output):
                lines, _ = self.assert_cache_state(200 * GIB, True, False, output=output)
                self.assertTrue(all("cannot read free space" in line for line in lines), lines)

    def test_failed_cache_directory_creation_fails_before_df(self):
        lines, calls = self.assert_cache_state(200 * GIB, False, False, mkdir_fails=True)
        self.assertTrue(all("cannot read free space" in line for line in lines), lines)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
