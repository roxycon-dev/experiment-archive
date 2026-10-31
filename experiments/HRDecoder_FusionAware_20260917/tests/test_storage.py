"""tests/test_storage.py: real PNG fixtures, >=2GiB headroom, mocked disk."""
import json, tempfile, unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import storage_preflight as sp


def make_png(path, w, h):
    """Real PNG bytes (PIL encodes valid IHDR/IDAT)."""
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(str(path))


class TestStoragePreflight(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.data = root / "data"
        (self.data / "images/train").mkdir(parents=True)
        make_png(self.data / "images/train/a.png", 4, 4)  # 4*4*16 = 256 B
        self.student = root / "student.bin"
        self.student.write_bytes(b"x" * 100)
        self.ref = root / "ref.bin"
        self.ref.write_bytes(b"y" * 50)
        self.cache = root / "cache"
        self.run = root / "run"
        self.out = root / "report.json"

    def _call(self, cache_free, run_free):
        ret = mock.Mock(free=cache_free, total=1 << 40, used=0)
        st = mock.Mock()
        st.st_dev = 1  # single filesystem -> combined min(free) branch
        with mock.patch("shutil.disk_usage", return_value=ret):
            return sp.check_storage(self.data, self.cache, self.run,
                                    self.student, self.ref, expected_count=1)

    def test_ok_with_2gib_headroom(self):
        r = self._call(cache_free=4 << 30, run_free=4 << 30)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["images"]["count"], 1)
        self.assertEqual(r["images"]["bytes"], 256)
        self.assertEqual(r["checkpoints"]["per_ckpt_bytes"], 2 * 150)

    def test_block_below_headroom(self):
        # 2 GiB headroom alone exceeds tiny free space
        r = self._call(cache_free=1 << 30, run_free=1 << 30)
        self.assertEqual(r["status"], "block")
        r = self._call(cache_free=10, run_free=10)
        self.assertEqual(r["status"], "block")

    def test_missing_inputs_block(self):
        r = sp.check_storage(self.data, self.cache, self.run,
                             self.student, self.tmp.name + "/nope.bin")
        self.assertEqual(r["status"], "block")
        self.assertIn("missing", r["reason"])

    def test_headroom_applied(self):
        r = self._call(cache_free=1 << 40, run_free=1 << 40)
        base = 256 + 24 * 300
        self.assertEqual(r["required_bytes"], int(base * 1.2) + (2 << 30))

    def test_nearest_existing_dir_stat(self):
        # cache/run do not exist: free_for walks to nearest existing parent,
        # which must still be stat-able for st_dev comparison.
        ret = mock.Mock(free=4 << 30, total=1 << 40, used=0)
        st = mock.Mock()
        st.st_dev = 1
        with mock.patch("shutil.disk_usage", return_value=ret) as du:
            r = sp.check_storage(self.data, self.cache, self.run,
                                 self.student, self.ref, expected_count=1)
            self.assertEqual(du.call_count, 2)
            self.assertTrue(self.data.exists())
            self.assertTrue(r["same_filesystem"])

    def test_cli_writes_only_out_json(self):
        import subprocess
        import sys
        before = sorted(str(p) for p in Path(self.tmp.name).rglob("*")
                        if p.is_file())
        cp = subprocess.run([sys.executable, "storage_preflight.py",
                             "--data-root", str(self.data),
                             "--cache-dir", str(self.cache),
                             "--run-dir", str(self.run),
                             "--student-checkpoint", str(self.student),
                             "--reference-checkpoint", str(self.ref),
                             "--out-json", str(self.out)],
                            capture_output=True)
        after = sorted(str(p) for p in Path(self.tmp.name).rglob("*")
                       if p.is_file())
        self.assertEqual(set(after) - set(before), {str(self.out)})
        self.assertIn(cp.returncode, (0, 2))
        json.loads(self.out.read_text())


if __name__ == "__main__":
    unittest.main()
