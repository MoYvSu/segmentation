# -*- coding: utf-8 -*-
"""在独立临时Git仓库验证保密检查，不使用任何真实赛题数据。"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

GUARD = Path(__file__).resolve().parents[1] / "tools/confidential_artifacts_guard.py"


class ConfidentialGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Guard Test")
        self.git("config", "user.email", "guard@example.invalid")

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root).decode().strip()

    def guard(self, mode, stdin=""):
        return subprocess.run([sys.executable, str(GUARD), mode], cwd=self.root,
                              input=stdin.encode(), capture_output=True)

    def add(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        self.git("add", "--", name)

    def test_plain_code_passes_and_output_json_is_blocked(self):
        self.add("model.py", b"print('ok')\n")
        self.assertEqual(self.guard("--staged").returncode, 0)
        self.add("output/prediction.json", b"{}")
        self.assertNotEqual(self.guard("--staged").returncode, 0)

    def test_renamed_image_and_embedded_image_are_blocked(self):
        self.add("notes.txt", bytes.fromhex("89504e470d0a1a0a") + b"synthetic")
        self.assertNotEqual(self.guard("--staged").returncode, 0)
        self.add("notes.txt", b"data:" + b"image/png;base64," + b"A" * 64)
        self.assertNotEqual(self.guard("--staged").returncode, 0)

    def test_clean_tip_does_not_hide_contaminated_ancestor(self):
        self.add("output/prediction.json", b"{}")
        self.git("commit", "-qm", "synthetic artifact")
        self.git("rm", "-q", "output/prediction.json")
        self.add("model.py", b"pass\n")
        self.git("commit", "-qm", "clean tip")
        oid = self.git("rev-parse", "HEAD")
        self.assertEqual(self.guard("--staged").returncode, 0)
        self.assertNotEqual(self.guard("--pre-push", f"refs/heads/main {oid} refs/heads/main {'0'*40}\n").returncode, 0)

    def test_internal_ref_blocked_and_branch_deletion_allowed(self):
        self.add("model.py", b"pass\n")
        self.git("commit", "-qm", "code")
        oid = self.git("rev-parse", "HEAD")
        self.assertNotEqual(self.guard("--pre-push", f"refs/codex/snapshot {oid} refs/codex/snapshot {'0'*40}\n").returncode, 0)
        self.assertEqual(self.guard("--pre-push", f"(delete) {'0'*40} refs/heads/old {oid}\n").returncode, 0)
        self.assertEqual(self.guard("--pre-push", f"refs/heads/main {oid} refs/heads/main {'0'*40}\n").returncode, 0)

    def test_installed_hooks_survive_branch_switch_and_preserve_existing_hook(self):
        hooks = self.root / ".git/hooks"
        (hooks / "pre-commit").write_text("#!/bin/sh\nexit 0\n")
        self.assertNotEqual(self.guard("--install").returncode, 0)
        self.assertEqual((hooks / "pre-commit").read_text(), "#!/bin/sh\nexit 0\n")
        (hooks / "pre-commit").unlink()
        self.assertEqual(self.guard("--install").returncode, 0)
        self.add("model.py", b"pass\n")
        self.git("commit", "-qm", "code")
        self.git("checkout", "-qb", "another-branch")
        self.assertTrue(self.git("check-ignore", "output/local.json"))
        self.add("renamed.dat", bytes.fromhex("89504e470d0a1a0a") + b"synthetic")
        result = subprocess.run(["git", "commit", "-qm", "must fail"], cwd=self.root, capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
