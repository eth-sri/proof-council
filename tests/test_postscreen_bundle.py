"""Lock the security behaviour of the postscreen audit bundler.

The bundle is uploaded to a third-party API, so ``_build_bundle`` must
never follow a symlink out of the run directory (``ZipFile.write``
follows symlinks and would exfiltrate the target's contents) and must
unconditionally strip secret material per the shared ``_secret_paths``
policy (``.codex-home/``, ``.ssh/``, ``auth.json``, ``.env``, ...).
Prebuilt ``--zip`` archives go through ``_sanitize_prebuilt_zip`` so the
same guarantee holds for them. ``main`` must refuse non-OpenAI model
configs because the uploaded bundle only rides an OpenAI
code_interpreter container.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "postscreen_run.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_postscreen_run_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildBundleTests(unittest.TestCase):
    def setUp(self):
        self.ps = _load_module()
        self.tmp = Path(tempfile.mkdtemp(prefix="psb_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "outside-secret.txt").write_text("TOP-SECRET")
        run = self.tmp / "run_x"
        (run / "subdir").mkdir(parents=True)
        (run / "normal.txt").write_text("hello")
        (run / "subdir" / "trace.json").write_text("{}")
        (run / "paper.pdf").write_text("pdf")
        (run / "compute_workspace_round_1.zip").write_text("z")
        (run / ".codex-home").mkdir()
        (run / ".codex-home" / "auth.json").write_text("CRED")
        (run / ".compute_codex_home" / "abc").mkdir(parents=True)
        (run / ".compute_codex_home" / "abc" / "auth.json").write_text("CRED")
        (run / "subdir" / ".codex").mkdir()
        (run / "subdir" / ".codex" / "config.toml").write_text("cfg")
        (run / "subdir" / "auth.json").write_text("CRED")
        (run / ".env").write_text("CRED")
        (run / "prod.env").write_text("CRED")
        (run / "subdir" / "credentials.json").write_text("CRED")
        (run / ".ssh").mkdir()
        (run / ".ssh" / "id_rsa").write_text("CRED")
        (run / "leak.txt").symlink_to("../outside-secret.txt")
        (run / "alias.txt").symlink_to("normal.txt")
        (run / "sneaky.json").symlink_to(".codex-home/auth.json")
        (run / "dangling.txt").symlink_to("does-not-exist")
        (run / "dirlink").symlink_to("subdir")
        self.run_dir = run

    def _bundle(self, **kwargs) -> list[str]:
        out_zip = self.tmp / "out" / "bundle.zip"
        kwargs.setdefault("include_workspace_zips", False)
        kwargs.setdefault("include_pdfs", False)
        self.ps._build_bundle(self.run_dir, out_zip, **kwargs)
        with zipfile.ZipFile(out_zip) as zf:
            self.blob = b"".join(zf.read(n) for n in zf.namelist())
            return sorted(zf.namelist())

    def test_symlink_out_of_run_dir_is_dropped(self):
        names = self._bundle()
        self.assertEqual(
            names, ["run_x/alias.txt", "run_x/normal.txt", "run_x/subdir/trace.json"]
        )
        self.assertNotIn(b"TOP-SECRET", self.blob)

    def test_internal_symlink_keeps_target_content(self):
        self._bundle()
        out_zip = self.tmp / "out" / "bundle.zip"
        with zipfile.ZipFile(out_zip) as zf:
            self.assertEqual(zf.read("run_x/alias.txt"), b"hello")

    def test_credentials_excluded_even_with_include_flags(self):
        names = self._bundle(include_workspace_zips=True, include_pdfs=True)
        self.assertNotIn(b"CRED", self.blob)
        secret_markers = (".codex", "auth.json", ".env", "credentials", ".ssh", "id_rsa")
        self.assertFalse(any(m in n for n in names for m in secret_markers), names)
        self.assertIn("run_x/paper.pdf", names)
        self.assertIn("run_x/compute_workspace_round_1.zip", names)

    def test_user_globs_compose_with_hard_exclusions(self):
        names = self._bundle(exclude=["subdir/*"])
        self.assertNotIn("run_x/subdir/trace.json", names)
        self.assertNotIn(b"CRED", self.blob)


class ExclusionsNoteTests(unittest.TestCase):
    def setUp(self):
        self.ps = _load_module()

    def test_built_bundle_always_mentions_credentials(self):
        note = self.ps._exclusions_note(
            from_zip=False, include_pdfs=True, include_workspace_zips=True
        )
        self.assertIn("credential", note)

    def test_prebuilt_zip_claims_nothing(self):
        note = self.ps._exclusions_note(
            from_zip=True, include_pdfs=False, include_workspace_zips=False
        )
        self.assertIn("prepared externally", note)
        self.assertNotIn("excluded when building", note)


class SanitizePrebuiltZipTests(unittest.TestCase):
    def setUp(self):
        self.ps = _load_module()
        self.tmp = Path(tempfile.mkdtemp(prefix="pss_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _make_zip(self, members: dict[str, str]) -> Path:
        src = self.tmp / "src.zip"
        with zipfile.ZipFile(src, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)
        return src

    def test_dirty_zip_is_rebuilt_without_secret_members(self):
        src = self._make_zip({
            "run/ok.txt": "hello",
            "run/.codex-home/auth.json": "CRED",
            "run/.env": "CRED",
            "run/sub/credentials.json": "CRED",
        })
        out = self.tmp / "out" / "sanitized.zip"
        bundle, dropped = self.ps._sanitize_prebuilt_zip(src, out)
        self.assertEqual(bundle, out)
        self.assertEqual(
            sorted(dropped),
            ["run/.codex-home/auth.json", "run/.env", "run/sub/credentials.json"],
        )
        with zipfile.ZipFile(bundle) as zf:
            self.assertEqual(zf.namelist(), ["run/ok.txt"])
            self.assertEqual(zf.read("run/ok.txt"), b"hello")
        with zipfile.ZipFile(src) as zf:
            self.assertIn("run/.env", zf.namelist())

    def test_clean_zip_is_used_as_is(self):
        src = self._make_zip({"run/ok.txt": "hello", "run/sub/log.md": "x"})
        out = self.tmp / "out" / "sanitized.zip"
        bundle, dropped = self.ps._sanitize_prebuilt_zip(src, out)
        self.assertEqual(bundle, src)
        self.assertEqual(dropped, [])
        self.assertFalse(out.exists())


class RewriteArtifactLinksTests(unittest.TestCase):
    def test_sandbox_and_bare_links_are_repointed(self):
        ps = _load_module()
        report = (
            "![p](sandbox:/mnt/data/audit_artifacts/progress.png) and "
            "see /mnt/data/audit_artifacts/rounds.csv plus "
            "![q](progress.png) but keep ![r](artifacts/progress.png) "
            "and unrelated /mnt/data/audit_artifacts/other.csv"
        )
        out = ps._rewrite_artifact_links(report, ["progress.png", "rounds.csv"])
        self.assertEqual(
            out,
            "![p](artifacts/progress.png) and "
            "see artifacts/rounds.csv plus "
            "![q](artifacts/progress.png) but keep ![r](artifacts/progress.png) "
            "and unrelated /mnt/data/audit_artifacts/other.csv",
        )


class ModelGuardTests(unittest.TestCase):
    def test_non_openai_model_is_rejected_before_upload(self):
        ps = _load_module()
        dummy = Path(tempfile.mkdtemp(prefix="psg_")) / "dummy.zip"
        self.addCleanup(shutil.rmtree, dummy.parent, True)
        with zipfile.ZipFile(dummy, "w") as zf:
            zf.writestr("x.txt", "x")
        argv = [
            "postscreen_run.py", "--zip", str(dummy), "--slug", "t",
            "--model", "models/anthropic/fable_5",
            "--out-dir", str(dummy.parent / "out"),
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            with self.assertRaisesRegex(SystemExit, "not an OpenAI Responses config"):
                ps.main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
