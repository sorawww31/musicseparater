"""tests/test_colab_runtime.py: Colab の取得・再実行・停止を GPU なしで検証する。"""

import ast
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from colab_runtime import artifacts, launcher, setup
from colab_runtime.config import ROOT


class ColabArtifactTests(unittest.TestCase):
    """ダウンロードの破損と、不完全な成果物定義を見逃さない。"""

    def test_download_checks_hash_and_reuses_verified_file(self):
        payload = b"verified model"
        checksum = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model.pth"
            with patch.object(artifacts.urllib.request, "urlopen", return_value=io.BytesIO(payload)) as request:
                artifacts.download("https://example.org/model", destination, checksum)
                artifacts.download("https://example.org/model", destination, checksum)
            request.assert_called_once()
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(destination.with_suffix(".pth.part").exists())

    def test_bad_download_never_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model.pth"
            destination.write_bytes(b"previous")
            with patch.object(artifacts.urllib.request, "urlopen", return_value=io.BytesIO(b"broken")):
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    artifacts.download("https://example.org/model", destination, "0" * 64)
            self.assertEqual(destination.read_bytes(), b"previous")
            self.assertFalse(destination.with_suffix(".pth.part").exists())

    def test_docker_definitions_cover_configured_models(self):
        from backend.src import config

        sources, checkpoints = artifacts.model_artifacts((ROOT / "backend/Dockerfile").read_text())
        paths = {directory for _, _, directory in sources}
        for model in (config.UNMIXX, config.SEPACAP, config.JACAPPELLA_DPTNET, config.MEDLEYVOX):
            self.assertIn(str(model.source_directory), paths)
        self.assertIn(str(config.JACAPPELLA_DPTNET.filterbank_directory), paths)
        hashes = {directory: checksum for checksum, _, directory in checkpoints}
        for checkpoint in config.SINGER_INFORMED.checkpoints:
            self.assertEqual(hashes[str(checkpoint.path)], checkpoint.sha256)
        self.assertEqual(
            hashes[str(config.SINGER_INFORMED.embedding_checkpoint_path)],
            config.SINGER_INFORMED.embedding_checkpoint_sha256,
        )

    def test_unknown_add_definition_fails_instead_of_skipping(self):
        dockerfile = (ROOT / "backend/Dockerfile").read_text()
        with self.assertRaises(ValueError):
            artifacts.model_artifacts(dockerfile + "\nADD https://example.org/new /opt/new\n")


class ColabSetupTests(unittest.TestCase):
    """既存 lock を同期し、Colab カーネルの環境を変更しないことを確認する。"""

    def test_uv_sync_uses_existing_lock_and_isolated_environment(self):
        torch = SimpleNamespace(__version__="colab-cuda", cuda=SimpleNamespace(
            is_available=lambda: True, get_device_name=lambda _: "test GPU",
        ))
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            python = work / "venv/bin/python"
            with patch.dict(sys.modules, {"torch": torch}), patch.object(setup, "WORK", work), \
                    patch.object(setup, "PYTHON", python), patch.object(setup.shutil, "which", return_value="uv"), \
                    patch.object(setup.subprocess, "run") as run:
                setup.prepare_python()
            create, sync, check = run.call_args_list
            self.assertIn("--system-site-packages", create.args[0])
            self.assertEqual(sync.args[0], [
                "uv", "sync", "--project", str(ROOT / "backend"), "--locked", "--no-install-project",
            ])
            self.assertEqual(sync.kwargs["env"]["UV_PROJECT_ENVIRONMENT"], str(python.parents[1]))
            self.assertEqual(check.args[0][0], str(python))

    def test_cpu_only_runtime_fails_before_installation(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"torch": torch}), patch.object(setup.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                setup.prepare_python()
        run.assert_not_called()

    def test_existing_supported_node_needs_no_download(self):
        with patch.object(setup.shutil, "which", return_value="/usr/bin/node"), \
                patch.object(setup.subprocess, "check_output", return_value="v22.12.0\n"), \
                patch.object(setup, "download") as download:
            env = setup.node_environment()
        download.assert_not_called()
        self.assertEqual(env["PATH"], setup.os.environ["PATH"])
        self.assertEqual(env["VITE_API_URL"], ".")

    def test_notebook_code_compiles_and_has_no_saved_output(self):
        notebook = json.loads((ROOT / "music_separater.ipynb").read_text())
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])


class ColabLauncherTests(unittest.TestCase):
    """重複サーバーや、準備失敗後の誤起動を防ぐ。"""

    def test_reopening_reuses_running_process(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(launcher, "_process", process), patch.object(launcher.subprocess, "Popen") as start:
            self.assertEqual(launcher.start(), launcher.PORT)
        start.assert_not_called()

    def test_missing_setup_fails_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(launcher, "WORK", Path(temporary)), patch.object(launcher, "_process", None):
                with self.assertRaisesRegex(RuntimeError, "①"):
                    launcher.start()

    def test_stop_terminates_owned_group_and_forces_hung_process(self):
        process = Mock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired("uvicorn", launcher.STOP_TIMEOUT), 0]
        with patch.object(launcher, "_process", process), patch.object(launcher.os, "killpg") as kill:
            launcher.stop()
            self.assertIsNone(launcher._process)
        self.assertEqual([call.args for call in kill.call_args_list], [
            (process.pid, launcher.signal.SIGTERM), (process.pid, launcher.signal.SIGKILL),
        ])


if __name__ == "__main__":
    unittest.main()
