"""tests/test_colab_server.py: UI 同梱時もアップロードから WAV 取得まで API が機能するか検証する。"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from colab_runtime import server


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "fastapi", "httpx", "onnxruntime")),
    "バックエンドの依存環境で実行してください（httpx はテスト用）。",
)
class ColabServerTests(unittest.TestCase):
    """StaticFiles が既存 API を隠さず、非公開ファイルを配信しないことを確認する。"""

    def test_ui_upload_job_and_download_share_one_origin(self):
        from fastapi.testclient import TestClient

        previous_cwd = Path.cwd()
        previous_path = sys.path.copy()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dist = root / "frontend/dist"
            dist.mkdir(parents=True)
            (dist / "index.html").write_text("<html>Music Separater</html>")
            (root / ".env").write_text("not-public")
            try:
                with patch.object(server, "ROOT", root), patch.dict(os.environ), \
                        patch("onnxruntime.preload_dlls"):
                    app = server.create_app()
                import main
                from src.config import AudioStorageConfig
                from src.job_manager import JobManager
                from src.storage import AudioStorage

                storage = AudioStorage(AudioStorageConfig(root_directory=root / "audio"))
                manager = JobManager(storage)

                def separate(_model, metadata, _num_vocals, _progress):
                    output = Path(metadata["output_dir"])
                    output.mkdir(parents=True)
                    paths = {stem: output / f"{stem}.wav" for stem in ("vocals", "instrumental")}
                    for path in paths.values():
                        path.write_bytes(b"test wav")
                    return {stem: str(path) for stem, path in paths.items()}

                with patch.object(main, "storage", storage), patch.object(main, "job_manager", manager), \
                        patch.object(main, "inference", side_effect=separate), TestClient(app) as client:
                    self.assertIn("Music Separater", client.get("/").text)
                    self.assertEqual(client.get("/.env").status_code, 404)
                    upload = client.post("/audios", files={"uploadfile": ("song.wav", b"source", "audio/wav")})
                    self.assertEqual(upload.status_code, 201)
                    created = client.post("/separations", json={
                        "audio_id": upload.json()["audio_id"], "model_id": "bs-polarformer",
                    })
                    self.assertEqual(created.status_code, 202)
                    manager.wait_for(created.json()["job_id"])
                    result = client.get(created.json()["status_url"]).json()
                    self.assertEqual(result["status"], "completed")
                    download = client.get(result["stems"]["vocals"] + "?download=true")
                    self.assertEqual(download.content, b"test wav")
                    self.assertIn("attachment", download.headers["content-disposition"])
            finally:
                # アプリの mount、cwd、import path を他のテストに持ち越さない。
                if "app" in locals():
                    app.router.routes[:] = [route for route in app.routes if route.name != "colab-ui"]
                os.chdir(previous_cwd)
                sys.path[:] = previous_path


if __name__ == "__main__":
    unittest.main()
