"""colab_runtime/server.py: 既存 API とビルド済み UI を Colab の同一ポートで配信する。"""

import os
import sys

from .config import BACKEND, ROOT, WORK


def create_app():
    """既存 API を先に登録し、公開対象を frontend/dist のみに限定する。"""
    from fastapi.staticfiles import StaticFiles

    # モデル子プロセスも backend を作業ディレクトリとして起動する必要がある。
    os.chdir(BACKEND)
    sys.path.insert(0, str(BACKEND))
    os.environ.setdefault("HF_HOME", str(WORK / "huggingface"))
    import onnxruntime as ort

    ort.preload_dlls()
    from main import app

    app.mount("/", StaticFiles(directory=ROOT / "frontend/dist", html=True), name="colab-ui")
    return app
