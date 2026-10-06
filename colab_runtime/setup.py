"""colab_runtime/setup.py: Colab の既存 CUDA 環境を保ち、アプリを準備する。"""

import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

from .artifacts import download, prepare_models
from .config import (
    DOWNLOAD_TIMEOUT, GPU_HELP, NODE_MINIMUM, NODE_MINIMUM_20, NODE_VERSION,
    PYTHON, ROOT, WORK,
)


def prepare_python() -> None:
    """notebook カーネルとは別の venv に追加依存を入れ、再起動を不要にする。"""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(GPU_HELP)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("このアプリは Python 3.12 対応です。Colab のランタイム版を確認してください。")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv が見つかりません。uv が入った Colab ランタイムを使用してください。")
    print(f"GPU: {torch.cuda.get_device_name(0)} / PyTorch {torch.__version__}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if not PYTHON.exists():
        subprocess.run([
            uv, "venv", "--system-site-packages", "--python", sys.executable,
            str(PYTHON.parents[1]),
        ], check=True)
    # Docker と同じ lock を使う。torch はプロジェクト依存に含めず Colab 版を継承する。
    subprocess.run([
        uv, "sync", "--project", str(ROOT / "backend"), "--locked", "--no-install-project",
    ], env=dict(os.environ, UV_PROJECT_ENVIRONMENT=str(PYTHON.parents[1])), check=True)
    subprocess.run([str(PYTHON), "-m", "colab_runtime.gpu"], cwd=ROOT, check=True)


def node_environment() -> dict[str, str]:
    """Colab の Node/npm を優先し、Vite 非対応の場合だけローカルに補う。"""
    env = dict(os.environ, VITE_API_URL=".")
    if shutil.which("node") and shutil.which("npm"):
        output = subprocess.check_output(["node", "--version"], text=True)
        version = tuple(int(part) for part in output.strip().lstrip("v").split("."))
        if version >= NODE_MINIMUM or (version[0] == 20 and version >= NODE_MINIMUM_20):
            print(f"Colab の Node.js {output.strip()} / npm を使います。", flush=True)
            return env
    print("UI のビルドに必要な Node.js / npm をアプリ専用フォルダーに準備します。", flush=True)
    name = f"node-v{NODE_VERSION}-linux-x64"
    directory = WORK / name
    archive = WORK / f"{name}.tar.xz"
    if not (directory / "bin/node").is_file():
        url = f"https://nodejs.org/dist/v{NODE_VERSION}"
        with urllib.request.urlopen(f"{url}/SHASUMS256.txt", timeout=DOWNLOAD_TIMEOUT) as response:
            sums = dict(line.split()[::-1] for line in response.read().decode().splitlines())
        download(f"{url}/{archive.name}", archive, sums[archive.name])
        with tarfile.open(archive) as stream:
            stream.extractall(WORK, filter="data")
    env["PATH"] = f"{directory / 'bin'}:{os.environ['PATH']}"
    return env


def prepare_frontend() -> None:
    """既存 UI をビルドし、API と同じ origin へ接続する。"""
    env = node_environment()
    frontend = ROOT / "frontend"
    subprocess.run(["npm", "ci"], cwd=frontend, env=env, check=True)
    subprocess.run([
        "npm", "run", "build", "--", "--base=./",
    ], cwd=frontend, env=env, check=True)


def main() -> None:
    """失敗した段階で止め、起動セルに不完全な準備を引き継がない。"""
    print("1/3 Python と GPU を確認しています…", flush=True)
    prepare_python()
    print("2/3 モデル用コードと重みを取得しています…", flush=True)
    prepare_models()
    print("3/3 操作画面を準備しています…", flush=True)
    prepare_frontend()
    (WORK / "ready").touch()
    print("準備ができました。次の「② アプリを開く」を実行してください。", flush=True)


if __name__ == "__main__":
    # 前回成功の印を残すと、途中で失敗した再セットアップを起動してしまう。
    (WORK / "ready").unlink(missing_ok=True)
    main()
