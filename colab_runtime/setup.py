"""colab_runtime/setup.py: Colab と独立した Python / CUDA 依存環境を uv で準備する。"""

import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

from .artifacts import download, prepare_models
from .config import (
    DOWNLOAD_TIMEOUT, GPU_HELP, NODE_MINIMUM, NODE_MINIMUM_20, NODE_VERSION,
    PYTHON, PYTHON_VERSION, ROOT, TORCH_INDEX, TORCH_REQUIREMENT, UV_VERSION, WORK,
)


def find_uv() -> str:
    """uv がなければ専用フォルダーに補い、カーネルのパッケージを変更しない。"""
    uv = shutil.which("uv")
    if uv:
        return uv
    target = WORK / "tools"
    binary = target / "bin/uv"
    if not binary.is_file():
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--target", str(target),
            f"uv=={UV_VERSION}",
        ], check=True)
    return str(binary)


def prepare_python() -> None:
    """ホストの Python / torch を使わず、指定版を独立環境へ導入する。"""
    # 大きな wheel を取得する前に、GPU が割り当てられているか確認する。
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError(GPU_HELP)
    gpu = subprocess.run([nvidia_smi, "-L"], capture_output=True, text=True)
    if gpu.returncode != 0 or not gpu.stdout.strip():
        raise RuntimeError(GPU_HELP)
    print(gpu.stdout.strip(), flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    uv = find_uv()
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(PYTHON.parents[1]))
    # sync が Python の取得と venv 作成も担当する。再実行時は追加した torch を残す。
    print(f"アプリ専用の Python {PYTHON_VERSION} と依存を準備しています…", flush=True)
    subprocess.run([
        uv, "sync", "--project", str(ROOT / "backend"), "--locked", "--no-install-project",
        "--python", PYTHON_VERSION, "--inexact",
    ], env=env, check=True)
    # torch の導入で既存 lock の共通依存が変更されないよう制約を渡す。
    constraints = WORK / "backend-constraints.txt"
    subprocess.run([
        uv, "export", "--project", str(ROOT / "backend"), "--locked",
        "--no-emit-project", "--no-hashes", "--output-file", str(constraints),
    ], env=env, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([
        uv, "pip", "install", "--python", str(PYTHON),
        "--index", TORCH_INDEX, "--constraints", str(constraints), TORCH_REQUIREMENT,
    ], check=True)
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
