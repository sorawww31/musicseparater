"""colab_runtime/artifacts.py: Docker と同じ固定成果物を Colab に取得する。"""

import hashlib
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .config import BACKEND, COPY_CHUNK_SIZE, DOWNLOAD_TIMEOUT


def download(url: str, destination: Path, sha256: str) -> None:
    """途中ファイルを正式な重みとして残さず、ハッシュ一致後にだけ配置する。"""
    def digest(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    if destination.is_file() and digest(destination) == sha256:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream, COPY_CHUNK_SIZE)
        if digest(temporary) != sha256:
            raise RuntimeError(f"取得ファイルの SHA-256 が一致しません: {destination.name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def model_artifacts(dockerfile: str) -> tuple[list[tuple], list[tuple]]:
    """Dockerfile の固定 ADD を唯一の取得元とし、設定の二重管理を避ける。"""
    sources = re.findall(
        r"^ADD (https://github\.com/\S+\.git)#([0-9a-f]{40}) (/opt/model-sources/\S+)$",
        dockerfile, re.MULTILINE,
    )
    checkpoints = re.findall(
        r"^ADD --checksum=sha256:([0-9a-f]{64}) (https://\S+) (/opt/model-checkpoints/\S+)$",
        dockerfile, re.MULTILINE,
    )
    # ADD の構文変更を黙って無視すると、画面上は選べるのに推論だけ失敗する。
    expected = [line for line in dockerfile.splitlines() if line.startswith("ADD ")]
    if not sources or not checkpoints or len(expected) != len(sources) + len(checkpoints):
        raise ValueError("Dockerfile のモデル取得定義を読み取れません。Colab 設定を確認してください。")
    return sources, checkpoints


def prepare_models() -> None:
    """研究コードは再配布せず、Colab の実行時に upstream から取得する。"""
    sources, checkpoints = model_artifacts((BACKEND / "Dockerfile").read_text())
    for url, revision, directory in sources:
        path = Path(directory)
        print(f"モデル用コードを準備: {path.name}", flush=True)
        if not (path / ".git").is_dir():
            path.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", str(path)], check=True)
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
        )
        if head.returncode == 0 and head.stdout.strip() == revision:
            continue
        subprocess.run(["git", "-C", str(path), "fetch", "--depth=1", url, revision], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", revision], check=True)
    for sha256, url, directory in checkpoints:
        path = Path(directory)
        print(f"モデルの重みを準備: {path.parent.name}/{path.name}", flush=True)
        download(url, path, sha256)
