"""colab_runtime/config.py: Colab 専用のパスと待機時間を集約する。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
WORK = ROOT / ".colab"
PYTHON = WORK / "venv/bin/python"
LOG = WORK / "server.log"
PORT = 8000
STARTUP_TIMEOUT = 120
REQUEST_TIMEOUT = 5
POLL_INTERVAL = 1
STOP_TIMEOUT = 15
DOWNLOAD_TIMEOUT = 120
COPY_CHUNK_SIZE = 1024 * 1024
LOG_TAIL_LINES = 40
IFRAME_HEIGHT = 1000
NODE_VERSION = "22.22.0"
# Vite が対応する Node.js の下限（20 系、22 系以降）。
NODE_MINIMUM_20 = (20, 19, 0)
NODE_MINIMUM = (22, 12, 0)
GPU_HELP = "Colab の「ランタイム → ランタイムのタイプを変更」で GPU を選んでください。"
