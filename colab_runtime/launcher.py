"""colab_runtime/launcher.py: Colab の再実行でサーバーを重複させず起動・停止する。"""

import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

from .config import (
    LOG, LOG_TAIL_LINES, POLL_INTERVAL, PORT, PYTHON, REQUEST_TIMEOUT,
    ROOT, STARTUP_TIMEOUT, STOP_TIMEOUT, WORK,
)

_process = None


def stop() -> None:
    """所有しているプロセス群だけを停止し、GPU 子プロセスも残さない。"""
    global _process
    if _process is None:
        return
    try:
        os.killpg(_process.pid, signal.SIGTERM)
        _process.wait(timeout=STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(_process.pid, signal.SIGKILL)
        _process.wait()
    except ProcessLookupError:
        pass
    finally:
        _process = None


def show_log() -> None:
    """初心者がセルから確認できるよう、起動・推論ログの末尾を表示する。"""
    if LOG.exists():
        print("\n".join(LOG.read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:]))
    else:
        print("まだ起動ログはありません。先に①と②を実行してください。")


def start() -> int:
    """再クリック時は既存サーバーを使い、HTTP 応答を確認してから画面を開く。"""
    global _process
    if _process is not None and _process.poll() is None:
        return PORT
    if not (WORK / "ready").exists():
        raise RuntimeError("先に「① 準備する」を最後まで実行してください。")
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", PORT))
        except OSError as error:
            raise RuntimeError("ポートが使用中です。ランタイムを再起動し①から実行してください。") from error
    with LOG.open("w") as log:
        _process = subprocess.Popen(
            [str(PYTHON), "-m", "uvicorn", "colab_runtime.server:create_app", "--factory",
             "--host", "0.0.0.0", "--port", str(PORT), "--workers", "1"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline and _process.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=REQUEST_TIMEOUT) as response:
                if response.status == 200:
                    return PORT
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(POLL_INTERVAL)
    stop()
    show_log()
    raise RuntimeError("画面を起動できませんでした。上のログを確認し①からやり直してください。")
