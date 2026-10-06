"""backend/src/model_process.py: GPU子プロセスの進捗転送と終了処理を一元管理する。"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from threading import Lock

from .config import JOB_RUNTIME

JobProgress = Callable[[str, int], None]

_PROCESS_LOCK = Lock()
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None


def run_model_process(
    settings_path: Path,
    progress: JobProgress,
    *,
    phase: str,
    start_percent: int,
    end_percent: int,
) -> None:
    """shellを介さずrunnerを実行し、構造化された進捗行だけを採用する。"""
    global _ACTIVE_PROCESS

    command = [sys.executable, "-m", "model_runtime.runner", str(settings_path)]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with _PROCESS_LOCK:
        _ACTIVE_PROCESS = process
    recent_output: list[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.rstrip()
            recent_output = [*recent_output[-(JOB_RUNTIME.child_log_tail_lines - 1):], stripped]
            fields = stripped.split()
            if len(fields) != 3 or fields[0] != "PROGRESS":
                continue
            try:
                completed, total = int(fields[1]), int(fields[2])
            except ValueError:
                continue
            if total > 0:
                percent = start_percent + round(
                    (end_percent - start_percent) * completed / total
                )
                progress(phase, percent)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError("子プロセスが失敗しました: " + "\n".join(recent_output))
    finally:
        with _PROCESS_LOCK:
            if _ACTIVE_PROCESS is process:
                _ACTIVE_PROCESS = None


def terminate_active_process() -> None:
    """API shutdown時に実行中のGPU子プロセスを停止する。"""
    with _PROCESS_LOCK:
        process = _ACTIVE_PROCESS
    if process is not None and process.poll() is None:
        process.terminate()
