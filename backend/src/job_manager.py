"""backend/src/job_manager.py: GPU推論を1本ずつ実行し、状態を原子的に保存する。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from .config import JOB_RUNTIME, SINGER_INFORMED, JobRuntimeConfig
from .model_catalog import ModelDefinition
from .storage import AudioStorage

InferenceRunner = Callable[..., dict[str, str]]


class JobManager:
    """単一GPUを守る直列 executor と永続 job metadata を管理する。"""

    def __init__(
        self,
        storage: AudioStorage,
        config: JobRuntimeConfig = JOB_RUNTIME,
    ) -> None:
        self.storage = storage
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix="separation")
        self._futures: dict[str, Future[None]] = {}
        self._metadata_lock = Lock()

    def submit(
        self,
        *,
        audio_id: str,
        source_path: Path,
        model: ModelDefinition,
        num_vocals: int | None,
        runner: InferenceRunner,
        reference_audio_id: str | None = None,
        reference_path: Path | None = None,
    ) -> str:
        """queued metadata を先に保存してからGPU executorへ投入する。"""
        job_id = self.storage.new_id()
        metadata = {
            "audio_id": audio_id,
            "reference_audio_id": reference_audio_id,
            "job_id": job_id,
            "model_id": model.model_id,
            "num_vocals": num_vocals,
            "status": "queued",
            "phase": "queued",
            "progress_percent": 0,
            "created_at": self.storage.timestamp(),
            "source": str(source_path),
            "reference_source": str(reference_path) if reference_path else None,
            "stems": None,
            "error": None,
        }
        self._write(job_id, metadata)
        future = self._executor.submit(self._run, job_id, model, runner)
        self._futures[job_id] = future
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        """保存済み job 状態を返す。"""
        return self.storage.read_metadata(self.storage.separation_paths(job_id)["metadata"])

    def wait_for(self, job_id: str, timeout: float = 10) -> None:
        """テストとローカル診断向けに指定 job の完了を待つ。"""
        self._futures[job_id].result(timeout=timeout)

    def recover_interrupted(self) -> None:
        """再開機能がないため、前回停止時の未完了 job を明示的に failed にする。"""
        for metadata_path in self.storage.separation_metadata_paths():
            metadata = self.storage.read_metadata(metadata_path)
            if metadata and metadata.get("status") in {"queued", "running"}:
                metadata.update({
                    "status": "failed",
                    "phase": "failed",
                    "error": "サーバー再起動により処理が中断されました",
                })
                self.storage.write_metadata(metadata_path, metadata)
                self.storage.cleanup_job_temporary_files(str(metadata["job_id"]))

    def shutdown(self) -> None:
        """待機 job をキャンセルし、executor の新規受付を終了する。"""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: str, model: ModelDefinition, runner: InferenceRunner) -> None:
        metadata = self.get(job_id)
        if metadata is None:
            return
        try:
            skips_vocal_extraction = (
                model.requires_enrollment and not SINGER_INFORMED.cascade_vocal_extraction
            )
            initial_phase = (
                "separating_singers" if skips_vocal_extraction else "extracting_vocals"
            )
            metadata.update({"status": "running", "phase": initial_phase})
            self._write(job_id, metadata)
            paths = self.storage.separation_paths(job_id)

            def progress(phase: str, percent: int) -> None:
                metadata["phase"] = phase
                metadata["progress_percent"] = max(
                    int(metadata["progress_percent"]), min(99, max(0, percent)),
                )
                self._write(job_id, metadata)

            runner_metadata = {
                "input_path": metadata["source"],
                "output_dir": paths["staging"],
            }
            if metadata.get("reference_source"):
                runner_metadata["enrollment_path"] = metadata["reference_source"]
            raw_paths = runner(
                model.model_id,
                runner_metadata,
                metadata["num_vocals"],
                progress,
            )
            if set(raw_paths) != set(model.stems):
                raise RuntimeError("モデルの公開ステム契約と生成結果が一致しません")
            if any(not Path(path).is_file() for path in raw_paths.values()):
                raise RuntimeError("生成されたステムが見つかりません")
            stems_directory = self.storage.promote_stems(job_id)
            metadata.update({
                "status": "completed",
                "phase": "completed",
                "progress_percent": 100,
                "stems": {stem: str(stems_directory / f"{stem}.wav") for stem in model.stems},
            })
            self._write(job_id, metadata)
        except Exception:
            logging.exception("separation job failed: %s", job_id)
            metadata.update({
                "status": "failed",
                "phase": "failed",
                "error": self.config.generic_error,
            })
            self._write(job_id, metadata)
        finally:
            self.storage.cleanup_job_temporary_files(job_id)

    def _write(self, job_id: str, metadata: dict[str, Any]) -> None:
        """同一プロセスの進捗更新を直列化する。"""
        with self._metadata_lock:
            self.storage.write_metadata(self.storage.separation_paths(job_id)["metadata"], metadata)
