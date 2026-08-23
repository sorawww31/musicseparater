"""backend/src/storage.py: アップロード音声と分離結果を要求された構造で保存する。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from .config import AUDIO_STORAGE, AudioStorageConfig


class AudioStorage:
    """音声 ID と不透明なジョブ ID の保存パスを一元管理する。"""

    def __init__(self, config: AudioStorageConfig = AUDIO_STORAGE) -> None:
        self.config = config

    def new_id(self) -> str:
        """外部入力をパスに使わず、衝突しにくい識別子を生成する。"""
        return uuid4().hex

    def source_paths(self, audio_id: str) -> dict[str, Path]:
        """`audio/sources/{audio_id}` 配下のパスを返す。"""
        source_directory = self.config.root_directory / "sources" / self._component(audio_id)
        return {
            "directory": source_directory,
            "original": source_directory / "original.wav",
            "metadata": source_directory / "metadata.json",
        }

    def separation_paths(self, job_id: str) -> dict[str, Path]:
        """`audio/separations/{job_id}` 配下のパスを返す。"""
        separation_directory = (
            self.config.root_directory / "separations" / self._component(job_id)
        )
        return {
            "directory": separation_directory,
            "metadata": separation_directory / "metadata.json",
            "stems": separation_directory / "stems",
        }

    def find_source_path(self, audio_id: str) -> Path | None:
        """公開IDから元音源を直接解決し、サーバー外のパスを返さない。"""
        self._component(audio_id)
        sources_root = (self.config.root_directory / "sources").resolve()
        candidate = sources_root / audio_id / "original.wav"
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(sources_root)
        except ValueError:
            # 保存領域内のシンボリックリンク経由で外部ファイルを返さない。
            return None
        return resolved_candidate if resolved_candidate.is_file() else None

    def find_stem_path(self, job_id: str, stem: str) -> Path | None:
        """公開IDから成果物を直接解決し、サーバー外のパスを返さない。"""
        self._component(job_id)
        self._component(stem)
        separations_root = (self.config.root_directory / "separations").resolve()
        candidate = separations_root / job_id / "stems" / f"{stem}.wav"
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(separations_root)
        except ValueError:
            # 保存領域内のシンボリックリンク経由で外部ファイルを返さない。
            return None
        return resolved_candidate if resolved_candidate.is_file() else None

    async def save_source(
        self,
        uploadfile: UploadFile,
        audio_id: str,
        *,
        metadata: Mapping[str, Any],
    ) -> dict[str, Path]:
        """UploadFile を読み込み、元ファイルと metadata.json を保存する。"""
        paths = self.source_paths(audio_id)
        paths["directory"].mkdir(parents=True, exist_ok=True)
        await uploadfile.seek(0)
        with paths["original"].open("wb") as destination:
            while chunk := await uploadfile.read(self.config.upload_chunk_size):
                destination.write(chunk)
        self.write_metadata(paths["metadata"], metadata)
        return paths

    def write_metadata(self, path: Path, metadata: Mapping[str, Any]) -> None:
        """UTF-8 の JSON metadata を親ディレクトリと一緒に作成する。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dict(metadata), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def timestamp() -> str:
        """metadata に保存する UTC 時刻を ISO 8601 形式で返す。"""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _component(value: str) -> str:
        """識別子を単一パス要素に制限し、パストラバーサルを防ぐ。"""
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError("audio_id と job_id は単一の識別子で指定してください")
        return value
