"""backend/src/separation_request.py: 公開推論 API の入力を安全な設定値へ正規化する。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.src.config import BS_POLARFORMER


@dataclass(frozen=True)
class SeparationRequest:
    """辞書で受け取る推論指定を、モデル生成に使える不変の設定へ変換する。"""

    input_path: Path
    output_dir: Path
    model_path: str | None
    cache_dir: str | None
    providers: tuple[str, ...]
    precision: str
    chunk_size: int | None

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> SeparationRequest:
        """既存の `input_path` / `audio_path` 契約を保ったまま入力を検証する。"""
        input_path = metadata.get("input_path", metadata.get("audio_path"))
        providers = metadata.get("providers")
        precision = metadata.get("precision", BS_POLARFORMER.default_precision)
        chunk_size = metadata.get("chunk_size")

        return cls(
            input_path=Path(input_path),
            output_dir=Path(metadata.get("output_dir", BS_POLARFORMER.output_directory)),
            model_path=cls._optional_path_value(metadata, "model_path"),
            cache_dir=cls._optional_path_value(metadata, "cache_dir"),
            providers=tuple(providers or ()),
            precision=precision,
            chunk_size=chunk_size,
        )

    @staticmethod
    def _optional_path_value(metadata: Mapping[str, Any], key: str) -> str | None:
        """従来どおり、空値は未指定として扱い、指定値は文字列化する。"""
        value = metadata.get(key)
        return str(value) if value else None