"""backend/src/separation_request.py: 公開推論 API の入力を安全な設定値へ正規化する。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BS_POLARFORMER, SEPARATION_RUNTIME, SINGER_INFORMED


@dataclass(frozen=True)
class SeparationRequest:
    """辞書で受け取る推論指定を、モデル生成に使える不変の設定へ変換する。"""

    input_path: Path
    enrollment_path: Path | None
    output_dir: Path
    model_path: str | None
    cache_dir: str | None
    providers: tuple[str, ...]
    precision: str
    chunk_size: int | None
    conditioning_lambda: str

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> SeparationRequest:
        """既存の `input_path` / `audio_path` 契約を保ったまま入力を検証する。"""
        input_path = metadata.get("input_path", metadata.get("audio_path"))
        providers = metadata.get("providers", SEPARATION_RUNTIME.providers)
        precision = metadata.get("precision", SEPARATION_RUNTIME.precision)
        chunk_size = metadata.get("chunk_size", SEPARATION_RUNTIME.chunk_size)

        if not input_path:
            raise ValueError("input_path が必要です")
        if not isinstance(providers, (list, tuple)) or not all(
            isinstance(provider, str) and provider for provider in providers
        ):
            raise ValueError("providers が不正です")
        if precision not in BS_POLARFORMER.supported_precisions:
            raise ValueError("precision が不正です")
        if chunk_size is not None and (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size < BS_POLARFORMER.win_length
        ):
            raise ValueError("chunk_size が不正です")
        conditioning_lambda = metadata.get("conditioning_lambda")
        if conditioning_lambda is not None and not isinstance(conditioning_lambda, str):
            raise ValueError("conditioning_lambda が不正です")
        # 未公開の条件はcheckpointを探す前に弾く。
        checkpoint = SINGER_INFORMED.checkpoint_for(conditioning_lambda)

        return cls(
            input_path=Path(input_path),
            enrollment_path=cls._optional_path(metadata, "enrollment_path"),
            output_dir=Path(metadata.get("output_dir", BS_POLARFORMER.output_directory)),
            model_path=cls._optional_path_value(metadata, "model_path", SEPARATION_RUNTIME.model_path),
            cache_dir=cls._optional_path_value(metadata, "cache_dir", SEPARATION_RUNTIME.cache_dir),
            providers=tuple(providers or ()),
            precision=precision,
            chunk_size=chunk_size,
            conditioning_lambda=checkpoint.conditioning_lambda,
        )

    @staticmethod
    def _optional_path_value(
        metadata: Mapping[str, Any], key: str, default: str | None = None
    ) -> str | None:
        """従来どおり、空値は未指定として扱い、指定値は文字列化する。"""
        value = metadata.get(key, default)
        return str(value) if value else None

    @staticmethod
    def _optional_path(metadata: Mapping[str, Any], key: str) -> Path | None:
        """任意のファイルパスを未指定または Path として正規化する。"""
        value = metadata.get(key)
        return Path(value) if value else None
