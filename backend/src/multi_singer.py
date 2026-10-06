"""backend/src/multi_singer.py: ボーカル抽出とGPU子プロセスを二段で実行する。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .audio_files import load_audio, stereo_at_length, write_stems
from .config import (
    BS_POLARFORMER,
    JACAPPELLA_DPTNET,
    JOB_RUNTIME,
    MEDLEYVOX,
    MULTI_SINGER_AUDIO,
    SEPACAP,
    SEPARATION_RUNTIME,
    UNMIXX,
)
from .model_process import run_model_process
from .separation_request import SeparationRequest

JobProgress = Callable[[str, int], None]
ExtractVocals = Callable[[Path, Path, Callable[[int, int], None]], dict[str, str]]

# 周波数と分割長は学習条件ごとに違うため、モデル設定そのものから引く。
_MULTI_SINGER_MODELS: dict[str, Any] = {
    "unmixx": UNMIXX,
    "sepacap": SEPACAP,
    "jacappella-dptnet": JACAPPELLA_DPTNET,
    "medleyvox": MEDLEYVOX,
}

def _model_config(model_id: str) -> Any:
    """model_id に対応する固定設定を返し、未対応は推論前に拒否する。"""
    model = _MULTI_SINGER_MODELS.get(model_id)
    if model is None:
        raise ValueError(f"未対応の複数歌声モデルです: {model_id}")
    return model


def _model_settings(model_id: str) -> dict[str, Any]:
    """子プロセスへ渡す JSON 化可能な固定設定を返す。"""
    model = _model_config(model_id)
    settings: dict[str, Any] = {
        "repo_id": model.repo_id,
        "revision": model.revision,
        "source_directory": str(model.source_directory),
        "checkpoint_filename": model.checkpoint_filename,
        "checkpoint_sha256": model.checkpoint_sha256,
        "precision": model.precision,
        "stems": list(model.stems),
    }
    if model_id in ("sepacap", "medleyvox"):
        # 構成ファイルが重みと同じ配布物に入っているため、重みと同じ revision で取る。
        settings["config_filename"] = model.config_filename
    if model_id in ("sepacap", "jacappella-dptnet", "medleyvox"):
        # UNMIXX 以外は重みを実行時に Hugging Face から取得する。
        settings["cache_dir"] = SEPARATION_RUNTIME.cache_dir or str(
            BS_POLARFORMER.model_cache_directory,
        )
    if model_id in ("jacappella-dptnet", "medleyvox"):
        # asteroid の encoder/decoder 実装は別リポジトリにある。
        settings["filterbank_directory"] = str(model.filterbank_directory)
    if model_id == "medleyvox":
        settings["asteroid_directory"] = str(model.asteroid_directory)
        settings["target_lufs"] = model.target_lufs
    return settings


def _runner_settings(
    model_id: str, input_path: Path, output_directory: Path,
) -> dict[str, Any]:
    """周波数・チャンク・精度を親側の設定から明示する。"""
    model = _model_config(model_id)
    return {
        "model_id": model_id,
        "input_path": str(input_path),
        "output_directory": str(output_directory),
        "sample_rate": model.sample_rate,
        "chunk_size": model.chunk_size,
        "hop_size": model.hop_size,
        "silence_epsilon": MULTI_SINGER_AUDIO.silence_epsilon,
        "cuda_device_id": BS_POLARFORMER.cuda_device_id,
        "model": _model_settings(model_id),
    }


def separate_multi_singer(
    model_id: str,
    request: SeparationRequest,
    extract_vocals: ExtractVocals,
    progress: JobProgress,
) -> dict[str, str]:
    """BS PolarFormer → singer model → 44.1 kHz成果物の順で処理する。"""
    work_directory = request.output_dir.parent / "work"
    bs_directory = work_directory / "bs"
    child_directory = work_directory / "singers"
    work_directory.mkdir(parents=True, exist_ok=True)
    progress("extracting_vocals", 0)
    first_stage = extract_vocals(
        request.input_path,
        bs_directory,
        lambda completed, total: progress(
            "extracting_vocals",
            round(JOB_RUNTIME.vocal_extraction_end_percent * completed / total),
        ),
    )

    settings_path = work_directory / "runner.json"
    settings_path.write_text(
        json.dumps(_runner_settings(model_id, Path(first_stage["vocals"]), child_directory)),
        encoding="utf-8",
    )
    progress("separating_singers", JOB_RUNTIME.vocal_extraction_end_percent)
    run_model_process(
        settings_path,
        progress,
        phase="separating_singers",
        start_percent=JOB_RUNTIME.vocal_extraction_end_percent,
        end_percent=JOB_RUNTIME.singer_separation_end_percent,
    )
    progress("finalizing", JOB_RUNTIME.singer_separation_end_percent)

    instrumental = stereo_at_length(
        first_stage["instrumental"], BS_POLARFORMER.sample_rate,
        np.asarray(load_audio(first_stage["instrumental"], BS_POLARFORMER.sample_rate)).shape[-1],
    )
    stems = {
        stem: stereo_at_length(
            child_directory / f"{stem}.wav", BS_POLARFORMER.sample_rate, instrumental.shape[1],
        )
        for stem in _model_settings(model_id)["stems"]
    }
    stems["instrumental"] = instrumental
    output_paths = write_stems(stems, BS_POLARFORMER.sample_rate, request.output_dir)
    progress("finalizing", JOB_RUNTIME.finalizing_percent)
    return output_paths
