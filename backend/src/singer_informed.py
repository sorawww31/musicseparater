"""backend/src/singer_informed.py: 参照音声付きtarget singer抽出を子プロセスへ委譲する。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .audio_files import load_audio, stereo_at_length, write_stems
from .config import BS_POLARFORMER, JOB_RUNTIME, SINGER_INFORMED
from .model_process import JobProgress, run_model_process
from .separation_request import SeparationRequest

ExtractVocals = Callable[[Path, Path, Callable[[int, int], None]], dict[str, str]]


def _model_settings(conditioning_lambda: str) -> dict[str, Any]:
    """選択されたλの検証済み重みと論文の固定アーキテクチャを子プロセスへ渡す。"""
    checkpoint = SINGER_INFORMED.checkpoint_for(conditioning_lambda)
    return {
        "paper_repo_revision": SINGER_INFORMED.paper_repo_revision,
        "conditioning_lambda": checkpoint.conditioning_lambda,
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_sha256": checkpoint.sha256,
        "embedding_checkpoint_path": str(SINGER_INFORMED.embedding_checkpoint_path),
        "embedding_checkpoint_sha256": SINGER_INFORMED.embedding_checkpoint_sha256,
        "channels": SINGER_INFORMED.channels,
        "n_fft": SINGER_INFORMED.n_fft,
        "hop_length": SINGER_INFORMED.hop_length,
        "max_bin": SINGER_INFORMED.max_bin,
        "hidden_size": SINGER_INFORMED.hidden_size,
        "lstm_layers": SINGER_INFORMED.lstm_layers,
        "lstm_dropout": SINGER_INFORMED.lstm_dropout,
        "embedding_size": SINGER_INFORMED.embedding_size,
        "embedding_layers": SINGER_INFORMED.embedding_layers,
        "embedding_n_fft": SINGER_INFORMED.embedding_n_fft,
        "embedding_hop_length": SINGER_INFORMED.embedding_hop_length,
        "precision": SINGER_INFORMED.precision,
        "stems": list(SINGER_INFORMED.stems),
    }


def _extract_enrollment_vocals(
    enrollment_path: Path, work_directory: Path, extract_vocals: ExtractVocals,
) -> Path:
    """参照音声の伴奏を落とし、clean embeddingの学習条件へそろえる。"""
    first_stage = extract_vocals(
        enrollment_path,
        work_directory / "bs-enrollment",
        lambda _completed, _total: None,
    )
    return Path(first_stage["vocals"])


def _merge_instrumental(
    child_directory: Path, instrumental_path: Path, output_dir: Path,
) -> dict[str, str]:
    """mono推定を44.1 kHzステレオへ戻し、前段の伴奏をresidualへ戻す。"""
    sample_rate = SINGER_INFORMED.sample_rate
    length = np.asarray(load_audio(instrumental_path, sample_rate)).shape[-1]
    instrumental = stereo_at_length(instrumental_path, sample_rate, length)
    # 前段で外した伴奏をresidualへ戻す。論文モデルはmono推定のため、
    # target_vocal + residual は歌声をmonoへ畳んだ元の楽曲になる。
    stems = {
        "target_vocal": stereo_at_length(
            child_directory / "target_vocal.wav", sample_rate, length,
        ),
        "residual": stereo_at_length(
            child_directory / "residual.wav", sample_rate, length,
        ) + instrumental,
    }
    return write_stems(stems, sample_rate, output_dir)


def separate_target_singer(
    request: SeparationRequest,
    extract_vocals: ExtractVocals,
    progress: JobProgress,
) -> dict[str, str]:
    """混合楽曲とenrollmentからtarget vocalとresidualを生成する。"""
    if request.enrollment_path is None:
        raise ValueError("対象歌手の参照音声が必要です")

    work_directory = request.output_dir.parent / "work"
    work_directory.mkdir(parents=True, exist_ok=True)

    mixture_path = request.input_path
    enrollment_path = request.enrollment_path
    instrumental_path: Path | None = None
    child_directory = request.output_dir
    start_percent = 0

    if SINGER_INFORMED.cascade_vocal_extraction:
        progress("extracting_vocals", 0)
        if SINGER_INFORMED.cascade_enrollment_extraction:
            # 参照音声は短いため、進捗はこの後の楽曲側の抽出だけで表す。
            enrollment_path = _extract_enrollment_vocals(
                request.enrollment_path, work_directory, extract_vocals,
            )
        first_stage = extract_vocals(
            request.input_path,
            work_directory / "bs",
            lambda completed, total: progress(
                "extracting_vocals",
                round(JOB_RUNTIME.vocal_extraction_end_percent * completed / total),
            ),
        )
        mixture_path = Path(first_stage["vocals"])
        instrumental_path = Path(first_stage["instrumental"])
        child_directory = work_directory / "singer-informed"
        start_percent = JOB_RUNTIME.vocal_extraction_end_percent

    settings_path = work_directory / "singer-informed-runner.json"
    settings = {
        "model_id": "singer-informed",
        "input_path": str(mixture_path),
        "enrollment_path": str(enrollment_path),
        "output_directory": str(child_directory),
        "sample_rate": SINGER_INFORMED.sample_rate,
        "enrollment_seconds": SINGER_INFORMED.enrollment_seconds,
        "enrollment_search_hop_size": SINGER_INFORMED.enrollment_search_hop_size,
        "enrollment_segment_count": SINGER_INFORMED.enrollment_segment_count,
        "mask_exponent": SINGER_INFORMED.mask_exponent,
        "mask_floor": SINGER_INFORMED.mask_floor,
        "silence_epsilon": SINGER_INFORMED.silence_epsilon,
        "cuda_device_id": BS_POLARFORMER.cuda_device_id,
        "model": _model_settings(request.conditioning_lambda),
    }
    settings_path.write_text(json.dumps(settings), encoding="utf-8")

    progress("separating_singers", start_percent)
    run_model_process(
        settings_path,
        progress,
        phase="separating_singers",
        start_percent=start_percent,
        end_percent=JOB_RUNTIME.singer_separation_end_percent,
    )
    progress("finalizing", JOB_RUNTIME.singer_separation_end_percent)
    if instrumental_path is not None:
        output_paths = _merge_instrumental(
            child_directory, instrumental_path, request.output_dir,
        )
    else:
        output_paths = {
            stem: str(request.output_dir / f"{stem}.wav")
            for stem in SINGER_INFORMED.stems
        }
    progress("finalizing", JOB_RUNTIME.finalizing_percent)
    return output_paths
