"""backend/src/audio_files.py: 音声ファイルの読み書きをモデル推論から分離する。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

def load_audio(source: str | Path, sample_rate: int) -> Any:
    """ファイルを検証し、指定サンプルレートのチャンネル先行 waveform として読む。"""
    # コーデック依存は実際の読み込み時まで遅延し、API 起動と単体テストを軽く保つ。
    import librosa

    source_path = Path(source).resolve()
    waveform, _ = librosa.load(source_path, sr=sample_rate, mono=False)
    return waveform


def stereo_at_length(path: str | Path, sample_rate: int, length: int) -> Any:
    """ステムを指定周波数のステレオへそろえ、基準の長さへ合わせる。"""
    import numpy as np

    audio = np.asarray(load_audio(path, sample_rate), dtype=np.float32)
    if audio.ndim == 1:
        audio = np.stack((audio, audio))
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    normalized = np.zeros((2, length), dtype=np.float32)
    copy_length = min(length, audio.shape[1])
    normalized[:, :copy_length] = audio[:2, :copy_length]
    if not np.isfinite(normalized).all():
        raise RuntimeError("分離結果に非有限値が含まれています")
    return normalized


def write_stems(stems: Mapping[str, Any], sample_rate: int, output_dir: str | Path) -> dict[str, str]:
    """チャンネル先行のステムを WAV に書き出し、ステム名ごとの出力パスを返す。"""
    import soundfile as sound_file

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_paths = {stem: destination / f"{stem}.wav" for stem in stems}
    for stem, path in output_paths.items():
        # soundfile は samples-first を要求するため、モデル内部表現をここでだけ転置する。
        sound_file.write(path, stems[stem].T, sample_rate)
    return {stem: str(path) for stem, path in output_paths.items()}
