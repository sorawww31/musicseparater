"""backend/src/audio_files.py: 音声ファイルの読み書きをモデル推論から分離する。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import librosa
import soundfile as sound_file


def load_audio(source: str | Path, sample_rate: int) -> Any:
    """ファイルを検証し、指定サンプルレートのチャンネル先行 waveform として読む。"""

    source_path = Path(source).resolve()
    waveform, _ = librosa.load(source_path, sr=sample_rate, mono=False)
    return waveform


def write_stems(stems: Mapping[str, Any], sample_rate: int, output_dir: str | Path) -> dict[str, str]:
    """チャンネル先行のステムを WAV に書き出し、ステム名ごとの出力パスを返す。"""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_paths = {stem: destination / f"{stem}.wav" for stem in stems}
    for stem, path in output_paths.items():
        # soundfile は samples-first を要求するため、モデル内部表現をここでだけ転置する。
        sound_file.write(path, stems[stem].T, sample_rate)
    return {stem: str(path) for stem, path in output_paths.items()}
