"""backend/tests/test_audio_files.py: モデルから分離した音声ファイル I/O を確認する。"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio_files import load_audio, write_stems  # noqa: E402


class AudioFileTests(unittest.TestCase):
    """重いコーデック処理をせず、I/O 境界の引数と出力契約を検証する。"""

    def test_load_audio_requests_stereo_waveform_at_target_rate(self) -> None:
        """librosa にはモデルで扱うサンプルレートと mono=False を渡す。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "song.wav"
            source_path.touch()
            waveform = np.ones((2, 3), dtype=np.float32)
            fake_librosa = SimpleNamespace(load=Mock(return_value=(waveform, 44_100)))

            with patch.dict(sys.modules, {"librosa": fake_librosa}):
                actual = load_audio(source_path, 44_100)

        np.testing.assert_array_equal(actual, waveform)
        fake_librosa.load.assert_called_once_with(source_path, sr=44_100, mono=False)

    def test_write_stems_transposes_waveforms_and_returns_paths(self) -> None:
        """WAV 出力だけで channels-first と samples-first の変換を吸収する。"""
        waveform = np.arange(6, dtype=np.float32).reshape(2, 3)
        fake_soundfile = SimpleNamespace(write=Mock())
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "output"
            with patch.dict(sys.modules, {"soundfile": fake_soundfile}):
                actual = write_stems({"vocals": waveform}, 44_100, output_dir)
            self.assertEqual(actual, {"vocals": str(output_dir / "vocals.wav")})
            self.assertTrue(output_dir.is_dir())

        np.testing.assert_array_equal(fake_soundfile.write.call_args.args[1], waveform.T)
        self.assertEqual(fake_soundfile.write.call_args.args[2], 44_100)
