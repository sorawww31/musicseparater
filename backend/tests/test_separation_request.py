"""backend/tests/test_separation_request.py: 公開推論 API の入力正規化を確認する。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.separation_request import SeparationRequest  # noqa: E402


class SeparationRequestTests(unittest.TestCase):
    """辞書入力を一箇所で検証し、既存の API 契約を維持する。"""

    def test_normalizes_legacy_audio_path_and_runtime_settings(self) -> None:
        """audio_path の別名と可変値を、モデル生成に使える安定した設定へ変換する。"""
        request = SeparationRequest.from_metadata(
            {
                "audio_path": "song.wav",
                "enrollment_path": "reference.wav",
                "output_dir": "out",
                "model_path": Path("model.onnx"),
                "cache_dir": Path("cache"),
                "providers": ["CPUExecutionProvider"],
                "precision": "fp32",
                "chunk_size": 44_100,
            }
        )

        self.assertEqual(request.input_path, Path("song.wav"))
        self.assertEqual(request.enrollment_path, Path("reference.wav"))
        self.assertEqual(request.output_dir, Path("out"))
        self.assertEqual(request.model_path, "model.onnx")
        self.assertEqual(request.cache_dir, "cache")
        self.assertEqual(request.providers, ("CPUExecutionProvider",))
        self.assertEqual(request.precision, "fp32")
        self.assertEqual(request.chunk_size, 44_100)

    def test_rejects_invalid_providers_and_boolean_chunk_size(self) -> None:
        """曖昧な provider 値と bool を、モデル生成前に明確なエラーで拒否する。"""
        with self.assertRaisesRegex(ValueError, "providers"):
            SeparationRequest.from_metadata({"input_path": "song.wav", "providers": "CPUExecutionProvider"})
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            SeparationRequest.from_metadata({"input_path": "song.wav", "chunk_size": True})


if __name__ == "__main__":
    unittest.main()
