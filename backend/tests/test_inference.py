"""backend/tests/test_inference.py: 公開する BS PolarFormer 呼び出し契約を確認する。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inference  # noqa: E402


class InferenceContractTests(unittest.TestCase):
    """重いモデルを取得せずに入力検証と出力契約をテストする。"""

    def setUp(self) -> None:
        """テスト間でプロセス内モデルキャッシュを共有しない。"""
        inference._separator_for.cache_clear()

    def tearDown(self) -> None:
        inference._separator_for.cache_clear()

    def test_bs_polarformer_delegates_to_separator(self) -> None:
        expected = {"vocals": "out/vocals.wav", "instrumental": "out/instrumental.wav"}
        with patch.object(inference, "BSPolarFormer") as separator_type:
            separator_type.return_value.separate_file.return_value = expected
            actual = inference.inference(
                "bs-polarformer",
                {
                    "input_path": "song.wav",
                    "output_dir": "out",
                    "providers": ["CPUExecutionProvider"],
                    "precision": "fp32",
                },
            )
        self.assertEqual(actual, expected)
        separator_type.assert_called_once_with(
            model_path=None,
            cache_dir=None,
            providers=["CPUExecutionProvider"],
            precision="fp32",
            chunk_size=None,
        )
        separator_type.return_value.separate_file.assert_called_once_with(Path("song.wav"), Path("out"))

    def test_rejects_unknown_model_and_missing_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "未対応"):
            inference.inference("htdemucs", {"input_path": "song.wav"})
        with self.assertRaisesRegex(ValueError, "input_path"):
            inference.inference("bs-polarformer", {})

    def test_rejects_unsupported_precision(self) -> None:
        """モデルカードにない精度をダウンロード前に拒否する。"""
        with self.assertRaisesRegex(ValueError, "precision"):
            inference.inference("bs-polarformer", {"input_path": "song.wav", "precision": "int8"})

    def test_reuses_one_separator_for_repeated_requests(self) -> None:
        """同じ設定の連続呼び出しでは、重い ONNX session を作り直さない。"""
        with patch.object(inference, "BSPolarFormer") as separator_type:
            separator_type.return_value.separate_file.return_value = {"vocals": "out/vocals.wav"}
            request = {"input_path": "song.wav", "output_dir": "out", "precision": "fp16"}
            inference.inference("bs-polarformer", request)
            inference.inference("bs-polarformer", request)

        separator_type.assert_called_once_with(
            model_path=None,
            cache_dir=None,
            providers=None,
            precision="fp16",
            chunk_size=None,
        )
        self.assertEqual(separator_type.return_value.separate_file.call_count, 2)

    def test_accepts_chunk_size_as_a_separate_session_configuration(self) -> None:
        """異なるチャンク長は別 separator にして、安全に VRAM 設定を反映する。"""
        with patch.object(inference, "BSPolarFormer") as separator_type:
            separator_type.return_value.separate_file.return_value = {"vocals": "out/vocals.wav"}
            inference.inference("bs-polarformer", {"input_path": "song.wav", "chunk_size": 44_100})

        separator_type.assert_called_once_with(
            model_path=None,
            cache_dir=None,
            providers=None,
            precision="fp16",
            chunk_size=44_100,
        )

    def test_rejects_invalid_chunk_size(self) -> None:
        """STFT 窓より小さいチャンクは実行前に拒否する。"""
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            inference.inference("bs-polarformer", {"input_path": "song.wav", "chunk_size": 1})


if __name__ == "__main__":
    unittest.main()
