"""backend/tests/test_models.py: 実行 provider ごとのモデル精度選択を確認する。"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import BSPolarFormer  # noqa: E402


class BSPolarFormerLoadTests(unittest.TestCase):
    """実モデルをロードせず、CUDA と CPU の精度選択を検証する。"""

    def test_load_converts_fp32_source_to_mixed_fp16_for_cuda(self) -> None:
        """CUDA 利用時は FP32 元モデルから生成した FP16 モデルをセッションへ渡す。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "source.onnx"
            converted_path = Path(temporary_directory) / "converted.onnx"
            source_path.touch()

            session_type = Mock()
            session_type.return_value.get_providers.return_value = ["CUDAExecutionProvider"]
            session_options = Mock()
            fake_ort = SimpleNamespace(
                get_available_providers=Mock(return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]),
                InferenceSession=session_type,
                SessionOptions=Mock(return_value=session_options),
            )
            fake_hub = SimpleNamespace(hf_hub_download=Mock())
            separator = BSPolarFormer(model_path=source_path, providers=["CUDAExecutionProvider"])

            with patch.dict(sys.modules, {"onnxruntime": fake_ort, "huggingface_hub": fake_hub}):
                with patch("src.models.convert_model_to_mixed_fp16", return_value=converted_path) as converter:
                    separator.load()

            converter.assert_called_once_with(
                source_path,
                separator.config.converted_fp16_model_path(source_path, None),
            )
            session_type.assert_called_once_with(
                str(converted_path),
                session_options,
                providers=[
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": separator.config.cuda_device_id,
                            "arena_extend_strategy": separator.config.cuda_arena_extend_strategy,
                            "do_copy_in_default_stream": True,
                        },
                    ),
                ],
            )
            self.assertEqual(session_options.log_severity_level, separator.config.onnxruntime_log_severity)
            fake_hub.hf_hub_download.assert_not_called()

    def test_load_downloads_official_fp16_model_without_conversion(self) -> None:
        """既定 FP16 では公式配布物をそのまま使い、ローカル変換を避ける。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            fp16_path = Path(temporary_directory) / "bs_polarformer_fp16.onnx"
            fp16_path.touch()

            session_type = Mock()
            session_type.return_value.get_providers.return_value = ["CUDAExecutionProvider"]
            fake_ort = SimpleNamespace(
                get_available_providers=Mock(return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]),
                InferenceSession=session_type,
                SessionOptions=Mock(return_value=Mock()),
            )
            fake_hub = SimpleNamespace(hf_hub_download=Mock(return_value=str(fp16_path)))
            separator = BSPolarFormer(providers=["CUDAExecutionProvider"])

            with patch.dict(sys.modules, {"onnxruntime": fake_ort, "huggingface_hub": fake_hub}):
                with patch("src.models.convert_model_to_mixed_fp16") as converter:
                    separator.load()

            converter.assert_not_called()
            fake_hub.hf_hub_download.assert_called_once_with(
                repo_id=separator.config.repo_id,
                filename=separator.config.fp16_model_filename,
                cache_dir=None,
            )

    def test_load_keeps_fp32_source_for_cpu(self) -> None:
        """CPU provider は FP16 演算非対応のため、変換せず元モデルを実行する。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "source.onnx"
            source_path.touch()

            session_type = Mock()
            session_type.return_value.get_providers.return_value = ["CPUExecutionProvider"]
            session_options = Mock()
            fake_ort = SimpleNamespace(
                get_available_providers=Mock(return_value=["CPUExecutionProvider"]),
                InferenceSession=session_type,
                SessionOptions=Mock(return_value=session_options),
            )
            fake_hub = SimpleNamespace(hf_hub_download=Mock())
            separator = BSPolarFormer(model_path=source_path, providers=["CPUExecutionProvider"])

            with patch.dict(sys.modules, {"onnxruntime": fake_ort, "huggingface_hub": fake_hub}):
                with patch("src.models.convert_model_to_mixed_fp16") as converter:
                    separator.load()

            converter.assert_not_called()
            session_type.assert_called_once_with(str(source_path), session_options, providers=["CPUExecutionProvider"])

    def test_chunk_progress_uses_the_configured_chunk_size(self) -> None:
        """長尺音声の進捗は、選択したチャンク数を total として表示する。"""
        import numpy as np

        separator = BSPolarFormer(chunk_size=4)
        separator._predict_vocals = Mock(side_effect=lambda audio: audio)
        progress = Mock(side_effect=lambda iterable, **_kwargs: iterable)
        fake_tqdm = SimpleNamespace(tqdm=progress)

        with patch.dict(sys.modules, {"tqdm": fake_tqdm}):
            result = separator._separate_chunks(np.ones((2, 5), dtype=np.float32))

        np.testing.assert_allclose(result, np.ones((2, 5), dtype=np.float32))
        progress.assert_called_once_with(range(0, 5, 2), desc="分離", total=3, unit="chunk")
