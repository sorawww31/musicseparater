"""backend/tests/test_model_conversion.py: mixed FP16 変換のキャッシュ契約を確認する。"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_conversion import convert_model_to_mixed_fp16  # noqa: E402


class ModelConversionTests(unittest.TestCase):
    """重い ONNX 変換を行わず、変換器への受け渡しとキャッシュを検証する。"""

    def test_converts_once_and_preserves_fp32_io(self) -> None:
        """変換済みキャッシュがなければ、FP32 I/O 指定で変換して保存する。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "source.onnx"
            target_path = temporary_path / "cache" / "source-fp16.onnx"
            source_path.touch()

            model = object()
            converted = SimpleNamespace(graph=SimpleNamespace(value_info=[object()]))
            converter = Mock(return_value=converted)

            def save_model(saved_model: object, destination: str) -> None:
                self.assertIs(saved_model, converted)
                Path(destination).write_bytes(b"converted")

            fake_onnx = SimpleNamespace(
                load_model=Mock(return_value=model),
                save_model=save_model,
                checker=SimpleNamespace(check_model=Mock()),
            )
            fake_converter_common = SimpleNamespace(float16=SimpleNamespace(convert_float_to_float16=converter))
            with patch.dict(
                sys.modules,
                {"onnx": fake_onnx, "onnxconverter_common": fake_converter_common},
            ):
                actual = convert_model_to_mixed_fp16(source_path, target_path)

            self.assertEqual(actual, target_path)
            self.assertEqual(target_path.read_bytes(), b"converted")
            fake_onnx.load_model.assert_called_once_with(str(source_path))
            converter.assert_called_once_with(
                model,
                keep_io_types=True,
                min_positive_val=5.96e-08,
                max_finite_val=65_504.0,
            )
            self.assertEqual(converted.graph.value_info, [])
            fake_onnx.checker.check_model.assert_called_once_with(converted)

    def test_reuses_existing_conversion_without_importing_dependencies(self) -> None:
        """変換済みファイルがあれば、依存関係なしでそのまま再利用する。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "source.onnx"
            target_path = temporary_path / "source-fp16.onnx"
            source_path.touch()
            target_path.write_bytes(b"cached")

            self.assertEqual(convert_model_to_mixed_fp16(source_path, target_path), target_path)
            self.assertEqual(target_path.read_bytes(), b"cached")
