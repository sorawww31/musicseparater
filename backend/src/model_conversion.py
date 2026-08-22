"""backend/src/model_conversion.py: FP32 ONNX を安全に mixed FP16 へ変換して再利用する。"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import onnx
from onnxconverter_common import float16

# IEEE 754 half precision が表せる範囲。変換器の緩い既定値より丸めを抑える。
_MIN_FP16_POSITIVE = 5.96e-08
_MAX_FP16_FINITE = 65_504.0


def convert_model_to_mixed_fp16(source_path: Path, target_path: Path) -> Path:
    """FP32 入出力を保った mixed FP16 モデルを一度だけ生成して返す。

    ONNX Runtime が CUDA で実行できない演算は FP32 に残す。対象ファイルを最後に置換するため、
    途中で失敗しても既存キャッシュを壊さない。
    """
    if target_path.is_file():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    model: Any = onnx.load_model(str(source_path))
    # 音声前後処理は NumPy / PyTorch で FP32 のため、ONNX の公開 I/O も FP32 に維持する。
    converted: Any = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        min_positive_val=_MIN_FP16_POSITIVE,
        max_finite_val=_MAX_FP16_FINITE,
    )
    # このモデルにはエクスポート時の中間テンソル型注釈が大量に含まれる。変換器は
    # 一部を更新できないため、古い注釈を残すと ONNX Runtime が型不整合として拒否する。
    # 公開 I/O の注釈は別管理なので削除されない。
    del converted.graph.value_info[:]
    onnx.checker.check_model(converted)

    with NamedTemporaryFile(dir=target_path.parent, suffix=".onnx", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        onnx.save_model(converted, str(temporary_path))
        temporary_path.replace(target_path)
    finally:
        # save_model の失敗時だけ一時ファイルを回収する。正常時は replace 済みで存在しない。
        if temporary_path.exists():
            temporary_path.unlink()

    return target_path
