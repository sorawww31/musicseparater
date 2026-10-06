"""backend/src/inference.py: モデル契約に従って各分離処理を直列実行する。"""

from __future__ import annotations

import argparse
import gc
from collections.abc import Mapping
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from threading import Lock

from .config import BS_POLARFORMER, JOB_RUNTIME
from .model_catalog import validate_model_request
from .models import BSPolarFormer
from .multi_singer import separate_multi_singer
from .separation_request import SeparationRequest
from .singer_informed import separate_target_singer

MODEL_ID = "bs-polarformer"
JobProgress = Callable[[str, int], None]

# ONNX session は重みと CUDA arena を保持するため、プロセス内では一つだけ再利用する。
# 同時に複数の曲を GPU へ送ると activation が重なって OOM になるため、実行も直列化する。
_SEPARATION_LOCK = Lock()


@lru_cache(maxsize=1)
def _separator_for(
    model_id: str,
    model_path: str | None,
    cache_dir: str | None,
    providers: tuple[str, ...],
    precision: str,
    chunk_size: int | None,
) -> BSPolarFormer:
    """同一設定の separator を返す。キャッシュは VRAM を一つのモデルに限定する。"""
    if model_id != MODEL_ID:
        raise ValueError(f"未対応のモデルです: {model_id}")
    return BSPolarFormer(
        model_path=model_path,
        cache_dir=cache_dir,
        providers=list(providers) or None,
        precision=precision,
        chunk_size=chunk_size,
    )


def unload_separator(separator: BSPolarFormer | None = None) -> None:
    """PyTorchモデル起動前にONNX sessionとCUDAキャッシュを解放する。"""
    if separator is not None:
        separator.session = None
    _separator_for.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def inference(
    model_id: str,
    meta_data: Mapping[str, object],
    num_vocals: int | str | None = None,
    progress: JobProgress | None = None,
) -> dict[str, str]:
    """指定モデルを直列実行し、ステム名とWAVパスを返す。"""
    if isinstance(num_vocals, str):
        raise ValueError("num_vocals は整数で指定してください")
    definition = validate_model_request(model_id, num_vocals)

    request = SeparationRequest.from_metadata(meta_data)

    # 生成・ロードも lock 内に置き、同時リクエストが別 session を確保することを防ぐ。
    with _SEPARATION_LOCK:
        def extract_vocals(
            input_path: Path, output_dir: Path, chunk_progress: Callable[[int, int], None],
        ) -> dict[str, str]:
            separator = _separator_for(
                MODEL_ID,
                request.model_path,
                request.cache_dir,
                request.providers,
                request.precision,
                request.chunk_size,
            )
            try:
                return separator.separate_file(input_path, output_dir, chunk_progress)
            finally:
                unload_separator(separator)

        if definition.requires_enrollment:
            if request.enrollment_path is None:
                raise ValueError("singer-informed には enrollment_path が必要です")
            return separate_target_singer(
                request,
                extract_vocals,
                progress or (lambda _phase, _percent: None),
            )
        if definition.multi_singer:
            return separate_multi_singer(
                model_id, request, extract_vocals, progress or (lambda _phase, _percent: None),
            )
        separator = _separator_for(
            MODEL_ID,
            request.model_path,
            request.cache_dir,
            request.providers,
            request.precision,
            request.chunk_size,
        )
        chunk_progress = None
        if progress is not None:
            chunk_progress = lambda completed, total: progress(
                "extracting_vocals",
                round(JOB_RUNTIME.singer_separation_end_percent * completed / total),
            )
        if chunk_progress is None:
            return separator.separate_file(request.input_path, request.output_dir)
        result = separator.separate_file(request.input_path, request.output_dir, chunk_progress)
        progress("finalizing", JOB_RUNTIME.finalizing_percent)
        return result


def main() -> None:
    """ローカル実行用 CLI。モデルは初回だけ Hugging Face キャッシュへ取得される。"""
    parser = argparse.ArgumentParser(description="BS PolarFormer で vocals / instrumental を分離します")
    parser.add_argument("input_path", help="入力音声ファイル（mp3 / wav / flac 等）")
    parser.add_argument("--output-dir", default=str(BS_POLARFORMER.output_directory), help="WAV 出力先")
    parser.add_argument("--model-path", help="ダウンロード済み ONNX ファイルを使う場合のパス")
    parser.add_argument("--cache-dir", help="Hugging Face モデルキャッシュの保存先")
    parser.add_argument("--cpu", action="store_true", help="GPU を使わず CPUExecutionProvider のみで実行")
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="1 チャンクのサンプル数（小さいほど peak VRAM は下がるが、処理時間は増える）",
    )
    parser.add_argument(
        "--precision",
        choices=BS_POLARFORMER.supported_precisions,
        default=BS_POLARFORMER.default_precision,
        help="CUDA 推論時の ONNX 演算精度（既定: fp16。CPU は fp32 で実行）",
    )
    args = parser.parse_args()

    output_paths = inference(
        MODEL_ID,
        {
            "input_path": args.input_path,
            "output_dir": args.output_dir,
            "model_path": args.model_path,
            "cache_dir": args.cache_dir,
            "providers": ["CPUExecutionProvider"] if args.cpu else None,
            "precision": args.precision,
            "chunk_size": args.chunk_size,
        },
    )
    for stem, path in output_paths.items():
        print(f"{stem}: {path}")


if __name__ == "__main__":
    main()
