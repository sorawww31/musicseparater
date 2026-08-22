"""backend/inference.py: BS PolarFormer を CLI とバックエンド呼び出しから実行する。"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

from config import BS_POLARFORMER
from src.models import BSPolarFormer

MODEL_ID = "bs-polarformer"

# ONNX session は重みと CUDA arena を保持するため、プロセス内では一つだけ再利用する。
# 同時に複数の曲を GPU へ送ると activation が重なって OOM になるため、実行も直列化する。
_SEPARATION_LOCK = Lock()


@lru_cache(maxsize=1)
def _separator_for(
    model_path: str | None,
    cache_dir: str | None,
    providers: tuple[str, ...],
    precision: str,
    chunk_size: int | None,
) -> BSPolarFormer:
    """同一設定の separator を返す。キャッシュは VRAM を一つのモデルに限定する。"""
    return BSPolarFormer(
        model_path=model_path,
        cache_dir=cache_dir,
        providers=list(providers) or None,
        precision=precision,
        chunk_size=chunk_size,
    )


def inference(model_id: str, meta_data: Mapping[str, Any]) -> dict[str, str]:
    """指定モデルで分離し、`{"vocals": path, "instrumental": path}` を返す。"""
    if model_id != MODEL_ID:
        raise ValueError(f"未対応のモデルです: {model_id}. 現在は {MODEL_ID} のみ対応しています")

    input_path = meta_data.get("input_path", meta_data.get("audio_path"))
    if not input_path:
        raise ValueError("meta_data に input_path を指定してください")
    output_dir = meta_data.get("output_dir", BS_POLARFORMER.output_directory)
    providers = meta_data.get("providers")
    if providers is not None and (not isinstance(providers, list) or not all(isinstance(p, str) for p in providers)):
        raise ValueError("providers は ONNX Runtime provider 名の文字列リストにしてください")
    precision = meta_data.get("precision", BS_POLARFORMER.default_precision)
    if precision not in BS_POLARFORMER.supported_precisions:
        supported = ", ".join(BS_POLARFORMER.supported_precisions)
        raise ValueError(f"precision は {supported} のいずれかを指定してください: {precision}")
    chunk_size = meta_data.get("chunk_size")
    if chunk_size is not None and (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < BS_POLARFORMER.win_length
    ):
        raise ValueError(f"chunk_size は {BS_POLARFORMER.win_length} 以上の整数にしてください")

    # 生成・ロードも lock 内に置き、同時リクエストが別 session を確保することを防ぐ。
    with _SEPARATION_LOCK:
        separator = _separator_for(
            str(meta_data["model_path"]) if meta_data.get("model_path") else None,
            str(meta_data["cache_dir"]) if meta_data.get("cache_dir") else None,
            tuple(providers or ()),
            precision,
            chunk_size,
        )
        return separator.separate_file(Path(input_path), Path(output_dir))


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
