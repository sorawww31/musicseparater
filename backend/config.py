"""backend/config.py: BS PolarFormer の実行時設定を一箇所に集約する。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BSPolarFormerConfig:
    """モデルカードで公開されている ONNX 推論用の固定パラメータ。"""

    repo_id: str = "bgkb/bs_polarformer"
    # Hugging Face には公式の FP32 / FP16 ONNX がそれぞれ配布されている。
    # FP16 は重みの容量を半減し、FP32 の入出力契約はそのまま維持する。
    default_precision: str = "fp16"
    supported_precisions: tuple[str, ...] = ("fp16", "fp32")
    fp32_model_filename: str = "bs_polarformer.onnx"
    fp16_model_filename: str = "bs_polarformer_fp16.onnx"
    # 変換済みモデルはダウンロード元を変更せず、専用キャッシュに保存する。
    model_cache_directory: Path = Path(".model-cache")
    converted_model_subdirectory: str = "converted"
    # 変換ロジックを更新した場合に、古い生成物を再利用しないためのキャッシュ世代。
    fp16_conversion_revision: str = "v2"
    sample_rate: int = 44_100
    channels: int = 2
    n_fft: int = 2_048
    hop_length: int = 512
    win_length: int = 2_048
    # 10 秒単位にして、公式既定の 20 秒より peak VRAM を抑える。重ねた領域は平均する。
    chunk_size: int = 441_000
    overlap_count: int = 2
    cuda_device_id: int = 0
    # 大きなチャンクを処理した後に CUDA arena が過剰に成長し続けないようにする。
    cuda_arena_extend_strategy: str = "kSameAsRequested"
    # ORT の constant folding は CUDA 対応演算について無害な warning を出すことがある。
    # 通常実行ではエラー以上だけを表示し、CUDA の選択結果はアプリ側で明示する。
    onnxruntime_log_severity: int = 3
    output_directory: Path = Path("output")

    def model_filename_for(self, precision: str) -> str:
        """指定精度で公式配布されている ONNX ファイル名を返す。"""
        if precision == "fp16":
            return self.fp16_model_filename
        if precision == "fp32":
            return self.fp32_model_filename
        supported = ", ".join(self.supported_precisions)
        raise ValueError(f"precision は {supported} のいずれかを指定してください: {precision}")

    def converted_fp16_model_path(self, source_path: Path, cache_dir: str | Path | None) -> Path:
        """変換元の更新を区別できる FP16 キャッシュパスを返す。"""
        source_stat = source_path.stat()
        cache_root = Path(cache_dir) if cache_dir else self.model_cache_directory
        filename = (
            f"{source_path.stem}-{source_stat.st_size}-{source_stat.st_mtime_ns}"
            f"-fp16-{self.fp16_conversion_revision}.onnx"
        )
        return cache_root / self.converted_model_subdirectory / filename


BS_POLARFORMER = BSPolarFormerConfig()
