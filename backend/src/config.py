"""backend/src/config.py: モデル実行と音声保存の調整値を一箇所に集約する。"""

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


@dataclass(frozen=True)
class SeparationRuntimeConfig:
    """HTTP リクエストから隠し、サーバー側で管理する推論設定。"""

    model_path: str | None = None
    cache_dir: str | None = None
    providers: tuple[str, ...] = ()
    precision: str = "fp16"
    chunk_size: int | None = None


@dataclass(frozen=True)
class AudioStorageConfig:
    """アップロード音声と分離結果の保存場所を定義する。"""

    root_directory: Path = Path("audio")
    upload_chunk_size: int = 1024 * 1024


@dataclass(frozen=True)
class MultiSingerAudioConfig:
    """複数歌声モデルに共通する音声と長尺処理の固定値。"""

    channels: int = 2
    silence_epsilon: float = 1e-8


@dataclass(frozen=True)
class UNMIXXConfig:
    """公式コードと checkpoint を再現可能な revision へ固定する。"""

    repo_id: str = "jihoojung0106/unmixx"
    revision: str = "8e750521b5942f4717656cac86a23cf0bd90dea5"
    source_directory: Path = Path("/opt/model-sources/unmixx")
    checkpoint_filename: str = "ckpt/best.ckpt"
    checkpoint_sha256: str = "edd8bd0782f85ebb8a6b26bec47808b3130d15c2ccd49c741d2ff5983cb6c359"
    precision: str = "fp32"
    sample_rate: int = 24_000
    chunk_size: int = 96_000
    hop_size: int = 48_000
    stems: tuple[str, ...] = ("singer_1", "singer_2")


@dataclass(frozen=True)
class SepACapConfig:
    """SepACap の公式 Hugging Face 成果物と声部順を固定する。"""

    repo_id: str = "Tino3141/sepacap"
    revision: str = "15aa140133f29e4358900f8ab0f3a3732c450fc5"
    source_directory: Path = Path("/opt/model-sources/sepacap")
    checkpoint_filename: str = "SepACap.pth"
    config_filename: str = "modelMusicSep.yaml"
    checkpoint_sha256: str = "423b2c7225dfe7b14e7370a651959c09aaa3301f60483b5548686341b2eacc43"
    precision: str = "bf16"
    sample_rate: int = 24_000
    chunk_size: int = 96_000
    hop_size: int = 48_000
    stems: tuple[str, ...] = (
        "alto", "bass", "finger_snap", "lead_vocal", "soprano", "tenor", "vocal_percussion",
    )


@dataclass(frozen=True)
class JaCappellaDPTNetConfig:
    """jaCappella 公式 DPTNet の重みと声部順を固定 revision へ固定する。"""

    repo_id: str = "jaCappella/DPTNet_jaCappella_VES_48k"
    revision: str = "00e6e6207007fdc87f465e2a823872ee87a4c351"
    # 公式の学習コードは asteroid の fork。モデル本体は upstream 0.6.1dev と同一で、
    # 差分は jaCappella 用 dataset と parser だけなので MedleyVox とソースを共有する。
    source_directory: Path = Path("/opt/model-sources/asteroid-jacappella")
    filterbank_directory: Path = Path("/opt/model-sources/asteroid-filterbanks")
    checkpoint_filename: str = "best_model.pth"
    checkpoint_sha256: str = "2d0738ae01145bdf074e8e3a2312f1a21ff2e0c96f2a4f42b1cd0d2c7f4780ac"
    precision: str = "fp32"
    sample_rate: int = 48_000
    # 学習時の seq_dur 5.046 秒とちょうど同じ 242,208 サンプルで推論し、半分を重ねる。
    chunk_size: int = 242_208
    hop_size: int = 121_104
    # conf.yml の data.sources と同じ並び。出力indexと声部の対応はこの順序でしか正しくない。
    stems: tuple[str, ...] = (
        "vocal_percussion", "bass", "alto", "tenor", "soprano", "lead_vocal",
    )


@dataclass(frozen=True)
class MedleyVoxConfig:
    """非公式の再学習 checkpoint と、その vocals.json が持つ構成を固定する。"""

    # 著者は重みを公開しておらず、これは cc-by-4.0 で配布された第三者の再学習版。
    repo_id: str = "Cyru5/MedleyVox"
    revision: str = "5c9e4e0d909e5a006c992b3422901ed416f4e57f"
    source_directory: Path = Path("/opt/model-sources/medleyvox")
    # upstream は asteroid==0.6.1dev を要求する。jaCappella fork の asteroid 本体が
    # まさにその版なので、同じソースを使い回す。
    asteroid_directory: Path = Path("/opt/model-sources/asteroid-jacappella")
    filterbank_directory: Path = Path("/opt/model-sources/asteroid-filterbanks")
    # 同リポジトリの中で学習が最も進んだ iSRNet 付き fine-tuning 版を使う。
    checkpoint_filename: str = "singing_librispeech_ft_iSRNet/vocals.pth"
    config_filename: str = "singing_librispeech_ft_iSRNet/vocals.json"
    checkpoint_sha256: str = "a46b206f4185cd01639cb7ce791a4d50bae2833b02483e6e3de85bc03938e820"
    precision: str = "fp32"
    sample_rate: int = 24_000
    # 学習時の seq_dur 3.0 秒に合わせ、72,000 サンプル単位で半分を重ねる。
    chunk_size: int = 72_000
    hop_size: int = 36_000
    # 公式 inference と同じ入力ラウドネス。商用ミックスの音量をそのまま入れると
    # 出力が実測で1/5まで痩せるため、正規化してから推論し、出力で利得を戻す。
    target_lufs: float = -24.0
    # n_src=2 固定のduetモデル。3人以上は分けられず、どちらが誰かも決まらない。
    stems: tuple[str, ...] = ("singer_1", "singer_2")


@dataclass(frozen=True)
class SingerInformedCheckpoint:
    """論文が公開するConcatenation系checkpointを1条件ずつ指す。"""

    # λはdual lossの学習時重み。推論時に変えられる値ではなく、重みの選択肢。
    conditioning_lambda: str
    path: Path
    sha256: str


# 4条件はいずれも同一アーキテクチャで、strict=True ロードを実機で確認済み。
SINGER_INFORMED_CHECKPOINTS: tuple[SingerInformedCheckpoint, ...] = (
    SingerInformedCheckpoint(
        "none",
        Path("/opt/model-checkpoints/singer-informed/concat-none/vocals.pth"),
        "ec69648c9b1628bfb836072be2d66ad341fe23a87ad09934cc6e7edbb837b674",
    ),
    SingerInformedCheckpoint(
        "0.05",
        Path("/opt/model-checkpoints/singer-informed/concat-0.05/vocals.pth"),
        "e38c0312577d1f345fe1fbe6baa525398df5ce7b7ab8cea77dc0ba015f591011",
    ),
    SingerInformedCheckpoint(
        "0.1",
        Path("/opt/model-checkpoints/singer-informed/concat-0.1/vocals.pth"),
        "691e52121e63529d29ea7d06dd4ce08e91789d47fe5ffeed1782a83c14c4ed48",
    ),
    SingerInformedCheckpoint(
        "0.2",
        Path("/opt/model-checkpoints/singer-informed/concat-0.2/vocals.pth"),
        "39d5e7a61b5f8470d47233badd4220fda6f617d6bf6b697adcf52f41b936c400",
    ),
)


@dataclass(frozen=True)
class SingerInformedConfig:
    """論文の公開モデルと singer embedding の推論契約を固定する。"""

    paper_repo_revision: str = "71f365be86fca580e208146b767a3cd24f5c5f0d"
    checkpoints: tuple[SingerInformedCheckpoint, ...] = SINGER_INFORMED_CHECKPOINTS
    # 論文でduetのtarget SI-SDRが最良だった条件を既定にする。
    default_conditioning_lambda: str = "0.1"
    embedding_checkpoint_path: Path = Path(
        "/opt/model-checkpoints/singer-informed/clean-embedding.pth"
    )
    embedding_checkpoint_sha256: str = (
        "5f52e199edc0a753378bfbf335408dd3044828c9989684f602765e7169fbc1fc"
    )
    sample_rate: int = 44_100
    channels: int = 1
    n_fft: int = 4_096
    hop_length: int = 1_024
    max_bin: int = 1_487
    hidden_size: int = 512
    lstm_layers: int = 3
    lstm_dropout: float = 0.4
    embedding_size: int = 32
    embedding_layers: int = 2
    embedding_n_fft: int = 1_024
    embedding_hop_length: int = 256
    enrollment_seconds: float = 3.0
    enrollment_search_hop_size: int = 4_410
    silence_epsilon: float = 1e-8
    precision: str = "fp32"
    stems: tuple[str, ...] = ("target_vocal", "residual")
    # 論文は混合曲を直接入力するが、16 kHz帯域・3層LSTMのOpen-Unmixでは伴奏が取り切れない。
    # 伴奏除去はBS PolarFormerに任せ、条件付きモデルには歌声だけを渡す。
    cascade_vocal_extraction: bool = True
    # clean embeddingは伴奏なしの歌声で学習されているため、参照音声も同じ前段へ通す。
    cascade_enrollment_extraction: bool = True
    # 単一区間のembeddingは1フレーズの癖に偏るため、重複しない上位区間を平均する。
    enrollment_segment_count: int = 5
    # ratio maskの鋭さ。1.0は従来の直接出力と一致し、2.0はWiener相当。
    # 大きいほど伴奏優位のbinを強く抑えるが、3.0を超えると musical noise が出やすい。
    mask_exponent: float = 2.0
    # ratio maskの下限。これを下回るbinは伴奏とみなして落とす。0.0で無効。
    mask_floor: float = 0.0

    @property
    def conditioning_lambdas(self) -> tuple[str, ...]:
        """APIとUIが受理するλの一覧を返す。"""
        return tuple(checkpoint.conditioning_lambda for checkpoint in self.checkpoints)

    def checkpoint_for(self, conditioning_lambda: str | None) -> SingerInformedCheckpoint:
        """λに対応する公開checkpointを返し、未公開の条件は実行前に拒否する。"""
        label = conditioning_lambda or self.default_conditioning_lambda
        for checkpoint in self.checkpoints:
            if checkpoint.conditioning_lambda == label:
                return checkpoint
        supported = ", ".join(self.conditioning_lambdas)
        raise ValueError(
            f"conditioning_lambda は {supported} のいずれかを指定してください: {label}"
        )


@dataclass(frozen=True)
class JobRuntimeConfig:
    """ジョブ実行とブラウザ接続に関する調整値。"""

    max_workers: int = 1
    allowed_origins: tuple[str, ...] = ("http://localhost:5173",)
    generic_error: str = "音声分離に失敗しました"
    vocal_extraction_end_percent: int = 40
    singer_separation_end_percent: int = 95
    finalizing_percent: int = 99
    child_log_tail_lines: int = 20


BS_POLARFORMER = BSPolarFormerConfig()
SEPARATION_RUNTIME = SeparationRuntimeConfig()
AUDIO_STORAGE = AudioStorageConfig()
MULTI_SINGER_AUDIO = MultiSingerAudioConfig()
UNMIXX = UNMIXXConfig()
SEPACAP = SepACapConfig()
JACAPPELLA_DPTNET = JaCappellaDPTNetConfig()
MEDLEYVOX = MedleyVoxConfig()
SINGER_INFORMED = SingerInformedConfig()
JOB_RUNTIME = JobRuntimeConfig()
