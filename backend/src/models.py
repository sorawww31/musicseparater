"""backend/src/models.py: ONNX BS PolarFormer を安全に実行するモデル層。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import BS_POLARFORMER, BSPolarFormerConfig
from src.model_conversion import convert_model_to_mixed_fp16


@dataclass(frozen=True)
class ModelMetadata:
    """呼び出し元に公開するモデルの音声仕様。"""

    sample_rate: int
    stems: tuple[str, ...]
    channels: int


@dataclass
class AudioData:
    """チャンネル先行 `(channels, samples)` の float32 音声。"""

    waveform: Any
    sample_rate: int


@dataclass
class SeparationResult:
    """分離済みステムと、そのサンプルレート。"""

    stems: dict[str, Any]
    sample_rate: int


@dataclass
class _StftData:
    """ONNX 入力と、iSTFT のために保持する複素 STFT。"""

    features: Any
    stft: Any
    window: Any
    length: int


class BaseSeparator(ABC):
    """モデル固有のロード・推論処理を分離する最小インターフェース。"""

    metadata: ModelMetadata

    @abstractmethod
    def load(self) -> None:
        """必要な重みと推論セッションを準備する。"""

    @abstractmethod
    def separate(self, audio: AudioData) -> SeparationResult:
        """入力音声をモデルが扱うステムへ分離する。"""


class BSPolarFormer(BaseSeparator):
    """bgkb/bs_polarformer の ONNX ボーカル分離実装。"""

    def __init__(
        self,
        config: BSPolarFormerConfig = BS_POLARFORMER,
        *,
        model_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        providers: list[str] | None = None,
        precision: str | None = None,
        chunk_size: int | None = None,
    ) -> None:
        self.config = config
        self.model_path = Path(model_path) if model_path else None
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.providers = providers
        # FP16 は CUDA 向けの mixed precision モデルを生成する設定である。STFT と
        # mask の公開入出力は FP32 の契約を保つため、半精度に変換してはならない。
        self.precision = precision or config.default_precision
        self.config.model_filename_for(self.precision)
        self.chunk_size = chunk_size or config.chunk_size
        self.session: Any | None = None
        self._uses_cuda = False
        self.metadata = ModelMetadata(config.sample_rate, ("vocals", "instrumental"), config.channels)

    def load(self) -> None:
        """モデルを必要時だけ Hugging Face から取得し、ONNX Runtime を起動する。"""
        if self.session is not None:
            return
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("依存関係を導入してください: uv sync") from exc

        model_path = self.model_path
        downloaded_from_hub = model_path is None
        if model_path is None:
            model_path = Path(hf_hub_download(
                repo_id=self.config.repo_id,
                filename=self.config.model_filename_for(self.precision),
                cache_dir=self.cache_dir,
            ))
        if not model_path.is_file():
            raise FileNotFoundError(f"BS PolarFormer の ONNX モデルが見つかりません: {model_path}")

        available = ort.get_available_providers()
        requested = self.providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        selected = [provider for provider in requested if provider in available]
        if not selected:
            raise RuntimeError(f"利用可能な ONNX Runtime provider がありません: {available}")

        # ONNX Runtime の CPU EP は FP16 演算を実行できない。CPU 実行時は元の FP32
        # モデルを使い、従来どおり安全に分離できるようにする。
        # ローカルの --model-path は従来どおり FP32 ONNX を想定し、その場合だけ変換する。
        # 公式配布の FP16 は変換不要なので、丸め warning と起動時間を発生させない。
        if self.precision == "fp16" and not downloaded_from_hub and "CUDAExecutionProvider" in selected:
            target_path = self.config.converted_fp16_model_path(model_path, self.cache_dir)
            model_path = convert_model_to_mixed_fp16(model_path, target_path)
        configured_providers: list[Any] = []
        for provider in selected:
            if provider == "CUDAExecutionProvider":
                configured_providers.append(
                    (
                        provider,
                        {
                            "device_id": self.config.cuda_device_id,
                            "arena_extend_strategy": self.config.cuda_arena_extend_strategy,
                            "do_copy_in_default_stream": True,
                        },
                    )
                )
            else:
                configured_providers.append(provider)
        session_options = ort.SessionOptions()
        session_options.log_severity_level = self.config.onnxruntime_log_severity
        self.session = ort.InferenceSession(
            str(model_path), session_options, providers=configured_providers,
        )
        self._uses_cuda = "CUDAExecutionProvider" in self.session.get_providers()
        if "CUDAExecutionProvider" in selected and not self._uses_cuda:
            raise RuntimeError("CUDAExecutionProvider の初期化に失敗しました。--cpu を指定するか CUDA 環境を確認してください")
        print(f"ONNX Runtime providers: {', '.join(self.session.get_providers())}")

    def separate(self, audio: AudioData) -> SeparationResult:
        """44.1 kHz ステレオ音声を vocals / instrumental に分離する。"""
        self._require_runtime()
        normalized = self._normalize_audio(audio.waveform)
        vocals = self._separate_chunks(normalized)
        return SeparationResult(
            stems={"vocals": vocals, "instrumental": normalized - vocals},
            sample_rate=self.config.sample_rate,
        )

    def separate_file(self, input_path: str | Path, output_dir: str | Path) -> dict[str, str]:
        """音声ファイルを読み、2 つの WAV ステムを書き出してパスを返す。"""
        self._require_runtime()
        try:
            import librosa
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("依存関係を導入してください: uv sync") from exc

        source = Path(input_path)
        if not source.is_file():
            raise FileNotFoundError(f"入力音声が見つかりません: {source}")
        waveform, _ = librosa.load(source, sr=self.config.sample_rate, mono=False)
        result = self.separate(AudioData(waveform=waveform, sample_rate=self.config.sample_rate))

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        output_paths = {name: destination / f"{name}.wav" for name in self.metadata.stems}
        for stem, path in output_paths.items():
            sf.write(path, result.stems[stem].T, result.sample_rate)
        return {stem: str(path) for stem, path in output_paths.items()}

    def _require_runtime(self) -> None:
        if self.session is None:
            self.load()

    def _normalize_audio(self, waveform: Any) -> Any:
        """mono / 多チャンネル入力をモデル入出力仕様の stereo float32 にそろえる。"""
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("依存関係を導入してください: uv sync") from exc

        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.stack((audio, audio))
        if audio.ndim != 2:
            raise ValueError("音声は (samples,) または (channels, samples) で指定してください")
        if audio.shape[0] < self.config.channels:
            audio = np.repeat(audio, self.config.channels, axis=0)
        return audio[: self.config.channels]

    def _separate_chunks(self, audio: Any) -> Any:
        """オーバーラップ平均で長尺音源を処理する。"""
        import numpy as np

        total_samples = audio.shape[1]
        try:
            from tqdm import tqdm
        except ImportError as exc:
            raise RuntimeError("進捗表示の依存関係を導入してください: uv sync") from exc

        step = self.chunk_size // self.config.overlap_count
        vocals = np.zeros_like(audio, dtype=np.float32)
        counts = np.zeros(total_samples, dtype=np.float32)
        starts = range(0, total_samples, step)
        for start in tqdm(starts, desc="分離", total=len(starts), unit="chunk"):
            end = min(start + self.chunk_size, total_samples)
            chunk = audio[:, start:end]
            # 完全なチャンクまでコピーする必要はない。最後の短いチャンクだけゼロ埋めする。
            padded = chunk
            if chunk.shape[1] < self.chunk_size:
                padded = np.pad(chunk, ((0, 0), (0, self.chunk_size - chunk.shape[1])))
            predicted = self._predict_vocals(padded)
            vocals[:, start:end] += predicted[:, : end - start]
            counts[start:end] += 1
        return vocals / np.maximum(counts, 1)[None, :]

    def _predict_vocals(self, audio: Any) -> Any:
        """STFT → ONNX mask → iSTFT というモデルカード指定の経路を実行する。"""
        device = self._cuda_device()
        prepared = self._prepare_stft(audio, device)
        if device is None:
            mask = self.session.run(None, {"stft_features": prepared.features})[0]
        else:
            mask = self._run_cuda_with_iobinding(prepared.features, device)
        return self._reconstruct_audio(prepared, mask)

    def _cuda_device(self) -> Any | None:
        """ORT と PyTorch の両方で CUDA を使える場合だけ、GPU 前後処理を有効にする。"""
        if not self._uses_cuda:
            return None
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.device("cuda", self.config.cuda_device_id)

    def _prepare_stft(self, audio: Any, device: Any | None) -> _StftData:
        import torch

        # mixed FP16 モデルも ONNX 入出力は FP32 固定。半精度の特徴量を渡すと
        # 入力契約に合わないため、常に float32 を生成する。
        raw = torch.from_numpy(audio).to(device=device, dtype=torch.float32)
        window = torch.hann_window(self.config.win_length, device=raw.device, dtype=raw.dtype)
        stft = torch.stft(raw, n_fft=self.config.n_fft, hop_length=self.config.hop_length,
                          win_length=self.config.win_length, window=window, return_complex=True)
        stft_features = torch.view_as_real(stft).permute(1, 0, 2, 3).reshape(1, -1, stft.shape[-1], 2)
        features = stft_features.permute(0, 2, 1, 3).reshape(1, stft.shape[-1], -1).contiguous()
        if device is None:
            features = features.numpy()
        return _StftData(features=features, stft=stft_features, window=window, length=audio.shape[-1])

    def _run_cuda_with_iobinding(self, features: Any, device: Any) -> Any:
        """GPU tensor を ORT の入出力へ直接 bind し、STFT mask の host 往復を避ける。"""
        import numpy as np
        import torch

        # PyTorch の STFT 完了後に ORT が同じ buffer を読むよう同期する。曲ごとに直列化して
        # いるため、この保守的な同期でも並列実行との競合は発生しない。
        torch.cuda.synchronize(device)
        features = features.contiguous()
        frame_count = features.shape[1]
        mask_shape = (1, 1, (self.config.n_fft // 2 + 1) * self.config.channels, frame_count, 2)
        mask = torch.empty(mask_shape, device=device, dtype=torch.float32)
        binding = self.session.io_binding()
        binding.bind_input(
            name="stft_features",
            device_type="cuda",
            device_id=self.config.cuda_device_id,
            element_type=np.float32,
            shape=tuple(features.shape),
            buffer_ptr=features.data_ptr(),
        )
        binding.bind_output(
            name="mask",
            device_type="cuda",
            device_id=self.config.cuda_device_id,
            element_type=np.float32,
            shape=mask_shape,
            buffer_ptr=mask.data_ptr(),
        )
        self.session.run_with_iobinding(binding)
        # ORT 完了後に PyTorch iSTFT が mask を読むため、別 stream 間の同期を明示する。
        torch.cuda.synchronize(device)
        return mask

    def _reconstruct_audio(self, prepared: _StftData, mask: Any) -> Any:
        import torch

        stft = torch.view_as_complex(prepared.stft.unsqueeze(1).contiguous())
        mask_tensor = mask if isinstance(mask, torch.Tensor) else torch.from_numpy(mask)
        masked = stft * torch.view_as_complex(mask_tensor.contiguous())
        masked = masked.reshape(1, 1, -1, self.config.channels, masked.shape[-1])
        masked = masked.permute(0, 1, 3, 2, 4).reshape(self.config.channels, -1, masked.shape[-1])
        masked[:, 0, :] = 0  # 参照実装と同じく DC 成分を除く。
        reconstructed = torch.istft(masked, n_fft=self.config.n_fft, hop_length=self.config.hop_length,
                                    win_length=self.config.win_length, window=prepared.window,
                                    length=prepared.length)
        return reconstructed.cpu().numpy()
