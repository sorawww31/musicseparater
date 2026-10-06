"""backend/model_runtime/runner.py: 単一モデルをCUDAで実行して即終了する子プロセス。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audio import (
    loudness_normalization_gain,
    normalize_stereo,
    rescale_estimates_to_mixture,
    select_enrollment_segments,
    separate_long_audio,
)
from .loaders import (
    load_jacappella_dptnet,
    load_medleyvox,
    load_sepacap,
    load_singer_informed,
    load_unmixx,
)


def _unmixx_predictor(model: Any, device: Any) -> Any:
    """UNMIXX 出力を `(sources, channels, samples)` へ変換する。"""
    import torch

    @torch.inference_mode()
    def predict(chunk: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(chunk).to(device=device, dtype=torch.float32)
        # upstream forward は batch と channel を平坦化した後の mixture consistency に
        # broadcasting があるため、stereoをbatch化せず1chずつ確実に2出力へ通す。
        channels = [
            model(tensor[index:index + 1].unsqueeze(1))[0][0]
            for index in range(tensor.shape[0])
        ]
        estimates = torch.stack(channels, dim=1)
        return estimates.float().cpu().numpy()

    return predict


def _sepacap_predictor(model: Any, device: Any, source_count: int) -> Any:
    """補助損失用 decoder を通さず SepACap 最終段だけを実行する。"""
    import torch

    @torch.inference_mode()
    def predict(chunk: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(chunk).to(device=device, dtype=torch.bfloat16)
        encoded = model.audio_encoder(tensor)
        projected = model.feature_projector(encoded)
        separated, _ = model.separator(projected)
        masked = model.out_layer(separated, encoded)
        estimates = torch.stack([model.audio_decoder(masked[index]) for index in range(source_count)])
        return estimates[..., : chunk.shape[-1]].float().cpu().numpy()

    return predict


def _monaural_predictor(model: Any, device: Any, source_count: int) -> Any:
    """mono学習のモデルを左右で個別に通し `(sources, channels, samples)` へ積む。"""
    import torch

    @torch.inference_mode()
    def predict(chunk: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(chunk).to(device=device, dtype=torch.float32)
        estimates = torch.stack(
            [model(tensor[index:index + 1].unsqueeze(1))[0] for index in range(tensor.shape[0])],
            dim=1,
        )
        if estimates.shape[0] != source_count:
            raise RuntimeError(f"モデル出力の source 数が設定と一致しません: {estimates.shape[0]}")
        return estimates[..., : chunk.shape[-1]].float().cpu().numpy()

    return predict


def _run_singer_informed(settings: dict[str, Any], device: Any) -> None:
    """参照embeddingでmono楽曲からtarget vocalとresidualを推定する。"""
    import librosa
    import soundfile as sound_file
    import torch

    # 親子は別プロセスなので、APIを再起動せずに更新すると古い設定が渡る。
    # 分かりにくいKeyErrorの代わりに、原因を名指しして即座に止める。
    missing = {"enrollment_segment_count", "mask_exponent", "mask_floor"} - settings.keys()
    if missing:
        raise RuntimeError(
            "runner設定の項目が不足しています。APIプロセスが古いコードを読み込んでいる"
            "可能性があります（backendコンテナを再起動してください）: "
            + ", ".join(sorted(missing))
        )

    model_settings = settings["model"]
    separator, embedding_model = load_singer_informed(model_settings, device)
    enrollment, _ = librosa.load(
        settings["enrollment_path"],
        sr=settings["sample_rate"],
        mono=True,
    )
    enrollment_segments = select_enrollment_segments(
        enrollment,
        sample_rate=settings["sample_rate"],
        duration_seconds=settings["enrollment_seconds"],
        search_hop_size=settings["enrollment_search_hop_size"],
        epsilon=settings["silence_epsilon"],
        segment_count=settings["enrollment_segment_count"],
    )

    with torch.inference_mode():
        enrollment_tensor = torch.from_numpy(enrollment_segments).to(
            device=device,
            dtype=torch.float32,
        )
        # 複数区間の平均embeddingにして、1フレーズの音高や母音へ条件付けが偏らないようにする。
        embedding = embedding_model.embedding(enrollment_tensor).mean(dim=0, keepdim=True)
        print("PROGRESS 1 2", flush=True)

        mixture, _ = librosa.load(
            settings["input_path"],
            sr=settings["sample_rate"],
            mono=True,
        )
        mixture = np.asarray(mixture, dtype=np.float32)
        if mixture.size == 0 or not np.isfinite(mixture).all():
            raise ValueError("対象楽曲が空か、非有限値を含んでいます")
        mixture_tensor = torch.from_numpy(mixture).unsqueeze(0).to(
            device=device,
            dtype=torch.float32,
        )
        window = torch.hann_window(model_settings["n_fft"], device=device)
        spectrum = torch.stft(
            mixture_tensor,
            n_fft=model_settings["n_fft"],
            hop_length=model_settings["hop_length"],
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            pad_mode="reflect",
            return_complex=True,
        )
        mixture_magnitude = torch.abs(spectrum)
        estimated_magnitude = separator(mixture_magnitude.unsqueeze(1), embedding).squeeze(1)
        # モデルの直接出力は混合の振幅を超えることがあり、伴奏binをそのまま増幅してしまう。
        # 補完推定との比でmaskへ正規化し、exponentで伴奏優位のbinを追加で抑える。
        # exponent=1.0 かつ推定振幅が混合以下なら、従来の直接出力と一致する。
        exponent = settings["mask_exponent"]
        target_power = estimated_magnitude.clamp_min(0.0) ** exponent
        residual_power = (mixture_magnitude - estimated_magnitude).clamp_min(0.0) ** exponent
        mask = target_power / (target_power + residual_power + settings["silence_epsilon"])
        if settings["mask_floor"] > 0.0:
            mask = torch.where(mask < settings["mask_floor"], torch.zeros_like(mask), mask)
        estimated_spectrum = spectrum * mask
        target = torch.istft(
            estimated_spectrum,
            n_fft=model_settings["n_fft"],
            hop_length=model_settings["hop_length"],
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=mixture.size,
        )[0].float().cpu().numpy()

    if not np.isfinite(target).all():
        raise RuntimeError("target singer出力に非有限値が含まれています")
    residual = mixture - target
    output_directory = Path(settings["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {"target_vocal": target, "residual": residual}
    for stem in model_settings["stems"]:
        # 論文モデルはmono学習のため、ブラウザ再生用WAVでは左右へ同じ信号を配置する。
        stereo = np.stack((outputs[stem], outputs[stem]), axis=1)
        sound_file.write(output_directory / f"{stem}.wav", stereo, settings["sample_rate"])
    print("PROGRESS 2 2", flush=True)


def run(settings_path: Path) -> None:
    """JSON 設定に従い、選択モデルの音声契約で全ステムへ分離する。"""
    import librosa
    import soundfile as sound_file
    import torch

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("複数歌声分離には CUDA GPU が必要です")
    device = torch.device("cuda", settings["cuda_device_id"])
    model_id = settings["model_id"]
    model_settings = settings["model"]
    if model_id == "singer-informed":
        _run_singer_informed(settings, device)
        return
    source_count = len(model_settings["stems"])
    # 匿名singerのモデルだけ左右とチャンク境界で出力順をそろえる。声部名が決まっている
    # モデルは index と stem の対応が固定なので、並べ替えてはならない。
    rescale_to_mixture = False
    if model_id == "unmixx":
        model = load_unmixx(model_settings, device)
        predictor = _unmixx_predictor(model, device)
        align_anonymous_sources = True
    elif model_id == "sepacap":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("SepACap には BF16 対応 CUDA GPU が必要です")
        model = load_sepacap(model_settings, device)
        predictor = _sepacap_predictor(model, device, source_count)
        align_anonymous_sources = False
    elif model_id == "jacappella-dptnet":
        model = load_jacappella_dptnet(model_settings, device)
        predictor = _monaural_predictor(model, device, source_count)
        align_anonymous_sources = False
        # sigmoid maskの直接出力は混合より十数倍小さい。公式 separate.py の既定と同じく
        # 最小二乗で利得を戻さないと、ブラウザ再生でほぼ無音になる。
        rescale_to_mixture = True
    elif model_id == "medleyvox":
        model = load_medleyvox(model_settings, device)
        predictor = _monaural_predictor(model, device, source_count)
        # duetモデルなので2出力に歌手の割り当てがなく、UNMIXXと同じ整列が必要。
        align_anonymous_sources = True
    else:
        raise ValueError(f"未対応の子プロセスモデルです: {model_id}")

    waveform, _ = librosa.load(settings["input_path"], sr=settings["sample_rate"], mono=False)
    audio = normalize_stereo(waveform)
    # ラウドネス前提があるモデルは、全長で一度だけ利得を決めてから分割推論へ渡す。
    input_gain = 1.0
    if "target_lufs" in model_settings:
        input_gain = loudness_normalization_gain(
            audio, settings["sample_rate"], model_settings["target_lufs"],
        )
        audio = audio * input_gain
    estimates = separate_long_audio(
        audio,
        predictor,
        source_count=source_count,
        chunk_size=settings["chunk_size"],
        hop_size=settings["hop_size"],
        align_anonymous_sources=align_anonymous_sources,
        epsilon=settings["silence_epsilon"],
        progress=lambda completed, total: print(f"PROGRESS {completed} {total}", flush=True),
    )
    if rescale_to_mixture:
        # チャンクごとに解くと利得が段差になるため、合成後の全長に対して一度だけ解く。
        estimates = rescale_estimates_to_mixture(estimates, audio, settings["silence_epsilon"])
    if input_gain != 1.0:
        # 入力へかけた利得を戻し、元の音量のステムとして書き出す。
        estimates = estimates / input_gain
    output_directory = Path(settings["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(model_settings["stems"]):
        sound_file.write(output_directory / f"{stem}.wav", estimates[index].T, settings["sample_rate"])


def main() -> None:
    """shell 展開を使わない単一 JSON 引数の CLI。"""
    parser = argparse.ArgumentParser(description="Multi-singer CUDA model runner")
    parser.add_argument("settings", type=Path)
    arguments = parser.parse_args()
    run(arguments.settings)


if __name__ == "__main__":
    main()
