"""backend/model_runtime/audio.py: 複数歌声向けの音声整形・整列・合成を担う。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def select_enrollment_segments(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    duration_seconds: float,
    search_hop_size: int,
    epsilon: float,
    segment_count: int = 1,
) -> np.ndarray:
    """参照音声から重複しない高エネルギー区間を選び、peak正規化して積み上げる。"""
    audio = np.asarray(waveform, dtype=np.float32)
    segment_size = round(sample_rate * duration_seconds)
    if audio.ndim != 1 or segment_size <= 0 or search_hop_size <= 0 or segment_count <= 0:
        raise ValueError("参照音声またはenrollment設定が不正です")
    if audio.size < segment_size:
        raise ValueError(f"参照音声は{duration_seconds:g}秒以上必要です")
    if not np.isfinite(audio).all():
        raise ValueError("参照音声に非有限値が含まれています")

    final_start = audio.size - segment_size
    starts = list(range(0, final_start + 1, search_hop_size))
    if starts[-1] != final_start:
        starts.append(final_start)
    # 安定ソートにして、同エネルギーなら従来どおり先頭の区間を優先する。
    ranked = sorted(
        starts,
        key=lambda candidate: -float(
            np.mean(audio[candidate:candidate + segment_size] ** 2)
        ),
    )

    segments: list[np.ndarray] = []
    selected: list[int] = []
    for candidate in ranked:
        if len(segments) >= segment_count:
            break
        # 同じフレーズから複数のembeddingを作らないよう、重なる区間は飛ばす。
        if any(abs(candidate - chosen) < segment_size for chosen in selected):
            continue
        segment = audio[candidate:candidate + segment_size]
        peak = float(np.max(np.abs(segment)))
        if peak <= epsilon:
            continue
        selected.append(candidate)
        segments.append(segment / peak)
    if not segments:
        raise ValueError("参照音声から有効な歌声区間を検出できません")
    return np.stack(segments)


def select_enrollment_segment(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    duration_seconds: float,
    search_hop_size: int,
    epsilon: float,
) -> np.ndarray:
    """最大エネルギーの区間だけを返す単一区間版。"""
    return select_enrollment_segments(
        waveform,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        search_hop_size=search_hop_size,
        epsilon=epsilon,
    )[0]


def normalize_stereo(waveform: np.ndarray) -> np.ndarray:
    """音声を `(2, samples)` の有限な float32 配列へそろえる。"""
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.stack((audio, audio))
    if audio.ndim != 2 or audio.shape[0] == 0:
        raise ValueError("音声は mono または channels-first 形式で指定してください")
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    audio = audio[:2]
    if not np.isfinite(audio).all():
        raise ValueError("入力音声に非有限値が含まれています")
    return audio


def chunk_starts(total_samples: int, chunk_size: int, hop_size: int) -> list[int]:
    """末尾チャンクを一度だけ含む開始位置を返す。"""
    if total_samples <= 0 or chunk_size <= 0 or not 0 < hop_size <= chunk_size:
        raise ValueError("チャンク設定または音声長が不正です")
    starts: list[int] = []
    start = 0
    while True:
        starts.append(start)
        if start + chunk_size >= total_samples:
            return starts
        start += hop_size


def _correlation(left: np.ndarray, right: np.ndarray, epsilon: float) -> float:
    """無音時に不安定化しない正規化相関を返す。"""
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= epsilon:
        return 0.0
    return float(np.sum(left * right) / denominator)


def align_stereo_sources(sources: np.ndarray, epsilon: float) -> np.ndarray:
    """UNMIXX の右チャンネルの2出力を左チャンネル順へ合わせる。"""
    direct = sum(_correlation(sources[index, 0], sources[index, 1], epsilon) for index in range(2))
    swapped = sum(_correlation(sources[index, 0], sources[1 - index, 1], epsilon) for index in range(2))
    if swapped > direct:
        sources = sources.copy()
        sources[:, 1] = sources[::-1, 1]
    return sources


def align_chunk_sources(
    current: np.ndarray, previous: np.ndarray, overlap: int, epsilon: float,
) -> np.ndarray:
    """隣接 UNMIXX チャンクの匿名 singer 順を相関が高い方へ合わせる。"""
    if overlap <= 0:
        return current
    previous_tail = previous[:, :, -overlap:]
    current_head = current[:, :, :overlap]
    energy = float(np.sum(previous_tail**2) + np.sum(current_head**2))
    if energy <= epsilon:
        return current
    direct = sum(_correlation(previous_tail[i], current_head[i], epsilon) for i in range(2))
    swapped = sum(_correlation(previous_tail[i], current_head[1 - i], epsilon) for i in range(2))
    return current[::-1].copy() if swapped > direct else current


def loudness_normalization_gain(
    audio: np.ndarray, sample_rate: int, target_lufs: float,
) -> float:
    """統合ラウドネスを目標値へ合わせる線形利得を返す。"""
    import pyloudnorm

    # 公式 inference は mono 入力を測って正規化するので、同じ条件になる downmix で測る。
    # 利得は左右へ同じ値をかけ、ステレオの定位を変えない。
    downmix = np.asarray(audio, dtype=np.float32).mean(axis=0)
    block_seconds = 0.400
    if downmix.size < round(sample_rate * block_seconds):
        # 1ブロックに満たない入力はラウドネスを測れないため、そのまま通す。
        return 1.0
    meter = pyloudnorm.Meter(sample_rate, block_size=block_seconds)
    loudness = float(meter.integrated_loudness(downmix))
    if not np.isfinite(loudness):
        # 無音では -inf になる。公式も同じ条件で正規化を飛ばしている。
        return 1.0
    return float(10.0 ** ((target_lufs - loudness) / 20.0))


def rescale_estimates_to_mixture(
    estimates: np.ndarray, mixture: np.ndarray, epsilon: float,
) -> np.ndarray:
    """各ステムの利得を最小二乗で混合へ合わせる、jaCappella公式の既定後処理。"""
    rescaled = np.empty_like(estimates)
    for channel in range(estimates.shape[1]):
        basis = estimates[:, channel, :].T
        # 全長ぶんのlstsqは数百MBになるため、同値なステム数ぶんの正規方程式をfloat64で解く。
        gram = np.einsum("ts,tu->su", basis, basis, dtype=np.float64)
        projection = np.einsum("ts,t->s", basis, mixture[channel], dtype=np.float64)
        if float(np.trace(gram)) <= epsilon:
            # 全ステムが無音のチャンネルでは利得が決まらないので、そのまま返す。
            rescaled[:, channel, :] = estimates[:, channel, :]
            continue
        gains = np.linalg.lstsq(gram, projection, rcond=None)[0]
        rescaled[:, channel, :] = (basis * gains.astype(np.float32)).T
    return rescaled


def separate_long_audio(
    audio: np.ndarray,
    predict: Callable[[np.ndarray], np.ndarray],
    *,
    source_count: int,
    chunk_size: int,
    hop_size: int,
    align_anonymous_sources: bool,
    epsilon: float,
    progress: Callable[[int, int], None],
) -> np.ndarray:
    """モデル推論をオーバーラップ合成し `(sources, 2, samples)` を返す。"""
    starts = chunk_starts(audio.shape[1], chunk_size, hop_size)
    accumulated = np.zeros((source_count, 2, audio.shape[1]), dtype=np.float32)
    weights = np.zeros(audio.shape[1], dtype=np.float32)
    previous: np.ndarray | None = None
    overlap = chunk_size - hop_size

    for index, start in enumerate(starts):
        end = min(start + chunk_size, audio.shape[1])
        valid_length = end - start
        chunk = np.zeros((2, chunk_size), dtype=np.float32)
        chunk[:, :valid_length] = audio[:, start:end]
        predicted = np.asarray(predict(chunk), dtype=np.float32)
        expected = (source_count, 2, chunk_size)
        if predicted.shape != expected or not np.isfinite(predicted).all():
            raise RuntimeError(f"モデル出力が不正です: expected={expected}, actual={predicted.shape}")
        if align_anonymous_sources:
            predicted = align_stereo_sources(predicted, epsilon)
            if previous is not None:
                predicted = align_chunk_sources(predicted, previous, overlap, epsilon)

        window = np.ones(valid_length, dtype=np.float32)
        fade_length = min(overlap, valid_length)
        if index > 0:
            window[:fade_length] *= np.linspace(0.0, 1.0, fade_length, dtype=np.float32)
        if index < len(starts) - 1:
            window[-fade_length:] *= np.linspace(1.0, 0.0, fade_length, dtype=np.float32)
        accumulated[:, :, start:end] += predicted[:, :, :valid_length] * window
        weights[start:end] += window
        previous = predicted
        progress(index + 1, len(starts))

    return accumulated / np.maximum(weights, epsilon)[None, None, :]
