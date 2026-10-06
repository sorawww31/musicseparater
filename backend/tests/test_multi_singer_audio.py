"""backend/tests/test_multi_singer_audio.py: 長尺分割と歌手順整列をモデルなしで検証する。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_runtime.audio import (  # noqa: E402
    align_stereo_sources,
    chunk_starts,
    loudness_normalization_gain,
    rescale_estimates_to_mixture,
    select_enrollment_segment,
    select_enrollment_segments,
    separate_long_audio,
)


class MultiSingerAudioTests(unittest.TestCase):
    """周波数変換後のチャンク境界でサンプル欠落を起こさないことを守る。"""

    def test_chunk_starts_does_not_add_redundant_tail(self) -> None:
        self.assertEqual(chunk_starts(96_000, 96_000, 48_000), [0])
        self.assertEqual(chunk_starts(96_001, 96_000, 48_000), [0, 48_000])

    def test_overlap_add_preserves_identity_and_length(self) -> None:
        audio = np.linspace(-1, 1, 130_000, dtype=np.float32)[None, :].repeat(2, axis=0)
        progress: list[tuple[int, int]] = []

        result = separate_long_audio(
            audio,
            lambda chunk: np.stack((chunk, -chunk)),
            source_count=2,
            chunk_size=96_000,
            hop_size=48_000,
            align_anonymous_sources=False,
            epsilon=1e-8,
            progress=lambda completed, total: progress.append((completed, total)),
        )

        self.assertEqual(result.shape, (2, 2, 130_000))
        np.testing.assert_allclose(result[0], audio, atol=1e-6)
        np.testing.assert_allclose(result[1], -audio, atol=1e-6)
        self.assertEqual(progress[-1], (2, 2))

    def test_unmixx_stereo_alignment_swaps_only_right_channel(self) -> None:
        source_a = np.array([1, 2, 3], dtype=np.float32)
        source_b = np.array([-3, 1, 2], dtype=np.float32)
        estimates = np.stack((
            np.stack((source_a, source_b)),
            np.stack((source_b, source_a)),
        ))

        aligned = align_stereo_sources(estimates, 1e-8)

        np.testing.assert_array_equal(aligned[0, 0], source_a)
        np.testing.assert_array_equal(aligned[0, 1], source_a)
        np.testing.assert_array_equal(aligned[1, 1], source_b)

    def test_enrollment_uses_loudest_three_second_window_and_normalizes_peak(self) -> None:
        """先頭の無音ではなく、有効な歌声区間を決定的にembeddingへ渡す。"""
        waveform = np.concatenate((
            np.zeros(4, dtype=np.float32),
            np.array([0.5, -1.0, 0.5, -0.5], dtype=np.float32),
        ))

        segment = select_enrollment_segment(
            waveform,
            sample_rate=2,
            duration_seconds=2,
            search_hop_size=2,
            epsilon=1e-8,
        )

        np.testing.assert_allclose(segment, np.array([0.5, -1.0, 0.5, -0.5]))

    def test_enrollment_collects_non_overlapping_windows_by_energy(self) -> None:
        """平均embedding用に、別フレーズの区間をエネルギー順で集める。"""
        waveform = np.array(
            [0.1, 0.1, 0.4, -0.4, 0.05, 0.05, 1.0, -1.0, 0.2, 0.2], dtype=np.float32,
        )

        segments = select_enrollment_segments(
            waveform,
            sample_rate=2,
            duration_seconds=1,
            search_hop_size=2,
            epsilon=1e-8,
            segment_count=3,
        )

        self.assertEqual(segments.shape, (3, 2))
        # 最大エネルギーの区間が先頭で、各区間はpeak正規化されている。
        np.testing.assert_allclose(segments[0], np.array([1.0, -1.0]))
        np.testing.assert_allclose(segments[1], np.array([1.0, -1.0]))
        np.testing.assert_allclose(segments[2], np.array([1.0, 1.0]))

    def test_enrollment_segment_count_is_capped_by_reference_length(self) -> None:
        """重複する区間を重ねず、取れる区間数だけを返す。"""
        waveform = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)

        segments = select_enrollment_segments(
            waveform,
            sample_rate=2,
            duration_seconds=1,
            search_hop_size=1,
            epsilon=1e-8,
            segment_count=8,
        )

        self.assertEqual(segments.shape, (2, 2))

    def test_enrollment_rejects_silent_reference(self) -> None:
        """無音の参照音声は条件付けに使わず、明示的に失敗させる。"""
        with self.assertRaisesRegex(ValueError, "有効な歌声区間"):
            select_enrollment_segments(
                np.zeros(8, dtype=np.float32),
                sample_rate=2,
                duration_seconds=1,
                search_hop_size=1,
                epsilon=1e-8,
                segment_count=3,
            )

    def test_mixture_rescaling_recovers_the_original_stem_gains(self) -> None:
        """jaCappella DPTNet の痩せた出力を、混合に合う音量へ戻す。"""
        first = np.array([[1.0, 0.0, -1.0, 0.5], [0.5, -0.5, 0.25, 0.0]], dtype=np.float32)
        second = np.array([[0.0, 1.0, 0.5, -0.5], [-0.25, 0.5, 0.0, 1.0]], dtype=np.float32)
        mixture = 3.0 * first + 7.0 * second
        # 推論結果が一律に小さい状況を作り、利得だけが復元されることを確かめる。
        estimates = np.stack((first * 0.05, second * 0.05))

        rescaled = rescale_estimates_to_mixture(estimates, mixture, 1e-8)

        np.testing.assert_allclose(rescaled[0], 3.0 * first, atol=1e-5)
        np.testing.assert_allclose(rescaled[1], 7.0 * second, atol=1e-5)
        np.testing.assert_allclose(rescaled.sum(axis=0), mixture, atol=1e-5)

    def test_mixture_rescaling_leaves_silent_channels_untouched(self) -> None:
        """全ステムが無音のチャンネルでは利得が決まらず、値を変えない。"""
        estimates = np.zeros((2, 2, 4), dtype=np.float32)
        estimates[0, 1] = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
        mixture = np.zeros((2, 4), dtype=np.float32)
        mixture[1] = estimates[0, 1]

        rescaled = rescale_estimates_to_mixture(estimates, mixture, 1e-8)

        np.testing.assert_array_equal(rescaled[:, 0], estimates[:, 0])
        np.testing.assert_allclose(rescaled[0, 1], estimates[0, 1], atol=1e-5)

    def test_loudness_gain_matches_the_target_and_skips_unmeasurable_input(self) -> None:
        """MedleyVox の前提である目標ラウドネスへ合わせ、測れない入力は等倍で通す。"""
        sample_rate = 24_000
        noise = np.random.default_rng(0).normal(0.0, 0.2, (2, sample_rate * 3)).astype(np.float32)

        gain = loudness_normalization_gain(noise, sample_rate, -24.0)
        adjusted = loudness_normalization_gain(noise * gain, sample_rate, -24.0)

        # 正規化後は追加の利得が不要になる（1倍へ収束する）。
        self.assertAlmostEqual(adjusted, 1.0, places=3)
        self.assertEqual(loudness_normalization_gain(noise[:, :100], sample_rate, -24.0), 1.0)
        self.assertEqual(
            loudness_normalization_gain(np.zeros((2, sample_rate), dtype=np.float32), sample_rate, -24.0),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
