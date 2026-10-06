"""backend/tests/test_multi_singer_audio.py: 長尺分割と歌手順整列をモデルなしで検証する。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_runtime.audio import (  # noqa: E402
    align_stereo_sources,
    chunk_starts,
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


if __name__ == "__main__":
    unittest.main()
