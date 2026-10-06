"""backend/tests/test_singer_informed.py: 参照歌手抽出の二段構成と設定受け渡しを検証する。"""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.singer_informed as singer_informed  # noqa: E402
from src.config import SINGER_INFORMED  # noqa: E402
from src.separation_request import SeparationRequest  # noqa: E402


class SingerInformedPipelineTests(unittest.TestCase):
    """GPUを使わず、前段の接続と子プロセス設定だけを確認する。"""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        root = Path(self._directory.name)
        self.request = SeparationRequest.from_metadata({
            "input_path": str(root / "song.wav"),
            "enrollment_path": str(root / "reference.wav"),
            "output_dir": str(root / "staging"),
        })
        self.extracted: list[Path] = []

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _extract_vocals(self, input_path, output_dir, chunk_progress):
        """BS PolarFormer を呼ばず、呼び出し順と出力契約だけを再現する。"""
        self.extracted.append(input_path)
        chunk_progress(1, 1)
        return {
            "vocals": str(output_dir / "vocals.wav"),
            "instrumental": str(output_dir / "instrumental.wav"),
        }

    def _written_settings(self) -> dict:
        work = self.request.output_dir.parent / "work"
        return json.loads((work / "singer-informed-runner.json").read_text(encoding="utf-8"))

    def test_requires_enrollment_audio(self) -> None:
        request = replace(self.request, enrollment_path=None)
        with self.assertRaisesRegex(ValueError, "参照音声"):
            singer_informed.separate_target_singer(
                request, self._extract_vocals, lambda _phase, _percent: None,
            )

    def test_cascade_feeds_the_vocal_stem_and_restores_the_instrumental(self) -> None:
        """伴奏はBS PolarFormerが外し、residualへ戻して元の楽曲を保つ。"""
        phases: list[tuple[str, int]] = []
        merged: dict[str, np.ndarray] = {}

        def write_stems(stems, sample_rate, output_dir):
            merged.update(stems)
            return {stem: str(Path(output_dir) / f"{stem}.wav") for stem in stems}

        def stereo_at_length(path, sample_rate, length):
            names = {"vocals": 0.0, "instrumental": 3.0, "target_vocal": 1.0, "residual": 2.0}
            return np.full((2, length), names[Path(path).stem], dtype=np.float32)

        with (
            patch.object(singer_informed, "run_model_process") as child,
            patch.object(singer_informed, "load_audio", return_value=np.zeros((2, 7))),
            patch.object(singer_informed, "stereo_at_length", stereo_at_length),
            patch.object(singer_informed, "write_stems", write_stems),
        ):
            output_paths = singer_informed.separate_target_singer(
                self.request, self._extract_vocals, lambda phase, percent: phases.append((phase, percent)),
            )

        work = self.request.output_dir.parent / "work"
        # 参照音声と楽曲の両方を前段へ通し、子プロセスには歌声だけを渡す。
        self.assertEqual(
            self.extracted, [self.request.enrollment_path, self.request.input_path],
        )
        settings = self._written_settings()
        self.assertEqual(settings["input_path"], str(work / "bs" / "vocals.wav"))
        self.assertEqual(
            settings["enrollment_path"], str(work / "bs-enrollment" / "vocals.wav"),
        )
        self.assertEqual(settings["output_directory"], str(work / "singer-informed"))
        self.assertEqual(child.call_args.kwargs["start_percent"], 40)

        self.assertEqual(set(output_paths), set(SINGER_INFORMED.stems))
        np.testing.assert_allclose(merged["target_vocal"], np.full((2, 7), 1.0))
        np.testing.assert_allclose(merged["residual"], np.full((2, 7), 5.0))
        self.assertEqual(phases[0], ("extracting_vocals", 0))
        self.assertEqual(phases[-1], ("finalizing", 99))

    def test_direct_mode_skips_the_vocal_extraction_stage(self) -> None:
        """論文どおり混合曲を直接入力する構成では、前段を呼ばない。"""
        with (
            patch.object(singer_informed, "SINGER_INFORMED", replace(
                SINGER_INFORMED, cascade_vocal_extraction=False,
            )),
            patch.object(singer_informed, "run_model_process") as child,
        ):
            output_paths = singer_informed.separate_target_singer(
                self.request, self._extract_vocals, lambda _phase, _percent: None,
            )

        self.assertEqual(self.extracted, [])
        settings = self._written_settings()
        self.assertEqual(settings["input_path"], str(self.request.input_path))
        self.assertEqual(settings["output_directory"], str(self.request.output_dir))
        self.assertEqual(child.call_args.kwargs["start_percent"], 0)
        self.assertEqual(
            output_paths["target_vocal"], str(self.request.output_dir / "target_vocal.wav"),
        )

    def test_selected_lambda_picks_the_matching_checkpoint(self) -> None:
        """λごとに公開されている別checkpointへ切り替える。"""
        request = replace(self.request, conditioning_lambda="0.2")
        expected = SINGER_INFORMED.checkpoint_for("0.2")
        with (
            patch.object(singer_informed, "SINGER_INFORMED", replace(
                SINGER_INFORMED, cascade_vocal_extraction=False,
            )),
            patch.object(singer_informed, "run_model_process"),
        ):
            singer_informed.separate_target_singer(
                request, self._extract_vocals, lambda _phase, _percent: None,
            )

        model = self._written_settings()["model"]
        self.assertEqual(model["conditioning_lambda"], "0.2")
        self.assertEqual(model["checkpoint_path"], str(expected.path))
        self.assertEqual(model["checkpoint_sha256"], expected.sha256)
        # 全λで同じアーキテクチャなので、embeddingは共通のものを使う。
        self.assertEqual(
            model["embedding_checkpoint_path"],
            str(SINGER_INFORMED.embedding_checkpoint_path),
        )

    def test_published_lambdas_have_distinct_verified_checkpoints(self) -> None:
        """4条件が別の重みを指し、UIの選択肢と一致することを守る。"""
        self.assertEqual(SINGER_INFORMED.conditioning_lambdas, ("none", "0.05", "0.1", "0.2"))
        hashes = {c.sha256 for c in SINGER_INFORMED.checkpoints}
        paths = {c.path for c in SINGER_INFORMED.checkpoints}
        self.assertEqual(len(hashes), 4)
        self.assertEqual(len(paths), 4)
        self.assertEqual(
            SINGER_INFORMED.checkpoint_for(None).conditioning_lambda,
            SINGER_INFORMED.default_conditioning_lambda,
        )
        with self.assertRaisesRegex(ValueError, "conditioning_lambda"):
            SINGER_INFORMED.checkpoint_for("0.5")

    def test_mask_tuning_reaches_the_child_process(self) -> None:
        """maskと平均embeddingの調整値は子プロセスへそのまま渡す。"""
        tuned = replace(
            SINGER_INFORMED,
            cascade_vocal_extraction=False,
            mask_exponent=2.5,
            mask_floor=0.05,
            enrollment_segment_count=7,
        )
        with (
            patch.object(singer_informed, "SINGER_INFORMED", tuned),
            patch.object(singer_informed, "run_model_process"),
        ):
            singer_informed.separate_target_singer(
                self.request, self._extract_vocals, lambda _phase, _percent: None,
            )

        settings = self._written_settings()
        self.assertEqual(settings["mask_exponent"], 2.5)
        self.assertEqual(settings["mask_floor"], 0.05)
        self.assertEqual(settings["enrollment_segment_count"], 7)


if __name__ == "__main__":
    unittest.main()
