"""backend/tests/test_multi_singer_models.py: モデルごとの音声契約と子プロセス設定を検証する。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.multi_singer as multi_singer  # noqa: E402
from src.model_catalog import MODEL_CATALOG  # noqa: E402
from src.separation_request import SeparationRequest  # noqa: E402


class MultiSingerModelSettingsTests(unittest.TestCase):
    """GPUを使わず、周波数・分割長・ソース指定の受け渡しだけを確認する。"""

    def _settings(self, model_id: str) -> dict:
        return multi_singer._runner_settings(model_id, Path("song.wav"), Path("out"))

    def test_existing_models_keep_their_24_khz_contract(self) -> None:
        """モデル別設定へ移しても、既存2モデルの音声契約は変えない。"""
        for model_id in ("unmixx", "sepacap"):
            with self.subTest(model_id=model_id):
                settings = self._settings(model_id)
                self.assertEqual(settings["sample_rate"], 24_000)
                self.assertEqual(settings["chunk_size"], 96_000)
                self.assertEqual(settings["hop_size"], 48_000)

    def test_jacappella_dptnet_runs_at_48_khz_with_the_trained_segment_length(self) -> None:
        """学習時の seq_dur 5.046 秒と同じ長さで分割する。"""
        settings = self._settings("jacappella-dptnet")

        self.assertEqual(settings["sample_rate"], 48_000)
        self.assertEqual(settings["chunk_size"], 242_208)
        self.assertEqual(settings["chunk_size"], round(48_000 * 5.046))
        self.assertEqual(settings["hop_size"], settings["chunk_size"] // 2)
        # 声部名つきのモデルなので、出力indexと声部の対応はconf.ymlの順で固定する。
        self.assertEqual(
            settings["model"]["stems"],
            ["vocal_percussion", "bass", "alto", "tenor", "soprano", "lead_vocal"],
        )
        self.assertIn("filterbank_directory", settings["model"])
        # ラウドネス前提を持たないモデルでは、入力の音量を変えない。
        self.assertNotIn("target_lufs", settings["model"])

    def test_medleyvox_runs_at_24_khz_and_carries_its_loudness_target(self) -> None:
        """duetモデルは2出力固定で、公式と同じ入力ラウドネスを子プロセスへ渡す。"""
        settings = self._settings("medleyvox")

        self.assertEqual(settings["sample_rate"], 24_000)
        self.assertEqual(settings["chunk_size"], 72_000)
        self.assertEqual(settings["hop_size"], 36_000)
        self.assertEqual(settings["model"]["stems"], ["singer_1", "singer_2"])
        self.assertEqual(settings["model"]["target_lufs"], -24.0)
        # asteroid 実装は別リポジトリから読むため、両方のソースを明示する。
        self.assertIn("asteroid_directory", settings["model"])
        self.assertIn("filterbank_directory", settings["model"])

    def test_every_model_pins_a_checkpoint_hash(self) -> None:
        """差し替えられた重みを検証なしでロードしないよう、全モデルでhashを持つ。"""
        for model_id in ("unmixx", "sepacap", "jacappella-dptnet", "medleyvox"):
            with self.subTest(model_id=model_id):
                model = self._settings(model_id)["model"]
                self.assertEqual(len(model["checkpoint_sha256"]), 64)
                self.assertTrue(model["revision"])

    def test_rejects_unknown_model_before_starting_a_child_process(self) -> None:
        with self.assertRaisesRegex(ValueError, "未対応"):
            self._settings("medleyvox-v2")

    def test_catalog_publishes_the_model_stems_plus_the_instrumental(self) -> None:
        """UIへ公開するステム契約と、子プロセスのステム順を一致させる。"""
        for model_id in ("jacappella-dptnet", "medleyvox"):
            with self.subTest(model_id=model_id):
                definition = MODEL_CATALOG[model_id]
                self.assertTrue(definition.multi_singer)
                self.assertFalse(definition.requires_enrollment)
                self.assertEqual(
                    list(definition.stems),
                    [*self._settings(model_id)["model"]["stems"], "instrumental"],
                )


class MultiSingerPipelineTests(unittest.TestCase):
    """前段のボーカル抽出から成果物書き出しまでの接続を確認する。"""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        root = Path(self._directory.name)
        self.request = SeparationRequest.from_metadata({
            "input_path": str(root / "song.wav"),
            "output_dir": str(root / "staging"),
        })

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _extract_vocals(self, input_path, output_dir, chunk_progress):
        """BS PolarFormer を呼ばず、呼び出し契約だけを再現する。"""
        chunk_progress(1, 1)
        return {
            "vocals": str(output_dir / "vocals.wav"),
            "instrumental": str(output_dir / "instrumental.wav"),
        }

    def test_child_settings_and_stems_follow_the_selected_model(self) -> None:
        """選択モデルの周波数で子プロセスを起動し、伴奏を足して44.1 kHzで返す。"""
        written: dict[str, np.ndarray] = {}

        def write_stems(stems, sample_rate, output_dir):
            written.update(stems)
            self.assertEqual(sample_rate, 44_100)
            return {stem: f"{output_dir}/{stem}.wav" for stem in stems}

        with (
            patch.object(multi_singer, "run_model_process") as child,
            patch.object(multi_singer, "load_audio", return_value=np.zeros((2, 5))),
            patch.object(
                multi_singer, "stereo_at_length", lambda *_: np.zeros((2, 5), dtype=np.float32),
            ),
            patch.object(multi_singer, "write_stems", write_stems),
        ):
            multi_singer.separate_multi_singer(
                "jacappella-dptnet",
                self.request,
                self._extract_vocals,
                lambda _phase, _percent: None,
            )

        child.assert_called_once()
        settings = json.loads(
            (self.request.output_dir.parent / "work" / "runner.json").read_text(encoding="utf-8"),
        )
        self.assertEqual(settings["model_id"], "jacappella-dptnet")
        self.assertEqual(settings["sample_rate"], 48_000)
        # 子プロセスには前段のボーカルだけを渡す。
        self.assertTrue(settings["input_path"].endswith("vocals.wav"))
        self.assertEqual(
            sorted(written),
            ["alto", "bass", "instrumental", "lead_vocal", "soprano", "tenor", "vocal_percussion"],
        )


if __name__ == "__main__":
    unittest.main()
