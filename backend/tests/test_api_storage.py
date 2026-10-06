"""backend/tests/test_api_storage.py: FastAPI の保存契約を重いモデルなしで確認する。"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from src.config import AudioStorageConfig  # noqa: E402
from src.job_manager import JobManager  # noqa: E402
from src.storage import AudioStorage  # noqa: E402


class ApiStorageTests(unittest.TestCase):
    """アップロード、モデル入力、成果物 metadata の境界を検証する。"""

    def test_upload_then_separate_reuses_audio_id_without_model_in_path(self) -> None:
        """アップロードと推論を分け、保存済み audio_id を推論へ渡す。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            previous_storage = main.storage
            previous_job_manager = main.job_manager
            main.storage = AudioStorage(
                AudioStorageConfig(root_directory=Path(temporary_directory) / "audio")
            )
            main.job_manager = JobManager(main.storage)

            def fake_inference(
                model_id: str,
                metadata: dict[str, object],
                num_vocals: int | None,
                progress: object,
            ) -> dict[str, str]:
                self.assertEqual(model_id, "bs-polarformer")
                self.assertEqual(num_vocals, 2)
                stems_directory = Path(metadata["output_dir"])
                stems_directory.mkdir(parents=True)
                vocals_path = stems_directory / "vocals.wav"
                instrumental_path = stems_directory / "instrumental.wav"
                vocals_path.write_bytes(b"stem")
                instrumental_path.write_bytes(b"stem")
                return {
                    "vocals": str(vocals_path),
                    "instrumental": str(instrumental_path),
                }

            try:
                uploadfile = UploadFile(
                    file=io.BytesIO(b"audio bytes"),
                    filename="song.mp3",
                    headers={"content-type": "audio/mpeg"},
                )
                upload_response = asyncio.run(main.upload_audio(uploadfile))
                payload = main.SeparationPayload(
                    audio_id=upload_response["audio_id"],
                    model_id="bs-polarformer",
                    num_vocals=2,
                )
                with patch.object(main, "inference", side_effect=fake_inference):
                    response = asyncio.run(main.separate_audio(payload))
                    main.job_manager.wait_for(response["job_id"])
                    status_response = asyncio.run(main.get_separation(response["job_id"]))
            finally:
                main.job_manager.shutdown()
                main.storage = previous_storage
                main.job_manager = previous_job_manager

            root = Path(temporary_directory)
            source_directory = root / "audio" / "sources" / upload_response["audio_id"]
            separation_directory = root / "audio" / "separations" / response["job_id"]
            self.assertTrue((source_directory / "original.wav").exists())
            self.assertEqual((source_directory / "original.wav").read_bytes(), b"audio bytes")
            self.assertEqual(
                upload_response["original_url"],
                f"/audios/{upload_response['audio_id']}/original",
            )
            metadata = json.loads((separation_directory / "metadata.json").read_text())
            self.assertEqual(metadata["audio_id"], upload_response["audio_id"])
            self.assertEqual(metadata["model_id"], "bs-polarformer")
            self.assertEqual(metadata["num_vocals"], 2)
            self.assertNotIn("bs-polarformer", str(separation_directory))
            self.assertTrue((separation_directory / "stems" / "vocals.wav").exists())
            self.assertEqual(response["status"], "queued")
            self.assertEqual(response["status_url"], f"/separations/{response['job_id']}")
            self.assertEqual(status_response["status"], "completed")
            self.assertEqual(status_response["progress_percent"], 100)
            self.assertEqual(
                status_response["stems"]["vocals"],
                f"/separations/{response['job_id']}/stems/vocals",
            )
            self.assertNotIn("audio/", str(status_response))

    def test_original_url_serves_source_without_exposing_storage_path(self) -> None:
        """元音源はaudio_idで取得し、保存先を直接公開しない。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            previous_storage = main.storage
            main.storage = AudioStorage(
                AudioStorageConfig(root_directory=Path(temporary_directory) / "audio")
            )
            audio_id = main.storage.new_id()
            source_path = main.storage.source_paths(audio_id)["original"]
            source_path.parent.mkdir(parents=True)
            source_path.write_bytes(b"source")

            try:
                response = asyncio.run(main.get_original_audio(audio_id))
                self.assertIsInstance(response, FileResponse)
                self.assertEqual(response.media_type, "audio/wav")
                self.assertEqual(response.path, source_path.resolve())
                self.assertNotIn("content-disposition", response.headers)
                with self.assertRaises(HTTPException) as error:
                    asyncio.run(main.get_original_audio("missing"))
                self.assertEqual(error.exception.status_code, 404)
            finally:
                main.storage = previous_storage

    def test_stem_url_serves_wav_without_exposing_storage_path(self) -> None:
        """API URLは保存先を直接公開せず、生成済みWAVだけを返す。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            previous_storage = main.storage
            main.storage = AudioStorage(
                AudioStorageConfig(root_directory=Path(temporary_directory) / "audio")
            )
            job_id = main.storage.new_id()
            stem_path = main.storage.separation_paths(job_id)["stems"] / "vocals.wav"
            stem_path.parent.mkdir(parents=True)
            stem_path.write_bytes(b"stem")

            try:
                response = asyncio.run(main.get_separation_stem(job_id, "vocals"))
                self.assertIsInstance(response, FileResponse)
                self.assertEqual(response.media_type, "audio/wav")
                self.assertEqual(response.path, stem_path.resolve())
                self.assertNotIn("content-disposition", response.headers)
                download_response = asyncio.run(
                    main.get_separation_stem(job_id, "vocals", download=True)
                )
                self.assertIn('filename="vocals.wav"', download_response.headers["content-disposition"])
                with self.assertRaises(HTTPException) as error:
                    asyncio.run(main.get_separation_stem(job_id, "missing"))
                self.assertEqual(error.exception.status_code, 404)
            finally:
                main.storage = previous_storage

    def test_separation_rejects_unknown_audio_id_before_inference(self) -> None:
        """存在しない audio_id では job を実行せず 404 を返す。"""
        payload = main.SeparationPayload(audio_id="missing", model_id="bs-polarformer")

        with patch.object(main, "inference") as mocked_inference:
            with self.assertRaises(HTTPException) as error:
                asyncio.run(main.separate_audio(payload))

        self.assertEqual(error.exception.status_code, 404)
        mocked_inference.assert_not_called()

    def test_singer_informed_requires_and_forwards_reference_audio(self) -> None:
        """参照IDを検証し、対象楽曲とは別の保存済み音源をrunnerへ渡す。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            previous_storage = main.storage
            previous_job_manager = main.job_manager
            main.storage = AudioStorage(
                AudioStorageConfig(root_directory=Path(temporary_directory) / "audio")
            )
            main.job_manager = JobManager(main.storage)
            audio_id = main.storage.new_id()
            reference_audio_id = main.storage.new_id()
            source_path = main.storage.source_paths(audio_id)["original"]
            reference_path = main.storage.source_paths(reference_audio_id)["original"]
            source_path.parent.mkdir(parents=True)
            reference_path.parent.mkdir(parents=True)
            source_path.write_bytes(b"mixture")
            reference_path.write_bytes(b"enrollment")

            def fake_inference(
                model_id: str,
                metadata: dict[str, object],
                num_vocals: int | None,
                progress: object,
            ) -> dict[str, str]:
                self.assertEqual(model_id, "singer-informed")
                self.assertIsNone(num_vocals)
                self.assertEqual(Path(metadata["enrollment_path"]), reference_path.resolve())
                output_directory = Path(metadata["output_dir"])
                output_directory.mkdir(parents=True)
                paths = {
                    stem: output_directory / f"{stem}.wav"
                    for stem in ("target_vocal", "residual")
                }
                for path in paths.values():
                    path.write_bytes(b"stem")
                return {stem: str(path) for stem, path in paths.items()}

            try:
                missing_reference = main.SeparationPayload(
                    audio_id=audio_id,
                    model_id="singer-informed",
                )
                with self.assertRaises(HTTPException) as error:
                    asyncio.run(main.separate_audio(missing_reference))
                self.assertEqual(error.exception.status_code, 422)

                payload = main.SeparationPayload(
                    audio_id=audio_id,
                    reference_audio_id=reference_audio_id,
                    model_id="singer-informed",
                )
                with patch.object(main, "inference", side_effect=fake_inference):
                    response = asyncio.run(main.separate_audio(payload))
                    main.job_manager.wait_for(response["job_id"])
            finally:
                main.job_manager.shutdown()
                main.storage = previous_storage
                main.job_manager = previous_job_manager

            metadata_path = (
                Path(temporary_directory)
                / "audio"
                / "separations"
                / response["job_id"]
                / "metadata.json"
            )
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata["reference_audio_id"], reference_audio_id)
            self.assertEqual(metadata["status"], "completed")

    def test_openapi_separates_multipart_upload_from_json_inference(self) -> None:
        """公開スキーマでも upload は form、separation は JSON として示す。"""
        paths = main.app.openapi()["paths"]

        upload_content = paths["/audios"]["post"]["requestBody"]["content"]
        separation_content = paths["/separations"]["post"]["requestBody"]["content"]

        self.assertIn("multipart/form-data", upload_content)
        self.assertIn("application/json", separation_content)


if __name__ == "__main__":
    unittest.main()
