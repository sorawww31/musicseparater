"""backend/main.py: FastAPI で音声を受け取り、分離結果を保存する入口。"""

from __future__ import annotations

from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.inference import inference as run_inference
from src.storage import AudioStorage
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="Music Separation Backend")
storage = AudioStorage()


class SeparationPayload(BaseModel):
    """保存済み音源に対する分離設定を表すJSON body。"""

    audio_id: str
    model_id: str
    num_vocals: int | None = None


def inference(*args: Any, **kwargs: Any) -> dict[str, str]:
    """重いモデル依存を API 起動時に読み込まず、推論時だけ委譲する。"""

    return run_inference(*args, **kwargs)


@app.post("/audios", status_code=201)
async def upload_audio(
    uploadfile: Annotated[UploadFile, File(description="分離する音声ファイル")],
) -> dict[str, Any]:
    """元音源を保存し、再利用可能な audio_id と取得URLを返す。"""
    audio_id = storage.new_id()
    source_paths = storage.source_paths(audio_id)
    source_metadata = {
        "audio_id": audio_id,
        "original_filename": uploadfile.filename,
        "content_type": uploadfile.content_type,
        "created_at": storage.timestamp(),
        "path": str(source_paths["original"]),
    }

    try:
        await storage.save_source(uploadfile, audio_id, metadata=source_metadata)
    except Exception as error:
        raise HTTPException(status_code=500, detail="音源の保存に失敗しました") from error
    finally:
        await uploadfile.close()

    return {
        "audio_id": audio_id,
        "original_url": f"/audios/{audio_id}/original",
    }


@app.post("/separations", status_code=201)
async def separate_audio(payload: SeparationPayload) -> dict[str, Any]:
    """保存済み audio_id を指定モデルで分離し、job_id と成果物URLを返す。"""
    try:
        source_path = storage.find_source_path(payload.audio_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="元音源が見つかりません") from error
    if source_path is None:
        raise HTTPException(status_code=404, detail="元音源が見つかりません")

    job_id = storage.new_id()
    separation_paths = storage.separation_paths(job_id)

    try:
        raw_output_paths = await run_in_threadpool(
            inference,
            payload.model_id,
            {
                "input_path": source_path,
                "output_dir": separation_paths["stems"],
            },
            payload.num_vocals,
        )
        output_paths = {stem: str(path) for stem, path in raw_output_paths.items()}

    except Exception as error:
        raise HTTPException(status_code=500, detail="音声分離に失敗しました") from error

    separation_metadata = {
        "audio_id": payload.audio_id,
        "job_id": job_id,
        "model_id": payload.model_id,
        "num_vocals": payload.num_vocals,
        "status": "completed",
        "created_at": storage.timestamp(),
        "source": str(source_path),
        "stems": output_paths,
    }
    storage.write_metadata(separation_paths["metadata"], separation_metadata)
    return {
        "job_id": job_id,
        "status": "completed",
        "stems": {
            stem: f"/separations/{job_id}/stems/{stem}"
            for stem in output_paths
        },
    }


@app.get("/separations/{job_id}/stems/{stem}", response_class=FileResponse)
async def get_separation_stem(job_id: str, stem: str) -> FileResponse:
    """内部保存パスを公開せず、分離済みWAVをAPI経由で返す。"""
    try:
        stem_path = storage.find_stem_path(job_id, stem)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="分離結果が見つかりません") from error
    if stem_path is None:
        raise HTTPException(status_code=404, detail="分離結果が見つかりません")
    return FileResponse(stem_path, media_type="audio/wav")


@app.get("/audios/{audio_id}/original", response_class=FileResponse)
async def get_original_audio(audio_id: str) -> FileResponse:
    """内部保存パスを公開せず、元音源をAPI経由で返す。"""
    try:
        source_path = storage.find_source_path(audio_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="元音源が見つかりません") from error
    if source_path is None:
        raise HTTPException(status_code=404, detail="元音源が見つかりません")

    return FileResponse(source_path, media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
