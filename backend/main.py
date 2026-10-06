"""backend/main.py: FastAPI で音声を受け取り、分離結果を保存する入口。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.config import JOB_RUNTIME, SINGER_INFORMED
from src.inference import inference as run_inference
from src.job_manager import JobManager
from src.model_catalog import validate_model_request
from src.model_process import terminate_active_process
from src.storage import AudioStorage

storage = AudioStorage()
job_manager = JobManager(storage)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """中断ジョブを回収し、終了時にGPU子プロセスを残さない。"""
    job_manager.recover_interrupted()
    try:
        yield
    finally:
        terminate_active_process()
        job_manager.shutdown()


app = FastAPI(title="Music Separation Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(JOB_RUNTIME.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class SeparationPayload(BaseModel):
    """保存済み音源に対する分離設定を表すJSON body。"""

    audio_id: str
    model_id: str
    num_vocals: int | None = None
    reference_audio_id: str | None = None
    # λは学習時のdual loss重みで、条件ごとに別checkpointが公開されている。
    conditioning_lambda: str | None = None


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


@app.post("/separations", status_code=202)
async def separate_audio(payload: SeparationPayload) -> dict[str, Any]:
    """分離要求を検証して直列GPUキューへ登録する。"""
    try:
        model = validate_model_request(payload.model_id, payload.num_vocals)
        if payload.conditioning_lambda is not None:
            SINGER_INFORMED.checkpoint_for(payload.conditioning_lambda)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    try:
        source_path = storage.find_source_path(payload.audio_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="元音源が見つかりません") from error
    if source_path is None:
        raise HTTPException(status_code=404, detail="元音源が見つかりません")

    reference_path = None
    if model.requires_enrollment:
        if not payload.reference_audio_id:
            raise HTTPException(status_code=422, detail="対象歌手の参照音声が必要です")
        try:
            reference_path = storage.find_source_path(payload.reference_audio_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail="参照音声が見つかりません") from error
        if reference_path is None:
            raise HTTPException(status_code=404, detail="参照音声が見つかりません")

    job_id = job_manager.submit(
        audio_id=payload.audio_id,
        source_path=source_path,
        reference_audio_id=payload.reference_audio_id,
        reference_path=reference_path,
        model=model,
        num_vocals=payload.num_vocals,
        conditioning_lambda=payload.conditioning_lambda,
        runner=inference,
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/separations/{job_id}",
    }


@app.get("/separations/{job_id}")
async def get_separation(job_id: str) -> dict[str, Any]:
    """進捗と、完了後だけ利用できるステムURLを返す。"""
    try:
        metadata = job_manager.get(job_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="分離ジョブが見つかりません") from error
    if metadata is None:
        raise HTTPException(status_code=404, detail="分離ジョブが見つかりません")
    stems = metadata.get("stems")
    return {
        "job_id": job_id,
        "status": metadata["status"],
        "phase": metadata["phase"],
        "progress_percent": metadata["progress_percent"],
        "stems": (
            {stem: f"/separations/{job_id}/stems/{stem}" for stem in stems}
            if stems else None
        ),
        "error": metadata.get("error"),
    }


@app.get("/separations/{job_id}/stems/{stem}", response_class=FileResponse)
async def get_separation_stem(job_id: str, stem: str, download: bool = False) -> FileResponse:
    """分離済みWAVを試聴用または明示的なdownload responseで返す。"""
    try:
        stem_path = storage.find_stem_path(job_id, stem)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="分離結果が見つかりません") from error
    if stem_path is None:
        raise HTTPException(status_code=404, detail="分離結果が見つかりません")
    filename = f"{stem}.wav" if download else None
    return FileResponse(stem_path, media_type="audio/wav", filename=filename)


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
