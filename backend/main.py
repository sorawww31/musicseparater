from typing import Annotated

from fastapi import FastAPI, Form, UploadFile

app = FastAPI()

@app.post("/separations")
async def separate_audio(
    file: UploadFile,
    model_id: Annotated[str, Form()],
    num_vocals: Annotated[int | None, Form()] = None,
):
    return {"filename": file.filename, "content_type": file.content_type, "model_id": model_id, "num_vocals": num_vocals}