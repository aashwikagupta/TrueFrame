from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.db import init_db, list_predictions, save_prediction
from app.utils import sha256_bytes, validate_and_decode
from app.vision import predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="TrueFrame API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    data = await file.read()

    try:
        image = validate_and_decode(data, file.content_type or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    label, confidence = predict(image)

    save_prediction(
        sha256_bytes(data),
        file.filename,
        label,
        confidence,
    )

    return {
        "filename": file.filename,
        "prediction": label,
        "confidence": round(confidence, 4),
        "note": "Prototype prediction; not proof that an image is real or fake.",
    }


@app.get("/history")
def history():
    return {"predictions": list_predictions()}