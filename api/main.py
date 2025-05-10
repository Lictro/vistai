from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
import os

from config import MODEL_PATH
from models.model import create_model
from utils.predict import predict_image_api
from data.prepare_data import get_transform

app = FastAPI()

# Cargar modelo entrenado
model = create_model()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
else:
    raise RuntimeError(f"No se encontró el modelo en {MODEL_PATH}. Ejecuta primero main.py para entrenar o guardar el modelo.")

transform = get_transform()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        prediction = predict_image_api(model, image, transform)
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
