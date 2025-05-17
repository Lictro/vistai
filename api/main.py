import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
from transformers import CLIPProcessor, CLIPModel

# Cargar el modelo CLIP de Hugging Face
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Tu modelo entrenado de severidad
from config import MODEL_PATH
from models.model import create_model
from utils.predict import predict_image_api
from data.prepare_data import get_transform

app = FastAPI()

# Cargar modelo de clasificación de severidad
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
        # Leer la imagen subida
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocesar la imagen para el modelo de clasificación
        inputs = transform(image).unsqueeze(0)  # Ajuste según tu pipeline de preprocesamiento
        
        # Realizar la predicción con tu modelo entrenado
        with torch.no_grad():
            prediction = model(inputs)
        
        # Obtenemos el índice de la clase con la mayor probabilidad
        severity_label = prediction.argmax(dim=1).item()  # Asumiendo que es una clasificación

        # Mapa de severidad: mapea los valores numéricos a etiquetas descriptivas
        severity_map = {
            0: "no signs of diabetic retinopathy",
            1: "mild non-proliferative diabetic retinopathy",
            2: "moderate non-proliferative diabetic retinopathy",
            3: "severe non-proliferative diabetic retinopathy",
            4: "proliferative diabetic retinopathy"
        }
        severity = severity_map.get(severity_label, "unknown")  # Si no está en el mapa, devuelve "desconocido"

        # Generar una respuesta textual usando CLIP
        # Creamos las descripciones de texto que se pueden asociar con la imagen
        texts = [
            f"This retinal image shows {severity}.",
            f"There are visible features of {severity} in this fundus photo.",
            f"The severity of diabetic retinopathy is classified as {severity}.",
            f"This is a case of {severity}.",
            f"Fundus scan indicating {severity}.",
            f"The image suggests {severity} based on clinical signs.",
            f"Retinal signs are consistent with {severity}.",
            f"DR severity level: {severity}.",
            f"Findings: {severity}.",
            f"Diagnosis: {severity}."
        ]

        # Preprocesar la imagen para CLIP
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

        # Realizamos la predicción de CLIP
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # Similitud entre texto e imagen
        similarity = logits_per_image.softmax(dim=1)  # Normalizamos con softmax

        # Tomamos el texto con mayor similitud
        response_text = texts[similarity.argmax().item()]

        return JSONResponse(content={"severity": severity, "response": response_text})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
