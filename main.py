import os
import torch
import pandas as pd
from io import StringIO
from torch.utils.data import DataLoader

from config import BASE_PATH, CSV_PATH, TRAIN_IMG_DIR, BATCH_SIZE, MODEL_PATH
from data.prepare_data import RetinaDataset, get_transform
from models.model import create_model
from models.train import train_model
from utils.predict import predict_image

if __name__ == "__main__":
    # Verificaciones
    if not os.path.exists(BASE_PATH):
        raise FileNotFoundError(f"No se encuentra la carpeta: {BASE_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No se encuentra el archivo CSV: {CSV_PATH}")
    if not os.path.exists(TRAIN_IMG_DIR):
        raise FileNotFoundError(f"No se encuentra la carpeta de imágenes: {TRAIN_IMG_DIR}")

    # Cargar datos
    all_data = pd.read_csv(CSV_PATH)
    sampled_data = all_data.sample(n=500, random_state=42).reset_index(drop=True)

    buffer = StringIO()
    sampled_data.to_csv(buffer, index=False)
    buffer.seek(0)

    dataset = RetinaDataset(buffer, TRAIN_IMG_DIR, get_transform())
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Crear modelo
    model = create_model()

    # Entrenar o cargar modelo
    if os.path.exists(MODEL_PATH):
        print(f"Modelo ya existe. Cargando desde {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("Entrenando modelo desde cero...")
        train_model(model, train_loader)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Modelo guardado en {MODEL_PATH}")

    # Predicción de ejemplo
    test_img = os.path.join(TRAIN_IMG_DIR, "163_left.jpeg")
    if os.path.exists(test_img):
        pred = predict_image(model, test_img)
        print(f"Predicción para '163_left.jpeg': {pred}")
    else:
        print(f"Imagen no encontrada: {test_img}")
