import os
import torch
import pandas as pd
from io import StringIO
from torch.utils.data import DataLoader

from config import BASE_PATH, CSV_PATH, TRAIN_IMG_DIR, BATCH_SIZE, MODEL_PATH, USE_S3
from data.prepare_data import RetinaDataset, get_transform
from models.model import create_model
from models.train import train_model
from utils.predict import predict_image

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No se encuentra el archivo CSV: {CSV_PATH}")

    all_data = pd.read_csv(CSV_PATH)
    sampled_data = all_data.sample(n=10000, random_state=42).reset_index(drop=True)

    buffer = StringIO()
    sampled_data.to_csv(buffer, index=False)
    buffer.seek(0)

    dataset = RetinaDataset(buffer, TRAIN_IMG_DIR if not USE_S3 else None, get_transform())
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = create_model()

    if os.path.exists(MODEL_PATH):
        print(f"Modelo ya existe. Cargando desde {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("Entrenando modelo desde cero...")
        train_model(model, train_loader)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Modelo guardado en {MODEL_PATH}")

    # Solo se hace predicción local si NO estás en S3
    if not USE_S3:
        test_img = os.path.join(TRAIN_IMG_DIR, "172_left.jpeg")
        if os.path.exists(test_img):
            pred = predict_image(model, test_img)
            print(f"Predicción para '172_left.jpeg': {pred}")
        else:
            print(f"Imagen no encontrada: {test_img}")
    else:
        print("Saltando predicción de ejemplo (modo S3 activado).")
