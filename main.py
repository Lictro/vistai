import os
import torch
import pandas as pd
import csv
from io import StringIO
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BASE_PATH, CSV_PATH, TRAIN_IMG_DIR, BATCH_SIZE,
    MODEL_PATH, USE_S3, TEST_IMG_DIR, DEVICE
)
from data.prepare_data import RetinaDataset, get_transform
from models.model import create_model
from models.train import train_model
from utils.predict import predict_image

def find_image_path(test_img_dir, filename):
    for ext in [".jpeg", ".jpg", ".png"]:
        path = os.path.join(test_img_dir, f"{filename}{ext}")
        if os.path.exists(path):
            return path
    return None

def evaluate_model(model, result_csv_path, test_img_dir):
    correct = 0
    total = 0

    with open(result_csv_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Salta el encabezado
        rows = list(reader)
        total = len(rows)

        print("🔄 Procesando imágenes...")  # Mensaje general antes de iniciar el procesamiento

        for filename, label in tqdm(rows, total=total, desc="Imágenes", dynamic_ncols=True):
            img_path = os.path.join(test_img_dir, f"{filename}.jpeg")
            if os.path.exists(img_path):
                pred = predict_image(model, img_path)  # Obtiene la predicción de la imagen
                true_label = int(label)

                # Actualizar la precisión
                correct += (pred == true_label)
            else:
                print(f"Imagen no encontrada: {img_path}")

            # Actualiza la barra de progreso con el nombre de la imagen
            tqdm.write(f"Procesando {filename}.jpeg...")  # Esto se actualiza con cada imagen procesada

        accuracy = (correct / total) * 100
        print(f"✅ Precisión del modelo en {result_csv_path}: {accuracy:.2f}% ({correct}/{total} aciertos)")


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No se encuentra el archivo CSV: {CSV_PATH}")

    all_data = pd.read_csv(CSV_PATH)
    sampled_data = all_data.reset_index(drop=True)

    buffer = StringIO()
    sampled_data.to_csv(buffer, index=False)
    buffer.seek(0)

    dataset = RetinaDataset(buffer, TRAIN_IMG_DIR if not USE_S3 else None, get_transform())
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = create_model()
    model.to(DEVICE)

    if os.path.exists(MODEL_PATH):
        print(f"✅ Modelo ya existe. Cargando desde {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("🚀 Entrenando modelo desde cero...")
        train_model(model, train_loader)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"💾 Modelo guardado en {MODEL_PATH}")

    # Evaluar modelo con result.csv si está disponible
    result_csv_path = os.path.join(BASE_PATH, "result.csv")
    evaluate_model(model, result_csv_path, TEST_IMG_DIR)

    # Predicción de ejemplo si no se usa S3
    if not USE_S3:
        test_img = os.path.join(TEST_IMG_DIR, "9_left.jpeg")
        if os.path.exists(test_img):
            pred = predict_image(model, test_img)
            print(f"\n🔍 Predicción para '9_left.jpeg': {pred}")
        else:
            print(f"⚠️ Imagen no encontrada: {test_img}")
    else:
        print("📦 Saltando predicción de ejemplo (modo S3 activado).")
