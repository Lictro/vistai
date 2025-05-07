import os
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from io import StringIO

# ========== CONFIGURACIÓN ==========
# Ajusta esta ruta al nombre real de tu disco
BASE_PATH = "/Volumes/WDDisk/diabetic-retinopathy-detection/retina-project"

CSV_PATH = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

# ========== DATASET PERSONALIZADO ==========
class RetinaDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path) if isinstance(csv_path, str) else pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        img_path = os.path.join(self.image_dir, f"{img_name}.jpeg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ========== TRANSFORMACIONES ==========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ========== CARGA DE DATOS ==========
all_data = pd.read_csv(CSV_PATH)
sampled_data = all_data.sample(n=500, random_state=42).reset_index(drop=True)  # Solo 500 muestras

# Guardamos temporalmente el CSV reducido en memoria
sample_csv_buffer = StringIO()
sampled_data.to_csv(sample_csv_buffer, index=False)
sample_csv_buffer.seek(0)

# Creamos el Dataset desde el buffer CSV
train_dataset = RetinaDataset(sample_csv_buffer, TRAIN_IMG_DIR, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========== MODELO CNN (ResNet18) ==========
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 5)
model = model.to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ========== ENTRENAMIENTO ==========
def train_model():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1} completed. Average loss: {total_loss / len(train_loader):.4f}")

# ========== PREDICCIÓN ==========
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(1), desc="Inference"):
            output = model(image)
        prediction = output.argmax(1).item()
    return prediction

# ========== MAIN ==========
if __name__ == "__main__":
    # Verifica que las rutas existen
    if not os.path.exists(BASE_PATH):
        raise FileNotFoundError(f"No se encuentra la carpeta: {BASE_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No se encuentra el archivo CSV: {CSV_PATH}")
    if not os.path.exists(TRAIN_IMG_DIR):
        raise FileNotFoundError(f"No se encuentra la carpeta de imágenes: {TRAIN_IMG_DIR}")

    # Entrenar el modelo
    train_model()

    # Predecir una imagen de ejemplo
    test_image_path = os.path.join(TRAIN_IMG_DIR, "184_left.jpeg")
    if os.path.exists(test_image_path):
        pred = predict_image(test_image_path)
        print(f"Predicted level for '184_left.jpeg': {pred}")
    else:
        print(f"Imagen de test no encontrada: {test_image_path}")
