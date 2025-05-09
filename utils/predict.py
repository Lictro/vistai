import torch
from PIL import Image
from tqdm import tqdm
from config import DEVICE
from data.prepare_data import get_transform

def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = get_transform()(image).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(1), desc="Inference"):
            output = model(image)
        prediction = output.argmax(1).item()
    return prediction
