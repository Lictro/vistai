import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from io import StringIO, BytesIO
from config import IMG_SIZE, USE_S3, S3_BUCKET, S3_IMAGE_PREFIX

if USE_S3:
    import boto3
    s3_client = boto3.client("s3")

class RetinaDataset(Dataset):
    def __init__(self, csv_path, image_dir=None, transform=None):
        self.data = pd.read_csv(csv_path) if isinstance(csv_path, str) else pd.read_csv(csv_path)
        self.image_dir = image_dir  # Solo se usa en modo local
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def load_image(self, img_name):
        if USE_S3:
            key = f"{S3_IMAGE_PREFIX}/{img_name}.jpeg"
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            return Image.open(BytesIO(response["Body"].read())).convert("RGB")
        else:
            img_path = os.path.join(self.image_dir, f"{img_name}.jpeg")
            return Image.open(img_path).convert("RGB")

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = self.load_image(img_name)
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
