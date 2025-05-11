import os
import torch

BASE_PATH = "/Volumes/WDDisk/diabetic-retinopathy-detection/retina-project"

CSV_PATH = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test")
MODEL_PATH = os.path.join(BASE_PATH, "retina_model.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

USE_S3 = True
S3_BUCKET = "retinographies"
S3_IMAGE_PREFIX = "train"
