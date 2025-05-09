import torch.nn as nn
from torchvision import models

def create_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 5)
    return model
