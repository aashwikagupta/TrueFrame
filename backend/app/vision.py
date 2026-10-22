from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

MODEL_PATH = Path(__file__).resolve().parents[1] / "ml" / "model.pt"
CLASSES = ["fake", "real"]

# CPU inference avoids the disk space issues reported during training.
device = torch.device("cpu")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict(image):
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    index = int(probabilities.argmax().item())
    return CLASSES[index], float(probabilities[index].item())