"""Fine-tune MobileNetV2 on a subset of the 140k Real and Fake Faces dataset.

Usage: python ml/train.py --data /path/to/real_vs_fake/real-vs-fake
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

OUT = Path(__file__).resolve().parent
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])


def subset(ds, per_class, seed=42):
    rng = random.Random(seed)
    idx = []
    for c in range(len(ds.classes)):
        class_idx = [i for i, t in enumerate(ds.targets) if t == c]
        idx += rng.sample(class_idx, per_class)
    return Subset(ds, idx)


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device)).argmax(1).cpu()
            y_true += y.tolist()
            y_pred += out.tolist()
    return y_true, y_pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()
    root = Path(args.data)

    train_ds = subset(datasets.ImageFolder(root / "train", TF), 2000)  # 4000 imgs
    val_ds = subset(datasets.ImageFolder(root / "valid", TF), 500)     # 1000 imgs
    test_ds = subset(datasets.ImageFolder(root / "test", TF), 500)     # 1000 imgs
    classes = datasets.ImageFolder(root / "test").classes              # ['fake','real']
    print("classes:", classes)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)
    test_dl = DataLoader(test_ds, batch_size=64)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        for i, (x, y) in enumerate(train_dl):
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            if i % 20 == 0:
                print(f"epoch {epoch+1} batch {i}/{len(train_dl)} loss {loss.item():.4f}")
        yt, yp = evaluate(model, val_dl, device)
        acc = accuracy_score(yt, yp)
        print(f"epoch {epoch+1} val acc {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUT / "model.pt")

    # Final evaluation on held-out test set using the best checkpoint
    model.load_state_dict(torch.load(OUT / "model.pt", map_location=device))
    yt, yp = evaluate(model, test_dl, device)
    metrics = {
        "accuracy": accuracy_score(yt, yp),
        "precision": precision_score(yt, yp),
        "recall": recall_score(yt, yp),
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
        "classes": classes,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
