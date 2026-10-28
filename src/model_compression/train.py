import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import IMAGE_SIZE, get_loader
from .model import LeNet

METRICS_PATH = Path("metrics/train.json")


def train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer
) -> float:
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    total, count = 0.0, 0
    for images, labels in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, count = 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            correct += int((model(images).argmax(dim=1) == labels).sum())
            count += len(labels)
    return correct / count


def train(
    prepared_data: str, learning_rate: float, epochs: int, model_path: str
) -> None:
    torch.manual_seed(0)
    train_loader = get_loader(f"{prepared_data}/train", IMAGE_SIZE, shuffle=True)
    test_loader = get_loader(f"{prepared_data}/test", IMAGE_SIZE)
    model = LeNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer)
        print(f"epoch {epoch + 1}/{epochs}: loss={loss:.3f}")
    accuracy = evaluate(model, test_loader)
    print(f"accuracy={accuracy:.3f}")
    torch.save(model.state_dict(), model_path)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps({"accuracy": accuracy, "loss": loss}))
