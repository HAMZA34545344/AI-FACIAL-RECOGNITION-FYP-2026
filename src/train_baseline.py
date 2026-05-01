import os
import sys
import json
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from src.dataloader import get_dataloaders
from models.cnn_baseline import CNNBaseline

DATA_DIR = r"C:\face-recognition-fyp\data\subset_200_7images_split"
OUTPUT_DIR = r"C:\face-recognition-fyp\outputs\cnn_baseline"

EPOCHS = 15
BATCH_SIZE = 16
IMAGE_SIZE = 224
LEARNING_RATE = 0.001
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, desc="Evaluation"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


def save_checkpoint(path, model, optimizer, epoch, best_val_acc, train_loss, train_acc, val_loss, val_acc, class_names):
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "class_names": class_names
    }
    torch.save(checkpoint, path)


def main():
    output_dir = prepare_output_dir()

    train_loader, val_loader, test_loader, class_names, num_classes = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS
    )

    model = CNNBaseline(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion, desc="Validation")
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")

        save_checkpoint(
            os.path.join(output_dir, "checkpoint.pth"),
            model, optimizer, epoch, best_val_acc,
            train_loss, train_acc, val_loss, val_acc, class_names
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(output_dir, "best_model.pth"))
            print("Best model saved.")

    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))

    model.load_state_dict(best_model_wts)
    test_loss, test_acc = evaluate(model, test_loader, criterion, desc="Testing")

    results = {
        "model": "CNNBaseline",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_classes": num_classes,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "class_names": class_names,
        "history": history
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        for k, v in results.items():
            if k != "history":
                f.write(f"{k}: {v}\n")

    print("Training completed.")
    print(f"Saved outputs in: {output_dir}")


if __name__ == "__main__":
    main()