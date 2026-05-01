import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader import get_dataloaders
from train import train_model, evaluate_model
from models.resnet_pretrained import ResNetPretrained

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Project root:", project_root)
    print("Loading dataset...")

    dataloaders, dataset_sizes, class_names, num_classes = get_dataloaders(
        data_dir=project_root / "data" / "split_dataset",
        batch_size=8,
        img_size=224,
        num_workers=0
    )

    print("Dataset loaded successfully.")
    print("Train size:", dataset_sizes["train"])
    print("Val size:", dataset_sizes["val"])
    print("Test size:", dataset_sizes["test"])
    print("Number of classes:", num_classes)

    print("Creating pretrained ResNet18 model...")
    model = ResNetPretrained(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001,
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2
    )

    save_path = project_root / "resnet_pretrained_best.pth"

    print("Starting training...")

    model = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=5,
        save_path=str(save_path),
        patience=2
    )

    print("Evaluating on test set...")
    evaluate_model(model, dataloaders, dataset_sizes, device)

if __name__ == "__main__":
    main()