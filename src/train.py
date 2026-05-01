import copy
import time
import torch

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device,
                num_epochs=20, save_path="best_model.pth", patience=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_batches = len(dataloaders[phase])

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

                if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                    print(f"{phase.upper()} Batch [{batch_idx + 1}/{total_batches}] - Loss: {loss.item():.4f}")

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

            print(f"{phase.upper()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "val":
                if scheduler is not None:
                    scheduler.step(epoch_loss)

                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_val_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), save_path)
                    print(f"Best model saved to {save_path}")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


def evaluate_model(model, dataloaders, dataset_sizes, device):
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels)

    test_acc = running_corrects.double().item() / dataset_sizes["test"]
    print(f"\nTEST Accuracy: {test_acc:.4f}")
    return test_acc