import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import create_datasets
from src.models import create_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy


def train_model(
    model_name,
    dataset_path,
    epochs=15,
    batch_size=16,
    learning_rate=None,
    train_ratio=0.8,
    seed=42,
    patience=4
):

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Model: {model_name}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_dataset, test_dataset, class_names = create_datasets(
        dataset_path=dataset_path,
        train_ratio=train_ratio,
        seed=seed
    )

    print(f"Number of classes: {len(class_names)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = create_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=True,
        freeze_backbone=(model_name in ["resnet", "vit"])
    )

    model = model.to(device)

    # --------------------------------------------------
    # Learning rate
    # --------------------------------------------------

    if learning_rate is None:

        if model_name == "cnn":
            learning_rate = 0.001

        elif model_name == "resnet":
            learning_rate = 0.001

        elif model_name == "vit":
            learning_rate = 0.001

    print(f"Learning rate: {learning_rate}")

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Optimizer
    # Only parameters requiring gradients are optimized.
    # --------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    best_accuracy = 0.0
    epochs_without_improvement = 0

    os.makedirs("models", exist_ok=True)

    model_path = os.path.join(
        "models",
        f"{model_name}_best.pth"
    )

    for epoch in range(epochs):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        scheduler.step(test_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.4f}"
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if test_accuracy > best_accuracy:

            best_accuracy = test_accuracy
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "model_name": model_name,
                    "best_accuracy": best_accuracy
                },
                model_path
            )

            print(
                f"Saved best model to {model_path}"
            )

        else:

            epochs_without_improvement += 1

        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------

        if epochs_without_improvement >= patience:

            print(
                f"\nEarly stopping triggered after "
                f"{epoch + 1} epochs."
            )

            break

    print("\nTraining completed.")
    print(f"Best test accuracy: {best_accuracy:.4f}")
    print(f"Model saved at: {model_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a human action recognition model."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn", "resnet", "vit"]
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4
    )

    args = parser.parse_args()

    train_model(
        model_name=args.model,
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience
    )