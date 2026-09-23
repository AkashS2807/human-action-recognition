import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from torch.utils.data import DataLoader

from src.dataset import create_datasets
from src.models import create_model


def evaluate_model(model_name, dataset_path, batch_size=32):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load dataset
    _, test_dataset, class_names = create_datasets(
        dataset_path=dataset_path,
        train_ratio=0.8,
        seed=42
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Create model
    model = create_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False
    )

    model_path = os.path.join(
        "models",
        f"{model_name}_best.pth"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    # Accuracy
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print("\n" + "=" * 60)
    print(f"{model_name.upper()} EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")

    # Classification report
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # Save results
    os.makedirs("results", exist_ok=True)

    report_path = os.path.join(
        "results",
        f"{model_name}_classification_report.txt"
    )

    with open(report_path, "w") as file:
        file.write(
            f"Model: {model_name}\n"
            f"Accuracy: {accuracy:.4f}\n\n"
        )

        file.write(report)

    # Confusion matrix
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names)))
    )

    # Only display class names if the number of classes is manageable
    if len(class_names) <= 30:

        fig, ax = plt.subplots(
            figsize=(14, 12)
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )

        display.plot(
            ax=ax,
            xticks_rotation=90,
            cmap="Blues",
            colorbar=False
        )

        plt.title(
            f"{model_name.upper()} Confusion Matrix"
        )

        plt.tight_layout()

        confusion_path = os.path.join(
            "results",
            f"{model_name}_confusion_matrix.png"
        )

        plt.savefig(
            confusion_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"\nConfusion matrix saved to: "
            f"{confusion_path}"
        )

    print(
        f"Classification report saved to: "
        f"{report_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate a trained action recognition model."
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
        "--batch-size",
        type=int,
        default=32
    )

    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        dataset_path=args.dataset,
        batch_size=args.batch_size
    )