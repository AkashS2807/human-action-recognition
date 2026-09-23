import argparse

import torch
from PIL import Image

from src.dataset import get_transforms
from src.models import create_model


def predict_image(model_name, image_path, model_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    class_names = checkpoint["class_names"]

    model = create_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=(model_name in ["resnet", "vit"])
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    _, eval_transform = get_transforms()

    image = Image.open(image_path).convert("RGB")
    image = eval_transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = class_names[prediction.item()]
    confidence = confidence.item()

    print("\nPrediction")
    print("-" * 40)
    print(f"Action: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")

    print("\nTop 3 Predictions")
    print("-" * 40)

    top_probabilities, top_indices = torch.topk(
        probabilities,
        min(3, len(class_names)),
        dim=1
    )

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):
        print(
            f"{class_names[index.item()]}: "
            f"{probability.item() * 100:.2f}%"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Predict human action from an image."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn", "resnet", "vit"]
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True
    )

    args = parser.parse_args()

    predict_image(
        model_name=args.model,
        image_path=args.image,
        model_path=args.model_path
    )