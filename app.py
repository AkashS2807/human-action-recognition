import os

import streamlit as st
import torch
from PIL import Image

from src.dataset import get_transforms
from src.models import create_model


MODEL_PATHS = {
    "CNN": "models/cnn_best.pth",
    "ResNet18": "models/resnet_best.pth",
    "ViT-Tiny": "models/vit_best.pth",
}

MODEL_NAMES = {
    "CNN": "cnn",
    "ResNet18": "resnet",
    "ViT-Tiny": "vit",
}


@st.cache_resource
def load_model(model_name):

    model_key = MODEL_NAMES[model_name]
    model_path = MODEL_PATHS[model_name]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    class_names = checkpoint["class_names"]

    model = create_model(
        model_name=model_key,
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=(model_key in ["resnet", "vit"])
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, class_names, device


def predict(model, class_names, device, image):

    _, eval_transform = get_transforms()

    image = image.convert("RGB")
    image = eval_transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    top_probabilities, top_indices = torch.topk(
        probabilities,
        min(3, len(class_names)),
        dim=1
    )

    results = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):
        results.append(
            (
                class_names[index.item()],
                probability.item()
            )
        )

    return results


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Human Action Recognition",
    page_icon="🎯",
    layout="centered"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🎯 Human Action Recognition")

st.write(
    "Classify human actions from images using "
    "CNN, ResNet18, and Vision Transformer models."
)

st.divider()


# --------------------------------------------------
# Model selection
# --------------------------------------------------

selected_model = st.selectbox(
    "Select Model",
    list(MODEL_PATHS.keys())
)


# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    model_path = MODEL_PATHS[selected_model]

    if not os.path.exists(model_path):

        st.error(
            f"Model file not found: {model_path}"
        )

    else:

        if st.button(
            "Predict Action",
            type="primary"
        ):

            with st.spinner("Analyzing image..."):

                model, class_names, device = load_model(
                    selected_model
                )

                results = predict(
                    model,
                    class_names,
                    device,
                    image
                )

            predicted_class = results[0][0]
            confidence = results[0][1]

            st.success(
                f"Predicted Action: {predicted_class}"
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.subheader("Top 3 Predictions")

            for action, probability in results:

                st.write(
                    f"**{action}** — "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )


st.divider()

st.caption(
    "Human Action Recognition | "
    "CNN • ResNet18 • ViT-Tiny"
)