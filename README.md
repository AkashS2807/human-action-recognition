# Human Action Recognition

A deep learning project for recognizing human actions from images extracted from videos using three different approaches:

- Custom CNN
- ResNet18
- Vision Transformer (ViT)

The project compares the performance of these models and provides an interactive Streamlit application for prediction.

---

## Project Overview

Human action recognition is a computer vision task where a machine learning model identifies the action being performed in an image or video.

This project uses frames extracted from the UCF101 human action recognition dataset.

The main objective is to compare:

1. A custom CNN trained from scratch
2. A pretrained ResNet18 using transfer learning
3. A pretrained Vision Transformer (ViT)

The best-performing model can then be used through the Streamlit web application.

---

## Models

### 1. Custom CNN

A custom convolutional neural network consisting of multiple convolution, batch normalization, ReLU and pooling layers.

### 2. ResNet18

A pretrained ResNet18 model using ImageNet weights. The final classification layer is replaced to match the action classes in the dataset.

### 3. Vision Transformer

A pretrained ViT-Base/16 model used for image-based action classification.

---

## Dataset

The project uses processed frames from the UCF101 dataset.

The dataset should follow the structure:

```text
processed_dataset/
├── ActionClass1/
│   ├── video1_frame0.jpg
│   ├── video1_frame1.jpg
│   └── ...
├── ActionClass2/
│   ├── video1_frame0.jpg
│   └── ...
└── ...
```

The dataset itself is not included in this repository because of its size.

---

## Data Splitting

Frames are grouped according to their original video before creating the train/test split.

This prevents frames from the same video from appearing in both training and testing sets and helps avoid data leakage.

---

## Project Structure

```text
Human-Action-Recognition/
│
├── notebooks/
│   ├── MlMiniProject_original.ipynb
│   └── MlMiniProject.ipynb
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
├── results/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Human-Action-Recognition
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

### Train CNN

```bash
python -m src.train --model cnn --dataset "PATH_TO_DATASET"
```

### Train ResNet18

```bash
python -m src.train --model resnet --dataset "PATH_TO_DATASET"
```

### Train Vision Transformer

```bash
python -m src.train --model vit --dataset "PATH_TO_DATASET"
```

You can also specify the number of epochs and batch size:

```bash
python -m src.train --model resnet --dataset "PATH_TO_DATASET" --epochs 10 --batch-size 32
```

The best model will be saved in the `models/` directory.

---

## Evaluation

Evaluate a trained model using:

```bash
python -m src.evaluate --model cnn --dataset "PATH_TO_DATASET"
```

For ResNet:

```bash
python -m src.evaluate --model resnet --dataset "PATH_TO_DATASET"
```

For ViT:

```bash
python -m src.evaluate --model vit --dataset "PATH_TO_DATASET"
```

The evaluation generates:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

Results are stored in the `results/` directory.

---

## Prediction

A trained model can be used to classify an individual image:

```bash
python -m src.predict --model resnet --image "path/to/image.jpg" --model-path "models/resnet_best.pth"
```

---

## Streamlit Application

Run the application using:

```bash
streamlit run app.py
```

The application allows users to:

1. Select a trained model
2. Upload an image
3. Predict the human action
4. View the prediction confidence
5. View the top predictions

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- timm
- NumPy
- Scikit-learn
- Matplotlib
- Pillow
- Streamlit

---

## Future Improvements

- Extend the application to support direct video uploads
- Use temporal information between video frames
- Improve model performance through hyperparameter tuning
- Add more comprehensive model comparison visualizations
- Deploy the Streamlit application

---

## Author

**Akash S**

Computer Science and Engineering
