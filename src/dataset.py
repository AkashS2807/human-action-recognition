import os
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_transforms(image_size=224):
    """
    Create transformations for training and evaluation.
    """

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


def get_video_name(image_path):
    """
    Extract the original video name from a frame filename.

    Example:
        v_Basketball_g01_c01_frame0.jpg
        -> v_Basketball_g01_c01
    """

    filename = os.path.basename(image_path)

    if "_frame" in filename:
        return filename.split("_frame")[0]

    return os.path.splitext(filename)[0]


def create_video_split(dataset, train_ratio=0.8, seed=42):
    """
    Split the dataset at the VIDEO level while keeping every class
    represented in both training and testing whenever possible.

    This prevents frames from the same video appearing in both sets.
    """

    class_video_groups = defaultdict(lambda: defaultdict(list))

    # Group frames by class and video
    for index, (image_path, label) in enumerate(dataset.samples):

        video_name = get_video_name(image_path)

        class_video_groups[label][video_name].append(index)

    rng = np.random.default_rng(seed)

    train_indices = []
    test_indices = []

    # Split videos separately for each class
    for label in sorted(class_video_groups.keys()):

        videos = list(class_video_groups[label].keys())

        rng.shuffle(videos)

        number_of_videos = len(videos)

        # If there is only one video, it cannot be present
        # in both training and testing without data leakage.
        if number_of_videos == 1:

            train_videos = videos
            test_videos = []

        else:

            # Keep at least one video for testing
            # and at least one video for training.
            test_count = max(
                1,
                round(number_of_videos * (1 - train_ratio))
            )

            test_count = min(
                test_count,
                number_of_videos - 1
            )

            test_videos = videos[:test_count]
            train_videos = videos[test_count:]

        # Add frame indices
        for video in train_videos:
            train_indices.extend(
                class_video_groups[label][video]
            )

        for video in test_videos:
            test_indices.extend(
                class_video_groups[label][video]
            )

    # Shuffle final indices
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices


class ActionDataset(Dataset):
    """
    PyTorch Dataset for human action recognition frames.
    """

    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def create_datasets(dataset_path, train_ratio=0.8, seed=42):
    """
    Load the image dataset and create training and testing datasets.
    """

    base_dataset = datasets.ImageFolder(dataset_path)

    train_indices, test_indices = create_video_split(
        base_dataset,
        train_ratio=train_ratio,
        seed=seed
    )

    train_transform, eval_transform = get_transforms()

    train_samples = [
        base_dataset.samples[index]
        for index in train_indices
    ]

    test_samples = [
        base_dataset.samples[index]
        for index in test_indices
    ]

    train_dataset = ActionDataset(
        train_samples,
        transform=train_transform
    )

    test_dataset = ActionDataset(
        test_samples,
        transform=eval_transform
    )

    return (
        train_dataset,
        test_dataset,
        base_dataset.classes
    )