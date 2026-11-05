"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 6 — MODEL OPTIMIZATION AND DEPLOYMENT

LESSON 76 — KNOWLEDGE DISTILLATION
======================================================================

OBJECTIVE
---------

Train a compact student neural network using knowledge transferred
from a larger teacher neural network.

EXPERIMENTS
-----------

1. Train a teacher CNN.
2. Train a student using ground-truth labels only.
3. Train another student using knowledge distillation.
4. Compare classification accuracy.
5. Compare inference latency.
6. Compare parameter counts.
7. Compare model storage.
8. Visualize temperature-scaled probabilities.
9. Visualize training curves.
10. Export a JSON experiment report.

No pretrained models or pretrained weights are used.

======================================================================
1. TEACHER AND STUDENT
======================================================================

Teacher:
    Larger model
    More parameters
    Greater representational capacity

Student:
    Smaller model
    Fewer parameters
    Lower computational requirements

The student learns to imitate the teacher.

======================================================================
2. HARD LABELS
======================================================================

Suppose the correct class is plastic:

    y = [0, 0, 1, 0, 0]

Cross entropy:

    L_CE = -sum(y_i * log(P_s[i]))

======================================================================
3. SOFT LABELS
======================================================================

A teacher might produce:

    cardboard: 0.05
    glass:     0.10
    plastic:   0.65
    metal:     0.15
    paper:     0.05

This probability distribution contains more information than
the correct class alone.

The student learns relationships between classes.

======================================================================
4. TEMPERATURE
======================================================================

Temperature-scaled softmax:

    P_i(T) = exp(z_i / T) / sum(exp(z_j / T))

When T increases:

    - Distribution becomes softer.
    - Smaller probabilities become more visible.
    - Relative class similarities become easier to observe.

T = 1:
    Ordinary softmax.

T > 1:
    Softer probability distribution.

======================================================================
5. KL-DIVERGENCE
======================================================================

    KL(P_t || P_s)

    = sum(P_t[i] * log(P_t[i] / P_s[i]))

PyTorch:

    F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    )

======================================================================
6. DISTILLATION LOSS
======================================================================

    L = (1-alpha)*L_CE + alpha*T^2*L_KL

Where:

    T     = temperature
    alpha = distillation weight
    L_CE  = hard-label cross entropy
    L_KL  = teacher-student KL divergence

The T^2 factor compensates for gradient scaling caused by
temperature.

======================================================================
7. TEACHER FREEZING
======================================================================

During student training:

    Teacher:
        eval()
        requires_grad_(False)
        torch.no_grad()

    Student:
        train()
        optimizer updates student parameters

The teacher is never optimized during distillation.

======================================================================
8. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    RGB cameras
        |
        v
    Large teacher perception model
        |
        v
    Teacher predictions
        |
        v
    Knowledge distillation
        |
        v
    Compact student perception model
        |
        v
    Object localization
        |
        v
    ACT manipulation policy
        |
        v
    Robot grasping

For deployment, evaluate:

    - Detection mAP
    - Small-object recall
    - Transparent object recognition
    - Occlusion robustness
    - Inference latency
    - GPU memory
    - Real robot grasp success rate

Classification accuracy alone is insufficient.

======================================================================
NEXT LESSON
-----------

Lesson 77 — ONNX Model Export.
======================================================================
"""

from __future__ import annotations

import argparse
import json
import random
import time

from pathlib import Path
from typing import Any

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)

import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

IMAGE_SIZE = 64

NUM_CLASSES = 10

DEFAULT_OUTPUT_DIR = Path("runs/lesson76")

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "organic",
    "fabric",
    "wood",
    "ceramic",
    "other",
]


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed: int = 42) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ======================================================================
# DEVICE
# ======================================================================

def select_device(
    requested: str,
) -> torch.device:

    if requested == "cpu":

        return torch.device("cpu")

    if requested == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        return torch.device("cuda")

    if requested == "auto":

        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    raise ValueError(
        f"Unsupported device: {requested}"
    )


def synchronize(
    device: torch.device,
) -> None:

    if device.type == "cuda":

        torch.cuda.synchronize(device)


# ======================================================================
# SYNTHETIC GEOMETRIC DATASET
# ======================================================================

class SyntheticWasteDataset(Dataset):
    """
    Controlled geometric classification dataset.

    Each class has a different geometric pattern.

    This dataset is used to verify the learning pipeline.
    It is NOT a substitute for real waste photographs.
    """

    def __init__(
        self,
        samples_per_class: int = 100,
        image_size: int = IMAGE_SIZE,
        seed: int = 42,
    ):

        self.images = []

        self.labels = []

        self.image_size = image_size

        rng = np.random.default_rng(seed)

        y, x = np.mgrid[
            0:image_size,
            0:image_size,
        ]

        center = image_size / 2

        for class_id in range(NUM_CLASSES):

            for _ in range(samples_per_class):

                image = np.zeros(
                    (3, image_size, image_size),
                    dtype=np.float32,
                )

                offset_x = rng.integers(-5, 6)

                offset_y = rng.integers(-5, 6)

                cx = center + offset_x

                cy = center + offset_y

                radius = rng.integers(
                    image_size // 7,
                    image_size // 4,
                )

                if class_id == 0:

                    pattern = (
                        np.abs((y - cy) % 12) < 3
                    )

                elif class_id == 1:

                    pattern = (
                        np.abs((x - cx) % 12) < 3
                    )

                elif class_id == 2:

                    pattern = (
                        np.abs((x + y) % 14) < 3
                    )

                elif class_id == 3:

                    distance = np.sqrt(
                        (x - cx) ** 2
                        +
                        (y - cy) ** 2
                    )

                    pattern = (
                        np.abs(distance - radius) < 4
                    )

                elif class_id == 4:

                    distance = np.sqrt(
                        (x - cx) ** 2
                        +
                        (y - cy) ** 2
                    )

                    pattern = distance < radius

                elif class_id == 5:

                    pattern = (
                        (np.abs(x - cx) < radius)
                        &
                        (np.abs(y - cy) < radius)
                    )

                elif class_id == 6:

                    pattern = (
                        (np.abs(x - cx) < 4)
                        |
                        (np.abs(y - cy) < 4)
                    )

                elif class_id == 7:

                    pattern = (
                        ((x // 8 + y // 8) % 2) == 0
                    )

                elif class_id == 8:

                    pattern = (
                        np.abs(x - y - offset_x) < 4
                    ) | (
                        np.abs(
                            x + y - image_size - offset_y
                        ) < 4
                    )

                else:

                    pattern = (
                        (
                            (np.abs(x - cx) < 3)
                            |
                            (np.abs(y - cy) < 3)
                        )
                        &
                        (((x + y) % 16) < 9)
                    )

                channel = class_id % 3

                image[channel, pattern] = 1.0

                image[(channel + 1) % 3, pattern] = 0.5

                noise = rng.normal(
                    0.0,
                    0.06,
                    size=image.shape,
                ).astype(np.float32)

                image = np.clip(
                    image + noise,
                    0.0,
                    1.0,
                )

                self.images.append(image)

                self.labels.append(class_id)

        self.images = torch.from_numpy(
            np.asarray(
                self.images,
                dtype=np.float32,
            )
        )

        self.labels = torch.tensor(
            self.labels,
            dtype=torch.long,
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        image = self.images[index]

        label = self.labels[index]

        image = image * 2.0 - 1.0

        return image, label


# ======================================================================
# REAL IMAGE FOLDER DATASET
# ======================================================================

class ImageFolderDataset(Dataset):
    """
    Minimal ImageFolder-style dataset.

    Expected directory:

        dataset/
            cardboard/
                image001.jpg
                image002.jpg
            plastic/
                image001.jpg
                image002.jpg
            metal/
                image001.jpg

    Each class is represented by a directory.
    """

    EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    def __init__(
        self,
        root: str,
        image_size: int = IMAGE_SIZE,
    ):

        from PIL import Image

        self.root = Path(root)

        if not self.root.exists():

            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root}"
            )

        class_dirs = sorted(
            path
            for path in self.root.iterdir()
            if path.is_dir()
        )

        if len(class_dirs) < 2:

            raise ValueError(
                "ImageFolder requires at least two class directories."
            )

        self.class_to_idx = {

            path.name: index

            for index, path in enumerate(class_dirs)

        }

        self.classes = list(
            self.class_to_idx.keys()
        )

        self.samples = []

        for class_dir in class_dirs:

            class_id = self.class_to_idx[
                class_dir.name
            ]

            for path in sorted(
                class_dir.rglob("*")
            ):

                if path.suffix.lower() in self.EXTENSIONS:

                    self.samples.append(
                        (path, class_id)
                    )

        if not self.samples:

            raise ValueError(
                "No supported images found."
            )

        self.image_size = image_size

        print(
            f"Loaded {len(self.samples)} images "
            f"from {len(self.classes)} classes."
        )

        print(
            f"Classes: {self.classes}"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        from PIL import Image

        path, label = self.samples[index]

        with Image.open(path) as image:

            image = image.convert("RGB")

            image = image.resize(
                (
                    self.image_size,
                    self.image_size,
                )
            )

            array = np.asarray(
                image,
                dtype=np.float32,
            ) / 255.0

        array = np.transpose(
            array,
            (2, 0, 1),
        )

        tensor = torch.from_numpy(
            array.copy()
        )

        tensor = tensor * 2.0 - 1.0

        return tensor, label


# ======================================================================
# CONVOLUTIONAL BLOCK
# ======================================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = True,
    ):

        super().__init__()

        layers = [

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

        ]

        if pool:

            layers.append(
                nn.MaxPool2d(2)
            )

        self.block = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return self.block(x)


# ======================================================================
# TEACHER CNN
# ======================================================================

class TeacherCNN(nn.Module):

    def __init__(
        self,
        num_classes: int,
    ):

        super().__init__()

        self.features = nn.Sequential(

            ConvBlock(3, 32),

            ConvBlock(32, 64),

            ConvBlock(64, 128),

            ConvBlock(128, 256),

        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, num_classes),

        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        return self.classifier(x)


# ======================================================================
# STUDENT CNN
# ======================================================================

class StudentCNN(nn.Module):

    def __init__(
        self,
        num_classes: int,
    ):

        super().__init__()

        self.features = nn.Sequential(

            ConvBlock(3, 16),

            ConvBlock(16, 32),

            ConvBlock(32, 64),

        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64, num_classes),

        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        return self.classifier(x)


# ======================================================================
# MODEL STATISTICS
# ======================================================================

def count_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def model_storage_bytes(
    model: nn.Module,
) -> int:

    total = 0

    for parameter in model.parameters():

        total += (
            parameter.numel()
            *
            parameter.element_size()
        )

    for buffer in model.buffers():

        total += (
            buffer.numel()
            *
            buffer.element_size()
        )

    return total


# ======================================================================
# TEMPERATURE SOFTMAX
# ======================================================================

def temperature_softmax(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:

    if temperature <= 0:

        raise ValueError(
            "Temperature must be positive."
        )

    return F.softmax(
        logits / temperature,
        dim=1,
    )


# ======================================================================
# KNOWLEDGE DISTILLATION LOSS
# ======================================================================

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> tuple[torch.Tensor, dict[str, float]]:

    """
    Combined objective:

        L = (1-alpha)*CE + alpha*T^2*KL

    Teacher logits are detached.

    Gradients are calculated only for the student.
    """

    if temperature <= 0:

        raise ValueError(
            "Temperature must be positive."
        )

    if not 0.0 <= alpha <= 1.0:

        raise ValueError(
            "Alpha must be between 0 and 1."
        )

    if student_logits.shape != teacher_logits.shape:

        raise ValueError(
            "Teacher and student logits must have identical shapes."
        )

    if student_logits.ndim != 2:

        raise ValueError(
            "Logits must have shape [batch_size, num_classes]."
        )

    hard_loss = F.cross_entropy(
        student_logits,
        labels,
    )

    teacher_probabilities = F.softmax(
        teacher_logits.detach() / temperature,
        dim=1,
    )

    student_log_probabilities = F.log_softmax(
        student_logits / temperature,
        dim=1,
    )

    soft_loss = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="batchmean",
    )

    weighted_soft_loss = (
        temperature ** 2
    ) * soft_loss

    total_loss = (
        (1.0 - alpha) * hard_loss
        +
        alpha * weighted_soft_loss
    )

    return total_loss, {

        "hard_loss": float(
            hard_loss.detach().item()
        ),

        "soft_loss": float(
            weighted_soft_loss.detach().item()
        ),

        "total_loss": float(
            total_loss.detach().item()
        ),

    }


# ======================================================================
# TEACHER TRAINING
# ======================================================================

def train_teacher(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> list[dict[str, float]]:

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    history = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = F.cross_entropy(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                *
                images.shape[0]
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.numel()

        epoch_loss = running_loss / max(
            total,
            1,
        )

        epoch_accuracy = correct / max(
            total,
            1,
        )

        history.append({

            "loss": epoch_loss,

            "accuracy": epoch_accuracy,

        })

        print(
            f"Teacher epoch {epoch + 1:02d}/{epochs:02d} "
            f"| loss={epoch_loss:.4f} "
            f"| accuracy={epoch_accuracy:.4f}"
        )

    return history


# ======================================================================
# STUDENT TRAINING
# ======================================================================

def train_student(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    temperature: float,
    alpha: float,
    use_distillation: bool,
) -> list[dict[str, float]]:

    student.to(device)

    teacher.to(device)

    # Freeze teacher.

    teacher.eval()

    teacher.requires_grad_(False)

    # Only student parameters are optimized.

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    history = []

    for epoch in range(epochs):

        student.train()

        # Keep teacher in evaluation mode.

        teacher.eval()

        total_loss = 0.0

        total_hard_loss = 0.0

        total_soft_loss = 0.0

        correct = 0

        total = 0

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            student_logits = student(images)

            if use_distillation:

                with torch.no_grad():

                    teacher_logits = teacher(
                        images
                    )

                loss, loss_parts = distillation_loss(

                    student_logits,

                    teacher_logits,

                    labels,

                    temperature=temperature,

                    alpha=alpha,

                )

                total_hard_loss += (
                    loss_parts["hard_loss"]
                    *
                    images.shape[0]
                )

                total_soft_loss += (
                    loss_parts["soft_loss"]
                    *
                    images.shape[0]
                )

            else:

                loss = F.cross_entropy(
                    student_logits,
                    labels,
                )

                total_hard_loss += (
                    loss.item()
                    *
                    images.shape[0]
                )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                *
                images.shape[0]
            )

            predictions = student_logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.numel()

        epoch_result = {

            "loss": total_loss / max(total, 1),

            "hard_loss": total_hard_loss / max(total, 1),

            "soft_loss": total_soft_loss / max(total, 1),

            "accuracy": correct / max(total, 1),

        }

        history.append(
            epoch_result
        )

        mode = (
            "distillation"
            if use_distillation
            else "hard-label"
        )

        print(
            f"Student [{mode}] "
            f"epoch {epoch + 1:02d}/{epochs:02d} "
            f"| loss={epoch_result['loss']:.4f} "
            f"| accuracy={epoch_result['accuracy']:.4f}"
        )

    # Verify that teacher parameters were not modified.

    if any(
        parameter.requires_grad
        for parameter in teacher.parameters()
    ):

        raise RuntimeError(
            "Teacher unexpectedly has trainable parameters."
        )

    return history


# ======================================================================
# EVALUATION
# ======================================================================

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:

    model.eval()

    model.to(device)

    correct = 0

    total = 0

    total_loss = 0.0

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            loss = F.cross_entropy(
                logits,
                labels,
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.numel()

            total_loss += (
                loss.item()
                *
                labels.numel()
            )

    return {

        "accuracy": correct / max(total, 1),

        "loss": total_loss / max(total, 1),

        "samples": total,

    }


# ======================================================================
# INFERENCE BENCHMARK
# ======================================================================

def benchmark_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup: int = 10,
    iterations: int = 50,
) -> dict[str, float]:

    if iterations <= 0:

        raise ValueError(
            "Benchmark iterations must be positive."
        )

    model.eval()

    model.to(device)

    first_images, _ = next(
        iter(loader)
    )

    images = first_images[:1].to(device)

    with torch.inference_mode():

        for _ in range(warmup):

            model(images)

    synchronize(device)

    timings = []

    for _ in range(iterations):

        synchronize(device)

        start = time.perf_counter()

        with torch.inference_mode():

            model(images)

        synchronize(device)

        elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        timings.append(elapsed)

    values = np.asarray(
        timings,
        dtype=np.float64,
    )

    mean_latency = float(
        np.mean(values)
    )

    return {

        "mean_latency_ms": mean_latency,

        "p50_latency_ms": float(
            np.percentile(values, 50)
        ),

        "p95_latency_ms": float(
            np.percentile(values, 95)
        ),

        "images_per_second": (
            1000.0 / mean_latency
            if mean_latency > 0
            else 0.0
        ),

    }


# ======================================================================
# TEMPERATURE DEMONSTRATION
# ======================================================================

def demonstrate_temperature(
    output_dir: Path,
) -> dict[str, list[float]]:

    logits = torch.tensor(
        [[
            5.0,
            3.0,
            1.0,
            0.5,
            -1.0,
        ]]
    )

    temperatures = [
        1.0,
        2.0,
        4.0,
        8.0,
    ]

    results = {}

    for temperature in temperatures:

        probabilities = temperature_softmax(
            logits,
            temperature,
        )

        results[str(temperature)] = (
            probabilities[0].tolist()
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = np.arange(
        logits.shape[1]
    )

    width = 0.18

    for index, temperature in enumerate(
        temperatures
    ):

        values = results[str(temperature)]

        ax.bar(
            x + index * width,
            values,
            width,
            label=f"T={temperature}",
        )

    ax.set_xticks(
        x + width * 1.5
    )

    ax.set_xticklabels(
        [f"Class {i}" for i in range(5)]
    )

    ax.set_ylabel("Probability")

    ax.set_title(
        "Effect of Softmax Temperature"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "temperature_comparison.png",
        dpi=150,
    )

    plt.close(fig)

    return results


# ======================================================================
# TRAINING VISUALIZATION
# ======================================================================

def plot_training_history(
    teacher_history: list[dict[str, float]],
    hard_history: list[dict[str, float]],
    distill_history: list[dict[str, float]],
    output_dir: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        [x["loss"] for x in teacher_history],
        label="Teacher",
    )

    ax.plot(
        [x["loss"] for x in hard_history],
        label="Student: hard labels",
    )

    ax.plot(
        [x["loss"] for x in distill_history],
        label="Student: distillation",
    )

    ax.set_xlabel("Epoch")

    ax.set_ylabel("Training loss")

    ax.set_title(
        "Training Loss Comparison"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "training_loss.png",
        dpi=150,
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        [x["accuracy"] for x in teacher_history],
        label="Teacher",
    )

    ax.plot(
        [x["accuracy"] for x in hard_history],
        label="Student: hard labels",
    )

    ax.plot(
        [x["accuracy"] for x in distill_history],
        label="Student: distillation",
    )

    ax.set_xlabel("Epoch")

    ax.set_ylabel("Training accuracy")

    ax.set_title(
        "Training Accuracy Comparison"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "training_accuracy.png",
        dpi=150,
    )

    plt.close(fig)


# ======================================================================
# FINAL COMPARISON VISUALIZATION
# ======================================================================

def plot_final_comparison(
    results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:

    names = list(
        results.keys()
    )

    accuracies = [

        results[name]["evaluation"]["accuracy"] * 100

        for name in names

    ]

    parameters = [

        results[name]["parameters"] / 1000

        for name in names

    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        names,
        accuracies,
    )

    ax.set_ylabel("Accuracy (%)")

    ax.set_title(
        "Final Classification Accuracy"
    )

    ax.set_ylim(0, 100)

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "final_accuracy.png",
        dpi=150,
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        names,
        parameters,
    )

    ax.set_ylabel("Parameters (thousands)")

    ax.set_title(
        "Model Parameter Comparison"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "parameter_comparison.png",
        dpi=150,
    )

    plt.close(fig)


# ======================================================================
# PREDICTION EXAMPLES
# ======================================================================

def save_prediction_examples(
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    output_dir: Path,
) -> None:

    images, labels = next(
        iter(loader)
    )

    images_device = images.to(device)

    teacher.eval()

    student.eval()

    with torch.inference_mode():

        teacher_logits = teacher(
            images_device
        )

        student_logits = student(
            images_device
        )

        teacher_prob = F.softmax(
            teacher_logits,
            dim=1,
        )

        student_prob = F.softmax(
            student_logits,
            dim=1,
        )

    count = min(
        8,
        images.shape[0],
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 7),
    )

    for index, ax in enumerate(
        axes.flat
    ):

        if index >= count:

            ax.axis("off")

            continue

        image = images[index].cpu()

        image = image.permute(
            1,
            2,
            0,
        ).numpy()

        image = (
            image + 1.0
        ) / 2.0

        image = np.clip(
            image,
            0.0,
            1.0,
        )

        teacher_class = int(
            teacher_prob[index].argmax()
        )

        student_class = int(
            student_prob[index].argmax()
        )

        true_class = int(
            labels[index]
        )

        ax.imshow(image)

        ax.set_title(
            f"True: {class_names[true_class]}\n"
            f"Teacher: {class_names[teacher_class]}\n"
            f"Student: {class_names[student_class]}",
            fontsize=8,
        )

        ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        output_dir / "prediction_examples.png",
        dpi=150,
    )

    plt.close(fig)


# ======================================================================
# TESTS
# ======================================================================

def run_tests() -> None:

    print("\nRunning Lesson 76 tests...\n")

    set_seed(42)

    teacher = TeacherCNN(
        NUM_CLASSES
    )

    student = StudentCNN(
        NUM_CLASSES
    )

    x = torch.randn(
        4,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    # Test 1:
    # Teacher forward pass.
    #
    # no_grad is appropriate here because the teacher is frozen.

    with torch.no_grad():

        teacher_output = teacher(x)

    assert teacher_output.shape == (
        4,
        NUM_CLASSES,
    )

    print("Test 1 passed: Teacher forward.")

    # Test 2:
    # Student forward pass.
    #
    # IMPORTANT:
    # Do NOT use torch.no_grad() here.
    # We need a computation graph for Test 8.

    student_output = student(x)

    assert student_output.shape == (
        4,
        NUM_CLASSES,
    )

    assert student_output.requires_grad

    print("Test 2 passed: Student forward with gradients.")

    # Test 3:
    # Student has fewer parameters.

    teacher_parameters = count_parameters(
        teacher
    )

    student_parameters = count_parameters(
        student
    )

    assert student_parameters < teacher_parameters

    print(
        "Test 3 passed: Student is smaller "
        f"({student_parameters:,} < {teacher_parameters:,})."
    )

    # Test 4:
    # Temperature 1 equals ordinary softmax.

    logits = torch.randn(
        4,
        NUM_CLASSES,
    )

    p1 = temperature_softmax(
        logits,
        1.0,
    )

    expected = F.softmax(
        logits,
        dim=1,
    )

    assert torch.allclose(
        p1,
        expected,
        atol=1e-6,
    )

    print("Test 4 passed: Temperature T=1.")

    # Test 5:
    # Probabilities sum to one.

    p4 = temperature_softmax(
        logits,
        4.0,
    )

    assert torch.allclose(
        p4.sum(dim=1),
        torch.ones(4),
        atol=1e-6,
    )

    print("Test 5 passed: Probability normalization.")

    # Test 6:
    # Higher temperature increases entropy.

    entropy_1 = -torch.sum(
        p1 * torch.log(p1.clamp_min(1e-12)),
        dim=1,
    ).mean()

    entropy_4 = -torch.sum(
        p4 * torch.log(p4.clamp_min(1e-12)),
        dim=1,
    ).mean()

    assert entropy_4 > entropy_1

    print("Test 6 passed: Higher temperature increases entropy.")

    # Test 7:
    # Distillation loss is finite.

    labels = torch.randint(
        0,
        NUM_CLASSES,
        (4,),
    )

    loss, components = distillation_loss(
        student_output,
        teacher_output,
        labels,
        temperature=4.0,
        alpha=0.7,
    )

    assert torch.isfinite(loss)

    print("Test 7 passed: Distillation loss is finite.")

    # Test 8:
    # Student receives gradients.

    student.zero_grad(
        set_to_none=True
    )

    loss.backward()

    gradient_exists = any(

        parameter.grad is not None

        and torch.isfinite(parameter.grad).all()

        for parameter in student.parameters()

    )

    assert gradient_exists

    print("Test 8 passed: Student gradients.")

    # Test 9:
    # Teacher is not part of the gradient graph.

    assert not teacher_output.requires_grad

    print("Test 9 passed: Teacher inference graph.")

    # Test 10:
    # Alpha zero equals ordinary cross entropy.

    student_logits = torch.randn(
        4,
        NUM_CLASSES,
        requires_grad=True,
    )

    teacher_logits = torch.randn(
        4,
        NUM_CLASSES,
    )

    combined, _ = distillation_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature=3.0,
        alpha=0.0,
    )

    ordinary = F.cross_entropy(
        student_logits,
        labels,
    )

    assert torch.allclose(
        combined,
        ordinary,
        atol=1e-6,
    )

    print("Test 10 passed: Alpha zero equals CE.")

    # Test 11:
    # Alpha one uses only the soft loss.

    combined, components = distillation_loss(
        student_logits,
        teacher_logits,
        labels,
        temperature=3.0,
        alpha=1.0,
    )

    assert np.isclose(
        combined.item(),
        components["soft_loss"],
        atol=1e-6,
    )

    print("Test 11 passed: Alpha one uses soft loss.")

    # Test 12:
    # Synthetic dataset shape and labels.

    dataset = SyntheticWasteDataset(
        samples_per_class=2,
        image_size=IMAGE_SIZE,
    )

    image, label = dataset[0]

    assert image.shape == (
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert 0 <= int(label) < NUM_CLASSES

    print("Test 12 passed: Synthetic dataset.")

    # Test 13:
    # Model storage measurement.

    assert model_storage_bytes(student) > 0

    print("Test 13 passed: Model storage measurement.")

    # Test 14:
    # Invalid temperature must raise an error.

    try:

        temperature_softmax(
            logits,
            0.0,
        )

        raise AssertionError(
            "Invalid temperature accepted."
        )

    except ValueError:

        pass

    print("Test 14 passed: Temperature validation.")

    # Test 15:
    # Invalid alpha must raise an error.

    try:

        distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            temperature=3.0,
            alpha=1.5,
        )

        raise AssertionError(
            "Invalid alpha accepted."
        )

    except ValueError:

        pass

    print("Test 15 passed: Alpha validation.")

    # Test 16:
    # Teacher freezing.

    teacher.requires_grad_(False)

    assert all(
        not parameter.requires_grad
        for parameter in teacher.parameters()
    )

    print("Test 16 passed: Teacher freezing.")

    # Test 17:
    # Teacher and student output shapes must match for KD.

    try:

        distillation_loss(
            torch.randn(4, 5),
            torch.randn(4, 6),
            labels,
        )

        raise AssertionError(
            "Mismatched logits accepted."
        )

    except ValueError:

        pass

    print("Test 17 passed: Logit dimension validation.")

    # Test 18:
    # Temperature must be positive.

    try:

        distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            temperature=-1.0,
        )

        raise AssertionError(
            "Negative temperature accepted."
        )

    except ValueError:

        pass

    print("Test 18 passed: Distillation temperature validation.")

    print("\nAll 18 tests passed.")


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Lesson 76 — Knowledge Distillation"
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Real ImageFolder dataset root.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--teacher-epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--student-epochs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.check_only:

        run_tests()

        return

    if args.image_size < 32:

        raise ValueError(
            "Image size must be at least 32."
        )

    if args.teacher_epochs <= 0:

        raise ValueError(
            "Teacher epochs must be positive."
        )

    if args.student_epochs <= 0:

        raise ValueError(
            "Student epochs must be positive."
        )

    if args.batch_size <= 0:

        raise ValueError(
            "Batch size must be positive."
        )

    if args.learning_rate <= 0:

        raise ValueError(
            "Learning rate must be positive."
        )

    if args.temperature <= 0:

        raise ValueError(
            "Temperature must be positive."
        )

    if not 0.0 <= args.alpha <= 1.0:

        raise ValueError(
            "Alpha must be between 0 and 1."
        )

    if args.workers < 0:

        raise ValueError(
            "Workers cannot be negative."
        )

    if args.benchmark_iterations <= 0:

        raise ValueError(
            "Benchmark iterations must be positive."
        )

    set_seed(42)

    device = select_device(
        args.device
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n" + "=" * 75)

    print("LESSON 76 — KNOWLEDGE DISTILLATION")

    print("=" * 75)

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(device)}"
        )

    print(
        f"Temperature: {args.temperature}"
    )

    print(
        f"Alpha: {args.alpha}"
    )

    # --------------------------------------------------------------
    # DATASET
    # --------------------------------------------------------------

    if args.data_dir:

        dataset = ImageFolderDataset(
            args.data_dir,
            image_size=args.image_size,
        )

        class_names = dataset.classes

        num_classes = len(
            class_names
        )

        dataset_type = "real_image_folder"

    else:

        dataset = SyntheticWasteDataset(
            samples_per_class=args.samples_per_class,
            image_size=args.image_size,
        )

        class_names = CLASS_NAMES

        num_classes = NUM_CLASSES

        dataset_type = "synthetic_geometric"

        print(
            "\nWARNING: Using synthetic geometric images."
        )

        print(
            "Use --data-dir with real photographs for real-world evaluation."
        )

    if len(dataset) < 4:

        raise ValueError(
            "Dataset needs at least four samples."
        )

    # --------------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------------

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(
        42
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"\nDataset: {dataset_type}"
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    print(
        f"Classes: {num_classes}"
    )

    # --------------------------------------------------------------
    # TEMPERATURE DEMONSTRATION
    # --------------------------------------------------------------

    temperature_results = demonstrate_temperature(
        output_dir
    )

    # --------------------------------------------------------------
    # TEACHER
    # --------------------------------------------------------------

    teacher = TeacherCNN(
        num_classes
    )

    print("\nTeacher model")

    print(
        f"Parameters: {count_parameters(teacher):,}"
    )

    print(
        f"Storage: {model_storage_bytes(teacher) / 1024**2:.2f} MB"
    )

    teacher_history = train_teacher(
        teacher,
        train_loader,
        device,
        epochs=args.teacher_epochs,
        learning_rate=args.learning_rate,
    )

    teacher_evaluation = evaluate(
        teacher,
        val_loader,
        device,
    )

    print(
        f"\nTeacher validation accuracy: "
        f"{teacher_evaluation['accuracy']:.4f}"
    )

    # --------------------------------------------------------------
    # STUDENT: HARD LABELS ONLY
    # --------------------------------------------------------------

    hard_student = StudentCNN(
        num_classes
    )

    print(
        "\nTraining student with hard labels only..."
    )

    hard_history = train_student(
        hard_student,
        teacher,
        train_loader,
        device,
        epochs=args.student_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        alpha=args.alpha,
        use_distillation=False,
    )

    hard_evaluation = evaluate(
        hard_student,
        val_loader,
        device,
    )

    # --------------------------------------------------------------
    # STUDENT: KNOWLEDGE DISTILLATION
    # --------------------------------------------------------------

    distill_student = StudentCNN(
        num_classes
    )

    print(
        "\nTraining student with knowledge distillation..."
    )

    distill_history = train_student(
        distill_student,
        teacher,
        train_loader,
        device,
        epochs=args.student_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        alpha=args.alpha,
        use_distillation=True,
    )

    distill_evaluation = evaluate(
        distill_student,
        val_loader,
        device,
    )

    # --------------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------------

    print(
        "\nBenchmarking inference..."
    )

    models = {

        "teacher": teacher,

        "student_hard": hard_student,

        "student_distilled": distill_student,

    }

    evaluations = {

        "teacher": teacher_evaluation,

        "student_hard": hard_evaluation,

        "student_distilled": distill_evaluation,

    }

    results = {}

    for name, model in models.items():

        benchmark = benchmark_inference(
            model,
            val_loader,
            device,
            iterations=args.benchmark_iterations,
        )

        results[name] = {

            "evaluation": evaluations[name],

            "parameters": count_parameters(model),

            "storage_bytes": model_storage_bytes(model),

            "benchmark": benchmark,

        }

    # --------------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------------

    print("\n" + "=" * 100)

    print("MODEL COMPARISON")

    print("=" * 100)

    print(
        f"{'Model':<22}"
        f"{'Accuracy':>12}"
        f"{'Parameters':>16}"
        f"{'Storage MB':>14}"
        f"{'Latency ms':>14}"
        f"{'FPS':>12}"
    )

    print("-" * 100)

    for name, result in results.items():

        print(
            f"{name:<22}"
            f"{result['evaluation']['accuracy']:>12.4f}"
            f"{result['parameters']:>16,}"
            f"{result['storage_bytes'] / 1024**2:>14.3f}"
            f"{result['benchmark']['mean_latency_ms']:>14.3f}"
            f"{result['benchmark']['images_per_second']:>12.2f}"
        )

    # --------------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------------

    plot_training_history(
        teacher_history,
        hard_history,
        distill_history,
        output_dir,
    )

    plot_final_comparison(
        results,
        output_dir,
    )

    save_prediction_examples(
        teacher,
        distill_student,
        val_loader,
        device,
        class_names,
        output_dir,
    )

    # --------------------------------------------------------------
    # SAVE MODELS
    # --------------------------------------------------------------

    torch.save(
        teacher.state_dict(),
        output_dir / "teacher.pth",
    )

    torch.save(
        hard_student.state_dict(),
        output_dir / "student_hard.pth",
    )

    torch.save(
        distill_student.state_dict(),
        output_dir / "student_distilled.pth",
    )

    # --------------------------------------------------------------
    # REPORT
    # --------------------------------------------------------------

    report = {

        "lesson": 76,

        "title": "Knowledge Distillation",

        "dataset_type": dataset_type,

        "dataset_size": len(dataset),

        "training_samples": len(train_dataset),

        "validation_samples": len(val_dataset),

        "number_of_classes": num_classes,

        "class_names": class_names,

        "image_size": args.image_size,

        "temperature": args.temperature,

        "alpha": args.alpha,

        "teacher_history": teacher_history,

        "hard_student_history": hard_history,

        "distillation_student_history": distill_history,

        "temperature_demonstration": temperature_results,

        "results": results,

    }

    with open(
        output_dir / "distillation_report.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------

    print("\nGenerated files:")

    for path in sorted(
        output_dir.iterdir()
    ):

        print(
            f"  {path}"
        )

    print("\n" + "=" * 75)

    print("LESSON 76 COMPLETED")

    print("=" * 75)

    print(
        "\nNext: Lesson 77 — ONNX Model Export."
    )


if __name__ == "__main__":

    main()