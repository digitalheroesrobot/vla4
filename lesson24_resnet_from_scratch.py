"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 24 — RESNET FROM SCRATCH
======================================================================

COURSE PHILOSOPHY
-----------------
1. No pretrained models.
2. No pretrained weights.
3. All model parameters are randomly initialized.
4. Understand the mathematics before using established architectures.
5. Implement the residual block explicitly.
6. Train a small ResNet from scratch.
7. Compare a plain CNN against a residual CNN.

LIBRARIES
---------
- NumPy
- Matplotlib
- PyTorch

No torchvision is required.

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1 — THE DEEP NETWORK PROBLEM
    Understand why deeper neural networks can become harder to train.

STEP 2 — RESIDUAL LEARNING
    Instead of directly learning H(x), learn:

        F(x) = H(x) - x

    Therefore:

        H(x) = F(x) + x

STEP 3 — IDENTITY SHORTCUT
    Implement:

        y = F(x) + x

    The input travels through two paths:
        1. Residual transformation.
        2. Identity shortcut.

STEP 4 — RESIDUAL BLOCK
    Implement a two-convolution residual block:

        x
        |
        +--------------------------+
        |                          |
        Conv -> BN -> ReLU          |
        |                          |
        Conv -> BN                  |
        |                          |
        +---------- ADD <-----------+
                   |
                 ReLU
                   |
                   y

STEP 5 — PROJECTION SHORTCUT
    When input and output dimensions differ:

        y = F(x) + W_s(x)

    W_s is a 1x1 convolution that changes channel count
    and/or spatial resolution.

STEP 6 — BUILD A SMALL RESNET
    Architecture:

        Input: 3 x 32 x 32

        Stem:
            Conv 3x3, 16 channels

        Stage 1:
            Residual Block, 16 channels
            Residual Block, 16 channels

        Stage 2:
            Residual Block, 32 channels, stride 2
            Residual Block, 32 channels

        Stage 3:
            Residual Block, 64 channels, stride 2
            Residual Block, 64 channels

        Global Average Pooling

        Fully Connected Layer

        Output: 3 classes

STEP 7 — TRAIN FROM SCRATCH
    Generate a synthetic image classification dataset.

    Classes:
        0: Horizontal patterns
        1: Vertical patterns
        2: Diagonal patterns

    Compare:
        - Plain CNN
        - Residual CNN

STEP 8 — GRADIENT FLOW
    Inspect gradients inside early and late network layers.

STEP 9 — REAL-WORLD ROBOTICS APPLICATION
    Consider an OpenArm V2 waste-sorting system.

    Camera image
        |
        v
    ResNet feature extraction
        |
        v
    Object classification
        |
        v
    Object category
        |
        v
    ACT manipulation policy
        |
        v
    Robot picks and sorts the object.

======================================================================
MATHEMATICAL DERIVATION
======================================================================

1. PLAIN NETWORK

    y = H(x)

    Gradient:

    dy/dx = dH(x)/dx

    During backpropagation, gradients pass through every
    transformation in the network.

2. RESIDUAL NETWORK

    y = F(x) + x

    Gradient:

    dy/dx = dF(x)/dx + 1

    The identity shortcut provides a direct gradient path.

    This does not guarantee perfect gradients, but it makes
    optimization of deeper networks easier in many situations.

3. DIMENSION MATCHING

    If input and output dimensions are equal:

        y = F(x) + x

    Otherwise:

        y = F(x) + W_s(x)

    W_s is usually implemented using a 1x1 convolution.

4. GLOBAL AVERAGE POOLING

    For feature map X with dimensions C x H x W:

        z[c] = (1 / HW) * sum(X[c, i, j])

    This converts spatial feature maps into a C-dimensional
    feature vector.

5. CROSS-ENTROPY LOSS

    L = -sum(y_true * log(y_pred))

    PyTorch CrossEntropyLoss combines log-softmax and
    negative log likelihood.

======================================================================
WHAT YOU SHOULD UNDERSTAND AFTER THIS LESSON
======================================================================

- Why deeper plain networks can become difficult to optimize.
- Why residual connections provide a direct gradient path.
- How identity shortcuts work.
- Why projection shortcuts are necessary.
- How a ResNet is constructed from residual blocks.
- How to train a ResNet without pretrained weights.
- How to inspect gradient flow.
- How to compare plain and residual architectures.

======================================================================
INSTALLATION
======================================================================

pip install numpy matplotlib torch

======================================================================
RUN
======================================================================

python lesson24_resnet_from_scratch.py

======================================================================
NEXT LESSON
======================================================================

LESSON 25 — DATA AUGMENTATION FROM SCRATCH

You will implement:
    - Random horizontal flipping
    - Random rotation
    - Random cropping
    - Color jitter
    - Random translation
    - Cutout
    - MixUp
    - CutMix

======================================================================
"""

import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

IMAGE_SIZE = 32
NUM_CLASSES = 3

TRAIN_SAMPLES = 900
TEST_SAMPLES = 300

BATCH_SIZE = 64

PLAIN_EPOCHS = 8
RESNET_EPOCHS = 8

LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# STEP 1 — SYNTHETIC DATASET
# ======================================================================

def draw_line_pattern(image, orientation, rng):
    """
    Draw a synthetic visual pattern.

    orientation:
        0 = horizontal
        1 = vertical
        2 = diagonal
    """

    height, width = image.shape

    intensity = rng.uniform(0.6, 1.0)

    if orientation == 0:

        y = rng.integers(5, height - 5)

        thickness = rng.integers(2, 5)

        image[
            max(0, y - thickness):min(height, y + thickness),
            :
        ] = intensity

    elif orientation == 1:

        x = rng.integers(5, width - 5)

        thickness = rng.integers(2, 5)

        image[
            :,
            max(0, x - thickness):min(width, x + thickness)
        ] = intensity

    elif orientation == 2:

        offset = rng.integers(-8, 8)

        thickness = rng.integers(2, 4)

        for y in range(height):

            x = y + offset

            if 0 <= x < width:

                image[
                    max(0, y - thickness):min(height, y + thickness),
                    max(0, x - thickness):min(width, x + thickness)
                ] = intensity

    return image


def generate_sample(label, rng):
    """
    Generate a 3-channel synthetic image.

    Random color and background variation prevent the model
    from relying on a single fixed pixel pattern.
    """

    image = np.zeros(
        (3, IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float32,
    )

    background = rng.uniform(
        0.0,
        0.2,
        size=(3, IMAGE_SIZE, IMAGE_SIZE),
    )

    image += background.astype(np.float32)

    # Create pattern on a grayscale canvas.
    pattern = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float32,
    )

    pattern = draw_line_pattern(
        pattern,
        label,
        rng,
    )

    # Apply different color weights.
    color = rng.uniform(
        0.5,
        1.0,
        size=(3, 1, 1),
    )

    image += color * pattern[None, :, :]

    # Add Gaussian noise.
    noise = rng.normal(
        0.0,
        0.07,
        size=image.shape,
    )

    image += noise.astype(np.float32)

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return image


def generate_dataset(num_samples, seed):
    rng = np.random.default_rng(seed)

    images = []
    labels = []

    for index in range(num_samples):

        label = index % NUM_CLASSES

        image = generate_sample(
            label,
            rng,
        )

        images.append(image)
        labels.append(label)

    images = np.asarray(
        images,
        dtype=np.float32,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    # Shuffle the dataset.
    indices = rng.permutation(num_samples)

    images = images[indices]
    labels = labels[indices]

    return images, labels


# ======================================================================
# STEP 2 — RESIDUAL BLOCK
# ======================================================================

class ResidualBlock(nn.Module):
    """
    Two-convolution residual block.

    Main path:

        Conv3x3
        BatchNorm
        ReLU
        Conv3x3
        BatchNorm

    Shortcut path:

        Identity

    If dimensions differ:

        Conv1x1 + BatchNorm

    Final:

        ReLU(main_path + shortcut)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        # Identity shortcut when dimensions match.
        if (
            stride == 1
            and in_channels == out_channels
        ):

            self.shortcut = nn.Identity()

        else:

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),

                nn.BatchNorm2d(
                    out_channels
                ),
            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)

        out = self.bn1(out)

        out = F.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out = out + identity

        out = F.relu(out)

        return out


# ======================================================================
# STEP 3 — PLAIN CNN
# ======================================================================

class PlainCNN(nn.Module):
    """
    A conventional CNN without residual connections.

    This provides a comparison baseline.
    """

    def __init__(self, num_classes=NUM_CLASSES):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 16, 3,
                padding=1,
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.Conv2d(
                16, 16, 3,
                padding=1,
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32, 3,
                padding=1,
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.Conv2d(
                32, 32, 3,
                padding=1,
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64, 3,
                padding=1,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.Conv2d(
                64, 64, 3,
                padding=1,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(
            64,
            num_classes,
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.classifier(x)

        return x


# ======================================================================
# STEP 4 — SMALL RESNET
# ======================================================================

class SmallResNet(nn.Module):
    """
    Small ResNet inspired by the original residual architecture.

    All parameters are randomly initialized.

    Input:
        B x 3 x 32 x 32

    Output:
        B x num_classes
    """

    def __init__(self, num_classes=NUM_CLASSES):

        super().__init__()

        # Initial feature extraction.
        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),
        )

        # Stage 1: 32 x 32.
        self.stage1 = nn.Sequential(

            ResidualBlock(
                16,
                16,
                stride=1,
            ),

            ResidualBlock(
                16,
                16,
                stride=1,
            ),
        )

        # Stage 2: 16 x 16.
        self.stage2 = nn.Sequential(

            ResidualBlock(
                16,
                32,
                stride=2,
            ),

            ResidualBlock(
                32,
                32,
                stride=1,
            ),
        )

        # Stage 3: 8 x 8.
        self.stage3 = nn.Sequential(

            ResidualBlock(
                32,
                64,
                stride=2,
            ),

            ResidualBlock(
                64,
                64,
                stride=1,
            ),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(
            64,
            num_classes,
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.global_pool(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.classifier(x)

        return x


# ======================================================================
# STEP 5 — MODEL UTILITIES
# ======================================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def initialize_model(model):
    """
    Explicitly initialize all trainable weights from scratch.

    Convolution:
        Kaiming initialization.

    Linear:
        Kaiming initialization.

    BatchNorm:
        Scale = 1
        Bias = 0
    """

    for module in model.modules():

        if isinstance(module, nn.Conv2d):

            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

        elif isinstance(module, nn.Linear):

            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(
            module,
            nn.BatchNorm2d,
        ):

            nn.init.ones_(module.weight)

            nn.init.zeros_(module.bias)

    return model


# ======================================================================
# STEP 6 — TRAINING
# ======================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    average_loss = total_loss / total

    accuracy = correct / total

    return average_loss, accuracy


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    return (
        total_loss / total,
        correct / total,
    )


def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    model_name,
):

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    print()
    print("=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(epochs):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
        )

        history["train_loss"].append(train_loss)

        history["train_accuracy"].append(
            train_accuracy
        )

        history["test_loss"].append(test_loss)

        history["test_accuracy"].append(
            test_accuracy
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2%} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.2%}"
        )

    elapsed = time.time() - start_time

    print(
        f"\nTraining time: {elapsed:.2f} seconds"
    )

    return model, history


# ======================================================================
# STEP 7 — GRADIENT FLOW INSPECTION
# ======================================================================

def inspect_gradient_flow(model, sample_images):
    """
    Inspect gradients after one backward pass.

    We compare gradient magnitudes in early and late layers.

    This is a diagnostic, not a proof that ResNet always has
    better gradient flow than every plain CNN.
    """

    model.train()

    model.zero_grad()

    sample_images = sample_images.to(DEVICE)

    labels = torch.zeros(
        sample_images.size(0),
        dtype=torch.long,
        device=DEVICE,
    )

    outputs = model(sample_images)

    loss = F.cross_entropy(
        outputs,
        labels,
    )

    loss.backward()

    gradients = {}

    for name, parameter in model.named_parameters():

        if (
            parameter.grad is not None
            and parameter.ndim >= 2
        ):

            gradients[name] = (
                parameter.grad.detach()
                .abs()
                .mean()
                .item()
            )

    return gradients


def print_gradient_report(name, gradients):

    print()
    print("=" * 70)
    print(f"GRADIENT REPORT: {name}")
    print("=" * 70)

    selected = list(gradients.items())

    if not selected:
        print("No gradients available.")
        return

    print("First trainable layers:")

    for layer_name, magnitude in selected[:5]:

        print(
            f"{layer_name:<45} "
            f"{magnitude:.8e}"
        )

    print("\nLast trainable layers:")

    for layer_name, magnitude in selected[-5:]:

        print(
            f"{layer_name:<45} "
            f"{magnitude:.8e}"
        )


# ======================================================================
# STEP 8 — VISUALIZATION
# ======================================================================

def visualize_dataset(images, labels):

    fig, axes = plt.subplots(
        3,
        6,
        figsize=(12, 6),
    )

    fig.suptitle(
        "Synthetic Classification Dataset",
        fontsize=15,
    )

    shown = [0, 1, 2]

    for row, class_id in enumerate(shown):

        indices = np.where(
            labels == class_id
        )[0][:6]

        for col, index in enumerate(indices):

            image = images[index].transpose(
                1, 2, 0
            )

            axes[row, col].imshow(image)

            axes[row, col].axis("off")

            axes[row, col].set_title(
                f"Class {class_id}"
            )

    plt.tight_layout()

    plt.show()


def visualize_training(
    plain_history,
    resnet_history,
):

    epochs = np.arange(
        1,
        len(plain_history["train_loss"]) + 1,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    fig.suptitle(
        "Plain CNN vs ResNet Training",
        fontsize=16,
    )

    # Training loss.
    axes[0, 0].plot(
        epochs,
        plain_history["train_loss"],
        marker="o",
        label="Plain CNN",
    )

    axes[0, 0].plot(
        epochs,
        resnet_history["train_loss"],
        marker="o",
        label="ResNet",
    )

    axes[0, 0].set_title(
        "Training Loss"
    )

    axes[0, 0].set_xlabel("Epoch")

    axes[0, 0].set_ylabel("Loss")

    axes[0, 0].legend()

    axes[0, 0].grid(True)

    # Test loss.
    axes[0, 1].plot(
        epochs,
        plain_history["test_loss"],
        marker="o",
        label="Plain CNN",
    )

    axes[0, 1].plot(
        epochs,
        resnet_history["test_loss"],
        marker="o",
        label="ResNet",
    )

    axes[0, 1].set_title(
        "Test Loss"
    )

    axes[0, 1].set_xlabel("Epoch")

    axes[0, 1].set_ylabel("Loss")

    axes[0, 1].legend()

    axes[0, 1].grid(True)

    # Training accuracy.
    axes[1, 0].plot(
        epochs,
        plain_history["train_accuracy"],
        marker="o",
        label="Plain CNN",
    )

    axes[1, 0].plot(
        epochs,
        resnet_history["train_accuracy"],
        marker="o",
        label="ResNet",
    )

    axes[1, 0].set_title(
        "Training Accuracy"
    )

    axes[1, 0].set_xlabel("Epoch")

    axes[1, 0].set_ylabel("Accuracy")

    axes[1, 0].legend()

    axes[1, 0].grid(True)

    # Test accuracy.
    axes[1, 1].plot(
        epochs,
        plain_history["test_accuracy"],
        marker="o",
        label="Plain CNN",
    )

    axes[1, 1].plot(
        epochs,
        resnet_history["test_accuracy"],
        marker="o",
        label="ResNet",
    )

    axes[1, 1].set_title(
        "Test Accuracy"
    )

    axes[1, 1].set_xlabel("Epoch")

    axes[1, 1].set_ylabel("Accuracy")

    axes[1, 1].legend()

    axes[1, 1].grid(True)

    plt.tight_layout()

    plt.show()


def visualize_residual_block():

    block = ResidualBlock(
        in_channels=16,
        out_channels=16,
    )

    block.eval()

    x = torch.randn(
        1,
        16,
        32,
        32,
    )

    with torch.no_grad():

        identity = block.shortcut(x)

        residual = block.bn1(
            block.conv1(x)
        )

        residual = F.relu(residual)

        residual = block.bn2(
            block.conv2(residual)
        )

        output = F.relu(
            residual + identity
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    axes[0].imshow(
        x[0, 0].numpy(),
        cmap="gray",
    )

    axes[0].set_title(
        "Input Feature Map"
    )

    axes[1].imshow(
        residual[0, 0].numpy(),
        cmap="gray",
    )

    axes[1].set_title(
        "Residual Transformation"
    )

    axes[2].imshow(
        output[0, 0].numpy(),
        cmap="gray",
    )

    axes[2].set_title(
        "Residual Block Output"
    )

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()

    plt.show()


# ======================================================================
# STEP 9 — NUMERICAL AND UNIT TESTS
# ======================================================================

def run_tests():

    print()
    print("=" * 70)
    print("RUNNING LESSON 24 TESTS")
    print("=" * 70)

    # --------------------------------------------------------------
    # TEST 1 — Identity shortcut
    # --------------------------------------------------------------

    block = ResidualBlock(
        16,
        16,
        stride=1,
    )

    assert isinstance(
        block.shortcut,
        nn.Identity,
    )

    print("Test 1 passed: Identity shortcut.")

    # --------------------------------------------------------------
    # TEST 2 — Projection shortcut
    # --------------------------------------------------------------

    block_projection = ResidualBlock(
        16,
        32,
        stride=2,
    )

    assert isinstance(
        block_projection.shortcut,
        nn.Sequential,
    )

    print("Test 2 passed: Projection shortcut.")

    # --------------------------------------------------------------
    # TEST 3 — Residual block shape
    # --------------------------------------------------------------

    x = torch.randn(
        2,
        16,
        32,
        32,
    )

    output = block(x)

    assert output.shape == (
        2,
        16,
        32,
        32,
    )

    print("Test 3 passed: Residual block output shape.")

    # --------------------------------------------------------------
    # TEST 4 — Projection block shape
    # --------------------------------------------------------------

    x = torch.randn(
        2,
        16,
        32,
        32,
    )

    output = block_projection(x)

    assert output.shape == (
        2,
        32,
        16,
        16,
    )

    print("Test 4 passed: Projection block output shape.")

    # --------------------------------------------------------------
    # TEST 5 — ResNet output
    # --------------------------------------------------------------

    model = SmallResNet()

    x = torch.randn(
        4,
        3,
        32,
        32,
    )

    output = model(x)

    assert output.shape == (
        4,
        NUM_CLASSES,
    )

    print("Test 5 passed: ResNet output shape.")

    # --------------------------------------------------------------
    # TEST 6 — Plain CNN output
    # --------------------------------------------------------------

    model = PlainCNN()

    output = model(x)

    assert output.shape == (
        4,
        NUM_CLASSES,
    )

    print("Test 6 passed: Plain CNN output shape.")

    # --------------------------------------------------------------
    # TEST 7 — Backpropagation
    # --------------------------------------------------------------

    model = SmallResNet()

    x = torch.randn(
        4,
        3,
        32,
        32,
    )

    labels = torch.randint(
        0,
        NUM_CLASSES,
        (4,),
    )

    output = model(x)

    loss = F.cross_entropy(
        output,
        labels,
    )

    loss.backward()

    gradients_exist = any(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    assert gradients_exist

    print("Test 7 passed: Backpropagation.")

    # --------------------------------------------------------------
    # TEST 8 — Gradient is finite
    # --------------------------------------------------------------

    for parameter in model.parameters():

        if parameter.grad is not None:

            assert torch.isfinite(
                parameter.grad
            ).all()

    print("Test 8 passed: Finite gradients.")

    # --------------------------------------------------------------
    # TEST 9 — Parameter initialization
    # --------------------------------------------------------------

    model = initialize_model(
        SmallResNet()
    )

    conv_weights = [
        module.weight
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    ]

    assert len(conv_weights) > 0

    assert all(
        torch.isfinite(weight).all()
        for weight in conv_weights
    )

    print("Test 9 passed: Random weight initialization.")

    # --------------------------------------------------------------
    # TEST 10 — Global average pooling
    # --------------------------------------------------------------

    pool = nn.AdaptiveAvgPool2d(1)

    x = torch.randn(
        2,
        64,
        8,
        8,
    )

    output = pool(x)

    assert output.shape == (
        2,
        64,
        1,
        1,
    )

    print("Test 10 passed: Global average pooling.")

    # --------------------------------------------------------------
    # TEST 11 — Parameter count
    # --------------------------------------------------------------

    plain = PlainCNN()

    resnet = SmallResNet()

    assert count_parameters(plain) > 0

    assert count_parameters(resnet) > 0

    print("Test 11 passed: Parameter counting.")

    print()
    print("ALL TESTS PASSED.")
    print("=" * 70)


# ======================================================================
# STEP 10 — MAIN EXPERIMENT
# ======================================================================

def run_experiment():

    print()
    print("=" * 70)
    print("LESSON 24 — RESNET FROM SCRATCH")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")

    print(f"Classes: {NUM_CLASSES}")

    print()

    # --------------------------------------------------------------
    # Generate data
    # --------------------------------------------------------------

    train_images, train_labels = generate_dataset(
        TRAIN_SAMPLES,
        seed=SEED,
    )

    test_images, test_labels = generate_dataset(
        TEST_SAMPLES,
        seed=SEED + 1,
    )

    print(
        f"Training samples: {len(train_images)}"
    )

    print(
        f"Test samples: {len(test_images)}"
    )

    train_dataset = TensorDataset(
        torch.from_numpy(train_images),
        torch.from_numpy(train_labels),
    )

    test_dataset = TensorDataset(
        torch.from_numpy(test_images),
        torch.from_numpy(test_labels),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    visualize_dataset(
        train_images,
        train_labels,
    )

    # --------------------------------------------------------------
    # Model 1 — Plain CNN
    # --------------------------------------------------------------

    plain_model = initialize_model(
        PlainCNN()
    )

    print(
        f"\nPlain CNN parameters: "
        f"{count_parameters(plain_model):,}"
    )

    plain_model, plain_history = train_model(
        model=plain_model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=PLAIN_EPOCHS,
        model_name="PLAIN CNN",
    )

    # --------------------------------------------------------------
    # Model 2 — ResNet
    # --------------------------------------------------------------

    resnet_model = initialize_model(
        SmallResNet()
    )

    print(
        f"\nResNet parameters: "
        f"{count_parameters(resnet_model):,}"
    )

    resnet_model, resnet_history = train_model(
        model=resnet_model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=RESNET_EPOCHS,
        model_name="RESNET",
    )

    # --------------------------------------------------------------
    # Gradient flow
    # --------------------------------------------------------------

    sample_images = torch.from_numpy(
        train_images[:8]
    )

    plain_gradients = inspect_gradient_flow(
        plain_model,
        sample_images,
    )

    resnet_gradients = inspect_gradient_flow(
        resnet_model,
        sample_images,
    )

    print_gradient_report(
        "PLAIN CNN",
        plain_gradients,
    )

    print_gradient_report(
        "RESNET",
        resnet_gradients,
    )

    # --------------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------------

    plain_accuracy = plain_history[
        "test_accuracy"
    ][-1]

    resnet_accuracy = resnet_history[
        "test_accuracy"
    ][-1]

    print()
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        f"Plain CNN test accuracy: "
        f"{plain_accuracy:.2%}"
    )

    print(
        f"ResNet test accuracy: "
        f"{resnet_accuracy:.2%}"
    )

    print(
        f"Accuracy difference: "
        f"{resnet_accuracy - plain_accuracy:+.2%}"
    )

    print()
    print(
        "Note: ResNet is not guaranteed to outperform the plain CNN "
        "on this small synthetic dataset. The primary objective is "
        "to understand residual architecture and gradient flow."
    )

    # --------------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------------

    visualize_training(
        plain_history,
        resnet_history,
    )

    visualize_residual_block()


# ======================================================================
# MAIN
# ======================================================================

def main():

    set_seed()

    print("Device:", DEVICE)

    run_tests()

    run_experiment()

    print()
    print("=" * 70)
    print("LESSON 24 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()