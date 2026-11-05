"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 6 — MODEL OPTIMIZATION AND DEPLOYMENT

LESSON 77 — ONNX MODEL EXPORT
======================================================================

OBJECTIVE
---------

Export a trained PyTorch CNN to ONNX format.

Then:

1. Load the ONNX graph.
2. Validate the graph.
3. Run ONNX Runtime inference.
4. Compare PyTorch and ONNX outputs.
5. Test multiple batch sizes.
6. Benchmark CPU inference.
7. Save export metadata.

No pretrained models or pretrained weights are used.

======================================================================
1. WHAT IS ONNX?
======================================================================

ONNX = Open Neural Network Exchange.

It is an open model representation format.

It separates:

    Model architecture / computation graph
    from
    Training framework

Example:

    PyTorch training
          |
          v
       ONNX export
          |
          v
      model.onnx
          |
          +------------------+
          |                  |
          v                  v
    ONNX Runtime         TensorRT
          |                  |
          v                  v
         CPU                GPU

======================================================================
2. COMPUTATION GRAPH
======================================================================

A CNN performs operations such as:

    Conv2D
    BatchNorm
    ReLU
    MaxPool
    Linear

ONNX represents these operations as graph nodes.

The graph also contains:

    - Inputs
    - Outputs
    - Initializers (model weights)
    - Tensor shapes
    - Operators

======================================================================
3. STATIC AND DYNAMIC SHAPES
======================================================================

Static input:

    [1, 3, 64, 64]

Only batch size 1.

Dynamic batch:

    [batch, 3, 64, 64]

Can accept:

    [1, 3, 64, 64]
    [4, 3, 64, 64]
    [8, 3, 64, 64]

Dynamic dimensions are useful in production pipelines.

======================================================================
4. EXPORT
======================================================================

PyTorch model:

    model.eval()

    torch.onnx.export(
        model,
        example_input,
        "model.onnx",
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

======================================================================
5. VALIDATION
======================================================================

Compare:

    PyTorch output
    ONNX Runtime output

Maximum absolute error:

    max(abs(Y_torch - Y_onnx))

Mean absolute error:

    mean(abs(Y_torch - Y_onnx))

Numerical equivalence does not guarantee identical performance.

======================================================================
6. ROBOTICS APPLICATION
======================================================================

OpenArm V2 perception pipeline:

    Camera
      |
      v
    Image preprocessing
      |
      v
    ONNX model
      |
      v
    Object classification / detection
      |
      v
    Object localization
      |
      v
    ACT policy
      |
      v
    Robot manipulation

ONNX is an intermediate deployment format.

It does not automatically make a model faster.
Performance depends on the execution provider, graph optimization,
hardware, operators, and precision.

======================================================================
NEXT LESSON
-----------

Lesson 78 — ONNX Runtime Optimization.
======================================================================
"""

from __future__ import annotations

import argparse
import json
import random
import time

from pathlib import Path

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

DEFAULT_OUTPUT_DIR = Path("runs/lesson77")

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

def set_seed(seed: int = 42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ======================================================================
# DEVICE
# ======================================================================

def select_device(requested: str):

    if requested == "cpu":

        return torch.device("cpu")

    if requested == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA is unavailable."
            )

        return torch.device("cuda")

    if requested == "auto":

        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    raise ValueError(
        f"Unsupported device: {requested}"
    )


def synchronize(device):

    if device.type == "cuda":

        torch.cuda.synchronize(device)


# ======================================================================
# DATASET
# ======================================================================

class SyntheticDataset(Dataset):

    def __init__(
        self,
        samples_per_class=100,
        image_size=IMAGE_SIZE,
        seed=42,
    ):

        self.images = []

        self.labels = []

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

                ox = rng.integers(-5, 6)

                oy = rng.integers(-5, 6)

                cx = center + ox

                cy = center + oy

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
                        np.abs(x - y - ox) < 4
                    ) | (
                        np.abs(x + y - image_size - oy) < 4
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

        self.images = torch.tensor(
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

        return image * 2.0 - 1.0, label


class ImageFolderDataset(Dataset):

    EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    def __init__(
        self,
        root,
        image_size=IMAGE_SIZE,
    ):

        from PIL import Image

        self.root = Path(root)

        if not self.root.exists():

            raise FileNotFoundError(
                self.root
            )

        directories = sorted(
            path
            for path in self.root.iterdir()
            if path.is_dir()
        )

        if len(directories) < 2:

            raise ValueError(
                "At least two class folders are required."
            )

        self.classes = [
            path.name
            for path in directories
        ]

        self.class_to_idx = {

            name: index

            for index, name in enumerate(self.classes)

        }

        self.samples = []

        for directory in directories:

            label = self.class_to_idx[
                directory.name
            ]

            for path in sorted(
                directory.rglob("*")
            ):

                if path.suffix.lower() in self.EXTENSIONS:

                    self.samples.append(
                        (path, label)
                    )

        if len(self.samples) == 0:

            raise ValueError(
                "No image files found."
            )

        self.image_size = image_size

        print(
            f"Loaded {len(self.samples)} images."
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

        return tensor * 2.0 - 1.0, label


# ======================================================================
# MODEL
# ======================================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):

        super().__init__()

        self.block = nn.Sequential(

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

            nn.MaxPool2d(2),

        )

    def forward(self, x):

        return self.block(x)


class VisionCNN(nn.Module):

    def __init__(
        self,
        num_classes,
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
# TRAINING
# ======================================================================

def train_model(
    model,
    loader,
    device,
    epochs,
    learning_rate,
):

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    history = []

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        correct = 0

        total = 0

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

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

            total_loss += (
                loss.item()
                *
                labels.size(0)
            )

            correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            total += labels.size(0)

        epoch_loss = total_loss / max(total, 1)

        accuracy = correct / max(total, 1)

        history.append({

            "loss": epoch_loss,

            "accuracy": accuracy,

        })

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| loss={epoch_loss:.4f} "
            f"| accuracy={accuracy:.4f}"
        )

    return history


def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    total = 0

    correct = 0

    total_loss = 0.0

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            logits = model(images)

            loss = F.cross_entropy(
                logits,
                labels,
            )

            total_loss += (
                loss.item()
                *
                labels.size(0)
            )

            correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            total += labels.size(0)

    return {

        "accuracy": correct / max(total, 1),

        "loss": total_loss / max(total, 1),

        "samples": total,

    }


# ======================================================================
# ONNX EXPORT
# ======================================================================

def export_onnx(
    model,
    output_path,
    image_size,
    dynamic_batch=True,
):

    model.eval()

    model.cpu()

    example_input = torch.randn(
        1,
        3,
        image_size,
        image_size,
        dtype=torch.float32,
    )

    dynamic_axes = None

    if dynamic_batch:

        dynamic_axes = {

            "images": {
                0: "batch_size",
            },

            "logits": {
                0: "batch_size",
            },

        }

    # Use the modern dynamo exporter when available.
    # The legacy exporter remains a fallback for environments
    # without the ONNX exporter dependencies required by dynamo.

    try:

        torch.onnx.export(

            model,

            example_input,

            str(output_path),

            input_names=["images"],

            output_names=["logits"],

            dynamic_axes=dynamic_axes,

            opset_version=17,

            dynamo=False,

            do_constant_folding=True,

        )

    except Exception as error:

        raise RuntimeError(
            f"ONNX export failed: {error}"
        ) from error

    print(
        f"\nONNX model exported: {output_path}"
    )


# ======================================================================
# ONNX VALIDATION
# ======================================================================

def validate_onnx(
    onnx_path,
):

    import onnx

    model = onnx.load(
        str(onnx_path)
    )

    onnx.checker.check_model(
        model
    )

    print(
        "\nONNX graph validation passed."
    )

    print(
        f"ONNX opset: {model.opset_import[0].version}"
    )

    print(
        f"Graph nodes: {len(model.graph.node)}"
    )

    return {

        "nodes": len(model.graph.node),

        "opset": model.opset_import[0].version,

    }


# ======================================================================
# ONNX RUNTIME SESSION
# ======================================================================

def create_onnx_session(
    onnx_path,
    providers=None,
):

    import onnxruntime as ort

    if providers is None:

        providers = [
            "CPUExecutionProvider"
        ]

    available = ort.get_available_providers()

    for provider in providers:

        if provider not in available:

            raise RuntimeError(
                f"Requested ONNX provider unavailable: {provider}. "
                f"Available providers: {available}"
            )

    options = ort.SessionOptions()

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(

        str(onnx_path),

        sess_options=options,

        providers=providers,

    )

    print(
        "\nONNX Runtime providers:"
    )

    print(
        session.get_providers()
    )

    return session


# ======================================================================
# NUMERICAL EQUIVALENCE
# ======================================================================

def compare_outputs(
    pytorch_model,
    onnx_session,
    batch_size,
    image_size,
    device,
):

    rng = np.random.default_rng(
        42 + batch_size
    )

    input_array = rng.normal(
        0.0,
        1.0,
        size=(
            batch_size,
            3,
            image_size,
            image_size,
        ),
    ).astype(np.float32)

    torch_input = torch.from_numpy(
        input_array
    ).to(device)

    pytorch_model.eval()

    with torch.inference_mode():

        torch_output = pytorch_model(
            torch_input
        ).cpu().numpy()

    onnx_output = onnx_session.run(

        ["logits"],

        {
            "images": input_array,
        },

    )[0]

    absolute_error = np.abs(
        torch_output - onnx_output
    )

    max_error = float(
        absolute_error.max()
    )

    mean_error = float(
        absolute_error.mean()
    )

    relative_error = float(
        np.linalg.norm(
            torch_output - onnx_output
        )
        /
        max(
            np.linalg.norm(torch_output),
            1e-12,
        )
    )

    prediction_agreement = float(
        np.mean(
            torch_output.argmax(axis=1)
            ==
            onnx_output.argmax(axis=1)
        )
    )

    return {

        "batch_size": batch_size,

        "max_absolute_error": max_error,

        "mean_absolute_error": mean_error,

        "relative_l2_error": relative_error,

        "prediction_agreement": prediction_agreement,

    }


# ======================================================================
# ONNX INFERENCE BENCHMARK
# ======================================================================

def benchmark_onnx(
    session,
    image_size,
    batch_size=1,
    warmup=10,
    iterations=100,
):

    rng = np.random.default_rng(123)

    images = rng.normal(
        0.0,
        1.0,
        size=(
            batch_size,
            3,
            image_size,
            image_size,
        ),
    ).astype(np.float32)

    for _ in range(warmup):

        session.run(
            ["logits"],
            {"images": images},
        )

    timings = []

    for _ in range(iterations):

        start = time.perf_counter()

        session.run(
            ["logits"],
            {"images": images},
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        timings.append(elapsed)

    values = np.asarray(
        timings,
        dtype=np.float64,
    )

    mean_latency = float(
        values.mean()
    )

    return {

        "batch_size": batch_size,

        "mean_latency_ms": mean_latency,

        "p50_latency_ms": float(
            np.percentile(values, 50)
        ),

        "p95_latency_ms": float(
            np.percentile(values, 95)
        ),

        "images_per_second": (
            batch_size * 1000.0 / mean_latency
        ),

    }


# ======================================================================
# PYTORCH INFERENCE BENCHMARK
# ======================================================================

def benchmark_pytorch(
    model,
    device,
    image_size,
    batch_size=1,
    warmup=10,
    iterations=100,
):

    model.eval()

    images = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        device=device,
    )

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

        timings.append(
            (time.perf_counter() - start) * 1000.0
        )

    values = np.asarray(
        timings,
        dtype=np.float64,
    )

    mean_latency = float(
        values.mean()
    )

    return {

        "batch_size": batch_size,

        "mean_latency_ms": mean_latency,

        "p50_latency_ms": float(
            np.percentile(values, 50)
        ),

        "p95_latency_ms": float(
            np.percentile(values, 95)
        ),

        "images_per_second": (
            batch_size * 1000.0 / mean_latency
        ),

    }


# ======================================================================
# FILE SIZE
# ======================================================================

def file_size_bytes(
    path,
):

    return Path(path).stat().st_size


# ======================================================================
# VISUALIZATION
# ======================================================================

def plot_equivalence(
    results,
    output_dir,
):

    batch_sizes = [

        item["batch_size"]

        for item in results

    ]

    max_errors = [

        item["max_absolute_error"]

        for item in results

    ]

    mean_errors = [

        item["mean_absolute_error"]

        for item in results

    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        batch_sizes,
        max_errors,
        marker="o",
        label="Maximum absolute error",
    )

    ax.plot(
        batch_sizes,
        mean_errors,
        marker="s",
        label="Mean absolute error",
    )

    ax.set_xlabel("Batch size")

    ax.set_ylabel("Numerical error")

    ax.set_title(
        "PyTorch vs ONNX Numerical Equivalence"
    )

    ax.set_xticks(
        batch_sizes
    )

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "onnx_equivalence.png",
        dpi=150,
    )

    plt.close(fig)


def plot_benchmark(
    pytorch_results,
    onnx_results,
    output_dir,
):

    batch_sizes = [

        item["batch_size"]

        for item in pytorch_results

    ]

    torch_latency = [

        item["mean_latency_ms"]

        for item in pytorch_results

    ]

    onnx_latency = [

        item["mean_latency_ms"]

        for item in onnx_results

    ]

    x = np.arange(
        len(batch_sizes)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        torch_latency,
        width,
        label="PyTorch",
    )

    ax.bar(
        x + width / 2,
        onnx_latency,
        width,
        label="ONNX Runtime CPU",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [str(size) for size in batch_sizes]
    )

    ax.set_xlabel("Batch size")

    ax.set_ylabel("Latency (ms)")

    ax.set_title(
        "PyTorch vs ONNX Runtime Latency"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "onnx_benchmark.png",
        dpi=150,
    )

    plt.close(fig)


# ======================================================================
# TESTS
# ======================================================================

def run_tests():

    print("\nRunning Lesson 77 tests...\n")

    import onnx
    import onnxruntime as ort

    set_seed(42)

    model = VisionCNN(
        NUM_CLASSES
    )

    model.eval()

    # Test 1: Model output shape.

    x = torch.randn(
        2,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    with torch.inference_mode():

        output = model(x)

    assert output.shape == (
        2,
        NUM_CLASSES,
    )

    print("Test 1 passed: PyTorch model output.")

    # Test 2: Model output finite.

    assert torch.isfinite(output).all()

    print("Test 2 passed: Finite model output.")

    # Test 3: Export ONNX.

    test_dir = Path(
        "runs/lesson77_tests"
    )

    test_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    onnx_path = (
        test_dir / "test_model.onnx"
    )

    export_onnx(
        model,
        onnx_path,
        IMAGE_SIZE,
        dynamic_batch=True,
    )

    assert onnx_path.exists()

    print("Test 3 passed: ONNX export.")

    # Test 4: ONNX graph validation.

    onnx_model = onnx.load(
        str(onnx_path)
    )

    onnx.checker.check_model(
        onnx_model
    )

    print("Test 4 passed: ONNX graph validation.")

    # Test 5: ONNX Runtime session.

    session = ort.InferenceSession(

        str(onnx_path),

        providers=[
            "CPUExecutionProvider"
        ],

    )

    assert session is not None

    print("Test 5 passed: ONNX Runtime session.")

    # Test 6: Dynamic batch size.

    for batch_size in [1, 2, 4]:

        test_input = np.random.randn(
            batch_size,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ).astype(np.float32)

        result = session.run(

            ["logits"],

            {
                "images": test_input,
            },

        )[0]

        assert result.shape == (
            batch_size,
            NUM_CLASSES,
        )

    print("Test 6 passed: Dynamic batch sizes.")

    # Test 7: Numerical equivalence.

    input_array = np.random.randn(
        3,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ).astype(np.float32)

    with torch.inference_mode():

        torch_result = model(
            torch.from_numpy(input_array)
        ).numpy()

    onnx_result = session.run(

        ["logits"],

        {
            "images": input_array,
        },

    )[0]

    assert np.allclose(
        torch_result,
        onnx_result,
        atol=1e-4,
        rtol=1e-4,
    )

    print("Test 7 passed: PyTorch/ONNX numerical equivalence.")

    # Test 8: Prediction agreement.

    torch_prediction = torch_result.argmax(
        axis=1
    )

    onnx_prediction = onnx_result.argmax(
        axis=1
    )

    assert np.array_equal(
        torch_prediction,
        onnx_prediction,
    )

    print("Test 8 passed: Prediction agreement.")

    # Test 9: ONNX input name.

    assert (
        session.get_inputs()[0].name
        ==
        "images"
    )

    print("Test 9 passed: ONNX input name.")

    # Test 10: ONNX output name.

    assert (
        session.get_outputs()[0].name
        ==
        "logits"
    )

    print("Test 10 passed: ONNX output name.")

    # Test 11: Input dimensions.

    input_shape = session.get_inputs()[0].shape

    assert input_shape[1:] == [
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ]

    print("Test 11 passed: Input dimensions.")

    # Test 12: File size.

    assert file_size_bytes(onnx_path) > 0

    print("Test 12 passed: ONNX file size.")

    # Test 13: Softmax output.

    probabilities = F.softmax(
        torch.from_numpy(torch_result),
        dim=1,
    )

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(3),
        atol=1e-6,
    )

    print("Test 13 passed: Probability normalization.")

    # Test 14: Prediction dimensions.

    assert torch_prediction.shape == (3,)

    print("Test 14 passed: Prediction dimensions.")

    print("\nAll 14 tests passed.")


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Lesson 77 — ONNX Model Export"
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
        "--epochs",
        type=int,
        default=5,
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
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=100,
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

    if args.epochs <= 0:

        raise ValueError(
            "Epochs must be positive."
        )

    if args.batch_size <= 0:

        raise ValueError(
            "Batch size must be positive."
        )

    if args.learning_rate <= 0:

        raise ValueError(
            "Learning rate must be positive."
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

    print("LESSON 77 — ONNX MODEL EXPORT")

    print("=" * 75)

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"Device: {device}"
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

        dataset = SyntheticDataset(
            samples_per_class=args.samples_per_class,
            image_size=args.image_size,
        )

        class_names = CLASS_NAMES

        num_classes = NUM_CLASSES

        dataset_type = "synthetic_geometric"

        print(
            "\nWARNING: Using synthetic geometric images."
        )

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
        f"\nDataset type: {dataset_type}"
    )

    print(
        f"Classes: {class_names}"
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    # --------------------------------------------------------------
    # MODEL
    # --------------------------------------------------------------

    model = VisionCNN(
        num_classes
    )

    model.to(device)

    print(
        f"\nParameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # --------------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------------

    history = train_model(
        model,
        train_loader,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    evaluation = evaluate(
        model,
        val_loader,
        device,
    )

    print(
        f"\nValidation accuracy: "
        f"{evaluation['accuracy']:.4f}"
    )

    # --------------------------------------------------------------
    # SAVE PYTORCH CHECKPOINT
    # --------------------------------------------------------------

    pytorch_path = (
        output_dir / "model.pth"
    )

    torch.save(
        model.cpu().state_dict(),
        pytorch_path,
    )

    # --------------------------------------------------------------
    # EXPORT ONNX
    # --------------------------------------------------------------

    onnx_path = (
        output_dir / "model.onnx"
    )

    export_onnx(
        model,
        onnx_path,
        args.image_size,
        dynamic_batch=True,
    )

    # --------------------------------------------------------------
    # VALIDATE ONNX
    # --------------------------------------------------------------

    graph_info = validate_onnx(
        onnx_path
    )

    # --------------------------------------------------------------
    # CREATE ONNX RUNTIME SESSION
    # --------------------------------------------------------------

    session = create_onnx_session(
        onnx_path
    )

    # --------------------------------------------------------------
    # NUMERICAL EQUIVALENCE
    # --------------------------------------------------------------

    equivalence_results = []

    for batch_size in [1, 2, 4, 8]:

        result = compare_outputs(
            model,
            session,
            batch_size,
            args.image_size,
            torch.device("cpu"),
        )

        equivalence_results.append(
            result
        )

        print(
            f"\nBatch {batch_size}: "
            f"max error={result['max_absolute_error']:.8e}, "
            f"mean error={result['mean_absolute_error']:.8e}, "
            f"agreement={result['prediction_agreement']:.4f}"
        )

    # --------------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------------

    pytorch_benchmarks = []

    onnx_benchmarks = []

    model.to(device)

    for batch_size in [1, 4, 8]:

        torch_result = benchmark_pytorch(
            model,
            device,
            args.image_size,
            batch_size=batch_size,
            iterations=args.benchmark_iterations,
        )

        onnx_result = benchmark_onnx(
            session,
            args.image_size,
            batch_size=batch_size,
            iterations=args.benchmark_iterations,
        )

        pytorch_benchmarks.append(
            torch_result
        )

        onnx_benchmarks.append(
            onnx_result
        )

    # --------------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------------

    plot_equivalence(
        equivalence_results,
        output_dir,
    )

    plot_benchmark(
        pytorch_benchmarks,
        onnx_benchmarks,
        output_dir,
    )

    # --------------------------------------------------------------
    # MODEL FILE SIZES
    # --------------------------------------------------------------

    pytorch_size = file_size_bytes(
        pytorch_path
    )

    onnx_size = file_size_bytes(
        onnx_path
    )

    # --------------------------------------------------------------
    # REPORT
    # --------------------------------------------------------------

    report = {

        "lesson": 77,

        "title": "ONNX Model Export",

        "dataset_type": dataset_type,

        "class_names": class_names,

        "image_size": args.image_size,

        "validation": evaluation,

        "training_history": history,

        "onnx_graph": graph_info,

        "equivalence": equivalence_results,

        "pytorch_benchmark": pytorch_benchmarks,

        "onnx_benchmark": onnx_benchmarks,

        "file_sizes": {

            "pytorch_bytes": pytorch_size,

            "onnx_bytes": onnx_size,

            "pytorch_mb": pytorch_size / 1024**2,

            "onnx_mb": onnx_size / 1024**2,

        },

    }

    with open(
        output_dir / "onnx_export_report.json",
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

    print("\n" + "=" * 75)

    print("EXPORT SUMMARY")

    print("=" * 75)

    print(
        f"PyTorch file: {pytorch_size / 1024**2:.3f} MB"
    )

    print(
        f"ONNX file: {onnx_size / 1024**2:.3f} MB"
    )

    print(
        f"Graph nodes: {graph_info['nodes']}"
    )

    print(
        f"ONNX opset: {graph_info['opset']}"
    )

    print("\nGenerated files:")

    for path in sorted(
        output_dir.iterdir()
    ):

        print(
            f"  {path}"
        )

    print("\nLESSON 77 COMPLETED")

    print(
        "Next: Lesson 78 — ONNX Runtime Optimization."
    )


if __name__ == "__main__":

    main()