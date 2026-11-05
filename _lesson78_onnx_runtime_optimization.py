"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 6 — MODEL OPTIMIZATION AND DEPLOYMENT

LESSON 78 — ONNX RUNTIME OPTIMIZATION
======================================================================

OBJECTIVES
----------

1. Train a CNN using real images.
2. Export the CNN to ONNX.
3. Run inference with ONNX Runtime.
4. Compare graph optimization levels.
5. Generate optimized ONNX files.
6. Compare inference latency and throughput.
7. Investigate CPU threading.
8. Profile ONNX Runtime execution.
9. Verify numerical equivalence.

NO pretrained models.
NO pretrained weights.

======================================================================
1. ONNX RUNTIME
======================================================================

ONNX is a model representation format.

ONNX Runtime is an inference engine.

    PyTorch
       |
       v
    model.onnx
       |
       v
    ONNX Runtime
       |
       v
    Execution Provider
       |
       +----------+
       |          |
       v          v
      CPU        GPU

Execution providers include:

    CPUExecutionProvider
    CUDAExecutionProvider
    TensorRTExecutionProvider

Availability depends on the installed ONNX Runtime package
and the hardware/software environment.

======================================================================
2. GRAPH OPTIMIZATION
======================================================================

ONNX Runtime provides four main graph optimization levels.

DISABLE:

    Disable graph optimizations.

BASIC:

    Basic graph optimizations, such as constant folding,
    redundant-node elimination, and simple graph simplification.

EXTENDED:

    Includes basic optimizations and more advanced operator
    fusions supported by the execution provider.

ALL:

    Includes all supported optimizations.

Examples of graph transformations:

    Conv + BatchNorm
           |
           v
    Fused Conv

    Identity -> Identity
           |
           v
       Removed

    Constant arithmetic
           |
           v
    Precomputed constant

The exact transformations depend on the model, runtime version,
execution provider, and hardware.

======================================================================
3. CPU THREADING
======================================================================

ONNX Runtime CPU execution can use multiple threads.

intra_op_num_threads:

    Number of threads used to parallelize individual operators.

inter_op_num_threads:

    Number of threads used to execute independent graph nodes
    in parallel when parallel execution mode is enabled.

More threads do not always mean faster inference.

Too many threads can cause:

    - CPU contention
    - Context switching
    - Memory bandwidth pressure
    - Worse latency for small models

======================================================================
4. BENCHMARKING
======================================================================

Latency:

    Time required for one inference request.

Throughput:

    Number of images processed per second.

    throughput = batch_size / latency_seconds

P50:

    Median latency.

P95:

    95th percentile latency.

P95 is useful when evaluating latency consistency.

======================================================================
5. NUMERICAL EQUIVALENCE
======================================================================

Compare:

    PyTorch output
    ONNX Runtime output

    max_error = max(abs(torch_output - onnx_output))

Graph optimization should preserve model behavior within
reasonable floating-point tolerances.

======================================================================
6. ROBOTICS APPLICATION
======================================================================

OpenArm V2 perception pipeline:

    Camera frames
         |
         v
    Image preprocessing
         |
         v
    ONNX Runtime
         |
         v
    Object perception
         |
         v
    Object localization
         |
         v
    ACT manipulation policy
         |
         v
    Robot control

For a real robot, optimize the complete pipeline, not just
the neural network:

    camera capture
    preprocessing
    inference
    postprocessing
    communication
    robot control

======================================================================
NEXT LESSON
-----------

Lesson 79 — NVIDIA TensorRT.
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
    DataLoader,
    Subset,
    random_split,
)

import matplotlib.pyplot as plt


# ======================================================================
# CONSTANTS
# ======================================================================

DEFAULT_IMAGE_SIZE = 64

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

OPTIMIZATION_LEVELS = [
    "DISABLE",
    "BASIC",
    "EXTENDED",
    "ALL",
]


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ======================================================================
# DEVICE
# ======================================================================

def get_device(name):

    if name == "auto":

        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if name == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        return torch.device("cuda")

    return torch.device("cpu")


def synchronize(device):

    if device.type == "cuda":

        torch.cuda.synchronize(device)


# ======================================================================
# MODEL ARCHITECTURE
# ======================================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=False
        )

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        x = self.relu(x)

        x = self.pool(x)

        return x


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

        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128, 128),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, num_classes),

        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x


# ======================================================================
# DATASET
# ======================================================================

def build_datasets(args):

    from torchvision import datasets
    from torchvision import transforms

    transform = transforms.Compose([

        transforms.Resize(
            (args.image_size, args.image_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),

    ])

    if args.data_dir is None:

        print(
            "\nDownloading/loading real CIFAR-10 images..."
        )

        train_dataset = datasets.CIFAR10(

            root=args.dataset_root,

            train=True,

            download=True,

            transform=transform,

        )

        test_dataset = datasets.CIFAR10(

            root=args.dataset_root,

            train=False,

            download=True,

            transform=transform,

        )

        class_names = CIFAR10_CLASSES

    else:

        from torchvision.datasets import ImageFolder

        root = Path(args.data_dir)

        if not root.exists():

            raise FileNotFoundError(
                f"Dataset directory does not exist: {root}"
            )

        dataset = ImageFolder(
            root=str(root),
            transform=transform,
        )

        if len(dataset.classes) < 2:

            raise ValueError(
                "ImageFolder requires at least two classes."
            )

        class_names = dataset.classes

        train_size = int(
            len(dataset) * 0.8
        )

        test_size = len(dataset) - train_size

        if train_size == 0 or test_size == 0:

            raise ValueError(
                "Dataset must contain enough images for a train/test split."
            )

        train_dataset, test_dataset = random_split(

            dataset,

            [train_size, test_size],

            generator=torch.Generator().manual_seed(42),

        )

    # Limit dataset sizes when requested.
    # This is useful for quick optimization experiments.

    if args.max_train_samples > 0:

        count = min(
            args.max_train_samples,
            len(train_dataset),
        )

        indices = list(range(count))

        train_dataset = Subset(
            train_dataset,
            indices,
        )

    if args.max_test_samples > 0:

        count = min(
            args.max_test_samples,
            len(test_dataset),
        )

        indices = list(range(count))

        test_dataset = Subset(
            test_dataset,
            indices,
        )

    return (
        train_dataset,
        test_dataset,
        class_names,
    )


# ======================================================================
# TRAINING
# ======================================================================

def train_model(
    model,
    train_loader,
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

        running_loss = 0.0

        correct = 0

        total = 0

        for images, labels in train_loader:

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
                loss.item() * labels.size(0)
            )

            correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            total += labels.size(0)

        epoch_loss = running_loss / max(total, 1)

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


def evaluate_model(
    model,
    loader,
    device,
):

    model.eval()

    correct = 0

    total = 0

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
                loss.item() * labels.size(0)
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
# EXPORT TO ONNX
# ======================================================================

def export_onnx(
    model,
    output_path,
    image_size,
):

    model.eval()

    model.cpu()

    example = torch.randn(
        1,
        3,
        image_size,
        image_size,
    )

    torch.onnx.export(

        model,

        example,

        str(output_path),

        input_names=["images"],

        output_names=["logits"],

        dynamic_axes={

            "images": {
                0: "batch_size",
            },

            "logits": {
                0: "batch_size",
            },

        },

        opset_version=17,

        do_constant_folding=True,

        dynamo=False,

    )

    print(
        f"\nExported ONNX model: {output_path}"
    )


# ======================================================================
# ONNX GRAPH VALIDATION
# ======================================================================

def validate_onnx(path):

    import onnx

    model = onnx.load(
        str(path)
    )

    onnx.checker.check_model(
        model
    )

    print(
        "\nONNX graph validation passed."
    )

    print(
        f"Nodes: {len(model.graph.node)}"
    )

    print(
        f"Opset: {model.opset_import[0].version}"
    )

    return {

        "node_count": len(model.graph.node),

        "opset": model.opset_import[0].version,

    }


# ======================================================================
# ONNX RUNTIME OPTIMIZATION CONFIGURATION
# ======================================================================

def create_session_options(
    optimization_level,
    intra_threads,
    inter_threads,
    optimized_path=None,
    enable_profiling=False,
    profile_prefix=None,
):

    import onnxruntime as ort

    options = ort.SessionOptions()

    levels = {

        "DISABLE": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,

        "BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,

        "EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,

        "ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,

    }

    options.graph_optimization_level = levels[
        optimization_level
    ]

    options.intra_op_num_threads = intra_threads

    options.inter_op_num_threads = inter_threads

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    options.enable_profiling = enable_profiling

    if profile_prefix is not None:

        options.profile_file_prefix = str(
            profile_prefix
        )

    if optimized_path is not None:

        options.optimized_model_filepath = str(
            optimized_path
        )

    return options


# ======================================================================
# CREATE SESSION
# ======================================================================

def create_session(
    onnx_path,
    optimization_level,
    intra_threads,
    inter_threads,
    optimized_path=None,
):

    import onnxruntime as ort

    options = create_session_options(

        optimization_level=optimization_level,

        intra_threads=intra_threads,

        inter_threads=inter_threads,

        optimized_path=optimized_path,

    )

    session = ort.InferenceSession(

        str(onnx_path),

        sess_options=options,

        providers=[
            "CPUExecutionProvider"
        ],

    )

    return session


# ======================================================================
# NUMERICAL EQUIVALENCE
# ======================================================================

def compare_outputs(
    pytorch_model,
    session,
    image_size,
    batch_size,
):

    rng = np.random.default_rng(
        100 + batch_size
    )

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

    pytorch_model.eval()

    with torch.inference_mode():

        torch_output = pytorch_model(
            torch.from_numpy(images)
        ).numpy()

    onnx_output = session.run(

        ["logits"],

        {
            "images": images,
        },

    )[0]

    difference = np.abs(
        torch_output - onnx_output
    )

    max_error = float(
        difference.max()
    )

    mean_error = float(
        difference.mean()
    )

    relative_l2 = float(

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

        "max_absolute_error": max_error,

        "mean_absolute_error": mean_error,

        "relative_l2_error": relative_l2,

        "prediction_agreement": prediction_agreement,

    }


# ======================================================================
# ONNX RUNTIME BENCHMARK
# ======================================================================

def benchmark_session(
    session,
    image_size,
    batch_size,
    warmup,
    iterations,
):

    rng = np.random.default_rng(
        42 + batch_size
    )

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

    inputs = {
        "images": images,
    }

    # Warm-up

    for _ in range(warmup):

        session.run(
            ["logits"],
            inputs,
        )

    # Timed execution

    timings = []

    for _ in range(iterations):

        start = time.perf_counter()

        session.run(
            ["logits"],
            inputs,
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        timings.append(elapsed)

    timings = np.asarray(
        timings,
        dtype=np.float64,
    )

    mean_latency = float(
        timings.mean()
    )

    return {

        "batch_size": batch_size,

        "mean_latency_ms": mean_latency,

        "p50_latency_ms": float(
            np.percentile(timings, 50)
        ),

        "p95_latency_ms": float(
            np.percentile(timings, 95)
        ),

        "min_latency_ms": float(
            timings.min()
        ),

        "max_latency_ms": float(
            timings.max()
        ),

        "throughput_images_per_second": (

            batch_size * 1000.0 / mean_latency

        ),

    }


# ======================================================================
# PYTORCH CPU BENCHMARK
# ======================================================================

def benchmark_pytorch(
    model,
    image_size,
    batch_size,
    warmup,
    iterations,
):

    model.eval()

    model.cpu()

    images = torch.randn(

        batch_size,

        3,

        image_size,

        image_size,

    )

    with torch.inference_mode():

        for _ in range(warmup):

            model(images)

    timings = []

    for _ in range(iterations):

        start = time.perf_counter()

        with torch.inference_mode():

            model(images)

        elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        timings.append(elapsed)

    timings = np.asarray(
        timings,
        dtype=np.float64,
    )

    mean_latency = float(
        timings.mean()
    )

    return {

        "batch_size": batch_size,

        "mean_latency_ms": mean_latency,

        "p50_latency_ms": float(
            np.percentile(timings, 50)
        ),

        "p95_latency_ms": float(
            np.percentile(timings, 95)
        ),

        "throughput_images_per_second": (

            batch_size * 1000.0 / mean_latency

        ),

    }


# ======================================================================
# ONNX RUNTIME PROFILING
# ======================================================================

def run_runtime_profile(
    onnx_path,
    output_dir,
    image_size,
    batch_size,
    intra_threads,
):

    import onnxruntime as ort

    profile_prefix = (
        output_dir / "ort_profile"
    )

    options = create_session_options(

        optimization_level="ALL",

        intra_threads=intra_threads,

        inter_threads=1,

        enable_profiling=True,

        profile_prefix=profile_prefix,

    )

    session = ort.InferenceSession(

        str(onnx_path),

        sess_options=options,

        providers=[
            "CPUExecutionProvider"
        ],

    )

    images = np.random.default_rng(
        42
    ).normal(

        0.0,

        1.0,

        size=(

            batch_size,

            3,

            image_size,

            image_size,

        ),

    ).astype(np.float32)

    for _ in range(10):

        session.run(

            ["logits"],

            {
                "images": images,
            },

        )

    profile_path = session.end_profiling()

    print(
        f"\nONNX Runtime profile saved: {profile_path}"
    )

    return profile_path


# ======================================================================
# VISUALIZATION
# ======================================================================

def plot_latency(
    benchmark_results,
    output_dir,
):

    levels = list(
        benchmark_results.keys()
    )

    batch_sizes = sorted({

        item["batch_size"]

        for level in levels

        for item in benchmark_results[level]

    })

    for batch_size in batch_sizes:

        labels = []

        values = []

        for level in levels:

            matching = [

                item

                for item in benchmark_results[level]

                if item["batch_size"] == batch_size

            ]

            if matching:

                labels.append(level)

                values.append(
                    matching[0]["mean_latency_ms"]
                )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.bar(
            labels,
            values,
        )

        ax.set_title(
            f"ONNX Runtime Optimization — Batch {batch_size}"
        )

        ax.set_xlabel(
            "Graph optimization level"
        )

        ax.set_ylabel(
            "Mean latency (ms)"
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        fig.tight_layout()

        fig.savefig(

            output_dir / (
                f"latency_batch_{batch_size}.png"
            ),

            dpi=150,

        )

        plt.close(fig)


def plot_throughput(
    benchmark_results,
    output_dir,
):

    levels = list(
        benchmark_results.keys()
    )

    batch_sizes = sorted({

        item["batch_size"]

        for level in levels

        for item in benchmark_results[level]

    })

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = np.arange(
        len(batch_sizes)
    )

    width = 0.8 / max(len(levels), 1)

    for index, level in enumerate(levels):

        values = []

        for batch_size in batch_sizes:

            matching = [

                item

                for item in benchmark_results[level]

                if item["batch_size"] == batch_size

            ]

            values.append(

                matching[0]["throughput_images_per_second"]

                if matching else 0.0

            )

        offset = (
            index - (len(levels) - 1) / 2
        ) * width

        ax.bar(
            x + offset,
            values,
            width,
            label=level,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [str(value) for value in batch_sizes]
    )

    ax.set_xlabel(
        "Batch size"
    )

    ax.set_ylabel(
        "Images per second"
    )

    ax.set_title(
        "ONNX Runtime Throughput Comparison"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "throughput_comparison.png",
        dpi=150,
    )

    plt.close(fig)


# ======================================================================
# TESTS
# ======================================================================

def run_tests():

    print(
        "\nRunning Lesson 78 tests...\n"
    )

    import onnx
    import onnxruntime as ort

    set_seed(42)

    test_dir = Path(
        "runs/lesson78_tests"
    )

    test_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = VisionCNN(
        num_classes=10
    )

    model.eval()

    # Test 1: Output shape.

    x = torch.randn(
        2,
        3,
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )

    with torch.inference_mode():

        y = model(x)

    assert y.shape == (2, 10)

    print(
        "Test 1 passed: Model output shape."
    )

    # Test 2: Finite outputs.

    assert torch.isfinite(y).all()

    print(
        "Test 2 passed: Finite model outputs."
    )

    # Test 3: ONNX export.

    onnx_path = (
        test_dir / "test_model.onnx"
    )

    export_onnx(
        model,
        onnx_path,
        DEFAULT_IMAGE_SIZE,
    )

    assert onnx_path.exists()

    print(
        "Test 3 passed: ONNX export."
    )

    # Test 4: ONNX graph validation.

    graph = onnx.load(
        str(onnx_path)
    )

    onnx.checker.check_model(
        graph
    )

    print(
        "Test 4 passed: ONNX graph validation."
    )

    # Test 5: Optimization level enumeration.

    assert OPTIMIZATION_LEVELS == [
        "DISABLE",
        "BASIC",
        "EXTENDED",
        "ALL",
    ]

    print(
        "Test 5 passed: Optimization level definitions."
    )

    # Test 6: Create all optimization sessions.

    sessions = {}

    for level in OPTIMIZATION_LEVELS:

        optimized_path = (
            test_dir / f"optimized_{level.lower()}.onnx"
        )

        session = create_session(

            onnx_path,

            level,

            intra_threads=1,

            inter_threads=1,

            optimized_path=optimized_path,

        )

        sessions[level] = session

    print(
        "Test 6 passed: All optimization sessions created."
    )

    # Test 7: Dynamic batch inference.

    for batch_size in [1, 2, 4]:

        images = np.random.randn(

            batch_size,

            3,

            DEFAULT_IMAGE_SIZE,

            DEFAULT_IMAGE_SIZE,

        ).astype(np.float32)

        output = sessions["ALL"].run(

            ["logits"],

            {
                "images": images,
            },

        )[0]

        assert output.shape == (
            batch_size,
            10,
        )

    print(
        "Test 7 passed: Dynamic batch inference."
    )

    # Test 8: Numerical equivalence.

    for level, session in sessions.items():

        result = compare_outputs(

            model,

            session,

            DEFAULT_IMAGE_SIZE,

            batch_size=3,

        )

        assert result["max_absolute_error"] < 1e-4, (
            f"{level}: numerical equivalence failed: {result}"
        )

    print(
        "Test 8 passed: Numerical equivalence for all optimization levels."
    )

    # Test 9: Prediction agreement.

    result = compare_outputs(

        model,

        sessions["ALL"],

        DEFAULT_IMAGE_SIZE,

        batch_size=4,

    )

    assert result["prediction_agreement"] == 1.0

    print(
        "Test 9 passed: Prediction agreement."
    )

    # Test 10: Session input metadata.

    inputs = sessions["ALL"].get_inputs()

    assert len(inputs) == 1

    assert inputs[0].name == "images"

    print(
        "Test 10 passed: Input metadata."
    )

    # Test 11: Session output metadata.

    outputs = sessions["ALL"].get_outputs()

    assert len(outputs) == 1

    assert outputs[0].name == "logits"

    print(
        "Test 11 passed: Output metadata."
    )

    # Test 12: CPU provider.

    assert (
        "CPUExecutionProvider"
        in sessions["ALL"].get_providers()
    )

    print(
        "Test 12 passed: CPU execution provider."
    )

    # Test 13: Optimized graph files.

    for level in OPTIMIZATION_LEVELS:

        path = (
            test_dir / f"optimized_{level.lower()}.onnx"
        )

        assert path.exists()

        assert path.stat().st_size > 0

    print(
        "Test 13 passed: Optimized graph files."
    )

    # Test 14: Softmax normalization.

    probabilities = F.softmax(
        y,
        dim=1,
    )

    assert torch.allclose(

        probabilities.sum(dim=1),

        torch.ones(2),

        atol=1e-6,

    )

    print(
        "Test 14 passed: Softmax normalization."
    )

    # Test 15: Benchmark output.

    benchmark = benchmark_session(

        sessions["ALL"],

        DEFAULT_IMAGE_SIZE,

        batch_size=1,

        warmup=2,

        iterations=5,

    )

    assert benchmark["mean_latency_ms"] > 0

    assert benchmark[
        "throughput_images_per_second"
    ] > 0

    print(
        "Test 15 passed: ONNX Runtime benchmark."
    )

    print(
        "\nAll 15 tests passed."
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Lesson 78 — ONNX Runtime Optimization"
        )

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
        help="Optional ImageFolder directory containing class folders.",
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./datasets",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--intra-threads",
        type=int,
        default=0,
        help="0 lets ONNX Runtime choose its default.",
    )

    parser.add_argument(
        "--inter-threads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/lesson78",
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
            "Epoch count must be positive."
        )

    if args.batch_size <= 0:

        raise ValueError(
            "Batch size must be positive."
        )

    if args.iterations <= 0:

        raise ValueError(
            "Benchmark iterations must be positive."
        )

    if args.warmup < 0:

        raise ValueError(
            "Warm-up count cannot be negative."
        )

    if args.intra_threads < 0:

        raise ValueError(
            "intra-threads cannot be negative."
        )

    if args.inter_threads <= 0:

        raise ValueError(
            "inter-threads must be positive."
        )

    set_seed(42)

    device = get_device(
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

    print(
        "LESSON 78 — ONNX RUNTIME OPTIMIZATION"
    )

    print("=" * 75)

    print(
        f"PyTorch version: {torch.__version__}"
    )

    print(
        f"Training device: {device}"
    )

    # --------------------------------------------------------------
    # DATASET
    # --------------------------------------------------------------

    train_dataset, test_dataset, class_names = build_datasets(
        args
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=args.workers,

        pin_memory=(device.type == "cuda"),

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.workers,

        pin_memory=(device.type == "cuda"),

    )

    num_classes = len(
        class_names
    )

    print(
        f"\nClasses: {class_names}"
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    # --------------------------------------------------------------
    # MODEL
    # --------------------------------------------------------------

    model = VisionCNN(
        num_classes
    )

    parameter_count = sum(

        parameter.numel()

        for parameter in model.parameters()

    )

    print(
        f"\nModel parameters: {parameter_count:,}"
    )

    # --------------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------------

    print(
        "\nTraining CNN..."
    )

    training_history = train_model(

        model,

        train_loader,

        device,

        epochs=args.epochs,

        learning_rate=args.learning_rate,

    )

    model_evaluation = evaluate_model(

        model,

        test_loader,

        device,

    )

    print(
        f"\nTest accuracy: {model_evaluation['accuracy']:.4f}"
    )

    # --------------------------------------------------------------
    # SAVE PYTORCH MODEL
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

    )

    graph_info = validate_onnx(
        onnx_path
    )

    # --------------------------------------------------------------
    # CREATE OPTIMIZED SESSIONS
    # --------------------------------------------------------------

    benchmark_results = {}

    equivalence_results = {}

    session_info = {}

    for level in OPTIMIZATION_LEVELS:

        print("\n" + "-" * 65)

        print(
            f"OPTIMIZATION LEVEL: {level}"
        )

        print("-" * 65)

        optimized_path = (

            output_dir

            / f"model_optimized_{level.lower()}.onnx"

        )

        session = create_session(

            onnx_path,

            optimization_level=level,

            intra_threads=args.intra_threads,

            inter_threads=args.inter_threads,

            optimized_path=optimized_path,

        )

        session_info[level] = {

            "providers": session.get_providers(),

            "optimized_model": str(optimized_path),

            "optimized_model_bytes": (
                optimized_path.stat().st_size
                if optimized_path.exists()
                else None
            ),

        }

        # ----------------------------------------------------------
        # EQUIVALENCE
        # ----------------------------------------------------------

        equivalence = compare_outputs(

            model,

            session,

            args.image_size,

            batch_size=4,

        )

        equivalence_results[level] = equivalence

        print(
            f"Maximum error: "
            f"{equivalence['max_absolute_error']:.8e}"
        )

        print(
            f"Mean error: "
            f"{equivalence['mean_absolute_error']:.8e}"
        )

        print(
            f"Prediction agreement: "
            f"{equivalence['prediction_agreement']:.4f}"
        )

        # ----------------------------------------------------------
        # BENCHMARK
        # ----------------------------------------------------------

        level_results = []

        for batch_size in [1, 4, 8]:

            result = benchmark_session(

                session,

                image_size=args.image_size,

                batch_size=batch_size,

                warmup=args.warmup,

                iterations=args.iterations,

            )

            level_results.append(
                result
            )

            print(

                f"Batch {batch_size}: "

                f"{result['mean_latency_ms']:.3f} ms "

                f"| P95={result['p95_latency_ms']:.3f} ms "

                f"| throughput="
                f"{result['throughput_images_per_second']:.1f} img/s"

            )

        benchmark_results[level] = level_results

    # --------------------------------------------------------------
    # PYTORCH CPU BENCHMARK
    # --------------------------------------------------------------

    print(
        "\nBenchmarking PyTorch CPU..."
    )

    pytorch_cpu_results = []

    model.cpu()

    for batch_size in [1, 4, 8]:

        result = benchmark_pytorch(

            model,

            args.image_size,

            batch_size,

            args.warmup,

            args.iterations,

        )

        pytorch_cpu_results.append(
            result
        )

        print(

            f"Batch {batch_size}: "

            f"{result['mean_latency_ms']:.3f} ms "

            f"| throughput="
            f"{result['throughput_images_per_second']:.1f} img/s"

        )

    # --------------------------------------------------------------
    # PROFILING
    # --------------------------------------------------------------

    profile_path = run_runtime_profile(

        onnx_path,

        output_dir,

        args.image_size,

        batch_size=1,

        intra_threads=args.intra_threads,

    )

    # --------------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------------

    plot_latency(
        benchmark_results,
        output_dir,
    )

    plot_throughput(
        benchmark_results,
        output_dir,
    )

    # --------------------------------------------------------------
    # MODEL FILE SIZES
    # --------------------------------------------------------------

    file_sizes = {

        "pytorch_model_bytes": pytorch_path.stat().st_size,

        "onnx_model_bytes": onnx_path.stat().st_size,

    }

    for level in OPTIMIZATION_LEVELS:

        path = (

            output_dir

            / f"model_optimized_{level.lower()}.onnx"

        )

        if path.exists():

            file_sizes[
                f"optimized_{level.lower()}_bytes"
            ] = path.stat().st_size

    # --------------------------------------------------------------
    # REPORT
    # --------------------------------------------------------------

    report = {

        "lesson": 78,

        "title": "ONNX Runtime Optimization",

        "dataset": {

            "classes": class_names,

            "train_samples": len(train_dataset),

            "test_samples": len(test_dataset),

            "image_size": args.image_size,

        },

        "model": {

            "architecture": "VisionCNN",

            "parameters": parameter_count,

            "test_metrics": model_evaluation,

            "training_history": training_history,

        },

        "onnx_graph": graph_info,

        "sessions": session_info,

        "equivalence": equivalence_results,

        "onnx_runtime_benchmarks": benchmark_results,

        "pytorch_cpu_benchmarks": pytorch_cpu_results,

        "file_sizes": file_sizes,

        "profile_file": profile_path,

        "thread_configuration": {

            "intra_op_num_threads": args.intra_threads,

            "inter_op_num_threads": args.inter_threads,

        },

    }

    report_path = (
        output_dir / "runtime_optimization_report.json"
    )

    with open(
        report_path,
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

    print(
        "LESSON 78 SUMMARY"
    )

    print("=" * 75)

    print(
        f"PyTorch accuracy: {model_evaluation['accuracy']:.4f}"
    )

    print(
        f"ONNX original size: "
        f"{file_sizes['onnx_model_bytes'] / 1024**2:.3f} MB"
    )

    print("\nOptimization results:")

    for level in OPTIMIZATION_LEVELS:

        values = benchmark_results[level]

        batch_one = next(

            item

            for item in values

            if item["batch_size"] == 1

        )

        print(

            f"{level:10s} "

            f"| batch-1 latency: "
            f"{batch_one['mean_latency_ms']:.3f} ms"

        )

    print(
        "\nOutput directory:"
    )

    print(
        output_dir.resolve()
    )

    print(
        "\nLESSON 78 COMPLETED."
    )

    print(
        "Next: Lesson 79 — NVIDIA TensorRT."
    )


if __name__ == "__main__":

    main()