"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 6 — MODEL OPTIMIZATION AND DEPLOYMENT

LESSON 72 — BATCH SIZE OPTIMIZATION
======================================================================

OBJECTIVE
---------

Understand how batch size affects neural network inference.

We will investigate:

    1. Batch processing.
    2. Batch latency.
    3. Per-image latency.
    4. Throughput.
    5. Memory consumption.
    6. CPU versus GPU batch performance.
    7. Latency constraints.
    8. Batch-size selection.
    9. Out-of-memory handling.
   10. Real-time robotics implications.

======================================================================
1. WHAT IS BATCH SIZE?
======================================================================

A batch is a group of inputs processed together.

Batch size = 1:

    Image 1
       |
       v
    Model
       |
       v
    Prediction 1

Batch size = 4:

    Image 1 ─┐
    Image 2 ─┤
    Image 3 ─┼──> Model ──> Predictions
    Image 4 ─┘

The neural network processes the batch as one tensor.

For RGB images of resolution 224 x 224:

    Batch 1:

        [1, 3, 224, 224]

    Batch 4:

        [4, 3, 224, 224]

    Batch 16:

        [16, 3, 224, 224]

======================================================================
2. BATCH LATENCY
======================================================================

Batch latency is the time required to process an entire batch.

Example:

    Batch size = 8

    Batch latency = 40 ms

The model processes eight images in 40 ms.

======================================================================
3. PER-IMAGE LATENCY
======================================================================

Average computational latency per image:

    Per-image latency = Batch latency / Batch size

Example:

    40 ms / 8 = 5 ms per image

IMPORTANT:

This is an average computational cost per image.

It is NOT necessarily the actual waiting time experienced by
each individual image in a live streaming system.

======================================================================
4. THROUGHPUT
======================================================================

Throughput measures images processed per second.

    Throughput = Batch size / Batch latency in seconds

Example:

    Batch size = 8
    Batch latency = 40 ms

    Throughput = 8 / 0.040

               = 200 images/second

======================================================================
5. WHY LARGER BATCHES CAN BE FASTER
======================================================================

GPUs are designed for parallel computation.

A larger batch can provide:

    - More parallel work.
    - Better hardware utilization.
    - More efficient matrix operations.
    - Lower overhead per image.

However, performance does not increase indefinitely.

======================================================================
6. WHY LARGER BATCHES CAN BE SLOWER
======================================================================

Larger batches require more memory.

Memory is consumed by:

    - Input images.
    - Intermediate feature maps.
    - Temporary tensors.
    - Model parameters.
    - Runtime allocations.

Eventually:

    - Memory usage becomes excessive.
    - Cache behavior may change.
    - The GPU may run out of memory.
    - Latency may become unacceptable.

======================================================================
7. LATENCY VS THROUGHPUT
======================================================================

Consider these hypothetical measurements:

    Batch 1:
        Latency = 10 ms
        Throughput = 100 FPS

    Batch 4:
        Latency = 20 ms
        Throughput = 200 FPS

    Batch 8:
        Latency = 40 ms
        Throughput = 200 FPS

Batch 4 and Batch 8 have similar throughput.

However, Batch 4 completes each batch sooner and uses less memory.

The best batch size depends on the application.

======================================================================
8. ROBOTICS EXAMPLE
======================================================================

Imagine an OpenArm V2 waste-sorting robot with five cameras:

    Ceiling
    Head Left
    Head Right
    Wrist Left
    Wrist Right

One possible approach:

    Five camera frames
           |
           v
       Batch size 5
           |
           v
     Object Detection
           |
           v
     Object Localization
           |
           v
       Grasp Planning

Batching can improve GPU efficiency.

BUT:

The cameras may produce frames at different times.

Waiting for every camera before inference can increase latency.

For a real-time robot, a batch size of 1 may be better than
a batch size of 5 if immediate responses are important.

======================================================================
9. OPTIMIZATION OBJECTIVES
======================================================================

There is no universally optimal batch size.

We can define different objectives:

A. Maximum throughput:

    Choose the batch size with the highest FPS.

B. Minimum latency:

    Choose the batch size with the lowest batch latency.

C. Latency-constrained throughput:

    Maximize throughput while:

        P95 batch latency <= latency budget

D. Memory-constrained throughput:

    Maximize throughput while remaining under a memory budget.

This lesson implements objective C.

======================================================================
10. BENCHMARKING METHODOLOGY
======================================================================

For every batch size:

    1. Create input tensor.
    2. Warm up the model.
    3. Reset CUDA peak-memory statistics.
    4. Measure individual batch latency.
    5. Calculate throughput.
    6. Measure memory consumption.
    7. Calculate P50, P95, and P99.
    8. Save results.

The benchmark uses:

    model.eval()
    torch.inference_mode()
    torch.cuda.synchronize()

CUDA synchronization is necessary for accurate wall-clock timing.

======================================================================
WHAT YOU WILL LEARN
======================================================================

After this lesson, you should be able to:

    - Explain batch processing.
    - Calculate throughput.
    - Interpret per-image latency.
    - Compare different batch sizes.
    - Understand batch-memory relationships.
    - Select a batch size under a latency budget.
    - Design a basic batching strategy for multiple cameras.

NEXT LESSON
-----------

Lesson 73 — Mixed Precision FP16.

======================================================================
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

DEFAULT_IMAGE_SIZE = 224

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16]

DEFAULT_WARMUP = 20

DEFAULT_ITERATIONS = 100

DEFAULT_LATENCY_BUDGET_MS = 50.0

DEFAULT_OUTPUT_DIR = Path("runs/lesson72")


# ======================================================================
# DEVICE UTILITIES
# ======================================================================

def synchronize(device: torch.device) -> None:

    if device.type == "cuda":

        torch.cuda.synchronize(device)


def select_device(
    requested_device: str,
) -> torch.device:

    if requested_device == "cpu":

        return torch.device("cpu")

    if requested_device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        return torch.device("cuda")

    if requested_device == "auto":

        if torch.cuda.is_available():

            return torch.device("cuda")

        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device: {requested_device}"
    )


# ======================================================================
# CNN ARCHITECTURE
# ======================================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            out_channels
        )

        self.activation = nn.ReLU(
            inplace=False
        )

    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        x = self.activation(x)

        return x


class BatchBenchmarkCNN(nn.Module):

    """
    CNN for batch-size experiments.

    Input:
        B x 3 x H x W

    Output:
        B x 10
    """

    def __init__(
        self,
        num_classes: int = 10,
    ):

        super().__init__()

        self.stem = ConvBlock(
            3,
            32,
            stride=2,
        )

        self.block1 = ConvBlock(
            32,
            64,
            stride=2,
        )

        self.block2 = ConvBlock(
            64,
            128,
            stride=2,
        )

        self.block3 = ConvBlock(
            128,
            256,
            stride=2,
        )

        self.block4 = ConvBlock(
            256,
            256,
            stride=1,
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Linear(
            256,
            num_classes,
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        return self.classifier(x)


# ======================================================================
# INPUT GENERATION
# ======================================================================

def load_real_image(
    image_path: str,
    image_size: int,
) -> torch.Tensor:

    from PIL import Image

    path = Path(image_path)

    if not path.exists():

        raise FileNotFoundError(path)

    image = Image.open(path).convert("RGB")

    image = image.resize(
        (image_size, image_size)
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

    return tensor.unsqueeze(0)


def create_batch(
    batch_size: int,
    image_size: int,
    base_image: torch.Tensor | None,
) -> torch.Tensor:

    if batch_size <= 0:

        raise ValueError(
            "Batch size must be positive."
        )

    if image_size <= 0:

        raise ValueError(
            "Image size must be positive."
        )

    if base_image is None:

        # Numerical input for benchmarking.
        # This is not a synthetic photograph.

        return torch.rand(
            batch_size,
            3,
            image_size,
            image_size,
            dtype=torch.float32,
        )

    if base_image.shape != (
        1,
        3,
        image_size,
        image_size,
    ):

        raise ValueError(
            "base_image must have shape "
            f"(1, 3, {image_size}, {image_size})."
        )

    return base_image.repeat(
        batch_size,
        1,
        1,
        1,
    )


# ======================================================================
# PREPROCESSING
# ======================================================================

def preprocess(
    x: torch.Tensor,
) -> torch.Tensor:

    return x * 2.0 - 1.0


# ======================================================================
# LATENCY STATISTICS
# ======================================================================

def calculate_statistics(
    values: list[float],
) -> dict[str, float]:

    if not values:

        raise ValueError(
            "Cannot calculate statistics from empty values."
        )

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {

        "mean_ms": float(np.mean(array)),

        "median_ms": float(np.median(array)),

        "std_ms": float(np.std(array)),

        "min_ms": float(np.min(array)),

        "max_ms": float(np.max(array)),

        "p50_ms": float(np.percentile(array, 50)),

        "p95_ms": float(np.percentile(array, 95)),

        "p99_ms": float(np.percentile(array, 99)),

    }


# ======================================================================
# MEMORY MEASUREMENT
# ======================================================================

def get_gpu_memory(
    device: torch.device,
) -> dict[str, float | None]:

    if device.type != "cuda":

        return {

            "allocated_mb": None,

            "reserved_mb": None,

            "peak_allocated_mb": None,

            "peak_reserved_mb": None,

        }

    synchronize(device)

    mb = 1024 ** 2

    return {

        "allocated_mb": (
            torch.cuda.memory_allocated(device) / mb
        ),

        "reserved_mb": (
            torch.cuda.memory_reserved(device) / mb
        ),

        "peak_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / mb
        ),

        "peak_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / mb
        ),

    }


def get_process_memory_mb() -> float | None:

    try:

        import psutil

        process = psutil.Process()

        return (
            process.memory_info().rss
            / (1024 ** 2)
        )

    except ImportError:

        return None


# ======================================================================
# BENCHMARK RESULT
# ======================================================================

@dataclass
class BatchResult:

    device: str

    batch_size: int

    image_size: int

    iterations: int

    latencies_ms: list[float]

    total_elapsed_seconds: float

    throughput_fps: float

    latency_per_image_ms: float

    memory: dict[str, float | None]

    successful: bool = True

    error: str | None = None

    @property
    def statistics(self):

        if not self.latencies_ms:

            return {}

        return calculate_statistics(
            self.latencies_ms
        )


# ======================================================================
# WARMUP
# ======================================================================

def warmup(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup_iterations: int,
) -> None:

    model.eval()

    with torch.inference_mode():

        for _ in range(warmup_iterations):

            model(x)

    synchronize(device)


# ======================================================================
# SINGLE BATCH BENCHMARK
# ======================================================================

def benchmark_batch_size(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
    base_image: torch.Tensor | None,
    warmup_iterations: int,
    iterations: int,
) -> BatchResult:

    """
    Benchmark one batch size.

    Every iteration processes one complete batch.

    Throughput:

        total_images / total_elapsed_seconds

    Per-image latency:

        mean_batch_latency / batch_size

    Memory is measured after model and input allocation and includes
    their effect on the PyTorch CUDA allocator.
    """

    x_cpu = create_batch(
        batch_size,
        image_size,
        base_image,
    )

    x = x_cpu.to(device)

    x = preprocess(x)

    model.eval()

    # Warmup.
    warmup(
        model,
        x,
        device,
        warmup_iterations,
    )

    if device.type == "cuda":

        torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)

    latencies = []

    synchronize(device)

    total_start = time.perf_counter()

    try:

        with torch.inference_mode():

            for _ in range(iterations):

                synchronize(device)

                start = time.perf_counter()

                output = model(x)

                synchronize(device)

                end = time.perf_counter()

                latencies.append(
                    (end - start) * 1000.0
                )

                if output.shape[0] != batch_size:

                    raise RuntimeError(
                        "Unexpected output batch dimension."
                    )

                del output

        total_end = time.perf_counter()

        total_elapsed = (
            total_end - total_start
        )

        total_images = (
            batch_size * iterations
        )

        throughput = (
            total_images / total_elapsed
        )

        mean_batch_latency = float(
            np.mean(latencies)
        )

        latency_per_image = (
            mean_batch_latency / batch_size
        )

        memory = get_gpu_memory(device)

        return BatchResult(

            device=str(device),

            batch_size=batch_size,

            image_size=image_size,

            iterations=iterations,

            latencies_ms=latencies,

            total_elapsed_seconds=total_elapsed,

            throughput_fps=throughput,

            latency_per_image_ms=latency_per_image,

            memory=memory,

        )

    except RuntimeError as error:

        # Handle CUDA out-of-memory gracefully.
        # Other runtime errors are also recorded so that the
        # experiment can report the failed batch size.

        if device.type == "cuda":

            torch.cuda.empty_cache()

        return BatchResult(

            device=str(device),

            batch_size=batch_size,

            image_size=image_size,

            iterations=iterations,

            latencies_ms=latencies,

            total_elapsed_seconds=0.0,

            throughput_fps=0.0,

            latency_per_image_ms=0.0,

            memory=get_gpu_memory(device),

            successful=False,

            error=str(error),

        )

    finally:

        del x
        del x_cpu


# ======================================================================
# BATCH SIZE EXPERIMENT
# ======================================================================

def run_batch_experiment(
    model: nn.Module,
    device: torch.device,
    batch_sizes: list[int],
    image_size: int,
    base_image: torch.Tensor | None,
    warmup_iterations: int,
    iterations: int,
) -> list[BatchResult]:

    results = []

    print("\n" + "=" * 75)

    print(
        f"BATCH SIZE EXPERIMENT — DEVICE: {device}"
    )

    print("=" * 75)

    for batch_size in batch_sizes:

        print(
            f"\nTesting batch size {batch_size}..."
        )

        result = benchmark_batch_size(
            model=model,
            device=device,
            batch_size=batch_size,
            image_size=image_size,
            base_image=base_image,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        )

        results.append(result)

        if not result.successful:

            print(
                f"  FAILED: {result.error}"
            )

            continue

        stats = result.statistics

        print(
            f"  Batch latency: "
            f"{stats['mean_ms']:.3f} ms"
        )

        print(
            f"  P95 latency: "
            f"{stats['p95_ms']:.3f} ms"
        )

        print(
            f"  Per-image latency: "
            f"{result.latency_per_image_ms:.3f} ms"
        )

        print(
            f"  Throughput: "
            f"{result.throughput_fps:.2f} images/s"
        )

        if result.memory["peak_allocated_mb"] is not None:

            print(
                f"  Peak GPU allocated memory: "
                f"{result.memory['peak_allocated_mb']:.2f} MB"
            )

    return results


# ======================================================================
# OPTIMAL BATCH SELECTION
# ======================================================================

def select_optimal_batch(
    results: list[BatchResult],
    latency_budget_ms: float,
    metric: str = "throughput",
) -> BatchResult | None:

    """
    Select the highest-throughput successful batch whose P95 batch
    latency does not exceed the latency budget.

    This is a batch-level latency constraint.

    It is not a per-camera frame deadline guarantee.
    """

    candidates = []

    for result in results:

        if not result.successful:

            continue

        stats = result.statistics

        if not stats:

            continue

        if stats["p95_ms"] > latency_budget_ms:

            continue

        candidates.append(result)

    if not candidates:

        return None

    if metric == "throughput":

        return max(
            candidates,
            key=lambda result: result.throughput_fps,
        )

    if metric == "latency":

        return min(
            candidates,
            key=lambda result: result.statistics["p95_ms"],
        )

    raise ValueError(
        f"Unknown selection metric: {metric}"
    )


# ======================================================================
# RESULTS TABLE
# ======================================================================

def print_results_table(
    results: list[BatchResult],
) -> None:

    print("\n" + "=" * 105)

    print("BATCH SIZE COMPARISON")

    print("=" * 105)

    header = (
        f"{'Batch':>7}"
        f"{'Mean ms':>12}"
        f"{'P95 ms':>12}"
        f"{'Per-image ms':>16}"
        f"{'Throughput':>15}"
        f"{'Peak VRAM MB':>17}"
        f"{'Status':>12}"
    )

    print(header)

    print("-" * 105)

    for result in results:

        if not result.successful:

            print(
                f"{result.batch_size:>7}"
                f"{'-':>12}"
                f"{'-':>12}"
                f"{'-':>16}"
                f"{'-':>15}"
                f"{'-':>17}"
                f"{'FAILED':>12}"
            )

            continue

        stats = result.statistics

        peak_memory = result.memory[
            "peak_allocated_mb"
        ]

        memory_text = (
            f"{peak_memory:.2f}"
            if peak_memory is not None
            else "N/A"
        )

        print(
            f"{result.batch_size:>7}"
            f"{stats['mean_ms']:>12.3f}"
            f"{stats['p95_ms']:>12.3f}"
            f"{result.latency_per_image_ms:>16.3f}"
            f"{result.throughput_fps:>15.2f}"
            f"{memory_text:>17}"
            f"{'OK':>12}"
        )

    print("=" * 105)


# ======================================================================
# VISUALIZATION
# ======================================================================

def plot_batch_latency(
    results: list[BatchResult],
    output_dir: Path,
) -> None:

    valid = [
        result
        for result in results
        if result.successful
    ]

    if not valid:
        return

    batch_sizes = [
        result.batch_size
        for result in valid
    ]

    mean_latency = [
        result.statistics["mean_ms"]
        for result in valid
    ]

    p95_latency = [
        result.statistics["p95_ms"]
        for result in valid
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        batch_sizes,
        mean_latency,
        marker="o",
        label="Mean batch latency",
    )

    ax.plot(
        batch_sizes,
        p95_latency,
        marker="s",
        label="P95 batch latency",
    )

    ax.set_xlabel("Batch size")

    ax.set_ylabel("Latency (ms)")

    ax.set_title(
        "Batch Size vs Batch Latency"
    )

    ax.grid(alpha=0.3)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "batch_latency.png",
        dpi=150,
    )

    plt.close(fig)


def plot_throughput(
    results: list[BatchResult],
    output_dir: Path,
) -> None:

    valid = [
        result
        for result in results
        if result.successful
    ]

    if not valid:
        return

    batch_sizes = [
        result.batch_size
        for result in valid
    ]

    throughput = [
        result.throughput_fps
        for result in valid
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        batch_sizes,
        throughput,
        marker="o",
    )

    ax.set_xlabel("Batch size")

    ax.set_ylabel("Throughput (images/second)")

    ax.set_title(
        "Batch Size vs Throughput"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "batch_throughput.png",
        dpi=150,
    )

    plt.close(fig)


def plot_per_image_latency(
    results: list[BatchResult],
    output_dir: Path,
) -> None:

    valid = [
        result
        for result in results
        if result.successful
    ]

    if not valid:
        return

    batch_sizes = [
        result.batch_size
        for result in valid
    ]

    latency = [
        result.latency_per_image_ms
        for result in valid
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        batch_sizes,
        latency,
        marker="o",
    )

    ax.set_xlabel("Batch size")

    ax.set_ylabel("Average computational latency per image (ms)")

    ax.set_title(
        "Batch Size vs Average Per-Image Latency"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "per_image_latency.png",
        dpi=150,
    )

    plt.close(fig)


def plot_memory(
    results: list[BatchResult],
    output_dir: Path,
) -> None:

    valid = [
        result
        for result in results
        if (
            result.successful
            and result.memory["peak_allocated_mb"] is not None
        )
    ]

    if not valid:
        return

    batch_sizes = [
        result.batch_size
        for result in valid
    ]

    memory = [
        result.memory["peak_allocated_mb"]
        for result in valid
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        batch_sizes,
        memory,
        marker="o",
    )

    ax.set_xlabel("Batch size")

    ax.set_ylabel("Peak allocated GPU memory (MB)")

    ax.set_title(
        "Batch Size vs GPU Memory"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "batch_memory.png",
        dpi=150,
    )

    plt.close(fig)


def plot_multiple_devices(
    device_results: dict[str, list[BatchResult]],
    output_dir: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    plotted = False

    for device_name, results in device_results.items():

        valid = [
            result
            for result in results
            if result.successful
        ]

        if not valid:
            continue

        ax.plot(
            [
                result.batch_size
                for result in valid
            ],
            [
                result.throughput_fps
                for result in valid
            ],
            marker="o",
            label=device_name,
        )

        plotted = True

    if not plotted:

        plt.close(fig)

        return

    ax.set_xlabel("Batch size")

    ax.set_ylabel("Throughput (images/second)")

    ax.set_title(
        "CPU vs GPU Batch Throughput"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "cpu_gpu_batch_comparison.png",
        dpi=150,
    )

    plt.close(fig)


# ======================================================================
# JSON REPORT
# ======================================================================

def serialize_results(
    results: list[BatchResult],
) -> list[dict[str, Any]]:

    serialized = []

    for result in results:

        row = {

            "device": result.device,

            "batch_size": result.batch_size,

            "image_size": result.image_size,

            "successful": result.successful,

            "error": result.error,

            "statistics": result.statistics,

            "throughput_fps": result.throughput_fps,

            "latency_per_image_ms": (
                result.latency_per_image_ms
            ),

            "memory": result.memory,

        }

        serialized.append(row)

    return serialized


# ======================================================================
# NUMERICAL TESTS
# ======================================================================

def run_tests() -> None:

    print("\nRunning Lesson 72 tests...\n")

    # Test 1: Model output shape.

    model = BatchBenchmarkCNN()

    x = torch.randn(
        3,
        3,
        64,
        64,
    )

    with torch.inference_mode():

        output = model(x)

    assert output.shape == (3, 10)

    print("Test 1 passed: CNN output shape.")

    # Test 2: Batch size 1.

    x1 = torch.randn(
        1,
        3,
        64,
        64,
    )

    with torch.inference_mode():

        output1 = model(x1)

    assert output1.shape == (1, 10)

    print("Test 2 passed: Batch size 1.")

    # Test 3: Batch size 8.

    x8 = torch.randn(
        8,
        3,
        64,
        64,
    )

    with torch.inference_mode():

        output8 = model(x8)

    assert output8.shape == (8, 10)

    print("Test 3 passed: Batch size 8.")

    # Test 4: Preprocessing.

    values = torch.tensor(
        [0.0, 0.5, 1.0]
    )

    normalized = preprocess(values)

    expected = torch.tensor(
        [-1.0, 0.0, 1.0]
    )

    assert torch.allclose(
        normalized,
        expected,
        atol=1e-6,
    )

    print("Test 4 passed: Preprocessing.")

    # Test 5: Batch creation.

    batch = create_batch(
        batch_size=4,
        image_size=32,
        base_image=None,
    )

    assert batch.shape == (
        4,
        3,
        32,
        32,
    )

    print("Test 5 passed: Batch creation.")

    # Test 6: Real-image batch replication.

    base_image = torch.rand(
        1,
        3,
        32,
        32,
    )

    replicated = create_batch(
        batch_size=5,
        image_size=32,
        base_image=base_image,
    )

    assert replicated.shape == (
        5,
        3,
        32,
        32,
    )

    for i in range(5):

        assert torch.equal(
            replicated[i],
            base_image[0],
        )

    print("Test 6 passed: Image batch replication.")

    # Test 7: Latency statistics.

    latencies = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]

    stats = calculate_statistics(
        latencies
    )

    assert np.isclose(
        stats["mean_ms"],
        3.0,
    )

    assert np.isclose(
        stats["median_ms"],
        3.0,
    )

    print("Test 7 passed: Latency statistics.")

    # Test 8: Throughput formula.

    batch_size = 8

    elapsed = 0.04

    throughput = batch_size / elapsed

    assert np.isclose(
        throughput,
        200.0,
    )

    print("Test 8 passed: Throughput calculation.")

    # Test 9: Per-image latency formula.

    batch_latency = 40.0

    per_image = batch_latency / 8

    assert np.isclose(
        per_image,
        5.0,
    )

    print("Test 9 passed: Per-image latency calculation.")

    # Test 10: Batch selection.

    test_results = [

        BatchResult(
            device="cpu",
            batch_size=1,
            image_size=64,
            iterations=10,
            latencies_ms=[5.0] * 10,
            total_elapsed_seconds=0.05,
            throughput_fps=200.0,
            latency_per_image_ms=5.0,
            memory={
                "allocated_mb": None,
                "reserved_mb": None,
                "peak_allocated_mb": None,
                "peak_reserved_mb": None,
            },
        ),

        BatchResult(
            device="cpu",
            batch_size=4,
            image_size=64,
            iterations=10,
            latencies_ms=[12.0] * 10,
            total_elapsed_seconds=0.12,
            throughput_fps=333.0,
            latency_per_image_ms=3.0,
            memory={
                "allocated_mb": None,
                "reserved_mb": None,
                "peak_allocated_mb": None,
                "peak_reserved_mb": None,
            },
        ),

        BatchResult(
            device="cpu",
            batch_size=8,
            image_size=64,
            iterations=10,
            latencies_ms=[60.0] * 10,
            total_elapsed_seconds=0.6,
            throughput_fps=133.0,
            latency_per_image_ms=7.5,
            memory={
                "allocated_mb": None,
                "reserved_mb": None,
                "peak_allocated_mb": None,
                "peak_reserved_mb": None,
            },
        ),

    ]

    selected = select_optimal_batch(
        test_results,
        latency_budget_ms=20.0,
    )

    assert selected is not None

    assert selected.batch_size == 4

    print("Test 10 passed: Latency-constrained batch selection.")

    # Test 11: Failed batches are excluded.

    failed = BatchResult(
        device="cuda",
        batch_size=32,
        image_size=64,
        iterations=10,
        latencies_ms=[],
        total_elapsed_seconds=0.0,
        throughput_fps=0.0,
        latency_per_image_ms=0.0,
        memory={
            "allocated_mb": None,
            "reserved_mb": None,
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
        },
        successful=False,
        error="Simulated out-of-memory error.",
    )

    selected = select_optimal_batch(
        test_results + [failed],
        latency_budget_ms=20.0,
    )

    assert selected is not None

    assert selected.batch_size == 4

    print("Test 11 passed: Failed batch exclusion.")

    # Test 12: Batch-size validation.

    try:

        create_batch(
            batch_size=0,
            image_size=32,
            base_image=None,
        )

        raise AssertionError(
            "Invalid batch size was not rejected."
        )

    except ValueError:

        pass

    print("Test 12 passed: Invalid batch size rejection.")

    # Test 13: Device memory reporting.

    memory = get_gpu_memory(
        torch.device("cpu")
    )

    assert memory["allocated_mb"] is None

    print("Test 13 passed: CPU memory reporting.")

    # Test 14: Actual CPU benchmark.

    small_model = nn.Linear(
        8,
        4,
    )

    result = benchmark_batch_size(
        model=small_model,
        device=torch.device("cpu"),
        batch_size=2,
        image_size=8,
        base_image=None,
        warmup_iterations=2,
        iterations=5,
    )

    assert result.successful

    assert len(result.latencies_ms) == 5

    assert result.throughput_fps > 0

    print("Test 14 passed: Actual CPU batch benchmark.")

    print("\nAll 14 tests passed successfully.")


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Lesson 72 — Batch Size Optimization"
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
    )

    parser.add_argument(
        "--latency-budget-ms",
        type=float,
        default=DEFAULT_LATENCY_BUDGET_MS,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    args = parser.parse_args()

    if args.check_only:

        run_tests()

        return

    if args.image_size <= 0:

        raise ValueError(
            "--image-size must be positive."
        )

    if args.iterations <= 0:

        raise ValueError(
            "--iterations must be positive."
        )

    if args.warmup < 0:

        raise ValueError(
            "--warmup cannot be negative."
        )

    if args.latency_budget_ms <= 0:

        raise ValueError(
            "--latency-budget-ms must be positive."
        )

    if not args.batch_sizes:

        raise ValueError(
            "At least one batch size is required."
        )

    if any(
        size <= 0
        for size in args.batch_sizes
    ):

        raise ValueError(
            "All batch sizes must be positive."
        )

    # Remove duplicate batch sizes and sort.

    batch_sizes = sorted(
        set(args.batch_sizes)
    )

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

    print("LESSON 72 — BATCH SIZE OPTIMIZATION")

    print("=" * 75)

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"Python: {sys.version.split()[0]}"
    )

    print(
        f"Platform: {platform.platform()}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Image size: {args.image_size}"
    )

    print(
        f"Batch sizes: {batch_sizes}"
    )

    print(
        f"Latency budget: {args.latency_budget_ms} ms"
    )

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(device)}"
        )

    # --------------------------------------------------------------
    # LOAD OPTIONAL REAL IMAGE
    # --------------------------------------------------------------

    base_image = None

    if args.image:

        base_image = load_real_image(
            args.image,
            args.image_size,
        )

        print(
            f"\nReal image loaded: {args.image}"
        )

    else:

        print(
            "\nNo image supplied. "
            "Using numerical benchmarking tensors."
        )

    # --------------------------------------------------------------
    # MODEL
    # --------------------------------------------------------------

    torch.manual_seed(42)

    np.random.seed(42)

    model = BatchBenchmarkCNN()

    model.eval()

    model = model.to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"\nModel parameters: {parameter_count:,}"
    )

    # --------------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------------

    results = run_batch_experiment(
        model=model,
        device=device,
        batch_sizes=batch_sizes,
        image_size=args.image_size,
        base_image=base_image,
        warmup_iterations=args.warmup,
        iterations=args.iterations,
    )

    print_results_table(
        results
    )

    # --------------------------------------------------------------
    # OPTIMAL BATCH SIZE
    # --------------------------------------------------------------

    selected = select_optimal_batch(
        results,
        latency_budget_ms=args.latency_budget_ms,
        metric="throughput",
    )

    print("\n" + "=" * 75)

    print("OPTIMAL BATCH SIZE ANALYSIS")

    print("=" * 75)

    if selected is None:

        print(
            "No successful batch satisfies the latency budget."
        )

        print(
            "Try a smaller image size or a larger latency budget."
        )

    else:

        print(
            f"Selected batch size: {selected.batch_size}"
        )

        print(
            f"Mean batch latency: "
            f"{selected.statistics['mean_ms']:.3f} ms"
        )

        print(
            f"P95 batch latency: "
            f"{selected.statistics['p95_ms']:.3f} ms"
        )

        print(
            f"Throughput: "
            f"{selected.throughput_fps:.2f} images/s"
        )

        print(
            f"Average computational latency per image: "
            f"{selected.latency_per_image_ms:.3f} ms"
        )

        print(
            f"Latency budget: "
            f"{args.latency_budget_ms:.3f} ms"
        )

    # --------------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------------

    print("\nGenerating visualizations...")

    plot_batch_latency(
        results,
        output_dir,
    )

    plot_throughput(
        results,
        output_dir,
    )

    plot_per_image_latency(
        results,
        output_dir,
    )

    plot_memory(
        results,
        output_dir,
    )

    # --------------------------------------------------------------
    # JSON REPORT
    # --------------------------------------------------------------

    report = {

        "lesson": 72,

        "title": "Batch Size Optimization",

        "device": str(device),

        "image_size": args.image_size,

        "batch_sizes": batch_sizes,

        "warmup_iterations": args.warmup,

        "benchmark_iterations": args.iterations,

        "latency_budget_ms": args.latency_budget_ms,

        "selected_batch_size": (
            selected.batch_size
            if selected is not None
            else None
        ),

        "results": serialize_results(
            results
        ),

    }

    report_path = (
        output_dir / "batch_optimization_report.json"
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
    # FINISH
    # --------------------------------------------------------------

    print("\n" + "=" * 75)

    print("LESSON 72 COMPLETED")

    print("=" * 75)

    print("\nGenerated files:")

    for path in sorted(
        output_dir.iterdir()
    ):

        print(
            f"  {path}"
        )

    print("\nKey takeaways:")

    print(
        "1. Larger batches can improve throughput."
    )

    print(
        "2. Larger batches usually consume more memory."
    )

    print(
        "3. Batch latency and per-image latency are different."
    )

    print(
        "4. Maximum throughput is not necessarily minimum latency."
    )

    print(
        "5. The optimal batch size depends on application constraints."
    )

    print(
        "\nNext: Lesson 73 — Mixed Precision FP16."
    )


if __name__ == "__main__":

    main()