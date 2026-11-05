"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 6 — MODEL OPTIMIZATION AND DEPLOYMENT

LESSON 70 — MODEL PROFILING
======================================================================

OBJECTIVE
---------
Understand how to measure computer vision model performance before
attempting optimization.

We will measure:

    1. Model parameter count.
    2. Approximate FLOPs.
    3. Preprocessing latency.
    4. Neural network inference latency.
    5. Postprocessing latency.
    6. End-to-end pipeline latency.
    7. Average, median, P95, and P99 latency.
    8. Throughput (FPS).
    9. CPU memory usage.
   10. GPU memory usage.
   11. Batch-size performance.

IMPORTANT
---------
The CNN in this lesson has randomly initialized weights.

This lesson measures computational performance, NOT classification
accuracy.

A randomly initialized model is useful for benchmarking the
computational cost of a known architecture without downloading
pretrained weights.

REAL IMAGE SUPPORT
------------------
You can supply a real image:

    python lesson70_model_profiling.py --image image.jpg

The image is loaded and resized before inference.

Without --image, the script generates numerical input tensors.
These tensors are benchmarking workloads, not real photographs.

CORE CONCEPTS
-------------

1. LATENCY

    Latency = time required to process one inference request.

    Example:

        20 ms per image

    The theoretical maximum sequential throughput is:

        FPS = 1000 / 20 = 50 FPS

    This simple relationship assumes one image per request and
    ignores pipeline overlap and batching.

2. THROUGHPUT

    Throughput = number of images processed / elapsed time.

    Example:

        1000 images / 10 seconds = 100 images/second

3. PERCENTILE LATENCY

    Average latency does not describe occasional slow requests.

    P50:
        Median latency.

    P95:
        95% of requests complete within this time.

    P99:
        99% of requests complete within this time.

4. WARMUP

    Initial inference can be slower because of:

        - CUDA initialization.
        - Memory allocation.
        - Kernel selection.
        - CPU cache effects.
        - Library initialization.

    We should warm up before measuring.

5. GPU SYNCHRONIZATION

    CUDA operations are asynchronous.

    A Python call can return before the GPU finishes its work.

    Therefore, accurate CPU wall-clock GPU measurements require:

        torch.cuda.synchronize()

    CUDA events are another useful way to measure GPU execution time.

6. FLOPs

    FLOPs estimate the number of floating-point operations.

    For convolution:

        FLOPs approximately equal:

        2 * Hout * Wout * Cout *
            (Cin / groups) * Kh * Kw

    This counts one multiplication and one addition as two operations.

    The estimate does not include every possible operation,
    such as activations, normalization, or data movement.

7. MEMORY

    GPU memory includes:

        - Model parameters.
        - Intermediate feature maps.
        - Temporary tensors.
        - CUDA runtime allocations.
        - Other processes using the GPU.

    PyTorch reports allocated and reserved CUDA memory separately.

8. BOTTLENECK ANALYSIS

    If preprocessing dominates:
        Optimize image loading, resizing, and transfers.

    If inference dominates:
        Consider FP16, quantization, pruning, or TensorRT.

    If postprocessing dominates:
        Optimize NMS, decoding, or CPU/GPU transfers.

    If memory dominates:
        Reduce batch size or intermediate activation storage.

WHAT YOU WILL LEARN
-------------------

After this lesson, you should be able to:

    - Build a reproducible profiling experiment.
    - Measure CPU and GPU inference correctly.
    - Interpret latency distributions.
    - Compare throughput across batch sizes.
    - Identify the computational bottleneck.
    - Understand why optimization must begin with measurement.

REAL-WORLD ROBOTICS EXAMPLE
---------------------------

Imagine an OpenArm V2 waste-sorting robot.

Its perception pipeline is:

    Camera
       |
       v
    Image preprocessing
       |
       v
    Object detection model
       |
       v
    Object localization
       |
       v
    Grasp planning
       |
       v
    ACT policy
       |
       v
    Robot control

If the complete pipeline takes 150 ms per observation, the robot
cannot react as quickly as a system that takes 30 ms.

Profiling helps determine which component is responsible.

A model with excellent accuracy is not necessarily suitable for
real-time robotics if its inference latency is too high.

======================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 1

DEFAULT_WARMUP = 20
DEFAULT_ITERATIONS = 100

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16]

DEFAULT_OUTPUT_DIR = Path("runs/lesson70")


# ======================================================================
# UTILITIES
# ======================================================================

def synchronize(device: torch.device) -> None:
    """
    Synchronize CUDA execution before measuring elapsed wall time.

    CPU execution is synchronous and requires no action.
    """

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    """
    Calculate a percentile.

    q is expressed as a percentage, e.g. 95 for P95.
    """

    if not values:
        return 0.0

    return float(np.percentile(np.asarray(values), q))


def calculate_latency_statistics(
    latencies_ms: list[float],
) -> dict[str, float]:

    if not latencies_ms:
        raise ValueError("Latency list cannot be empty.")

    values = np.asarray(latencies_ms, dtype=np.float64)

    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "std_ms": float(np.std(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "p50_ms": percentile(latencies_ms, 50),
        "p90_ms": percentile(latencies_ms, 90),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
    }


# ======================================================================
# CNN MODEL
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

        self.bn = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        return x


class ProfilingCNN(nn.Module):
    """
    Small CNN designed for profiling.

    Input:
        B x 3 x 224 x 224

    Output:
        B x 10

    This is a computational benchmark architecture.

    It is not intended to achieve meaningful classification accuracy.
    """

    def __init__(self, num_classes: int = 10):

        super().__init__()

        self.stem = ConvBlock(3, 32, stride=2)

        self.block1 = ConvBlock(32, 64, stride=2)

        self.block2 = ConvBlock(64, 128, stride=2)

        self.block3 = ConvBlock(128, 256, stride=2)

        self.block4 = ConvBlock(256, 256, stride=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.stem(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)

        return x


# ======================================================================
# MODEL INFORMATION
# ======================================================================

def count_parameters(model: nn.Module) -> dict[str, int]:

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
    }


def estimate_model_size_mb(model: nn.Module) -> float:

    total_bytes = 0

    for parameter in model.parameters():
        total_bytes += parameter.numel() * parameter.element_size()

    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()

    return total_bytes / (1024 ** 2)


# ======================================================================
# FLOPs ESTIMATION
# ======================================================================

class FLOPsCounter:
    """
    Estimate convolution and linear layer FLOPs.

    Counts multiplication and addition as two operations.

    This is an approximation, not a complete hardware instruction count.
    """

    def __init__(self, model: nn.Module):

        self.model = model

        self.total_flops = 0

        self.handles = []

    def _conv_hook(self, module, inputs, output):

        batch_size = output.shape[0]

        out_channels = output.shape[1]

        out_h = output.shape[2]

        out_w = output.shape[3]

        kernel_h, kernel_w = module.kernel_size

        input_channels_per_group = (
            module.in_channels // module.groups
        )

        operations_per_output = (
            input_channels_per_group
            * kernel_h
            * kernel_w
            * 2
        )

        flops = (
            batch_size
            * out_channels
            * out_h
            * out_w
            * operations_per_output
        )

        self.total_flops += int(flops)

    def _linear_hook(self, module, inputs, output):

        batch_size = output.shape[0]

        flops = (
            batch_size
            * module.in_features
            * module.out_features
            * 2
        )

        self.total_flops += int(flops)

    def profile(self, input_tensor: torch.Tensor) -> int:

        self.total_flops = 0

        self.handles = []

        for module in self.model.modules():

            if isinstance(module, nn.Conv2d):

                handle = module.register_forward_hook(
                    self._conv_hook
                )

                self.handles.append(handle)

            elif isinstance(module, nn.Linear):

                handle = module.register_forward_hook(
                    self._linear_hook
                )

                self.handles.append(handle)

        was_training = self.model.training

        self.model.eval()

        with torch.inference_mode():

            self.model(input_tensor)

        for handle in self.handles:
            handle.remove()

        self.handles = []

        self.model.train(was_training)

        return int(self.total_flops)


# ======================================================================
# IMAGE INPUT
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


def create_benchmark_input(
    batch_size: int,
    image_size: int,
    device: torch.device,
    base_image: torch.Tensor | None = None,
) -> torch.Tensor:

    if base_image is None:

        # Numerical input for computational benchmarking.
        # This is not a synthetic photograph.

        tensor = torch.rand(
            batch_size,
            3,
            image_size,
            image_size,
            dtype=torch.float32,
        )

    else:

        if base_image.shape[0] != 1:
            raise ValueError(
                "base_image must have batch size 1."
            )

        tensor = base_image.repeat(
            batch_size,
            1,
            1,
            1,
        )

    return tensor.to(device)


# ======================================================================
# PREPROCESSING AND POSTPROCESSING
# ======================================================================

def preprocess(
    tensor: torch.Tensor,
) -> torch.Tensor:

    # Example image preprocessing:
    #   - convert input to float
    #   - normalize from [0, 1] to [-1, 1]

    tensor = tensor.float()

    tensor = tensor * 2.0 - 1.0

    return tensor


def postprocess(
    logits: torch.Tensor,
) -> torch.Tensor:

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    predictions = torch.argmax(
        probabilities,
        dim=1,
    )

    return predictions


# ======================================================================
# BENCHMARKING
# ======================================================================

@dataclass
class BenchmarkResult:

    name: str

    batch_size: int

    iterations: int

    latencies_ms: list[float]

    total_images: int

    total_time_s: float

    @property
    def statistics(self):

        return calculate_latency_statistics(
            self.latencies_ms
        )

    @property
    def images_per_second(self):

        if self.total_time_s <= 0:
            return 0.0

        return self.total_images / self.total_time_s


def benchmark_callable(
    name: str,
    function: Callable,
    device: torch.device,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:

    if warmup < 0:
        raise ValueError("warmup cannot be negative.")

    if iterations <= 0:
        raise ValueError("iterations must be positive.")

    # Warmup.
    for _ in range(warmup):

        function()

    synchronize(device)

    latencies = []

    synchronize(device)

    total_start = time.perf_counter()

    for _ in range(iterations):

        synchronize(device)

        start = time.perf_counter()

        function()

        synchronize(device)

        end = time.perf_counter()

        latencies.append(
            (end - start) * 1000.0
        )

    total_end = time.perf_counter()

    total_time = total_end - total_start

    return BenchmarkResult(
        name=name,
        batch_size=batch_size,
        iterations=iterations,
        latencies_ms=latencies,
        total_images=batch_size * iterations,
        total_time_s=total_time,
    )


# ======================================================================
# CUDA MEMORY
# ======================================================================

def get_cuda_memory(
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

    allocated = torch.cuda.memory_allocated(device)

    reserved = torch.cuda.memory_reserved(device)

    peak_allocated = torch.cuda.max_memory_allocated(device)

    peak_reserved = torch.cuda.max_memory_reserved(device)

    mb = 1024 ** 2

    return {
        "allocated_mb": allocated / mb,
        "reserved_mb": reserved / mb,
        "peak_allocated_mb": peak_allocated / mb,
        "peak_reserved_mb": peak_reserved / mb,
    }


# ======================================================================
# CPU MEMORY
# ======================================================================

def get_process_memory_mb() -> float | None:

    """
    Get current process resident memory when psutil is installed.

    This is process-level memory, not model-only memory.
    """

    try:

        import psutil

        process = psutil.Process(os.getpid())

        return process.memory_info().rss / (1024 ** 2)

    except ImportError:

        return None


# ======================================================================
# STAGE PROFILING
# ======================================================================

def profile_pipeline_stages(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, BenchmarkResult]:

    """
    Measure pipeline stages independently.

    Stage 1:
        Preprocessing.

    Stage 2:
        Model inference.

    Stage 3:
        Postprocessing.

    Stage 4:
        Complete pipeline.

    Important:
        Independent stage measurements do not necessarily sum to
        end-to-end latency because they have separate measurement
        boundaries and synchronization overhead.
    """

    prepared = preprocess(input_tensor)

    with torch.inference_mode():
        logits = model(prepared)

    # Stage 1: preprocessing.

    def run_preprocessing():

        return preprocess(input_tensor)

    preprocessing_result = benchmark_callable(
        "Preprocessing",
        run_preprocessing,
        device,
        input_tensor.shape[0],
        warmup,
        iterations,
    )

    # Stage 2: inference.

    def run_inference():

        with torch.inference_mode():

            return model(prepared)

    inference_result = benchmark_callable(
        "Model inference",
        run_inference,
        device,
        input_tensor.shape[0],
        warmup,
        iterations,
    )

    # Stage 3: postprocessing.

    def run_postprocessing():

        return postprocess(logits)

    postprocessing_result = benchmark_callable(
        "Postprocessing",
        run_postprocessing,
        device,
        input_tensor.shape[0],
        warmup,
        iterations,
    )

    # Stage 4: complete pipeline.

    def run_complete_pipeline():

        processed = preprocess(input_tensor)

        with torch.inference_mode():

            output = model(processed)

            prediction = postprocess(output)

        return prediction

    full_pipeline_result = benchmark_callable(
        "Complete pipeline",
        run_complete_pipeline,
        device,
        input_tensor.shape[0],
        warmup,
        iterations,
    )

    return {
        "preprocessing": preprocessing_result,
        "inference": inference_result,
        "postprocessing": postprocessing_result,
        "full_pipeline": full_pipeline_result,
    }


# ======================================================================
# BATCH SIZE PROFILING
# ======================================================================

def profile_batch_sizes(
    model: nn.Module,
    device: torch.device,
    image_size: int,
    batch_sizes: list[int],
    warmup: int,
    iterations: int,
    base_image: torch.Tensor | None = None,
) -> list[dict[str, Any]]:

    results = []

    for batch_size in batch_sizes:

        print(
            f"\nProfiling batch size: {batch_size}"
        )

        input_tensor = create_benchmark_input(
            batch_size=batch_size,
            image_size=image_size,
            device=device,
            base_image=base_image,
        )

        def run_model():

            with torch.inference_mode():

                return model(input_tensor)

        result = benchmark_callable(
            name=f"Batch {batch_size}",
            function=run_model,
            device=device,
            batch_size=batch_size,
            warmup=warmup,
            iterations=iterations,
        )

        stats = result.statistics

        row = {
            "batch_size": batch_size,
            "mean_latency_ms": stats["mean_ms"],
            "median_latency_ms": stats["median_ms"],
            "p95_latency_ms": stats["p95_ms"],
            "p99_latency_ms": stats["p99_ms"],
            "throughput_fps": result.images_per_second,
            "latency_per_image_ms": (
                stats["mean_ms"] / batch_size
            ),
        }

        results.append(row)

        print(
            f"  Mean batch latency: "
            f"{stats['mean_ms']:.3f} ms"
        )

        print(
            f"  P95 latency: "
            f"{stats['p95_ms']:.3f} ms"
        )

        print(
            f"  Throughput: "
            f"{result.images_per_second:.2f} images/s"
        )

    return results


# ======================================================================
# PYTORCH PROFILER
# ======================================================================

def run_torch_profiler(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    output_dir: Path,
) -> str:

    """
    Use PyTorch Profiler to inspect operator-level execution.

    This is optional because detailed profiling adds overhead.

    The trace can be opened with Perfetto or compatible trace viewers.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    activities = [
        torch.profiler.ProfilerActivity.CPU
    ]

    if device.type == "cuda":

        activities.append(
            torch.profiler.ProfilerActivity.CUDA
        )

    trace_path = output_dir / "torch_profiler_trace.json"

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:

        for _ in range(5):

            with torch.inference_mode():

                model(input_tensor)

            synchronize(device)

            profiler.step()

    profiler.export_chrome_trace(
        str(trace_path)
    )

    print("\nTop PyTorch profiler operators:")

    print(
        profiler.key_averages().table(
            sort_by=(
                "cuda_time_total"
                if device.type == "cuda"
                else "cpu_time_total"
            ),
            row_limit=15,
        )
    )

    return str(trace_path)


# ======================================================================
# VISUALIZATION
# ======================================================================

def plot_stage_latency(
    results: dict[str, BenchmarkResult],
    output_dir: Path,
) -> None:

    names = []

    means = []

    p95s = []

    for result in results.values():

        names.append(result.name)

        means.append(
            result.statistics["mean_ms"]
        )

        p95s.append(
            result.statistics["p95_ms"]
        )

    x = np.arange(len(names))

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        x - width / 2,
        means,
        width,
        label="Mean latency",
    )

    ax.bar(
        x + width / 2,
        p95s,
        width,
        label="P95 latency",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        names,
        rotation=15,
        ha="right",
    )

    ax.set_ylabel("Latency (ms)")

    ax.set_title(
        "Lesson 70 — Pipeline Stage Latency"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "stage_latency.png",
        dpi=150,
    )

    plt.close(fig)


def plot_latency_distribution(
    results: dict[str, BenchmarkResult],
    output_dir: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    values = [
        result.latencies_ms
        for result in results.values()
    ]

    labels = [
        result.name
        for result in results.values()
    ]

    ax.boxplot(
        values,
        tick_labels=labels,
        showfliers=True,
    )

    ax.set_ylabel("Latency (ms)")

    ax.set_title(
        "Latency Distribution"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.xticks(
        rotation=15,
        ha="right",
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "latency_distribution.png",
        dpi=150,
    )

    plt.close(fig)


def plot_batch_results(
    batch_results: list[dict[str, Any]],
    output_dir: Path,
) -> None:

    batch_sizes = [
        row["batch_size"]
        for row in batch_results
    ]

    throughput = [
        row["throughput_fps"]
        for row in batch_results
    ]

    latency = [
        row["mean_latency_ms"]
        for row in batch_results
    ]

    latency_per_image = [
        row["latency_per_image_ms"]
        for row in batch_results
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

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        batch_sizes,
        latency,
        marker="o",
        label="Batch latency",
    )

    ax.plot(
        batch_sizes,
        latency_per_image,
        marker="s",
        label="Latency per image",
    )

    ax.set_xlabel("Batch size")

    ax.set_ylabel("Latency (ms)")

    ax.set_title(
        "Batch Size vs Latency"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "batch_latency.png",
        dpi=150,
    )

    plt.close(fig)


# ======================================================================
# REPORT
# ======================================================================

def print_benchmark_report(
    results: dict[str, BenchmarkResult],
) -> None:

    print("\n" + "=" * 75)

    print("PIPELINE PROFILING REPORT")

    print("=" * 75)

    for result in results.values():

        stats = result.statistics

        print(f"\n{result.name}")

        print("-" * 50)

        print(
            f"Mean:       {stats['mean_ms']:.3f} ms"
        )

        print(
            f"Median:     {stats['median_ms']:.3f} ms"
        )

        print(
            f"Std:        {stats['std_ms']:.3f} ms"
        )

        print(
            f"Minimum:    {stats['min_ms']:.3f} ms"
        )

        print(
            f"Maximum:    {stats['max_ms']:.3f} ms"
        )

        print(
            f"P95:        {stats['p95_ms']:.3f} ms"
        )

        print(
            f"P99:        {stats['p99_ms']:.3f} ms"
        )

        print(
            f"Throughput: {result.images_per_second:.2f} images/s"
        )


# ======================================================================
# NUMERICAL TESTS
# ======================================================================

def run_tests() -> None:

    print("\nRunning Lesson 70 tests...\n")

    # Test 1: Model output shape.

    model = ProfilingCNN(num_classes=10)

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    output = model(x)

    assert output.shape == (2, 10)

    print("Test 1 passed: CNN output shape.")

    # Test 2: Parameter counting.

    parameters = count_parameters(model)

    assert parameters["total_parameters"] > 0

    assert (
        parameters["trainable_parameters"]
        <= parameters["total_parameters"]
    )

    print("Test 2 passed: Parameter counting.")

    # Test 3: Model size.

    size_mb = estimate_model_size_mb(model)

    assert size_mb > 0

    print("Test 3 passed: Model size estimation.")

    # Test 4: FLOPs.

    counter = FLOPsCounter(model)

    flops = counter.profile(
        torch.randn(1, 3, 224, 224)
    )

    assert flops > 0

    print("Test 4 passed: FLOPs estimation.")

    # Test 5: Latency statistics.

    sample_latencies = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]

    stats = calculate_latency_statistics(
        sample_latencies
    )

    assert math.isclose(
        stats["mean_ms"],
        3.0,
    )

    assert math.isclose(
        stats["median_ms"],
        3.0,
    )

    print("Test 5 passed: Latency statistics.")

    # Test 6: Percentile behavior.

    assert (
        stats["p95_ms"]
        >= stats["p50_ms"]
    )

    print("Test 6 passed: Percentile ordering.")

    # Test 7: Preprocessing.

    test_tensor = torch.tensor(
        [0.0, 0.5, 1.0]
    )

    processed = preprocess(test_tensor)

    expected = torch.tensor(
        [-1.0, 0.0, 1.0]
    )

    assert torch.allclose(
        processed,
        expected,
        atol=1e-6,
    )

    print("Test 7 passed: Preprocessing normalization.")

    # Test 8: Postprocessing.

    logits = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.8, 0.1, 0.2],
        ]
    )

    predictions = postprocess(logits)

    assert predictions.tolist() == [1, 0]

    print("Test 8 passed: Postprocessing.")

    # Test 9: Benchmark output.

    device = torch.device("cpu")

    small_model = nn.Linear(8, 4)

    small_input = torch.randn(1, 8)

    result = benchmark_callable(
        name="test",
        function=lambda: small_model(small_input),
        device=device,
        batch_size=1,
        warmup=2,
        iterations=5,
    )

    assert len(result.latencies_ms) == 5

    assert result.statistics["mean_ms"] >= 0

    assert result.images_per_second > 0

    print("Test 9 passed: Benchmark execution.")

    # Test 10: Batch input generation.

    generated = create_benchmark_input(
        batch_size=3,
        image_size=32,
        device=device,
    )

    assert generated.shape == (3, 3, 32, 32)

    print("Test 10 passed: Benchmark input generation.")

    # Test 11: FLOPs increase with batch size.

    flops_one = counter.profile(
        torch.randn(1, 3, 224, 224)
    )

    flops_two = counter.profile(
        torch.randn(2, 3, 224, 224)
    )

    assert flops_two == 2 * flops_one

    print("Test 11 passed: FLOPs scale with batch size.")

    # Test 12: Parameter count remains unchanged after profiling.

    before = count_parameters(model)

    counter.profile(
        torch.randn(1, 3, 224, 224)
    )

    after = count_parameters(model)

    assert before == after

    print("Test 12 passed: Profiling does not modify parameters.")

    print("\nAll 12 tests passed successfully.")


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Lesson 70 — Model Profiling"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
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
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
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
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    parser.add_argument(
        "--torch-profiler",
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
        raise ValueError("--image-size must be positive.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive.")

    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")

    if any(size <= 0 for size in args.batch_sizes):
        raise ValueError("All batch sizes must be positive.")

    # --------------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------------

    if args.device == "auto":

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    else:

        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA requested, but CUDA is not available."
        )

    print("\n" + "=" * 75)

    print("LESSON 70 — MODEL PROFILING")

    print("=" * 75)

    print(f"Python:       {sys.version.split()[0]}")

    print(f"PyTorch:      {torch.__version__}")

    print(f"Platform:     {platform.platform()}")

    print(f"Device:       {device}")

    if device.type == "cuda":

        print(
            f"GPU:          {torch.cuda.get_device_name(device)}"
        )

    # --------------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------------

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # MODEL
    # --------------------------------------------------------------

    torch.manual_seed(42)

    np.random.seed(42)

    model = ProfilingCNN(
        num_classes=10
    ).to(device)

    model.eval()

    parameter_info = count_parameters(model)

    model_size_mb = estimate_model_size_mb(model)

    # --------------------------------------------------------------
    # INPUT
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
            "\nNo real image supplied. "
            "Using numerical benchmark input."
        )

    input_tensor = create_benchmark_input(
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=device,
        base_image=base_image,
    )

    # --------------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------------

    print("\nMODEL INFORMATION")

    print("-" * 50)

    print(
        f"Total parameters: "
        f"{parameter_info['total_parameters']:,}"
    )

    print(
        f"Trainable parameters: "
        f"{parameter_info['trainable_parameters']:,}"
    )

    print(
        f"Model parameter/buffer size: "
        f"{model_size_mb:.3f} MB"
    )

    print(
        f"Input shape: {tuple(input_tensor.shape)}"
    )

    # --------------------------------------------------------------
    # FLOPs
    # --------------------------------------------------------------

    flops_counter = FLOPsCounter(model)

    flops = flops_counter.profile(
        input_tensor
    )

    flops_per_image = (
        flops / args.batch_size
    )

    print(
        f"Approximate FLOPs per batch: {flops:,}"
    )

    print(
        f"Approximate FLOPs per image: "
        f"{flops_per_image:,}"
    )

    print(
        f"Approximate GFLOPs per image: "
        f"{flops_per_image / 1e9:.4f}"
    )

    # --------------------------------------------------------------
    # GPU MEMORY BASELINE
    # --------------------------------------------------------------

    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats(device)

    # --------------------------------------------------------------
    # PIPELINE STAGES
    # --------------------------------------------------------------

    print("\nProfiling pipeline stages...")

    stage_results = profile_pipeline_stages(
        model=model,
        input_tensor=input_tensor,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    print_benchmark_report(stage_results)

    # --------------------------------------------------------------
    # BATCH SIZE EXPERIMENT
    # --------------------------------------------------------------

    print("\n" + "=" * 75)

    print("BATCH SIZE EXPERIMENT")

    print("=" * 75)

    batch_results = profile_batch_sizes(
        model=model,
        device=device,
        image_size=args.image_size,
        batch_sizes=args.batch_sizes,
        warmup=args.warmup,
        iterations=args.iterations,
        base_image=base_image,
    )

    # --------------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------------

    memory_info = get_cuda_memory(device)

    process_memory_mb = get_process_memory_mb()

    print("\nMEMORY REPORT")

    print("-" * 50)

    print(
        f"Model parameter/buffer size: "
        f"{model_size_mb:.3f} MB"
    )

    print(
        f"Process RSS memory: "
        f"{process_memory_mb if process_memory_mb is not None else 'N/A'} MB"
    )

    for name, value in memory_info.items():

        if value is None:

            print(f"{name}: N/A")

        else:

            print(
                f"{name}: {value:.3f} MB"
            )

    # --------------------------------------------------------------
    # OPTIONAL TORCH PROFILER
    # --------------------------------------------------------------

    trace_path = None

    if args.torch_profiler:

        print("\nRunning detailed PyTorch profiler...")

        prepared = preprocess(input_tensor)

        trace_path = run_torch_profiler(
            model=model,
            input_tensor=prepared,
            device=device,
            output_dir=output_dir,
        )

        print(
            f"Profiler trace saved: {trace_path}"
        )

    # --------------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------------

    print("\nGenerating visualizations...")

    plot_stage_latency(
        stage_results,
        output_dir,
    )

    plot_latency_distribution(
        stage_results,
        output_dir,
    )

    plot_batch_results(
        batch_results,
        output_dir,
    )

    # --------------------------------------------------------------
    # JSON REPORT
    # --------------------------------------------------------------

    report = {

        "lesson": 70,

        "title": "Model Profiling",

        "device": str(device),

        "pytorch_version": torch.__version__,

        "image_size": args.image_size,

        "batch_size": args.batch_size,

        "warmup_iterations": args.warmup,

        "benchmark_iterations": args.iterations,

        "model": {

            "architecture": "ProfilingCNN",

            "parameters": parameter_info,

            "model_size_mb": model_size_mb,

            "approximate_flops_per_batch": flops,

            "approximate_flops_per_image": (
                flops_per_image
            ),

        },

        "stage_results": {

            name: {

                "statistics": result.statistics,

                "throughput_images_per_second": (
                    result.images_per_second
                ),

            }

            for name, result in stage_results.items()

        },

        "batch_results": batch_results,

        "memory": {

            "process_rss_mb": process_memory_mb,

            "cuda": memory_info,

        },

        "profiler_trace": trace_path,

        "input_source": (
            "real_image"
            if args.image
            else "numerical_benchmark_tensor"
        ),

    }

    report_path = (
        output_dir / "profiling_report.json"
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

    print("LESSON 70 COMPLETED")

    print("=" * 75)

    print("\nGenerated files:")

    for path in sorted(output_dir.iterdir()):

        print(f"  {path}")

    print("\nImportant interpretation:")

    print(
        "1. Mean latency describes typical computational cost."
    )

    print(
        "2. P95 and P99 reveal occasional slow inference."
    )

    print(
        "3. Throughput measures total processing capacity."
    )

    print(
        "4. Larger batches may improve throughput but increase "
        "latency and memory usage."
    )

    print(
        "5. Profiling identifies bottlenecks before optimization."
    )

    print("\nNext lesson: Lesson 71 — CPU vs GPU Inference.")


if __name__ == "__main__":

    main()