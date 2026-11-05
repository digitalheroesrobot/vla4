"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 6 — MODEL OPTIMIZATION AND DEPLOYMENT

LESSON 73 — MIXED PRECISION FP16
======================================================================

OBJECTIVE
---------

Understand and implement mixed-precision neural network inference.

We compare:

    1. FP32 inference.
    2. FP16 automatic mixed precision (AMP).
    3. Numerical differences between FP32 and FP16.
    4. Inference latency.
    5. Throughput.
    6. GPU memory consumption.
    7. Prediction consistency.

IMPORTANT:

FP16 acceleration is hardware-dependent.

This lesson uses CUDA FP16 autocasting when an NVIDIA GPU is
available.

On CPU, FP32 is benchmarked. CUDA FP16 is not silently substituted
with a different precision mode.

======================================================================
1. FLOATING-POINT REPRESENTATION
======================================================================

Neural networks use floating-point numbers to represent:

    - Model parameters.
    - Activations.
    - Gradients.
    - Intermediate calculations.

Common formats:

    FP32:
        32-bit floating point.

    FP16:
        16-bit floating point.

    BF16:
        16-bit brain floating point.

======================================================================
2. FP32
======================================================================

FP32 contains:

    1 sign bit
    8 exponent bits
    23 fraction bits

Total:

    32 bits

Advantages:

    - High numerical precision.
    - Large dynamic range.
    - Stable numerical calculations.

Disadvantages:

    - More memory.
    - Greater memory bandwidth requirements.
    - Potentially slower inference on hardware optimized for FP16.

======================================================================
3. FP16
======================================================================

FP16 contains:

    1 sign bit
    5 exponent bits
    10 fraction bits

Total:

    16 bits

FP16 requires approximately half the storage of FP32.

For example:

    FP32 tensor:

        1000 elements x 4 bytes

        = 4000 bytes

    FP16 tensor:

        1000 elements x 2 bytes

        = 2000 bytes

The theoretical storage reduction is:

    50%

However, actual model memory reduction depends on which tensors
are converted and how the inference runtime manages memory.

======================================================================
4. FP16 NUMERICAL RANGE
======================================================================

FP16 has a smaller dynamic range than FP32.

Approximate maximum finite value:

    FP16: 65504

Approximate minimum positive normal value:

    FP16: 6.10e-5

Very small values may underflow.

Very large values may overflow.

This can affect:

    - Softmax.
    - Normalization.
    - Large activations.
    - Small gradients.
    - Sensitive numerical operations.

======================================================================
5. MIXED PRECISION
======================================================================

Mixed precision does not necessarily convert every operation
to FP16.

Instead, compatible operations use FP16 while selected operations
retain FP32 precision.

Example:

    Input FP32
        |
        v
    Autocast
        |
        +---- Convolution: FP16
        |
        +---- Matrix multiplication: FP16
        |
        +---- Numerically sensitive operations: FP32
        |
        v
    Output

PyTorch automatically chooses operation precision according to
the autocast policy.

======================================================================
6. PYTORCH AUTOMATIC MIXED PRECISION
======================================================================

Example:

    model.eval()

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            output = model(input)

This enables automatic mixed precision for supported CUDA operations.

The model parameters can remain FP32.

======================================================================
7. FP16 MODEL STORAGE VS AMP
======================================================================

There are two different approaches.

A. Explicit FP16 conversion:

    model.half()

    input.half()

This converts model parameters and buffers to FP16.

B. Automatic Mixed Precision:

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):

        output = model(input)

AMP allows operations to use appropriate precision.

This lesson focuses on AMP.

======================================================================
8. NUMERICAL ERROR
======================================================================

FP16 predictions can differ slightly from FP32 predictions.

We measure:

    Maximum absolute error:

        max(|FP32 - FP16|)

    Mean absolute error:

        mean(|FP32 - FP16|)

    Root mean square error:

        sqrt(mean((FP32 - FP16)^2))

    Cosine similarity:

        dot(A, B) / (||A|| * ||B||)

Cosine similarity close to 1 indicates similar output directions.

======================================================================
9. CLASSIFICATION CONSISTENCY
======================================================================

For classification:

    FP32 prediction:

        class 3

    FP16 prediction:

        class 3

The predicted class is unchanged.

However, numerical output differences can still exist.

We therefore measure both:

    - Numerical similarity.
    - Top-1 prediction agreement.

======================================================================
10. MEMORY COMPARISON
======================================================================

For CUDA benchmarking:

    torch.cuda.reset_peak_memory_stats()

    ...

    torch.cuda.max_memory_allocated()

We measure peak allocated memory during inference.

IMPORTANT:

AMP may retain FP32 model parameters, so do not expect total
memory usage to be exactly 50% lower.

FP16 activations and intermediate tensors can still provide
substantial savings.

======================================================================
11. ROBOTICS APPLICATION
======================================================================

Consider an OpenArm V2 waste-sorting system.

Five cameras provide images:

    Ceiling
    Head Left
    Head Right
    Wrist Left
    Wrist Right

A perception model processes the images.

The inference pipeline:

    Camera images
         |
         v
    Preprocessing
         |
         v
    FP16 object detection
         |
         v
    Object localization
         |
         v
    3-D coordinate estimation
         |
         v
    Grasp planning
         |
         v
    ACT policy

FP16 can improve inference efficiency and allow more images
to fit into GPU memory.

However:

    - Verify detection accuracy.
    - Verify small-object performance.
    - Check numerical stability.
    - Test real-time latency.
    - Validate robot grasp success.

A small numerical difference can sometimes change a detection
near a confidence threshold.

======================================================================
12. IMPORTANT DISTINCTION
======================================================================

FP16 does not automatically make a model accurate.

It does not improve the learned representation.

It changes numerical computation and storage.

The objective is:

    Lower resource consumption
    +
    Acceptable numerical differences
    +
    Improved inference performance

======================================================================
WHAT YOU WILL LEARN
======================================================================

After this lesson, you should understand:

    - FP32 versus FP16.
    - Floating-point precision.
    - Dynamic range.
    - Automatic mixed precision.
    - CUDA autocasting.
    - Numerical error measurement.
    - Memory benchmarking.
    - Precision selection for deployment.

NEXT LESSON
-----------

Lesson 74 — INT8 Quantization.

======================================================================
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

from dataclasses import dataclass
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

DEFAULT_BATCH_SIZES = [1, 2, 4, 8]

DEFAULT_WARMUP = 20

DEFAULT_ITERATIONS = 100

DEFAULT_OUTPUT_DIR = Path("runs/lesson73")


# ======================================================================
# DEVICE UTILITIES
# ======================================================================

def select_device(
    requested: str,
) -> torch.device:

    if requested == "cpu":

        return torch.device("cpu")

    if requested == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        return torch.device("cuda")

    if requested == "auto":

        if torch.cuda.is_available():

            return torch.device("cuda")

        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device: {requested}"
    )


def synchronize(
    device: torch.device,
) -> None:

    if device.type == "cuda":

        torch.cuda.synchronize(device)


# ======================================================================
# MODEL ARCHITECTURE
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

        self.relu = nn.ReLU(
            inplace=False
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv(x)

        x = self.bn(x)

        x = self.relu(x)

        return x


class PrecisionBenchmarkCNN(nn.Module):

    """
    CNN used to compare FP32 and FP16 inference.

    Input:
        B x 3 x H x W

    Output:
        B x 10

    The model is randomly initialized.

    No pretrained weights are used.
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
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Linear(
            256,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

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
# INPUT PREPARATION
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


def create_input(
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

        # Random numerical input for benchmarking.
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
            "Expected base image shape "
            f"(1, 3, {image_size}, {image_size})."
        )

    return base_image.repeat(
        batch_size,
        1,
        1,
        1,
    )


def preprocess(
    x: torch.Tensor,
) -> torch.Tensor:

    return x * 2.0 - 1.0


# ======================================================================
# PRECISION MODES
# ======================================================================

def forward_fp32(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:

    """
    Full FP32 inference.

    Model parameters and inputs remain FP32.
    """

    with torch.inference_mode():

        return model(x)


def forward_fp16_amp(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:

    """
    CUDA automatic mixed precision.

    Model weights remain FP32.

    Supported operations are automatically executed using FP16
    or another precision selected by the autocast policy.

    Output is converted to FP32 for consistent numerical comparison.
    """

    if device.type != "cuda":

        raise RuntimeError(
            "FP16 AMP benchmark requires CUDA."
        )

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            output = model(x)

        return output.float()


# ======================================================================
# FP16 CAPABILITY INFORMATION
# ======================================================================

def get_precision_information(
    device: torch.device,
) -> dict[str, Any]:

    info = {

        "device": str(device),

        "cuda_available": torch.cuda.is_available(),

        "fp16_amp_supported": (
            device.type == "cuda"
        ),

        "gpu_name": None,

        "compute_capability": None,

    }

    if device.type == "cuda":

        info["gpu_name"] = (
            torch.cuda.get_device_name(device)
        )

        capability = torch.cuda.get_device_capability(
            device
        )

        info["compute_capability"] = (
            f"{capability[0]}.{capability[1]}"
        )

    return info


# ======================================================================
# NUMERICAL COMPARISON
# ======================================================================

def compare_outputs(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:

    """
    Compare FP16 output against FP32 reference.

    All calculations are performed in FP64 for reporting.
    """

    if reference.shape != candidate.shape:

        raise ValueError(
            "Output shapes must match."
        )

    a = reference.detach().double()

    b = candidate.detach().double()

    difference = a - b

    absolute_difference = difference.abs()

    mse = torch.mean(
        difference ** 2
    )

    rmse = torch.sqrt(mse)

    mae = torch.mean(
        absolute_difference
    )

    max_error = torch.max(
        absolute_difference
    )

    reference_norm = torch.linalg.vector_norm(
        a
    )

    candidate_norm = torch.linalg.vector_norm(
        b
    )

    denominator = (
        reference_norm * candidate_norm
    )

    if denominator.item() == 0:

        cosine_similarity = 1.0

    else:

        cosine_similarity = torch.sum(
            a * b
        ) / denominator

    relative_error = (
        torch.linalg.vector_norm(difference)
        /
        torch.linalg.vector_norm(a).clamp_min(1e-12)
    )

    return {

        "mae": float(mae.item()),

        "rmse": float(rmse.item()),

        "max_absolute_error": float(max_error.item()),

        "relative_l2_error": float(relative_error.item()),

        "cosine_similarity": float(
            cosine_similarity.item()
        ),

    }


def compare_class_predictions(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:

    reference_classes = torch.argmax(
        reference,
        dim=1,
    )

    candidate_classes = torch.argmax(
        candidate,
        dim=1,
    )

    agreement = (
        reference_classes == candidate_classes
    )

    agreement_rate = (
        agreement.float().mean().item()
    )

    return {

        "agreement_rate": float(agreement_rate),

        "matching_samples": int(
            agreement.sum().item()
        ),

        "total_samples": int(
            agreement.numel()
        ),

        "fp32_classes": (
            reference_classes.cpu().tolist()
        ),

        "fp16_classes": (
            candidate_classes.cpu().tolist()
        ),

    }


# ======================================================================
# GPU MEMORY
# ======================================================================

def reset_gpu_memory(
    device: torch.device,
) -> None:

    if device.type == "cuda":

        synchronize(device)

        torch.cuda.reset_peak_memory_stats(
            device
        )


def measure_gpu_memory(
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


# ======================================================================
# BENCHMARK RESULT
# ======================================================================

@dataclass
class PrecisionBenchmarkResult:

    precision: str

    batch_size: int

    image_size: int

    mean_latency_ms: float

    p50_latency_ms: float

    p95_latency_ms: float

    p99_latency_ms: float

    std_latency_ms: float

    throughput_fps: float

    memory: dict[str, float | None]

    successful: bool = True

    error: str | None = None


# ======================================================================
# LATENCY STATISTICS
# ======================================================================

def latency_statistics(
    latencies: list[float],
) -> dict[str, float]:

    if not latencies:

        raise ValueError(
            "No latency measurements."
        )

    values = np.asarray(
        latencies,
        dtype=np.float64,
    )

    return {

        "mean": float(np.mean(values)),

        "median": float(np.median(values)),

        "p50": float(np.percentile(values, 50)),

        "p95": float(np.percentile(values, 95)),

        "p99": float(np.percentile(values, 99)),

        "std": float(np.std(values)),

    }


# ======================================================================
# WARMUP
# ======================================================================

def warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    precision: str,
    warmup_iterations: int,
) -> None:

    model.eval()

    for _ in range(warmup_iterations):

        if precision == "fp32":

            output = forward_fp32(
                model,
                x,
            )

        elif precision == "fp16_amp":

            output = forward_fp16_amp(
                model,
                x,
                device,
            )

        else:

            raise ValueError(
                f"Unsupported precision: {precision}"
            )

        del output

    synchronize(device)


# ======================================================================
# SINGLE PRECISION BENCHMARK
# ======================================================================

def benchmark_precision(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    precision: str,
    warmup_iterations: int,
    iterations: int,
) -> PrecisionBenchmarkResult:

    batch_size = x.shape[0]

    image_size = x.shape[-1]

    try:

        warmup_model(
            model,
            x,
            device,
            precision,
            warmup_iterations,
        )

        reset_gpu_memory(device)

        latencies = []

        synchronize(device)

        total_start = time.perf_counter()

        for _ in range(iterations):

            synchronize(device)

            start = time.perf_counter()

            if precision == "fp32":

                output = forward_fp32(
                    model,
                    x,
                )

            elif precision == "fp16_amp":

                output = forward_fp16_amp(
                    model,
                    x,
                    device,
                )

            else:

                raise ValueError(
                    f"Unsupported precision: {precision}"
                )

            synchronize(device)

            end = time.perf_counter()

            latencies.append(
                (end - start) * 1000.0
            )

            del output

        synchronize(device)

        total_end = time.perf_counter()

        total_elapsed = (
            total_end - total_start
        )

        stats = latency_statistics(
            latencies
        )

        throughput = (
            batch_size * iterations
            / total_elapsed
        )

        memory = measure_gpu_memory(
            device
        )

        return PrecisionBenchmarkResult(

            precision=precision,

            batch_size=batch_size,

            image_size=image_size,

            mean_latency_ms=stats["mean"],

            p50_latency_ms=stats["p50"],

            p95_latency_ms=stats["p95"],

            p99_latency_ms=stats["p99"],

            std_latency_ms=stats["std"],

            throughput_fps=throughput,

            memory=memory,

        )

    except RuntimeError as error:

        if device.type == "cuda":

            torch.cuda.empty_cache()

        return PrecisionBenchmarkResult(

            precision=precision,

            batch_size=batch_size,

            image_size=image_size,

            mean_latency_ms=0.0,

            p50_latency_ms=0.0,

            p95_latency_ms=0.0,

            p99_latency_ms=0.0,

            std_latency_ms=0.0,

            throughput_fps=0.0,

            memory=measure_gpu_memory(device),

            successful=False,

            error=str(error),

        )


# ======================================================================
# PRECISION EXPERIMENT
# ======================================================================

def run_precision_experiment(
    model: nn.Module,
    device: torch.device,
    batch_sizes: list[int],
    image_size: int,
    base_image: torch.Tensor | None,
    warmup_iterations: int,
    iterations: int,
) -> list[dict[str, Any]]:

    results = []

    precisions = ["fp32"]

    if device.type == "cuda":

        precisions.append("fp16_amp")

    for batch_size in batch_sizes:

        print("\n" + "=" * 75)

        print(
            f"BATCH SIZE: {batch_size}"
        )

        print("=" * 75)

        x = create_input(
            batch_size,
            image_size,
            base_image,
        )

        x = preprocess(x).to(device)

        fp32_output = None

        for precision in precisions:

            print(
                f"\nBenchmarking {precision}..."
            )

            if precision == "fp32":

                with torch.inference_mode():

                    fp32_output = forward_fp32(
                        model,
                        x,
                    ).detach()

            result = benchmark_precision(
                model=model,
                x=x,
                device=device,
                precision=precision,
                warmup_iterations=warmup_iterations,
                iterations=iterations,
            )

            results.append({

                "benchmark": result,

                "numerical_error": None,

                "prediction_comparison": None,

            })

            if not result.successful:

                print(
                    f"FAILED: {result.error}"
                )

                continue

            print(
                f"Mean latency: "
                f"{result.mean_latency_ms:.4f} ms"
            )

            print(
                f"P95 latency: "
                f"{result.p95_latency_ms:.4f} ms"
            )

            print(
                f"Throughput: "
                f"{result.throughput_fps:.2f} images/s"
            )

            if result.memory["peak_allocated_mb"] is not None:

                print(
                    f"Peak allocated GPU memory: "
                    f"{result.memory['peak_allocated_mb']:.2f} MB"
                )

            # ------------------------------------------------------
            # Numerical comparison
            # ------------------------------------------------------

            if precision == "fp16_amp":

                with torch.inference_mode():

                    fp16_output = forward_fp16_amp(
                        model,
                        x,
                        device,
                    )

                numerical_error = compare_outputs(
                    fp32_output,
                    fp16_output,
                )

                prediction_comparison = compare_class_predictions(
                    fp32_output,
                    fp16_output,
                )

                results[-1]["numerical_error"] = (
                    numerical_error
                )

                results[-1]["prediction_comparison"] = (
                    prediction_comparison
                )

                print("\nNumerical comparison:")

                print(
                    f"  MAE: "
                    f"{numerical_error['mae']:.8e}"
                )

                print(
                    f"  RMSE: "
                    f"{numerical_error['rmse']:.8e}"
                )

                print(
                    f"  Maximum absolute error: "
                    f"{numerical_error['max_absolute_error']:.8e}"
                )

                print(
                    f"  Cosine similarity: "
                    f"{numerical_error['cosine_similarity']:.8f}"
                )

                print(
                    f"  Prediction agreement: "
                    f"{prediction_comparison['agreement_rate'] * 100:.2f}%"
                )

                del fp16_output

        del x

        if fp32_output is not None:

            del fp32_output

    return results


# ======================================================================
# REPORTING
# ======================================================================

def print_results(
    results: list[dict[str, Any]],
) -> None:

    print("\n" + "=" * 105)

    print("PRECISION BENCHMARK RESULTS")

    print("=" * 105)

    print(
        f"{'Precision':>12}"
        f"{'Batch':>8}"
        f"{'Mean ms':>12}"
        f"{'P95 ms':>12}"
        f"{'FPS':>14}"
        f"{'Peak MB':>13}"
        f"{'Status':>12}"
    )

    print("-" * 105)

    for item in results:

        result = item["benchmark"]

        if not result.successful:

            print(
                f"{result.precision:>12}"
                f"{result.batch_size:>8}"
                f"{'-':>12}"
                f"{'-':>12}"
                f"{'-':>14}"
                f"{'-':>13}"
                f"{'FAILED':>12}"
            )

            continue

        peak = result.memory[
            "peak_allocated_mb"
        ]

        peak_text = (
            f"{peak:.2f}"
            if peak is not None
            else "N/A"
        )

        print(
            f"{result.precision:>12}"
            f"{result.batch_size:>8}"
            f"{result.mean_latency_ms:>12.4f}"
            f"{result.p95_latency_ms:>12.4f}"
            f"{result.throughput_fps:>14.2f}"
            f"{peak_text:>13}"
            f"{'OK':>12}"
        )

    print("=" * 105)


def serialize_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    output = []

    for item in results:

        benchmark = item["benchmark"]

        output.append({

            "precision": benchmark.precision,

            "batch_size": benchmark.batch_size,

            "image_size": benchmark.image_size,

            "successful": benchmark.successful,

            "error": benchmark.error,

            "mean_latency_ms": benchmark.mean_latency_ms,

            "p50_latency_ms": benchmark.p50_latency_ms,

            "p95_latency_ms": benchmark.p95_latency_ms,

            "p99_latency_ms": benchmark.p99_latency_ms,

            "std_latency_ms": benchmark.std_latency_ms,

            "throughput_fps": benchmark.throughput_fps,

            "memory": benchmark.memory,

            "numerical_error": item["numerical_error"],

            "prediction_comparison": (
                item["prediction_comparison"]
            ),

        })

    return output


# ======================================================================
# VISUALIZATION
# ======================================================================

def plot_precision_latency(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    precisions = sorted(
        set(
            item["benchmark"].precision
            for item in results
        )
    )

    plotted = False

    for precision in precisions:

        valid = [

            item["benchmark"]

            for item in results

            if (
                item["benchmark"].precision == precision
                and item["benchmark"].successful
            )

        ]

        if not valid:

            continue

        ax.plot(
            [
                item.batch_size
                for item in valid
            ],
            [
                item.mean_latency_ms
                for item in valid
            ],
            marker="o",
            label=precision,
        )

        plotted = True

    if plotted:

        ax.set_xlabel("Batch size")

        ax.set_ylabel("Mean batch latency (ms)")

        ax.set_title(
            "FP32 vs FP16 Inference Latency"
        )

        ax.grid(alpha=0.3)

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            output_dir / "precision_latency.png",
            dpi=150,
        )

    plt.close(fig)


def plot_precision_throughput(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    precisions = sorted(
        set(
            item["benchmark"].precision
            for item in results
        )
    )

    plotted = False

    for precision in precisions:

        valid = [

            item["benchmark"]

            for item in results

            if (
                item["benchmark"].precision == precision
                and item["benchmark"].successful
            )

        ]

        if not valid:

            continue

        ax.plot(
            [
                item.batch_size
                for item in valid
            ],
            [
                item.throughput_fps
                for item in valid
            ],
            marker="o",
            label=precision,
        )

        plotted = True

    if plotted:

        ax.set_xlabel("Batch size")

        ax.set_ylabel("Throughput (images/second)")

        ax.set_title(
            "FP32 vs FP16 Throughput"
        )

        ax.grid(alpha=0.3)

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            output_dir / "precision_throughput.png",
            dpi=150,
        )

    plt.close(fig)


def plot_precision_memory(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    precisions = sorted(
        set(
            item["benchmark"].precision
            for item in results
        )
    )

    plotted = False

    for precision in precisions:

        valid = [

            item["benchmark"]

            for item in results

            if (
                item["benchmark"].precision == precision
                and item["benchmark"].successful
                and item["benchmark"].memory[
                    "peak_allocated_mb"
                ] is not None
            )

        ]

        if not valid:

            continue

        ax.plot(
            [
                item.batch_size
                for item in valid
            ],
            [
                item.memory["peak_allocated_mb"]
                for item in valid
            ],
            marker="o",
            label=precision,
        )

        plotted = True

    if plotted:

        ax.set_xlabel("Batch size")

        ax.set_ylabel("Peak allocated GPU memory (MB)")

        ax.set_title(
            "FP32 vs FP16 GPU Memory"
        )

        ax.grid(alpha=0.3)

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            output_dir / "precision_memory.png",
            dpi=150,
        )

    plt.close(fig)


def plot_numerical_error(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:

    valid = [

        item

        for item in results

        if (
            item["benchmark"].precision == "fp16_amp"
            and item["benchmark"].successful
            and item["numerical_error"] is not None
        )

    ]

    if not valid:

        return

    batch_sizes = [

        item["benchmark"].batch_size

        for item in valid

    ]

    mae = [

        item["numerical_error"]["mae"]

        for item in valid

    ]

    rmse = [

        item["numerical_error"]["rmse"]

        for item in valid

    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        batch_sizes,
        mae,
        marker="o",
        label="MAE",
    )

    ax.plot(
        batch_sizes,
        rmse,
        marker="s",
        label="RMSE",
    )

    ax.set_xlabel("Batch size")

    ax.set_ylabel("Numerical error")

    ax.set_title(
        "FP16 Numerical Error vs Batch Size"
    )

    ax.set_yscale("log")

    ax.grid(alpha=0.3)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "fp16_numerical_error.png",
        dpi=150,
    )

    plt.close(fig)


# ======================================================================
# NUMERICAL TESTS
# ======================================================================

def run_tests() -> None:

    print("\nRunning Lesson 73 tests...\n")

    # Test 1: FP32 model output.

    model = PrecisionBenchmarkCNN()

    x = torch.randn(
        2,
        3,
        64,
        64,
    )

    model.eval()

    with torch.inference_mode():

        output = model(x)

    assert output.shape == (2, 10)

    assert output.dtype == torch.float32

    print("Test 1 passed: FP32 model inference.")

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

    print("Test 2 passed: Single-image inference.")

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

    print("Test 3 passed: Batched inference.")

    # Test 4: Numerical comparison of identical tensors.

    a = torch.randn(
        4,
        10,
    )

    comparison = compare_outputs(
        a,
        a.clone(),
    )

    assert comparison["mae"] == 0.0

    assert comparison["rmse"] == 0.0

    assert comparison["max_absolute_error"] == 0.0

    assert np.isclose(
        comparison["cosine_similarity"],
        1.0,
    )

    print("Test 4 passed: Identical-output comparison.")

    # Test 5: Numerical comparison of slightly different tensors.

    b = a + 0.001

    comparison = compare_outputs(
        a,
        b,
    )

    assert comparison["mae"] > 0.0

    assert comparison["rmse"] > 0.0

    assert comparison["max_absolute_error"] > 0.0

    print("Test 5 passed: Numerical error detection.")

    # Test 6: Classification agreement.

    logits = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.8, 0.2, 0.1],
        ]
    )

    prediction_comparison = compare_class_predictions(
        logits,
        logits.clone(),
    )

    assert prediction_comparison["agreement_rate"] == 1.0

    assert prediction_comparison["matching_samples"] == 2

    print("Test 6 passed: Prediction agreement.")

    # Test 7: Batch creation.

    batch = create_input(
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

    assert batch.dtype == torch.float32

    print("Test 7 passed: Input batch creation.")

    # Test 8: Real image replication.

    base_image = torch.rand(
        1,
        3,
        32,
        32,
    )

    replicated = create_input(
        batch_size=3,
        image_size=32,
        base_image=base_image,
    )

    assert replicated.shape == (
        3,
        3,
        32,
        32,
    )

    assert torch.equal(
        replicated[0],
        replicated[2],
    )

    print("Test 8 passed: Image batch replication.")

    # Test 9: Preprocessing.

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
    )

    print("Test 9 passed: Input normalization.")

    # Test 10: Latency statistics.

    stats = latency_statistics(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    assert np.isclose(
        stats["mean"],
        3.0,
    )

    assert np.isclose(
        stats["median"],
        3.0,
    )

    assert np.isclose(
        stats["p50"],
        3.0,
    )

    print("Test 10 passed: Latency statistics.")

    # Test 11: FP32 and FP16 storage size.

    fp32_tensor = torch.zeros(
        1000,
        dtype=torch.float32,
    )

    fp16_tensor = torch.zeros(
        1000,
        dtype=torch.float16,
    )

    assert fp32_tensor.element_size() == 4

    assert fp16_tensor.element_size() == 2

    assert (
        fp16_tensor.numel() * fp16_tensor.element_size()
        ==
        fp32_tensor.numel() * fp32_tensor.element_size() / 2
    )

    print("Test 11 passed: FP16 storage reduction.")

    # Test 12: FP16 numeric range.

    assert torch.finfo(
        torch.float16
    ).max == 65504.0

    assert torch.finfo(
        torch.float32
    ).max > torch.finfo(
        torch.float16
    ).max

    print("Test 12 passed: Floating-point range.")

    # Test 13: FP32 benchmark on CPU.

    cpu_model = PrecisionBenchmarkCNN()

    cpu_model.eval()

    cpu_x = torch.rand(
        2,
        3,
        64,
        64,
    )

    result = benchmark_precision(
        model=cpu_model,
        x=cpu_x,
        device=torch.device("cpu"),
        precision="fp32",
        warmup_iterations=2,
        iterations=5,
    )

    assert result.successful

    assert result.mean_latency_ms > 0

    assert result.throughput_fps > 0

    print("Test 13 passed: CPU FP32 benchmark.")

    # Test 14: CPU cannot use this CUDA FP16 path.

    try:

        forward_fp16_amp(
            cpu_model,
            cpu_x,
            torch.device("cpu"),
        )

        raise AssertionError(
            "CUDA FP16 AMP should reject CPU."
        )

    except RuntimeError:

        pass

    print("Test 14 passed: CUDA FP16 device validation.")

    # Test 15: Output shape consistency.

    with torch.inference_mode():

        reference = forward_fp32(
            cpu_model,
            cpu_x,
        )

    assert reference.shape == (2, 10)

    print("Test 15 passed: Reference output shape.")

    print("\nAll 15 tests passed successfully.")


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Lesson 73 — Mixed Precision FP16"
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

    if args.image_size <= 0:

        raise ValueError(
            "Image size must be positive."
        )

    if args.iterations <= 0:

        raise ValueError(
            "Iterations must be positive."
        )

    if args.warmup < 0:

        raise ValueError(
            "Warmup iterations cannot be negative."
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

    print("LESSON 73 — MIXED PRECISION FP16")

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

    precision_info = get_precision_information(
        device
    )

    print(
        f"\nDevice: {device}"
    )

    if precision_info["gpu_name"]:

        print(
            f"GPU: {precision_info['gpu_name']}"
        )

        print(
            f"Compute capability: "
            f"{precision_info['compute_capability']}"
        )

    print(
        f"CUDA FP16 AMP available: "
        f"{precision_info['fp16_amp_supported']}"
    )

    print(
        f"Image size: {args.image_size}"
    )

    print(
        f"Batch sizes: {batch_sizes}"
    )

    # --------------------------------------------------------------
    # OPTIONAL REAL IMAGE
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
            "\nUsing numerical benchmarking tensors."
        )

    # --------------------------------------------------------------
    # MODEL INITIALIZATION
    # --------------------------------------------------------------

    torch.manual_seed(42)

    np.random.seed(42)

    model = PrecisionBenchmarkCNN()

    model.eval()

    model = model.to(
        device=device,
        dtype=torch.float32,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    fp32_parameter_bytes = sum(

        parameter.numel()
        * parameter.element_size()

        for parameter in model.parameters()

    )

    print(
        f"\nModel parameters: {parameter_count:,}"
    )

    print(
        f"FP32 parameter storage: "
        f"{fp32_parameter_bytes / (1024 ** 2):.2f} MB"
    )

    # --------------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------------

    results = run_precision_experiment(
        model=model,
        device=device,
        batch_sizes=batch_sizes,
        image_size=args.image_size,
        base_image=base_image,
        warmup_iterations=args.warmup,
        iterations=args.iterations,
    )

    print_results(
        results
    )

    # --------------------------------------------------------------
    # VISUALIZATIONS
    # --------------------------------------------------------------

    print("\nGenerating visualizations...")

    plot_precision_latency(
        results,
        output_dir,
    )

    plot_precision_throughput(
        results,
        output_dir,
    )

    plot_precision_memory(
        results,
        output_dir,
    )

    plot_numerical_error(
        results,
        output_dir,
    )

    # --------------------------------------------------------------
    # JSON REPORT
    # --------------------------------------------------------------

    report = {

        "lesson": 73,

        "title": "Mixed Precision FP16",

        "precision_information": precision_info,

        "image_size": args.image_size,

        "batch_sizes": batch_sizes,

        "warmup_iterations": args.warmup,

        "benchmark_iterations": args.iterations,

        "model_parameters": parameter_count,

        "fp32_parameter_storage_bytes": fp32_parameter_bytes,

        "results": serialize_results(
            results
        ),

    }

    report_path = (
        output_dir / "fp16_benchmark_report.json"
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

    print("LESSON 73 COMPLETED")

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
        "1. FP16 uses half the storage per element compared with FP32."
    )

    print(
        "2. Mixed precision allows compatible operations to use FP16."
    )

    print(
        "3. FP16 can improve GPU throughput on supported hardware."
    )

    print(
        "4. FP16 may introduce numerical differences."
    )

    print(
        "5. AMP does not automatically halve total GPU memory usage."
    )

    print(
        "6. Always validate numerical behavior and task accuracy."
    )

    print(
        "\nNext: Lesson 74 — INT8 Quantization."
    )


if __name__ == "__main__":

    main()