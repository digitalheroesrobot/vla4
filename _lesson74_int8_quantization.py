"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 6 — MODEL OPTIMIZATION AND DEPLOYMENT

LESSON 74 — INT8 QUANTIZATION
======================================================================

OBJECTIVE
---------

Implement and understand INT8 quantization.

Compare:

    1. FP32 floating-point representation.
    2. INT8 integer representation.
    3. Symmetric quantization.
    4. Asymmetric quantization.
    5. Per-tensor quantization.
    6. Per-channel quantization.
    7. Quantization error.
    8. Dynamic quantization.
    9. Static quantization.
    10. FP32 versus INT8 inference.

No pretrained weights are used.

The CNN is initialized randomly.

======================================================================
1. WHAT IS QUANTIZATION?
======================================================================

Quantization converts high-precision numerical values into
lower-precision representations.

Example:

    FP32:

        0.123456
        0.234567
        0.345678

    INT8:

        12
        24
        35

The integer representation is smaller.

However, the conversion introduces approximation error.

======================================================================
2. INT8 REPRESENTATION
======================================================================

Signed INT8 range:

    -128 to 127

Total possible values:

    256

FP32 uses 4 bytes per element.

INT8 uses 1 byte per element.

Therefore:

    INT8 storage = 25% of FP32 storage

The theoretical reduction is 75%.

Actual model-size savings depend on metadata, unsupported
operations, and the quantization implementation.

======================================================================
3. LINEAR QUANTIZATION
======================================================================

The general quantization equation is:

    q = round(x / scale) + zero_point

Where:

    x:
        Original floating-point value.

    q:
        Quantized integer.

    scale:
        Conversion scale.

    zero_point:
        Integer representing real zero.

The integer result is clipped to the allowed range.

For signed INT8:

    q = clamp(round(x / scale) + zero_point, -128, 127)

======================================================================
4. DEQUANTIZATION
======================================================================

Recover an approximate floating-point value:

    x_approx = scale * (q - zero_point)

Quantization and dequantization are not exact inverses.

The original information cannot always be recovered.

======================================================================
5. SYMMETRIC QUANTIZATION
======================================================================

Symmetric quantization uses:

    zero_point = 0

For signed INT8:

    q = round(x / scale)

A common scale calculation is:

    scale = max(abs(x)) / 127

Advantages:

    - Simple.
    - Efficient.
    - Convenient for weights.

Disadvantages:

    - May waste representable values for asymmetric distributions.

======================================================================
6. ASYMMETRIC QUANTIZATION
======================================================================

Asymmetric quantization maps the observed minimum and maximum
to the available integer range.

For INT8:

    q_min = -128
    q_max = 127

Scale:

    scale = (x_max - x_min) / (q_max - q_min)

Zero point:

    zero_point = round(
        q_min - x_min / scale
    )

Then clamp zero_point to INT8 range.

Advantages:

    - Can represent asymmetric distributions more efficiently.

Disadvantages:

    - Requires handling zero_point.
    - Slightly more complicated arithmetic.

======================================================================
7. PER-TENSOR QUANTIZATION
======================================================================

One scale and one zero-point are used for an entire tensor.

    Tensor
       |
       v
    One scale
       |
       v
    INT8 tensor

Simple and efficient.

However, different channels may have very different ranges.

======================================================================
8. PER-CHANNEL QUANTIZATION
======================================================================

Each output channel gets its own quantization scale.

For a convolution weight tensor:

    [out_channels, in_channels, height, width]

We can calculate one scale for each output channel.

This often preserves weight precision better.

======================================================================
9. CALIBRATION
======================================================================

Static quantization requires information about activation ranges.

Calibration collects representative model inputs and measures
activation statistics.

Example:

    Camera images
         |
         v
    FP32 model
         |
         v
    Observe activation ranges
         |
         v
    Calculate quantization parameters
         |
         v
    Quantized model

Calibration data should represent real deployment conditions.

For waste sorting, include:

    - Different lighting.
    - Different waste materials.
    - Different object sizes.
    - Different camera angles.
    - Different backgrounds.

======================================================================
10. DYNAMIC QUANTIZATION
======================================================================

Dynamic quantization calculates activation quantization
parameters during inference.

It is convenient for certain operations, especially Linear layers.

It does not mean every operation in a CNN is executed in INT8.

======================================================================
11. STATIC QUANTIZATION
======================================================================

Static quantization uses calibration statistics to determine
activation quantization parameters ahead of inference.

This can support quantized convolution and linear operations
on compatible backends.

======================================================================
12. QUANTIZATION-AWARE TRAINING
======================================================================

Quantization-aware training simulates quantization effects
during training.

The forward pass uses fake quantization:

    FP32
      |
      v
    Quantize
      |
      v
    Dequantize
      |
      v
    Continue FP32 training

The model learns to tolerate quantization error.

This lesson introduces the concept but does not train a QAT model.

======================================================================
13. QUANTIZATION ERROR
======================================================================

We measure:

    MAE:

        mean(abs(x - x_approx))

    RMSE:

        sqrt(mean((x - x_approx)^2))

    Maximum error:

        max(abs(x - x_approx))

    Relative L2 error:

        ||x - x_approx|| / ||x||

======================================================================
14. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste-sorting perception:

    RGB camera
        |
        v
    FP32 object detection
        |
        v
    INT8 object detection
        |
        v
    Bounding boxes
        |
        v
    Object localization
        |
        v
    Grasp point estimation
        |
        v
    ACT policy
        |
        v
    Robot control

The important question is not only whether INT8 is faster.

We must also ask:

    - Are small objects still detected?
    - Are confidence scores stable?
    - Are bounding boxes accurate?
    - Does grasp localization change?
    - Does the robot still succeed?

======================================================================
WHAT YOU WILL LEARN
======================================================================

After this lesson:

    - Understand INT8 representation.
    - Implement symmetric quantization.
    - Implement asymmetric quantization.
    - Implement per-channel quantization.
    - Perform calibration.
    - Measure quantization error.
    - Understand PyTorch quantization.
    - Compare FP32 and quantized inference.

NEXT LESSON
-----------

Lesson 75 — Model Pruning.

======================================================================
"""

from __future__ import annotations

import argparse
import copy
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

DEFAULT_BATCH_SIZES = [1, 4, 8]

DEFAULT_WARMUP = 10

DEFAULT_ITERATIONS = 50

DEFAULT_OUTPUT_DIR = Path("runs/lesson74")

INT8_MIN = -128

INT8_MAX = 127


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


def synchronize(
    device: torch.device,
) -> None:

    if device.type == "cuda":

        torch.cuda.synchronize(device)


# ======================================================================
# FLOATING-POINT MODEL
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

        return self.relu(
            self.bn(
                self.conv(x)
            )
        )


class QuantizationCNN(nn.Module):

    def __init__(
        self,
        num_classes: int = 10,
    ):

        super().__init__()

        self.features = nn.Sequential(

            ConvBlock(3, 32, 2),

            ConvBlock(32, 64, 2),

            ConvBlock(64, 128, 2),

            ConvBlock(128, 256, 2),

            ConvBlock(256, 256),

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

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        return self.classifier(x)


# ======================================================================
# REAL IMAGE
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

    return torch.from_numpy(
        array.copy()
    ).unsqueeze(0)


def create_input(
    batch_size: int,
    image_size: int,
    base_image: torch.Tensor | None,
) -> torch.Tensor:

    if base_image is None:

        return torch.rand(
            batch_size,
            3,
            image_size,
            image_size,
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
# SYMMETRIC QUANTIZATION
# ======================================================================

@dataclass
class QuantizationParameters:

    scale: torch.Tensor

    zero_point: torch.Tensor

    qmin: int

    qmax: int

    mode: str

    axis: int | None = None


def symmetric_quantize(
    x: torch.Tensor,
    qmin: int = INT8_MIN,
    qmax: int = INT8_MAX,
) -> tuple[torch.Tensor, QuantizationParameters]:

    """
    Symmetric per-tensor quantization.

    scale = max(abs(x)) / 127

    zero_point = 0
    """

    if not x.is_floating_point():

        raise TypeError(
            "Input must be floating point."
        )

    if not torch.isfinite(x).all():

        raise ValueError(
            "Input contains non-finite values."
        )

    max_abs = torch.max(
        torch.abs(x)
    )

    scale = max_abs / max(
        abs(qmin),
        abs(qmax),
    )

    scale = torch.clamp(
        scale,
        min=1e-12,
    )

    zero_point = torch.tensor(
        0,
        device=x.device,
        dtype=torch.int32,
    )

    quantized = torch.round(
        x / scale
    )

    quantized = torch.clamp(
        quantized,
        qmin,
        qmax,
    ).to(torch.int8)

    parameters = QuantizationParameters(

        scale=scale,

        zero_point=zero_point,

        qmin=qmin,

        qmax=qmax,

        mode="symmetric_per_tensor",

    )

    return quantized, parameters


# ======================================================================
# ASYMMETRIC QUANTIZATION
# ======================================================================

def asymmetric_quantize(
    x: torch.Tensor,
    qmin: int = INT8_MIN,
    qmax: int = INT8_MAX,
) -> tuple[torch.Tensor, QuantizationParameters]:

    """
    Asymmetric per-tensor quantization.

    scale = (xmax - xmin) / (qmax - qmin)

    zero_point = round(qmin - xmin / scale)
    """

    if not x.is_floating_point():

        raise TypeError(
            "Input must be floating point."
        )

    if not torch.isfinite(x).all():

        raise ValueError(
            "Input contains non-finite values."
        )

    xmin = torch.min(x)

    xmax = torch.max(x)

    scale = (
        xmax - xmin
    ) / (
        qmax - qmin
    )

    scale = torch.clamp(
        scale,
        min=1e-12,
    )

    zero_point = torch.round(
        qmin - xmin / scale
    )

    zero_point = torch.clamp(
        zero_point,
        qmin,
        qmax,
    ).to(torch.int32)

    quantized = torch.round(
        x / scale + zero_point.float()
    )

    quantized = torch.clamp(
        quantized,
        qmin,
        qmax,
    ).to(torch.int8)

    parameters = QuantizationParameters(

        scale=scale,

        zero_point=zero_point,

        qmin=qmin,

        qmax=qmax,

        mode="asymmetric_per_tensor",

    )

    return quantized, parameters


# ======================================================================
# DEQUANTIZATION
# ======================================================================

def dequantize(
    quantized: torch.Tensor,
    parameters: QuantizationParameters,
) -> torch.Tensor:

    if quantized.dtype != torch.int8:

        raise TypeError(
            "Expected torch.int8 tensor."
        )

    return (
        quantized.float()
        - parameters.zero_point.float()
    ) * parameters.scale.float()


# ======================================================================
# PER-CHANNEL QUANTIZATION
# ======================================================================

def symmetric_per_channel_quantize(
    weights: torch.Tensor,
    axis: int = 0,
) -> tuple[torch.Tensor, QuantizationParameters]:

    """
    Per-channel symmetric quantization.

    Example convolution weights:

        [out_channels, in_channels, kernel_h, kernel_w]

    axis=0:

        One scale per output channel.
    """

    if not weights.is_floating_point():

        raise TypeError(
            "Weights must be floating point."
        )

    if not torch.isfinite(weights).all():

        raise ValueError(
            "Weights contain non-finite values."
        )

    if not -weights.ndim <= axis < weights.ndim:

        raise ValueError(
            "Invalid quantization axis."
        )

    axis = axis % weights.ndim

    reduce_axes = tuple(

        index

        for index in range(weights.ndim)

        if index != axis

    )

    max_abs = torch.amax(
        torch.abs(weights),
        dim=reduce_axes,
        keepdim=True,
    )

    scale = torch.clamp(
        max_abs / 127.0,
        min=1e-12,
    )

    quantized = torch.round(
        weights / scale
    )

    quantized = torch.clamp(
        quantized,
        -127,
        127,
    ).to(torch.int8)

    parameters = QuantizationParameters(

        scale=scale,

        zero_point=torch.tensor(
            0,
            device=weights.device,
            dtype=torch.int32,
        ),

        qmin=-127,

        qmax=127,

        mode="symmetric_per_channel",

        axis=axis,

    )

    return quantized, parameters


# ======================================================================
# QUANTIZATION METRICS
# ======================================================================

def quantization_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict[str, float]:

    a = original.double()

    b = reconstructed.double()

    error = a - b

    absolute_error = torch.abs(
        error
    )

    mae = torch.mean(
        absolute_error
    )

    mse = torch.mean(
        error ** 2
    )

    rmse = torch.sqrt(mse)

    maximum_error = torch.max(
        absolute_error
    )

    relative_error = (
        torch.linalg.vector_norm(error)
        /
        torch.linalg.vector_norm(a).clamp_min(1e-12)
    )

    return {

        "mae": float(mae.item()),

        "mse": float(mse.item()),

        "rmse": float(rmse.item()),

        "max_error": float(maximum_error.item()),

        "relative_l2_error": float(
            relative_error.item()
        ),

    }


# ======================================================================
# QUANTIZATION ERROR ANALYSIS
# ======================================================================

def run_quantization_analysis(
    x: torch.Tensor,
) -> dict[str, Any]:

    print("\n" + "=" * 75)

    print("QUANTIZATION ANALYSIS")

    print("=" * 75)

    # Symmetric

    q_sym, params_sym = symmetric_quantize(x)

    reconstructed_sym = dequantize(
        q_sym,
        params_sym,
    )

    symmetric_metrics = quantization_metrics(
        x,
        reconstructed_sym,
    )

    # Asymmetric

    q_asym, params_asym = asymmetric_quantize(x)

    reconstructed_asym = dequantize(
        q_asym,
        params_asym,
    )

    asymmetric_metrics = quantization_metrics(
        x,
        reconstructed_asym,
    )

    print("\nSymmetric quantization:")

    print(
        f"  Scale: {params_sym.scale.item():.8e}"
    )

    print(
        f"  Zero point: {params_sym.zero_point.item()}"
    )

    print(
        f"  MAE: {symmetric_metrics['mae']:.8e}"
    )

    print(
        f"  RMSE: {symmetric_metrics['rmse']:.8e}"
    )

    print("\nAsymmetric quantization:")

    print(
        f"  Scale: {params_asym.scale.item():.8e}"
    )

    print(
        f"  Zero point: {params_asym.zero_point.item()}"
    )

    print(
        f"  MAE: {asymmetric_metrics['mae']:.8e}"
    )

    print(
        f"  RMSE: {asymmetric_metrics['rmse']:.8e}"
    )

    return {

        "symmetric": {

            "parameters": {

                "scale": float(
                    params_sym.scale.item()
                ),

                "zero_point": int(
                    params_sym.zero_point.item()
                ),

            },

            "metrics": symmetric_metrics,

        },

        "asymmetric": {

            "parameters": {

                "scale": float(
                    params_asym.scale.item()
                ),

                "zero_point": int(
                    params_asym.zero_point.item()
                ),

            },

            "metrics": asymmetric_metrics,

        },

    }


# ======================================================================
# CALIBRATION
# ======================================================================

class ActivationCalibrator:

    """
    Collect activation ranges for calibration.

    This is a simplified educational calibration implementation.

    It records activation statistics from selected layers.

    It does not automatically convert the entire model into a
    production-ready INT8 graph.
    """

    def __init__(
        self,
        model: nn.Module,
    ):

        self.model = model

        self.statistics = {}

        self.handles = []

    def _create_hook(
        self,
        name: str,
    ):

        def hook(
            module,
            inputs,
            output,
        ):

            if not isinstance(
                output,
                torch.Tensor,
            ):

                return

            if not output.is_floating_point():

                return

            values = output.detach()

            current_min = float(
                values.min().item()
            )

            current_max = float(
                values.max().item()
            )

            if name not in self.statistics:

                self.statistics[name] = {

                    "min": current_min,

                    "max": current_max,

                    "observations": 0,

                }

            else:

                self.statistics[name]["min"] = min(
                    self.statistics[name]["min"],
                    current_min,
                )

                self.statistics[name]["max"] = max(
                    self.statistics[name]["max"],
                    current_max,
                )

            self.statistics[name]["observations"] += 1

        return hook

    def attach(self) -> None:

        for name, module in self.model.named_modules():

            if isinstance(
                module,
                (
                    nn.Conv2d,
                    nn.Linear,
                    nn.ReLU,
                ),
            ):

                handle = module.register_forward_hook(
                    self._create_hook(name)
                )

                self.handles.append(handle)

    def remove(self) -> None:

        for handle in self.handles:

            handle.remove()

        self.handles.clear()

    def calibrate(
        self,
        batches: list[torch.Tensor],
        device: torch.device,
    ) -> dict[str, dict[str, float | int]]:

        self.model.eval()

        self.attach()

        try:

            with torch.inference_mode():

                for batch in batches:

                    batch = batch.to(device)

                    self.model(batch)

        finally:

            self.remove()

        return self.statistics


# ======================================================================
# PYTORCH DYNAMIC QUANTIZATION
# ======================================================================

def try_dynamic_quantization(
    model: nn.Module,
) -> tuple[nn.Module | None, str | None]:

    """
    Dynamic quantization demonstration.

    PyTorch dynamic quantization commonly supports nn.Linear
    and selected recurrent layers.

    This CNN's convolution layers remain floating point.

    Therefore, this is not a fully INT8 CNN conversion.
    """

    try:

        quantized_model = torch.ao.quantization.quantize_dynamic(

            copy.deepcopy(model).cpu().eval(),

            {
                nn.Linear,
            },

            dtype=torch.qint8,

        )

        return quantized_model, None

    except Exception as error:

        return None, str(error)


# ======================================================================
# BENCHMARK
# ======================================================================

@dataclass
class BenchmarkResult:

    mode: str

    batch_size: int

    mean_ms: float

    p50_ms: float

    p95_ms: float

    throughput_fps: float

    successful: bool

    error: str | None = None


def benchmark_model(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    mode: str,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:

    model.eval()

    batch_size = x.shape[0]

    try:

        for _ in range(warmup):

            with torch.inference_mode():

                output = model(x)

            del output

        synchronize(device)

        timings = []

        total_start = time.perf_counter()

        for _ in range(iterations):

            synchronize(device)

            start = time.perf_counter()

            with torch.inference_mode():

                output = model(x)

            synchronize(device)

            end = time.perf_counter()

            timings.append(
                (end - start) * 1000.0
            )

            del output

        total_end = time.perf_counter()

        total_seconds = (
            total_end - total_start
        )

        values = np.asarray(
            timings,
            dtype=np.float64,
        )

        throughput = (
            batch_size * iterations
            / total_seconds
        )

        return BenchmarkResult(

            mode=mode,

            batch_size=batch_size,

            mean_ms=float(
                np.mean(values)
            ),

            p50_ms=float(
                np.percentile(values, 50)
            ),

            p95_ms=float(
                np.percentile(values, 95)
            ),

            throughput_fps=float(
                throughput
            ),

            successful=True,

        )

    except Exception as error:

        return BenchmarkResult(

            mode=mode,

            batch_size=batch_size,

            mean_ms=0.0,

            p50_ms=0.0,

            p95_ms=0.0,

            throughput_fps=0.0,

            successful=False,

            error=str(error),

        )


# ======================================================================
# MODEL SIZE
# ======================================================================

def model_parameter_storage(
    model: nn.Module,
) -> int:

    total = 0

    for parameter in model.parameters():

        total += (
            parameter.numel()
            * parameter.element_size()
        )

    for buffer in model.buffers():

        total += (
            buffer.numel()
            * buffer.element_size()
        )

    return total


# ======================================================================
# VISUALIZATION
# ======================================================================

def plot_quantization_distribution(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    output_dir: Path,
) -> None:

    original_np = (
        original.detach().cpu().numpy().reshape(-1)
    )

    reconstructed_np = (
        reconstructed.detach().cpu().numpy().reshape(-1)
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.hist(
        original_np,
        bins=80,
        alpha=0.5,
        label="FP32",
    )

    ax.hist(
        reconstructed_np,
        bins=80,
        alpha=0.5,
        label="INT8 reconstructed",
    )

    ax.set_title(
        "Original and Quantized Value Distributions"
    )

    ax.set_xlabel("Value")

    ax.set_ylabel("Frequency")

    ax.legend()

    ax.grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        output_dir / "quantization_distribution.png",
        dpi=150,
    )

    plt.close(fig)


def plot_quantization_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    output_dir: Path,
) -> None:

    error = (
        original - reconstructed
    ).detach().cpu().numpy().reshape(-1)

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.hist(
        error,
        bins=80,
    )

    ax.set_title(
        "INT8 Quantization Error Distribution"
    )

    ax.set_xlabel("Quantization error")

    ax.set_ylabel("Frequency")

    ax.grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        output_dir / "quantization_error.png",
        dpi=150,
    )

    plt.close(fig)


def plot_benchmark(
    results: list[BenchmarkResult],
    output_dir: Path,
) -> None:

    successful = [
        item
        for item in results
        if item.successful
    ]

    if not successful:

        return

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    modes = sorted(
        set(
            item.mode
            for item in successful
        )
    )

    for mode in modes:

        subset = [

            item

            for item in successful

            if item.mode == mode

        ]

        ax.plot(
            [
                item.batch_size
                for item in subset
            ],
            [
                item.mean_ms
                for item in subset
            ],
            marker="o",
            label=mode,
        )

    ax.set_xlabel("Batch size")

    ax.set_ylabel("Mean latency (ms)")

    ax.set_title(
        "FP32 vs Dynamic INT8 Benchmark"
    )

    ax.grid(alpha=0.3)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "quantization_benchmark.png",
        dpi=150,
    )

    plt.close(fig)


# ======================================================================
# TESTS
# ======================================================================

def run_tests() -> None:

    print("\nRunning Lesson 74 tests...\n")

    # Test 1: Symmetric quantization.

    x = torch.tensor(
        [-1.0, -0.5, 0.0, 0.5, 1.0]
    )

    q, params = symmetric_quantize(x)

    assert q.dtype == torch.int8

    assert q.min() >= -128

    assert q.max() <= 127

    assert params.zero_point.item() == 0

    print("Test 1 passed: Symmetric quantization.")

    # Test 2: Dequantization.

    reconstructed = dequantize(
        q,
        params,
    )

    assert reconstructed.shape == x.shape

    assert torch.allclose(
        reconstructed,
        x,
        atol=0.01,
    )

    print("Test 2 passed: Dequantization.")

    # Test 3: Asymmetric quantization.

    x_asym = torch.tensor(
        [0.0, 1.0, 2.0, 3.0, 4.0]
    )

    q_asym, params_asym = asymmetric_quantize(
        x_asym
    )

    reconstructed_asym = dequantize(
        q_asym,
        params_asym,
    )

    assert q_asym.dtype == torch.int8

    assert torch.allclose(
        reconstructed_asym,
        x_asym,
        atol=0.02,
    )

    print("Test 3 passed: Asymmetric quantization.")

    # Test 4: Quantization bounds.

    random_tensor = torch.randn(
        1000
    ) * 5

    q_random, _ = symmetric_quantize(
        random_tensor
    )

    assert q_random.min() >= -128

    assert q_random.max() <= 127

    print("Test 4 passed: INT8 bounds.")

    # Test 5: Numerical error.

    metrics = quantization_metrics(
        x,
        reconstructed,
    )

    assert metrics["mae"] >= 0

    assert metrics["rmse"] >= 0

    assert metrics["max_error"] >= 0

    print("Test 5 passed: Quantization metrics.")

    # Test 6: Per-channel quantization.

    weights = torch.randn(
        8,
        4,
        3,
        3,
    )

    q_weights, channel_params = (
        symmetric_per_channel_quantize(
            weights,
            axis=0,
        )
    )

    assert q_weights.dtype == torch.int8

    assert channel_params.scale.shape == (
        8,
        1,
        1,
        1,
    )

    print("Test 6 passed: Per-channel quantization.")

    # Test 7: Per-channel reconstruction.

    reconstructed_weights = dequantize(
        q_weights,
        channel_params,
    )

    assert reconstructed_weights.shape == weights.shape

    print("Test 7 passed: Per-channel dequantization.")

    # Test 8: Storage comparison.

    fp32 = torch.zeros(
        1000,
        dtype=torch.float32,
    )

    int8 = torch.zeros(
        1000,
        dtype=torch.int8,
    )

    assert fp32.element_size() == 4

    assert int8.element_size() == 1

    assert (
        fp32.numel() * fp32.element_size()
        ==
        4 * int8.numel() * int8.element_size()
    )

    print("Test 8 passed: INT8 storage reduction.")

    # Test 9: Calibration.

    model = QuantizationCNN().eval()

    calibrator = ActivationCalibrator(
        model
    )

    batches = [

        torch.randn(
            2,
            3,
            64,
            64,
        ),

        torch.randn(
            2,
            3,
            64,
            64,
        ),

    ]

    statistics = calibrator.calibrate(
        batches,
        torch.device("cpu"),
    )

    assert len(statistics) > 0

    assert all(
        item["observations"] > 0
        for item in statistics.values()
    )

    print("Test 9 passed: Activation calibration.")

    # Test 10: CNN forward pass.

    with torch.inference_mode():

        output = model(
            torch.randn(
                2,
                3,
                64,
                64,
            )
        )

    assert output.shape == (2, 10)

    print("Test 10 passed: CNN inference.")

    # Test 11: Model storage.

    storage = model_parameter_storage(
        model
    )

    assert storage > 0

    print("Test 11 passed: Model storage calculation.")

    # Test 12: Dynamic quantization.

    quantized_model, error = try_dynamic_quantization(
        model
    )

    if quantized_model is not None:

        with torch.inference_mode():

            output = quantized_model(
                torch.randn(
                    2,
                    3,
                    64,
                    64,
                )
            )

        assert output.shape == (2, 10)

        print(
            "Test 12 passed: Dynamic quantization."
        )

    else:

        print(
            "Test 12 skipped: Backend unavailable:",
            error,
        )

    # Test 13: Input generation.

    image = create_input(
        4,
        64,
        None,
    )

    assert image.shape == (
        4,
        3,
        64,
        64,
    )

    print("Test 13 passed: Input generation.")

    # Test 14: Invalid integer input.

    try:

        symmetric_quantize(
            torch.tensor(
                [1, 2, 3],
                dtype=torch.int32,
            )
        )

        raise AssertionError(
            "Integer input should be rejected."
        )

    except TypeError:

        pass

    print("Test 14 passed: Input validation.")

    # Test 15: Quantization parameters are finite.

    assert torch.isfinite(
        params.scale
    ).all()

    assert params.scale.item() > 0

    print("Test 15 passed: Valid scale.")

    print("\nAll Lesson 74 tests passed.")


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Lesson 74 — INT8 Quantization"
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
            "Warmup cannot be negative."
        )

    if any(
        size <= 0
        for size in args.batch_sizes
    ):

        raise ValueError(
            "Batch sizes must be positive."
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

    torch.manual_seed(42)

    np.random.seed(42)

    print("\n" + "=" * 75)

    print("LESSON 74 — INT8 QUANTIZATION")

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

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(device)}"
        )

    # --------------------------------------------------------------
    # REAL IMAGE
    # --------------------------------------------------------------

    base_image = None

    if args.image:

        base_image = load_real_image(
            args.image,
            args.image_size,
        )

        print(
            f"Real image: {args.image}"
        )

    # --------------------------------------------------------------
    # MODEL
    # --------------------------------------------------------------

    model = QuantizationCNN()

    model.eval()

    model = model.to(
        device
    )

    fp32_storage = model_parameter_storage(
        model
    )

    print(
        f"\nFP32 model storage: "
        f"{fp32_storage / (1024 ** 2):.2f} MB"
    )

    # --------------------------------------------------------------
    # WEIGHT QUANTIZATION DEMONSTRATION
    # --------------------------------------------------------------

    first_conv = model.features[0].conv.weight.detach()

    q_weights, weight_params = (
        symmetric_per_channel_quantize(
            first_conv,
            axis=0,
        )
    )

    reconstructed_weights = dequantize(
        q_weights,
        weight_params,
    )

    weight_metrics = quantization_metrics(
        first_conv,
        reconstructed_weights,
    )

    print("\nFirst convolution weight quantization:")

    print(
        f"Original storage: "
        f"{first_conv.numel() * first_conv.element_size()} bytes"
    )

    print(
        f"INT8 storage: "
        f"{q_weights.numel() * q_weights.element_size()} bytes"
    )

    print(
        f"Weight MAE: "
        f"{weight_metrics['mae']:.8e}"
    )

    # --------------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------------

    calibration_batches = [

        create_input(
            batch_size=2,
            image_size=args.image_size,
            base_image=base_image,
        ),

        create_input(
            batch_size=2,
            image_size=args.image_size,
            base_image=base_image,
        ),

        create_input(
            batch_size=2,
            image_size=args.image_size,
            base_image=base_image,
        ),

    ]

    calibration_batches = [

        preprocess(batch).to(device)

        for batch in calibration_batches

    ]

    calibrator = ActivationCalibrator(
        model
    )

    calibration_statistics = calibrator.calibrate(
        calibration_batches,
        device,
    )

    print("\nCalibration statistics:")

    for name, statistics in calibration_statistics.items():

        print(
            f"  {name}: "
            f"min={statistics['min']:.5f}, "
            f"max={statistics['max']:.5f}, "
            f"observations={statistics['observations']}"
        )

    # --------------------------------------------------------------
    # QUANTIZATION ERROR
    # --------------------------------------------------------------

    analysis_input = torch.randn(
        1,
        3,
        128,
        128,
    )

    quantization_report = run_quantization_analysis(
        analysis_input
    )

    q, params = symmetric_quantize(
        analysis_input
    )

    reconstructed = dequantize(
        q,
        params,
    )

    plot_quantization_distribution(
        analysis_input,
        reconstructed,
        output_dir,
    )

    plot_quantization_error(
        analysis_input,
        reconstructed,
        output_dir,
    )

    # --------------------------------------------------------------
    # DYNAMIC QUANTIZATION
    # --------------------------------------------------------------

    dynamic_model, dynamic_error = try_dynamic_quantization(
        model
    )

    dynamic_available = (
        dynamic_model is not None
    )

    if dynamic_available:

        print(
            "\nDynamic quantization succeeded."
        )

        print(
            "Important: this quantizes supported Linear layers; "
            "the CNN convolution layers remain floating point."
        )

    else:

        print(
            "\nDynamic quantization unavailable:",
            dynamic_error,
        )

    # --------------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------------

    benchmark_results = []

    for batch_size in sorted(
        set(args.batch_sizes)
    ):

        x = create_input(
            batch_size,
            args.image_size,
            base_image,
        )

        x = preprocess(x).to(device)

        # FP32

        fp32_result = benchmark_model(
            model=model,
            x=x,
            device=device,
            mode="fp32",
            warmup=args.warmup,
            iterations=args.iterations,
        )

        benchmark_results.append(
            fp32_result
        )

        # Dynamic quantized model is CPU-oriented.

        if dynamic_available:

            x_cpu = x.cpu()

            dynamic_result = benchmark_model(
                model=dynamic_model,
                x=x_cpu,
                device=torch.device("cpu"),
                mode="dynamic_int8_linear",
                warmup=args.warmup,
                iterations=args.iterations,
            )

            benchmark_results.append(
                dynamic_result
            )

        del x

    # --------------------------------------------------------------
    # PRINT BENCHMARK
    # --------------------------------------------------------------

    print("\n" + "=" * 90)

    print("INFERENCE BENCHMARK")

    print("=" * 90)

    print(
        f"{'Mode':>25}"
        f"{'Batch':>8}"
        f"{'Mean ms':>14}"
        f"{'P95 ms':>14}"
        f"{'FPS':>14}"
        f"{'Status':>12}"
    )

    print("-" * 90)

    for result in benchmark_results:

        print(
            f"{result.mode:>25}"
            f"{result.batch_size:>8}"
            f"{result.mean_ms:>14.4f}"
            f"{result.p95_ms:>14.4f}"
            f"{result.throughput_fps:>14.2f}"
            f"{'OK' if result.successful else 'FAILED':>12}"
        )

    print("=" * 90)

    # --------------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------------

    plot_benchmark(
        benchmark_results,
        output_dir,
    )

    # --------------------------------------------------------------
    # JSON
    # --------------------------------------------------------------

    report = {

        "lesson": 74,

        "title": "INT8 Quantization",

        "device": str(device),

        "image_size": args.image_size,

        "fp32_model_storage_bytes": fp32_storage,

        "dynamic_quantization_available": dynamic_available,

        "dynamic_quantization_error": dynamic_error,

        "weight_quantization": {

            "mode": weight_params.mode,

            "axis": weight_params.axis,

            "scale_shape": list(
                weight_params.scale.shape
            ),

            "metrics": weight_metrics,

        },

        "quantization_analysis": quantization_report,

        "calibration": calibration_statistics,

        "benchmarks": [

            {

                "mode": item.mode,

                "batch_size": item.batch_size,

                "mean_ms": item.mean_ms,

                "p50_ms": item.p50_ms,

                "p95_ms": item.p95_ms,

                "throughput_fps": item.throughput_fps,

                "successful": item.successful,

                "error": item.error,

            }

            for item in benchmark_results

        ],

    }

    report_path = (
        output_dir / "int8_quantization_report.json"
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

    print("\nGenerated files:")

    for path in sorted(
        output_dir.iterdir()
    ):

        print(
            f"  {path}"
        )

    print("\nLesson 74 completed.")

    print(
        "\nNext: Lesson 75 — Model Pruning."
    )


if __name__ == "__main__":

    main()