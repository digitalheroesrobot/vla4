"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 6 — MODEL OPTIMIZATION AND DEPLOYMENT

LESSON 71 — CPU VS GPU INFERENCE
======================================================================

OBJECTIVE
---------

Understand and experimentally compare neural network inference
performance on CPU and GPU.

We will investigate:

    1. CPU inference.
    2. GPU inference.
    3. Model initialization and weight consistency.
    4. CPU and GPU warmup.
    5. CPU/GPU synchronization.
    6. Host-to-device memory transfer.
    7. Compute-only latency.
    8. End-to-end inference latency.
    9. Batch-size effects.
   10. Throughput.
   11. GPU memory usage.

IMPORTANT
---------

This lesson uses a randomly initialized CNN.

We are studying computational performance, not classification
accuracy.

The same model architecture and weights are used for CPU and GPU.

A GPU is not automatically faster for every workload.

For example:

    - Small models may execute faster on CPU.
    - Small batches may not fully utilize a GPU.
    - Data transfer can reduce GPU performance.
    - Large batches can improve GPU utilization.
    - Large images generally increase computational cost.

======================================================================
1. CPU COMPUTATION
======================================================================

A CPU typically has:

    - A relatively small number of powerful cores.
    - Large caches.
    - Complex control logic.
    - Strong performance for sequential workloads.
    - Efficient execution of many small operations.

CPUs are useful for:

    - Image preprocessing.
    - Lightweight inference.
    - Data loading.
    - Control logic.
    - Low-volume inference requests.

======================================================================
2. GPU COMPUTATION
======================================================================

A GPU typically has:

    - Many parallel processing units.
    - High memory bandwidth.
    - Specialized matrix multiplication hardware.
    - Efficient parallel tensor operations.

GPUs are useful for:

    - Large CNNs.
    - Transformer inference.
    - Large batches.
    - High-resolution image processing.
    - Multiple simultaneous inference requests.

======================================================================
3. HOST-TO-DEVICE TRANSFER
======================================================================

A GPU generally has its own memory.

CPU memory:

    Host memory (RAM)

GPU memory:

    Device memory (VRAM)

Transferring an image:

    CPU RAM
       |
       | H2D transfer
       v
    GPU VRAM
       |
       v
    GPU inference
       |
       v
    GPU output

The transfer itself takes time.

For a small model, the transfer cost can be significant compared
with the actual inference time.

======================================================================
4. COMPUTE-ONLY VS END-TO-END
======================================================================

COMPUTE-ONLY:

    Input already resides on the target device.

    CPU:
        CPU input -> CPU model

    GPU:
        GPU input -> GPU model

END-TO-END:

    Input starts in CPU memory.

    CPU:
        CPU input -> preprocessing -> CPU inference

    GPU:
        CPU input -> preprocessing -> H2D -> GPU inference

For a realistic GPU pipeline, transfer costs must not be ignored.

This lesson measures both.

======================================================================
5. CUDA SYNCHRONIZATION
======================================================================

CUDA operations are asynchronous.

For example:

    output = model(input)

The Python call may return before the GPU finishes processing.

Therefore, this is not sufficient for accurate wall-clock timing:

    start = time.perf_counter()
    output = model(input)
    elapsed = time.perf_counter() - start

Instead:

    torch.cuda.synchronize()

    start = time.perf_counter()

    output = model(input)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

======================================================================
6. THROUGHPUT AND LATENCY
======================================================================

Latency:

    Time required to process a request.

Throughput:

    Number of images processed per second.

For batch size B:

    Throughput = B / batch_latency_seconds

Example:

    Batch size = 8
    Batch latency = 20 ms

    Throughput = 8 / 0.020

               = 400 images/second

But this does NOT mean one image has 2.5 ms end-to-end latency
in a live system.

Batching introduces waiting time and scheduling considerations.

======================================================================
7. REAL-WORLD ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste-sorting pipeline:

    RGB Camera
         |
         v
    CPU image capture
         |
         v
    Image preprocessing
         |
         v
    CPU RAM -> GPU VRAM
         |
         v
    Object Detection
         |
         v
    Object Localization
         |
         v
    Grasp Planning
         |
         v
    ACT Policy
         |
         v
    Robot Control

For real-time robotics, we care about:

    - End-to-end latency.
    - Consistent response time.
    - GPU utilization.
    - Memory transfer.
    - CPU/GPU workload distribution.

======================================================================
WHAT YOU WILL LEARN
======================================================================

After this lesson, you should understand:

    - How to compare CPU and GPU fairly.
    - Why warmup matters.
    - Why CUDA synchronization matters.
    - Why GPU memory transfer matters.
    - Why batch size changes throughput.
    - When CPU inference can be preferable.
    - How to interpret device benchmarking results.

NEXT LESSON
-----------

Lesson 72 — Batch Size Optimization.

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
import torch.nn.functional as F

import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

DEFAULT_IMAGE_SIZE = 224

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16]

DEFAULT_WARMUP = 20

DEFAULT_ITERATIONS = 100

DEFAULT_OUTPUT_DIR = Path("runs/lesson71")


# ======================================================================
# DEVICE UTILITIES
# ======================================================================

def synchronize(device: torch.device) -> None:

    """
    Ensure all queued CUDA operations have finished.

    CPU operations are synchronous.
    """

    if device.type == "cuda":

        torch.cuda.synchronize(device)


def get_available_devices(
    requested_device: str,
) -> list[torch.device]:

    if requested_device == "cpu":

        return [torch.device("cpu")]

    if requested_device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested but no CUDA device is available."
            )

        return [torch.device("cuda")]

    if requested_device == "auto":

        if torch.cuda.is_available():

            return [
                torch.device("cpu"),
                torch.device("cuda"),
            ]

        return [torch.device("cpu")]

    raise ValueError(
        f"Unknown device selection: {requested_device}"
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

        self.bn = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        x = self.activation(x)

        return x


class BenchmarkCNN(nn.Module):

    """
    CNN used to compare CPU and GPU inference.

    Input:
        B x 3 x H x W

    Output:
        B x 10

    Architecture:

        Conv(3, 32)
            |
        Conv(32, 64)
            |
        Conv(64, 128)
            |
        Conv(128, 256)
            |
        Conv(256, 256)
            |
        Global Average Pooling
            |
        Linear(256, 10)
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
# IMAGE INPUT
# ======================================================================

def load_image(
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


# ======================================================================
# PREPROCESSING
# ======================================================================

def preprocess(
    tensor: torch.Tensor,
) -> torch.Tensor:

    """
    Simple normalization.

    Input:
        [0, 1]

    Output:
        [-1, 1]
    """

    return tensor * 2.0 - 1.0


# ======================================================================
# BENCHMARK RESULT
# ======================================================================

@dataclass
class BenchmarkResult:

    device_name: str

    batch_size: int

    compute_latencies_ms: list[float]

    end_to_end_latencies_ms: list[float]

    compute_throughput_fps: float

    end_to_end_throughput_fps: float

    transfer_latencies_ms: list[float]

    @property
    def compute_mean_ms(self):

        return float(
            np.mean(self.compute_latencies_ms)
        )

    @property
    def compute_p95_ms(self):

        return float(
            np.percentile(
                self.compute_latencies_ms,
                95,
            )
        )

    @property
    def end_to_end_mean_ms(self):

        return float(
            np.mean(self.end_to_end_latencies_ms)
        )

    @property
    def end_to_end_p95_ms(self):

        return float(
            np.percentile(
                self.end_to_end_latencies_ms,
                95,
            )
        )

    @property
    def transfer_mean_ms(self):

        if not self.transfer_latencies_ms:

            return 0.0

        return float(
            np.mean(self.transfer_latencies_ms)
        )


# ======================================================================
# WARMUP
# ======================================================================

def warmup_model(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup_iterations: int,
) -> None:

    model.eval()

    with torch.inference_mode():

        for _ in range(warmup_iterations):

            model(input_tensor)

    synchronize(device)


# ======================================================================
# COMPUTE-ONLY BENCHMARK
# ======================================================================

def benchmark_compute_only(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    iterations: int,
) -> list[float]:

    """
    Measure model execution with the input already on the target device.

    Does not include host-to-device transfer.
    """

    latencies = []

    with torch.inference_mode():

        for _ in range(iterations):

            synchronize(device)

            start = time.perf_counter()

            model(input_tensor)

            synchronize(device)

            end = time.perf_counter()

            latencies.append(
                (end - start) * 1000.0
            )

    return latencies


# ======================================================================
# TRANSFER BENCHMARK
# ======================================================================

def benchmark_transfer(
    input_cpu: torch.Tensor,
    device: torch.device,
    iterations: int,
) -> list[float]:

    """
    Measure CPU -> target-device transfer.

    CPU:
        Transfer is effectively a no-op.

    CUDA:
        Measures blocking CPU-to-GPU transfer.

    This is a simple baseline, not an optimized asynchronous
    transfer pipeline.
    """

    latencies = []

    for _ in range(iterations):

        synchronize(device)

        start = time.perf_counter()

        input_device = input_cpu.to(
            device,
            non_blocking=False,
        )

        synchronize(device)

        end = time.perf_counter()

        latencies.append(
            (end - start) * 1000.0
        )

        del input_device

    return latencies


# ======================================================================
# END-TO-END BENCHMARK
# ======================================================================

def benchmark_end_to_end(
    model: nn.Module,
    input_cpu: torch.Tensor,
    device: torch.device,
    iterations: int,
) -> list[float]:

    """
    Measure:

        CPU input
            |
            v
        Device transfer
            |
            v
        Preprocessing
            |
            v
        Model inference
            |
            v
        Output

    The input begins in CPU memory for both devices.
    """

    latencies = []

    with torch.inference_mode():

        for _ in range(iterations):

            synchronize(device)

            start = time.perf_counter()

            input_device = input_cpu.to(
                device,
                non_blocking=False,
            )

            processed = preprocess(
                input_device
            )

            output = model(
                processed
            )

            synchronize(device)

            end = time.perf_counter()

            latencies.append(
                (end - start) * 1000.0
            )

            del input_device
            del processed
            del output

    return latencies


# ======================================================================
# FULL DEVICE BENCHMARK
# ======================================================================

def benchmark_device(
    model: nn.Module,
    input_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int,
    warmup_iterations: int,
    iterations: int,
) -> BenchmarkResult:

    model_device = model.to(device)

    model_device.eval()

    input_device = input_cpu.to(device)

    # Warm up both model execution and the complete path.

    warmup_model(
        model_device,
        preprocess(input_device),
        device,
        warmup_iterations,
    )

    # Warmup end-to-end transfer path.

    for _ in range(min(warmup_iterations, 10)):

        with torch.inference_mode():

            x = input_cpu.to(device)

            model_device(
                preprocess(x)
            )

    synchronize(device)

    # Compute-only.

    compute_latencies = benchmark_compute_only(
        model_device,
        preprocess(input_device),
        device,
        iterations,
    )

    # Transfer-only.

    transfer_latencies = benchmark_transfer(
        input_cpu,
        device,
        iterations,
    )

    # End-to-end.

    end_to_end_latencies = benchmark_end_to_end(
        model_device,
        input_cpu,
        device,
        iterations,
    )

    # Throughput is measured separately from individual request
    # latency to reduce per-iteration timer overhead.

    with torch.inference_mode():

        synchronize(device)

        start = time.perf_counter()

        for _ in range(iterations):

            model_device(
                preprocess(input_device)
            )

        synchronize(device)

        end = time.perf_counter()

    compute_elapsed = end - start

    compute_throughput = (
        batch_size * iterations / compute_elapsed
    )

    with torch.inference_mode():

        synchronize(device)

        start = time.perf_counter()

        for _ in range(iterations):

            x = input_cpu.to(device)

            model_device(
                preprocess(x)
            )

        synchronize(device)

        end = time.perf_counter()

    end_to_end_elapsed = end - start

    end_to_end_throughput = (
        batch_size * iterations / end_to_end_elapsed
    )

    return BenchmarkResult(
        device_name=str(device),
        batch_size=batch_size,
        compute_latencies_ms=compute_latencies,
        end_to_end_latencies_ms=end_to_end_latencies,
        compute_throughput_fps=compute_throughput,
        end_to_end_throughput_fps=end_to_end_throughput,
        transfer_latencies_ms=transfer_latencies,
    )


# ======================================================================
# DEVICE INFORMATION
# ======================================================================

def get_device_information(
    device: torch.device,
) -> dict[str, Any]:

    if device.type == "cpu":

        return {
            "device": "CPU",
            "processor": platform.processor(),
            "cpu_threads": torch.get_num_threads(),
            "inter_op_threads": torch.get_num_interop_threads(),
        }

    properties = torch.cuda.get_device_properties(
        device
    )

    return {
        "device": "CUDA GPU",
        "name": properties.name,
        "total_memory_gb": (
            properties.total_memory / (1024 ** 3)
        ),
        "multiprocessors": properties.multi_processor_count,
        "compute_capability": (
            f"{properties.major}.{properties.minor}"
        ),
    }


# ======================================================================
# GPU MEMORY
# ======================================================================

def get_gpu_memory(
    device: torch.device,
) -> dict[str, float | None]:

    if device.type != "cuda":

        return {
            "allocated_mb": None,
            "reserved_mb": None,
            "peak_allocated_mb": None,
        }

    torch.cuda.synchronize(device)

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
    }


# ======================================================================
# REPORT
# ======================================================================

def print_result(
    result: BenchmarkResult,
) -> None:

    print("\n" + "-" * 65)

    print(
        f"Device: {result.device_name}"
    )

    print(
        f"Batch size: {result.batch_size}"
    )

    print("-" * 65)

    print(
        f"Compute-only mean: "
        f"{result.compute_mean_ms:.3f} ms"
    )

    print(
        f"Compute-only P95: "
        f"{result.compute_p95_ms:.3f} ms"
    )

    print(
        f"Compute-only throughput: "
        f"{result.compute_throughput_fps:.2f} images/s"
    )

    print(
        f"Transfer mean: "
        f"{result.transfer_mean_ms:.3f} ms"
    )

    print(
        f"End-to-end mean: "
        f"{result.end_to_end_mean_ms:.3f} ms"
    )

    print(
        f"End-to-end P95: "
        f"{result.end_to_end_p95_ms:.3f} ms"
    )

    print(
        f"End-to-end throughput: "
        f"{result.end_to_end_throughput_fps:.2f} images/s"
    )


# ======================================================================
# VISUALIZATION
# ======================================================================

def plot_comparison(
    results: dict[str, list[BenchmarkResult]],
    output_dir: Path,
) -> None:

    """
    Compare CPU and GPU across batch sizes.

    GPU plots are omitted if CUDA is unavailable.
    """

    device_names = list(results.keys())

    batch_sizes = [
        result.batch_size
        for result in results[device_names[0]]
    ]

    # --------------------------------------------------------------
    # COMPUTE LATENCY
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for device_name, device_results in results.items():

        values = [
            result.compute_mean_ms
            for result in device_results
        ]

        ax.plot(
            batch_sizes,
            values,
            marker="o",
            label=device_name,
        )

    ax.set_xlabel("Batch size")

    ax.set_ylabel("Compute-only latency (ms)")

    ax.set_title("CPU vs GPU — Compute Latency")

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "compute_latency_comparison.png",
        dpi=150,
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # END-TO-END LATENCY
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for device_name, device_results in results.items():

        values = [
            result.end_to_end_mean_ms
            for result in device_results
        ]

        ax.plot(
            batch_sizes,
            values,
            marker="o",
            label=device_name,
        )

    ax.set_xlabel("Batch size")

    ax.set_ylabel("End-to-end latency (ms)")

    ax.set_title("CPU vs GPU — End-to-End Latency")

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "end_to_end_latency_comparison.png",
        dpi=150,
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # THROUGHPUT
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for device_name, device_results in results.items():

        values = [
            result.compute_throughput_fps
            for result in device_results
        ]

        ax.plot(
            batch_sizes,
            values,
            marker="o",
            label=device_name,
        )

    ax.set_xlabel("Batch size")

    ax.set_ylabel("Throughput (images/second)")

    ax.set_title("CPU vs GPU — Throughput")

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "throughput_comparison.png",
        dpi=150,
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # TRANSFER COST
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for device_name, device_results in results.items():

        values = [
            result.transfer_mean_ms
            for result in device_results
        ]

        ax.plot(
            batch_sizes,
            values,
            marker="o",
            label=device_name,
        )

    ax.set_xlabel("Batch size")

    ax.set_ylabel("Transfer latency (ms)")

    ax.set_title("CPU vs GPU — Input Transfer Cost")

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "transfer_latency_comparison.png",
        dpi=150,
    )

    plt.close(fig)


# ======================================================================
# JSON SERIALIZATION
# ======================================================================

def build_report(
    results: dict[str, list[BenchmarkResult]],
    device_information: dict[str, Any],
    image_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:

    report = {

        "lesson": 71,

        "title": "CPU vs GPU Inference",

        "image_size": image_size,

        "warmup_iterations": warmup,

        "benchmark_iterations": iterations,

        "devices": device_information,

        "results": {},

    }

    for device_name, device_results in results.items():

        report["results"][device_name] = []

        for result in device_results:

            report["results"][device_name].append({

                "batch_size": result.batch_size,

                "compute_mean_ms": result.compute_mean_ms,

                "compute_p95_ms": result.compute_p95_ms,

                "compute_throughput_fps": (
                    result.compute_throughput_fps
                ),

                "transfer_mean_ms": result.transfer_mean_ms,

                "end_to_end_mean_ms": (
                    result.end_to_end_mean_ms
                ),

                "end_to_end_p95_ms": (
                    result.end_to_end_p95_ms
                ),

                "end_to_end_throughput_fps": (
                    result.end_to_end_throughput_fps
                ),

            })

    return report


# ======================================================================
# NUMERICAL TESTS
# ======================================================================

def run_tests() -> None:

    print("\nRunning Lesson 71 tests...\n")

    # Test 1: CNN output shape.

    model = BenchmarkCNN()

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.inference_mode():

        output = model(x)

    assert output.shape == (2, 10)

    print("Test 1 passed: CNN output shape.")

    # Test 2: CPU inference.

    cpu_model = model.to("cpu")

    cpu_output = cpu_model(x)

    assert cpu_output.shape == (2, 10)

    print("Test 2 passed: CPU inference.")

    # Test 3: CPU/GPU weight consistency.

    state_dict = model.state_dict()

    cloned_model = BenchmarkCNN()

    cloned_model.load_state_dict(state_dict)

    for key in state_dict:

        assert torch.equal(
            state_dict[key].cpu(),
            cloned_model.state_dict()[key].cpu(),
        )

    print("Test 3 passed: Model weights can be copied exactly.")

    # Test 4: Preprocessing.

    sample = torch.tensor(
        [0.0, 0.5, 1.0]
    )

    processed = preprocess(sample)

    expected = torch.tensor(
        [-1.0, 0.0, 1.0]
    )

    assert torch.allclose(
        processed,
        expected,
        atol=1e-6,
    )

    print("Test 4 passed: Preprocessing.")

    # Test 5: Input generation.

    generated = create_input(
        batch_size=4,
        image_size=64,
        base_image=None,
    )

    assert generated.shape == (
        4,
        3,
        64,
        64,
    )

    print("Test 5 passed: Input generation.")

    # Test 6: Batch replication.

    base = torch.rand(
        1,
        3,
        32,
        32,
    )

    replicated = create_input(
        batch_size=5,
        image_size=32,
        base_image=base,
    )

    assert replicated.shape == (
        5,
        3,
        32,
        32,
    )

    assert torch.allclose(
        replicated[0],
        replicated[4],
    )

    print("Test 6 passed: Batch replication.")

    # Test 7: CPU transfer.

    cpu_input = torch.rand(
        2,
        3,
        32,
        32,
    )

    cpu_device_input = cpu_input.to(
        torch.device("cpu")
    )

    assert torch.equal(
        cpu_input,
        cpu_device_input,
    )

    print("Test 7 passed: CPU device transfer.")

    # Test 8: Benchmark output.

    small_model = nn.Linear(8, 4)

    small_input = torch.rand(
        1,
        8,
    )

    latencies = benchmark_compute_only(
        small_model,
        small_input,
        torch.device("cpu"),
        iterations=5,
    )

    assert len(latencies) == 5

    assert all(
        latency >= 0
        for latency in latencies
    )

    print("Test 8 passed: Compute benchmark.")

    # Test 9: End-to-end CPU execution.

    end_to_end = benchmark_end_to_end(
        small_model,
        small_input,
        torch.device("cpu"),
        iterations=5,
    )

    assert len(end_to_end) == 5

    assert all(
        latency >= 0
        for latency in end_to_end
    )

    print("Test 9 passed: End-to-end benchmark.")

    # Test 10: CPU memory report.

    memory = get_gpu_memory(
        torch.device("cpu")
    )

    assert memory["allocated_mb"] is None

    print("Test 10 passed: CPU memory reporting.")

    # Test 11: Device enumeration.

    devices = get_available_devices("cpu")

    assert len(devices) == 1

    assert devices[0].type == "cpu"

    print("Test 11 passed: Device enumeration.")

    # Test 12: Model parameter consistency.

    original_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    cloned_parameters = sum(
        p.numel()
        for p in cloned_model.parameters()
    )

    assert original_parameters == cloned_parameters

    print("Test 12 passed: Parameter count consistency.")

    # Test 13: GPU test, if available.

    if torch.cuda.is_available():

        gpu_model = BenchmarkCNN().cuda()

        gpu_model.load_state_dict(
            model.state_dict()
        )

        gpu_input = x.cuda()

        with torch.inference_mode():

            gpu_output = gpu_model(gpu_input)

        assert gpu_output.shape == (2, 10)

        print("Test 13 passed: CUDA inference.")

    else:

        print(
            "Test 13 skipped: CUDA is unavailable."
        )

    print("\nAll available tests passed.")


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Lesson 71 — CPU vs GPU Inference"
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
        "--cpu-threads",
        type=int,
        default=None,
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

    if not args.batch_sizes or any(
        batch <= 0
        for batch in args.batch_sizes
    ):

        raise ValueError(
            "Batch sizes must be positive."
        )

    if args.cpu_threads is not None:

        if args.cpu_threads <= 0:

            raise ValueError(
                "--cpu-threads must be positive."
            )

        torch.set_num_threads(
            args.cpu_threads
        )

    # --------------------------------------------------------------
    # DEVICE SELECTION
    # --------------------------------------------------------------

    devices = get_available_devices(
        args.device
    )

    print("\n" + "=" * 75)

    print("LESSON 71 — CPU VS GPU INFERENCE")

    print("=" * 75)

    print(
        f"PyTorch version: {torch.__version__}"
    )

    print(
        f"Python version: {sys.version.split()[0]}"
    )

    print(
        f"Image size: {args.image_size} x {args.image_size}"
    )

    print(
        f"CPU threads: {torch.get_num_threads()}"
    )

    # --------------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------------

    base_image = None

    if args.image:

        base_image = load_image(
            args.image,
            args.image_size,
        )

        print(
            f"\nReal image loaded: {args.image}"
        )

    else:

        print(
            "\nNo real image supplied. "
            "Using numerical benchmarking tensors."
        )

    # --------------------------------------------------------------
    # IDENTICAL MODEL WEIGHTS
    # --------------------------------------------------------------

    torch.manual_seed(42)

    np.random.seed(42)

    reference_model = BenchmarkCNN()

    reference_model.eval()

    reference_state = {
        key: value.detach().cpu().clone()
        for key, value in reference_model.state_dict().items()
    }

    # --------------------------------------------------------------
    # BENCHMARK ALL DEVICES
    # --------------------------------------------------------------

    all_results: dict[str, list[BenchmarkResult]] = {}

    device_information = {}

    for device in devices:

        device_name = str(device)

        print("\n" + "=" * 75)

        print(
            f"DEVICE: {device_name.upper()}"
        )

        print("=" * 75)

        device_information[device_name] = (
            get_device_information(device)
        )

        print(
            json.dumps(
                device_information[device_name],
                indent=4,
            )
        )

        model = BenchmarkCNN()

        model.load_state_dict(
            reference_state
        )

        model = model.to(device)

        model.eval()

        all_results[device_name] = []

        for batch_size in args.batch_sizes:

            print("\n" + "-" * 65)

            print(
                f"Testing {device_name}, "
                f"batch size {batch_size}"
            )

            print("-" * 65)

            input_cpu = create_input(
                batch_size=batch_size,
                image_size=args.image_size,
                base_image=base_image,
            )

            if device.type == "cuda":

                torch.cuda.reset_peak_memory_stats(
                    device
                )

            result = benchmark_device(
                model=model,
                input_cpu=input_cpu,
                device=device,
                batch_size=batch_size,
                warmup_iterations=args.warmup,
                iterations=args.iterations,
            )

            print_result(result)

            all_results[device_name].append(
                result
            )

            if device.type == "cuda":

                memory = get_gpu_memory(
                    device
                )

                print("\nGPU MEMORY")

                for name, value in memory.items():

                    print(
                        f"{name}: {value:.3f} MB"
                    )

            del input_cpu

        del model

        if device.type == "cuda":

            torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------------

    print("\n" + "=" * 75)

    print("CPU VS GPU COMPARISON")

    print("=" * 75)

    if (
        "cpu" in all_results
        and "cuda" in all_results
    ):

        print(
            "\nGPU speedup based on compute-only latency:"
        )

        cpu_results = all_results["cpu"]

        gpu_results = all_results["cuda"]

        for cpu_result, gpu_result in zip(
            cpu_results,
            gpu_results,
        ):

            if cpu_result.batch_size != gpu_result.batch_size:

                continue

            speedup = (
                cpu_result.compute_mean_ms
                / max(gpu_result.compute_mean_ms, 1e-12)
            )

            end_to_end_speedup = (
                cpu_result.end_to_end_mean_ms
                / max(gpu_result.end_to_end_mean_ms, 1e-12)
            )

            print(
                f"\nBatch size {cpu_result.batch_size}"
            )

            print(
                f"  Compute-only GPU speedup: "
                f"{speedup:.2f}x"
            )

            print(
                f"  End-to-end GPU speedup: "
                f"{end_to_end_speedup:.2f}x"
            )

    else:

        print(
            "\nOnly one device was benchmarked. "
            "Run with --device auto on a CUDA-enabled system "
            "to compare CPU and GPU."
        )

    # --------------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------------

    print("\nGenerating comparison charts...")

    plot_comparison(
        all_results,
        output_dir,
    )

    # --------------------------------------------------------------
    # JSON
    # --------------------------------------------------------------

    report = build_report(
        results=all_results,
        device_information=device_information,
        image_size=args.image_size,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    report_path = (
        output_dir / "cpu_gpu_comparison.json"
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

    print("LESSON 71 COMPLETED")

    print("=" * 75)

    print("\nGenerated files:")

    for path in sorted(output_dir.iterdir()):

        print(
            f"  {path}"
        )

    print("\nKey takeaways:")

    print(
        "1. CPU and GPU use the same architecture and weights."
    )

    print(
        "2. Compute-only latency excludes input transfer."
    )

    print(
        "3. End-to-end latency includes CPU-to-device transfer."
    )

    print(
        "4. GPU synchronization is essential for correct timing."
    )

    print(
        "5. Larger batches may improve throughput."
    )

    print(
        "6. GPU performance depends on workload and hardware."
    )

    print(
        "\nNext: Lesson 72 — Batch Size Optimization."
    )


if __name__ == "__main__":

    main()