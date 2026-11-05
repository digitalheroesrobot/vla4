"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 6 — MODEL OPTIMIZATION AND DEPLOYMENT

LESSON 83 — MODEL BENCHMARKING
======================================================================

LEARNING OBJECTIVES
-------------------

1. Build a reproducible inference benchmarking framework.
2. Understand warm-up and measured inference.
3. Measure latency and throughput.
4. Calculate P50, P90, P95, and P99 latency.
5. Compare different batch sizes.
6. Compare CPU thread configurations.
7. Compare available ONNX Runtime execution providers.
8. Measure benchmark stability.
9. Validate numerical consistency across configurations.
10. Generate benchmark reports and visualizations.

NO pretrained models are created by this lesson.

A supplied ONNX model may contain trained weights.
The fallback model is randomly initialized and is used only
to demonstrate benchmarking mechanics.

======================================================================
1. WHY BENCHMARKING?
======================================================================

Model optimization is an experimental process.

Suppose we have three configurations:

    Configuration A:
        Batch size = 1
        CPU threads = 1

    Configuration B:
        Batch size = 4
        CPU threads = 2

    Configuration C:
        Batch size = 8
        CPU threads = 4

We need to measure which configuration works best.

Do not assume that increasing batch size or thread count
always improves performance.

======================================================================
2. LATENCY
======================================================================

Latency is the time required to complete an inference request.

    latency = completion_time - submission_time

Important statistics:

    Mean:
        Average latency.

    Median:
        Middle observation.

    P50:
        50th percentile latency.

    P90:
        90th percentile latency.

    P95:
        95th percentile latency.

    P99:
        99th percentile latency.

P95 is especially useful when evaluating real-time systems.

For example:

    Mean latency = 12 ms
    P95 latency  = 25 ms

This means most requests complete quickly, but some
requests take substantially longer.

======================================================================
3. THROUGHPUT
======================================================================

Throughput measures completed images per second.

    throughput = completed_images / elapsed_seconds

Example:

    1000 images / 10 seconds = 100 images/s

For batch inference, throughput counts individual images,
not inference calls.

======================================================================
4. WARM-UP
======================================================================

The first inference may be slower because of:

    - Runtime initialization
    - Memory allocation
    - Kernel initialization
    - CPU cache effects
    - GPU initialization

Therefore:

    Warm-up runs:
        Execute inference but do not include measurements.

    Benchmark runs:
        Measure inference after warm-up.

======================================================================
5. BATCH SIZE
======================================================================

Compare:

    Batch size 1
    Batch size 2
    Batch size 4
    Batch size 8
    Batch size 16

Larger batches can improve throughput but increase:

    - Memory usage
    - Per-request waiting time
    - End-to-end latency

A real-time robot may prefer batch size 1 even when
larger batches achieve better total throughput.

======================================================================
6. CPU THREADS
======================================================================

ONNX Runtime supports configurable CPU threading.

Compare:

    1 thread
    2 threads
    4 threads
    8 threads

Too many threads can cause oversubscription.

The optimal value depends on the CPU, model, and workload.

======================================================================
7. EXECUTION PROVIDERS
======================================================================

ONNX Runtime can use different execution providers:

    CPUExecutionProvider
    CUDAExecutionProvider
    TensorRTExecutionProvider

This lesson benchmarks the selected provider.

Use a provider only when it is available in your environment.

======================================================================
8. NUMERICAL CONSISTENCY
======================================================================

Different execution providers or optimization settings may
produce slightly different floating-point results.

We compare predictions and numerical output differences.

Important metrics:

    Maximum absolute error:

        max(abs(output_A - output_B))

    Mean absolute error:

        mean(abs(output_A - output_B))

A small numerical difference does not necessarily indicate
an incorrect model.

======================================================================
9. BENCHMARK STABILITY
======================================================================

A single measurement is not enough.

Run multiple benchmark repetitions.

Measure:

    - Mean throughput
    - Throughput standard deviation
    - Mean latency
    - P95 latency
    - Latency variability

======================================================================
10. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    Ceiling camera
         |
    Head camera
         |
    Wrist cameras
         |
         v
    Perception model
         |
         v
    Object detection / segmentation
         |
         v
    Object localization
         |
         v
    ACT manipulation

A useful deployment benchmark should evaluate:

    - Single-camera inference
    - Multiple-camera inference
    - Different image resolutions
    - Different batch sizes
    - CPU and GPU execution
    - End-to-end latency

A high-throughput model is not automatically suitable
for real-time robotic manipulation.

======================================================================
NEXT LESSON
-----------

Lesson 84 — Edge AI Deployment.
======================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import psutil


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)


# ======================================================================
# MEMORY
# ======================================================================

MIB = 1024.0 * 1024.0


def get_memory_mib():

    process = psutil.Process()

    return process.memory_info().rss / MIB


# ======================================================================
# DATASET
# ======================================================================

def load_images(args):

    from torchvision.datasets import CIFAR10, ImageFolder

    if args.data_dir is None:

        print("\nLoading CIFAR-10 test images...")

        dataset = CIFAR10(

            root=args.dataset_root,

            train=False,

            download=True,

        )

        raw_images = dataset.data

        raw_labels = dataset.targets

        class_names = dataset.classes

        count = min(
            args.num_requests,
            len(dataset),
        )

        rng = np.random.default_rng(42)

        indices = rng.choice(

            len(dataset),

            size=count,

            replace=False,

        )

        images = [

            Image.fromarray(
                raw_images[index]
            ).convert("RGB")

            for index in indices

        ]

        labels = [

            int(raw_labels[index])

            for index in indices

        ]

    else:

        dataset = ImageFolder(
            args.data_dir
        )

        if len(dataset) == 0:

            raise RuntimeError(
                "ImageFolder dataset is empty."
            )

        count = min(
            args.num_requests,
            len(dataset),
        )

        rng = np.random.default_rng(42)

        indices = rng.choice(

            len(dataset),

            size=count,

            replace=False,

        )

        images = []

        labels = []

        for index in indices:

            image, label = dataset[index]

            images.append(
                image.convert("RGB")
            )

            labels.append(
                int(label)
            )

        class_names = dataset.classes

    print(
        f"Loaded {len(images)} real images."
    )

    return images, labels, class_names


# ======================================================================
# PREPROCESSING
# ======================================================================

def preprocess_image(
    image,
    image_size,
):

    image = image.convert("RGB")

    image = image.resize(

        (image_size, image_size),

        Image.Resampling.BILINEAR,

    )

    array = np.asarray(

        image,

        dtype=np.float32,

    )

    array /= 255.0

    array -= 0.5

    array /= 0.5

    array = np.transpose(

        array,

        (2, 0, 1),

    )

    return np.ascontiguousarray(
        array
    )


def create_input_batch(
    images,
    start,
    batch_size,
    image_size,
):

    selected = images[
        start:start + batch_size
    ]

    if len(selected) != batch_size:

        raise ValueError(
            "Incomplete batch."
        )

    batch = np.stack(

        [

            preprocess_image(
                image,
                image_size,
            )

            for image in selected

        ],

        axis=0,

    )

    return np.ascontiguousarray(
        batch,
        dtype=np.float32,
    )


# ======================================================================
# FALLBACK MODEL
# ======================================================================

def create_fallback_onnx(
    output_path,
    image_size,
    num_classes=10,
):

    """
    Export a randomly initialized CNN.

    This model is for benchmarking infrastructure only.
    Its predictions are not meaningful.
    """

    import torch
    import torch.nn as nn

    class SmallCNN(nn.Module):

        def __init__(self):

            super().__init__()

            self.features = nn.Sequential(

                nn.Conv2d(
                    3,
                    32,
                    3,
                    padding=1,
                ),

                nn.ReLU(),

                nn.MaxPool2d(2),

                nn.Conv2d(
                    32,
                    64,
                    3,
                    padding=1,
                ),

                nn.ReLU(),

                nn.MaxPool2d(2),

                nn.Conv2d(
                    64,
                    128,
                    3,
                    padding=1,
                ),

                nn.ReLU(),

                nn.AdaptiveAvgPool2d(1),

            )

            self.classifier = nn.Sequential(

                nn.Flatten(),

                nn.Linear(
                    128,
                    128,
                ),

                nn.ReLU(),

                nn.Linear(
                    128,
                    num_classes,
                ),

            )

        def forward(self, x):

            return self.classifier(
                self.features(x)
            )

    torch.manual_seed(42)

    model = SmallCNN().eval()

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

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

        dynamo=False,

    )

    print(
        "\nCreated randomly initialized fallback model."
    )

    return output_path


# ======================================================================
# MODEL RESOLUTION
# ======================================================================

def resolve_model(
    requested_path,
    output_dir,
    image_size,
):

    if requested_path:

        path = Path(
            requested_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        return path

    candidates = [

        Path("runs/lesson78/model.onnx"),

        Path("runs/lesson77/model.onnx"),

        Path("runs/lesson80/fallback_random_model.onnx"),

        Path("runs/lesson82/fallback_random_model.onnx"),

    ]

    for candidate in candidates:

        if candidate.exists():

            print(
                f"Using existing model: {candidate}"
            )

            return candidate

    return create_fallback_onnx(

        output_dir / "fallback_random_model.onnx",

        image_size,

    )


# ======================================================================
# SESSION CREATION
# ======================================================================

def create_session(
    model_path,
    provider,
    intra_threads,
    inter_threads=1,
):

    import onnxruntime as ort

    available = ort.get_available_providers()

    if provider not in available:

        raise RuntimeError(

            f"Provider unavailable: {provider}\n"

            f"Available providers: {available}"

        )

    options = ort.SessionOptions()

    options.graph_optimization_level = (

        ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    )

    options.intra_op_num_threads = intra_threads

    options.inter_op_num_threads = inter_threads

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    session = ort.InferenceSession(

        str(model_path),

        sess_options=options,

        providers=[provider],

    )

    return session


# ======================================================================
# MODEL VALIDATION
# ======================================================================

def validate_model(
    session,
    image_size,
):

    inputs = session.get_inputs()

    outputs = session.get_outputs()

    if len(inputs) != 1:

        raise RuntimeError(
            "Expected exactly one input."
        )

    if len(outputs) != 1:

        raise RuntimeError(
            "Expected exactly one output."
        )

    input_info = inputs[0]

    output_info = outputs[0]

    if len(input_info.shape) != 4:

        raise RuntimeError(
            "Expected NCHW image input."
        )

    if len(output_info.shape) != 2:

        raise RuntimeError(
            "Expected [batch, classes] output."
        )

    if input_info.type != "tensor(float)":

        raise RuntimeError(
            "This benchmark expects float32 input."
        )

    if input_info.shape[1] != 3:

        raise RuntimeError(
            "Expected three-channel RGB input."
        )

    height = input_info.shape[2]

    width = input_info.shape[3]

    if isinstance(height, int):

        if height != image_size:

            raise ValueError(

                f"Model height is {height}; "
                f"requested image-size is {image_size}."

            )

    if isinstance(width, int):

        if width != image_size:

            raise ValueError(

                f"Model width is {width}; "
                f"requested image-size is {image_size}."

            )

    batch_dimension = input_info.shape[0]

    if isinstance(batch_dimension, int):

        if batch_dimension != 1:

            raise RuntimeError(

                "This lesson expects a dynamic batch dimension "
                "or a fixed batch size of 1."

            )

    return {

        "input_name": input_info.name,

        "output_name": output_info.name,

        "input_shape": input_info.shape,

        "output_shape": output_info.shape,

    }


# ======================================================================
# WARM-UP
# ======================================================================

def warmup_session(
    session,
    images,
    image_size,
    batch_size,
    warmup_runs,
):

    """
    Warm-up calls are excluded from benchmark statistics.
    """

    input_name = session.get_inputs()[0].name

    selected = [

        images[index % len(images)]

        for index in range(batch_size)

    ]

    batch = np.stack(

        [

            preprocess_image(
                image,
                image_size,
            )

            for image in selected

        ],

        axis=0,

    ).astype(np.float32)

    for _ in range(warmup_runs):

        session.run(

            None,

            {
                input_name: batch,
            },

        )

    return batch


# ======================================================================
# ONE BENCHMARK REPETITION
# ======================================================================

def benchmark_one_repetition(
    session,
    images,
    image_size,
    batch_size,
    warmup_runs,
    target_images,
):

    """
    Benchmark one session configuration.

    Batch construction and preprocessing are intentionally
    performed before the timed inference call.

    This isolates model inference runtime.

    The benchmark therefore does NOT measure complete camera-to-model
    application latency.

    Batch inference is timed as a whole, then its per-image
    throughput and per-batch-call latency are reported separately.
    """

    input_name = session.get_inputs()[0].name

    # Warm-up.

    warmup_session(

        session,

        images,

        image_size,

        batch_size,

        warmup_runs,

    )

    gc.collect()

    # Build the benchmark batches before timing.

    number_of_batches = (
        target_images + batch_size - 1
    ) // batch_size

    batches = []

    valid_counts = []

    for batch_index in range(number_of_batches):

        start = (
            batch_index * batch_size
        )

        selected = images[
            start:min(
                start + batch_size,
                target_images,
            )
        ]

        valid_count = len(
            selected
        )

        if valid_count == 0:

            continue

        # Pad the final batch if needed.

        if valid_count < batch_size:

            padded = list(selected)

            while len(padded) < batch_size:

                padded.append(
                    selected[-1]
                )

            selected = padded

        batch = np.stack(

            [

                preprocess_image(
                    image,
                    image_size,
                )

                for image in selected

            ],

            axis=0,

        ).astype(np.float32)

        batches.append(batch)

        valid_counts.append(
            valid_count
        )

    latencies_ms = []

    total_images = 0

    all_outputs = []

    start_time = time.perf_counter()

    for batch, valid_count in zip(
        batches,
        valid_counts,
    ):

        request_start = time.perf_counter()

        output = session.run(

            None,

            {
                input_name: batch,
            },

        )[0]

        request_end = time.perf_counter()

        latencies_ms.append(

            (request_end - request_start) * 1000.0

        )

        total_images += valid_count

        # Retain outputs for numerical validation.
        # Only the measured benchmark uses this list.

        all_outputs.append(
            output[:valid_count].copy()
        )

    elapsed = (
        time.perf_counter() - start_time
    )

    concatenated = np.concatenate(

        all_outputs,

        axis=0,

    )

    return {

        "latencies_ms": latencies_ms,

        "elapsed_seconds": elapsed,

        "images_processed": total_images,

        "throughput": total_images / elapsed,

        "outputs": concatenated,

    }


# ======================================================================
# STATISTICS
# ======================================================================

def calculate_latency_statistics(values):

    values = np.asarray(

        values,

        dtype=np.float64,

    )

    if values.size == 0:

        raise ValueError(
            "No latency measurements."
        )

    return {

        "mean_ms": float(
            np.mean(values)
        ),

        "median_ms": float(
            np.median(values)
        ),

        "std_ms": float(
            np.std(values)
        ),

        "min_ms": float(
            np.min(values)
        ),

        "max_ms": float(
            np.max(values)
        ),

        "p50_ms": float(
            np.percentile(values, 50)
        ),

        "p90_ms": float(
            np.percentile(values, 90)
        ),

        "p95_ms": float(
            np.percentile(values, 95)
        ),

        "p99_ms": float(
            np.percentile(values, 99)
        ),

    }


# ======================================================================
# CONFIGURATION BENCHMARK
# ======================================================================

def benchmark_configuration(
    model_path,
    provider,
    intra_threads,
    images,
    image_size,
    batch_size,
    warmup_runs,
    repetitions,
    target_images,
):

    print(
        "\n" + "-" * 75
    )

    print(
        f"Provider: {provider}"
    )

    print(
        f"Intra-op threads: {intra_threads}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        "-" * 75
    )

    session = create_session(

        model_path,

        provider,

        intra_threads,

    )

    validate_model(
        session,
        image_size,
    )

    repetition_results = []

    reference_outputs = None

    for repetition in range(repetitions):

        result = benchmark_one_repetition(

            session=session,

            images=images,

            image_size=image_size,

            batch_size=batch_size,

            warmup_runs=warmup_runs,

            target_images=target_images,

        )

        repetition_results.append(
            result
        )

        if reference_outputs is None:

            reference_outputs = result["outputs"]

        else:

            if not np.allclose(

                reference_outputs,

                result["outputs"],

                rtol=1e-4,

                atol=1e-5,

            ):

                raise RuntimeError(

                    "Repeated benchmark outputs are inconsistent."

                )

        print(

            f"Repetition {repetition + 1}/{repetitions}: "

            f"{result['throughput']:.2f} images/s"

        )

    all_latencies = np.concatenate(

        [

            result["latencies_ms"]

            for result in repetition_results

        ]

    )

    throughput_values = np.asarray(

        [

            result["throughput"]

            for result in repetition_results

        ],

        dtype=np.float64,

    )

    elapsed_values = [

        result["elapsed_seconds"]

        for result in repetition_results

    ]

    latency_stats = calculate_latency_statistics(
        all_latencies
    )

    summary = {

        "provider": provider,

        "intra_threads": intra_threads,

        "batch_size": batch_size,

        "repetitions": repetitions,

        "images_per_repetition": target_images,

        "mean_throughput_images_per_second": float(
            np.mean(throughput_values)
        ),

        "std_throughput_images_per_second": float(
            np.std(throughput_values)
        ),

        "min_throughput_images_per_second": float(
            np.min(throughput_values)
        ),

        "max_throughput_images_per_second": float(
            np.max(throughput_values)
        ),

        "mean_repetition_seconds": float(
            np.mean(elapsed_values)
        ),

        "batch_call_latency": latency_stats,

        "outputs": reference_outputs,

    }

    print(

        f"Average throughput: "
        f"{summary['mean_throughput_images_per_second']:.2f} images/s"

    )

    print(

        f"Average batch-call latency: "
        f"{latency_stats['mean_ms']:.3f} ms"

    )

    print(

        f"P95 batch-call latency: "
        f"{latency_stats['p95_ms']:.3f} ms"

    )

    return summary


# ======================================================================
# BATCH SIZE AND THREAD CONFIGURATION GENERATION
# ======================================================================

def parse_integer_list(value):

    values = [

        int(item.strip())

        for item in value.split(",")

        if item.strip()

    ]

    if not values:

        raise ValueError(
            "Configuration list cannot be empty."
        )

    if any(item <= 0 for item in values):

        raise ValueError(
            "All configuration values must be positive."
        )

    return sorted(
        set(values)
    )


# ======================================================================
# COMPARISON PLOTS
# ======================================================================

def plot_throughput(
    summaries,
    output_dir,
):

    labels = [

        f"B{s['batch_size']}\nT{s['intra_threads']}"

        for s in summaries

    ]

    values = [

        s["mean_throughput_images_per_second"]

        for s in summaries

    ]

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_title(
        "Model Throughput by Batch Size and CPU Threads"
    )

    ax.set_ylabel(
        "Images per second"
    )

    ax.set_xlabel(
        "Batch size / Intra-op threads"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(

        output_dir / "throughput_by_configuration.png",

        dpi=150,

    )

    plt.close(fig)


def plot_latency(
    summaries,
    output_dir,
):

    labels = [

        f"B{s['batch_size']}\nT{s['intra_threads']}"

        for s in summaries

    ]

    mean_values = [

        s["batch_call_latency"]["mean_ms"]

        for s in summaries

    ]

    p95_values = [

        s["batch_call_latency"]["p95_ms"]

        for s in summaries

    ]

    x = np.arange(
        len(labels)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(

        x - width / 2,

        mean_values,

        width,

        label="Mean",

    )

    ax.bar(

        x + width / 2,

        p95_values,

        width,

        label="P95",

    )

    ax.set_xticks(x)

    ax.set_xticklabels(labels)

    ax.set_title(
        "Batch-call Latency Comparison"
    )

    ax.set_ylabel(
        "Latency (ms)"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(

        output_dir / "latency_by_configuration.png",

        dpi=150,

    )

    plt.close(fig)


def plot_batch_scaling(
    summaries,
    output_dir,
):

    thread_values = sorted(

        set(
            item["intra_threads"]
            for item in summaries
        )

    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for threads in thread_values:

        selected = [

            item

            for item in summaries

            if item["intra_threads"] == threads

        ]

        selected.sort(
            key=lambda item: item["batch_size"]
        )

        batch_sizes = [

            item["batch_size"]

            for item in selected

        ]

        throughputs = [

            item["mean_throughput_images_per_second"]

            for item in selected

        ]

        ax.plot(

            batch_sizes,

            throughputs,

            marker="o",

            label=f"{threads} CPU threads",

        )

    ax.set_title(
        "Throughput Scaling with Batch Size"
    )

    ax.set_xlabel(
        "Batch size"
    )

    ax.set_ylabel(
        "Images per second"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(

        output_dir / "batch_scaling.png",

        dpi=150,

    )

    plt.close(fig)


def plot_repetition_stability(
    summaries,
    output_dir,
):

    labels = [

        f"B{s['batch_size']}\nT{s['intra_threads']}"

        for s in summaries

    ]

    means = [

        s["mean_throughput_images_per_second"]

        for s in summaries

    ]

    deviations = [

        s["std_throughput_images_per_second"]

        for s in summaries

    ]

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(

        labels,

        means,

        yerr=deviations,

        capsize=4,

    )

    ax.set_title(
        "Benchmark Repeatability"
    )

    ax.set_ylabel(
        "Throughput (images/s)"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(

        output_dir / "benchmark_stability.png",

        dpi=150,

    )

    plt.close(fig)


# ======================================================================
# BEST CONFIGURATION
# ======================================================================

def select_best_configuration(
    summaries,
):

    return max(

        summaries,

        key=lambda item: (
            item["mean_throughput_images_per_second"]
        ),

    )


# ======================================================================
# AUTOMATED TESTS
# ======================================================================

def create_test_model(
    path,
    image_size=32,
):

    import torch
    import torch.nn as nn

    class TestCNN(nn.Module):

        def __init__(self):

            super().__init__()

            self.network = nn.Sequential(

                nn.Conv2d(
                    3,
                    8,
                    3,
                    padding=1,
                ),

                nn.ReLU(),

                nn.AdaptiveAvgPool2d(1),

                nn.Flatten(),

                nn.Linear(
                    8,
                    10,
                ),

            )

        def forward(self, x):

            return self.network(x)

    torch.manual_seed(42)

    model = TestCNN().eval()

    torch.onnx.export(

        model,

        torch.randn(

            1,

            3,

            image_size,

            image_size,

        ),

        str(path),

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

        dynamo=False,

    )


def make_test_image(seed):

    rng = np.random.default_rng(seed)

    array = rng.integers(

        0,

        256,

        size=(48, 64, 3),

        dtype=np.uint8,

    )

    return Image.fromarray(
        array
    )


def run_tests():

    print(
        "\nRunning Lesson 83 tests...\n"
    )

    test_dir = Path(
        "runs/lesson83_tests"
    )

    test_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    set_seed(42)

    # --------------------------------------------------------------
    # Test 1: Image preprocessing
    # --------------------------------------------------------------

    image = make_test_image(1)

    tensor = preprocess_image(
        image,
        32,
    )

    assert tensor.shape == (
        3,
        32,
        32,
    )

    print(
        "Test 1 passed: Image preprocessing."
    )

    # --------------------------------------------------------------
    # Test 2: Input dtype
    # --------------------------------------------------------------

    assert tensor.dtype == np.float32

    print(
        "Test 2 passed: Float32 input."
    )

    # --------------------------------------------------------------
    # Test 3: Input normalization
    # --------------------------------------------------------------

    assert tensor.min() >= -1.01

    assert tensor.max() <= 1.01

    print(
        "Test 3 passed: Input normalization."
    )

    # --------------------------------------------------------------
    # Test 4: Contiguous array
    # --------------------------------------------------------------

    assert tensor.flags["C_CONTIGUOUS"]

    print(
        "Test 4 passed: Contiguous input."
    )

    # --------------------------------------------------------------
    # Test 5: Batch construction
    # --------------------------------------------------------------

    images = [

        make_test_image(index)

        for index in range(4)

    ]

    batch = create_input_batch(

        images,

        0,

        4,

        32,

    )

    assert batch.shape == (
        4,
        3,
        32,
        32,
    )

    print(
        "Test 5 passed: Batch construction."
    )

    # --------------------------------------------------------------
    # Test 6: Batch dtype
    # --------------------------------------------------------------

    assert batch.dtype == np.float32

    print(
        "Test 6 passed: Batch dtype."
    )

    # --------------------------------------------------------------
    # Test 7: Batch values finite
    # --------------------------------------------------------------

    assert np.isfinite(
        batch
    ).all()

    print(
        "Test 7 passed: Finite batch values."
    )

    # --------------------------------------------------------------
    # Test 8: Latency statistics
    # --------------------------------------------------------------

    stats = calculate_latency_statistics(

        [1, 2, 3, 4, 5]

    )

    assert stats["mean_ms"] == 3.0

    assert stats["median_ms"] == 3.0

    print(
        "Test 8 passed: Latency statistics."
    )

    # --------------------------------------------------------------
    # Test 9: P50
    # --------------------------------------------------------------

    assert stats["p50_ms"] == 3.0

    print(
        "Test 9 passed: P50 calculation."
    )

    # --------------------------------------------------------------
    # Test 10: P95
    # --------------------------------------------------------------

    assert stats["p95_ms"] >= stats["p50_ms"]

    print(
        "Test 10 passed: P95 calculation."
    )

    # --------------------------------------------------------------
    # Test 11: P99
    # --------------------------------------------------------------

    assert stats["p99_ms"] >= stats["p95_ms"]

    print(
        "Test 11 passed: P99 calculation."
    )

    # --------------------------------------------------------------
    # Test 12: Standard deviation
    # --------------------------------------------------------------

    assert stats["std_ms"] >= 0

    print(
        "Test 12 passed: Standard deviation."
    )

    # --------------------------------------------------------------
    # Test 13: Configuration parser
    # --------------------------------------------------------------

    values = parse_integer_list(
        "1,2,4,8"
    )

    assert values == [
        1,
        2,
        4,
        8,
    ]

    print(
        "Test 13 passed: Configuration parser."
    )

    # --------------------------------------------------------------
    # Test 14: Duplicate configurations
    # --------------------------------------------------------------

    values = parse_integer_list(
        "1,2,2,4"
    )

    assert values == [
        1,
        2,
        4,
    ]

    print(
        "Test 14 passed: Duplicate removal."
    )

    # --------------------------------------------------------------
    # Test 15: Invalid configuration
    # --------------------------------------------------------------

    try:

        parse_integer_list(
            "1,0,4"
        )

        raise AssertionError(
            "Expected ValueError."
        )

    except ValueError:

        pass

    print(
        "Test 15 passed: Invalid configuration detection."
    )

    # --------------------------------------------------------------
    # Test 16: ONNX export
    # --------------------------------------------------------------

    model_path = (
        test_dir / "test_model.onnx"
    )

    create_test_model(
        model_path,
        32,
    )

    assert model_path.exists()

    print(
        "Test 16 passed: ONNX export."
    )

    # --------------------------------------------------------------
    # Test 17: ONNX validation
    # --------------------------------------------------------------

    import onnx

    model = onnx.load(
        str(model_path)
    )

    onnx.checker.check_model(
        model
    )

    print(
        "Test 17 passed: ONNX graph validation."
    )

    # --------------------------------------------------------------
    # Test 18: ONNX Runtime
    # --------------------------------------------------------------

    import onnxruntime as ort

    session = create_session(

        model_path,

        "CPUExecutionProvider",

        1,

    )

    assert session is not None

    print(
        "Test 18 passed: ONNX Runtime session."
    )

    # --------------------------------------------------------------
    # Test 19: Model metadata
    # --------------------------------------------------------------

    metadata = validate_model(
        session,
        32,
    )

    assert metadata["input_shape"][1:] == [
        3,
        32,
        32,
    ]

    print(
        "Test 19 passed: Model metadata."
    )

    # --------------------------------------------------------------
    # Test 20: Batch-size-one inference
    # --------------------------------------------------------------

    input_name = metadata["input_name"]

    output = session.run(

        None,

        {
            input_name: batch[:1],
        },

    )[0]

    assert output.shape == (
        1,
        10,
    )

    print(
        "Test 20 passed: Batch-size-one inference."
    )

    # --------------------------------------------------------------
    # Test 21: Batch-size-four inference
    # --------------------------------------------------------------

    output = session.run(

        None,

        {
            input_name: batch,
        },

    )[0]

    assert output.shape == (
        4,
        10,
    )

    print(
        "Test 21 passed: Batch-size-four inference."
    )

    # --------------------------------------------------------------
    # Test 22: Dynamic batch consistency
    # --------------------------------------------------------------

    individual = [

        session.run(

            None,

            {
                input_name: batch[index:index + 1],
            },

        )[0]

        for index in range(4)

    ]

    individual = np.concatenate(
        individual,
        axis=0,
    )

    assert np.allclose(

        output,

        individual,

        atol=1e-5,

    )

    print(
        "Test 22 passed: Dynamic batch consistency."
    )

    # --------------------------------------------------------------
    # Test 23: Warm-up
    # --------------------------------------------------------------

    warmup = warmup_session(

        session,

        images,

        32,

        2,

        2,

    )

    assert warmup.shape == (
        2,
        3,
        32,
        32,
    )

    print(
        "Test 23 passed: Warm-up execution."
    )

    # --------------------------------------------------------------
    # Test 24: Benchmark repetition
    # --------------------------------------------------------------

    result = benchmark_one_repetition(

        session,

        images,

        32,

        2,

        1,

        4,

    )

    assert result["images_processed"] == 4

    assert result["throughput"] > 0

    print(
        "Test 24 passed: Benchmark execution."
    )

    # --------------------------------------------------------------
    # Test 25: Benchmark output shape
    # --------------------------------------------------------------

    assert result["outputs"].shape == (
        4,
        10,
    )

    print(
        "Test 25 passed: Benchmark output shape."
    )

    # --------------------------------------------------------------
    # Test 26: Latency count
    # --------------------------------------------------------------

    assert len(
        result["latencies_ms"]
    ) == 2

    print(
        "Test 26 passed: Batch-call latency count."
    )

    # --------------------------------------------------------------
    # Test 27: Positive timing
    # --------------------------------------------------------------

    assert result["elapsed_seconds"] > 0

    assert all(

        item >= 0

        for item in result["latencies_ms"]

    )

    print(
        "Test 27 passed: Positive benchmark timing."
    )

    # --------------------------------------------------------------
    # Test 28: Numerical repeatability
    # --------------------------------------------------------------

    repeated = benchmark_one_repetition(

        session,

        images,

        32,

        2,

        1,

        4,

    )

    assert np.allclose(

        result["outputs"],

        repeated["outputs"],

        atol=1e-5,

    )

    print(
        "Test 28 passed: Repeated output consistency."
    )

    # --------------------------------------------------------------
    # Test 29: Best configuration
    # --------------------------------------------------------------

    dummy = [

        {

            "batch_size": 1,

            "intra_threads": 1,

            "mean_throughput_images_per_second": 10.0,

        },

        {

            "batch_size": 4,

            "intra_threads": 2,

            "mean_throughput_images_per_second": 30.0,

        },

    ]

    best = select_best_configuration(
        dummy
    )

    assert best["batch_size"] == 4

    print(
        "Test 29 passed: Best configuration selection."
    )

    # --------------------------------------------------------------
    # Test 30: Memory measurement
    # --------------------------------------------------------------

    assert get_memory_mib() > 0

    print(
        "Test 30 passed: Process memory measurement."
    )

    print(
        "\nAll 30 tests passed."
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Lesson 83 — Model Benchmarking"
        )

    )

    parser.add_argument(

        "--onnx-path",

        type=str,

        default=None,

    )

    parser.add_argument(

        "--data-dir",

        type=str,

        default=None,

    )

    parser.add_argument(

        "--dataset-root",

        type=str,

        default="./datasets",

    )

    parser.add_argument(

        "--provider",

        type=str,

        default="CPUExecutionProvider",

    )

    parser.add_argument(

        "--image-size",

        type=int,

        default=64,

    )

    parser.add_argument(

        "--num-requests",

        type=int,

        default=256,

    )

    parser.add_argument(

        "--batch-sizes",

        type=str,

        default="1,2,4,8",

    )

    parser.add_argument(

        "--thread-counts",

        type=str,

        default="1,2,4",

    )

    parser.add_argument(

        "--warmup-runs",

        type=int,

        default=10,

    )

    parser.add_argument(

        "--repetitions",

        type=int,

        default=3,

    )

    parser.add_argument(

        "--output-dir",

        type=str,

        default="runs/lesson83",

    )

    parser.add_argument(

        "--check-only",

        action="store_true",

    )

    args = parser.parse_args()

    if args.check_only:

        run_tests()

        return

    if args.num_requests <= 0:

        raise ValueError(
            "num-requests must be positive."
        )

    if args.image_size < 32:

        raise ValueError(
            "image-size must be at least 32."
        )

    if args.warmup_runs < 0:

        raise ValueError(
            "warmup-runs cannot be negative."
        )

    if args.repetitions <= 0:

        raise ValueError(
            "repetitions must be positive."
        )

    batch_sizes = parse_integer_list(
        args.batch_sizes
    )

    thread_counts = parse_integer_list(
        args.thread_counts
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    set_seed(42)

    print(
        "\n" + "=" * 75
    )

    print(
        "LESSON 83 — MODEL BENCHMARKING"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------------
    # LOAD REAL IMAGES
    # --------------------------------------------------------------

    images, labels, class_names = load_images(
        args
    )

    # --------------------------------------------------------------
    # RESOLVE MODEL
    # --------------------------------------------------------------

    model_path = resolve_model(

        args.onnx_path,

        output_dir,

        args.image_size,

    )

    print(
        f"\nModel: {model_path}"
    )

    print(
        f"Provider: {args.provider}"
    )

    # --------------------------------------------------------------
    # CHECK PROVIDER
    # --------------------------------------------------------------

    import onnxruntime as ort

    available = ort.get_available_providers()

    if args.provider not in available:

        raise RuntimeError(

            f"Requested provider {args.provider} is unavailable.\n"

            f"Available providers: {available}"

        )

    print(
        f"Available providers: {available}"
    )

    # --------------------------------------------------------------
    # CONFIGURATION BENCHMARKS
    # --------------------------------------------------------------

    summaries = []

    reference_output = None

    target_images = min(

        args.num_requests,

        len(images),

    )

    for threads in thread_counts:

        for batch_size in batch_sizes:

            # Skip batch sizes unsupported by a fixed-batch model.

            session_check = create_session(

                model_path,

                args.provider,

                threads,

            )

            metadata = validate_model(

                session_check,

                args.image_size,

            )

            fixed_batch = metadata["input_shape"][0]

            if isinstance(fixed_batch, int):

                if batch_size != fixed_batch:

                    print(

                        f"Skipping batch {batch_size}: "
                        f"model requires batch {fixed_batch}."

                    )

                    del session_check

                    continue

            del session_check

            summary = benchmark_configuration(

                model_path=model_path,

                provider=args.provider,

                intra_threads=threads,

                images=images,

                image_size=args.image_size,

                batch_size=batch_size,

                warmup_runs=args.warmup_runs,

                repetitions=args.repetitions,

                target_images=target_images,

            )

            outputs = summary.pop(
                "outputs"
            )

            if reference_output is None:

                reference_output = outputs

            else:

                if not np.allclose(

                    reference_output,

                    outputs,

                    rtol=1e-4,

                    atol=1e-5,

                ):

                    raise RuntimeError(

                        "Numerical output differs between configurations."

                    )

            summaries.append(
                summary
            )

    if not summaries:

        raise RuntimeError(
            "No valid benchmark configurations were executed."
        )

    # --------------------------------------------------------------
    # SELECT BEST CONFIGURATION
    # --------------------------------------------------------------

    best = select_best_configuration(
        summaries
    )

    # --------------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "BENCHMARK SUMMARY"
    )

    print(
        "=" * 75
    )

    for item in summaries:

        stats = item["batch_call_latency"]

        print(

            f"\nBatch={item['batch_size']} "

            f"Threads={item['intra_threads']}"

        )

        print(

            f"Throughput: "
            f"{item['mean_throughput_images_per_second']:.2f} images/s"

        )

        print(

            f"Throughput std: "
            f"{item['std_throughput_images_per_second']:.2f} images/s"

        )

        print(

            f"Mean batch latency: "
            f"{stats['mean_ms']:.3f} ms"

        )

        print(

            f"P50: {stats['p50_ms']:.3f} ms | "

            f"P95: {stats['p95_ms']:.3f} ms | "

            f"P99: {stats['p99_ms']:.3f} ms"

        )

    print(
        "\n" + "=" * 75
    )

    print(
        "BEST THROUGHPUT CONFIGURATION"
    )

    print(
        "=" * 75
    )

    print(
        f"Batch size: {best['batch_size']}"
    )

    print(
        f"CPU threads: {best['intra_threads']}"
    )

    print(

        f"Throughput: "
        f"{best['mean_throughput_images_per_second']:.2f} images/s"

    )

    print(
        "\nNumerical consistency: PASSED."
    )

    # --------------------------------------------------------------
    # GENERATE PLOTS
    # --------------------------------------------------------------

    plot_throughput(
        summaries,
        output_dir,
    )

    plot_latency(
        summaries,
        output_dir,
    )

    plot_batch_scaling(
        summaries,
        output_dir,
    )

    plot_repetition_stability(
        summaries,
        output_dir,
    )

    # --------------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------------

    import csv

    csv_path = (
        output_dir / "benchmark_results.csv"
    )

    with open(

        csv_path,

        "w",

        newline="",

        encoding="utf-8",

    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=[

                "provider",

                "batch_size",

                "intra_threads",

                "mean_throughput_images_per_second",

                "std_throughput_images_per_second",

                "mean_latency_ms",

                "p50_ms",

                "p90_ms",

                "p95_ms",

                "p99_ms",

            ],

        )

        writer.writeheader()

        for item in summaries:

            stats = item["batch_call_latency"]

            writer.writerow({

                "provider": item["provider"],

                "batch_size": item["batch_size"],

                "intra_threads": item["intra_threads"],

                "mean_throughput_images_per_second": (
                    item["mean_throughput_images_per_second"]
                ),

                "std_throughput_images_per_second": (
                    item["std_throughput_images_per_second"]
                ),

                "mean_latency_ms": stats["mean_ms"],

                "p50_ms": stats["p50_ms"],

                "p90_ms": stats["p90_ms"],

                "p95_ms": stats["p95_ms"],

                "p99_ms": stats["p99_ms"],

            })

    # --------------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------------

    report = {

        "lesson": 83,

        "title": "Model Benchmarking",

        "model": str(model_path),

        "provider": args.provider,

        "dataset": {

            "num_images": target_images,

            "class_names": class_names,

            "image_size": args.image_size,

        },

        "benchmark_configuration": {

            "batch_sizes": batch_sizes,

            "thread_counts": thread_counts,

            "warmup_runs": args.warmup_runs,

            "repetitions": args.repetitions,

        },

        "results": summaries,

        "best_throughput_configuration": {

            "batch_size": best["batch_size"],

            "intra_threads": best["intra_threads"],

            "throughput": (
                best["mean_throughput_images_per_second"]
            ),

        },

        "numerical_consistency": True,

    }

    report_path = (

        output_dir / "benchmark_report.json"

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

    print(
        "\n" + "=" * 75
    )

    print(
        "LESSON 83 COMPLETED."
    )

    print(
        "=" * 75
    )

    print(
        f"Output directory: {output_dir.resolve()}"
    )

    print(
        "\nGenerated files:"
    )

    print(
        "  throughput_by_configuration.png"
    )

    print(
        "  latency_by_configuration.png"
    )

    print(
        "  batch_scaling.png"
    )

    print(
        "  benchmark_stability.png"
    )

    print(
        "  benchmark_results.csv"
    )

    print(
        "  benchmark_report.json"
    )

    print(
        "\nNext: Lesson 84 — Edge AI Deployment."
    )


if __name__ == "__main__":

    main()