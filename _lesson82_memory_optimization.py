"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 6 — MODEL OPTIMIZATION AND DEPLOYMENT

LESSON 82 — MEMORY OPTIMIZATION
======================================================================

LEARNING OBJECTIVES
-------------------

1. Understand model memory, activation memory, and inference buffers.
2. Measure CPU RAM usage.
3. Inspect GPU memory when CUDA is available.
4. Avoid repeatedly allocating input arrays.
5. Reuse inference input and output buffers.
6. Understand ONNX Runtime I/O Binding.
7. Avoid accumulating inference outputs unnecessarily.
8. Stream images instead of loading an entire dataset into RAM.
9. Measure memory consumption and inference speed.
10. Understand memory/performance trade-offs.

NO pretrained weights.
NO pretrained models.

======================================================================
1. WHERE DOES INFERENCE MEMORY GO?
======================================================================

A computer vision inference application uses memory for:

    Model parameters
    Intermediate activations
    Input images
    Preprocessing buffers
    Output tensors
    Runtime execution buffers
    Queues and cached frames

A simplified memory model:

    Total memory =
        model memory
        + activation memory
        + input memory
        + output memory
        + temporary memory
        + application overhead

For a multi-camera robot, memory can grow quickly.

Example:

    4 cameras
    640 x 480 RGB
    uint8 images

Memory per image:

    640 * 480 * 3 = 921,600 bytes

Four cameras:

    3,686,400 bytes

This is approximately 3.52 MiB for one set of four images.

If the application keeps 100 frames per camera:

    3.52 MiB * 100 = 352 MiB

This does not include model activations or intermediate copies.

======================================================================
2. MODEL MEMORY
======================================================================

For a model with P parameters:

    FP32 parameter memory = P * 4 bytes
    FP16 parameter memory = P * 2 bytes
    INT8 parameter memory = P * 1 byte

For example:

    10 million FP32 parameters:

    10,000,000 * 4 = 40 MB

This is parameter storage only.

It does not include activations or runtime buffers.

======================================================================
3. ACTIVATION MEMORY
======================================================================

Intermediate feature maps can consume substantial memory.

For a feature map:

    [B, C, H, W]

FP32 activation memory:

    B * C * H * W * 4 bytes

Example:

    [1, 128, 80, 80]

    1 * 128 * 80 * 80 * 4
    = 3,276,800 bytes

Approximately 3.125 MiB.

======================================================================
4. MEMORY ALLOCATION
======================================================================

A naive inference loop repeatedly creates arrays:

    for image in images:

        tensor = preprocess(image)

        output = session.run(...)

Repeated allocations increase allocator activity.

A more efficient approach:

    input_buffer = allocate_once()

    for image in images:

        preprocess_into(image, input_buffer)

        run_inference(input_buffer)

The input buffer is reused.

======================================================================
5. ONNX RUNTIME I/O BINDING
======================================================================

Ordinary inference:

    output = session.run(...)

ONNX Runtime allocates output memory and returns it.

I/O Binding can bind input and output tensors to existing buffers.

This can reduce repeated allocation and copying.

This lesson demonstrates CPU-buffer I/O Binding.

For GPU deployment, device-side I/O Binding can avoid
unnecessary CPU/GPU transfers, but requires correctly managed
device memory and compatible execution providers.

======================================================================
6. STREAMING INFERENCE
======================================================================

Avoid retaining every output if the application only needs
the current prediction.

Bad:

    all_outputs = []

    for image in images:

        output = session.run(...)

        all_outputs.append(output)

Better:

    for image in images:

        output = session.run(...)

        prediction = process(output)

        discard_temporary_output()

For large datasets, use a streaming data source rather than
loading every image into memory.

======================================================================
7. CPU AND GPU MEMORY
======================================================================

CPU memory:

    Process resident set size (RSS).

GPU memory:

    CUDA allocated memory
    CUDA reserved memory

Important:

    PyTorch CUDA memory metrics do not necessarily include
    memory allocated internally by ONNX Runtime.

Use NVIDIA tools such as nvidia-smi for total device usage.

======================================================================
8. EXPERIMENTS
======================================================================

Compare:

    A. Standard ONNX inference.
    B. Reusable input buffer.
    C. Reusable input and output buffers using I/O Binding.

Measure:

    - Throughput
    - Mean latency
    - P95 latency
    - Initial RSS
    - Peak RSS
    - Final RSS
    - Memory growth

Do not assume buffer reuse always improves speed.
Measure it on the target hardware.

======================================================================
9. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    Camera frames
         |
         v
    Bounded camera buffers
         |
         v
    Reusable preprocessing buffers
         |
         v
    ONNX Runtime inference
         |
         v
    Reusable output buffers
         |
         v
    Detection / segmentation
         |
         v
    Object localization
         |
         v
    ACT manipulation

A robot should not accumulate old camera frames or inference
outputs indefinitely.

Memory use should remain bounded during long-running operation.

======================================================================
NEXT LESSON
-----------

Lesson 83 — Model Benchmarking.
======================================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import threading
import time

from dataclasses import dataclass
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
# MEMORY UTILITIES
# ======================================================================

BYTES_PER_MIB = 1024.0 * 1024.0


def bytes_to_mib(value):

    return float(value) / BYTES_PER_MIB


def get_rss_bytes():

    process = psutil.Process()

    return process.memory_info().rss


def get_rss_mib():

    return bytes_to_mib(
        get_rss_bytes()
    )


def get_torch_cuda_memory():

    """
    Return PyTorch CUDA memory statistics when available.

    Important:
        These statistics do not necessarily include allocations
        made internally by ONNX Runtime.
    """

    try:

        import torch

        if not torch.cuda.is_available():

            return None

        return {

            "allocated_mib": bytes_to_mib(

                torch.cuda.memory_allocated()

            ),

            "reserved_mib": bytes_to_mib(

                torch.cuda.memory_reserved()

            ),

            "max_allocated_mib": bytes_to_mib(

                torch.cuda.max_memory_allocated()

            ),

        }

    except Exception:

        return None


class MemorySampler:
    """
    Periodically sample process RSS.

    The sampler runs independently of inference.

    This provides an approximate observed peak. Very short-lived
    allocations may occur between sampling intervals.
    """

    def __init__(
        self,
        interval_seconds=0.01,
    ):

        self.interval_seconds = interval_seconds

        self.samples_mib = []

        self.stop_event = threading.Event()

        self.thread = None

    def _sample_loop(self):

        while not self.stop_event.is_set():

            try:

                self.samples_mib.append(
                    get_rss_mib()
                )

            except Exception:

                pass

            self.stop_event.wait(
                self.interval_seconds
            )

    def start(self):

        self.samples_mib = []

        self.stop_event.clear()

        self.thread = threading.Thread(

            target=self._sample_loop,

            daemon=True,

        )

        self.thread.start()

    def stop(self):

        self.stop_event.set()

        if self.thread is not None:

            self.thread.join()

        # Capture a final sample.

        self.samples_mib.append(
            get_rss_mib()
        )

    def summary(self):

        if not self.samples_mib:

            return {

                "initial_rss_mib": get_rss_mib(),

                "peak_rss_mib": get_rss_mib(),

                "final_rss_mib": get_rss_mib(),

                "observed_rss_growth_mib": 0.0,

            }

        values = np.asarray(
            self.samples_mib,
            dtype=np.float64,
        )

        initial = float(values[0])

        peak = float(np.max(values))

        final = float(values[-1])

        return {

            "initial_rss_mib": initial,

            "peak_rss_mib": peak,

            "final_rss_mib": final,

            "observed_rss_growth_mib": peak - initial,

        }


# ======================================================================
# IMAGE LOADING
# ======================================================================

def load_images(args):

    from torchvision.datasets import CIFAR10, ImageFolder

    if args.data_dir is None:

        print(
            "\nLoading CIFAR-10 test images..."
        )

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
    """
    Allocate a new normalized NCHW float32 array.
    """

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
        array[None, ...]
    )


def preprocess_into(
    image,
    image_size,
    input_buffer,
):
    """
    Preprocess an image into an existing input buffer.

    The final NCHW buffer is reused.

    PIL resizing and intermediate conversion still allocate
    temporary objects. This is not a zero-allocation pipeline.
    """

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

    chw = np.transpose(

        array,

        (2, 0, 1),

    )

    np.copyto(

        input_buffer[0],

        chw,

    )

    return input_buffer


# ======================================================================
# FALLBACK ONNX MODEL
# ======================================================================

def create_fallback_onnx(
    output_path,
    image_size,
    num_classes=10,
):
    """
    Export a small randomly initialized CNN.

    This model is for memory and runtime experiments only.
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
        "\nCreated random-weight fallback ONNX model."
    )

    print(
        "Use a trained model for meaningful predictions."
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

    ]

    for path in candidates:

        if path.exists():

            print(
                f"Using existing model: {path}"
            )

            return path

    return create_fallback_onnx(

        output_dir / "fallback_random_model.onnx",

        image_size,

    )


# ======================================================================
# ONNX SESSION
# ======================================================================

def create_session(
    model_path,
    provider,
    intra_threads,
):

    import onnxruntime as ort

    available = ort.get_available_providers()

    if provider not in available:

        raise RuntimeError(

            f"Provider {provider} is unavailable.\n"

            f"Available providers: {available}"

        )

    options = ort.SessionOptions()

    options.graph_optimization_level = (

        ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    )

    options.intra_op_num_threads = intra_threads

    options.inter_op_num_threads = 1

    options.enable_mem_pattern = True

    options.enable_cpu_mem_arena = True

    session = ort.InferenceSession(

        str(model_path),

        sess_options=options,

        providers=[provider],

    )

    print(
        f"Active providers: {session.get_providers()}"
    )

    for item in session.get_inputs():

        print(
            f"Input: {item.name}, shape={item.shape}, type={item.type}"
        )

    for item in session.get_outputs():

        print(
            f"Output: {item.name}, shape={item.shape}, type={item.type}"
        )

    return session


# ======================================================================
# MODEL METADATA
# ======================================================================

def inspect_model(session):

    if len(session.get_inputs()) != 1:

        raise RuntimeError(
            "This lesson expects one model input."
        )

    if len(session.get_outputs()) != 1:

        raise RuntimeError(
            "This lesson expects one model output."
        )

    input_info = session.get_inputs()[0]

    output_info = session.get_outputs()[0]

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
            "This lesson expects float32 model input."
        )

    return input_info, output_info


# ======================================================================
# MODEL SIZE
# ======================================================================

def get_model_file_size(model_path):

    return bytes_to_mib(
        Path(model_path).stat().st_size
    )


# ======================================================================
# STANDARD INFERENCE
# ======================================================================

def run_standard(
    session,
    images,
    image_size,
):
    """
    Standard inference.

    Each request creates a new preprocessed input array.
    ONNX Runtime returns a newly allocated output array.
    """

    input_name = session.get_inputs()[0].name

    predictions = []

    latencies = []

    sampler = MemorySampler()

    gc.collect()

    sampler.start()

    start = time.perf_counter()

    for image in images:

        request_start = time.perf_counter()

        tensor = preprocess_image(

            image,

            image_size,

        )

        output = session.run(

            None,

            {
                input_name: tensor,
            },

        )[0]

        prediction = int(
            np.argmax(output[0])
        )

        predictions.append(
            prediction
        )

        latencies.append(

            (

                time.perf_counter()

                - request_start

            ) * 1000.0

        )

        # The next iteration replaces the references.
        # No output arrays are retained.

    elapsed = (
        time.perf_counter() - start
    )

    sampler.stop()

    return {

        "mode": "standard",

        "predictions": predictions,

        "latencies_ms": latencies,

        "elapsed_seconds": elapsed,

        "throughput": len(images) / elapsed,

        "memory": sampler.summary(),

    }


# ======================================================================
# REUSABLE INPUT BUFFER
# ======================================================================

def run_reusable_input(
    session,
    images,
    image_size,
):
    """
    Reuse a single input array.

    Output memory is still allocated by session.run().
    """

    input_name = session.get_inputs()[0].name

    input_buffer = np.empty(

        (

            1,

            3,

            image_size,

            image_size,

        ),

        dtype=np.float32,

    )

    predictions = []

    latencies = []

    sampler = MemorySampler()

    gc.collect()

    sampler.start()

    start = time.perf_counter()

    for image in images:

        request_start = time.perf_counter()

        preprocess_into(

            image,

            image_size,

            input_buffer,

        )

        output = session.run(

            None,

            {
                input_name: input_buffer,
            },

        )[0]

        prediction = int(
            np.argmax(output[0])
        )

        predictions.append(
            prediction
        )

        latencies.append(

            (

                time.perf_counter()

                - request_start

            ) * 1000.0

        )

    elapsed = (
        time.perf_counter() - start
    )

    sampler.stop()

    return {

        "mode": "reusable_input",

        "predictions": predictions,

        "latencies_ms": latencies,

        "elapsed_seconds": elapsed,

        "throughput": len(images) / elapsed,

        "memory": sampler.summary(),

    }


# ======================================================================
# OUTPUT TYPE MAPPING
# ======================================================================

def output_numpy_dtype(onnx_type):

    mapping = {

        "tensor(float)": np.float32,

        "tensor(float16)": np.float16,

        "tensor(double)": np.float64,

    }

    if onnx_type not in mapping:

        raise RuntimeError(

            f"Unsupported output type for this lesson: {onnx_type}"

        )

    return mapping[onnx_type]


# ======================================================================
# REUSABLE INPUT + OUTPUT BUFFERS
# ======================================================================

class ReusableIOBinding:
    """
    Reusable CPU input and output buffers using ONNX Runtime I/O Binding.

    The output buffer is allocated once after a warm-up inference.

    The model must have an output shape compatible with the
    allocated buffer.

    This lesson uses batch size 1.

    Important:
        CPU I/O Binding demonstrates buffer reuse. It does not
        guarantee a speed improvement. GPU deployment can use
        device-side binding to reduce host/device transfers.
    """

    def __init__(
        self,
        session,
        image_size,
    ):

        self.session = session

        self.image_size = image_size

        self.input_info, self.output_info = inspect_model(
            session
        )

        self.input_name = (
            self.input_info.name
        )

        self.output_name = (
            self.output_info.name
        )

        self.input_buffer = np.empty(

            (

                1,

                3,

                image_size,

                image_size,

            ),

            dtype=np.float32,

        )

        # Determine the output shape and dtype with one warm-up run.

        warmup_output = session.run(

            [self.output_name],

            {
                self.input_name: self.input_buffer,
            },

        )[0]

        self.output_shape = tuple(
            warmup_output.shape
        )

        self.output_dtype = output_numpy_dtype(
            self.output_info.type
        )

        if warmup_output.dtype != self.output_dtype:

            raise RuntimeError(
                "Output dtype does not match ONNX metadata."
            )

        if len(self.output_shape) != 2:

            raise RuntimeError(
                "Expected a two-dimensional output."
            )

        self.output_buffer = np.empty(

            self.output_shape,

            dtype=self.output_dtype,

        )

        self.binding = session.io_binding()

        self.binding.bind_input(

            name=self.input_name,

            device_type="cpu",

            device_id=0,

            element_type=np.float32,

            shape=self.input_buffer.shape,

            buffer_ptr=self.input_buffer.ctypes.data,

        )

        self.binding.bind_output(

            name=self.output_name,

            device_type="cpu",

            device_id=0,

            element_type=self.output_dtype,

            shape=self.output_buffer.shape,

            buffer_ptr=self.output_buffer.ctypes.data,

        )

    def infer(self, image):

        preprocess_into(

            image,

            self.image_size,

            self.input_buffer,

        )

        self.session.run_with_iobinding(
            self.binding
        )

        # Copy only the scalar prediction.
        # Do not retain a reference to the reusable output buffer.

        prediction = int(
            np.argmax(self.output_buffer[0])
        )

        return prediction


def run_reusable_io(
    session,
    images,
    image_size,
):

    engine = ReusableIOBinding(

        session,

        image_size,

    )

    predictions = []

    latencies = []

    sampler = MemorySampler()

    gc.collect()

    sampler.start()

    start = time.perf_counter()

    for image in images:

        request_start = time.perf_counter()

        prediction = engine.infer(
            image
        )

        predictions.append(
            prediction
        )

        latencies.append(

            (

                time.perf_counter()

                - request_start

            ) * 1000.0

        )

    elapsed = (
        time.perf_counter() - start
    )

    sampler.stop()

    return {

        "mode": "reusable_input_output",

        "predictions": predictions,

        "latencies_ms": latencies,

        "elapsed_seconds": elapsed,

        "throughput": len(images) / elapsed,

        "memory": sampler.summary(),

    }


# ======================================================================
# SUMMARY
# ======================================================================

def summarize_result(result):

    latencies = np.asarray(

        result["latencies_ms"],

        dtype=np.float64,

    )

    return {

        "mode": result["mode"],

        "requests": len(latencies),

        "elapsed_seconds": result["elapsed_seconds"],

        "throughput_images_per_second": result["throughput"],

        "mean_latency_ms": float(
            np.mean(latencies)
        ),

        "p50_latency_ms": float(
            np.percentile(latencies, 50)
        ),

        "p95_latency_ms": float(
            np.percentile(latencies, 95)
        ),

        "max_latency_ms": float(
            np.max(latencies)
        ),

        "memory": result["memory"],

    }


# ======================================================================
# VISUALIZATION
# ======================================================================

def plot_memory_comparison(
    summaries,
    output_dir,
):

    labels = [

        item["mode"]

        for item in summaries

    ]

    initial = [

        item["memory"]["initial_rss_mib"]

        for item in summaries

    ]

    peak = [

        item["memory"]["peak_rss_mib"]

        for item in summaries

    ]

    final = [

        item["memory"]["final_rss_mib"]

        for item in summaries

    ]

    x = np.arange(
        len(labels)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(

        x - width,

        initial,

        width,

        label="Initial RSS",

    )

    ax.bar(

        x,

        peak,

        width,

        label="Peak RSS",

    )

    ax.bar(

        x + width,

        final,

        width,

        label="Final RSS",

    )

    ax.set_xticks(x)

    ax.set_xticklabels(labels)

    ax.set_ylabel(
        "Process RSS (MiB)"
    )

    ax.set_title(
        "Memory Usage Comparison"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(

        output_dir / "memory_comparison.png",

        dpi=150,

    )

    plt.close(fig)


def plot_throughput_comparison(
    summaries,
    output_dir,
):

    labels = [

        item["mode"]

        for item in summaries

    ]

    values = [

        item["throughput_images_per_second"]

        for item in summaries

    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Images per second"
    )

    ax.set_title(
        "Inference Throughput Comparison"
    )

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


def plot_latency_comparison(
    summaries,
    output_dir,
):

    labels = [

        item["mode"]

        for item in summaries

    ]

    means = [

        item["mean_latency_ms"]

        for item in summaries

    ]

    p95 = [

        item["p95_latency_ms"]

        for item in summaries

    ]

    x = np.arange(
        len(labels)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(

        x - width / 2,

        means,

        width,

        label="Mean",

    )

    ax.bar(

        x + width / 2,

        p95,

        width,

        label="P95",

    )

    ax.set_xticks(x)

    ax.set_xticklabels(labels)

    ax.set_ylabel(
        "Latency (ms)"
    )

    ax.set_title(
        "Inference Latency Comparison"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(

        output_dir / "latency_comparison.png",

        dpi=150,

    )

    plt.close(fig)


# ======================================================================
# AUTOMATED TEST HELPERS
# ======================================================================

def create_test_model(
    output_path,
    image_size=32,
):

    import torch
    import torch.nn as nn

    class TinyCNN(nn.Module):

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

    model = TinyCNN().eval()

    torch.onnx.export(

        model,

        torch.randn(

            1,

            3,

            image_size,

            image_size,

        ),

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
    ).convert("RGB")


# ======================================================================
# AUTOMATED TESTS
# ======================================================================

def run_tests():

    print(
        "\nRunning Lesson 82 tests...\n"
    )

    test_dir = Path(
        "runs/lesson82_tests"
    )

    test_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    set_seed(42)

    # --------------------------------------------------------------
    # Test 1: Image preprocessing shape
    # --------------------------------------------------------------

    image = make_test_image(1)

    tensor = preprocess_image(
        image,
        32,
    )

    assert tensor.shape == (
        1,
        3,
        32,
        32,
    )

    print(
        "Test 1 passed: Image preprocessing shape."
    )

    # --------------------------------------------------------------
    # Test 2: Input dtype
    # --------------------------------------------------------------

    assert tensor.dtype == np.float32

    print(
        "Test 2 passed: Float32 input."
    )

    # --------------------------------------------------------------
    # Test 3: Normalized range
    # --------------------------------------------------------------

    assert tensor.min() >= -1.01

    assert tensor.max() <= 1.01

    print(
        "Test 3 passed: Normalization range."
    )

    # --------------------------------------------------------------
    # Test 4: Contiguous memory
    # --------------------------------------------------------------

    assert tensor.flags["C_CONTIGUOUS"]

    print(
        "Test 4 passed: Contiguous input memory."
    )

    # --------------------------------------------------------------
    # Test 5: Reusable input buffer
    # --------------------------------------------------------------

    buffer = np.empty(

        (1, 3, 32, 32),

        dtype=np.float32,

    )

    returned = preprocess_into(

        image,

        32,

        buffer,

    )

    assert returned is buffer

    print(
        "Test 5 passed: Input buffer reused."
    )

    # --------------------------------------------------------------
    # Test 6: Reusable input numerical equivalence
    # --------------------------------------------------------------

    assert np.allclose(
        tensor,
        buffer,
        atol=1e-6,
    )

    print(
        "Test 6 passed: Reusable input numerical equivalence."
    )

    # --------------------------------------------------------------
    # Test 7: Input buffer shape
    # --------------------------------------------------------------

    assert buffer.shape == (
        1,
        3,
        32,
        32,
    )

    print(
        "Test 7 passed: Input buffer dimensions."
    )

    # --------------------------------------------------------------
    # Test 8: MiB conversion
    # --------------------------------------------------------------

    assert np.isclose(
        bytes_to_mib(1024 * 1024),
        1.0,
    )

    print(
        "Test 8 passed: Memory unit conversion."
    )

    # --------------------------------------------------------------
    # Test 9: RSS measurement
    # --------------------------------------------------------------

    rss = get_rss_bytes()

    assert rss > 0

    print(
        "Test 9 passed: Process RSS measurement."
    )

    # --------------------------------------------------------------
    # Test 10: Memory sampler
    # --------------------------------------------------------------

    sampler = MemorySampler(
        interval_seconds=0.002
    )

    sampler.start()

    time.sleep(0.02)

    sampler.stop()

    summary = sampler.summary()

    assert summary["peak_rss_mib"] > 0

    print(
        "Test 10 passed: Memory sampling."
    )

    # --------------------------------------------------------------
    # Test 11: Model export
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
        "Test 11 passed: ONNX model export."
    )

    # --------------------------------------------------------------
    # Test 12: ONNX graph validation
    # --------------------------------------------------------------

    import onnx

    graph = onnx.load(
        str(model_path)
    )

    onnx.checker.check_model(
        graph
    )

    print(
        "Test 12 passed: ONNX graph validation."
    )

    # --------------------------------------------------------------
    # Test 13: ONNX Runtime session
    # --------------------------------------------------------------

    import onnxruntime as ort

    session = ort.InferenceSession(

        str(model_path),

        providers=[
            "CPUExecutionProvider"
        ],

    )

    assert session is not None

    print(
        "Test 13 passed: ONNX Runtime session."
    )

    # --------------------------------------------------------------
    # Test 14: Model metadata
    # --------------------------------------------------------------

    input_info, output_info = inspect_model(
        session
    )

    assert input_info.shape[1:] == [
        3,
        32,
        32,
    ]

    assert output_info.shape[1] == 10

    print(
        "Test 14 passed: Model metadata."
    )

    # --------------------------------------------------------------
    # Test 15: Standard inference
    # --------------------------------------------------------------

    input_name = input_info.name

    output = session.run(

        None,

        {
            input_name: tensor,
        },

    )[0]

    assert output.shape == (
        1,
        10,
    )

    print(
        "Test 15 passed: Standard inference."
    )

    # --------------------------------------------------------------
    # Test 16: Reusable input inference
    # --------------------------------------------------------------

    predictions = []

    for index in range(3):

        test_image = make_test_image(index)

        preprocess_into(

            test_image,

            32,

            buffer,

        )

        output = session.run(

            None,

            {
                input_name: buffer,
            },

        )[0]

        predictions.append(
            int(np.argmax(output[0]))
        )

    assert len(predictions) == 3

    print(
        "Test 16 passed: Reusable input inference."
    )

    # --------------------------------------------------------------
    # Test 17: Reusable I/O binding construction
    # --------------------------------------------------------------

    engine = ReusableIOBinding(

        session,

        32,

    )

    assert engine.input_buffer.shape == (
        1,
        3,
        32,
        32,
    )

    assert engine.output_buffer.shape == (
        1,
        10,
    )

    print(
        "Test 17 passed: Reusable I/O buffers."
    )

    # --------------------------------------------------------------
    # Test 18: I/O binding inference
    # --------------------------------------------------------------

    prediction = engine.infer(
        image
    )

    assert 0 <= prediction < 10

    print(
        "Test 18 passed: I/O Binding inference."
    )

    # --------------------------------------------------------------
    # Test 19: Standard and reusable predictions
    # --------------------------------------------------------------

    standard_output = session.run(

        None,

        {
            input_name: preprocess_image(
                image,
                32,
            ),
        },

    )[0]

    standard_prediction = int(
        np.argmax(standard_output[0])
    )

    reusable_prediction = engine.infer(
        image
    )

    assert standard_prediction == reusable_prediction

    print(
        "Test 19 passed: Prediction equivalence."
    )

    # --------------------------------------------------------------
    # Test 20: Output buffer remains allocated
    # --------------------------------------------------------------

    output_pointer_before = (
        engine.output_buffer.ctypes.data
    )

    engine.infer(
        make_test_image(20)
    )

    output_pointer_after = (
        engine.output_buffer.ctypes.data
    )

    assert (
        output_pointer_before
        ==
        output_pointer_after
    )

    print(
        "Test 20 passed: Output buffer reuse."
    )

    # --------------------------------------------------------------
    # Test 21: Input buffer remains allocated
    # --------------------------------------------------------------

    input_pointer_before = (
        engine.input_buffer.ctypes.data
    )

    engine.infer(
        make_test_image(21)
    )

    input_pointer_after = (
        engine.input_buffer.ctypes.data
    )

    assert (
        input_pointer_before
        ==
        input_pointer_after
    )

    print(
        "Test 21 passed: Input buffer reuse."
    )

    # --------------------------------------------------------------
    # Test 22: Repeated I/O Binding inference
    # --------------------------------------------------------------

    for index in range(10):

        prediction = engine.infer(
            make_test_image(index)
        )

        assert 0 <= prediction < 10

    print(
        "Test 22 passed: Repeated I/O Binding inference."
    )

    # --------------------------------------------------------------
    # Test 23: Output buffer finite values
    # --------------------------------------------------------------

    assert np.isfinite(
        engine.output_buffer
    ).all()

    print(
        "Test 23 passed: Finite output buffer."
    )

    # --------------------------------------------------------------
    # Test 24: Streaming result count
    # --------------------------------------------------------------

    test_images = [

        make_test_image(index)

        for index in range(5)

    ]

    result = run_reusable_io(

        session,

        test_images,

        32,

    )

    assert len(
        result["predictions"]
    ) == 5

    print(
        "Test 24 passed: Streaming result count."
    )

    # --------------------------------------------------------------
    # Test 25: Benchmark timing
    # --------------------------------------------------------------

    assert result["elapsed_seconds"] > 0

    assert result["throughput"] > 0

    print(
        "Test 25 passed: Benchmark timing."
    )

    # --------------------------------------------------------------
    # Test 26: Benchmark latency count
    # --------------------------------------------------------------

    assert len(
        result["latencies_ms"]
    ) == 5

    print(
        "Test 26 passed: Latency measurement."
    )

    # --------------------------------------------------------------
    # Test 27: Memory report
    # --------------------------------------------------------------

    assert result["memory"]["peak_rss_mib"] > 0

    print(
        "Test 27 passed: Memory report."
    )

    # --------------------------------------------------------------
    # Test 28: Prediction class range
    # --------------------------------------------------------------

    assert all(

        0 <= value < 10

        for value in result["predictions"]

    )

    print(
        "Test 28 passed: Prediction class range."
    )

    # --------------------------------------------------------------
    # Test 29: Memory summary consistency
    # --------------------------------------------------------------

    memory = result["memory"]

    assert memory["peak_rss_mib"] >= 0

    assert memory["final_rss_mib"] >= 0

    print(
        "Test 29 passed: Memory summary validity."
    )

    # --------------------------------------------------------------
    # Test 30: Model file size
    # --------------------------------------------------------------

    size = get_model_file_size(
        model_path
    )

    assert size > 0

    print(
        "Test 30 passed: Model file size measurement."
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
            "Lesson 82 — Memory Optimization"
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

        choices=[

            "CPUExecutionProvider",

            "CUDAExecutionProvider",

        ],

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

        "--intra-threads",

        type=int,

        default=1,

    )

    parser.add_argument(

        "--output-dir",

        type=str,

        default="runs/lesson82",

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

    if args.intra_threads < 0:

        raise ValueError(
            "intra-threads cannot be negative."
        )

    set_seed(42)

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    print(
        "\n" + "=" * 75
    )

    print(
        "LESSON 82 — MEMORY OPTIMIZATION"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------------
    # LOAD IMAGES
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
        f"\nModel file size: {get_model_file_size(model_path):.3f} MiB"
    )

    # --------------------------------------------------------------
    # CREATE SESSION
    # --------------------------------------------------------------

    session = create_session(

        model_path,

        args.provider,

        args.intra_threads,

    )

    input_info, output_info = inspect_model(
        session
    )

    # --------------------------------------------------------------
    # INPUT SIZE VALIDATION
    # --------------------------------------------------------------

    expected_height = input_info.shape[2]

    expected_width = input_info.shape[3]

    if isinstance(expected_height, int):

        if expected_height != args.image_size:

            raise ValueError(

                f"Model expects height {expected_height}; "
                f"you supplied image-size {args.image_size}."

            )

    if isinstance(expected_width, int):

        if expected_width != args.image_size:

            raise ValueError(

                f"Model expects width {expected_width}; "
                f"you supplied image-size {args.image_size}."

            )

    # --------------------------------------------------------------
    # BASELINE MEMORY
    # --------------------------------------------------------------

    gc.collect()

    baseline_rss = get_rss_mib()

    print(
        f"\nProcess RSS before benchmarks: {baseline_rss:.2f} MiB"
    )

    cuda_before = get_torch_cuda_memory()

    # --------------------------------------------------------------
    # STANDARD INFERENCE
    # --------------------------------------------------------------

    print(
        "\nRunning standard inference..."
    )

    standard = run_standard(

        session,

        images,

        args.image_size,

    )

    # --------------------------------------------------------------
    # REUSABLE INPUT
    # --------------------------------------------------------------

    print(
        "\nRunning reusable-input inference..."
    )

    reusable_input = run_reusable_input(

        session,

        images,

        args.image_size,

    )

    # --------------------------------------------------------------
    # REUSABLE INPUT AND OUTPUT
    # --------------------------------------------------------------

    print(
        "\nRunning reusable input/output inference..."
    )

    reusable_io = run_reusable_io(

        session,

        images,

        args.image_size,

    )

    # --------------------------------------------------------------
    # PREDICTION VALIDATION
    # --------------------------------------------------------------

    reference = standard["predictions"]

    if reusable_input["predictions"] != reference:

        raise RuntimeError(
            "Reusable-input predictions differ from standard."
        )

    if reusable_io["predictions"] != reference:

        raise RuntimeError(
            "Reusable I/O predictions differ from standard."
        )

    print(
        "\nPrediction equivalence: PASSED."
    )

    # --------------------------------------------------------------
    # SUMMARIES
    # --------------------------------------------------------------

    results = [

        standard,

        reusable_input,

        reusable_io,

    ]

    summaries = [

        summarize_result(item)

        for item in results

    ]

    # --------------------------------------------------------------
    # PERFORMANCE REPORT
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "INFERENCE PERFORMANCE"
    )

    print(
        "=" * 75
    )

    for item in summaries:

        print(
            f"\n{item['mode'].upper()}"
        )

        print(

            f"Throughput: "
            f"{item['throughput_images_per_second']:.2f} images/s"

        )

        print(

            f"Mean latency: "
            f"{item['mean_latency_ms']:.3f} ms"

        )

        print(

            f"P50 latency: "
            f"{item['p50_latency_ms']:.3f} ms"

        )

        print(

            f"P95 latency: "
            f"{item['p95_latency_ms']:.3f} ms"

        )

        print(

            f"Peak RSS: "
            f"{item['memory']['peak_rss_mib']:.2f} MiB"

        )

        print(

            f"Observed RSS growth: "
            f"{item['memory']['observed_rss_growth_mib']:.2f} MiB"

        )

    # --------------------------------------------------------------
    # CUDA MEMORY
    # --------------------------------------------------------------

    cuda_after = get_torch_cuda_memory()

    print(
        "\nCUDA memory information:"
    )

    if cuda_before is None:

        print(
            "PyTorch CUDA memory metrics unavailable."
        )

        print(
            "For ONNX Runtime GPU memory, inspect nvidia-smi."
        )

    else:

        print(
            f"Before: {cuda_before}"
        )

        print(
            f"After: {cuda_after}"
        )

        print(
            "Note: PyTorch metrics do not include all ONNX Runtime allocations."
        )

    # --------------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------------

    plot_memory_comparison(

        summaries,

        output_dir,

    )

    plot_throughput_comparison(

        summaries,

        output_dir,

    )

    plot_latency_comparison(

        summaries,

        output_dir,

    )

    # --------------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------------

    report = {

        "lesson": 82,

        "title": "Memory Optimization",

        "model": str(model_path),

        "model_file_size_mib": get_model_file_size(
            model_path
        ),

        "provider": args.provider,

        "dataset": {

            "num_images": len(images),

            "class_names": class_names,

            "image_size": args.image_size,

        },

        "process_rss_before_benchmarks_mib": baseline_rss,

        "benchmarks": summaries,

        "prediction_equivalence": True,

        "cuda_memory_before": cuda_before,

        "cuda_memory_after": cuda_after,

    }

    report_path = (

        output_dir / "memory_optimization_report.json"

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
    # FINAL OUTPUT
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        "LESSON 82 COMPLETED."
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
        "  memory_comparison.png"
    )

    print(
        "  throughput_comparison.png"
    )

    print(
        "  latency_comparison.png"
    )

    print(
        "  memory_optimization_report.json"
    )

    print(
        "\nNext: Lesson 83 — Model Benchmarking."
    )


if __name__ == "__main__":

    main()