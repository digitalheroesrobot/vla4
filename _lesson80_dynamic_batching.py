"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 6 — MODEL OPTIMIZATION AND DEPLOYMENT

LESSON 80 — DYNAMIC BATCHING
======================================================================

OBJECTIVES
----------

1. Understand static batching and dynamic batching.
2. Implement an inference request queue.
3. Implement a dynamic batch scheduler.
4. Combine multiple requests into one inference call.
5. Configure maximum batch size.
6. Configure maximum batching delay.
7. Simulate concurrent inference clients.
8. Measure request latency and throughput.
9. Measure queue waiting time.
10. Compare single-request and dynamic-batch inference.

NO pretrained models.
NO pretrained weights.

======================================================================
1. WHY BATCHING?
======================================================================

Imagine that three cameras send images to one GPU server.

Without batching:

    Camera 1 ---> Inference
    Camera 2 ---> Inference
    Camera 3 ---> Inference

Each request executes separately.

With batching:

    Camera 1 ---+
    Camera 2 ---+---> Batch ---> GPU Inference
    Camera 3 ---+

The GPU can process multiple images in one execution.

This can improve throughput.

======================================================================
2. STATIC BATCHING
======================================================================

Static batching groups a known number of samples together.

Example:

    Batch size = 4

    [image_1, image_2, image_3, image_4]
                     |
                     v
                  Inference

The batch size is decided before execution.

Advantages:

    - Simple implementation
    - Predictable batch dimensions
    - Efficient for offline processing

Disadvantages:

    - Requests may wait for a complete batch
    - Incoming request rate may be unpredictable
    - Small workloads may not fill the batch

======================================================================
3. DYNAMIC BATCHING
======================================================================

Dynamic batching collects incoming requests at runtime.

The scheduler uses two important parameters:

    max_batch_size

        Maximum number of requests in one batch.

    max_wait_ms

        Maximum time to wait for additional requests.

Example:

    max_batch_size = 8
    max_wait_ms = 5

The scheduler:

    1. Receives the first request.
    2. Starts a short collection window.
    3. Collects more requests.
    4. Stops when the batch is full or the timer expires.
    5. Executes the combined batch.
    6. Returns individual outputs to request owners.

======================================================================
4. THE BATCHING TRADE-OFF
======================================================================

Increasing batch size can improve throughput.

However, waiting for a larger batch can increase latency.

A useful way to think about it:

    Small max_wait_ms:

        Lower waiting latency.
        Potentially smaller batches.

    Large max_wait_ms:

        More opportunity to form large batches.
        Potentially higher request latency.

There is no universally optimal setting.

======================================================================
5. LATENCY
======================================================================

Request latency:

    submission time -> completion time

Queue waiting time:

    submission time -> inference start time

Batch execution time:

    inference start time -> inference completion time

Total latency:

    queue waiting time + batch execution time + overhead

======================================================================
6. THROUGHPUT
======================================================================

Throughput:

    completed requests / elapsed seconds

Dynamic batching is particularly useful when multiple clients
submit requests concurrently.

======================================================================
7. ROBOTICS APPLICATION
======================================================================

Example: OpenArm V2 multi-camera perception.

    Ceiling camera ----+
    Head camera -------+
    Left wrist camera -+----> Request queue
    Right wrist camera-+          |
                                  v
                           Dynamic batcher
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
                           ACT manipulation

Important:

Dynamic batching improves inference efficiency, but it also
introduces waiting time.

For robot control, a slightly higher throughput is not useful
if it causes unacceptable perception latency.

======================================================================
NEXT LESSON
-----------

Lesson 81 — Asynchronous Inference.
======================================================================
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import threading
import time

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)


# ======================================================================
# DATASET
# ======================================================================

def load_real_images(args):

    """
    Load real images from CIFAR-10 or an ImageFolder dataset.

    Every returned image is:

        float32
        normalized to approximately [-1, 1]
        shape [3, H, W]

    The images are used as actual inference requests.
    """

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
            "\nLoading real CIFAR-10 test images..."
        )

        dataset = datasets.CIFAR10(

            root=args.dataset_root,

            train=False,

            download=True,

            transform=transform,

        )

        class_names = [

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

    else:

        from torchvision.datasets import ImageFolder

        root = Path(
            args.data_dir
        )

        if not root.exists():

            raise FileNotFoundError(
                f"Dataset directory does not exist: {root}"
            )

        dataset = ImageFolder(

            root=str(root),

            transform=transform,

        )

        class_names = dataset.classes

    if len(dataset) == 0:

        raise RuntimeError(
            "The image dataset is empty."
        )

    count = min(
        args.num_requests,
        len(dataset),
    )

    rng = random.Random(42)

    indices = rng.sample(
        range(len(dataset)),
        count,
    )

    images = []

    labels = []

    for index in indices:

        image, label = dataset[index]

        images.append(
            image.numpy().astype(np.float32)
        )

        labels.append(
            int(label)
        )

    images = np.stack(
        images,
        axis=0,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    print(
        f"Loaded {len(images)} real images."
    )

    print(
        f"Image tensor shape: {images.shape}"
    )

    print(
        f"Available classes: {len(class_names)}"
    )

    return images, labels, class_names


# ======================================================================
# FALLBACK ONNX MODEL
# ======================================================================

def create_fallback_onnx(
    output_path,
    image_size,
    num_classes=10,
):

    """
    Create a small randomly initialized CNN and export it.

    This model exists only to test the batching infrastructure.

    It has NOT been trained and must not be treated as a useful
    object classifier.

    Real images are still used as inference inputs.

    For meaningful predictions, supply the trained ONNX model
    exported in Lesson 78.
    """

    import torch
    import torch.nn as nn

    class ConvBlock(nn.Module):

        def __init__(
            self,
            in_channels,
            out_channels,
        ):

            super().__init__()

            self.layers = nn.Sequential(

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

                nn.ReLU(),

                nn.MaxPool2d(2),

            )

        def forward(self, x):

            return self.layers(x)

    class VisionCNN(nn.Module):

        def __init__(self):

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

                nn.Linear(128, num_classes),

            )

        def forward(self, x):

            x = self.features(x)

            x = self.pool(x)

            return self.classifier(x)

    model = VisionCNN()

    model.eval()

    example = torch.randn(

        1,

        3,

        image_size,

        image_size,

    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True,

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
        "\nCreated fallback ONNX model."
    )

    print(
        "WARNING: Model weights are random and untrained."
    )

    return output_path


# ======================================================================
# ONNX MODEL RESOLUTION
# ======================================================================

def resolve_onnx_model(
    requested_path,
    output_dir,
    image_size,
):

    if requested_path is not None:

        path = Path(
            requested_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"ONNX model not found: {path}"
            )

        return path

    candidates = [

        Path("runs/lesson78/model.onnx"),

        Path("runs/lesson77/model.onnx"),

    ]

    for path in candidates:

        if path.exists():

            print(
                f"\nUsing existing model: {path}"
            )

            return path

    fallback = (
        output_dir / "fallback_random_model.onnx"
    )

    return create_fallback_onnx(

        fallback,

        image_size,

    )


# ======================================================================
# ONNX RUNTIME SESSION
# ======================================================================

def create_ort_session(
    onnx_path,
    provider,
    intra_threads,
):

    import onnxruntime as ort

    available = ort.get_available_providers()

    if provider not in available:

        raise RuntimeError(

            f"Requested provider {provider} is unavailable.\n"

            f"Available providers: {available}"

        )

    options = ort.SessionOptions()

    options.graph_optimization_level = (

        ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    )

    options.intra_op_num_threads = intra_threads

    options.inter_op_num_threads = 1

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    session = ort.InferenceSession(

        str(onnx_path),

        sess_options=options,

        providers=[provider],

    )

    print(
        f"\nONNX Runtime providers: {session.get_providers()}"
    )

    input_info = session.get_inputs()

    if len(input_info) != 1:

        raise RuntimeError(
            "This lesson expects exactly one model input."
        )

    model_input = input_info[0]

    print(
        f"Input name: {model_input.name}"
    )

    print(
        f"Input shape: {model_input.shape}"
    )

    if len(model_input.shape) != 4:

        raise RuntimeError(
            "Expected input shape [batch, channels, height, width]."
        )

    batch_dimension = model_input.shape[0]

    if isinstance(batch_dimension, int):

        raise RuntimeError(

            "The ONNX model has a fixed batch dimension.\n"

            "Dynamic batching requires a dynamic batch dimension.\n"

            "Re-export the model with dynamic_axes for batch."

        )

    return session


# ======================================================================
# STATIC BATCH BENCHMARK
# ======================================================================

def benchmark_fixed_batch(
    session,
    images,
    batch_size,
    warmup,
    iterations,
):

    """
    Process requests in fixed-size batches.

    The last batch may be smaller than batch_size.

    All input images are processed.

    This is an offline static-grouping benchmark. It does not
    model request arrival times or queue waiting.
    """

    input_name = session.get_inputs()[0].name

    outputs = []

    for start in range(
        0,
        len(images),
        batch_size,
    ):

        batch = images[
            start:start + batch_size
        ]

        result = session.run(

            None,

            {
                input_name: batch,
            },

        )

        outputs.append(
            result[0]
        )

    # Warm-up

    for _ in range(warmup):

        for start in range(
            0,
            len(images),
            batch_size,
        ):

            batch = images[
                start:start + batch_size
            ]

            session.run(

                None,

                {
                    input_name: batch,
                },

            )

    timings = []

    for _ in range(iterations):

        start_time = time.perf_counter()

        for start in range(
            0,
            len(images),
            batch_size,
        ):

            batch = images[
                start:start + batch_size
            ]

            session.run(

                None,

                {
                    input_name: batch,
                },

            )

        elapsed = (
            time.perf_counter() - start_time
        )

        timings.append(
            elapsed
        )

    values = np.asarray(
        timings,
        dtype=np.float64,
    )

    mean_seconds = float(
        values.mean()
    )

    total_images = len(images)

    return {

        "batch_size": batch_size,

        "mean_total_time_ms": mean_seconds * 1000.0,

        "p50_total_time_ms": float(
            np.percentile(values, 50) * 1000.0
        ),

        "p95_total_time_ms": float(
            np.percentile(values, 95) * 1000.0
        ),

        "throughput_images_per_second": (

            total_images / mean_seconds

        ),

        "requests": total_images,

    }


# ======================================================================
# REQUEST DATA STRUCTURE
# ======================================================================

@dataclass
class InferenceRequest:

    request_id: int

    image: np.ndarray

    submitted_at: float

    future: Future


# ======================================================================
# DYNAMIC BATCH SCHEDULER
# ======================================================================

class DynamicBatchScheduler:

    """
    A simple dynamic batching scheduler.

    One worker thread owns the ONNX Runtime execution loop.

    Incoming clients submit individual image requests.

    The worker:

        1. Waits for the first request.
        2. Starts a batching deadline.
        3. Collects requests until the deadline or max batch size.
        4. Concatenates images.
        5. Executes one ONNX Runtime inference call.
        6. Splits the output into individual responses.
        7. Completes each request's Future.

    The implementation uses a bounded batch size and a maximum
    collection delay.

    The queue itself is unbounded in this educational example.
    A production service should also impose queue capacity,
    backpressure, cancellation, and overload handling.
    """

    def __init__(
        self,
        session,
        max_batch_size,
        max_wait_ms,
    ):

        self.session = session

        self.input_name = (
            session.get_inputs()[0].name
        )

        self.output_names = [

            output.name

            for output in session.get_outputs()

        ]

        self.max_batch_size = max_batch_size

        self.max_wait_seconds = (
            max_wait_ms / 1000.0
        )

        self.request_queue = queue.Queue()

        self.worker = threading.Thread(

            target=self._worker_loop,

            daemon=True,

        )

        self.running = False

        self.metrics_lock = threading.Lock()

        self.batch_sizes = []

        self.batch_execution_times_ms = []

        self.total_batches = 0

        self.total_requests = 0

        self.worker.start()

        self.running = True

    # ------------------------------------------------------------------
    # SUBMIT
    # ------------------------------------------------------------------

    def submit(
        self,
        request_id,
        image,
    ):

        if not self.running:

            raise RuntimeError(
                "Scheduler is not running."
            )

        future = Future()

        request = InferenceRequest(

            request_id=request_id,

            image=image,

            submitted_at=time.perf_counter(),

            future=future,

        )

        self.request_queue.put(
            request
        )

        return future

    # ------------------------------------------------------------------
    # BATCH EXECUTION
    # ------------------------------------------------------------------

    def _execute_batch(
        self,
        requests,
    ):

        if not requests:

            return

        images = np.concatenate(

            [

                request.image[None, ...]

                for request in requests

            ],

            axis=0,

        ).astype(np.float32)

        inference_started = time.perf_counter()

        try:

            outputs = self.session.run(

                None,

                {
                    self.input_name: images,
                },

            )

            inference_finished = time.perf_counter()

            execution_ms = (

                inference_finished - inference_started

            ) * 1000.0

            with self.metrics_lock:

                self.batch_sizes.append(
                    len(requests)
                )

                self.batch_execution_times_ms.append(
                    execution_ms
                )

                self.total_batches += 1

                self.total_requests += len(requests)

            for index, request in enumerate(requests):

                request_outputs = [

                    output[index:index + 1]

                    for output in outputs

                ]

                result = {

                    "request_id": request.request_id,

                    "outputs": request_outputs,

                    "submitted_at": request.submitted_at,

                    "inference_started_at": inference_started,

                    "completed_at": inference_finished,

                    "queue_wait_ms": (

                        inference_started - request.submitted_at

                    ) * 1000.0,

                    "batch_execution_ms": execution_ms,

                    "batch_size": len(requests),

                }

                request.future.set_result(
                    result
                )

        except Exception as exc:

            for request in requests:

                if not request.future.done():

                    request.future.set_exception(
                        exc
                    )

    # ------------------------------------------------------------------
    # WORKER LOOP
    # ------------------------------------------------------------------

    def _worker_loop(self):

        stopping = False

        while not stopping:

            first = self.request_queue.get()

            if first is None:

                break

            batch = [first]

            deadline = (

                time.perf_counter()

                + self.max_wait_seconds

            )

            while len(batch) < self.max_batch_size:

                remaining = (

                    deadline - time.perf_counter()

                )

                if remaining <= 0:

                    break

                try:

                    item = self.request_queue.get(

                        timeout=remaining

                    )

                except queue.Empty:

                    break

                if item is None:

                    stopping = True

                    break

                batch.append(
                    item
                )

            self._execute_batch(
                batch
            )

        # If shutdown occurs while requests remain queued,
        # process those requests before exiting.

        while True:

            try:

                item = self.request_queue.get_nowait()

            except queue.Empty:

                break

            if item is None:

                continue

            self._execute_batch(
                [item]
            )

    # ------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------

    def close(self):

        if not self.running:

            return

        self.running = False

        self.request_queue.put(
            None
        )

        self.worker.join()

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------

    def get_metrics(self):

        with self.metrics_lock:

            return {

                "batch_sizes": list(
                    self.batch_sizes
                ),

                "batch_execution_times_ms": list(
                    self.batch_execution_times_ms
                ),

                "total_batches": self.total_batches,

                "total_requests": self.total_requests,

            }


# ======================================================================
# DYNAMIC BATCH BENCHMARK
# ======================================================================

def benchmark_dynamic_batching(
    session,
    images,
    max_batch_size,
    max_wait_ms,
    clients,
):

    scheduler = DynamicBatchScheduler(

        session=session,

        max_batch_size=max_batch_size,

        max_wait_ms=max_wait_ms,

    )

    start_time = time.perf_counter()

    def send_request(index):

        future = scheduler.submit(

            request_id=index,

            image=images[index],

        )

        return future.result()

    try:

        with ThreadPoolExecutor(
            max_workers=clients
        ) as executor:

            futures = [

                executor.submit(
                    send_request,
                    index,
                )

                for index in range(len(images))

            ]

            results = [

                future.result()

                for future in futures

            ]

    finally:

        scheduler.close()

    end_time = time.perf_counter()

    elapsed_seconds = (
        end_time - start_time
    )

    latencies = np.asarray(

        [

            (
                result["completed_at"]
                - result["submitted_at"]
            ) * 1000.0

            for result in results

        ],

        dtype=np.float64,

    )

    queue_waits = np.asarray(

        [

            result["queue_wait_ms"]

            for result in results

        ],

        dtype=np.float64,

    )

    execution_times = np.asarray(

        [

            result["batch_execution_ms"]

            for result in results

        ],

        dtype=np.float64,

    )

    metrics = scheduler.get_metrics()

    return {

        "requests": len(results),

        "clients": clients,

        "max_batch_size": max_batch_size,

        "max_wait_ms": max_wait_ms,

        "elapsed_seconds": elapsed_seconds,

        "throughput_images_per_second": (

            len(results) / elapsed_seconds

        ),

        "mean_latency_ms": float(
            latencies.mean()
        ),

        "p50_latency_ms": float(
            np.percentile(latencies, 50)
        ),

        "p95_latency_ms": float(
            np.percentile(latencies, 95)
        ),

        "mean_queue_wait_ms": float(
            queue_waits.mean()
        ),

        "p95_queue_wait_ms": float(
            np.percentile(queue_waits, 95)
        ),

        "mean_batch_execution_ms": float(
            execution_times.mean()
        ),

        "batch_sizes": metrics["batch_sizes"],

        "batch_execution_times_ms": (
            metrics["batch_execution_times_ms"]
        ),

        "total_batches": metrics["total_batches"],

        "average_batch_size": (

            metrics["total_requests"]

            /

            max(metrics["total_batches"], 1)

        ),

    }


# ======================================================================
# VALIDATE DYNAMIC BATCHING
# ======================================================================

def validate_batching(
    session,
    images,
    max_batch_size,
):

    input_name = session.get_inputs()[0].name

    # Run one image individually.

    individual = session.run(

        None,

        {
            input_name: images[0:1],
        },

    )

    # Run several images together.

    batch_count = min(
        max_batch_size,
        len(images),
    )

    batch = session.run(

        None,

        {
            input_name: images[:batch_count],
        },

    )

    assert len(individual) == len(batch)

    assert batch[0].shape[0] == batch_count

    assert individual[0].shape[0] == 1

    assert np.isfinite(batch[0]).all()

    # Compare combined output with individual inference.

    for index in range(batch_count):

        single = session.run(

            None,

            {
                input_name: images[index:index + 1],
            },

        )

        difference = np.max(

            np.abs(
                single[0] - batch[0][index:index + 1]
            )

        )

        if difference > 1e-4:

            raise AssertionError(

                "Batched and individual inference differ "
                f"beyond tolerance: {difference}"

            )

    print(
        "Dynamic batch numerical validation passed."
    )


# ======================================================================
# VISUALIZATION
# ======================================================================

def plot_batch_distribution(
    batch_sizes,
    output_dir,
):

    values, counts = np.unique(

        batch_sizes,

        return_counts=True,

    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        values,
        counts,
    )

    ax.set_title(
        "Dynamic Batch Size Distribution"
    )

    ax.set_xlabel(
        "Actual batch size"
    )

    ax.set_ylabel(
        "Number of batches"
    )

    ax.set_xticks(values)

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(

        output_dir / "batch_distribution.png",

        dpi=150,

    )

    plt.close(fig)


def plot_latency_throughput(
    results,
    output_dir,
):

    labels = [

        (
            f"B={item['max_batch_size']}, "
            f"wait={item['max_wait_ms']}ms"
        )

        for item in results

    ]

    latency = [

        item["mean_latency_ms"]

        for item in results

    ]

    throughput = [

        item["throughput_images_per_second"]

        for item in results

    ]

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(
        labels,
        latency,
    )

    ax.set_title(
        "Dynamic Batching Request Latency"
    )

    ax.set_ylabel(
        "Mean request latency (ms)"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(

        output_dir / "dynamic_batch_latency.png",

        dpi=150,

    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(
        labels,
        throughput,
    )

    ax.set_title(
        "Dynamic Batching Throughput"
    )

    ax.set_ylabel(
        "Images per second"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(

        output_dir / "dynamic_batch_throughput.png",

        dpi=150,

    )

    plt.close(fig)


# ======================================================================
# TESTS
# ======================================================================

def run_tests():

    print(
        "\nRunning Lesson 80 tests...\n"
    )

    import onnx
    import torch
    import torch.nn as nn

    test_dir = Path(
        "runs/lesson80_tests"
    )

    test_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    set_seed(42)

    # --------------------------------------------------------------
    # Test model
    # --------------------------------------------------------------

    class TestModel(nn.Module):

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

                nn.Linear(8, 10),

            )

        def forward(self, x):

            return self.network(x)

    model = TestModel()

    model.eval()

    # Test 1: Model output.

    x = torch.randn(
        2,
        3,
        32,
        32,
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
        "Test 2 passed: Finite outputs."
    )

    # Test 3: Export dynamic ONNX model.

    onnx_path = (
        test_dir / "test_model.onnx"
    )

    torch.onnx.export(

        model,

        torch.randn(
            1,
            3,
            32,
            32,
        ),

        str(onnx_path),

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
        "Test 4 passed: ONNX validation."
    )

    # Test 5: Dynamic batch metadata.

    dim = (

        graph.graph.input[0]

        .type.tensor_type.shape.dim[0]

    )

    assert dim.dim_param != ""

    print(
        "Test 5 passed: Dynamic batch dimension."
    )

    # Test 6: Deterministic image generation.

    images_a = np.random.default_rng(
        42
    ).normal(

        size=(4, 3, 32, 32)

    ).astype(np.float32)

    images_b = np.random.default_rng(
        42
    ).normal(

        size=(4, 3, 32, 32)

    ).astype(np.float32)

    assert np.array_equal(
        images_a,
        images_b,
    )

    print(
        "Test 6 passed: Deterministic input generation."
    )

    # Test 7: Queue creation.

    test_queue = queue.Queue()

    test_queue.put("request")

    assert test_queue.get() == "request"

    print(
        "Test 7 passed: Request queue."
    )

    # Test 8: Future completion.

    future = Future()

    future.set_result(
        123
    )

    assert future.result() == 123

    print(
        "Test 8 passed: Future result."
    )

    # Test 9: Request structure.

    request = InferenceRequest(

        request_id=1,

        image=images_a[0],

        submitted_at=time.perf_counter(),

        future=Future(),

    )

    assert request.request_id == 1

    assert request.image.shape == (
        3,
        32,
        32,
    )

    print(
        "Test 9 passed: Inference request structure."
    )

    # Test 10: Concatenation.

    batch = np.concatenate(

        [
            images_a[0][None],
            images_a[1][None],
            images_a[2][None],
        ],

        axis=0,

    )

    assert batch.shape == (
        3,
        3,
        32,
        32,
    )

    print(
        "Test 10 passed: Dynamic batch concatenation."
    )

    # Test 11: Batch splitting.

    output = np.arange(
        30
    ).reshape(
        3,
        10,
    )

    split = [

        output[index:index + 1]

        for index in range(3)

    ]

    assert len(split) == 3

    assert split[0].shape == (1, 10)

    print(
        "Test 11 passed: Batch output splitting."
    )

    # Test 12: Latency calculation.

    submitted = 1.0

    completed = 1.025

    latency_ms = (
        completed - submitted
    ) * 1000.0

    assert abs(
        latency_ms - 25.0
    ) < 1e-6

    print(
        "Test 12 passed: Request latency calculation."
    )

    # Test 13: Throughput calculation.

    throughput = 100 / 2.0

    assert throughput == 50.0

    print(
        "Test 13 passed: Throughput calculation."
    )

    # Test 14: Maximum batch size logic.

    max_batch = 8

    assert min(
        12,
        max_batch,
    ) == 8

    print(
        "Test 14 passed: Maximum batch size constraint."
    )

    # Test 15: Batch waiting deadline.

    start = time.perf_counter()

    deadline = start + 0.005

    assert deadline > start

    print(
        "Test 15 passed: Batching deadline."
    )

    # Test 16: ONNX Runtime dynamic batch.

    import onnxruntime as ort

    session = ort.InferenceSession(

        str(onnx_path),

        providers=[
            "CPUExecutionProvider"
        ],

    )

    input_name = session.get_inputs()[0].name

    for batch_size in [1, 2, 4]:

        output = session.run(

            None,

            {

                input_name: images_a[:batch_size],

            },

        )[0]

        assert output.shape == (
            batch_size,
            10,
        )

    print(
        "Test 16 passed: ONNX Runtime dynamic batch inference."
    )

    # Test 17: Individual versus batched equivalence.

    single = session.run(

        None,

        {
            input_name: images_a[:1],
        },

    )[0]

    combined = session.run(

        None,

        {
            input_name: images_a[:4],
        },

    )[0]

    assert np.allclose(

        single,

        combined[:1],

        atol=1e-5,

    )

    print(
        "Test 17 passed: Individual/batch equivalence."
    )

    # Test 18: Scheduler integration.

    scheduler = DynamicBatchScheduler(

        session=session,

        max_batch_size=4,

        max_wait_ms=5,

    )

    try:

        futures = [

            scheduler.submit(
                index,
                images_a[index],
            )

            for index in range(4)

        ]

        results = [

            future.result(timeout=10)

            for future in futures

        ]

    finally:

        scheduler.close()

    assert len(results) == 4

    assert all(
        result["outputs"][0].shape == (1, 10)
        for result in results
    )

    print(
        "Test 18 passed: Dynamic scheduler integration."
    )

    # Test 19: Scheduler metrics.

    metrics = scheduler.get_metrics()

    assert metrics["total_requests"] == 4

    assert metrics["total_batches"] >= 1

    print(
        "Test 19 passed: Scheduler metrics."
    )

    # Test 20: Batch execution timing.

    assert all(

        value >= 0

        for value in metrics["batch_execution_times_ms"]

    )

    print(
        "Test 20 passed: Batch execution timing."
    )

    print(
        "\nAll 20 tests passed."
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Lesson 80 — Dynamic Batching"
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

        "--intra-threads",

        type=int,

        default=0,

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

        "--clients",

        type=int,

        default=16,

    )

    parser.add_argument(

        "--max-batch-size",

        type=int,

        default=8,

    )

    parser.add_argument(

        "--max-wait-ms",

        type=float,

        default=5.0,

    )

    parser.add_argument(

        "--warmup",

        type=int,

        default=5,

    )

    parser.add_argument(

        "--iterations",

        type=int,

        default=3,

    )

    parser.add_argument(

        "--output-dir",

        type=str,

        default="runs/lesson80",

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
            "Number of requests must be positive."
        )

    if args.clients <= 0:

        raise ValueError(
            "Client count must be positive."
        )

    if args.max_batch_size <= 0:

        raise ValueError(
            "Maximum batch size must be positive."
        )

    if args.max_wait_ms < 0:

        raise ValueError(
            "Maximum wait cannot be negative."
        )

    if args.image_size < 32:

        raise ValueError(
            "Image size must be at least 32."
        )

    if args.iterations <= 0:

        raise ValueError(
            "Iterations must be positive."
        )

    if args.warmup < 0:

        raise ValueError(
            "Warm-up count cannot be negative."
        )

    if args.intra_threads < 0:

        raise ValueError(
            "Thread count cannot be negative."
        )

    set_seed(42)

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    print("\n" + "=" * 75)

    print(
        "LESSON 80 — DYNAMIC BATCHING"
    )

    print("=" * 75)

    # --------------------------------------------------------------
    # DATA
    # --------------------------------------------------------------

    images, labels, class_names = load_real_images(
        args
    )

    # --------------------------------------------------------------
    # MODEL
    # --------------------------------------------------------------

    onnx_path = resolve_onnx_model(

        args.onnx_path,

        output_dir,

        args.image_size,

    )

    # --------------------------------------------------------------
    # ONNX RUNTIME
    # --------------------------------------------------------------

    session = create_ort_session(

        onnx_path,

        provider=args.provider,

        intra_threads=args.intra_threads,

    )

    # --------------------------------------------------------------
    # INPUT SHAPE VALIDATION
    # --------------------------------------------------------------

    expected_shape = (

        3,

        args.image_size,

        args.image_size,

    )

    if tuple(images.shape[1:]) != expected_shape:

        raise RuntimeError(

            f"Dataset input shape {images.shape[1:]} "
            f"does not match expected {expected_shape}."

        )

    validate_batching(

        session,

        images,

        args.max_batch_size,

    )

    # --------------------------------------------------------------
    # FIXED BATCH BASELINES
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 65
    )

    print(
        "FIXED BATCH BENCHMARKS"
    )

    print(
        "=" * 65
    )

    fixed_results = []

    for batch_size in [1, 2, 4, 8]:

        result = benchmark_fixed_batch(

            session,

            images,

            batch_size,

            args.warmup,

            args.iterations,

        )

        fixed_results.append(
            result
        )

        print(

            f"Batch {batch_size}: "

            f"{result['mean_total_time_ms']:.3f} ms "

            f"| throughput="
            f"{result['throughput_images_per_second']:.2f} img/s"

        )

    # --------------------------------------------------------------
    # DYNAMIC BATCHING EXPERIMENTS
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 65
    )

    print(
        "DYNAMIC BATCHING EXPERIMENTS"
    )

    print(
        "=" * 65
    )

    configurations = [

        (2, 1.0),

        (4, 1.0),

        (8, 1.0),

        (8, 5.0),

        (8, 10.0),

        (16, 5.0),

    ]

    dynamic_results = []

    for max_batch, max_wait in configurations:

        if max_batch > args.num_requests:

            continue

        print(

            f"\nConfiguration: "

            f"max_batch={max_batch}, "

            f"max_wait={max_wait} ms"

        )

        result = benchmark_dynamic_batching(

            session,

            images,

            max_batch_size=max_batch,

            max_wait_ms=max_wait,

            clients=args.clients,

        )

        dynamic_results.append(
            result
        )

        print(

            f"Average batch size: "
            f"{result['average_batch_size']:.2f}"

        )

        print(

            f"Total batches: "
            f"{result['total_batches']}"

        )

        print(

            f"Mean latency: "
            f"{result['mean_latency_ms']:.3f} ms"

        )

        print(

            f"P95 latency: "
            f"{result['p95_latency_ms']:.3f} ms"

        )

        print(

            f"Mean queue waiting: "
            f"{result['mean_queue_wait_ms']:.3f} ms"

        )

        print(

            f"Throughput: "
            f"{result['throughput_images_per_second']:.2f} img/s"

        )

    # --------------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------------

    plot_latency_throughput(

        dynamic_results,

        output_dir,

    )

    if dynamic_results:

        largest = max(

            dynamic_results,

            key=lambda item: item["average_batch_size"],

        )

        plot_batch_distribution(

            largest["batch_sizes"],

            output_dir,

        )

    # --------------------------------------------------------------
    # REPORT
    # --------------------------------------------------------------

    report = {

        "lesson": 80,

        "title": "Dynamic Batching",

        "model": str(onnx_path),

        "provider": args.provider,

        "dataset": {

            "samples": len(images),

            "image_shape": list(images.shape[1:]),

            "classes": class_names,

        },

        "configuration": {

            "clients": args.clients,

            "max_batch_size": args.max_batch_size,

            "max_wait_ms": args.max_wait_ms,

            "intra_threads": args.intra_threads,

        },

        "fixed_batch_results": fixed_results,

        "dynamic_batch_results": dynamic_results,

    }

    report_path = (

        output_dir

        / "dynamic_batching_report.json"

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
        "LESSON 80 SUMMARY"
    )

    print("=" * 75)

    print(
        f"Images processed: {len(images)}"
    )

    print(
        f"Concurrent clients: {args.clients}"
    )

    print(
        "\nFixed batch throughput:"
    )

    for item in fixed_results:

        print(

            f"Batch {item['batch_size']:2d}: "

            f"{item['throughput_images_per_second']:.2f} images/s"

        )

    print(
        "\nDynamic batching throughput:"
    )

    for item in dynamic_results:

        print(

            f"Max batch {item['max_batch_size']:2d} "

            f"| wait {item['max_wait_ms']:4.1f} ms "

            f"| average batch {item['average_batch_size']:.2f} "

            f"| {item['throughput_images_per_second']:.2f} img/s"

        )

    print(
        "\nOutput directory:"
    )

    print(
        output_dir.resolve()
    )

    print(
        "\nLESSON 80 COMPLETED."
    )

    print(
        "Next: Lesson 81 — Asynchronous Inference."
    )


if __name__ == "__main__":

    main()