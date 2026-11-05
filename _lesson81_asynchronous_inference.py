"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 6 — MODEL OPTIMIZATION AND DEPLOYMENT

LESSON 81 — ASYNCHRONOUS INFERENCE
======================================================================

LEARNING OBJECTIVES
-------------------

1. Understand synchronous and asynchronous inference.
2. Build a producer-consumer pipeline.
3. Separate image acquisition, preprocessing, inference, and results.
4. Use independent worker threads for different pipeline stages.
5. Use bounded queues to control memory and backpressure.
6. Measure end-to-end latency and throughput.
7. Compare synchronous and asynchronous execution.
8. Understand queue waiting time and stale-frame problems.
9. Design a multi-camera robotics perception pipeline.

NO pretrained models.
NO pretrained weights.

======================================================================
1. SYNCHRONOUS INFERENCE
======================================================================

A synchronous pipeline executes each stage sequentially:

    Capture image
         |
         v
    Preprocess image
         |
         v
    Run model inference
         |
         v
    Postprocess prediction
         |
         v
    Capture next image

If each stage takes:

    Capture       = 10 ms
    Preprocessing =  5 ms
    Inference     = 30 ms
    Postprocessing=  2 ms

Total time per image:

    10 + 5 + 30 + 2 = 47 ms

The pipeline cannot start processing the next image until
the current image has completed all stages.

======================================================================
2. ASYNCHRONOUS INFERENCE
======================================================================

An asynchronous pipeline overlaps independent stages:

    Acquisition worker
          |
          v
       Raw queue
          |
          v
    Preprocessing workers
          |
          v
     Inference queue
          |
          v
     Inference workers
          |
          v
      Result queue
          |
          v
    Result consumer

While one image is being processed by the model, another image
can be preprocessed and another can be acquired.

The ideal steady-state throughput is limited by the slowest
pipeline stage, not necessarily the sum of all stage times.

======================================================================
3. PRODUCER-CONSUMER ARCHITECTURE
======================================================================

Producer:

    Generates or captures incoming images.

Consumer:

    Processes images from a queue.

A queue separates production from consumption.

This is useful when camera acquisition and GPU inference
run at different speeds.

======================================================================
4. BOUNDED QUEUES AND BACKPRESSURE
======================================================================

An unbounded queue can grow indefinitely if image acquisition
is faster than inference.

A bounded queue limits the number of pending frames.

Example:

    max_queue_size = 16

When the queue is full, the producer waits until capacity
becomes available.

This is called backpressure.

In robotics, another possible strategy is dropping old frames
and processing the newest frame. This lesson implements
backpressure first, because it makes request accounting and
correctness testing easier.

======================================================================
5. THREADS AND ONNX RUNTIME
======================================================================

Python ThreadPoolExecutor and threading allow independent
pipeline stages to overlap.

ONNX Runtime provides its own internal execution threads.

Be careful:

    Too many application workers
    +
    Too many ONNX Runtime threads
    =
    CPU oversubscription.

The best configuration depends on the hardware and model.

======================================================================
6. LATENCY
======================================================================

End-to-end latency:

    result completion time - request submission time

Queue waiting time:

    time spent waiting for a worker.

Stage latency:

    preprocessing time
    inference time
    postprocessing time

Throughput:

    completed images / elapsed seconds

Higher throughput does not automatically mean lower latency.

======================================================================
7. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    Ceiling camera ----+
    Head camera -------+
    Wrist-left camera -+----> Acquisition
    Wrist-right camera-+          |
                                  v
                              Raw queue
                                  |
                                  v
                             Preprocessing
                                  |
                                  v
                             Inference queue
                                  |
                                  v
                           ONNX Runtime workers
                                  |
                                  v
                              Postprocessing
                                  |
                                  v
                         Object localization
                                  |
                                  v
                           ACT manipulation

For closed-loop robot control, always consider frame timestamps.
A high-throughput pipeline that returns stale observations
can be worse than a slower pipeline with fresh observations.

======================================================================
8. EXPERIMENTS
======================================================================

Compare:

    A. Synchronous processing
    B. One preprocessing worker + one inference worker
    C. Multiple preprocessing workers + multiple inference workers

Measure:

    - Throughput
    - Mean latency
    - P50 latency
    - P95 latency
    - Queue waiting time
    - Preprocessing time
    - Inference time

======================================================================
NEXT LESSON
-----------

Lesson 82 — Memory Optimization.
======================================================================
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)


# ======================================================================
# DATASET LOADING
# ======================================================================

def load_images(args):
    """
    Load real images.

    Default:
        CIFAR-10 test images.

    Optional:
        ImageFolder directory.

    Returns PIL images and integer labels.

    Preprocessing is deliberately performed later inside
    the asynchronous pipeline.
    """

    from torchvision.datasets import CIFAR10, ImageFolder

    if args.data_dir is None:

        print("\nDownloading/loading CIFAR-10 test images...")

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

    print(
        f"Number of classes: {len(class_names)}"
    )

    return images, labels, class_names


# ======================================================================
# IMAGE PREPROCESSING
# ======================================================================

def preprocess_image(
    image,
    image_size,
):
    """
    Convert a real RGB image into normalized NCHW float32 input.

    Output shape:

        [1, 3, H, W]

    Normalization:

        x_normalized = (x / 255 - 0.5) / 0.5

    This maps RGB values approximately to [-1, 1].
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

    array = array / 255.0

    array = (array - 0.5) / 0.5

    array = np.transpose(

        array,

        (2, 0, 1),

    )

    return np.ascontiguousarray(
        array[None, ...],
        dtype=np.float32,
    )


# ======================================================================
# FALLBACK ONNX MODEL
# ======================================================================

def create_fallback_onnx(
    output_path,
    image_size,
    num_classes=10,
):
    """
    Create a small randomly initialized CNN.

    This fallback exists only to test the inference pipeline.

    It is not trained and must not be used as a meaningful
    classifier.

    For meaningful inference, supply a trained ONNX model.
    """

    import torch
    import torch.nn as nn

    class VisionCNN(nn.Module):

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

    model = VisionCNN().eval()

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
        "Predictions from this model are not meaningful."
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
                f"Model does not exist: {path}"
            )

        return path

    candidates = [

        Path("runs/lesson78/model.onnx"),

        Path("runs/lesson77/model.onnx"),

    ]

    for candidate in candidates:

        if candidate.exists():

            print(
                f"Using existing ONNX model: {candidate}"
            )

            return candidate

    return create_fallback_onnx(

        output_dir / "fallback_random_model.onnx",

        image_size,

    )


# ======================================================================
# ONNX RUNTIME SESSION
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

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    session = ort.InferenceSession(

        str(model_path),

        sess_options=options,

        providers=[provider],

    )

    input_info = session.get_inputs()

    if len(input_info) != 1:

        raise RuntimeError(
            "This lesson expects exactly one model input."
        )

    if len(input_info[0].shape) != 4:

        raise RuntimeError(
            "Expected NCHW image input."
        )

    batch_dimension = input_info[0].shape[0]

    if isinstance(batch_dimension, int):

        raise RuntimeError(
            "Model must have a dynamic batch dimension."
        )

    print(
        f"Active providers: {session.get_providers()}"
    )

    print(
        f"Model input: {input_info[0].shape}"
    )

    return session


# ======================================================================
# REQUEST STRUCTURE
# ======================================================================

@dataclass
class FrameRequest:

    frame_id: int

    image: Image.Image

    label: int

    submitted_at: float


@dataclass
class PreparedRequest:

    request: FrameRequest

    tensor: np.ndarray

    preprocessing_ms: float

    preprocessing_finished_at: float


@dataclass
class InferenceResult:

    frame_id: int

    predicted_class: int

    confidence: float

    label: int

    submitted_at: float

    completed_at: float

    raw_queue_wait_ms: float

    preprocessing_ms: float

    inference_queue_wait_ms: float

    inference_ms: float

    total_latency_ms: float


# ======================================================================
# SYNCHRONOUS BASELINE
# ======================================================================

def run_synchronous(
    session,
    images,
    labels,
    image_size,
):

    """
    Sequential baseline:

        preprocess -> inference -> postprocess

    One frame must complete before the next frame starts.
    """

    input_name = session.get_inputs()[0].name

    results = []

    start_time = time.perf_counter()

    for index, image in enumerate(images):

        submitted_at = time.perf_counter()

        preprocess_start = time.perf_counter()

        tensor = preprocess_image(

            image,

            image_size,

        )

        preprocess_end = time.perf_counter()

        inference_start = time.perf_counter()

        output = session.run(

            None,

            {
                input_name: tensor,
            },

        )[0]

        inference_end = time.perf_counter()

        logits = output[0]

        predicted = int(
            np.argmax(logits)
        )

        confidence = float(
            np.max(
                softmax(logits)
            )
        )

        completed_at = time.perf_counter()

        results.append({

            "frame_id": index,

            "predicted_class": predicted,

            "confidence": confidence,

            "label": labels[index],

            "submitted_at": submitted_at,

            "completed_at": completed_at,

            "preprocessing_ms": (

                preprocess_end - preprocess_start

            ) * 1000.0,

            "inference_ms": (

                inference_end - inference_start

            ) * 1000.0,

            "total_latency_ms": (

                completed_at - submitted_at

            ) * 1000.0,

        })

    elapsed = (
        time.perf_counter() - start_time
    )

    return {

        "mode": "synchronous",

        "results": results,

        "elapsed_seconds": elapsed,

        "throughput": len(results) / elapsed,

    }


# ======================================================================
# SOFTMAX
# ======================================================================

def softmax(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    shifted = values - np.max(
        values
    )

    exp_values = np.exp(
        shifted
    )

    return exp_values / np.sum(
        exp_values
    )


# ======================================================================
# ASYNCHRONOUS PIPELINE
# ======================================================================

class AsyncInferencePipeline:
    """
    Threaded producer-consumer inference pipeline.

    Stages:

        Producer
            |
            v
        Raw queue
            |
            v
        Preprocessing workers
            |
            v
        Inference queue
            |
            v
        Inference workers
            |
            v
        Result queue
            |
            v
        Main-thread result collector

    Each queue is bounded.

    Queue.put() blocks when the queue is full, implementing
    backpressure.

    One ONNX Runtime session is shared among inference workers.
    ONNX Runtime InferenceSession supports concurrent run calls.

    Worker exceptions are propagated through an error queue.
    """

    def __init__(
        self,
        session,
        image_size,
        preprocessing_workers=2,
        inference_workers=2,
        queue_size=16,
    ):

        self.session = session

        self.image_size = image_size

        self.preprocessing_workers = (
            preprocessing_workers
        )

        self.inference_workers = (
            inference_workers
        )

        self.raw_queue = queue.Queue(
            maxsize=queue_size
        )

        self.inference_queue = queue.Queue(
            maxsize=queue_size
        )

        self.result_queue = queue.Queue(
            maxsize=queue_size
        )

        self.error_queue = queue.Queue()

        self.input_name = (
            session.get_inputs()[0].name
        )

        self.threads = []

        self.running = False

        self.stop_event = threading.Event()

        self.metrics_lock = threading.Lock()

        self.raw_queue_waits = []

        self.inference_queue_waits = []

        self.preprocessing_times = []

        self.inference_times = []

        self.completed_count = 0

    # ------------------------------------------------------------------
    # START
    # ------------------------------------------------------------------

    def start(self):

        if self.running:

            raise RuntimeError(
                "Pipeline is already running."
            )

        self.running = True

        self.stop_event.clear()

        for worker_id in range(
            self.preprocessing_workers
        ):

            thread = threading.Thread(

                target=self._preprocessing_worker,

                args=(worker_id,),

                daemon=True,

            )

            thread.start()

            self.threads.append(
                thread
            )

        for worker_id in range(
            self.inference_workers
        ):

            thread = threading.Thread(

                target=self._inference_worker,

                args=(worker_id,),

                daemon=True,

            )

            thread.start()

            self.threads.append(
                thread
            )

    # ------------------------------------------------------------------
    # SUBMIT IMAGE
    # ------------------------------------------------------------------

    def submit(
        self,
        frame_id,
        image,
        label,
    ):

        if not self.running:

            raise RuntimeError(
                "Pipeline has not been started."
            )

        request = FrameRequest(

            frame_id=frame_id,

            image=image,

            label=label,

            submitted_at=time.perf_counter(),

        )

        self.raw_queue.put(
            request
        )

    # ------------------------------------------------------------------
    # PREPROCESSING WORKER
    # ------------------------------------------------------------------

    def _preprocessing_worker(
        self,
        worker_id,
    ):

        while True:

            item = self.raw_queue.get()

            if item is None:

                self.raw_queue.task_done()

                break

            try:

                start = time.perf_counter()

                prepared_tensor = preprocess_image(

                    item.image,

                    self.image_size,

                )

                end = time.perf_counter()

                prepared = PreparedRequest(

                    request=item,

                    tensor=prepared_tensor,

                    preprocessing_ms=(end - start) * 1000.0,

                    preprocessing_finished_at=end,

                )

                with self.metrics_lock:

                    self.preprocessing_times.append(
                        prepared.preprocessing_ms
                    )

                    self.raw_queue_waits.append(

                        max(
                            0.0,
                            (start - item.submitted_at) * 1000.0,
                        )

                    )

                self.inference_queue.put(
                    prepared
                )

            except Exception as exc:

                self.error_queue.put(exc)

                self.stop_event.set()

            finally:

                self.raw_queue.task_done()

    # ------------------------------------------------------------------
    # INFERENCE WORKER
    # ------------------------------------------------------------------

    def _inference_worker(
        self,
        worker_id,
    ):

        while True:

            item = self.inference_queue.get()

            if item is None:

                self.inference_queue.task_done()

                break

            try:

                inference_start = time.perf_counter()

                output = self.session.run(

                    None,

                    {
                        self.input_name: item.tensor,
                    },

                )[0]

                inference_end = time.perf_counter()

                logits = output[0]

                probabilities = softmax(
                    logits
                )

                predicted = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    probabilities[predicted]
                )

                completed_at = time.perf_counter()

                request = item.request

                result = InferenceResult(

                    frame_id=request.frame_id,

                    predicted_class=predicted,

                    confidence=confidence,

                    label=request.label,

                    submitted_at=request.submitted_at,

                    completed_at=completed_at,

                    raw_queue_wait_ms=max(

                        0.0,

                        (

                            item.preprocessing_finished_at

                            - request.submitted_at

                            - item.preprocessing_ms / 1000.0

                        ) * 1000.0,

                    ),

                    preprocessing_ms=item.preprocessing_ms,

                    inference_queue_wait_ms=(

                        inference_start

                        - item.preprocessing_finished_at

                    ) * 1000.0,

                    inference_ms=(

                        inference_end

                        - inference_start

                    ) * 1000.0,

                    total_latency_ms=(

                        completed_at

                        - request.submitted_at

                    ) * 1000.0,

                )

                with self.metrics_lock:

                    self.inference_times.append(
                        result.inference_ms
                    )

                    self.completed_count += 1

                self.result_queue.put(
                    result
                )

            except Exception as exc:

                self.error_queue.put(exc)

                self.stop_event.set()

            finally:

                self.inference_queue.task_done()

    # ------------------------------------------------------------------
    # FINISH PRODUCER
    # ------------------------------------------------------------------

    def finish_input(self):

        """
        Signal that no more camera frames will arrive.

        One sentinel is sent to each preprocessing worker.
        """

        for _ in range(
            self.preprocessing_workers
        ):

            self.raw_queue.put(
                None
            )

    # ------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------

    def shutdown(self):

        """
        Wait for preprocessing to finish, then stop inference workers.

        All inference tasks are allowed to finish before shutdown.
        """

        self.raw_queue.join()

        self.inference_queue.join()

        for _ in range(
            self.inference_workers
        ):

            self.inference_queue.put(
                None
            )

        self.inference_queue.join()

        for thread in self.threads:

            thread.join()

        self.running = False

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------

    def get_metrics(self):

        with self.metrics_lock:

            return {

                "raw_queue_waits_ms": list(
                    self.raw_queue_waits
                ),

                "preprocessing_ms": list(
                    self.preprocessing_times
                ),

                "inference_queue_waits_ms": list(
                    self.inference_queue_waits
                ),

                "inference_ms": list(
                    self.inference_times
                ),

                "completed_count": self.completed_count,

            }


# ======================================================================
# ASYNCHRONOUS BENCHMARK
# ======================================================================

def run_asynchronous(
    session,
    images,
    labels,
    image_size,
    preprocessing_workers,
    inference_workers,
    queue_size,
):

    pipeline = AsyncInferencePipeline(

        session=session,

        image_size=image_size,

        preprocessing_workers=preprocessing_workers,

        inference_workers=inference_workers,

        queue_size=queue_size,

    )

    results = []

    start_time = time.perf_counter()

    pipeline.start()

    try:

        # Simulate camera acquisition.

        for index, image in enumerate(images):

            if not pipeline.error_queue.empty():

                raise pipeline.error_queue.get()

            pipeline.submit(

                frame_id=index,

                image=image,

                label=labels[index],

            )

        pipeline.finish_input()

        # Collect one output for every submitted frame.

        for _ in range(len(images)):

            if not pipeline.error_queue.empty():

                raise pipeline.error_queue.get()

            result = pipeline.result_queue.get(
                timeout=120
            )

            results.append(
                result
            )

        pipeline.shutdown()

        if not pipeline.error_queue.empty():

            raise pipeline.error_queue.get()

    except Exception:

        # Ensure workers are not abandoned after an error.
        # This is an educational pipeline; production systems
        # should implement cancellation and coordinated shutdown.

        if pipeline.running:

            pipeline.finish_input()

            try:

                pipeline.shutdown()

            except Exception:

                pass

        raise

    elapsed = (
        time.perf_counter() - start_time
    )

    results.sort(
        key=lambda item: item.frame_id
    )

    metrics = pipeline.get_metrics()

    if len(results) != len(images):

        raise RuntimeError(

            f"Expected {len(images)} results, "
            f"received {len(results)}."

        )

    return {

        "mode": "asynchronous",

        "results": results,

        "elapsed_seconds": elapsed,

        "throughput": len(results) / elapsed,

        "metrics": metrics,

    }


# ======================================================================
# SUMMARY STATISTICS
# ======================================================================

def summarize_results(result):

    records = result["results"]

    latencies = np.asarray(

        [

            item["total_latency_ms"]

            if isinstance(item, InferenceResult)

            else item["total_latency_ms"]

            for item in records

        ],

        dtype=np.float64,

    )

    return {

        "mode": result["mode"],

        "requests": len(records),

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

    }


# ======================================================================
# PLOTS
# ======================================================================

def plot_comparison(
    summaries,
    output_dir,
):

    labels = [
        item["mode"]
        for item in summaries
    ]

    throughput = [

        item["throughput_images_per_second"]

        for item in summaries

    ]

    mean_latency = [

        item["mean_latency_ms"]

        for item in summaries

    ]

    p95_latency = [

        item["p95_latency_ms"]

        for item in summaries

    ]

    # --------------------------------------------------------------
    # Throughput
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        labels,
        throughput,
    )

    ax.set_title(
        "Synchronous vs Asynchronous Throughput"
    )

    ax.set_ylabel(
        "Images per second"
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

    # --------------------------------------------------------------
    # Latency
    # --------------------------------------------------------------

    x = np.arange(
        len(labels)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(

        x - width / 2,

        mean_latency,

        width,

        label="Mean",

    )

    ax.bar(

        x + width / 2,

        p95_latency,

        width,

        label="P95",

    )

    ax.set_xticks(x)

    ax.set_xticklabels(labels)

    ax.set_title(
        "End-to-End Latency Comparison"
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

        output_dir / "latency_comparison.png",

        dpi=150,

    )

    plt.close(fig)


def plot_pipeline_stages(
    metrics,
    output_dir,
):

    names = [

        "Raw queue wait",

        "Preprocessing",

        "Inference",

    ]

    values = [

        metrics["raw_queue_waits_ms"],

        metrics["preprocessing_ms"],

        metrics["inference_ms"],

    ]

    means = [

        float(
            np.mean(value)
        ) if value else 0.0

        for value in values

    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        names,
        means,
    )

    ax.set_title(
        "Asynchronous Pipeline Stage Times"
    )

    ax.set_ylabel(
        "Mean time (ms)"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(

        output_dir / "pipeline_stage_times.png",

        dpi=150,

    )

    plt.close(fig)


# ======================================================================
# TEST HELPERS
# ======================================================================

def create_test_model(
    output_path,
    image_size=32,
):

    import torch
    import torch.nn as nn

    class TestCNN(nn.Module):

        def __init__(self):

            super().__init__()

            self.net = nn.Sequential(

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

            return self.net(x)

    model = TestCNN().eval()

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


# ======================================================================
# AUTOMATED TESTS
# ======================================================================

def run_tests():

    print(
        "\nRunning Lesson 81 tests...\n"
    )

    test_dir = Path(
        "runs/lesson81_tests"
    )

    test_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    set_seed(42)

    # --------------------------------------------------------------
    # Test 1: Real image preprocessing
    # --------------------------------------------------------------

    image = Image.fromarray(

        np.random.default_rng(1).integers(

            0,

            256,

            size=(48, 64, 3),

            dtype=np.uint8,

        )

    )

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
    # Test 2: Float32 dtype
    # --------------------------------------------------------------

    assert tensor.dtype == np.float32

    print(
        "Test 2 passed: Float32 preprocessing."
    )

    # --------------------------------------------------------------
    # Test 3: Normalization range
    # --------------------------------------------------------------

    assert tensor.min() >= -1.01

    assert tensor.max() <= 1.01

    print(
        "Test 3 passed: Image normalization."
    )

    # --------------------------------------------------------------
    # Test 4: Contiguous tensor
    # --------------------------------------------------------------

    assert tensor.flags["C_CONTIGUOUS"]

    print(
        "Test 4 passed: Contiguous tensor."
    )

    # --------------------------------------------------------------
    # Test 5: Softmax sums to one
    # --------------------------------------------------------------

    probabilities = softmax(

        np.asarray(
            [1.0, 2.0, 3.0]
        )

    )

    assert np.isclose(
        probabilities.sum(),
        1.0,
    )

    print(
        "Test 5 passed: Softmax normalization."
    )

    # --------------------------------------------------------------
    # Test 6: Softmax finite values
    # --------------------------------------------------------------

    assert np.isfinite(
        probabilities
    ).all()

    print(
        "Test 6 passed: Softmax finite values."
    )

    # --------------------------------------------------------------
    # Test 7: Queue functionality
    # --------------------------------------------------------------

    test_queue = queue.Queue(
        maxsize=2
    )

    test_queue.put(1)

    test_queue.put(2)

    assert test_queue.get() == 1

    assert test_queue.get() == 2

    print(
        "Test 7 passed: Queue producer-consumer behavior."
    )

    # --------------------------------------------------------------
    # Test 8: Bounded queue
    # --------------------------------------------------------------

    assert test_queue.maxsize == 2

    print(
        "Test 8 passed: Bounded queue configuration."
    )

    # --------------------------------------------------------------
    # Test 9: Frame request
    # --------------------------------------------------------------

    request = FrameRequest(

        frame_id=5,

        image=image,

        label=2,

        submitted_at=time.perf_counter(),

    )

    assert request.frame_id == 5

    assert request.label == 2

    print(
        "Test 9 passed: Frame request structure."
    )

    # --------------------------------------------------------------
    # Test 10: Prepared request
    # --------------------------------------------------------------

    prepared = PreparedRequest(

        request=request,

        tensor=tensor,

        preprocessing_ms=1.0,

        preprocessing_finished_at=time.perf_counter(),

    )

    assert prepared.tensor.shape == (
        1,
        3,
        32,
        32,
    )

    print(
        "Test 10 passed: Prepared request structure."
    )

    # --------------------------------------------------------------
    # Test 11: Inference result
    # --------------------------------------------------------------

    result = InferenceResult(

        frame_id=5,

        predicted_class=2,

        confidence=0.9,

        label=2,

        submitted_at=1.0,

        completed_at=1.1,

        raw_queue_wait_ms=2.0,

        preprocessing_ms=3.0,

        inference_queue_wait_ms=1.0,

        inference_ms=5.0,

        total_latency_ms=100.0,

    )

    assert result.total_latency_ms == 100.0

    print(
        "Test 11 passed: Inference result structure."
    )

    # --------------------------------------------------------------
    # Test 12: Dynamic ONNX model export
    # --------------------------------------------------------------

    onnx_path = (
        test_dir / "test_model.onnx"
    )

    create_test_model(
        onnx_path,
        32,
    )

    assert onnx_path.exists()

    print(
        "Test 12 passed: ONNX model export."
    )

    # --------------------------------------------------------------
    # Test 13: ONNX graph validation
    # --------------------------------------------------------------

    import onnx

    graph = onnx.load(
        str(onnx_path)
    )

    onnx.checker.check_model(
        graph
    )

    print(
        "Test 13 passed: ONNX graph validation."
    )

    # --------------------------------------------------------------
    # Test 14: ONNX Runtime session
    # --------------------------------------------------------------

    import onnxruntime as ort

    session = ort.InferenceSession(

        str(onnx_path),

        providers=[
            "CPUExecutionProvider"
        ],

    )

    assert session is not None

    print(
        "Test 14 passed: ONNX Runtime session."
    )

    # --------------------------------------------------------------
    # Test 15: Single-image inference
    # --------------------------------------------------------------

    input_name = session.get_inputs()[0].name

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
        "Test 15 passed: Single-image inference."
    )

    # --------------------------------------------------------------
    # Test 16: Multiple individual inferences
    # --------------------------------------------------------------

    for _ in range(3):

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
        "Test 16 passed: Repeated inference."
    )

    # --------------------------------------------------------------
    # Test 17: Concurrent inference calls
    # --------------------------------------------------------------

    def infer_once():

        return session.run(

            None,

            {
                input_name: tensor,
            },

        )[0]

    with ThreadPoolExecutor(
        max_workers=4
    ) as executor:

        outputs = list(
            executor.map(
                lambda _: infer_once(),
                range(8),
            )
        )

    assert len(outputs) == 8

    assert all(
        item.shape == (1, 10)
        for item in outputs
    )

    print(
        "Test 17 passed: Concurrent ONNX Runtime calls."
    )

    # --------------------------------------------------------------
    # Test 18: Pipeline startup and shutdown
    # --------------------------------------------------------------

    pipeline = AsyncInferencePipeline(

        session=session,

        image_size=32,

        preprocessing_workers=2,

        inference_workers=2,

        queue_size=4,

    )

    pipeline.start()

    pipeline.finish_input()

    pipeline.shutdown()

    assert not pipeline.running

    print(
        "Test 18 passed: Pipeline startup and shutdown."
    )

    # --------------------------------------------------------------
    # Test 19: Pipeline end-to-end processing
    # --------------------------------------------------------------

    pipeline = AsyncInferencePipeline(

        session=session,

        image_size=32,

        preprocessing_workers=2,

        inference_workers=2,

        queue_size=4,

    )

    test_images = [

        Image.fromarray(

            np.random.default_rng(index).integers(

                0,

                256,

                size=(32, 32, 3),

                dtype=np.uint8,

            )

        )

        for index in range(8)

    ]

    pipeline.start()

    try:

        for index, test_image in enumerate(test_images):

            pipeline.submit(

                frame_id=index,

                image=test_image,

                label=index % 10,

            )

        pipeline.finish_input()

        received = [

            pipeline.result_queue.get(
                timeout=30
            )

            for _ in range(len(test_images))

        ]

        pipeline.shutdown()

    except Exception:

        if pipeline.running:

            pipeline.finish_input()

            pipeline.shutdown()

        raise

    assert len(received) == 8

    print(
        "Test 19 passed: End-to-end asynchronous processing."
    )

    # --------------------------------------------------------------
    # Test 20: Every frame completed exactly once
    # --------------------------------------------------------------

    frame_ids = [

        item.frame_id

        for item in received

    ]

    assert sorted(frame_ids) == list(
        range(8)
    )

    print(
        "Test 20 passed: No missing or duplicate frame results."
    )

    # --------------------------------------------------------------
    # Test 21: Prediction validity
    # --------------------------------------------------------------

    assert all(

        0 <= item.predicted_class < 10

        for item in received

    )

    print(
        "Test 21 passed: Prediction class range."
    )

    # --------------------------------------------------------------
    # Test 22: Confidence validity
    # --------------------------------------------------------------

    assert all(

        0.0 <= item.confidence <= 1.0

        for item in received

    )

    print(
        "Test 22 passed: Prediction confidence range."
    )

    # --------------------------------------------------------------
    # Test 23: Latency validity
    # --------------------------------------------------------------

    assert all(

        item.total_latency_ms >= 0

        for item in received

    )

    print(
        "Test 23 passed: End-to-end latency."
    )

    # --------------------------------------------------------------
    # Test 24: Preprocessing timing
    # --------------------------------------------------------------

    assert all(

        item.preprocessing_ms >= 0

        for item in received

    )

    print(
        "Test 24 passed: Preprocessing timing."
    )

    # --------------------------------------------------------------
    # Test 25: Inference timing
    # --------------------------------------------------------------

    assert all(

        item.inference_ms >= 0

        for item in received

    )

    print(
        "Test 25 passed: Inference timing."
    )

    print(
        "\nAll 25 tests passed."
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Lesson 81 — Asynchronous Inference"
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

        "--preprocessing-workers",

        type=int,

        default=2,

    )

    parser.add_argument(

        "--inference-workers",

        type=int,

        default=2,

    )

    parser.add_argument(

        "--queue-size",

        type=int,

        default=16,

    )

    parser.add_argument(

        "--intra-threads",

        type=int,

        default=1,

    )

    parser.add_argument(

        "--output-dir",

        type=str,

        default="runs/lesson81",

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

    if args.preprocessing_workers <= 0:

        raise ValueError(
            "preprocessing-workers must be positive."
        )

    if args.inference_workers <= 0:

        raise ValueError(
            "inference-workers must be positive."
        )

    if args.queue_size <= 0:

        raise ValueError(
            "queue-size must be positive."
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

    print("\n" + "=" * 75)

    print(
        "LESSON 81 — ASYNCHRONOUS INFERENCE"
    )

    print("=" * 75)

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

    # --------------------------------------------------------------
    # CREATE ONNX SESSION
    # --------------------------------------------------------------

    session = create_session(

        model_path,

        args.provider,

        args.intra_threads,

    )

    # --------------------------------------------------------------
    # VERIFY MODEL INPUT SIZE
    # --------------------------------------------------------------

    model_shape = session.get_inputs()[0].shape

    height = model_shape[2]

    width = model_shape[3]

    if isinstance(height, int) and height != args.image_size:

        raise ValueError(

            f"Model expects height {height}, "
            f"but image-size is {args.image_size}."

        )

    if isinstance(width, int) and width != args.image_size:

        raise ValueError(

            f"Model expects width {width}, "
            f"but image-size is {args.image_size}."

        )

    # --------------------------------------------------------------
    # SYNCHRONOUS BASELINE
    # --------------------------------------------------------------

    print(
        "\nRunning synchronous baseline..."
    )

    synchronous = run_synchronous(

        session,

        images,

        labels,

        args.image_size,

    )

    synchronous_summary = summarize_results(
        synchronous
    )

    print(
        json.dumps(
            synchronous_summary,
            indent=4,
        )
    )

    # --------------------------------------------------------------
    # ASYNCHRONOUS PIPELINE
    # --------------------------------------------------------------

    print(
        "\nRunning asynchronous pipeline..."
    )

    print(
        f"Preprocessing workers: {args.preprocessing_workers}"
    )

    print(
        f"Inference workers: {args.inference_workers}"
    )

    print(
        f"Queue capacity: {args.queue_size}"
    )

    asynchronous = run_asynchronous(

        session,

        images,

        labels,

        args.image_size,

        args.preprocessing_workers,

        args.inference_workers,

        args.queue_size,

    )

    asynchronous_summary = summarize_results(
        asynchronous
    )

    print(
        json.dumps(
            asynchronous_summary,
            indent=4,
        )
    )

    # --------------------------------------------------------------
    # PIPELINE STAGE METRICS
    # --------------------------------------------------------------

    metrics = asynchronous["metrics"]

    print(
        "\nPipeline stage averages:"
    )

    for name, key in [

        ("Raw queue waiting", "raw_queue_waits_ms"),

        ("Preprocessing", "preprocessing_ms"),

        ("Inference", "inference_ms"),

    ]:

        values = metrics[key]

        average = (

            float(np.mean(values))

            if values

            else 0.0

        )

        print(
            f"{name}: {average:.3f} ms"
        )

    # --------------------------------------------------------------
    # PREDICTION CONSISTENCY
    # --------------------------------------------------------------

    sync_predictions = [

        item["predicted_class"]

        for item in synchronous["results"]

    ]

    async_predictions = [

        item.predicted_class

        for item in asynchronous["results"]

    ]

    if sync_predictions != async_predictions:

        raise RuntimeError(

            "Synchronous and asynchronous predictions differ."

        )

    print(
        "\nPrediction consistency: PASSED."
    )

    # --------------------------------------------------------------
    # PERFORMANCE COMPARISON
    # --------------------------------------------------------------

    summaries = [

        synchronous_summary,

        asynchronous_summary,

    ]

    print(
        "\n" + "=" * 75
    )

    print(
        "PERFORMANCE COMPARISON"
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

            f"P95 latency: "
            f"{item['p95_latency_ms']:.3f} ms"

        )

    throughput_gain = (

        asynchronous["throughput"]

        /

        synchronous["throughput"]

    )

    print(

        f"\nAsynchronous throughput ratio: "
        f"{throughput_gain:.2f}x"

    )

    print(
        "Note: asynchronous execution is not guaranteed to be faster."
    )

    # --------------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------------

    plot_comparison(

        summaries,

        output_dir,

    )

    plot_pipeline_stages(

        metrics,

        output_dir,

    )

    # --------------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------------

    report = {

        "lesson": 81,

        "title": "Asynchronous Inference",

        "model": str(model_path),

        "provider": args.provider,

        "dataset": {

            "num_images": len(images),

            "class_names": class_names,

            "image_size": args.image_size,

        },

        "pipeline_configuration": {

            "preprocessing_workers": (
                args.preprocessing_workers
            ),

            "inference_workers": (
                args.inference_workers
            ),

            "queue_size": args.queue_size,

            "intra_threads": args.intra_threads,

        },

        "synchronous": synchronous_summary,

        "asynchronous": asynchronous_summary,

        "asynchronous_stage_metrics": {

            key: {

                "mean_ms": (

                    float(np.mean(values))

                    if values

                    else 0.0

                ),

                "p95_ms": (

                    float(np.percentile(values, 95))

                    if values

                    else 0.0

                ),

            }

            for key, values in metrics.items()

            if isinstance(values, list)

        },

        "throughput_ratio": throughput_gain,

        "prediction_consistency": True,

    }

    report_path = (

        output_dir / "asynchronous_inference_report.json"

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
        "LESSON 81 COMPLETED."
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
        "  throughput_comparison.png"
    )

    print(
        "  latency_comparison.png"
    )

    print(
        "  pipeline_stage_times.png"
    )

    print(
        "  asynchronous_inference_report.json"
    )

    print(
        "\nNext: Lesson 82 — Memory Optimization."
    )


if __name__ == "__main__":

    main()