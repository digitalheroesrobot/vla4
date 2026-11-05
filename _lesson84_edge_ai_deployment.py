#!/usr/bin/env python3
"""
===============================================================================
LESSON 84 — EDGE AI DEPLOYMENT
===============================================================================

COURSE: Computer Vision From Beginner to Advanced
PHASE 6: Model Optimization and Deployment

-------------------------------------------------------------------------------
1. LEARNING OBJECTIVES
-------------------------------------------------------------------------------

By the end of this lesson, you will understand how to:

1. Inspect and validate an ONNX model before deployment.
2. Discover available ONNX Runtime execution providers.
3. Configure CPU, CUDA, and other available providers.
4. Build a reliable image preprocessing pipeline.
5. Run model warm-up and health checks.
6. Build a bounded real-time inference pipeline.
7. Implement a latest-frame queue policy.
8. Measure:
       - Queue waiting time
       - Preprocessing latency
       - Inference latency
       - End-to-end latency
       - Throughput
       - Dropped frames
9. Monitor process memory and CPU usage.
10. Generate a deployment manifest and deployment report.
11. Test the deployment pipeline automatically.

-------------------------------------------------------------------------------
2. EDGE AI THEORY
-------------------------------------------------------------------------------

Cloud inference:

    Camera -> Network -> Cloud GPU -> Network -> Robot

Advantages:
    - Large computing resources
    - Centralized model management
    - Easier scaling

Disadvantages:
    - Network dependency
    - Network latency
    - Potential privacy concerns
    - Unpredictable communication delays

Edge inference:

    Camera -> Local Computer -> Model -> Robot Controller

Advantages:
    - Lower communication latency
    - Local operation
    - Better control over data
    - More predictable architecture

Potential deployment targets:

    CPU mini PC:
        ONNX Runtime CPUExecutionProvider

    NVIDIA Jetson:
        ONNX Runtime CUDAExecutionProvider or TensorRTExecutionProvider,
        depending on the installed runtime and supported configuration.

    Raspberry Pi:
        ONNX Runtime CPUExecutionProvider or another compatible runtime.

    Industrial PC:
        CPU, CUDA, or vendor-specific acceleration.

IMPORTANT:
    A model running successfully on your development computer does not
    guarantee that it will run successfully on an edge device.

    Always test on the actual target hardware.

-------------------------------------------------------------------------------
3. MODEL INPUT CONTRACT
-------------------------------------------------------------------------------

A model needs a clearly defined input contract.

This lesson assumes:

    Layout:       NCHW
    Color:        RGB
    Data type:    float32
    Range:        normalized floating-point values
    Batch:        dynamic or fixed

Input:

    [batch, 3, height, width]

Default normalization:

    normalized = (pixel / 255 - mean) / std

The actual normalization must match the model used during training.

A model trained with a different preprocessing pipeline must use its
original preprocessing configuration.

-------------------------------------------------------------------------------
4. MODEL OUTPUT CONTRACT
-------------------------------------------------------------------------------

Computer vision models do not all produce classification outputs.

Examples:

    Classification:
        [batch, classes]

    Object detection:
        [batch, candidates, attributes]

    Segmentation:
        [batch, classes, height, width]

    Feature extraction:
        [batch, channels, height, width]

This lesson inspects arbitrary output tensors.

It does not assume that every model returns [batch, classes].

-------------------------------------------------------------------------------
5. REAL-TIME STREAMING AND BACKPRESSURE
-------------------------------------------------------------------------------

Imagine a camera producing 30 frames per second.

The model can only process 10 frames per second.

An unlimited FIFO queue will accumulate old frames.

For robotics, this is dangerous:

    Frame 1: 3 seconds old
    Frame 2: 2.9 seconds old
    Frame 3: 2.8 seconds old

The robot may react to an outdated observation.

A latest-frame policy is often more appropriate:

    Queue capacity: 4

    When the queue is full:
        Remove the oldest waiting frame.
        Insert the newest frame.

This preserves freshness at the cost of dropping some frames.

This is useful for:
    - Object tracking
    - Robot visual feedback
    - Real-time detection
    - Camera-based manipulation

It is NOT appropriate for every application.

For offline dataset processing, every frame should normally be preserved.

-------------------------------------------------------------------------------
6. OPENARM V2 APPLICATION
-------------------------------------------------------------------------------

A possible edge architecture:

    Head Camera --------+
                       |
    Wrist Left Camera -+--> Edge Computer
                       |        |
    Wrist Right Camera +        v
                              ONNX Runtime
                                   |
                                   v
                          Object Detection
                                   |
                                   v
                          Object Segmentation
                                   |
                                   v
                          Object Localization
                                   |
                                   v
                            Grasp Target
                                   |
                                   v
                              ACT Policy
                                   |
                                   v
                           Robot Controller

The edge inference pipeline should not directly issue arbitrary actuator
commands.

A separate safety layer should validate:
    - Object confidence
    - Object position
    - Workspace limits
    - Collision constraints
    - Robot state
    - Emergency-stop state

If inference fails, the robot should enter a defined safe state.

-------------------------------------------------------------------------------
7. INSTALLATION
-------------------------------------------------------------------------------

Recommended dependencies:

    pip install numpy matplotlib pillow psutil onnx onnxruntime torch torchvision

For NVIDIA GPU inference, install an ONNX Runtime build compatible with
your CUDA environment.

Do not assume CUDA or TensorRT providers are installed.

-------------------------------------------------------------------------------
8. EXAMPLES
-------------------------------------------------------------------------------

Run automated tests:

    python lesson84_edge_ai_deployment.py --check-only

Deploy an existing ONNX model on CPU:

    python lesson84_edge_ai_deployment.py \
        --onnx-path runs/lesson78/model.onnx \
        --provider CPUExecutionProvider \
        --num-frames 128

Deploy using CUDA if available:

    python lesson84_edge_ai_deployment.py \
        --onnx-path runs/lesson78/model.onnx \
        --provider CUDAExecutionProvider \
        --num-frames 256 \
        --max-queue-size 4

Use an image dataset:

    python lesson84_edge_ai_deployment.py \
        --data-dir ./TACO/data \
        --image-size 64 \
        --provider CPUExecutionProvider

Run only a deployment smoke test:

    python lesson84_edge_ai_deployment.py \
        --onnx-path runs/lesson78/model.onnx \
        --run-mode smoke

-------------------------------------------------------------------------------
9. EXPECTED OUTPUT
-------------------------------------------------------------------------------

runs/lesson84/
    deployment_manifest.json
    deployment_report.json
    edge_latency.png
    edge_stage_timing.png
    edge_throughput.png
    deployment.log

-------------------------------------------------------------------------------
10. IMPORTANT LIMITATIONS
-------------------------------------------------------------------------------

This script demonstrates a deployment architecture.

It does not emulate:
    - Jetson thermal throttling
    - Raspberry Pi memory bandwidth
    - Industrial camera acquisition
    - Hardware-specific GPU scheduling
    - Actual robot communication latency

Run the script on your real target device to evaluate deployment performance.

The fallback model is randomly initialized. It is only for testing the
deployment infrastructure. Its predictions are NOT meaningful.

===============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import queue
import random
import statistics
import sys
import threading
import time

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import psutil
except ImportError:
    psutil = None

try:
    import onnx
    from onnx import TensorProto
except ImportError:
    onnx = None
    TensorProto = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    from PIL import Image
except ImportError:
    Image = None


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_OUTPUT_DIR = Path("runs/lesson84")

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

LOGGER = logging.getLogger("edge_deployment")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Frame:
    frame_id: int
    image: np.ndarray
    capture_time: float
    enqueue_time: float


@dataclass
class InferenceResult:
    frame_id: int
    queue_wait_ms: float
    preprocess_ms: float
    inference_ms: float
    end_to_end_ms: float
    output_shapes: list
    success: bool
    error: Optional[str] = None


@dataclass
class ModelInformation:
    path: str
    sha256: str
    file_size_bytes: int
    input_name: str
    input_shape: list
    input_type: str
    output_information: list
    opset: Optional[int]


# =============================================================================
# LOGGING
# =============================================================================

def configure_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "deployment.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w"),
        ],
        force=True,
    )


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def json_safe(value: Any):
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            json_safe(data),
            file,
            indent=4,
            ensure_ascii=False,
        )


def percentile(values, percentile_value):
    if len(values) == 0:
        return None

    return float(np.percentile(values, percentile_value))


def summarize(values):
    values = [float(value) for value in values]

    if not values:
        return {}

    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


# =============================================================================
# MODEL EXPORT
# =============================================================================

class FallbackCNN(nn.Module):
    """
    Randomly initialized CNN.

    This model is only used to demonstrate ONNX deployment mechanics when
    no existing ONNX model is supplied.

    Its predictions are not meaningful.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def export_fallback_model(output_path: Path, image_size: int):
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required to export the fallback model."
        )

    if onnx is None:
        raise RuntimeError(
            "ONNX is required to export the fallback model."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)

    model = FallbackCNN()
    model.eval()

    dummy = torch.randn(
        1,
        3,
        image_size,
        image_size,
        dtype=torch.float32,
    )

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["predictions"],
        dynamic_axes={
            "images": {0: "batch"},
            "predictions": {0: "batch"},
        },
        opset_version=17,
    )

    onnx.checker.check_model(
        onnx.load(str(output_path))
    )

    LOGGER.warning(
        "Exported a RANDOM fallback model. "
        "Its predictions are not meaningful."
    )

    return output_path


# =============================================================================
# MODEL DISCOVERY
# =============================================================================

def resolve_model_path(
    supplied_path: Optional[str],
    output_dir: Path,
    image_size: int,
):
    if supplied_path:
        path = Path(supplied_path).expanduser()

        if not path.exists():
            raise FileNotFoundError(
                f"ONNX model does not exist: {path}"
            )

        return path.resolve(), False

    candidates = [
        Path("runs/lesson78/model.onnx"),
        Path("runs/lesson77/model.onnx"),
        Path("runs/lesson80/model.onnx"),
        Path("runs/lesson82/model.onnx"),
    ]

    for candidate in candidates:
        if candidate.exists():
            LOGGER.info(
                "Using previously exported model: %s",
                candidate,
            )
            return candidate.resolve(), False

    fallback_path = output_dir / "fallback_random_model.onnx"

    export_fallback_model(
        fallback_path,
        image_size,
    )

    return fallback_path.resolve(), True


# =============================================================================
# MODEL INSPECTION
# =============================================================================

def inspect_onnx_model(path: Path) -> ModelInformation:
    if onnx is None:
        raise RuntimeError(
            "ONNX is required for model inspection."
        )

    model = onnx.load(str(path))

    onnx.checker.check_model(model)

    graph = model.graph

    initializer_names = {
        initializer.name
        for initializer in graph.initializer
    }

    inputs = [
        item
        for item in graph.input
        if item.name not in initializer_names
    ]

    if len(inputs) != 1:
        raise ValueError(
            f"This lesson supports one model input, "
            f"but found {len(inputs)}."
        )

    model_input = inputs[0]

    input_shape = []

    for dimension in model_input.type.tensor_type.shape.dim:
        if dimension.dim_value > 0:
            input_shape.append(int(dimension.dim_value))
        elif dimension.dim_param:
            input_shape.append(str(dimension.dim_param))
        else:
            input_shape.append(None)

    input_type = model_input.type.tensor_type.elem_type

    if input_type != TensorProto.FLOAT:
        raise ValueError(
            "This lesson's preprocessing pipeline supports float32 "
            "ONNX inputs. Model input type is not float32."
        )

    if len(input_shape) != 4:
        raise ValueError(
            "Expected a 4-dimensional NCHW input, "
            f"received shape {input_shape}."
        )

    if input_shape[1] != 3:
        raise ValueError(
            "Expected NCHW input with 3 RGB channels. "
            f"Received shape {input_shape}."
        )

    outputs = []

    for output in graph.output:
        shape = []

        tensor_type = output.type.tensor_type

        for dimension in tensor_type.shape.dim:
            if dimension.dim_value > 0:
                shape.append(int(dimension.dim_value))
            elif dimension.dim_param:
                shape.append(str(dimension.dim_param))
            else:
                shape.append(None)

        outputs.append({
            "name": output.name,
            "shape": shape,
            "type": tensor_type.elem_type,
        })

    opset = None

    if model.opset_import:
        opset = max(
            item.version
            for item in model.opset_import
        )

    return ModelInformation(
        path=str(path),
        sha256=sha256_file(path),
        file_size_bytes=path.stat().st_size,
        input_name=model_input.name,
        input_shape=input_shape,
        input_type="float32",
        output_information=outputs,
        opset=opset,
    )


# =============================================================================
# RUNTIME DISCOVERY
# =============================================================================

def discover_runtime():
    if ort is None:
        raise RuntimeError(
            "ONNX Runtime is not installed. "
            "Install onnxruntime or a compatible GPU-enabled build."
        )

    available = ort.get_available_providers()

    LOGGER.info(
        "ONNX Runtime version: %s",
        ort.__version__,
    )

    LOGGER.info(
        "Available providers: %s",
        available,
    )

    return available


def create_runtime_session(
    model_path: Path,
    provider: str,
    intra_op_threads: int = 0,
    inter_op_threads: int = 1,
):
    available = discover_runtime()

    if provider not in available:
        raise RuntimeError(
            f"Requested provider '{provider}' is unavailable.\n"
            f"Available providers: {available}"
        )

    options = ort.SessionOptions()

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=[provider],
    )

    actual_providers = session.get_providers()

    LOGGER.info(
        "Session providers: %s",
        actual_providers,
    )

    if provider not in actual_providers:
        raise RuntimeError(
            f"Requested provider {provider} is not active."
        )

    return session, actual_providers


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(
    image: np.ndarray,
    image_size: int,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
):
    if Image is None:
        raise RuntimeError(
            "Pillow is required for image preprocessing."
        )

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.ndim != 3:
        raise ValueError(
            f"Expected HWC image, got shape {image.shape}."
        )

    if image.shape[2] == 4:
        image = image[:, :, :3]

    if image.shape[2] != 3:
        raise ValueError(
            "Expected a 3-channel RGB image."
        )

    image = np.asarray(image, dtype=np.uint8)

    pil_image = Image.fromarray(image, mode="RGB")

    pil_image = pil_image.resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )

    image = np.asarray(
        pil_image,
        dtype=np.float32,
    ) / 255.0

    mean_array = np.asarray(
        mean,
        dtype=np.float32,
    ).reshape(1, 1, 3)

    std_array = np.asarray(
        std,
        dtype=np.float32,
    ).reshape(1, 1, 3)

    if np.any(std_array <= 0):
        raise ValueError("Normalization std must be positive.")

    image = (image - mean_array) / std_array

    image = np.transpose(
        image,
        (2, 0, 1),
    )

    image = np.ascontiguousarray(
        image,
        dtype=np.float32,
    )

    return image


def get_model_image_size(
    model_information: ModelInformation,
    fallback_size: int,
):
    height = model_information.input_shape[2]
    width = model_information.input_shape[3]

    if isinstance(height, int) and isinstance(width, int):
        if height != width:
            raise ValueError(
                "This lesson uses square image resizing. "
                f"Model expects {height}x{width}."
            )

        return height

    return fallback_size


# =============================================================================
# DATASET LOADING
# =============================================================================

def find_image_paths(root: Path):
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    paths = []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            paths.append(path)

    return sorted(paths)


def load_image_dataset(
    data_dir: Optional[str],
    dataset_root: Path,
    num_frames: int,
):
    if Image is None:
        raise RuntimeError("Pillow is required.")

    paths = []

    if data_dir:
        root = Path(data_dir).expanduser()

        if not root.exists():
            raise FileNotFoundError(root)

        paths = find_image_paths(root)

    if not paths:
        paths = find_image_paths(dataset_root)

    if not paths:
        raise RuntimeError(
            "No real image files were found.\n"
            "Provide --data-dir or place an image dataset in "
            f"{dataset_root}."
        )

    selected = []

    for index in range(num_frames):
        selected.append(
            paths[index % len(paths)]
        )

    images = []

    for path in selected:
        with Image.open(path) as image:
            image = image.convert("RGB")
            images.append(np.asarray(image).copy())

    LOGGER.info(
        "Loaded %d real images from %d available image files.",
        len(images),
        len(paths),
    )

    return images, [str(path) for path in selected]


def load_cifar10_images(
    num_frames: int,
    dataset_root: Path,
):
    """
    Load actual CIFAR-10 photographs.

    Downloading CIFAR-10 requires an internet connection unless the dataset
    is already cached.
    """

    try:
        from torchvision.datasets import CIFAR10
    except Exception as exc:
        raise RuntimeError(
            "Could not import torchvision CIFAR-10. "
            "Use --data-dir to provide your own image dataset."
        ) from exc

    dataset = CIFAR10(
        root=str(dataset_root),
        train=False,
        download=True,
    )

    images = []

    for index in range(num_frames):
        image, _ = dataset[index % len(dataset)]
        images.append(np.asarray(image.convert("RGB")).copy())

    LOGGER.info(
        "Loaded %d real CIFAR-10 images.",
        len(images),
    )

    return images


# =============================================================================
# OUTPUT VALIDATION
# =============================================================================

def validate_outputs(outputs):
    if not isinstance(outputs, (list, tuple)):
        raise TypeError("Runtime outputs must be a list or tuple.")

    if len(outputs) == 0:
        raise ValueError("Model returned no output tensors.")

    shapes = []

    for output in outputs:
        array = np.asarray(output)

        if array.size == 0:
            raise ValueError("Model returned an empty tensor.")

        if not np.all(np.isfinite(array)):
            raise ValueError(
                "Model output contains NaN or infinity."
            )

        shapes.append(list(array.shape))

    return shapes


# =============================================================================
# MODEL HEALTH CHECK
# =============================================================================

def health_check(
    session,
    model_information: ModelInformation,
    image_size: int,
    warmup_runs: int = 5,
):
    LOGGER.info("Starting model health check.")

    test_image = np.zeros(
        (image_size, image_size, 3),
        dtype=np.uint8,
    )

    input_tensor = preprocess_image(
        test_image,
        image_size,
    )[None, ...]

    for index in range(warmup_runs):
        outputs = session.run(
            None,
            {
                model_information.input_name: input_tensor
            },
        )

        shapes = validate_outputs(outputs)

        LOGGER.info(
            "Warm-up %d/%d successful: %s",
            index + 1,
            warmup_runs,
            shapes,
        )

    output_information = []

    for output in session.get_outputs():
        output_information.append({
            "name": output.name,
            "shape": output.shape,
            "type": output.type,
        })

    return {
        "success": True,
        "warmup_runs": warmup_runs,
        "input_shape": list(input_tensor.shape),
        "outputs": output_information,
    }


# =============================================================================
# BOUNDED LATEST-FRAME QUEUE
# =============================================================================

class LatestFrameQueue:
    """
    Bounded queue that drops the oldest waiting frame when full.

    The currently processing frame is not interrupted.

    Every removed item receives task_done(), so queue accounting remains
    consistent.
    """

    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError(
                "Queue size must be at least 1."
            )

        self.queue = queue.Queue(
            maxsize=max_size
        )

        self.max_size = max_size

        self.dropped_frames = 0
        self.accepted_frames = 0

        self.lock = threading.Lock()

    def put_latest(self, frame: Frame):
        dropped = False

        with self.lock:
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()

                    self.dropped_frames += 1
                    dropped = True

                except queue.Empty:
                    pass

            self.queue.put_nowait(frame)
            self.accepted_frames += 1

        return dropped

    def get(self, timeout=None):
        return self.queue.get(timeout=timeout)

    def task_done(self):
        self.queue.task_done()

    def qsize(self):
        return self.queue.qsize()

    def join(self):
        self.queue.join()


# =============================================================================
# STREAMING INFERENCE
# =============================================================================

SENTINEL = object()


class EdgeInferencePipeline:
    def __init__(
        self,
        session,
        model_information,
        image_size,
        max_queue_size,
    ):
        self.session = session
        self.model_information = model_information
        self.image_size = image_size

        self.frame_queue = LatestFrameQueue(
            max_queue_size
        )

        self.results = []

        self.results_lock = threading.Lock()

        self.worker_errors = []

        self.stop_event = threading.Event()

        self.process = (
            psutil.Process(os.getpid())
            if psutil is not None
            else None
        )

        self.cpu_percent = []
        self.memory_mb = []

        self.worker = threading.Thread(
            target=self._worker,
            name="edge-inference-worker",
            daemon=False,
        )

    def start(self):
        self.worker.start()

    def submit(self, frame: Frame):
        return self.frame_queue.put_latest(frame)

    def _worker(self):
        while True:
            item = self.frame_queue.get()

            try:
                if item is SENTINEL:
                    return

                result = self._process_frame(item)

                with self.results_lock:
                    self.results.append(result)

            except Exception as exc:
                LOGGER.exception(
                    "Inference worker encountered an error."
                )

                self.worker_errors.append(str(exc))

            finally:
                self.frame_queue.task_done()

    def _process_frame(self, frame: Frame):
        now = time.perf_counter()

        queue_wait_ms = (
            now - frame.enqueue_time
        ) * 1000.0

        preprocess_start = time.perf_counter()

        input_image = preprocess_image(
            frame.image,
            self.image_size,
        )

        input_tensor = input_image[None, ...]

        preprocess_ms = (
            time.perf_counter() - preprocess_start
        ) * 1000.0

        inference_start = time.perf_counter()

        outputs = self.session.run(
            None,
            {
                self.model_information.input_name: input_tensor
            },
        )

        inference_ms = (
            time.perf_counter() - inference_start
        ) * 1000.0

        output_shapes = validate_outputs(outputs)

        end_to_end_ms = (
            time.perf_counter() - frame.capture_time
        ) * 1000.0

        return InferenceResult(
            frame_id=frame.frame_id,
            queue_wait_ms=queue_wait_ms,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            end_to_end_ms=end_to_end_ms,
            output_shapes=output_shapes,
            success=True,
        )

    def monitor_resources(self):
        if self.process is None:
            return

        try:
            cpu = self.process.cpu_percent(
                interval=None
            )

            memory = (
                self.process.memory_info().rss
                / (1024 ** 2)
            )

            self.cpu_percent.append(float(cpu))
            self.memory_mb.append(float(memory))

        except Exception:
            LOGGER.warning(
                "Could not collect process resource metrics."
            )

    def finish(self):
        """
        Graceful shutdown.

        Wait for pending frames, then send a sentinel to the worker.
        """

        self.frame_queue.join()

        # Wait if a producer previously filled the queue.
        while True:
            try:
                self.frame_queue.queue.put(
                    SENTINEL,
                    timeout=0.1,
                )
                break
            except queue.Full:
                continue

        self.worker.join(timeout=30.0)

        if self.worker.is_alive():
            raise RuntimeError(
                "Inference worker did not shut down."
            )

    def report(self):
        with self.results_lock:
            results = list(self.results)

        successful = [
            item
            for item in results
            if item.success
        ]

        return {
            "submitted_frames": (
                self.frame_queue.accepted_frames
            ),
            "processed_frames": len(successful),
            "dropped_frames": (
                self.frame_queue.dropped_frames
            ),
            "worker_errors": self.worker_errors,
            "queue_max_size": self.frame_queue.max_size,
            "queue_wait_ms": summarize([
                item.queue_wait_ms
                for item in successful
            ]),
            "preprocess_ms": summarize([
                item.preprocess_ms
                for item in successful
            ]),
            "inference_ms": summarize([
                item.inference_ms
                for item in successful
            ]),
            "end_to_end_ms": summarize([
                item.end_to_end_ms
                for item in successful
            ]),
            "cpu_percent_samples": self.cpu_percent,
            "memory_mb_samples": self.memory_mb,
            "frame_results": [
                asdict(item)
                for item in successful
            ],
        }


# =============================================================================
# STREAM PRODUCER
# =============================================================================

def produce_frames(
    pipeline: EdgeInferencePipeline,
    images,
    frame_interval_ms: float,
):
    start = time.perf_counter()

    for frame_id, image in enumerate(images):
        capture_time = time.perf_counter()

        frame = Frame(
            frame_id=frame_id,
            image=image,
            capture_time=capture_time,
            enqueue_time=time.perf_counter(),
        )

        pipeline.submit(frame)

        pipeline.monitor_resources()

        if frame_interval_ms > 0:
            time.sleep(
                frame_interval_ms / 1000.0
            )

    return time.perf_counter() - start


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_latency(report, output_dir: Path):
    frames = report["frame_results"]

    values = [
        item["end_to_end_ms"]
        for item in frames
    ]

    if not values:
        LOGGER.warning(
            "No successful inference results to plot."
        )
        return

    plt.figure(figsize=(10, 5))

    plt.hist(
        values,
        bins=30,
        edgecolor="black",
    )

    plt.xlabel("End-to-end latency (ms)")
    plt.ylabel("Number of frames")
    plt.title("Edge Inference Latency Distribution")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "edge_latency.png",
        dpi=150,
    )

    plt.close()


def plot_stage_timing(report, output_dir: Path):
    stages = [
        "queue_wait_ms",
        "preprocess_ms",
        "inference_ms",
    ]

    labels = [
        "Queue wait",
        "Preprocessing",
        "Inference",
    ]

    means = []

    for stage in stages:
        summary = report.get(stage, {})
        means.append(summary.get("mean", 0.0))

    plt.figure(figsize=(9, 5))

    plt.bar(labels, means)

    plt.ylabel("Mean latency (ms)")
    plt.title("Edge Inference Stage Timing")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "edge_stage_timing.png",
        dpi=150,
    )

    plt.close()


def plot_throughput(report, producer_seconds, output_dir: Path):
    processed = report["processed_frames"]
    submitted = report["submitted_frames"]

    throughput = (
        processed / producer_seconds
        if producer_seconds > 0
        else 0.0
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        ["Accepted", "Processed", "Dropped"],
        [
            submitted,
            processed,
            report["dropped_frames"],
        ],
    )

    plt.ylabel("Frame count")
    plt.title(
        f"Edge Pipeline Frame Statistics "
        f"(Throughput: {throughput:.2f} FPS)"
    )

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "edge_throughput.png",
        dpi=150,
    )

    plt.close()


# =============================================================================
# AUTOMATED TESTS
# =============================================================================

class TestSuite:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name, condition):
        if condition:
            self.passed += 1
            print(f"[PASS] {name}")
        else:
            self.failed += 1
            print(f"[FAIL] {name}")

    def run(self):
        print("\n" + "=" * 70)
        print("LESSON 84 — AUTOMATED TESTS")
        print("=" * 70)

        # Test 1: Image preprocessing
        image = np.full(
            (80, 120, 3),
            128,
            dtype=np.uint8,
        )

        processed = preprocess_image(
            image,
            image_size=64,
        )

        self.check(
            "Preprocessing produces NCHW-compatible image",
            processed.shape == (3, 64, 64),
        )

        # Test 2: Float32
        self.check(
            "Preprocessing returns float32",
            processed.dtype == np.float32,
        )

        # Test 3: Finite preprocessing
        self.check(
            "Preprocessing output is finite",
            np.all(np.isfinite(processed)),
        )

        # Test 4: RGB validation
        try:
            preprocess_image(
                np.zeros((10, 10), dtype=np.uint8),
                32,
            )
            valid = False
        except ValueError:
            valid = True

        self.check(
            "Reject invalid image dimensions",
            valid,
        )

        # Test 5: Queue initialization
        bounded = LatestFrameQueue(2)

        self.check(
            "Bounded queue has configured capacity",
            bounded.max_size == 2,
        )

        # Test 6: Queue accepts frame
        frame1 = Frame(
            1,
            np.zeros((10, 10, 3), dtype=np.uint8),
            0.0,
            0.0,
        )

        bounded.put_latest(frame1)

        self.check(
            "Queue accepts frames",
            bounded.qsize() == 1,
        )

        # Test 7: Queue maximum size
        bounded.put_latest(
            Frame(
                2,
                frame1.image,
                0.0,
                0.0,
            )
        )

        bounded.put_latest(
            Frame(
                3,
                frame1.image,
                0.0,
                0.0,
            )
        )

        self.check(
            "Queue never exceeds configured capacity",
            bounded.qsize() <= 2,
        )

        # Test 8: Latest-frame dropping
        self.check(
            "Full queue drops oldest waiting frame",
            bounded.dropped_frames == 1,
        )

        # Test 9: Latest-frame identity
        first = bounded.get()

        self.check(
            "Oldest waiting frame was removed",
            first.frame_id == 2,
        )

        bounded.task_done()

        second = bounded.get()

        self.check(
            "Newest frame is preserved",
            second.frame_id == 3,
        )

        bounded.task_done()

        bounded.join()

        # Test 10: Queue accounting
        self.check(
            "Queue task accounting completes",
            bounded.qsize() == 0,
        )

        # Test 11: Output validation with rank 2
        outputs = [
            np.ones((1, 10), dtype=np.float32)
        ]

        shapes = validate_outputs(outputs)

        self.check(
            "Rank-2 output accepted",
            shapes == [[1, 10]],
        )

        # Test 12: Rank-3 output
        outputs = [
            np.ones((1, 100, 85), dtype=np.float32)
        ]

        shapes = validate_outputs(outputs)

        self.check(
            "Rank-3 detection output accepted",
            shapes == [[1, 100, 85]],
        )

        # Test 13: Rank-4 output
        outputs = [
            np.ones((1, 20, 32, 32), dtype=np.float32)
        ]

        shapes = validate_outputs(outputs)

        self.check(
            "Rank-4 segmentation output accepted",
            shapes == [[1, 20, 32, 32]],
        )

        # Test 14: Multiple outputs
        outputs = [
            np.ones((1, 10), dtype=np.float32),
            np.ones((1, 20, 32, 32), dtype=np.float32),
        ]

        shapes = validate_outputs(outputs)

        self.check(
            "Multiple model outputs accepted",
            len(shapes) == 2,
        )

        # Test 15: Reject NaN
        try:
            validate_outputs([
                np.array([np.nan], dtype=np.float32)
            ])
            valid = False
        except ValueError:
            valid = True

        self.check(
            "Reject NaN model output",
            valid,
        )

        # Test 16: Reject infinity
        try:
            validate_outputs([
                np.array([np.inf], dtype=np.float32)
            ])
            valid = False
        except ValueError:
            valid = True

        self.check(
            "Reject infinite model output",
            valid,
        )

        # Test 17: Reject empty output
        try:
            validate_outputs([
                np.array([], dtype=np.float32)
            ])
            valid = False
        except ValueError:
            valid = True

        self.check(
            "Reject empty output",
            valid,
        )

        # Test 18: Checksum repeatability
        temp = Path("lesson84_checksum_test.txt")

        temp.write_text(
            "edge-ai-test",
            encoding="utf-8",
        )

        checksum1 = sha256_file(temp)
        checksum2 = sha256_file(temp)

        temp.unlink()

        self.check(
            "SHA-256 checksum is repeatable",
            checksum1 == checksum2,
        )

        # Test 19: Non-negative summary
        summary = summarize([1.0, 2.0, 3.0])

        self.check(
            "Latency summary returns valid mean",
            summary["mean"] == 2.0,
        )

        # Test 20: Percentile
        self.check(
            "P50 calculation works",
            percentile([1, 2, 3, 4, 5], 50) == 3.0,
        )

        # Test 21: JSON serialization
        temp_json = Path("lesson84_test.json")

        save_json(
            temp_json,
            {
                "values": np.array([1, 2, 3]),
                "mean": np.float32(2.0),
            },
        )

        with temp_json.open("r", encoding="utf-8") as file:
            loaded = json.load(file)

        temp_json.unlink()

        self.check(
            "Deployment report JSON serialization",
            loaded["values"] == [1, 2, 3],
        )

        # Test 22: Reject invalid queue capacity
        try:
            LatestFrameQueue(0)
            valid = False
        except ValueError:
            valid = True

        self.check(
            "Reject zero queue capacity",
            valid,
        )

        # Test 23: Frame identifiers
        frame_ids = [
            Frame(
                index,
                np.zeros((4, 4, 3), dtype=np.uint8),
                0.0,
                0.0,
            ).frame_id
            for index in range(5)
        ]

        self.check(
            "Frame identifiers remain unique and ordered",
            frame_ids == [0, 1, 2, 3, 4],
        )

        # Test 24: Resource monitor availability
        self.check(
            "Resource monitor handles optional psutil",
            psutil is None or hasattr(psutil, "Process"),
        )

        # Test 25: Model metadata dataclass
        info = ModelInformation(
            path="model.onnx",
            sha256="abc",
            file_size_bytes=100,
            input_name="images",
            input_shape=["batch", 3, 64, 64],
            input_type="float32",
            output_information=[],
            opset=17,
        )

        self.check(
            "Model metadata structure works",
            info.input_shape[1] == 3,
        )

        # Test 26: Reject invalid std
        try:
            preprocess_image(
                np.zeros((8, 8, 3), dtype=np.uint8),
                16,
                std=(0.0, 1.0, 1.0),
            )
            valid = False
        except ValueError:
            valid = True

        self.check(
            "Reject zero normalization standard deviation",
            valid,
        )

        # Test 27: RGB output shape
        rgb = np.zeros(
            (10, 10, 4),
            dtype=np.uint8,
        )

        rgba_processed = preprocess_image(
            rgb,
            16,
        )

        self.check(
            "RGBA image is converted to RGB",
            rgba_processed.shape == (3, 16, 16),
        )

        # Test 28: Image dataset extension filter
        temp_dir = Path("lesson84_image_test")
        temp_dir.mkdir(exist_ok=True)

        (temp_dir / "image.jpg").write_bytes(b"test")
        (temp_dir / "notes.txt").write_text("test")

        paths = find_image_paths(temp_dir)

        for path in temp_dir.iterdir():
            path.unlink()

        temp_dir.rmdir()

        self.check(
            "Dataset discovery filters image extensions",
            len(paths) == 1 and paths[0].suffix == ".jpg",
        )

        # Test 29: Model input size selection
        model_info = ModelInformation(
            path="model.onnx",
            sha256="abc",
            file_size_bytes=100,
            input_name="images",
            input_shape=["batch", 3, 128, 128],
            input_type="float32",
            output_information=[],
            opset=17,
        )

        self.check(
            "Fixed ONNX input size is discovered",
            get_model_image_size(model_info, 64) == 128,
        )

        # Test 30: Dynamic model input size fallback
        model_info.input_shape = [
            "batch",
            3,
            "height",
            "width",
        ]

        self.check(
            "Dynamic ONNX input uses configured size",
            get_model_image_size(model_info, 64) == 64,
        )

        print("\n" + "=" * 70)
        print(
            f"RESULT: {self.passed} passed, "
            f"{self.failed} failed"
        )
        print("=" * 70)

        return self.failed == 0


# =============================================================================
# DEPLOYMENT MANIFEST
# =============================================================================

def build_manifest(
    model_information,
    provider,
    actual_providers,
    args,
    image_size,
    fallback_model,
):
    return {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "model": asdict(model_information),

        "runtime": {
            "name": "ONNX Runtime",
            "version": ort.__version__,
            "requested_provider": provider,
            "active_providers": actual_providers,
        },

        "configuration": {
            "image_size": image_size,
            "max_queue_size": args.max_queue_size,
            "num_frames": args.num_frames,
            "run_mode": args.run_mode,
            "normalization_mean": DEFAULT_MEAN,
            "normalization_std": DEFAULT_STD,
        },

        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory_monitor_available": psutil is not None,
        },

        "fallback_random_model": fallback_model,

        "limitations": [
            "Target hardware must be tested directly.",
            "This benchmark excludes physical camera acquisition.",
            "This benchmark excludes robot communication latency.",
            "Random fallback model predictions are meaningless.",
        ],
    }


# =============================================================================
# MAIN DEPLOYMENT
# =============================================================================

def run_deployment(args):
    output_dir = Path(args.output_dir)

    configure_logging(output_dir)

    LOGGER.info("=" * 70)
    LOGGER.info("LESSON 84 — EDGE AI DEPLOYMENT")
    LOGGER.info("=" * 70)

    model_path, fallback_model = resolve_model_path(
        args.onnx_path,
        output_dir,
        args.image_size,
    )

    model_information = inspect_onnx_model(
        model_path
    )

    image_size = get_model_image_size(
        model_information,
        args.image_size,
    )

    LOGGER.info(
        "Model input: %s %s",
        model_information.input_name,
        model_information.input_shape,
    )

    LOGGER.info(
        "Model outputs: %s",
        model_information.output_information,
    )

    session, actual_providers = create_runtime_session(
        model_path,
        args.provider,
        args.intra_op_threads,
        args.inter_op_threads,
    )

    manifest = build_manifest(
        model_information,
        args.provider,
        actual_providers,
        args,
        image_size,
        fallback_model,
    )

    save_json(
        output_dir / "deployment_manifest.json",
        manifest,
    )

    health = health_check(
        session,
        model_information,
        image_size,
        warmup_runs=args.warmup_runs,
    )

    LOGGER.info("Model health check passed.")

    if args.run_mode == "smoke":
        report = {
            "mode": "smoke",
            "health_check": health,
            "model": asdict(model_information),
            "runtime_providers": actual_providers,
        }

        save_json(
            output_dir / "deployment_report.json",
            report,
        )

        LOGGER.info("Smoke test completed.")
        return

    if args.data_dir:
        images, image_paths = load_image_dataset(
            args.data_dir,
            Path(args.dataset_root),
            args.num_frames,
        )

    elif args.use_cifar10:
        images = load_cifar10_images(
            args.num_frames,
            Path(args.dataset_root),
        )

        image_paths = [
            f"CIFAR10:{index}"
            for index in range(len(images))
        ]

    else:
        raise RuntimeError(
            "No image dataset selected. "
            "Use --data-dir or --use-cifar10."
        )

    pipeline = EdgeInferencePipeline(
        session=session,
        model_information=model_information,
        image_size=image_size,
        max_queue_size=args.max_queue_size,
    )

    pipeline.start()

    producer_seconds = produce_frames(
        pipeline,
        images,
        args.frame_interval_ms,
    )

    pipeline.finish()

    report = pipeline.report()

    report["mode"] = args.run_mode
    report["producer_seconds"] = producer_seconds
    report["health_check"] = health
    report["runtime_providers"] = actual_providers
    report["image_paths"] = image_paths

    report["throughput_fps"] = (
        report["processed_frames"] / producer_seconds
        if producer_seconds > 0
        else 0.0
    )

    report["model"] = asdict(model_information)

    plot_latency(
        report,
        output_dir,
    )

    plot_stage_timing(
        report,
        output_dir,
    )

    plot_throughput(
        report,
        producer_seconds,
        output_dir,
    )

    save_json(
        output_dir / "deployment_report.json",
        report,
    )

    LOGGER.info("=" * 70)
    LOGGER.info("DEPLOYMENT SUMMARY")
    LOGGER.info("=" * 70)

    LOGGER.info(
        "Submitted frames: %d",
        report["submitted_frames"],
    )

    LOGGER.info(
        "Processed frames: %d",
        report["processed_frames"],
    )

    LOGGER.info(
        "Dropped frames: %d",
        report["dropped_frames"],
    )

    LOGGER.info(
        "Throughput: %.2f FPS",
        report["throughput_fps"],
    )

    LOGGER.info(
        "Mean inference latency: %.3f ms",
        report["inference_ms"].get("mean", 0.0),
    )

    LOGGER.info(
        "Mean end-to-end latency: %.3f ms",
        report["end_to_end_ms"].get("mean", 0.0),
    )

    LOGGER.info(
        "Mean process memory: %.2f MB",
        (
            statistics.mean(report["memory_mb_samples"])
            if report["memory_mb_samples"]
            else 0.0
        ),
    )

    LOGGER.info(
        "Deployment results saved to: %s",
        output_dir.resolve(),
    )

    LOGGER.info("=" * 70)


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Lesson 84 — Edge AI Deployment"
    )

    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Path to an ONNX model.",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        help="ONNX Runtime execution provider.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing real images.",
    )

    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./datasets",
        help="Dataset download/cache directory.",
    )

    parser.add_argument(
        "--use-cifar10",
        action="store_true",
        help="Download/use real CIFAR-10 test images.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--frame-interval-ms",
        type=float,
        default=0.0,
        help="Simulated camera frame interval. Zero means fastest possible input.",
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--inter-op-threads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--run-mode",
        choices=["smoke", "stream", "benchmark"],
        default="stream",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    args = parse_arguments()

    if args.image_size < 1:
        raise ValueError("Image size must be positive.")

    if args.num_frames < 1:
        raise ValueError("Number of frames must be positive.")

    if args.max_queue_size < 1:
        raise ValueError("Maximum queue size must be positive.")

    if args.frame_interval_ms < 0:
        raise ValueError("Frame interval cannot be negative.")

    if args.warmup_runs < 1:
        raise ValueError("Warm-up runs must be positive.")

    if args.check_only:
        success = TestSuite().run()

        if not success:
            sys.exit(1)

        return

    if args.run_mode == "stream" or args.run_mode == "benchmark":
        if not args.data_dir and not args.use_cifar10:
            LOGGER.warning(
                "No image dataset selected. "
                "CIFAR-10 will be used as the default real-image dataset."
            )

            args.use_cifar10 = True

    run_deployment(args)


if __name__ == "__main__":
    main()