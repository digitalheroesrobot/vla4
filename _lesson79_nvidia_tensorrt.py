"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 6 — MODEL OPTIMIZATION AND DEPLOYMENT

LESSON 79 — NVIDIA TENSORRT
======================================================================

OBJECTIVES
----------

1. Understand TensorRT inference architecture.
2. Parse an ONNX graph.
3. Build a TensorRT serialized engine.
4. Configure GPU workspace memory.
5. Create dynamic shape optimization profiles.
6. Enable FP16 when supported.
7. Allocate GPU input/output tensors.
8. Execute inference asynchronously.
9. Compare TensorRT and ONNX Runtime outputs.
10. Benchmark inference latency and throughput.

NO pretrained models.
NO pretrained weights.

======================================================================
1. TENSORRT ARCHITECTURE
======================================================================

Training:

    PyTorch
       |
       v
    model.pth

Export:

    PyTorch
       |
       v
    model.onnx

Deployment:

    model.onnx
       |
       v
    TensorRT Builder
       |
       v
    TensorRT Engine
       |
       v
    TensorRT Runtime
       |
       v
    NVIDIA GPU

The serialized TensorRT engine contains an optimized execution
representation for the target GPU and TensorRT environment.

======================================================================
2. WHY TENSORRT?
======================================================================

TensorRT can optimize inference through:

    - Layer fusion
    - Kernel selection
    - Precision selection
    - Memory planning
    - Tensor layout optimization
    - GPU-specific execution planning

For example:

    Conv -> BatchNorm -> ReLU

may be combined into a more efficient execution path.

TensorRT chooses implementations based on the target hardware
and available kernels.

======================================================================
3. PRECISION
======================================================================

FP32:

    32-bit floating-point operations.

FP16:

    16-bit floating-point operations where supported.

FP16 can reduce memory use and improve throughput on compatible
NVIDIA GPUs.

However:

    FP16 is not guaranteed to be faster for every model.

    FP16 may introduce numerical differences.

This lesson compares FP32 and FP16 where supported.

This is not INT8 quantization. INT8 was covered in Lesson 74.

======================================================================
4. DYNAMIC SHAPES
======================================================================

An ONNX input may contain a dynamic batch dimension:

    [batch, 3, 64, 64]

TensorRT needs an optimization profile:

    Minimum: [1, 3, 64, 64]
    Optimum: [4, 3, 64, 64]
    Maximum: [16, 3, 64, 64]

The profile tells TensorRT which input shapes the engine
must support and which shape should be optimized.

======================================================================
5. ASYNCHRONOUS INFERENCE
======================================================================

TensorRT execution can be scheduled on a CUDA stream.

    Input GPU tensor
          |
          v
    TensorRT execution
          |
          v
    Output GPU tensor

execute_async_v3() schedules inference asynchronously.

CUDA synchronization is needed when the CPU must wait for
the GPU result or when measuring completed execution time.

======================================================================
6. BENCHMARKING
======================================================================

Compare:

    ONNX Runtime CPU
    TensorRT FP32
    TensorRT FP16 (if supported)

Measure:

    - Mean latency
    - Median latency
    - P95 latency
    - Throughput
    - Numerical difference

Do not compare unsynchronized GPU timings with CPU wall-clock
timings. This script synchronizes the GPU around measurements.

======================================================================
7. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste-sorting perception:

    RGB camera
        |
        v
    Image preprocessing
        |
        v
    TensorRT perception model
        |
        v
    Object detection / classification
        |
        v
    Object localization
        |
        v
    ACT manipulation policy
        |
        v
    Robot control

TensorRT is useful when inference latency is a bottleneck
in a GPU-based perception pipeline.

======================================================================
NEXT LESSON
-----------

Lesson 80 — Dynamic Batching.
======================================================================
"""

from __future__ import annotations

import argparse
import json
import random
import time

from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


# ======================================================================
# CONSTANTS
# ======================================================================

IMAGE_SIZE = 64

DEFAULT_BATCHES = [1, 4, 8]

TRT_PRECISIONS = ["FP32", "FP16"]


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ======================================================================
# CNN MODEL
# ======================================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):

        super().__init__()

        self.conv = nn.Conv2d(

            in_channels,

            out_channels,

            kernel_size=3,

            padding=1,

            bias=False,

        )

        self.bn = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=False
        )

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)

        x = self.relu(x)

        x = self.pool(x)

        return x


class VisionCNN(nn.Module):

    def __init__(
        self,
        num_classes=10,
    ):

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


# ======================================================================
# TENSORRT IMPORT
# ======================================================================

def import_tensorrt():

    try:

        import tensorrt as trt

        return trt

    except ImportError as exc:

        raise RuntimeError(

            "TensorRT Python bindings are unavailable.\n"

            "Install a TensorRT 10.x environment compatible with "
            "your NVIDIA GPU, driver, CUDA, and Python version."

        ) from exc


def require_cuda():

    if not torch.cuda.is_available():

        raise RuntimeError(

            "CUDA is unavailable.\n"

            "TensorRT GPU execution requires an NVIDIA GPU "
            "and a working CUDA environment."

        )

    return torch.device("cuda")


# ======================================================================
# CREATE A FALLBACK ONNX MODEL
# ======================================================================

def create_fallback_onnx(
    output_path,
    image_size,
    num_classes,
):

    """
    Creates a randomly initialized CNN and exports it to ONNX.

    This fallback is only for testing the TensorRT conversion
    and inference pipeline.

    It is NOT a trained classification model.

    For meaningful perception inference, use a trained ONNX
    model from Lesson 77 or Lesson 78.
    """

    import onnx

    model = VisionCNN(
        num_classes=num_classes
    )

    model.eval()

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

        do_constant_folding=True,

        dynamo=False,

    )

    graph = onnx.load(
        str(output_path)
    )

    onnx.checker.check_model(
        graph
    )

    print(
        "\nCreated fallback ONNX graph."
    )

    print(
        "WARNING: This graph has random weights and is not trained."
    )

    return model


# ======================================================================
# LOCATE INPUT ONNX MODEL
# ======================================================================

def resolve_onnx_path(
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
                f"\nUsing existing ONNX model: {path}"
            )

            return path

    fallback_path = (
        output_dir / "fallback_random_model.onnx"
    )

    create_fallback_onnx(

        fallback_path,

        image_size=image_size,

        num_classes=10,

    )

    return fallback_path


# ======================================================================
# ONNX MODEL INSPECTION
# ======================================================================

def inspect_onnx(path):

    import onnx

    model = onnx.load(
        str(path)
    )

    onnx.checker.check_model(
        model
    )

    print(
        "\nONNX model information"
    )

    print(
        f"Graph name: {model.graph.name}"
    )

    print(
        f"Node count: {len(model.graph.node)}"
    )

    print(
        f"Opset: {model.opset_import[0].version}"
    )

    inputs = []

    for value in model.graph.input:

        dims = []

        for dim in value.type.tensor_type.shape.dim:

            if dim.dim_param:

                dims.append(dim.dim_param)

            else:

                dims.append(dim.dim_value)

        inputs.append({

            "name": value.name,

            "shape": dims,

        })

        print(
            f"Input: {value.name} | shape={dims}"
        )

    for value in model.graph.output:

        print(
            f"Output: {value.name}"
        )

    return {

        "node_count": len(model.graph.node),

        "opset": model.opset_import[0].version,

        "inputs": inputs,

    }


# ======================================================================
# TENSORRT LOGGER
# ======================================================================

def create_logger(trt):

    return trt.Logger(
        trt.Logger.WARNING
    )


# ======================================================================
# OPTIMIZATION PROFILE
# ======================================================================

def configure_dynamic_profile(
    builder,
    network,
    config,
    min_batch,
    opt_batch,
    max_batch,
    image_size,
):

    trt = import_tensorrt()

    if network.num_inputs < 1:

        raise RuntimeError(
            "ONNX graph has no input tensors."
        )

    input_tensor = network.get_input(0)

    input_name = input_tensor.name

    original_shape = tuple(
        input_tensor.shape
    )

    print(
        f"\nTensorRT input: {input_name}"
    )

    print(
        f"Original input shape: {original_shape}"
    )

    if len(original_shape) != 4:

        raise RuntimeError(

            "This lesson expects a 4-D image input "
            "[batch, channels, height, width]."

        )

    min_shape = []

    opt_shape = []

    max_shape = []

    for index, dim in enumerate(original_shape):

        if dim >= 0:

            min_shape.append(dim)

            opt_shape.append(dim)

            max_shape.append(dim)

        elif index == 0:

            min_shape.append(min_batch)

            opt_shape.append(opt_batch)

            max_shape.append(max_batch)

        elif index in [2, 3]:

            min_shape.append(image_size)

            opt_shape.append(image_size)

            max_shape.append(image_size)

        else:

            raise RuntimeError(

                f"Unsupported dynamic dimension at index {index}."

            )

    dynamic = any(
        dim < 0 for dim in original_shape
    )

    if not dynamic:

        print(
            "Input has static dimensions; no optimization profile required."
        )

        return {

            "dynamic": False,

            "input_name": input_name,

            "min_shape": list(min_shape),

            "opt_shape": list(opt_shape),

            "max_shape": list(max_shape),

        }

    profile = builder.create_optimization_profile()

    success = profile.set_shape(

        input_name,

        tuple(min_shape),

        tuple(opt_shape),

        tuple(max_shape),

    )

    if not success:

        raise RuntimeError(
            "TensorRT rejected the dynamic shape profile."
        )

    config.add_optimization_profile(
        profile
    )

    print(
        f"Minimum shape: {tuple(min_shape)}"
    )

    print(
        f"Optimal shape: {tuple(opt_shape)}"
    )

    print(
        f"Maximum shape: {tuple(max_shape)}"
    )

    return {

        "dynamic": True,

        "input_name": input_name,

        "min_shape": list(min_shape),

        "opt_shape": list(opt_shape),

        "max_shape": list(max_shape),

    }


# ======================================================================
# BUILD TENSORRT ENGINE
# ======================================================================

def build_engine(
    onnx_path,
    engine_path,
    precision,
    workspace_mb,
    image_size,
    min_batch,
    opt_batch,
    max_batch,
):

    trt = import_tensorrt()

    require_cuda()

    logger = create_logger(trt)

    builder = trt.Builder(
        logger
    )

    network_flags = 0

    if hasattr(
        trt.NetworkDefinitionCreationFlag,
        "EXPLICIT_BATCH",
    ):

        network_flags = (

            1 << int(
                trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
            )

        )

    network = builder.create_network(
        network_flags
    )

    parser = trt.OnnxParser(
        network,
        logger,
    )

    print(
        "\nParsing ONNX graph..."
    )

    with open(
        onnx_path,
        "rb",
    ) as file:

        parsed = parser.parse(
            file.read()
        )

    if not parsed:

        errors = [

            str(parser.get_error(index))

            for index in range(parser.num_errors)

        ]

        raise RuntimeError(

            "TensorRT ONNX parsing failed:\n"

            + "\n".join(errors)

        )

    config = builder.create_builder_config()

    workspace_bytes = (
        workspace_mb * 1024 * 1024
    )

    if not hasattr(
        config,
        "set_memory_pool_limit",
    ):

        raise RuntimeError(

            "This script requires TensorRT 10.x "
            "builder configuration APIs."

        )

    config.set_memory_pool_limit(

        trt.MemoryPoolType.WORKSPACE,

        workspace_bytes,

    )

    precision_info = {

        "requested": precision,

        "enabled": False,

        "reason": None,

    }

    if precision == "FP16":

        if builder.platform_has_fast_fp16:

            config.set_flag(
                trt.BuilderFlag.FP16
            )

            precision_info["enabled"] = True

            print(
                "\nFP16 enabled."
            )

        else:

            precision_info["reason"] = (
                "GPU does not report fast FP16 support."
            )

            print(
                "\nWARNING: FP16 is not supported efficiently by this GPU."
            )

            print(
                "Building an FP32 engine instead."
            )

    else:

        print(
            "\nBuilding FP32 engine."
        )

    profile_info = configure_dynamic_profile(

        builder,

        network,

        config,

        min_batch=min_batch,

        opt_batch=opt_batch,

        max_batch=max_batch,

        image_size=image_size,

    )

    print(
        "\nBuilding TensorRT engine..."
    )

    print(
        "This can take some time."
    )

    start = time.perf_counter()

    serialized_engine = builder.build_serialized_network(

        network,

        config,

    )

    build_seconds = (
        time.perf_counter() - start
    )

    if serialized_engine is None:

        raise RuntimeError(
            "TensorRT engine build failed."
        )

    engine_path = Path(
        engine_path
    )

    engine_path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    with open(
        engine_path,
        "wb",
    ) as file:

        file.write(
            serialized_engine
        )

    print(
        f"\nEngine saved: {engine_path}"
    )

    print(
        f"Engine size: {engine_path.stat().st_size / 1024**2:.3f} MB"
    )

    print(
        f"Build time: {build_seconds:.2f} seconds"
    )

    return {

        "engine_path": str(engine_path),

        "engine_size_bytes": engine_path.stat().st_size,

        "build_seconds": build_seconds,

        "precision": precision_info,

        "workspace_mb": workspace_mb,

        "profile": profile_info,

    }


# ======================================================================
# LOAD TENSORRT ENGINE
# ======================================================================

def load_engine(engine_path):

    trt = import_tensorrt()

    logger = create_logger(trt)

    runtime = trt.Runtime(
        logger
    )

    with open(
        engine_path,
        "rb",
    ) as file:

        engine_bytes = file.read()

    engine = runtime.deserialize_cuda_engine(
        engine_bytes
    )

    if engine is None:

        raise RuntimeError(
            "Could not deserialize TensorRT engine."
        )

    context = engine.create_execution_context()

    if context is None:

        raise RuntimeError(
            "Could not create TensorRT execution context."
        )

    return runtime, engine, context


# ======================================================================
# TENSORRT DATA TYPE CONVERSION
# ======================================================================

def trt_dtype_to_torch(
    trt_dtype,
):

    trt = import_tensorrt()

    mapping = {

        trt.float32: torch.float32,

        trt.float16: torch.float16,

        trt.int8: torch.int8,

        trt.int32: torch.int32,

        trt.bool: torch.bool,

    }

    if trt_dtype not in mapping:

        raise TypeError(
            f"Unsupported TensorRT dtype: {trt_dtype}"
        )

    return mapping[trt_dtype]


# ======================================================================
# TENSORRT INFERENCE SESSION
# ======================================================================

class TensorRTSession:

    def __init__(
        self,
        engine_path,
        batch_size,
        image_size,
    ):

        require_cuda()

        self.runtime, self.engine, self.context = load_engine(
            engine_path
        )

        self.stream = torch.cuda.Stream()

        self.input_name = None

        self.output_names = []

        self.batch_size = batch_size

        self.image_size = image_size

        for index in range(
            self.engine.num_io_tensors
        ):

            name = self.engine.get_tensor_name(
                index
            )

            mode = self.engine.get_tensor_mode(
                name
            )

            if mode == self._input_mode():

                self.input_name = name

            else:

                self.output_names.append(
                    name
                )

        if self.input_name is None:

            raise RuntimeError(
                "TensorRT engine has no input tensor."
            )

        if len(self.output_names) == 0:

            raise RuntimeError(
                "TensorRT engine has no output tensors."
            )

        self.input_dtype = trt_dtype_to_torch(

            self.engine.get_tensor_dtype(
                self.input_name
            )

        )

        self.output_dtypes = {

            name: trt_dtype_to_torch(

                self.engine.get_tensor_dtype(
                    name
                )

            )

            for name in self.output_names

        }

        self.input_shape = (

            batch_size,

            3,

            image_size,

            image_size,

        )

        self.context.set_input_shape(

            self.input_name,

            self.input_shape,

        )

        if not self.context.all_binding_shapes_specified:

            raise RuntimeError(
                "TensorRT input dimensions were not fully specified."
            )

        self.input_tensor = torch.empty(

            self.input_shape,

            dtype=self.input_dtype,

            device="cuda",

        )

        self.context.set_tensor_address(

            self.input_name,

            self.input_tensor.data_ptr(),

        )

        self.output_tensors = {}

        for name in self.output_names:

            shape = tuple(

                self.context.get_tensor_shape(
                    name
                )

            )

            if any(
                dim < 0 for dim in shape
            ):

                raise RuntimeError(

                    f"Unresolved output shape for {name}: {shape}"

                )

            tensor = torch.empty(

                shape,

                dtype=self.output_dtypes[name],

                device="cuda",

            )

            self.output_tensors[name] = tensor

            self.context.set_tensor_address(

                name,

                tensor.data_ptr(),

            )

    def _input_mode(self):

        trt = import_tensorrt()

        return trt.TensorIOMode.INPUT

    def infer(
        self,
        images,
    ):

        if tuple(images.shape) != self.input_shape:

            raise ValueError(

                f"Expected input shape {self.input_shape}, "
                f"received {tuple(images.shape)}."

            )

        if images.device.type != "cuda":

            raise ValueError(
                "TensorRT input must be a CUDA tensor."
            )

        if images.dtype != self.input_dtype:

            images = images.to(
                dtype=self.input_dtype
            )

        with torch.cuda.stream(
            self.stream
        ):

            self.input_tensor.copy_(
                images
            )

            success = self.context.execute_async_v3(

                stream_handle=self.stream.cuda_stream

            )

            if not success:

                raise RuntimeError(
                    "TensorRT asynchronous execution failed."
                )

        return self.output_tensors

    def synchronize(self):

        self.stream.synchronize()


# ======================================================================
# CREATE TEST INPUT
# ======================================================================

def make_input(
    batch_size,
    image_size,
    seed=42,
):

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed
    )

    images = torch.randn(

        batch_size,

        3,

        image_size,

        image_size,

        generator=generator,

        dtype=torch.float32,

    )

    return images


# ======================================================================
# BENCHMARK TENSORRT
# ======================================================================

def benchmark_tensorrt(
    session,
    batch_size,
    image_size,
    warmup,
    iterations,
):

    images = make_input(

        batch_size,

        image_size,

    ).to("cuda")

    # Warm-up

    for _ in range(warmup):

        session.infer(images)

    session.synchronize()

    timings = []

    start_event = torch.cuda.Event(
        enable_timing=True
    )

    end_event = torch.cuda.Event(
        enable_timing=True
    )

    for _ in range(iterations):

        start_event.record(
            session.stream
        )

        session.infer(images)

        end_event.record(
            session.stream
        )

        end_event.synchronize()

        timings.append(
            start_event.elapsed_time(end_event)
        )

    values = np.asarray(
        timings,
        dtype=np.float64,
    )

    mean_latency = float(
        values.mean()
    )

    return {

        "batch_size": batch_size,

        "mean_latency_ms": mean_latency,

        "p50_latency_ms": float(
            np.percentile(values, 50)
        ),

        "p95_latency_ms": float(
            np.percentile(values, 95)
        ),

        "min_latency_ms": float(
            values.min()
        ),

        "max_latency_ms": float(
            values.max()
        ),

        "throughput_images_per_second": (

            batch_size * 1000.0 / mean_latency

        ),

    }


# ======================================================================
# ONNX RUNTIME REFERENCE
# ======================================================================

def benchmark_onnxruntime(
    onnx_path,
    batch_size,
    image_size,
    warmup,
    iterations,
):

    try:

        import onnxruntime as ort

    except ImportError:

        print(
            "\nONNX Runtime unavailable; skipping CPU reference."
        )

        return None

    session_options = ort.SessionOptions()

    session_options.graph_optimization_level = (

        ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    )

    session_options.intra_op_num_threads = 1

    session = ort.InferenceSession(

        str(onnx_path),

        sess_options=session_options,

        providers=[
            "CPUExecutionProvider"
        ],

    )

    images = make_input(

        batch_size,

        image_size,

    ).numpy()

    inputs = {
        "images": images,
    }

    for _ in range(warmup):

        session.run(
            None,
            inputs,
        )

    timings = []

    for _ in range(iterations):

        start = time.perf_counter()

        session.run(
            None,
            inputs,
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        timings.append(
            elapsed
        )

    values = np.asarray(
        timings,
        dtype=np.float64,
    )

    mean_latency = float(
        values.mean()
    )

    return {

        "batch_size": batch_size,

        "mean_latency_ms": mean_latency,

        "p50_latency_ms": float(
            np.percentile(values, 50)
        ),

        "p95_latency_ms": float(
            np.percentile(values, 95)
        ),

        "throughput_images_per_second": (

            batch_size * 1000.0 / mean_latency

        ),

    }


# ======================================================================
# NUMERICAL COMPARISON
# ======================================================================

def compare_tensorrt_onnx(
    onnx_path,
    engine_path,
    batch_size,
    image_size,
):

    try:

        import onnxruntime as ort

    except ImportError:

        print(
            "ONNX Runtime unavailable; skipping numerical comparison."
        )

        return None

    ort_session = ort.InferenceSession(

        str(onnx_path),

        providers=[
            "CPUExecutionProvider"
        ],

    )

    images = make_input(

        batch_size,

        image_size,

    )

    onnx_output = ort_session.run(

        None,

        {
            "images": images.numpy(),
        },

    )[0]

    trt_session = TensorRTSession(

        engine_path,

        batch_size,

        image_size,

    )

    gpu_images = images.to(
        "cuda"
    )

    output_tensors = trt_session.infer(
        gpu_images
    )

    trt_session.synchronize()

    trt_output = next(
        iter(output_tensors.values())
    ).float().cpu().numpy()

    difference = np.abs(
        onnx_output - trt_output
    )

    result = {

        "max_absolute_error": float(
            difference.max()
        ),

        "mean_absolute_error": float(
            difference.mean()
        ),

        "relative_l2_error": float(

            np.linalg.norm(
                onnx_output - trt_output
            )

            /

            max(
                np.linalg.norm(onnx_output),
                1e-12,
            )

        ),

        "prediction_agreement": float(

            np.mean(

                onnx_output.argmax(axis=1)

                ==

                trt_output.argmax(axis=1)

            )

        ),

    }

    del trt_session

    return result


# ======================================================================
# VISUALIZATION
# ======================================================================

def plot_latency(
    results,
    output_dir,
):

    labels = []

    latencies = []

    for precision, measurements in results.items():

        for item in measurements:

            labels.append(

                f"{precision}\nB={item['batch_size']}"

            )

            latencies.append(
                item["mean_latency_ms"]
            )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        labels,
        latencies,
    )

    ax.set_title(
        "TensorRT Inference Latency"
    )

    ax.set_ylabel(
        "Mean latency (ms)"
    )

    ax.set_xlabel(
        "Precision and batch size"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(

        output_dir / "tensorrt_latency.png",

        dpi=150,

    )

    plt.close(fig)


def plot_throughput(
    results,
    output_dir,
):

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for precision, measurements in results.items():

        batches = [

            item["batch_size"]

            for item in measurements

        ]

        throughput = [

            item["throughput_images_per_second"]

            for item in measurements

        ]

        ax.plot(

            batches,

            throughput,

            marker="o",

            label=precision,

        )

    ax.set_title(
        "TensorRT Inference Throughput"
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

        output_dir / "tensorrt_throughput.png",

        dpi=150,

    )

    plt.close(fig)


# ======================================================================
# TESTS
# ======================================================================

def run_tests():

    print(
        "\nRunning Lesson 79 tests..."
    )

    set_seed(42)

    test_dir = Path(
        "runs/lesson79_tests"
    )

    test_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Test 1: CNN output shape.

    model = VisionCNN(
        num_classes=10
    )

    model.eval()

    x = torch.randn(
        2,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    with torch.inference_mode():

        y = model(x)

    assert y.shape == (2, 10)

    print(
        "Test 1 passed: CNN output shape."
    )

    # Test 2: Finite model outputs.

    assert torch.isfinite(y).all()

    print(
        "Test 2 passed: Finite model outputs."
    )

    # Test 3: Input generation.

    x1 = make_input(
        2,
        IMAGE_SIZE,
        seed=42,
    )

    x2 = make_input(
        2,
        IMAGE_SIZE,
        seed=42,
    )

    assert torch.equal(x1, x2)

    print(
        "Test 3 passed: Deterministic input generation."
    )

    # Test 4: Input shape.

    assert x1.shape == (
        2,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    print(
        "Test 4 passed: Input dimensions."
    )

    # Test 5: Input dtype.

    assert x1.dtype == torch.float32

    print(
        "Test 5 passed: Input dtype."
    )

    # Test 6: Dynamic batch dimensions in export.

    onnx_path = (
        test_dir / "test_model.onnx"
    )

    create_fallback_onnx(

        onnx_path,

        IMAGE_SIZE,

        10,

    )

    assert onnx_path.exists()

    print(
        "Test 6 passed: ONNX export."
    )

    # Test 7: ONNX graph validation.

    import onnx

    graph = onnx.load(
        str(onnx_path)
    )

    onnx.checker.check_model(
        graph
    )

    print(
        "Test 7 passed: ONNX graph validation."
    )

    # Test 8: Dynamic batch metadata.

    dimensions = (

        graph.graph.input[0]

        .type.tensor_type.shape.dim

    )

    assert dimensions[0].dim_param != ""

    print(
        "Test 8 passed: Dynamic batch dimension."
    )

    # Test 9: ONNX input name.

    assert (
        graph.graph.input[0].name == "images"
    )

    print(
        "Test 9 passed: ONNX input name."
    )

    # Test 10: ONNX output name.

    assert (
        graph.graph.output[0].name == "logits"
    )

    print(
        "Test 10 passed: ONNX output name."
    )

    # Test 11: Workspace calculation.

    workspace_mb = 1024

    workspace_bytes = (
        workspace_mb * 1024 * 1024
    )

    assert workspace_bytes == 1073741824

    print(
        "Test 11 passed: Workspace byte conversion."
    )

    # Test 12: Precision configuration.

    assert TRT_PRECISIONS == [
        "FP32",
        "FP16",
    ]

    print(
        "Test 12 passed: Precision configuration."
    )

    # Test 13: Batch sizes.

    assert DEFAULT_BATCHES == [
        1,
        4,
        8,
    ]

    print(
        "Test 13 passed: Benchmark batch sizes."
    )

    # Test 14: Model parameter count.

    parameter_count = sum(

        parameter.numel()

        for parameter in model.parameters()

    )

    assert parameter_count > 0

    print(
        "Test 14 passed: Model parameter count."
    )

    # Test 15: Softmax.

    probabilities = torch.softmax(
        y,
        dim=1,
    )

    assert torch.allclose(

        probabilities.sum(dim=1),

        torch.ones(2),

        atol=1e-6,

    )

    print(
        "Test 15 passed: Probability normalization."
    )

    print(
        "\nAll 15 CPU-compatible tests passed."
    )

    print(
        "TensorRT engine tests require a compatible NVIDIA GPU."
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Lesson 79 — NVIDIA TensorRT"
        )

    )

    parser.add_argument(

        "--onnx-path",

        type=str,

        default=None,

        help="Input ONNX model. Auto-detects Lesson 78/77 output if omitted.",

    )

    parser.add_argument(

        "--output-dir",

        type=str,

        default="runs/lesson79",

    )

    parser.add_argument(

        "--image-size",

        type=int,

        default=IMAGE_SIZE,

    )

    parser.add_argument(

        "--precision",

        choices=["FP32", "FP16", "BOTH"],

        default="BOTH",

    )

    parser.add_argument(

        "--workspace-mb",

        type=int,

        default=2048,

    )

    parser.add_argument(

        "--min-batch",

        type=int,

        default=1,

    )

    parser.add_argument(

        "--opt-batch",

        type=int,

        default=4,

    )

    parser.add_argument(

        "--max-batch",

        type=int,

        default=16,

    )

    parser.add_argument(

        "--warmup",

        type=int,

        default=30,

    )

    parser.add_argument(

        "--iterations",

        type=int,

        default=100,

    )

    parser.add_argument(

        "--check-only",

        action="store_true",

    )

    args = parser.parse_args()

    if args.check_only:

        run_tests()

        return

    if args.workspace_mb <= 0:

        raise ValueError(
            "Workspace must be positive."
        )

    if not (
        1 <= args.min_batch
        <= args.opt_batch
        <= args.max_batch
    ):

        raise ValueError(
            "Batch sizes must satisfy min <= opt <= max."
        )

    if args.iterations <= 0:

        raise ValueError(
            "Iterations must be positive."
        )

    if args.warmup < 0:

        raise ValueError(
            "Warm-up cannot be negative."
        )

    if args.image_size < 32:

        raise ValueError(
            "Image size must be at least 32."
        )

    set_seed(42)

    require_cuda()

    import tensorrt as trt

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    print("\n" + "=" * 75)

    print(
        "LESSON 79 — NVIDIA TENSORRT"
    )

    print("=" * 75)

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"TensorRT: {trt.__version__}"
    )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )

    # --------------------------------------------------------------
    # INPUT MODEL
    # --------------------------------------------------------------

    onnx_path = resolve_onnx_path(

        args.onnx_path,

        output_dir,

        args.image_size,

    )

    graph_info = inspect_onnx(
        onnx_path
    )

    # --------------------------------------------------------------
    # PRECISION SELECTION
    # --------------------------------------------------------------

    if args.precision == "BOTH":

        precisions = [
            "FP32",
            "FP16",
        ]

    else:

        precisions = [
            args.precision
        ]

    build_results = {}

    benchmark_results = {}

    numerical_results = {}

    # --------------------------------------------------------------
    # BUILD ENGINES
    # --------------------------------------------------------------

    for precision in precisions:

        print("\n" + "=" * 65)

        print(
            f"BUILDING {precision} ENGINE"
        )

        print("=" * 65)

        engine_path = (

            output_dir

            / f"vision_model_{precision.lower()}.engine"

        )

        build_result = build_engine(

            onnx_path,

            engine_path,

            precision=precision,

            workspace_mb=args.workspace_mb,

            image_size=args.image_size,

            min_batch=args.min_batch,

            opt_batch=args.opt_batch,

            max_batch=args.max_batch,

        )

        build_results[precision] = build_result

        # ----------------------------------------------------------
        # BENCHMARK
        # ----------------------------------------------------------

        precision_results = []

        for batch_size in DEFAULT_BATCHES:

            if batch_size < args.min_batch:

                continue

            if batch_size > args.max_batch:

                continue

            print(
                f"\nBenchmarking {precision}, batch={batch_size}"
            )

            session = TensorRTSession(

                engine_path,

                batch_size,

                args.image_size,

            )

            result = benchmark_tensorrt(

                session,

                batch_size,

                args.image_size,

                args.warmup,

                args.iterations,

            )

            precision_results.append(
                result
            )

            print(

                f"Mean latency: "
                f"{result['mean_latency_ms']:.4f} ms"

            )

            print(

                f"P95 latency: "
                f"{result['p95_latency_ms']:.4f} ms"

            )

            print(

                f"Throughput: "
                f"{result['throughput_images_per_second']:.2f} images/s"

            )

            del session

        benchmark_results[precision] = precision_results

        # ----------------------------------------------------------
        # NUMERICAL COMPARISON
        # ----------------------------------------------------------

        comparison_batch = min(
            args.opt_batch,
            args.max_batch,
        )

        numerical = compare_tensorrt_onnx(

            onnx_path,

            engine_path,

            comparison_batch,

            args.image_size,

        )

        numerical_results[precision] = numerical

        if numerical is not None:

            print(
                f"\n{precision} numerical comparison:"
            )

            print(
                f"Max absolute error: "
                f"{numerical['max_absolute_error']:.8e}"
            )

            print(
                f"Mean absolute error: "
                f"{numerical['mean_absolute_error']:.8e}"
            )

            print(
                f"Prediction agreement: "
                f"{numerical['prediction_agreement']:.4f}"
            )

    # --------------------------------------------------------------
    # ONNX RUNTIME CPU REFERENCE
    # --------------------------------------------------------------

    onnx_reference = {}

    for batch_size in DEFAULT_BATCHES:

        if batch_size > args.max_batch:

            continue

        result = benchmark_onnxruntime(

            onnx_path,

            batch_size,

            args.image_size,

            args.warmup,

            args.iterations,

        )

        if result is not None:

            onnx_reference[str(batch_size)] = result

    # --------------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------------

    plot_latency(
        benchmark_results,
        output_dir,
    )

    plot_throughput(
        benchmark_results,
        output_dir,
    )

    # --------------------------------------------------------------
    # REPORT
    # --------------------------------------------------------------

    report = {

        "lesson": 79,

        "title": "NVIDIA TensorRT",

        "hardware": {

            "gpu": torch.cuda.get_device_name(0),

            "cuda": torch.version.cuda,

            "pytorch": torch.__version__,

            "tensorrt": trt.__version__,

        },

        "onnx_model": str(onnx_path),

        "onnx_graph": graph_info,

        "build_results": build_results,

        "tensorrt_benchmarks": benchmark_results,

        "onnxruntime_cpu_reference": onnx_reference,

        "numerical_comparison": numerical_results,

    }

    report_path = (

        output_dir

        / "tensorrt_report.json"

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
        "LESSON 79 SUMMARY"
    )

    print("=" * 75)

    print(
        f"Input ONNX: {onnx_path}"
    )

    for precision, measurements in benchmark_results.items():

        print(
            f"\n{precision}:"
        )

        for item in measurements:

            print(

                f"Batch {item['batch_size']}: "

                f"{item['mean_latency_ms']:.4f} ms "

                f"| {item['throughput_images_per_second']:.2f} img/s"

            )

    print(
        "\nReport:"
    )

    print(
        report_path.resolve()
    )

    print(
        "\nLESSON 79 COMPLETED."
    )

    print(
        "Next: Lesson 80 — Dynamic Batching."
    )


if __name__ == "__main__":

    main()