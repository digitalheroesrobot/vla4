"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 6 — MODEL OPTIMIZATION AND DEPLOYMENT

LESSON 75 — MODEL PRUNING
======================================================================

OBJECTIVE
---------

Understand and implement neural network pruning.

We investigate:

    1. Unstructured pruning.
    2. Structured pruning.
    3. Magnitude-based pruning.
    4. Global pruning.
    5. Layer-wise pruning.
    6. Sparsity measurement.
    7. Model parameter count.
    8. Model storage.
    9. Prediction differences.
    10. Fine-tuning after pruning.
    11. Pruning benchmarks.

The CNN uses randomly initialized weights.

No pretrained models are used.

======================================================================
1. WHAT IS MODEL PRUNING?
======================================================================

A neural network contains parameters:

    W = [w1, w2, w3, ..., wn]

Some parameters may have relatively small magnitudes.

Magnitude pruning removes parameters with small absolute values.

Example:

    Original:

        [0.9, 0.001, -0.8, 0.002, 0.7]

    After pruning:

        [0.9, 0.0, -0.8, 0.0, 0.7]

The network now has fewer nonzero weights.

======================================================================
2. UNSTRUCTURED PRUNING
======================================================================

Unstructured pruning removes individual weights.

Example:

    Original matrix:

        [ 0.9   0.01  -0.8 ]
        [ 0.02  0.7    0.03]
        [-0.6   0.01   0.5 ]

    Pruned matrix:

        [ 0.9   0.0   -0.8 ]
        [ 0.0   0.7    0.0 ]
        [-0.6   0.0    0.5 ]

Advantages:

    - Fine-grained pruning.
    - Can achieve high sparsity.
    - Simple magnitude-based implementation.

Disadvantages:

    - Irregular memory access.
    - Standard dense GPU operations may not become faster.
    - Requires sparse kernels to exploit sparsity efficiently.

======================================================================
3. STRUCTURED PRUNING
======================================================================

Structured pruning removes entire structures.

Examples:

    - Neurons.
    - Channels.
    - Filters.
    - Attention heads.
    - Blocks.

For a CNN, an entire output channel can be removed.

Original:

    Conv2D:
        64 output channels

Pruned:

    Conv2D:
        48 output channels

The model becomes physically smaller when the architecture
is rebuilt using fewer channels.

Advantages:

    - Reduced tensor dimensions.
    - Reduced computation.
    - Better compatibility with standard dense kernels.

Disadvantages:

    - May remove useful features.
    - Requires careful architecture updates.
    - Can cause a larger accuracy drop.

======================================================================
4. MAGNITUDE PRUNING
======================================================================

Magnitude pruning uses:

    importance(w) = |w|

Small weights are considered less important.

For target sparsity s:

    Number of weights to prune:

        k = round(s * N)

Where:

    N = total number of eligible weights.

The smallest k weights are set to zero.

======================================================================
5. GLOBAL PRUNING
======================================================================

Global pruning considers parameters from multiple layers together.

All eligible weights are ranked globally.

This allows different layers to receive different sparsity levels.

======================================================================
6. LAYER-WISE PRUNING
======================================================================

Layer-wise pruning selects a separate threshold for each layer.

Each layer receives the same target sparsity, but the actual
importance distribution differs.

Global and layer-wise pruning can produce very different networks.

======================================================================
7. SPARSITY
======================================================================

Sparsity:

    sparsity = zero_weights / total_weights

Example:

    Total weights = 1,000,000

    Zero weights = 500,000

    Sparsity = 0.5 = 50%

Density:

    density = nonzero_weights / total_weights

    density = 1 - sparsity

======================================================================
8. PYTORCH PRUNING
======================================================================

PyTorch provides:

    torch.nn.utils.prune

Example:

    prune.l1_unstructured(
        module,
        name="weight",
        amount=0.3,
    )

This creates a pruning reparameterization:

    weight_orig
    weight_mask

The effective weight becomes:

    weight = weight_orig * weight_mask

To make pruning permanent:

    prune.remove(
        module,
        "weight",
    )

After removal, zero weights remain in the parameter tensor.

======================================================================
9. PRUNING AND INFERENCE SPEED
======================================================================

IMPORTANT:

A dense matrix containing 50% zero values is not automatically
twice as fast.

A standard dense convolution may still process every element.

To gain actual speed:

    - Use structured pruning.
    - Rebuild smaller dense layers.
    - Use sparse kernels where supported.
    - Benchmark the actual deployment backend.

======================================================================
10. FINE-TUNING AFTER PRUNING
======================================================================

Pruning can reduce prediction quality.

Fine-tuning allows remaining weights to adapt.

Typical workflow:

    Train model
        |
        v
    Prune parameters
        |
        v
    Fine-tune
        |
        v
    Evaluate
        |
        v
    Repeat if necessary

This lesson demonstrates a small supervised training experiment
to illustrate fine-tuning after pruning.

======================================================================
11. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    Five camera streams
          |
          v
    Vision backbone
          |
          v
    Object detection
          |
          v
    Object localization
          |
          v
    Grasp planning
          |
          v
    ACT policy

A smaller structured-pruned backbone may:

    - Reduce computation.
    - Reduce activation memory.
    - Improve inference latency.
    - Leave more GPU resources for ACT.

But pruning must be evaluated on real objects, especially
small waste items and partially occluded objects.

======================================================================
WHAT YOU WILL LEARN
======================================================================

After this lesson:

    - Understand model sparsity.
    - Implement magnitude pruning.
    - Compare global and layer-wise pruning.
    - Implement structured channel pruning.
    - Measure parameter reduction.
    - Evaluate prediction changes.
    - Fine-tune a pruned model.
    - Understand practical deployment limitations.

NEXT LESSON
-----------

Lesson 76 — Knowledge Distillation.

======================================================================
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time

from pathlib import Path
from typing import Any

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.prune as prune

import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

DEFAULT_IMAGE_SIZE = 128

DEFAULT_BATCH_SIZES = [1, 4, 8]

DEFAULT_SPARSITIES = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
]

DEFAULT_OUTPUT_DIR = Path(
    "runs/lesson75"
)


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
                "CUDA requested but unavailable."
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
# MODEL
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


class PruningCNN(nn.Module):

    def __init__(
        self,
        num_classes: int = 10,
        channels: tuple[int, int, int, int] = (
            32,
            64,
            128,
            256,
        ),
    ):

        super().__init__()

        c1, c2, c3, c4 = channels

        self.channels = channels

        self.features = nn.Sequential(

            ConvBlock(
                3,
                c1,
                2,
            ),

            ConvBlock(
                c1,
                c2,
                2,
            ),

            ConvBlock(
                c2,
                c3,
                2,
            ),

            ConvBlock(
                c3,
                c4,
                2,
            ),

        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Linear(
            c4,
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
# PARAMETER UTILITIES
# ======================================================================

def count_parameters(
    model: nn.Module,
) -> dict[str, int]:

    total = 0

    trainable = 0

    for parameter in model.parameters():

        total += parameter.numel()

        if parameter.requires_grad:

            trainable += parameter.numel()

    return {

        "total": total,

        "trainable": trainable,

    }


def count_nonzero_parameters(
    model: nn.Module,
) -> int:

    count = 0

    for parameter in model.parameters():

        count += torch.count_nonzero(
            parameter.detach()
        ).item()

    return int(count)


def calculate_sparsity(
    model: nn.Module,
) -> dict[str, float | int]:

    total = count_parameters(
        model
    )["total"]

    nonzero = count_nonzero_parameters(
        model
    )

    zeros = total - nonzero

    sparsity = (
        zeros / total
        if total > 0
        else 0.0
    )

    density = (
        nonzero / total
        if total > 0
        else 0.0
    )

    return {

        "total_parameters": total,

        "nonzero_parameters": nonzero,

        "zero_parameters": zeros,

        "sparsity": float(sparsity),

        "density": float(density),

    }


def parameter_storage_bytes(
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
# MODULE DISCOVERY
# ======================================================================

def get_prunable_modules(
    model: nn.Module,
) -> list[tuple[str, nn.Module]]:

    modules = []

    for name, module in model.named_modules():

        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.Linear,
            ),
        ):

            modules.append(
                (name, module)
            )

    return modules


# ======================================================================
# LAYER-WISE UNSTRUCTURED PRUNING
# ======================================================================

def apply_layerwise_pruning(
    model: nn.Module,
    amount: float,
) -> nn.Module:

    """
    Prune each eligible layer independently.

    amount:
        Fraction of weights to prune in each layer.
    """

    if not 0.0 <= amount < 1.0:

        raise ValueError(
            "Pruning amount must be in [0, 1)."
        )

    for _, module in get_prunable_modules(model):

        prune.l1_unstructured(
            module,
            name="weight",
            amount=amount,
        )

    return model


# ======================================================================
# GLOBAL UNSTRUCTURED PRUNING
# ======================================================================

def apply_global_pruning(
    model: nn.Module,
    amount: float,
) -> nn.Module:

    """
    Rank all eligible weights globally.

    The smallest absolute values across all selected layers
    are pruned first.
    """

    if not 0.0 <= amount < 1.0:

        raise ValueError(
            "Pruning amount must be in [0, 1)."
        )

    parameters_to_prune = [

        (module, "weight")

        for _, module in get_prunable_modules(model)

    ]

    prune.global_unstructured(

        parameters_to_prune,

        pruning_method=prune.L1Unstructured,

        amount=amount,

    )

    return model


# ======================================================================
# MAKE PRUNING PERMANENT
# ======================================================================

def make_pruning_permanent(
    model: nn.Module,
) -> nn.Module:

    """
    Remove pruning reparameterization.

    The resulting parameter tensor contains actual zeros.

    This does not create a sparse storage format.
    """

    for _, module in get_prunable_modules(model):

        if hasattr(
            module,
            "weight_orig",
        ):

            prune.remove(
                module,
                "weight",
            )

    return model


# ======================================================================
# STRUCTURED CHANNEL PRUNING
# ======================================================================

def structured_prune_first_layer(
    model: PruningCNN,
    amount: float,
) -> PruningCNN:

    """
    Remove complete output channels from the first convolution.

    Rebuild the downstream architecture using fewer channels.

    This creates a genuinely smaller dense model.

    The function assumes the four-stage architecture defined above.
    """

    if not 0.0 <= amount < 1.0:

        raise ValueError(
            "Structured pruning amount must be in [0, 1)."
        )

    old_channels = list(
        model.channels
    )

    old_conv = model.features[0].conv

    old_bn = model.features[0].bn

    original_count = old_conv.out_channels

    new_count = max(
        1,
        int(round(original_count * (1.0 - amount))),
    )

    importance = torch.sum(
        torch.abs(
            old_conv.weight.detach()
        ),
        dim=(1, 2, 3),
    )

    selected = torch.argsort(
        importance,
        descending=True,
    )[:new_count]

    selected = torch.sort(
        selected
    ).values

    selected_cpu = selected.cpu()

    new_channels = [
        new_count,
        old_channels[1],
        old_channels[2],
        old_channels[3],
    ]

    smaller = PruningCNN(
        num_classes=model.classifier.out_features,
        channels=tuple(new_channels),
    )

    smaller.load_state_dict(
        model.state_dict(),
        strict=False,
    )

    with torch.no_grad():

        # First convolution output channels.

        smaller.features[0].conv.weight.copy_(
            old_conv.weight.detach()[
                selected
            ]
        )

        # First BatchNorm channel parameters.

        smaller.features[0].bn.weight.copy_(
            old_bn.weight.detach()[
                selected
            ]
        )

        smaller.features[0].bn.bias.copy_(
            old_bn.bias.detach()[
                selected
            ]
        )

        smaller.features[0].bn.running_mean.copy_(
            old_bn.running_mean[
                selected
            ]
        )

        smaller.features[0].bn.running_var.copy_(
            old_bn.running_var[
                selected
            ]
        )

        smaller.features[0].bn.num_batches_tracked.copy_(
            old_bn.num_batches_tracked
        )

        # Next convolution consumes the selected channels.

        second_conv = model.features[1].conv

        smaller.features[1].conv.weight.copy_(
            second_conv.weight.detach()[
                :,
                selected,
                :,
                :
            ]
        )

    return smaller


# ======================================================================
# OUTPUT COMPARISON
# ======================================================================

def compare_outputs(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:

    difference = (
        reference.double()
        - candidate.double()
    )

    mae = torch.mean(
        torch.abs(difference)
    )

    rmse = torch.sqrt(
        torch.mean(difference ** 2)
    )

    max_error = torch.max(
        torch.abs(difference)
    )

    relative_error = (
        torch.linalg.vector_norm(difference)
        /
        torch.linalg.vector_norm(
            reference.double()
        ).clamp_min(1e-12)
    )

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
    ).float().mean()

    return {

        "mae": float(mae.item()),

        "rmse": float(rmse.item()),

        "max_error": float(max_error.item()),

        "relative_l2_error": float(
            relative_error.item()
        ),

        "top1_agreement": float(
            agreement.item()
        ),

    }


# ======================================================================
# FINE-TUNING
# ======================================================================

def make_training_data(
    samples: int,
    image_size: int,
    num_classes: int,
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor]:

    """
    Numerical training data for controlled fine-tuning tests.

    This is not intended to represent a real image dataset.
    """

    generator = torch.Generator()

    generator.manual_seed(seed)

    images = torch.randn(
        samples,
        3,
        image_size,
        image_size,
        generator=generator,
    )

    labels = torch.randint(
        0,
        num_classes,
        (samples,),
        generator=generator,
    )

    return images, labels


def fine_tune(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    steps: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
) -> list[float]:

    """
    Demonstrate supervised fine-tuning after pruning.

    For real model deployment, replace the numerical dataset with
    the actual labeled training dataset.
    """

    model.train()

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    losses = []

    sample_count = images.shape[0]

    for step in range(steps):

        indices = torch.randint(
            0,
            sample_count,
            (batch_size,),
        )

        x = images[indices].to(device)

        y = labels[indices].to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits,
            y,
        )

        loss.backward()

        optimizer.step()

        losses.append(
            float(loss.item())
        )

    model.eval()

    return losses


# ======================================================================
# BENCHMARK
# ======================================================================

def benchmark_model(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup: int = 10,
    iterations: int = 50,
) -> dict[str, float]:

    model.eval()

    model.to(device)

    x = x.to(device)

    with torch.inference_mode():

        for _ in range(warmup):

            model(x)

    synchronize(device)

    timings = []

    total_start = time.perf_counter()

    for _ in range(iterations):

        synchronize(device)

        start = time.perf_counter()

        with torch.inference_mode():

            model(x)

        synchronize(device)

        timings.append(
            (time.perf_counter() - start) * 1000.0
        )

    total_elapsed = (
        time.perf_counter() - total_start
    )

    values = np.asarray(
        timings,
        dtype=np.float64,
    )

    return {

        "mean_latency_ms": float(
            np.mean(values)
        ),

        "p50_latency_ms": float(
            np.percentile(values, 50)
        ),

        "p95_latency_ms": float(
            np.percentile(values, 95)
        ),

        "throughput_fps": float(
            x.shape[0] * iterations / total_elapsed
        ),

    }


# ======================================================================
# MODEL EXPERIMENT
# ======================================================================

def evaluate_pruned_model(
    original_model: nn.Module,
    candidate: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    name: str,
) -> dict[str, Any]:

    original_model.eval()

    candidate.eval()

    original_model.to(device)

    candidate.to(device)

    x = x.to(device)

    with torch.inference_mode():

        reference = original_model(x)

        prediction = candidate(x)

    comparison = compare_outputs(
        reference,
        prediction,
    )

    sparsity = calculate_sparsity(
        candidate
    )

    parameters = count_parameters(
        candidate
    )

    storage = parameter_storage_bytes(
        candidate
    )

    return {

        "name": name,

        "comparison": comparison,

        "sparsity": sparsity,

        "parameters": parameters,

        "storage_bytes": storage,

    }


# ======================================================================
# VISUALIZATION
# ======================================================================

def plot_weight_distribution(
    model: nn.Module,
    output_dir: Path,
) -> None:

    weights = []

    for _, module in get_prunable_modules(model):

        weights.append(
            module.weight.detach().cpu().numpy().reshape(-1)
        )

    if not weights:

        return

    weights = np.concatenate(
        weights
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.hist(
        weights,
        bins=100,
    )

    ax.set_title(
        "Network Weight Distribution"
    )

    ax.set_xlabel("Weight value")

    ax.set_ylabel("Frequency")

    ax.grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        output_dir / "weight_distribution.png",
        dpi=150,
    )

    plt.close(fig)


def plot_sparsity(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:

    names = [

        item["name"]

        for item in results

    ]

    sparsities = [

        item["sparsity"]["sparsity"] * 100.0

        for item in results

    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        names,
        sparsities,
    )

    ax.set_ylabel("Sparsity (%)")

    ax.set_title(
        "Model Pruning Sparsity Comparison"
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
        output_dir / "pruning_sparsity.png",
        dpi=150,
    )

    plt.close(fig)


def plot_finetuning(
    losses: list[float],
    output_dir: Path,
) -> None:

    if not losses:

        return

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        losses
    )

    ax.set_xlabel("Training step")

    ax.set_ylabel("Cross-entropy loss")

    ax.set_title(
        "Fine-Tuning After Pruning"
    )

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "pruning_finetuning.png",
        dpi=150,
    )

    plt.close(fig)


# ======================================================================
# TESTS
# ======================================================================

def run_tests() -> None:

    print("\nRunning Lesson 75 tests...\n")

    torch.manual_seed(42)

    # Test 1: Model forward.

    model = PruningCNN().eval()

    x = torch.randn(
        2,
        3,
        64,
        64,
    )

    with torch.inference_mode():

        output = model(x)

    assert output.shape == (2, 10)

    print("Test 1 passed: CNN forward.")

    # Test 2: Parameter count.

    counts = count_parameters(
        model
    )

    assert counts["total"] > 0

    print("Test 2 passed: Parameter counting.")

    # Test 3: Initial sparsity.

    sparsity = calculate_sparsity(
        model
    )

    assert sparsity["sparsity"] == 0.0

    print("Test 3 passed: Initial sparsity.")

    # Test 4: Layer-wise pruning.

    layerwise = copy.deepcopy(
        model
    )

    apply_layerwise_pruning(
        layerwise,
        0.3,
    )

    assert any(

        torch.count_nonzero(
            module.weight
        ).item() < module.weight.numel()

        for _, module in get_prunable_modules(layerwise)

    )

    print("Test 4 passed: Layer-wise pruning.")

    # Test 5: Global pruning.

    global_model = copy.deepcopy(
        model
    )

    apply_global_pruning(
        global_model,
        0.3,
    )

    assert calculate_sparsity(
        global_model
    )["sparsity"] > 0.0

    print("Test 5 passed: Global pruning.")

    # Test 6: Permanent pruning.

    make_pruning_permanent(
        layerwise
    )

    for _, module in get_prunable_modules(layerwise):

        assert not hasattr(
            module,
            "weight_orig",
        )

        assert not hasattr(
            module,
            "weight_mask",
        )

    print("Test 6 passed: Permanent pruning.")

    # Test 7: Permanent zeros.

    permanent_sparsity = calculate_sparsity(
        layerwise
    )

    assert permanent_sparsity["sparsity"] > 0.0

    print("Test 7 passed: Pruned weights remain zero.")

    # Test 8: Model still works after pruning.

    layerwise.eval()

    with torch.inference_mode():

        pruned_output = layerwise(x)

    assert pruned_output.shape == (2, 10)

    print("Test 8 passed: Pruned model inference.")

    # Test 9: Structured pruning.

    structured = structured_prune_first_layer(
        model,
        amount=0.25,
    )

    assert (
        structured.features[0].conv.out_channels
        <
        model.features[0].conv.out_channels
    )

    print("Test 9 passed: Structured channel reduction.")

    # Test 10: Structured model forward.

    with torch.inference_mode():

        structured_output = structured(x)

    assert structured_output.shape == (2, 10)

    print("Test 10 passed: Structured model inference.")

    # Test 11: Storage calculation.

    original_storage = parameter_storage_bytes(
        model
    )

    structured_storage = parameter_storage_bytes(
        structured
    )

    assert original_storage > 0

    assert structured_storage < original_storage

    print("Test 11 passed: Structured storage reduction.")

    # Test 12: Output comparison.

    comparison = compare_outputs(
        output,
        output.clone(),
    )

    assert comparison["mae"] == 0.0

    assert comparison["top1_agreement"] == 1.0

    print("Test 12 passed: Output comparison.")

    # Test 13: Invalid pruning amount.

    try:

        apply_global_pruning(
            copy.deepcopy(model),
            1.0,
        )

        raise AssertionError(
            "Invalid pruning amount accepted."
        )

    except ValueError:

        pass

    print("Test 13 passed: Pruning validation.")

    # Test 14: Training data.

    images, labels = make_training_data(
        samples=16,
        image_size=32,
        num_classes=10,
    )

    assert images.shape == (
        16,
        3,
        32,
        32,
    )

    assert labels.shape == (16,)

    print("Test 14 passed: Training data creation.")

    # Test 15: Fine-tuning.

    small_model = PruningCNN().cpu()

    losses = fine_tune(
        small_model,
        images,
        labels,
        device=torch.device("cpu"),
        steps=2,
        batch_size=4,
    )

    assert len(losses) == 2

    assert all(
        np.isfinite(loss)
        for loss in losses
    )

    print("Test 15 passed: Fine-tuning.")

    print("\nAll 15 tests passed.")


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Lesson 75 — Model Pruning"
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
        "--sparsities",
        type=float,
        nargs="+",
        default=DEFAULT_SPARSITIES,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
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

    if any(
        not 0.0 <= amount < 1.0
        for amount in args.sparsities
    ):

        raise ValueError(
            "Sparsities must be in [0, 1)."
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

    print("LESSON 75 — MODEL PRUNING")

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
    # INPUT
    # --------------------------------------------------------------

    base_image = None

    if args.image:

        base_image = load_real_image(
            args.image,
            args.image_size,
        )

        print(
            f"Real image loaded: {args.image}"
        )

    # --------------------------------------------------------------
    # ORIGINAL MODEL
    # --------------------------------------------------------------

    original = PruningCNN().to(device).eval()

    plot_weight_distribution(
        original,
        output_dir,
    )

    print(
        f"\nOriginal parameter count: "
        f"{count_parameters(original)['total']:,}"
    )

    print(
        f"Original storage: "
        f"{parameter_storage_bytes(original) / (1024 ** 2):.2f} MB"
    )

    # --------------------------------------------------------------
    # INPUT BATCH
    # --------------------------------------------------------------

    x = create_input(
        batch_size=4,
        image_size=args.image_size,
        base_image=base_image,
    )

    x = preprocess(x).to(device)

    with torch.inference_mode():

        reference_output = original(x)

    # --------------------------------------------------------------
    # EXPERIMENTS
    # --------------------------------------------------------------

    experiment_results = []

    for amount in args.sparsities:

        print(
            f"\nRunning global pruning: {amount * 100:.1f}%"
        )

        candidate = copy.deepcopy(
            original
        )

        apply_global_pruning(
            candidate,
            amount,
        )

        make_pruning_permanent(
            candidate
        )

        result = evaluate_pruned_model(
            original,
            candidate,
            x,
            device,
            f"global_{amount:.2f}",
        )

        result["target_sparsity"] = amount

        experiment_results.append(
            result
        )

        print(
            f"Actual sparsity: "
            f"{result['sparsity']['sparsity'] * 100:.2f}%"
        )

        print(
            f"Top-1 agreement: "
            f"{result['comparison']['top1_agreement'] * 100:.2f}%"
        )

        print(
            f"Parameter storage: "
            f"{result['storage_bytes'] / (1024 ** 2):.2f} MB"
        )

    # --------------------------------------------------------------
    # LAYER-WISE PRUNING
    # --------------------------------------------------------------

    layerwise_model = copy.deepcopy(
        original
    )

    apply_layerwise_pruning(
        layerwise_model,
        0.5,
    )

    make_pruning_permanent(
        layerwise_model
    )

    layerwise_result = evaluate_pruned_model(
        original,
        layerwise_model,
        x,
        device,
        "layerwise_0.50",
    )

    experiment_results.append(
        layerwise_result
    )

    # --------------------------------------------------------------
    # STRUCTURED PRUNING
    # --------------------------------------------------------------

    structured_model = structured_prune_first_layer(
        original,
        amount=0.25,
    )

    structured_result = evaluate_pruned_model(
        original,
        structured_model,
        x,
        device,
        "structured_0.25",
    )

    experiment_results.append(
        structured_result
    )

    # --------------------------------------------------------------
    # FINE-TUNING
    # --------------------------------------------------------------

    print(
        "\nFine-tuning the 50% layer-wise pruned model..."
    )

    train_images, train_labels = make_training_data(
        samples=64,
        image_size=args.image_size,
        num_classes=10,
    )

    finetune_losses = fine_tune(
        layerwise_model,
        train_images,
        train_labels,
        device,
        steps=20,
        batch_size=4,
    )

    plot_finetuning(
        finetune_losses,
        output_dir,
    )

    # --------------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------------

    benchmark_results = []

    benchmark_models = {

        "fp32_original": original,

        "unstructured_50": layerwise_model,

        "structured_25": structured_model,

    }

    for name, benchmark_model_instance in benchmark_models.items():

        for batch_size in sorted(
            set(args.batch_sizes)
        ):

            batch = create_input(
                batch_size,
                args.image_size,
                base_image,
            )

            batch = preprocess(
                batch
            )

            result = benchmark_model(
                benchmark_model_instance,
                batch,
                device,
                warmup=args.warmup,
                iterations=args.iterations,
            )

            benchmark_results.append({

                "model": name,

                "batch_size": batch_size,

                **result,

            })

    print("\n" + "=" * 95)

    print("INFERENCE BENCHMARK")

    print("=" * 95)

    print(
        f"{'Model':>22}"
        f"{'Batch':>8}"
        f"{'Mean ms':>14}"
        f"{'P95 ms':>14}"
        f"{'FPS':>14}"
    )

    print("-" * 95)

    for item in benchmark_results:

        print(
            f"{item['model']:>22}"
            f"{item['batch_size']:>8}"
            f"{item['mean_latency_ms']:>14.4f}"
            f"{item['p95_latency_ms']:>14.4f}"
            f"{item['throughput_fps']:>14.2f}"
        )

    # --------------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------------

    plot_sparsity(
        experiment_results,
        output_dir,
    )

    # --------------------------------------------------------------
    # JSON REPORT
    # --------------------------------------------------------------

    report = {

        "lesson": 75,

        "title": "Model Pruning",

        "device": str(device),

        "image_size": args.image_size,

        "original_parameters": count_parameters(
            original
        ),

        "original_storage_bytes": parameter_storage_bytes(
            original
        ),

        "pruning_experiments": experiment_results,

        "benchmark_results": benchmark_results,

        "finetuning_losses": finetune_losses,

    }

    report_path = (
        output_dir / "model_pruning_report.json"
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

    print("\nLESSON 75 COMPLETED")

    print(
        "\nNext: Lesson 76 — Knowledge Distillation."
    )


if __name__ == "__main__":

    main()