"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 2 — DEEP LEARNING FOR COMPUTER VISION
LESSON 23 — CNN OPERATIONS AND FEATURE MAPS
======================================================================

COURSE RULE
-----------
No pretrained models.
No pretrained weights.
No pretrained feature extractors.

All demonstrations use synthetic images, explicitly defined filters,
or randomly initialized filters.

Libraries:
    NumPy       : convolution and feature-map mathematics
    OpenCV      : image generation and practical comparison
    Matplotlib  : visualization

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1 — CNN INTERNAL STRUCTURE
    Understand how images become feature maps.

STEP 2 — CONVOLUTION FILTERS
    Apply edge, blur, sharpening, and randomly initialized filters.

STEP 3 — MULTI-CHANNEL CONVOLUTION
    Understand input channels and output channels.

STEP 4 — FEATURE MAPS
    Visualize the activation produced by each filter.

STEP 5 — STRIDE
    Understand how stride changes spatial resolution.

STEP 6 — PADDING
    Compare valid and same convolution.

STEP 7 — KERNEL SIZE
    Compare 3x3, 5x5, and 7x7 kernels.

STEP 8 — ACTIVATION FUNCTIONS
    Compare linear convolution output and ReLU output.

STEP 9 — RECEPTIVE FIELD
    Calculate how much of the original image influences a feature.

STEP 10 — CHANNEL VISUALIZATION
    Understand why CNNs can have many feature channels.

STEP 11 — OPENCV COMPARISON
    Compare NumPy convolution with cv2.filter2D.

======================================================================
1. CNN FEATURE EXTRACTION
======================================================================

A convolutional network transforms an image through layers:

    Input Image
        |
        v
    Convolution
        |
        v
    Feature Maps
        |
        v
       ReLU
        |
        v
      Pooling
        |
        v
    More Convolution
        |
        v
    Higher-Level Features
        |
        v
    Classification

A feature map represents how strongly a filter responds
at different image locations.

======================================================================
2. CONVOLUTION FILTERS
======================================================================

A filter is a small matrix of trainable parameters.

Example horizontal edge filter:

    [-1 -1 -1]
    [ 0  0  0]
    [ 1  1  1]

Example vertical edge filter:

    [-1  0  1]
    [-1  0  1]
    [-1  0  1]

A CNN does not need these filters to be manually designed.

During training, backpropagation updates filter weights
to minimize the task loss.

IMPORTANT:
    The filters in this lesson are demonstrations.
    Handcrafted filters and random filters are NOT the same
    as filters learned through training.

======================================================================
3. CONVOLUTION OUTPUT SIZE
======================================================================

For an input dimension H:

    H_out = floor((H + 2P - K) / S) + 1

where:

    H = input height
    P = padding
    K = kernel size
    S = stride

Example:

    Input = 32
    Kernel = 3
    Padding = 1
    Stride = 1

    H_out = floor((32 + 2 - 3) / 1) + 1
          = 32

With stride 2:

    H_out = floor((32 + 2 - 3) / 2) + 1
          = 16

======================================================================
4. MULTI-CHANNEL CONVOLUTION
======================================================================

Input:

    H x W x C_in

Filter:

    K x K x C_in x C_out

Output:

    H_out x W_out x C_out

Each output channel combines information from all input channels.

For one output channel:

    Y[i,j,o] =
        sum_c sum_a sum_b
        X[i+a,j+b,c] * W[a,b,c,o]

A CNN with 3 input channels and 16 output filters
creates 16 feature maps.

======================================================================
5. RECEPTIVE FIELD
======================================================================

The receptive field is the region of the original input
that can influence one activation.

For a sequence of convolution/pooling layers:

    receptive_field_l =
        receptive_field_(l-1)
        + (kernel_l - 1) * jump_(l-1)

    jump_l = jump_(l-1) * stride_l

Initialize:

    receptive_field_0 = 1
    jump_0 = 1

Example:

    Conv 3x3, stride 1:
        RF = 3

    Pool 2x2, stride 2:
        RF = 4

    Conv 3x3, stride 1:
        RF = 8

The feature at the final layer can depend on an 8x8
region of the original image.

======================================================================
6. FEATURE MAP INTERPRETATION
======================================================================

Early CNN layers often learn local structures:

    - Edges
    - Corners
    - Lines
    - Color transitions
    - Simple textures

Deeper layers can combine these into more complex patterns:

    - Object parts
    - Curves
    - Textures
    - Shape combinations

This is an observed tendency, not a guarantee that every
individual filter has an easily interpretable meaning.

======================================================================
REAL-WORLD ROBOTICS APPLICATION
======================================================================

OPENARM V2 — WASTE SORTING

Imagine an overhead camera looking at a conveyor.

CNN filters may respond to:

    - Bottle boundaries
    - Cup rims
    - Paper edges
    - Plastic texture
    - Object contours

The resulting feature maps can be used by later layers
for classification, detection, or segmentation.

For robotic grasping, visual features must eventually be
combined with object localization and camera-to-robot geometry.

======================================================================
RUN
======================================================================

    python lesson23_cnn_operations_feature_maps.py

======================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# CONFIGURATION
# =====================================================================

SEED = 42

IMAGE_SIZE = 96


# =====================================================================
# STEP 1 — SYNTHETIC IMAGE GENERATION
# =====================================================================

def generate_scene():

    image = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.float32
    )

    # Rectangle.
    cv2.rectangle(
        image,
        (10, 12),
        (43, 55),
        180,
        -1
    )

    # Circle.
    cv2.circle(
        image,
        (70, 30),
        17,
        255,
        -1
    )

    # Triangle.
    points = np.array([
        [20, 75],
        [5, 91],
        [38, 91]
    ], dtype=np.int32)

    cv2.fillPoly(
        image,
        [points],
        220
    )

    # Diagonal line.
    cv2.line(
        image,
        (50, 60),
        (90, 90),
        255,
        3
    )

    # Add texture.
    for x in range(48, 92, 6):

        cv2.line(
            image,
            (x, 5),
            (x, 45),
            120,
            1
        )

    # Add a little noise.
    rng = np.random.default_rng(SEED)

    noise = rng.normal(
        0,
        5,
        image.shape
    )

    image = np.clip(
        image + noise,
        0,
        255
    )

    return image / 255.0


# =====================================================================
# STEP 2 — CONVOLUTION OUTPUT SIZE
# =====================================================================

def calculate_output_size(
    input_size,
    kernel_size,
    padding,
    stride
):

    return (
        (input_size + 2 * padding - kernel_size)
        // stride
    ) + 1


# =====================================================================
# STEP 3 — NUMPY CONVOLUTION
# =====================================================================

def conv2d(
    image,
    kernel,
    stride=1,
    padding=0
):

    """
    Single-channel 2D cross-correlation.

    Input:
        image: H x W
        kernel: K x K

    Output:
        H_out x W_out

    Note:
        CNN implementations commonly use cross-correlation
        rather than mathematically flipped convolution.
    """

    if image.ndim != 2 or kernel.ndim != 2:

        raise ValueError(
            "conv2d expects 2D image and kernel."
        )

    h, w = image.shape

    kh, kw = kernel.shape

    if stride < 1:

        raise ValueError(
            "Stride must be >= 1."
        )

    if padding < 0:

        raise ValueError(
            "Padding must be >= 0."
        )

    if padding > 0:

        image = np.pad(
            image,
            (
                (padding, padding),
                (padding, padding)
            ),
            mode="constant"
        )

    h_out = calculate_output_size(
        h,
        kh,
        padding,
        stride
    )

    w_out = calculate_output_size(
        w,
        kw,
        padding,
        stride
    )

    if h_out <= 0 or w_out <= 0:

        raise ValueError(
            "Kernel is too large for the padded input."
        )

    output = np.zeros(
        (h_out, w_out),
        dtype=np.float32
    )

    for y in range(h_out):

        for x in range(w_out):

            patch = image[
                y * stride:y * stride + kh,
                x * stride:x * stride + kw
            ]

            output[y, x] = np.sum(
                patch * kernel
            )

    return output


# =====================================================================
# STEP 4 — MULTI-CHANNEL CONVOLUTION
# =====================================================================

def conv2d_multi_channel(
    image,
    kernels,
    stride=1,
    padding=0
):

    """
    image:
        H,W,C_in

    kernels:
        K,K,C_in,C_out

    output:
        H_out,W_out,C_out
    """

    h, w, input_channels = image.shape

    kh, kw, kernel_channels, output_channels = kernels.shape

    if input_channels != kernel_channels:

        raise ValueError(
            "Input channel count must match kernel channel count."
        )

    output_channels_list = []

    for output_channel in range(output_channels):

        combined = None

        for input_channel in range(input_channels):

            channel_output = conv2d(
                image[:, :, input_channel],
                kernels[:, :, input_channel, output_channel],
                stride=stride,
                padding=padding
            )

            if combined is None:

                combined = channel_output

            else:

                combined += channel_output

        output_channels_list.append(
            combined
        )

    return np.stack(
        output_channels_list,
        axis=-1
    )


# =====================================================================
# STEP 5 — ACTIVATION
# =====================================================================

def relu(x):

    return np.maximum(
        x,
        0
    )


def sigmoid(x):

    x = np.clip(
        x,
        -50,
        50
    )

    return 1.0 / (
        1.0 + np.exp(-x)
    )


def normalize_for_display(x):

    minimum = np.min(x)

    maximum = np.max(x)

    if maximum - minimum < 1e-12:

        return np.zeros_like(x)

    return (
        x - minimum
    ) / (
        maximum - minimum
    )


# =====================================================================
# STEP 6 — DEFINE FILTERS
# =====================================================================

def create_filters():

    filters = {}

    filters["Horizontal edge"] = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)

    filters["Vertical edge"] = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    filters["Diagonal edge"] = np.array([
        [-1, -1,  0],
        [-1,  0,  1],
        [ 0,  1,  1]
    ], dtype=np.float32)

    filters["Sharpen"] = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    filters["Gaussian blur"] = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32) / 16.0

    filters["Laplacian"] = np.array([
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0]
    ], dtype=np.float32)

    return filters


# =====================================================================
# STEP 7 — RANDOM FILTER INITIALIZATION
# =====================================================================

def initialize_random_filters(
    num_filters=8,
    kernel_size=3,
    seed=SEED
):

    """
    He-style initialization for convolution filters.

    For a ReLU network:

        std = sqrt(2 / fan_in)

    fan_in = kernel_height * kernel_width * input_channels
    """

    rng = np.random.default_rng(seed)

    fan_in = (
        kernel_size *
        kernel_size
    )

    std = np.sqrt(
        2.0 / fan_in
    )

    filters = rng.normal(
        0,
        std,
        size=(
            num_filters,
            kernel_size,
            kernel_size
        )
    )

    return filters.astype(
        np.float32
    )


# =====================================================================
# STEP 8 — STRIDE EXPERIMENT
# =====================================================================

def stride_experiment(image):

    results = {}

    for stride in [1, 2, 3, 4]:

        output = conv2d(
            image,
            np.ones(
                (3, 3),
                dtype=np.float32
            ) / 9.0,
            stride=stride,
            padding=1
        )

        results[stride] = output

    return results


# =====================================================================
# STEP 9 — PADDING EXPERIMENT
# =====================================================================

def padding_experiment(image):

    kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    results = {}

    for padding in [0, 1, 2]:

        output = conv2d(
            image,
            kernel,
            stride=1,
            padding=padding
        )

        results[padding] = output

    return results


# =====================================================================
# STEP 10 — KERNEL SIZE EXPERIMENT
# =====================================================================

def kernel_size_experiment(image):

    kernels = {
        3: np.ones((3, 3), dtype=np.float32) / 9.0,
        5: np.ones((5, 5), dtype=np.float32) / 25.0,
        7: np.ones((7, 7), dtype=np.float32) / 49.0
    }

    results = {}

    for kernel_size, kernel in kernels.items():

        output = conv2d(
            image,
            kernel,
            stride=1,
            padding=kernel_size // 2
        )

        results[kernel_size] = output

    return results


# =====================================================================
# STEP 11 — RECEPTIVE FIELD CALCULATION
# =====================================================================

def calculate_receptive_field(layers):

    """
    Each layer is represented by:

        (kernel_size, stride)

    Example:

        Conv 3x3, stride 1
        Pool 2x2, stride 2
        Conv 3x3, stride 1

    The function returns receptive field and effective jump
    after each layer.
    """

    receptive_field = 1

    jump = 1

    results = []

    for kernel_size, stride in layers:

        receptive_field += (
            kernel_size - 1
        ) * jump

        jump *= stride

        results.append({
            "kernel": kernel_size,
            "stride": stride,
            "receptive_field": receptive_field,
            "jump": jump
        })

    return results


# =====================================================================
# STEP 12 — OPENCV COMPARISON
# =====================================================================

def compare_with_opencv(image, kernel):
    """
    Compare NumPy 2D cross-correlation with OpenCV filter2D.

    Both implementations use:
        - float64 input
        - Cross-correlation (no kernel flipping)
        - Same-size output
        - Zero-padding
    """

    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if image.ndim != 2:
        raise ValueError("Image must be a 2D grayscale array.")

    if kernel.ndim != 2:
        raise ValueError("Kernel must be a 2D array.")

    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel dimensions must be odd.")

    # ----------------------------------------------------------
    # STEP 1 — NUMPY CROSS-CORRELATION
    # ----------------------------------------------------------

    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0,
    )

    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (kh, kw),
    )

    numpy_result = np.einsum(
        "ijkl,kl->ij",
        windows,
        kernel,
    )

    # ----------------------------------------------------------
    # STEP 2 — OPENCV CROSS-CORRELATION
    # ----------------------------------------------------------

    opencv_result = cv2.filter2D(
        src=image,
        ddepth=cv2.CV_64F,
        kernel=kernel,
        borderType=cv2.BORDER_CONSTANT,
    )

    # ----------------------------------------------------------
    # STEP 3 — NUMERICAL COMPARISON
    # ----------------------------------------------------------

    error = float(
        np.max(np.abs(numpy_result - opencv_result))
    )

    return numpy_result, opencv_result, error

# =====================================================================
# STEP 13 — VISUALIZATION
# =====================================================================

def visualize_original(image):

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.title(
        "Original Synthetic Image"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()


def visualize_handcrafted_filters(
    image,
    filters
):

    fig, axes = plt.subplots(
        2,
        len(filters) + 1,
        figsize=(18, 7)
    )

    axes[0, 0].imshow(
        image,
        cmap="gray"
    )

    axes[0, 0].set_title(
        "Input"
    )

    axes[0, 0].axis("off")

    for index, (name, kernel) in enumerate(
        filters.items()
    ):

        axes[0, index + 1].imshow(
            kernel,
            cmap="coolwarm"
        )

        axes[0, index + 1].set_title(
            name
        )

        axes[0, index + 1].axis("off")

        feature_map = conv2d(
            image,
            kernel,
            padding=1
        )

        axes[1, index + 1].imshow(
            feature_map,
            cmap="gray"
        )

        axes[1, index + 1].set_title(
            f"Response: {name}"
        )

        axes[1, index + 1].axis("off")

    axes[1, 0].imshow(
        image,
        cmap="gray"
    )

    axes[1, 0].set_title(
        "Input Reference"
    )

    axes[1, 0].axis("off")

    plt.suptitle(
        "Handcrafted Convolution Filters and Feature Maps"
    )

    plt.tight_layout()

    plt.show()


def visualize_random_filters(image):

    filters = initialize_random_filters(
        num_filters=8,
        kernel_size=3
    )

    fig, axes = plt.subplots(
        2,
        8,
        figsize=(16, 6)
    )

    for index, kernel in enumerate(filters):

        feature_map = conv2d(
            image,
            kernel,
            padding=1
        )

        activated = relu(
            feature_map
        )

        axes[0, index].imshow(
            kernel,
            cmap="coolwarm"
        )

        axes[0, index].set_title(
            f"Filter {index + 1}"
        )

        axes[0, index].axis("off")

        axes[1, index].imshow(
            normalize_for_display(activated),
            cmap="viridis"
        )

        axes[1, index].set_title(
            "ReLU response"
        )

        axes[1, index].axis("off")

    plt.suptitle(
        "Randomly Initialized CNN Filters and Their Feature Maps"
    )

    plt.tight_layout()

    plt.show()


def visualize_stride(results):

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(14, 4)
    )

    for ax, (stride, output) in zip(
        axes,
        results.items()
    ):

        ax.imshow(
            normalize_for_display(output),
            cmap="gray"
        )

        ax.set_title(
            f"Stride = {stride}\n"
            f"Output = {output.shape}"
        )

        ax.axis("off")

    plt.suptitle(
        "Effect of Stride on Feature Map Resolution"
    )

    plt.tight_layout()

    plt.show()


def visualize_padding(results):

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(12, 4)
    )

    for ax, (padding, output) in zip(
        axes,
        results.items()
    ):

        ax.imshow(
            normalize_for_display(output),
            cmap="gray"
        )

        ax.set_title(
            f"Padding = {padding}\n"
            f"Shape = {output.shape}"
        )

        ax.axis("off")

    plt.suptitle(
        "Effect of Padding"
    )

    plt.tight_layout()

    plt.show()


def visualize_kernel_sizes(results):

    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(12, 4)
    )

    for ax, (kernel_size, output) in zip(
        axes,
        results.items()
    ):

        ax.imshow(
            output,
            cmap="gray"
        )

        ax.set_title(
            f"Kernel = {kernel_size}x{kernel_size}"
        )

        ax.axis("off")

    plt.suptitle(
        "Effect of Kernel Size on Smoothing"
    )

    plt.tight_layout()

    plt.show()


def visualize_multichannel_feature_maps():

    """
    Demonstrate three input channels and four output channels.
    """

    rng = np.random.default_rng(SEED)

    image = rng.random(
        (32, 32, 3)
    ).astype(np.float32)

    kernels = rng.normal(
        0,
        0.2,
        size=(3, 3, 3, 4)
    ).astype(np.float32)

    output = conv2d_multi_channel(
        image,
        kernels,
        padding=1
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(12, 4)
    )

    for index, ax in enumerate(axes):

        ax.imshow(
            normalize_for_display(
                output[:, :, index]
            ),
            cmap="viridis"
        )

        ax.set_title(
            f"Output channel {index + 1}"
        )

        ax.axis("off")

    plt.suptitle(
        "Multi-Channel Convolution: 3 Input Channels → 4 Output Channels"
    )

    plt.tight_layout()

    plt.show()


# =====================================================================
# STEP 14 — TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print("RUNNING LESSON 23 TESTS")

    print("=" * 70)

    rng = np.random.default_rng(123)

    # ---------------------------------------------------------------
    # TEST 1 — Output size
    # ---------------------------------------------------------------

    result = calculate_output_size(
        32,
        3,
        1,
        1
    )

    assert result == 32

    print("Test 1 passed: convolution output size.")

    # ---------------------------------------------------------------
    # TEST 2 — Valid convolution
    # ---------------------------------------------------------------

    image = np.ones(
        (5, 5),
        dtype=np.float32
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.float32
    )

    output = conv2d(
        image,
        kernel,
        padding=0
    )

    assert output.shape == (
        3,
        3
    )

    assert np.allclose(
        output,
        9.0
    )

    print("Test 2 passed: valid convolution.")

    # ---------------------------------------------------------------
    # TEST 3 — Same convolution
    # ---------------------------------------------------------------

    output = conv2d(
        image,
        kernel,
        padding=1
    )

    assert output.shape == (
        5,
        5
    )

    print("Test 3 passed: same-size padded convolution.")

    # ---------------------------------------------------------------
    # TEST 4 — Stride
    # ---------------------------------------------------------------

    output = conv2d(
        np.ones((8, 8), dtype=np.float32),
        np.ones((3, 3), dtype=np.float32),
        stride=2,
        padding=1
    )

    assert output.shape == (
        4,
        4
    )

    print("Test 4 passed: stride output dimensions.")

    # ---------------------------------------------------------------
    # TEST 5 — Multi-channel output shape
    # ---------------------------------------------------------------

    multi_image = rng.normal(
        size=(12, 12, 3)
    ).astype(np.float32)

    multi_kernels = rng.normal(
        size=(3, 3, 3, 5)
    ).astype(np.float32)

    multi_output = conv2d_multi_channel(
        multi_image,
        multi_kernels,
        padding=1
    )

    assert multi_output.shape == (
        12,
        12,
        5
    )

    print("Test 5 passed: multi-channel convolution.")

    # ---------------------------------------------------------------
    # TEST 6 — OpenCV comparison
    # ---------------------------------------------------------------

    test_image = rng.normal(
        size=(16, 16)
    ).astype(np.float32)

    test_kernel = rng.normal(
        size=(3, 3)
    ).astype(np.float32)

    numpy_result, cv_result, error = compare_with_opencv(
        test_image,
        test_kernel
    )

    assert error < 1e-5

    print(
        f"Test 6 passed: NumPy/OpenCV agreement. "
        f"Max error = {error:.8e}"
    )

    # ---------------------------------------------------------------
    # TEST 7 — ReLU
    # ---------------------------------------------------------------

    values = np.array([
        [-2.0, -1.0, 0.0, 2.0]
    ])

    activated = relu(values)

    assert np.allclose(
        activated,
        [[0.0, 0.0, 0.0, 2.0]]
    )

    print("Test 7 passed: ReLU activation.")

    # ---------------------------------------------------------------
    # TEST 8 — Receptive field
    # ---------------------------------------------------------------

    layers = [
        (3, 1),
        (2, 2),
        (3, 1)
    ]

    receptive_fields = calculate_receptive_field(
        layers
    )

    assert receptive_fields[-1][
        "receptive_field"
    ] == 8

    print("Test 8 passed: receptive field calculation.")

    # ---------------------------------------------------------------
    # TEST 9 — Random filter initialization
    # ---------------------------------------------------------------

    filters = initialize_random_filters(
        num_filters=8,
        kernel_size=3
    )

    assert filters.shape == (
        8,
        3,
        3
    )

    assert np.all(
        np.isfinite(filters)
    )

    print("Test 9 passed: random filter initialization.")

    # ---------------------------------------------------------------
    # TEST 10 — Kernel size experiment
    # ---------------------------------------------------------------

    test_image = rng.random(
        (32, 32)
    ).astype(np.float32)

    results = kernel_size_experiment(
        test_image
    )

    for kernel_size, result in results.items():

        assert result.shape == (
            32,
            32
        )

    print("Test 10 passed: different kernel sizes.")

    print("\nALL LESSON 23 TESTS PASSED.")


# =====================================================================
# STEP 15 — MAIN EXPERIMENT
# =====================================================================

def run_experiment():

    image = generate_scene()

    filters = create_filters()

    print("\nINPUT IMAGE")

    print(
        f"Shape: {image.shape}"
    )

    print(
        f"Minimum: {image.min():.4f}"
    )

    print(
        f"Maximum: {image.max():.4f}"
    )

    # ---------------------------------------------------------------
    # Handcrafted filters
    # ---------------------------------------------------------------

    visualize_handcrafted_filters(
        image,
        filters
    )

    # ---------------------------------------------------------------
    # Random filters
    # ---------------------------------------------------------------

    visualize_random_filters(
        image
    )

    # ---------------------------------------------------------------
    # Stride
    # ---------------------------------------------------------------

    stride_results = stride_experiment(
        image
    )

    visualize_stride(
        stride_results
    )

    # ---------------------------------------------------------------
    # Padding
    # ---------------------------------------------------------------

    padding_results = padding_experiment(
        image
    )

    visualize_padding(
        padding_results
    )

    # ---------------------------------------------------------------
    # Kernel size
    # ---------------------------------------------------------------

    kernel_results = kernel_size_experiment(
        image
    )

    visualize_kernel_sizes(
        kernel_results
    )

    # ---------------------------------------------------------------
    # Multi-channel convolution
    # ---------------------------------------------------------------

    visualize_multichannel_feature_maps()

    # ---------------------------------------------------------------
    # Receptive field
    # ---------------------------------------------------------------

    layers = [
        (3, 1),
        (2, 2),
        (3, 1),
        (2, 2),
        (3, 1)
    ]

    results = calculate_receptive_field(
        layers
    )

    print("\nRECEPTIVE FIELD ANALYSIS")

    print("-" * 70)

    for index, result in enumerate(results):

        print(
            f"Layer {index + 1}: "
            f"Kernel={result['kernel']} | "
            f"Stride={result['stride']} | "
            f"Receptive field={result['receptive_field']} | "
            f"Jump={result['jump']}"
        )

    # ---------------------------------------------------------------
    # OpenCV comparison
    # ---------------------------------------------------------------

    test_kernel = filters[
        "Vertical edge"
    ]

    numpy_result, cv_result, error = compare_with_opencv(
        image,
        test_kernel
    )

    print("\nOPENCV COMPARISON")

    print(
        f"Maximum absolute difference: {error:.10f}"
    )

    assert error < 1e-5

    print(
        "NumPy and OpenCV results agree."
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)

    print("COMPUTER VISION — PHASE 2 — LESSON 23")

    print("CNN OPERATIONS AND FEATURE MAPS")

    print("=" * 70)

    run_tests()

    run_experiment()

    print("\n" + "=" * 70)

    print("LESSON 23 COMPLETED")

    print("=" * 70)

    print(
        "NEXT: LESSON 24 — RESNET FROM SCRATCH"
    )


if __name__ == "__main__":

    main()