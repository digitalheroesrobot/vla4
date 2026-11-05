"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 04 — IMAGE FILTERING AND CONVOLUTION
======================================================================

COURSE RULE
-----------
Everything important is implemented from scratch.

NO:
    - pretrained models
    - pretrained weights
    - cv2 filtering functions for the main implementation
    - scipy convolution for the main implementation

We use:
    - NumPy
    - Matplotlib

----------------------------------------------------------------------
WHAT YOU LEARN
----------------------------------------------------------------------

STEP 1 — Why image filtering?

Real camera images contain:

    - sensor noise
    - compression artifacts
    - illumination variation
    - small unwanted details

Filtering allows us to modify an image based on its local
neighborhood.

Conceptually:

                neighborhood
                    |
                    v
              +-----------+
              | x x x x x |
              | x x x x x |
              | x x x x x |
              +-----------+
                    |
                    v
              output pixel

A filter examines nearby pixels and calculates a new value.

----------------------------------------------------------------------
STEP 2 — KERNELS
----------------------------------------------------------------------

A kernel is a small matrix of numbers.

Example:

        3 × 3 averaging kernel

        1/9  1/9  1/9
        1/9  1/9  1/9
        1/9  1/9  1/9

The kernel slides over the image.

At each position:

        multiply
            ↓
        add
            ↓
        produce output pixel

----------------------------------------------------------------------
STEP 3 — CONVOLUTION MATHEMATICS
----------------------------------------------------------------------

For a 2-D image I and kernel K:

        O(i,j)
        =
        Σ Σ I(i+m, j+n) K(m,n)

In strict mathematical convolution, the kernel is flipped
before multiplication.

In many computer vision and deep-learning implementations,
the operation actually performed is cross-correlation:

        O(i,j)
        =
        Σ Σ I(i+m, j+n) K(m,n)

without flipping the kernel.

For symmetric filters such as averaging and Gaussian kernels,
the distinction does not matter because:

        K = flipped(K)

We will explicitly implement the local weighted-sum operation.

----------------------------------------------------------------------
STEP 4 — PADDING
----------------------------------------------------------------------

A 3 × 3 kernel cannot be centered on a border pixel unless
we define values outside the image.

One solution is zero padding.

Example:

        original image
             ↓
        add border of zeros
             ↓
        apply kernel
             ↓
        output same size

Other padding strategies include:

        - replicate
        - reflect
        - circular
        - constant

We will implement zero padding manually.

----------------------------------------------------------------------
STEP 5 — AVERAGING FILTER
----------------------------------------------------------------------

A simple smoothing kernel:

        [1 1 1]
        [1 1 1] × 1/9
        [1 1 1]

This reduces local variation.

Effect:

        noise ↓
        sharpness ↓

----------------------------------------------------------------------
STEP 6 — GAUSSIAN FILTER
----------------------------------------------------------------------

The Gaussian function is:

        G(x,y)
        =
        1/(2πσ²)
        exp(
            -(x²+y²)/(2σ²)
        )

A discrete Gaussian kernel approximates this function.

Gaussian filtering gives greater weight to pixels near the
center of the kernel.

For example:

        low weight    medium weight    low weight
        medium        high             medium
        low weight    medium weight    low weight

Gaussian smoothing is generally more natural than simple
uniform averaging.

----------------------------------------------------------------------
STEP 7 — NOISE
----------------------------------------------------------------------

We will artificially add Gaussian noise:

        noisy = image + ε

where:

        ε ~ N(0, σ²)

This lets us directly measure whether filtering reduces noise.

----------------------------------------------------------------------
STEP 8 — FILTER SIZE
----------------------------------------------------------------------

Common kernel sizes:

        3 × 3
        5 × 5
        7 × 7

Larger kernels:

        stronger smoothing
        more computation
        more loss of fine detail

There is always a trade-off.

----------------------------------------------------------------------
STEP 9 — STRIDE
----------------------------------------------------------------------

Stride describes how far the kernel moves at each step.

Stride = 1:

        examine every neighboring position.

Stride = 2:

        move two pixels at a time.

Later, this becomes an important concept in CNNs.

----------------------------------------------------------------------
STEP 10 — CONNECTION TO CNNs
----------------------------------------------------------------------

This lesson is extremely important for deep learning.

A CNN also performs local weighted operations:

        image
          ↓
        kernel
          ↓
        feature map

The major difference is:

Classical filtering:
    kernel values are manually designed.

CNN:
    kernel values are learned from data.

Therefore:

    Classical filtering
        ↓
    mathematical foundation
        ↓
    convolution
        ↓
    CNN convolution
        ↓
    learned visual features

----------------------------------------------------------------------
REAL-WORLD ROBOTICS EXAMPLE
----------------------------------------------------------------------

Consider an OpenArm V2 waste-sorting robot.

A camera observes:

        plastic
        paper
        cans
        bottles
        food waste

The camera image may contain sensor noise.

Before extracting edges or other visual information:

        camera
          ↓
        noisy image
          ↓
        smoothing filter
          ↓
        cleaner image
          ↓
        edge detection
          ↓
        object perception

However:

    Too much smoothing can remove important object boundaries.

Therefore filtering must be chosen carefully.

----------------------------------------------------------------------
LESSON EXPERIMENTS
----------------------------------------------------------------------

We will:

    1. Create a clean image.
    2. Add Gaussian noise.
    3. Implement convolution from scratch.
    4. Apply an averaging filter.
    5. Create a Gaussian kernel.
    6. Apply Gaussian filtering.
    7. Compare noise statistics.
    8. Compare different kernel sizes.
    9. Visualize the result.
   10. Test the implementation numerically.

----------------------------------------------------------------------
EXERCISES
----------------------------------------------------------------------

EXERCISE 1
----------
Change the noise level.

Try:

    sigma = 5
    sigma = 15
    sigma = 30
    sigma = 50

Observe how filtering behaves.

EXERCISE 2
----------
Change the averaging kernel from:

    3 × 3

to:

    5 × 5

and:

    7 × 7

Compare the results.

EXERCISE 3
----------
Create a Gaussian kernel with:

    size = 5
    sigma = 1.0

Then try:

    sigma = 0.5
    sigma = 2.0

Observe how the kernel changes.

EXERCISE 4
----------
Implement a sharpening kernel:

        [ 0 -1  0]
        [-1  5 -1]
        [ 0 -1  0]

Apply it to an image.

EXERCISE 5
----------
Implement a horizontal edge detector:

        [-1 -1 -1]
        [ 0  0  0]
        [ 1  1  1]

Observe the result.

EXERCISE 6
----------
Implement a vertical edge detector:

        [-1  0  1]
        [-1  0  1]
        [-1  0  1]

These kernels will prepare you for the next lessons.

----------------------------------------------------------------------
NEXT LESSON
----------------------------------------------------------------------

LESSON 05 — EDGE DETECTION FROM SCRATCH

We will implement:

    - image gradients
    - Sobel operators
    - gradient magnitude
    - gradient direction
    - edge thresholding
    - Canny-style edge detection concepts

======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# STEP 1 — CREATE A CLEAN SYNTHETIC IMAGE
# ======================================================================

HEIGHT = 256
WIDTH = 384

image = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

# Background
image[:, :] = 30.0

# Gray rectangle
image[40:130, 40:160] = 100.0

# Bright rectangle
image[70:200, 210:350] = 220.0

# White square
image[160:220, 70:150] = 255.0

# Circular object
yy, xx = np.ogrid[:HEIGHT, :WIDTH]

center_y = 125
center_x = 190
radius = 40

circle_mask = (
    (yy - center_y) ** 2
    +
    (xx - center_x) ** 2
    <= radius ** 2
)

image[circle_mask] = 180.0


# ======================================================================
# STEP 2 — ADD GAUSSIAN NOISE
# ======================================================================
#
# Gaussian noise:
#
#     ε ~ N(0, σ²)
#
# noisy image:
#
#     I_noisy = I + ε
#
# We use a fixed random seed so that the experiment is reproducible.

rng = np.random.default_rng(42)

noise_sigma = 25.0

noise = rng.normal(
    loc=0.0,
    scale=noise_sigma,
    size=image.shape
)

noisy_image = image + noise

noisy_image = np.clip(
    noisy_image,
    0,
    255
).astype(np.float32)


# ======================================================================
# STEP 3 — IMPLEMENT PADDING
# ======================================================================
#
# For a kernel of size K:
#
#     padding = K // 2
#
# For K = 3:
#
#     padding = 1
#
# For K = 5:
#
#     padding = 2

def zero_pad(image, padding):
    """
    Add zero padding around a 2-D image.

    Example:

        input:
            H × W

        padding = 1

        output:
            (H+2) × (W+2)
    """

    padded = np.pad(
        image,
        (
            (padding, padding),
            (padding, padding)
        ),
        mode="constant",
        constant_values=0
    )

    return padded


# ======================================================================
# STEP 4 — IMPLEMENT CONVOLUTION / CROSS-CORRELATION
# ======================================================================
#
# This is the central implementation of this lesson.
#
# We deliberately use explicit loops.
#
# This is not the fastest implementation.
#
# The purpose is to make the mathematics visible.
#
# For every output pixel:
#
#     neighborhood × kernel
#                 ↓
#              sum
#                 ↓
#            output pixel

def convolve2d(image, kernel):
    """
    Apply a 2-D kernel to a grayscale image.

    This implementation uses:
        - zero padding
        - stride = 1
        - explicit loops

    The operation is cross-correlation.

    For symmetric kernels such as averaging and Gaussian kernels,
    cross-correlation and mathematical convolution produce the
    same result.
    """

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    assert kernel_height % 2 == 1
    assert kernel_width % 2 == 1

    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    padded = zero_pad(
        image,
        max(pad_y, pad_x)
    )

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    for y in range(image_height):

        for x in range(image_width):

            region = padded[
                y:y + kernel_height,
                x:x + kernel_width
            ]

            output[y, x] = np.sum(
                region * kernel
            )

    return output


# ======================================================================
# STEP 5 — AVERAGING KERNEL
# ======================================================================

average_kernel = (
    np.ones((3, 3), dtype=np.float32)
    / 9.0
)


# ======================================================================
# STEP 6 — APPLY AVERAGING FILTER
# ======================================================================

average_filtered = convolve2d(
    noisy_image,
    average_kernel
)

average_filtered = np.clip(
    average_filtered,
    0,
    255
)


# ======================================================================
# STEP 7 — CREATE GAUSSIAN KERNEL
# ======================================================================
#
# Gaussian equation:
#
#     G(x,y)
#     =
#     exp(
#         -(x²+y²)/(2σ²)
#     )
#
# We normalize the kernel:
#
#     K = K / ΣK
#
# so that the overall brightness is approximately preserved.

def gaussian_kernel(size, sigma):
    """
    Create a normalized 2-D Gaussian kernel from scratch.
    """

    assert size % 2 == 1
    assert sigma > 0

    radius = size // 2

    coordinates = np.arange(
        -radius,
        radius + 1
    )

    kernel = np.zeros(
        (size, size),
        dtype=np.float32
    )

    for y in range(size):

        for x in range(size):

            dx = coordinates[x]
            dy = coordinates[y]

            exponent = -(
                dx ** 2 + dy ** 2
            ) / (
                2.0 * sigma ** 2
            )

            kernel[y, x] = np.exp(
                exponent
            )

    kernel /= kernel.sum()

    return kernel


# ======================================================================
# STEP 8 — CREATE GAUSSIAN KERNEL
# ======================================================================

gaussian_3 = gaussian_kernel(
    size=3,
    sigma=1.0
)

gaussian_5 = gaussian_kernel(
    size=5,
    sigma=1.0
)

gaussian_7 = gaussian_kernel(
    size=7,
    sigma=1.5
)


# ======================================================================
# STEP 9 — APPLY GAUSSIAN FILTER
# ======================================================================

gaussian_filtered_3 = convolve2d(
    noisy_image,
    gaussian_3
)

gaussian_filtered_5 = convolve2d(
    noisy_image,
    gaussian_5
)

gaussian_filtered_7 = convolve2d(
    noisy_image,
    gaussian_7
)


gaussian_filtered_3 = np.clip(
    gaussian_filtered_3,
    0,
    255
)

gaussian_filtered_5 = np.clip(
    gaussian_filtered_5,
    0,
    255
)

gaussian_filtered_7 = np.clip(
    gaussian_filtered_7,
    0,
    255
)


# ======================================================================
# STEP 10 — IMAGE ERROR / NOISE MEASUREMENT
# ======================================================================
#
# Since we know the original clean synthetic image, we can calculate
# the mean absolute error.
#
# MAE:
#
#     MAE = mean(|I_original - I_filtered|)
#
# Lower MAE means the filtered image is numerically closer to the
# original synthetic image.

def mean_absolute_error(reference, prediction):
    return np.mean(
        np.abs(
            reference - prediction
        )
    )


noisy_mae = mean_absolute_error(
    image,
    noisy_image
)

average_mae = mean_absolute_error(
    image,
    average_filtered
)

gaussian_mae_3 = mean_absolute_error(
    image,
    gaussian_filtered_3
)

gaussian_mae_5 = mean_absolute_error(
    image,
    gaussian_filtered_5
)

gaussian_mae_7 = mean_absolute_error(
    image,
    gaussian_filtered_7
)


# ======================================================================
# STEP 11 — PRINT BASIC INFORMATION
# ======================================================================

print("=" * 72)
print("LESSON 04 — IMAGE FILTERING AND CONVOLUTION")
print("=" * 72)

print("\nIMAGE")
print("-" * 72)

print("Shape:", image.shape)
print("Data type:", image.dtype)

print("\nNOISE")
print("-" * 72)

print("Noise sigma:", noise_sigma)

print("\nKERNELS")
print("-" * 72)

print("Averaging kernel:")
print(average_kernel)

print("\nGaussian 3x3 kernel:")
print(gaussian_3)

print("\nGaussian 5x5 kernel:")
print(gaussian_5)

print("\nGaussian 7x7 kernel:")
print(gaussian_7)

print("\nKERNEL SUMS")
print("-" * 72)

print("Average kernel sum:", average_kernel.sum())
print("Gaussian 3 sum:", gaussian_3.sum())
print("Gaussian 5 sum:", gaussian_5.sum())
print("Gaussian 7 sum:", gaussian_7.sum())


# ======================================================================
# STEP 12 — ERROR COMPARISON
# ======================================================================

print("\nFILTERING ERROR")
print("-" * 72)

print(
    f"Noisy image MAE       : {noisy_mae:.4f}"
)

print(
    f"Average 3x3 MAE       : {average_mae:.4f}"
)

print(
    f"Gaussian 3x3 MAE      : {gaussian_mae_3:.4f}"
)

print(
    f"Gaussian 5x5 MAE      : {gaussian_mae_5:.4f}"
)

print(
    f"Gaussian 7x7 MAE      : {gaussian_mae_7:.4f}"
)


# ======================================================================
# STEP 13 — TEST KERNEL PROPERTIES
# ======================================================================

print("\nRUNNING TESTS")
print("-" * 72)

# Averaging kernel shape
assert average_kernel.shape == (3, 3)

# Averaging kernel sum
assert np.isclose(
    average_kernel.sum(),
    1.0
)

# Gaussian kernel shapes
assert gaussian_3.shape == (3, 3)
assert gaussian_5.shape == (5, 5)
assert gaussian_7.shape == (7, 7)

# Gaussian kernels should sum to approximately one.
assert np.isclose(
    gaussian_3.sum(),
    1.0
)

assert np.isclose(
    gaussian_5.sum(),
    1.0
)

assert np.isclose(
    gaussian_7.sum(),
    1.0
)

# Gaussian kernels should be symmetric.
assert np.allclose(
    gaussian_3,
    np.flipud(np.fliplr(gaussian_3))
)

assert np.allclose(
    gaussian_5,
    np.flipud(np.fliplr(gaussian_5))
)

# Output shape must equal input shape.
assert average_filtered.shape == image.shape
assert gaussian_filtered_3.shape == image.shape
assert gaussian_filtered_5.shape == image.shape
assert gaussian_filtered_7.shape == image.shape

# Output must remain within image range after clipping.
assert average_filtered.min() >= 0
assert average_filtered.max() <= 255

assert gaussian_filtered_3.min() >= 0
assert gaussian_filtered_3.max() <= 255

# MAE must be non-negative.
assert noisy_mae >= 0
assert average_mae >= 0
assert gaussian_mae_3 >= 0

print("All tests passed.")


# ======================================================================
# STEP 14 — VISUALIZE ORIGINAL AND NOISY IMAGES
# ======================================================================

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)

plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Original Clean Image")
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    noisy_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"Noisy Image (σ={noise_sigma})"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 15 — COMPARE FILTERS
# ======================================================================

plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)

plt.imshow(
    noisy_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"Noisy\nMAE={noisy_mae:.2f}"
)

plt.axis("off")


plt.subplot(2, 3, 2)

plt.imshow(
    average_filtered,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"Average 3x3\nMAE={average_mae:.2f}"
)

plt.axis("off")


plt.subplot(2, 3, 3)

plt.imshow(
    gaussian_filtered_3,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"Gaussian 3x3\nMAE={gaussian_mae_3:.2f}"
)

plt.axis("off")


plt.subplot(2, 3, 4)

plt.imshow(
    gaussian_filtered_5,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"Gaussian 5x5\nMAE={gaussian_mae_5:.2f}"
)

plt.axis("off")


plt.subplot(2, 3, 5)

plt.imshow(
    gaussian_filtered_7,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    f"Gaussian 7x7\nMAE={gaussian_mae_7:.2f}"
)

plt.axis("off")


plt.subplot(2, 3, 6)

plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Original")
plt.axis("off")


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 16 — VISUALIZE GAUSSIAN KERNELS
# ======================================================================

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)

plt.imshow(
    gaussian_3,
    cmap="gray"
)

plt.title("Gaussian 3×3")
plt.colorbar()


plt.subplot(1, 3, 2)

plt.imshow(
    gaussian_5,
    cmap="gray"
)

plt.title("Gaussian 5×5")
plt.colorbar()


plt.subplot(1, 3, 3)

plt.imshow(
    gaussian_7,
    cmap="gray"
)

plt.title("Gaussian 7×7")
plt.colorbar()


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 17 — TEST CONVOLUTION WITH AN IMPULSE
# ======================================================================
#
# An impulse image contains:
#
#     all zeros
#
# except one pixel:
#
#     center = 1
#
# When we filter an impulse image, the output should reproduce
# the kernel around the impulse location.
#
# This is a fundamental property of linear filtering.

impulse = np.zeros(
    (21, 21),
    dtype=np.float32
)

impulse[10, 10] = 1.0

impulse_response = convolve2d(
    impulse,
    gaussian_5
)


# Extract the central region.
central_response = impulse_response[
    8:13,
    8:13
]


print("\nIMPULSE RESPONSE TEST")
print("-" * 72)

print("Kernel:")
print(gaussian_5)

print("\nImpulse response:")
print(central_response)


assert np.allclose(
    central_response,
    gaussian_5,
    atol=1e-6
)

print("\nImpulse response matches Gaussian kernel.")


# ======================================================================
# STEP 18 — VISUALIZE IMPULSE RESPONSE
# ======================================================================

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)

plt.imshow(
    gaussian_5,
    cmap="gray"
)

plt.title("Gaussian Kernel")
plt.colorbar()


plt.subplot(1, 2, 2)

plt.imshow(
    impulse_response,
    cmap="gray"
)

plt.title("Impulse Response")
plt.colorbar()


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 19 — SIMPLE SHARPENING FILTER
# ======================================================================
#
# A sharpening kernel:
#
#         0  -1   0
#        -1   5  -1
#         0  -1   0
#
# It emphasizes differences between a pixel and its neighbors.

sharpen_kernel = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ],
    dtype=np.float32
)

sharpened = convolve2d(
    image,
    sharpen_kernel
)

sharpened = np.clip(
    sharpened,
    0,
    255
)


# ======================================================================
# STEP 20 — EDGE-RELATED FILTERS
# ======================================================================
#
# These are simple derivative-like filters.
#
# Horizontal gradient:
#
#         -1 -1 -1
#          0  0  0
#          1  1  1
#
# Vertical gradient:
#
#         -1  0  1
#         -1  0  1
#         -1  0  1

horizontal_kernel = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ],
    dtype=np.float32
)

vertical_kernel = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ],
    dtype=np.float32
)

horizontal_response = convolve2d(
    image,
    horizontal_kernel
)

vertical_response = convolve2d(
    image,
    vertical_kernel
)


# ======================================================================
# STEP 21 — VISUALIZE SHARPENING AND EDGE RESPONSES
# ======================================================================

plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)

plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Original")
plt.axis("off")


plt.subplot(2, 2, 2)

plt.imshow(
    sharpened,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Sharpened")
plt.axis("off")


plt.subplot(2, 2, 3)

plt.imshow(
    horizontal_response,
    cmap="gray"
)

plt.title("Horizontal Gradient-like Response")
plt.axis("off")


plt.subplot(2, 2, 4)

plt.imshow(
    vertical_response,
    cmap="gray"
)

plt.title("Vertical Gradient-like Response")
plt.axis("off")


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 22 — REAL-WORLD INTERPRETATION
# ======================================================================
#
# Imagine a robot camera observing a pile of waste.
#
# Raw camera frame:
#
#     ┌─────────────────────────────┐
#     │ noisy visual information    │
#     └──────────────┬──────────────┘
#                    ↓
#             noise reduction
#                    ↓
#             image filtering
#                    ↓
#             edge extraction
#                    ↓
#             object boundaries
#                    ↓
#             object detection
#                    ↓
#             grasp planning
#
# The filter is NOT the final perception system.
#
# It is one of the mathematical building blocks.
#
# Later:
#
#     filtering
#         ↓
#     gradients
#         ↓
#     edges
#         ↓
#     features
#         ↓
#     objects
#         ↓
#     robot actions
#
# This is the beginning of the classical computer vision pipeline.


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 72)
print("LESSON 04 COMPLETE")
print("=" * 72)

print(
    """
You learned:

1. Images can be processed using local neighborhoods.
2. A kernel defines a local weighted operation.
3. Convolution/cross-correlation can be implemented with loops.
4. Padding allows filtering near image boundaries.
5. Averaging filters reduce local noise.
6. Gaussian filters provide distance-dependent weighting.
7. Larger kernels generally produce stronger smoothing.
8. Filtering creates a trade-off between noise reduction and detail.
9. Edge-like filters can be created using derivative kernels.
10. The same mathematical idea of convolution becomes fundamental
    to convolutional neural networks.

The most important connection:

        CLASSICAL COMPUTER VISION
                 |
                 v
             KERNEL
                 |
                 v
            CONVOLUTION
                 |
                 v
             FEATURE
                 |
                 |
                 v
        CONVOLUTIONAL NEURAL
             NETWORK
                 |
                 v
        LEARNED KERNELS

In classical CV, we design the kernel.

In deep learning, the network learns the kernel.

NEXT:
LESSON 05 — EDGE DETECTION FROM SCRATCH
"""
)