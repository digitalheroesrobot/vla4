"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 05 — EDGE DETECTION FROM SCRATCH
======================================================================

COURSE RULE
-----------
This course is built from scratch.

NO:
    - pretrained models
    - pretrained weights
    - pretrained feature extractors
    - cv2 edge detection for the main implementation
    - black-box edge detectors

We use:
    - NumPy
    - Matplotlib

----------------------------------------------------------------------
WHAT YOU LEARN
----------------------------------------------------------------------

STEP 1 — What is an edge?

An edge is a location where image intensity changes significantly.

Example:

    dark region
    -----------
    30 30 30 30 30
    30 30 30 30 30
    30 30 30 30 30
    -----------

              ↓

    bright region
    -----------
    220 220 220 220
    220 220 220 220

The transition from approximately 30 → 220 is an edge.

Edges often correspond to:

    - object boundaries
    - surface boundaries
    - texture changes
    - shadows
    - depth discontinuities

----------------------------------------------------------------------
STEP 2 — IMAGE GRADIENT
----------------------------------------------------------------------

For an image I(x,y), we can calculate how intensity changes
in the horizontal and vertical directions.

Horizontal derivative:

        ∂I/∂x

Vertical derivative:

        ∂I/∂y

We approximate these derivatives numerically.

A simple horizontal derivative is:

        [-1  0  1]

A simple vertical derivative is:

        [-1]
        [ 0]
        [ 1]

----------------------------------------------------------------------
STEP 3 — GRADIENT VECTOR
----------------------------------------------------------------------

At every pixel:

        ∇I =
        [ Ix ]
        [ Iy ]

where:

        Ix = ∂I/∂x
        Iy = ∂I/∂y

The gradient points toward the direction of greatest
intensity increase.

----------------------------------------------------------------------
STEP 4 — GRADIENT MAGNITUDE
----------------------------------------------------------------------

The strength of the gradient is:

        |∇I|
        =
        sqrt(Ix² + Iy²)

A large magnitude indicates a strong intensity change.

Therefore:

        large gradient → possible edge

----------------------------------------------------------------------
STEP 5 — GRADIENT DIRECTION
----------------------------------------------------------------------

The gradient direction is:

        θ = atan2(Iy, Ix)

This tells us the direction in which intensity changes
most strongly.

IMPORTANT:

The gradient direction is perpendicular to the edge direction.

Example:

If an object has a vertical boundary:

        object | background

the intensity changes horizontally.

Therefore:

        gradient direction → horizontal

while:

        edge direction → vertical

----------------------------------------------------------------------
STEP 6 — SOBEL OPERATORS
----------------------------------------------------------------------

The Sobel operators combine:

    - differentiation
    - local smoothing

Horizontal Sobel kernel:

        [-1  0  1]
        [-2  0  2]
        [-1  0  1]

Vertical Sobel kernel:

        [-1 -2 -1]
        [ 0  0  0]
        [ 1  2  1]

We will implement these kernels using the convolution
implementation from Lesson 4.

----------------------------------------------------------------------
STEP 7 — EDGE MAGNITUDE
----------------------------------------------------------------------

After calculating:

        Gx
        Gy

we calculate:

        magnitude =
        sqrt(Gx² + Gy²)

We then normalize the magnitude for visualization.

----------------------------------------------------------------------
STEP 8 — EDGE THRESHOLDING
----------------------------------------------------------------------

Not every gradient is considered an edge.

We can define:

        edge(x,y) =
            1 if magnitude(x,y) > T
            0 otherwise

where T is a threshold.

This creates a binary edge map.

----------------------------------------------------------------------
STEP 9 — WHY EDGES ARE IMPORTANT
----------------------------------------------------------------------

Edges are useful because they often survive changes in
absolute brightness better than raw pixel values.

For example:

    dark object
    bright background

and:

    bright object
    very bright background

may have different pixel values, but the boundary can still
produce a strong gradient.

Edges therefore provide structural information.

----------------------------------------------------------------------
REAL-WORLD ROBOTICS EXAMPLE
----------------------------------------------------------------------

Imagine an OpenArm V2 robot sorting waste.

The camera sees:

        ┌──────────────────────────┐
        │ background               │
        │                          │
        │       ______             │
        │      /      \            │
        │     | bottle |           │
        │      \______/             │
        │                          │
        └──────────────────────────┘

The raw RGB image contains thousands of pixel values.

An edge detector can reduce this to something closer to:

        object boundary
              ↓
        ┌──────────┐
        │          │
        │  EDGE    │
        │          │
        └──────────┘

This structural information can later be used by:

    - contour detection
    - shape analysis
    - object localization
    - segmentation
    - feature extraction

----------------------------------------------------------------------
LESSON EXPERIMENTS
----------------------------------------------------------------------

We will:

    1. Create synthetic objects.
    2. Add image noise.
    3. Calculate horizontal gradients.
    4. Calculate vertical gradients.
    5. Calculate gradient magnitude.
    6. Calculate gradient direction.
    7. Apply Sobel operators.
    8. Threshold the gradient.
    9. Compare thresholds.
   10. Test the mathematical properties.

----------------------------------------------------------------------
IMPORTANT NUMERICAL ISSUE
----------------------------------------------------------------------

Derivative filters produce negative values.

For example:

    dark → bright

may produce:

    positive response

while:

    bright → dark

may produce:

    negative response

Therefore we should NOT immediately store derivative results
in uint8.

Instead use:

    float32

for gradient calculations.

This is a very important computer vision programming rule.

----------------------------------------------------------------------
EXERCISES
----------------------------------------------------------------------

EXERCISE 1
----------
Change the edge threshold.

Try:

    20
    40
    80
    120

Observe how the edge map changes.


EXERCISE 2
----------
Add Gaussian noise to the image.

Compare:

    edges without noise
    edges with noise


EXERCISE 3
----------
Apply Gaussian smoothing before Sobel edge detection.

Compare the result with the noisy image.

Pipeline:

    noisy image
         ↓
    Gaussian filter
         ↓
    Sobel
         ↓
    edge magnitude


EXERCISE 4
----------
Implement simple derivative kernels:

    [-1, 0, 1]

and:

    [-1]
    [ 0]
    [ 1]

Compare them with Sobel.


EXERCISE 5
----------
Calculate the histogram of gradient magnitude.


EXERCISE 6
----------
Create a diagonal edge.

Measure the gradient direction.

----------------------------------------------------------------------
NEXT LESSON
----------------------------------------------------------------------

LESSON 06 — THRESHOLDING AND BINARY IMAGE SEGMENTATION

We will implement:

    - global thresholding
    - adaptive thresholding
    - Otsu thresholding from scratch
    - binary masks
    - foreground/background separation
    - segmentation evaluation

======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# STEP 1 — CREATE A SYNTHETIC IMAGE
# ======================================================================

HEIGHT = 256
WIDTH = 384

image = np.zeros(
    (HEIGHT, WIDTH),
    dtype=np.float32
)

# Background
image[:, :] = 30.0

# Large gray rectangle
image[40:140, 40:170] = 110.0

# Bright rectangle
image[80:210, 220:350] = 220.0

# White square
image[160:220, 60:140] = 255.0


# Circle
yy, xx = np.ogrid[:HEIGHT, :WIDTH]

circle_center_y = 120
circle_center_x = 195
circle_radius = 42

circle_mask = (
    (yy - circle_center_y) ** 2
    +
    (xx - circle_center_x) ** 2
    <= circle_radius ** 2
)

image[circle_mask] = 180.0


# ======================================================================
# STEP 2 — ADD A DIAGONAL OBJECT
# ======================================================================
#
# This helps demonstrate gradient direction.

diagonal_mask = (
    np.abs(
        yy.astype(np.float32)
        -
        (0.5 * xx.astype(np.float32) + 20)
    )
    < 3
)

image[diagonal_mask] = 255.0


# ======================================================================
# STEP 3 — REUSE FROM-SCRATCH CONVOLUTION
# ======================================================================

def zero_pad(image, padding):
    """
    Add zero padding to a 2-D image.
    """

    return np.pad(
        image,
        (
            (padding, padding),
            (padding, padding)
        ),
        mode="constant",
        constant_values=0
    )


def convolve2d(image, kernel):
    """
    Apply a 2-D kernel using explicit loops.

    This is intentionally written from scratch so the local
    filtering operation is visible.

    The operation is cross-correlation.

    For symmetric kernels the result is equivalent to convolution.
    """

    image_height, image_width = image.shape

    kernel_height, kernel_width = kernel.shape

    assert kernel_height % 2 == 1
    assert kernel_width % 2 == 1

    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    padding = max(
        pad_y,
        pad_x
    )

    padded = zero_pad(
        image,
        padding
    )

    output = np.zeros(
        image.shape,
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
# STEP 4 — SIMPLE DERIVATIVE FILTERS
# ======================================================================
#
# Horizontal derivative:
#
#       [-1  0  1]
#
# Vertical derivative:
#
#       [-1]
#       [ 0]
#       [ 1]

simple_x_kernel = np.array(
    [
        [-1, 0, 1]
    ],
    dtype=np.float32
)

simple_y_kernel = np.array(
    [
        [-1],
        [0],
        [1]
    ],
    dtype=np.float32
)


# ======================================================================
# STEP 5 — SOBEL KERNELS
# ======================================================================
#
# Sobel X:
#
#       [-1  0  1]
#       [-2  0  2]
#       [-1  0  1]
#
# Sobel Y:
#
#       [-1 -2 -1]
#       [ 0  0  0]
#       [ 1  2  1]

sobel_x = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ],
    dtype=np.float32
)

sobel_y = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ],
    dtype=np.float32
)


# ======================================================================
# STEP 6 — CALCULATE SIMPLE GRADIENTS
# ======================================================================

simple_gx = convolve2d(
    image,
    simple_x_kernel
)

simple_gy = convolve2d(
    image,
    simple_y_kernel
)


# ======================================================================
# STEP 7 — CALCULATE SOBEL GRADIENTS
# ======================================================================

gx = convolve2d(
    image,
    sobel_x
)

gy = convolve2d(
    image,
    sobel_y
)


# ======================================================================
# STEP 8 — GRADIENT MAGNITUDE
# ======================================================================
#
# Formula:
#
#       M = sqrt(Gx² + Gy²)
#
# We use np.hypot() because it computes:
#
#       sqrt(x² + y²)
#
# in a numerically stable way.

gradient_magnitude = np.hypot(
    gx,
    gy
)


# ======================================================================
# STEP 9 — GRADIENT DIRECTION
# ======================================================================
#
# Formula:
#
#       θ = atan2(Gy, Gx)
#
# Result is in radians.
#
# Convert to degrees for easier interpretation.

gradient_direction = np.arctan2(
    gy,
    gx
)

gradient_direction_degrees = (
    np.degrees(
        gradient_direction
    )
)


# ======================================================================
# STEP 10 — NORMALIZE GRADIENT MAGNITUDE
# ======================================================================
#
# For visualization:
#
#       M_norm = M / max(M) × 255

max_magnitude = gradient_magnitude.max()

if max_magnitude > 0:

    magnitude_normalized = (
        gradient_magnitude
        /
        max_magnitude
        *
        255.0
    )

else:

    magnitude_normalized = np.zeros_like(
        gradient_magnitude
    )


# ======================================================================
# STEP 11 — THRESHOLD EDGE MAGNITUDE
# ======================================================================
#
# Binary edge map:
#
#       edge = 1 if M > threshold
#       edge = 0 otherwise

threshold = 80.0

edge_map = (
    gradient_magnitude > threshold
)


# ======================================================================
# STEP 12 — DIFFERENT THRESHOLDS
# ======================================================================

threshold_low = 30.0
threshold_medium = 80.0
threshold_high = 150.0

edges_low = (
    gradient_magnitude > threshold_low
)

edges_medium = (
    gradient_magnitude > threshold_medium
)

edges_high = (
    gradient_magnitude > threshold_high
)


# ======================================================================
# STEP 13 — PRINT INFORMATION
# ======================================================================

print("=" * 72)
print("LESSON 05 — EDGE DETECTION FROM SCRATCH")
print("=" * 72)

print("\nIMAGE")
print("-" * 72)

print("Shape:", image.shape)
print("Min intensity:", image.min())
print("Max intensity:", image.max())
print("Mean intensity:", image.mean())


print("\nSOBEL KERNELS")
print("-" * 72)

print("Sobel X:")
print(sobel_x)

print("\nSobel Y:")
print(sobel_y)


print("\nGRADIENT INFORMATION")
print("-" * 72)

print("Gx min:", gx.min())
print("Gx max:", gx.max())

print("Gy min:", gy.min())
print("Gy max:", gy.max())

print(
    "Magnitude min:",
    gradient_magnitude.min()
)

print(
    "Magnitude max:",
    gradient_magnitude.max()
)

print(
    "Direction min:",
    gradient_direction_degrees.min()
)

print(
    "Direction max:",
    gradient_direction_degrees.max()
)


# ======================================================================
# STEP 14 — EDGE PIXEL STATISTICS
# ======================================================================

num_edges_low = np.sum(
    edges_low
)

num_edges_medium = np.sum(
    edges_medium
)

num_edges_high = np.sum(
    edges_high
)

total_pixels = image.size


print("\nEDGE STATISTICS")
print("-" * 72)

print(
    f"Low threshold ({threshold_low}): "
    f"{num_edges_low} pixels "
    f"({100 * num_edges_low / total_pixels:.2f}%)"
)

print(
    f"Medium threshold ({threshold_medium}): "
    f"{num_edges_medium} pixels "
    f"({100 * num_edges_medium / total_pixels:.2f}%)"
)

print(
    f"High threshold ({threshold_high}): "
    f"{num_edges_high} pixels "
    f"({100 * num_edges_high / total_pixels:.2f}%)"
)


# ======================================================================
# STEP 15 — NUMERICAL TESTS
# ======================================================================

print("\nRUNNING TESTS")
print("-" * 72)

# Kernel shapes
assert sobel_x.shape == (3, 3)
assert sobel_y.shape == (3, 3)

# Gradient shapes
assert gx.shape == image.shape
assert gy.shape == image.shape

# Magnitude shape
assert gradient_magnitude.shape == image.shape

# Magnitude must never be negative.
assert gradient_magnitude.min() >= 0

# Normalized magnitude must be in [0,255].
assert magnitude_normalized.min() >= 0
assert magnitude_normalized.max() <= 255 + 1e-6

# Binary edge maps.
assert edges_low.dtype == np.bool_
assert edges_medium.dtype == np.bool_
assert edges_high.dtype == np.bool_

# Lower threshold should produce at least as many edges
# as a higher threshold.
assert num_edges_low >= num_edges_medium
assert num_edges_medium >= num_edges_high

# atan2 output must remain within [-pi, pi].
assert gradient_direction.min() >= -np.pi
assert gradient_direction.max() <= np.pi

print("All tests passed.")


# ======================================================================
# STEP 16 — VISUALIZE ORIGINAL AND GRADIENTS
# ======================================================================

plt.figure(figsize=(16, 10))


plt.subplot(2, 3, 1)

plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Original Image")
plt.axis("off")


plt.subplot(2, 3, 2)

plt.imshow(
    gx,
    cmap="gray"
)

plt.title("Sobel X — Gx")
plt.axis("off")


plt.subplot(2, 3, 3)

plt.imshow(
    gy,
    cmap="gray"
)

plt.title("Sobel Y — Gy")
plt.axis("off")


plt.subplot(2, 3, 4)

plt.imshow(
    magnitude_normalized,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Gradient Magnitude")
plt.axis("off")


plt.subplot(2, 3, 5)

plt.imshow(
    gradient_direction_degrees,
    cmap="hsv",
    vmin=-180,
    vmax=180
)

plt.title("Gradient Direction")
plt.colorbar(
    fraction=0.046,
    pad=0.04
)
plt.axis("off")


plt.subplot(2, 3, 6)

plt.imshow(
    edge_map,
    cmap="gray"
)

plt.title(
    f"Binary Edge Map\nThreshold={threshold}"
)

plt.axis("off")


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 17 — COMPARE THRESHOLDS
# ======================================================================

plt.figure(figsize=(16, 5))


plt.subplot(1, 3, 1)

plt.imshow(
    edges_low,
    cmap="gray"
)

plt.title(
    f"Low Threshold = {threshold_low}"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    edges_medium,
    cmap="gray"
)

plt.title(
    f"Medium Threshold = {threshold_medium}"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    edges_high,
    cmap="gray"
)

plt.title(
    f"High Threshold = {threshold_high}"
)

plt.axis("off")


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 18 — SIMPLE VS SOBEL GRADIENTS
# ======================================================================

simple_magnitude = np.hypot(
    simple_gx,
    simple_gy
)

simple_max = simple_magnitude.max()

if simple_max > 0:

    simple_magnitude_normalized = (
        simple_magnitude
        /
        simple_max
        *
        255.0
    )

else:

    simple_magnitude_normalized = np.zeros_like(
        simple_magnitude
    )


plt.figure(figsize=(14, 5))


plt.subplot(1, 2, 1)

plt.imshow(
    simple_magnitude_normalized,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Simple Derivative Gradient")
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    magnitude_normalized,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Sobel Gradient")
plt.axis("off")


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 19 — EXAMINE A SINGLE EDGE PROFILE
# ======================================================================
#
# We now take one horizontal row across a strong object boundary.
#
# This demonstrates the connection:
#
#       intensity
#           ↓
#       derivative
#           ↓
#       edge
#
# Suppose intensity changes like:
#
#       30 30 30 30 30 220 220 220
#
# The derivative around the transition becomes large.

row = 100

intensity_profile = image[row, :]

horizontal_gradient_profile = gx[row, :]


plt.figure(figsize=(14, 8))


plt.subplot(2, 1, 1)

plt.plot(
    intensity_profile
)

plt.title(
    f"Image Intensity Profile — Row {row}"
)

plt.xlabel("X coordinate")
plt.ylabel("Intensity")


plt.subplot(2, 1, 2)

plt.plot(
    horizontal_gradient_profile
)

plt.title(
    "Horizontal Gradient Profile"
)

plt.xlabel("X coordinate")
plt.ylabel("Gradient")


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 20 — FIND STRONGEST GRADIENT
# ======================================================================

strongest_index = np.argmax(
    gradient_magnitude
)

strongest_y, strongest_x = np.unravel_index(
    strongest_index,
    gradient_magnitude.shape
)

strongest_magnitude = gradient_magnitude[
    strongest_y,
    strongest_x
]

strongest_direction = gradient_direction_degrees[
    strongest_y,
    strongest_x
]


print("\nSTRONGEST GRADIENT")
print("-" * 72)

print(
    "Location:",
    f"(x={strongest_x}, y={strongest_y})"
)

print(
    "Magnitude:",
    strongest_magnitude
)

print(
    "Direction:",
    strongest_direction,
    "degrees"
)


# ======================================================================
# STEP 21 — VISUALIZE STRONGEST GRADIENT
# ======================================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.scatter(
    [strongest_x],
    [strongest_y],
    s=100,
    marker="x"
)

plt.title(
    "Location of Strongest Gradient"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 22 — REAL-WORLD ROBOTICS INTERPRETATION
# ======================================================================
#
# Consider the OpenArm V2 waste-sorting system.
#
# A possible classical vision pipeline is:
#
#       CAMERA
#         |
#         v
#       RGB IMAGE
#         |
#         v
#       GRAYSCALE
#         |
#         v
#       GAUSSIAN FILTER
#         |
#         v
#       IMAGE GRADIENT
#         |
#         v
#       SOBEL
#         |
#         v
#       EDGE MAGNITUDE
#         |
#         v
#       THRESHOLD
#         |
#         v
#       EDGE MAP
#         |
#         v
#       CONTOURS
#         |
#         v
#       OBJECT BOUNDARIES
#         |
#         v
#       OBJECT LOCATION
#
# For example, a bottle may produce strong edges around:
#
#       top
#       bottom
#       left boundary
#       right boundary
#
# These boundaries can later be grouped into contours.
#
# This is one reason edge detection is such an important
# foundation of classical computer vision.
#
# Later, neural networks will learn filters that can detect
# much more complex visual patterns automatically.
#
# But underneath those learned filters is the same basic idea:
#
#       local image values
#               ↓
#       weighted operation
#               ↓
#       feature response


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 72)
print("LESSON 05 COMPLETE")
print("=" * 72)

print(
    """
You learned:

1. Edges correspond to significant intensity changes.
2. Image gradients measure intensity change.
3. Gx measures horizontal intensity change.
4. Gy measures vertical intensity change.
5. Gradient magnitude is:

       sqrt(Gx² + Gy²)

6. Gradient direction is:

       atan2(Gy, Gx)

7. Sobel operators combine differentiation and local smoothing.
8. Gradient magnitude can be thresholded to produce an edge map.
9. Lower thresholds generally detect more edges.
10. Higher thresholds generally detect fewer stronger edges.
11. Gradient calculations should use floating-point values.
12. Edge direction and gradient direction are perpendicular.
13. Edge detection is a fundamental bridge between raw pixels
    and higher-level object structure.

The key computer vision chain is now:

    PIXELS
      ↓
    FILTERING
      ↓
    GRADIENTS
      ↓
    EDGES
      ↓
    STRUCTURE
      ↓
    OBJECTS

NEXT:
LESSON 06 — THRESHOLDING AND BINARY IMAGE SEGMENTATION
"""
)