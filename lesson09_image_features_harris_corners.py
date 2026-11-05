"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 09 — IMAGE FEATURES AND HARRIS CORNER DETECTION
======================================================================

COURSE RULE
-----------
This course is built from scratch.

NO:
    - pretrained models
    - pretrained weights
    - cv2.cornerHarris()
    - cv2.goodFeaturesToTrack()
    - black-box feature detectors

We implement the mathematics using:

    - NumPy
    - Matplotlib

----------------------------------------------------------------------
WHAT YOU LEARN
----------------------------------------------------------------------

STEP 1 — WHAT IS AN IMAGE FEATURE?
----------------------------------------------------------------------

A feature is an image structure useful for identifying or comparing
parts of an image.

Examples:

    - edges
    - corners
    - blobs
    - texture
    - keypoints

A useful feature should ideally be:

    - distinctive
    - localizable
    - stable under small image changes

Today we focus on:

    CORNERS

----------------------------------------------------------------------
STEP 2 — WHY ARE CORNERS IMPORTANT?
----------------------------------------------------------------------

A local image region can approximately be:

    FLAT

        --------
        --------
        --------

    EDGE

        ||||||||
        ||||||||
        ||||||||

    CORNER

        ||||
        ||||
        ------
        ------

A corner contains strong intensity variation in two directions.

Corners are useful for:

    - feature matching
    - image registration
    - optical flow
    - visual odometry
    - SLAM
    - camera calibration
    - tracking
    - robot localization

----------------------------------------------------------------------
STEP 3 — IMAGE GRADIENT
----------------------------------------------------------------------

From earlier lessons:

    Ix = ∂I/∂x
    Iy = ∂I/∂y

Gradient:

    ∇I = [Ix, Iy]

Gradient magnitude:

    |∇I| = sqrt(Ix² + Iy²)

The gradient tells us how image intensity changes.

----------------------------------------------------------------------
STEP 4 — LOCAL PATCH ANALYSIS
----------------------------------------------------------------------

For a small image displacement:

    (u, v)

we use the first-order Taylor approximation:

    I(x+u,y+v)
        ≈
    I(x,y) + Ix*u + Iy*v

Therefore:

    I(x+u,y+v) - I(x,y)
        ≈
    Ix*u + Iy*v

The squared error over a local window is:

    E(u,v)
        =
    Σ w(x,y)
      [Ix*u + Iy*v]²

This becomes:

    E(u,v)
        =
    [u v] M [u v]^T

where M is the structure tensor.

----------------------------------------------------------------------
STEP 5 — STRUCTURE TENSOR
----------------------------------------------------------------------

The structure tensor is:

             [ Σ Ix²      Σ IxIy ]
    M   =    [                ]
             [ Σ IxIy      Σ Iy²  ]

Using a local weighting function:

             [ Σ w Ix²       Σ w IxIy ]
    M   =    [                       ]
             [ Σ w IxIy      Σ w Iy²  ]

The local sums describe image variation around each pixel.

----------------------------------------------------------------------
STEP 6 — EIGENVALUES
----------------------------------------------------------------------

Let the eigenvalues of M be:

    λ1
    λ2

They describe local intensity variation in two principal
directions.

FLAT:

    λ1 ≈ 0
    λ2 ≈ 0

EDGE:

    λ1 >> 0
    λ2 ≈ 0

CORNER:

    λ1 >> 0
    λ2 >> 0

----------------------------------------------------------------------
STEP 7 — HARRIS CORNER RESPONSE
----------------------------------------------------------------------

Harris avoids explicitly computing eigenvalues at every pixel.

The response is:

    R = det(M) - k(trace(M))²

where:

    det(M) = λ1 λ2

and:

    trace(M) = λ1 + λ2

Typical:

    k ≈ 0.04

Interpretation:

    large positive R
        → corner

    negative R
        → edge

    R near zero
        → flat region

----------------------------------------------------------------------
STEP 8 — NON-MAXIMUM SUPPRESSION
----------------------------------------------------------------------

Harris produces a response at every pixel.

A single physical corner can produce several strong neighboring
pixels.

Non-maximum suppression keeps only the strongest response inside
a local neighborhood.

Pipeline:

    Harris response
          ↓
    threshold
          ↓
    non-maximum suppression
          ↓
    sparse keypoints

----------------------------------------------------------------------
STEP 9 — REAL-WORLD ROBOTICS EXAMPLE
----------------------------------------------------------------------

For an OpenArm V2 waste-sorting system:

    camera
       ↓
    image
       ↓
    keypoint detection
       ↓
    feature matching
       ↓
    object/camera motion estimation
       ↓
    geometric transformation
       ↓
    robot workspace coordinates
       ↓
    grasping

Harris corners can be useful for:

    - box corners
    - conveyor landmarks
    - visual tracking
    - camera motion estimation
    - visual odometry
    - localization

However:

    Harris corner detector
        !=
    object detector

Irregular waste may contain few strong corners.

----------------------------------------------------------------------
STEP 10 — IMPORTANT IMPLEMENTATION DETAIL
----------------------------------------------------------------------

Our convolution function uses ZERO PADDING.

That is useful for preserving image size, but it creates an
important boundary effect:

    constant image
        +
    zero padding
        =
    artificial boundary gradient

Therefore, when testing whether a flat image has zero gradient,
we must test the INTERIOR of the image rather than the boundary.

This is an important computer-vision numerical lesson:

    padding strategy affects the result.

----------------------------------------------------------------------
STEP 11 — FROM-SCRATCH IMPLEMENTATION
----------------------------------------------------------------------

We implement:

    1. synthetic image
    2. convolution
    3. image gradients
    4. local structure tensor
    5. determinant
    6. trace
    7. Harris response
    8. thresholding
    9. non-maximum suppression
   10. eigenvalue analysis
   11. visualization
   12. numerical tests

----------------------------------------------------------------------
EXERCISES
----------------------------------------------------------------------

1. Change Harris k:

       0.02
       0.04
       0.06
       0.10

2. Change the local window size.

3. Add Gaussian noise.

4. Compare corner counts before and after noise.

5. Rotate the image.

6. Scale the image.

7. Build a chessboard.

8. Build a cross-shaped pattern.

9. Compare Harris with a simple gradient detector.

10. Use the detected points for feature matching.

----------------------------------------------------------------------
NEXT LESSON
----------------------------------------------------------------------

LESSON 10 — FEATURE DESCRIPTORS AND IMAGE MATCHING

We will build:

    - local image patches
    - normalized descriptors
    - descriptor distance
    - SSD matching
    - normalized cross correlation
    - feature matching
    - outlier rejection

======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# STEP 1 — CREATE SYNTHETIC IMAGE
# ======================================================================

HEIGHT = 320
WIDTH = 480

image = np.zeros(
    (HEIGHT, WIDTH),
    dtype=np.float32
)

image[:] = 30.0


# Large rectangle
image[
    50:180,
    60:220
] = 210.0


# Small square
image[
    210:280,
    80:150
] = 180.0


# Second rectangle
image[
    70:220,
    280:410
] = 200.0


# Thin horizontal structure
image[
    240:255,
    200:390
] = 230.0


# Circle
yy, xx = np.ogrid[
    :HEIGHT,
    :WIDTH
]

circle = (
    (xx - 390) ** 2
    +
    (yy - 270) ** 2
    <= 35 ** 2
)

image[circle] = 160.0


# Diagonal line
for y in range(30, 180):

    x = int(
        230 + 0.6 * y
    )

    if 0 <= x < WIDTH:

        image[
            y,
            x
        ] = 255.0


# ======================================================================
# STEP 2 — FROM-SCRATCH CONVOLUTION
# ======================================================================

def zero_pad(
    image,
    kernel_shape
):
    """
    Zero-pad an image.

    IMPORTANT:
    This intentionally uses zero padding because it makes the
    convolution implementation easy to understand.

    Boundary pixels can therefore contain artificial gradients.
    """

    kh, kw = kernel_shape

    pad_y = kh // 2
    pad_x = kw // 2

    return np.pad(
        image,
        (
            (pad_y, pad_y),
            (pad_x, pad_x)
        ),
        mode="constant",
        constant_values=0.0
    )


def convolve2d(
    image,
    kernel
):
    """
    Basic 2-D convolution from scratch.

    Mathematical convolution flips the kernel before applying it.
    """

    image = image.astype(
        np.float32
    )

    kernel = kernel.astype(
        np.float32
    )

    flipped = np.flip(
        kernel
    )

    padded = zero_pad(
        image,
        kernel.shape
    )

    height, width = image.shape

    output = np.zeros(
        image.shape,
        dtype=np.float32
    )

    kh, kw = kernel.shape

    for y in range(height):

        for x in range(width):

            region = padded[
                y:y + kh,
                x:x + kw
            ]

            output[y, x] = np.sum(
                region * flipped
            )

    return output


# ======================================================================
# STEP 3 — SOBEL IMAGE GRADIENT
# ======================================================================

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


Ix = convolve2d(
    image,
    sobel_x
)

Iy = convolve2d(
    image,
    sobel_y
)


gradient_magnitude = np.sqrt(
    Ix ** 2
    +
    Iy ** 2
)


gradient_direction = np.arctan2(
    Iy,
    Ix
)


# ======================================================================
# STEP 4 — LOCAL BOX FILTER
# ======================================================================

def box_filter(
    image,
    window_size
):
    """
    Local mean filter implemented from scratch.

    This computes:

        mean = sum(local pixels) / N

    It is used here to approximate local structure-tensor sums.
    """

    assert window_size % 2 == 1

    kernel = np.ones(
        (
            window_size,
            window_size
        ),
        dtype=np.float32
    )

    kernel /= (
        window_size
        *
        window_size
    )

    return convolve2d(
        image,
        kernel
    )


# ======================================================================
# STEP 5 — STRUCTURE TENSOR
# ======================================================================

Ix2 = Ix * Ix
Iy2 = Iy * Iy
IxIy = Ix * Iy

window_size = 7

Sxx = box_filter(
    Ix2,
    window_size
)

Syy = box_filter(
    Iy2,
    window_size
)

Sxy = box_filter(
    IxIy,
    window_size
)


# ======================================================================
# STEP 6 — DETERMINANT AND TRACE
# ======================================================================

det_M = (
    Sxx * Syy
    -
    Sxy * Sxy
)

trace_M = (
    Sxx
    +
    Syy
)


# ======================================================================
# STEP 7 — HARRIS RESPONSE
# ======================================================================

k = 0.04

harris_response = (
    det_M
    -
    k * trace_M ** 2
)


# ======================================================================
# STEP 8 — NORMALIZE RESPONSE
# ======================================================================

max_abs_response = np.max(
    np.abs(
        harris_response
    )
)

if max_abs_response > 0:

    normalized_response = (
        harris_response
        /
        max_abs_response
    )

else:

    normalized_response = (
        np.zeros_like(
            harris_response
        )
    )


# ======================================================================
# STEP 9 — PRINT IMAGE STATISTICS
# ======================================================================

print("=" * 72)
print("LESSON 09 — IMAGE FEATURES AND HARRIS CORNERS")
print("=" * 72)

print("\nIMAGE")
print("-" * 72)

print(
    "Shape:",
    image.shape
)

print(
    "Minimum:",
    image.min()
)

print(
    "Maximum:",
    image.max()
)

print(
    "Mean:",
    image.mean()
)

print(
    "Standard deviation:",
    image.std()
)


print("\nGRADIENT")
print("-" * 72)

print(
    "Ix min:",
    Ix.min()
)

print(
    "Ix max:",
    Ix.max()
)

print(
    "Iy min:",
    Iy.min()
)

print(
    "Iy max:",
    Iy.max()
)

print(
    "Gradient magnitude max:",
    gradient_magnitude.max()
)


print("\nHARRIS RESPONSE")
print("-" * 72)

print(
    "Minimum:",
    harris_response.min()
)

print(
    "Maximum:",
    harris_response.max()
)

print(
    "Mean:",
    harris_response.mean()
)


# ======================================================================
# STEP 10 — THRESHOLD CORNER RESPONSE
# ======================================================================

relative_threshold = 0.05

corner_candidate_mask = (
    normalized_response
    >
    relative_threshold
)


# ======================================================================
# STEP 11 — NON-MAXIMUM SUPPRESSION
# ======================================================================

def non_maximum_suppression(
    response,
    candidate_mask,
    radius
):
    """
    Keep candidate pixels that are local maxima.

    A candidate is retained when its response is equal to the
    maximum response in its local neighborhood.

    The >= comparison intentionally permits plateaus. This is useful
    for a simple from-scratch implementation and is tested below.
    """

    height, width = response.shape

    output = np.zeros(
        response.shape,
        dtype=bool
    )

    for y in range(height):

        for x in range(width):

            if not candidate_mask[y, x]:
                continue

            y0 = max(
                0,
                y - radius
            )

            y1 = min(
                height,
                y + radius + 1
            )

            x0 = max(
                0,
                x - radius
            )

            x1 = min(
                width,
                x + radius + 1
            )

            local_region = response[
                y0:y1,
                x0:x1
            ]

            local_maximum = np.max(
                local_region
            )

            if (
                response[y, x]
                >= local_maximum
            ):

                output[y, x] = True

    return output


corner_mask = non_maximum_suppression(
    harris_response,
    corner_candidate_mask,
    radius=6
)


# ======================================================================
# STEP 12 — EXTRACT CORNER COORDINATES
# ======================================================================

corner_y, corner_x = np.where(
    corner_mask
)

corner_scores = harris_response[
    corner_y,
    corner_x
]


if len(corner_scores) > 0:

    order = np.argsort(
        corner_scores
    )[::-1]

    corner_x = corner_x[
        order
    ]

    corner_y = corner_y[
        order
    ]

    corner_scores = corner_scores[
        order
    ]


print("\nDETECTED CORNERS")
print("-" * 72)

print(
    "Number of corners:",
    len(corner_x)
)

print(
    "\nTop corner coordinates:"
)

for i in range(
    min(20, len(corner_x))
):

    print(
        f"{i + 1:2d}: "
        f"x={corner_x[i]:4d}, "
        f"y={corner_y[i]:4d}, "
        f"R={corner_scores[i]:.3e}"
    )


# ======================================================================
# STEP 13 — EIGENVALUE ANALYSIS
# ======================================================================

def structure_tensor_eigenvalues(
    Sxx_value,
    Sxy_value,
    Syy_value
):
    """
    Compute eigenvalues of:

        [Sxx Sxy]
        [Sxy Syy]

    For a 2x2 symmetric matrix:

        trace = Sxx + Syy

        determinant = Sxx*Syy - Sxy²

    Eigenvalues:

        λ1,2 =
            (trace ± sqrt(trace² - 4*determinant)) / 2
    """

    trace = (
        Sxx_value
        +
        Syy_value
    )

    determinant = (
        Sxx_value
        *
        Syy_value
        -
        Sxy_value ** 2
    )

    discriminant = (
        trace ** 2
        -
        4.0 * determinant
    )

    # Numerical round-off can make an expression that should be zero
    # slightly negative.
    discriminant = max(
        discriminant,
        0.0
    )

    sqrt_discriminant = np.sqrt(
        discriminant
    )

    lambda1 = (
        trace
        +
        sqrt_discriminant
    ) / 2.0

    lambda2 = (
        trace
        -
        sqrt_discriminant
    ) / 2.0

    return (
        lambda1,
        lambda2
    )


print("\nEIGENVALUE INTERPRETATION")
print("-" * 72)

if len(corner_x) > 0:

    for i in range(
        min(5, len(corner_x))
    ):

        x = corner_x[i]
        y = corner_y[i]

        lambda1, lambda2 = (
            structure_tensor_eigenvalues(
                Sxx[y, x],
                Sxy[y, x],
                Syy[y, x]
            )
        )

        print(
            f"Corner {i + 1}: "
            f"lambda1={lambda1:.3e}, "
            f"lambda2={lambda2:.3e}"
        )


# ======================================================================
# STEP 14 — VISUALIZE ORIGINAL IMAGE
# ======================================================================

plt.figure(
    figsize=(10, 7)
)

plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Synthetic Image"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 15 — VISUALIZE GRADIENTS
# ======================================================================

plt.figure(
    figsize=(15, 5)
)

plt.subplot(1, 3, 1)

plt.imshow(
    Ix,
    cmap="gray"
)

plt.title(
    "Ix — Horizontal Gradient"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    Iy,
    cmap="gray"
)

plt.title(
    "Iy — Vertical Gradient"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    gradient_magnitude,
    cmap="gray"
)

plt.title(
    "Gradient Magnitude"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 16 — STRUCTURE TENSOR COMPONENTS
# ======================================================================

plt.figure(
    figsize=(15, 5)
)

plt.subplot(1, 3, 1)

plt.imshow(
    Sxx,
    cmap="gray"
)

plt.title(
    "Sxx = Local Ix²"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    Syy,
    cmap="gray"
)

plt.title(
    "Syy = Local Iy²"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    Sxy,
    cmap="gray"
)

plt.title(
    "Sxy = Local IxIy"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 17 — HARRIS RESPONSE
# ======================================================================

plt.figure(
    figsize=(15, 5)
)

plt.subplot(1, 3, 1)

plt.imshow(
    harris_response,
    cmap="coolwarm"
)

plt.colorbar(
    fraction=0.046,
    pad=0.04
)

plt.title(
    "Harris Response R"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    normalized_response,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.colorbar(
    fraction=0.046,
    pad=0.04
)

plt.title(
    "Normalized Harris Response"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    corner_candidate_mask,
    cmap="gray"
)

plt.title(
    "Corner Candidates"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 18 — FINAL CORNERS
# ======================================================================

plt.figure(
    figsize=(11, 7)
)

plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.scatter(
    corner_x,
    corner_y,
    s=50,
    facecolors="none",
    edgecolors="red"
)

plt.title(
    f"Harris Corners — {len(corner_x)} detected"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 19 — TOP CORNERS
# ======================================================================

top_k = min(
    30,
    len(corner_x)
)

plt.figure(
    figsize=(11, 7)
)

plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)

if top_k > 0:

    plt.scatter(
        corner_x[:top_k],
        corner_y[:top_k],
        s=80,
        facecolors="none",
        edgecolors="red"
    )

    for i in range(
        top_k
    ):

        plt.text(
            corner_x[i] + 4,
            corner_y[i] + 4,
            str(i + 1)
        )

plt.title(
    f"Top {top_k} Harris Corners"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 20 — CORNER SCORE DISTRIBUTION
# ======================================================================

if len(corner_scores) > 0:

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        np.arange(
            len(corner_scores)
        ),
        corner_scores
    )

    plt.xlabel(
        "Corner Rank"
    )

    plt.ylabel(
        "Harris Response"
    )

    plt.title(
        "Detected Corner Response Strength"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ======================================================================
# STEP 21 — IDEAL FLAT / EDGE / CORNER PATCHES
# ======================================================================

flat_patch = np.ones(
    (21, 21),
    dtype=np.float32
) * 100.0


edge_patch = np.zeros(
    (21, 21),
    dtype=np.float32
)

edge_patch[
    :,
    10:
] = 255.0


corner_patch = np.zeros(
    (21, 21),
    dtype=np.float32
)

corner_patch[
    10:,
    10:
] = 255.0


# ======================================================================
# STEP 22 — ANALYZE IDEAL PATCH
# ======================================================================

def analyze_patch(
    patch,
    name
):
    """
    Analyze a small synthetic patch.

    We use the center pixel for the final structure tensor.

    This avoids boundary artifacts introduced by zero padding.
    """

    gx = convolve2d(
        patch,
        sobel_x
    )

    gy = convolve2d(
        patch,
        sobel_y
    )

    gx2 = gx * gx
    gy2 = gy * gy
    gxgy = gx * gy

    sxx = box_filter(
        gx2,
        5
    )

    syy = box_filter(
        gy2,
        5
    )

    sxy = box_filter(
        gxgy,
        5
    )

    center_y = patch.shape[0] // 2
    center_x = patch.shape[1] // 2

    lambda1, lambda2 = (
        structure_tensor_eigenvalues(
            sxx[center_y, center_x],
            sxy[center_y, center_x],
            syy[center_y, center_x]
        )
    )

    det = (
        sxx[center_y, center_x]
        *
        syy[center_y, center_x]
        -
        sxy[center_y, center_x] ** 2
    )

    trace = (
        sxx[center_y, center_x]
        +
        syy[center_y, center_x]
    )

    R = (
        det
        -
        k * trace ** 2
    )

    print(
        f"{name:10s} | "
        f"lambda1={lambda1:12.3f} | "
        f"lambda2={lambda2:12.3f} | "
        f"R={R:15.3f}"
    )

    return (
        lambda1,
        lambda2,
        R
    )


print("\nFLAT / EDGE / CORNER ANALYSIS")
print("-" * 72)

print(
    "Region     | lambda1       | lambda2       | Harris R"
)

flat_values = analyze_patch(
    flat_patch,
    "Flat"
)

edge_values = analyze_patch(
    edge_patch,
    "Edge"
)

corner_values = analyze_patch(
    corner_patch,
    "Corner"
)


# ======================================================================
# STEP 23 — VISUALIZE IDEAL STRUCTURES
# ======================================================================

plt.figure(
    figsize=(15, 5)
)

plt.subplot(1, 3, 1)

plt.imshow(
    flat_patch,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Flat Region"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    edge_patch,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Edge"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    corner_patch,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Corner"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 24 — NUMERICAL TESTS
# ======================================================================

print("\nRUNNING TESTS")
print("-" * 72)


# ----------------------------------------------------------------------
# Test 1 — image dimensions
# ----------------------------------------------------------------------

assert image.shape == (
    HEIGHT,
    WIDTH
)

print(
    "Image shape test passed."
)


# ----------------------------------------------------------------------
# Test 2 — gradient dimensions
# ----------------------------------------------------------------------

assert Ix.shape == image.shape
assert Iy.shape == image.shape

print(
    "Gradient shape test passed."
)


# ----------------------------------------------------------------------
# Test 3 — structure tensor dimensions
# ----------------------------------------------------------------------

assert Sxx.shape == image.shape
assert Syy.shape == image.shape
assert Sxy.shape == image.shape

print(
    "Structure tensor shape test passed."
)


# ----------------------------------------------------------------------
# Test 4 — Harris response dimensions
# ----------------------------------------------------------------------

assert (
    harris_response.shape
    ==
    image.shape
)

print(
    "Harris response shape test passed."
)


# ----------------------------------------------------------------------
# Test 5 — normalized response range
# ----------------------------------------------------------------------

assert (
    normalized_response.min()
    >= -1.000001
)

assert (
    normalized_response.max()
    <= 1.000001
)

print(
    "Normalized response range test passed."
)


# ----------------------------------------------------------------------
# Test 6 — determinant relationship
# ----------------------------------------------------------------------

random_y = 100
random_x = 150

expected_det = (
    Sxx[random_y, random_x]
    *
    Syy[random_y, random_x]
    -
    Sxy[random_y, random_x] ** 2
)

assert np.isclose(
    det_M[random_y, random_x],
    expected_det,
    rtol=1e-6,
    atol=1e-6
)

print(
    "Structure tensor determinant test passed."
)


# ----------------------------------------------------------------------
# Test 7 — trace relationship
# ----------------------------------------------------------------------

expected_trace = (
    Sxx[random_y, random_x]
    +
    Syy[random_y, random_x]
)

assert np.isclose(
    trace_M[random_y, random_x],
    expected_trace,
    rtol=1e-6,
    atol=1e-6
)

print(
    "Structure tensor trace test passed."
)


# ----------------------------------------------------------------------
# Test 8 — Harris equation
# ----------------------------------------------------------------------

expected_R = (
    det_M[random_y, random_x]
    -
    k
    *
    trace_M[random_y, random_x] ** 2
)

assert np.isclose(
    harris_response[random_y, random_x],
    expected_R,
    rtol=1e-6,
    atol=1e-6
)

print(
    "Harris equation test passed."
)


# ----------------------------------------------------------------------
# Test 9 — eigenvalue sum equals trace
# ----------------------------------------------------------------------

test_x = 100
test_y = 100

lambda1, lambda2 = (
    structure_tensor_eigenvalues(
        Sxx[test_y, test_x],
        Sxy[test_y, test_x],
        Syy[test_y, test_x]
    )
)

assert np.isclose(
    lambda1 + lambda2,
    trace_M[test_y, test_x],
    rtol=1e-5,
    atol=1e-5
)

print(
    "Eigenvalue-trace test passed."
)


# ----------------------------------------------------------------------
# Test 10 — eigenvalue product equals determinant
# ----------------------------------------------------------------------

assert np.isclose(
    lambda1 * lambda2,
    det_M[test_y, test_x],
    rtol=1e-5,
    atol=1e-5
)

print(
    "Eigenvalue-determinant test passed."
)


# ----------------------------------------------------------------------
# Test 11 — NMS output
# ----------------------------------------------------------------------

assert (
    corner_mask.dtype
    ==
    np.bool_
)

assert (
    corner_mask.shape
    ==
    image.shape
)

print(
    "Non-maximum suppression test passed."
)


# ----------------------------------------------------------------------
# Test 12 — corner coordinates
# ----------------------------------------------------------------------

assert np.all(
    corner_x >= 0
)

assert np.all(
    corner_x < WIDTH
)

assert np.all(
    corner_y >= 0
)

assert np.all(
    corner_y < HEIGHT
)

print(
    "Corner-coordinate bounds test passed."
)


# ----------------------------------------------------------------------
# Test 13 — detected corners satisfy threshold
# ----------------------------------------------------------------------

if len(corner_x) > 0:

    detected_scores = normalized_response[
        corner_y,
        corner_x
    ]

    assert np.all(
        detected_scores
        >
        relative_threshold
        -
        1e-12
    )

print(
    "Corner-threshold test passed."
)


# ----------------------------------------------------------------------
# Test 14 — flat region interior has zero gradient
# ----------------------------------------------------------------------
#
# IMPORTANT FIX
# -------------
#
# The previous version tested the entire flat image:
#
#     np.max(abs(flat_gradient))
#
# That is incorrect with ZERO PADDING.
#
# At the image boundary:
#
#     100 100 100 | 0 0
#
# creates an artificial intensity transition.
#
# Therefore we test only the interior where the convolution kernel
# sees only the constant 100-valued region.

flat_gradient = convolve2d(
    flat_patch,
    sobel_x
)

flat_interior = flat_gradient[
    2:-2,
    2:-2
]

assert np.allclose(
    flat_interior,
    0.0,
    rtol=0.0,
    atol=1e-6
)

print(
    "Flat-region interior gradient test passed."
)


# ----------------------------------------------------------------------
# Test 15 — flat patch has approximately zero eigenvalues
# ----------------------------------------------------------------------

flat_lambda1 = flat_values[0]
flat_lambda2 = flat_values[1]

assert np.isclose(
    flat_lambda1,
    0.0,
    atol=1e-6
)

assert np.isclose(
    flat_lambda2,
    0.0,
    atol=1e-6
)

print(
    "Flat-region eigenvalue test passed."
)


# ----------------------------------------------------------------------
# Test 16 — edge should have one significant eigenvalue
# ----------------------------------------------------------------------

edge_lambda1 = edge_values[0]
edge_lambda2 = edge_values[1]

assert (
    edge_lambda1 > 0
)

assert (
    edge_lambda2 >= -1e-6
)

# For the ideal edge, one eigenvalue should dominate the other.
assert (
    edge_lambda1
    >
    100.0 * max(
        edge_lambda2,
        1e-12
    )
)

print(
    "Edge eigenvalue test passed."
)


# ----------------------------------------------------------------------
# Test 17 — corner should have two positive eigenvalues
# ----------------------------------------------------------------------

corner_lambda1 = corner_values[0]
corner_lambda2 = corner_values[1]

assert (
    corner_lambda1 > 0
)

assert (
    corner_lambda2 > 0
)

print(
    "Corner eigenvalue test passed."
)


# ----------------------------------------------------------------------
# Test 18 — corner should have positive Harris response
# ----------------------------------------------------------------------

corner_R = corner_values[2]

assert (
    corner_R > 0
)

print(
    "Corner Harris-response test passed."
)


# ----------------------------------------------------------------------
# Test 19 — edge should have negative Harris response
# ----------------------------------------------------------------------

edge_R = edge_values[2]

assert (
    edge_R < 0
)

print(
    "Edge Harris-response test passed."
)


# ----------------------------------------------------------------------
# Test 20 — flat Harris response should be approximately zero
# ----------------------------------------------------------------------

flat_R = flat_values[2]

assert np.isclose(
    flat_R,
    0.0,
    atol=1e-6
)

print(
    "Flat Harris-response test passed."
)


print("\nALL TESTS PASSED.")


# ======================================================================
# STEP 25 — EFFECT OF HARRIS K
# ======================================================================

k_values = [
    0.02,
    0.04,
    0.06,
    0.10
]

plt.figure(
    figsize=(16, 10)
)

for i, k_test in enumerate(
    k_values
):

    R_test = (
        det_M
        -
        k_test
        *
        trace_M ** 2
    )

    max_abs = np.max(
        np.abs(
            R_test
        )
    )

    if max_abs > 0:

        R_norm = (
            R_test
            /
            max_abs
        )

    else:

        R_norm = np.zeros_like(
            R_test
        )

    plt.subplot(
        2,
        2,
        i + 1
    )

    plt.imshow(
        R_norm,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.title(
        f"Normalized Harris Response, k={k_test}"
    )

    plt.axis("off")


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 26 — EFFECT OF LOCAL WINDOW SIZE
# ======================================================================

window_sizes = [
    3,
    5,
    7,
    11
]

plt.figure(
    figsize=(16, 10)
)

for i, ws in enumerate(
    window_sizes
):

    local_sxx = box_filter(
        Ix2,
        ws
    )

    local_syy = box_filter(
        Iy2,
        ws
    )

    local_sxy = box_filter(
        IxIy,
        ws
    )

    local_det = (
        local_sxx
        *
        local_syy
        -
        local_sxy ** 2
    )

    local_trace = (
        local_sxx
        +
        local_syy
    )

    local_R = (
        local_det
        -
        k
        *
        local_trace ** 2
    )

    max_abs = np.max(
        np.abs(
            local_R
        )
    )

    if max_abs > 0:

        local_R = (
            local_R
            /
            max_abs
        )

    plt.subplot(
        2,
        2,
        i + 1
    )

    plt.imshow(
        local_R,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.title(
        f"Window Size = {ws}×{ws}"
    )

    plt.axis("off")


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 27 — ROBOTICS INTERPRETATION
# ======================================================================

print(
    """
======================================================================
ROBOTICS INTERPRETATION
======================================================================

Classical vision pipeline:

    CAMERA
       ↓
    IMAGE
       ↓
    GRADIENT
       ↓
    LOCAL STRUCTURE
       ↓
    HARRIS CORNERS
       ↓
    FEATURE POINTS
       ↓
    FEATURE MATCHING
       ↓
    MOTION ESTIMATION
       ↓
    ROBOT VISION

For OpenArm V2 waste sorting:

    camera
       ↓
    visual landmarks
       ↓
    object/camera movement
       ↓
    geometric transformation
       ↓
    workspace coordinates
       ↓
    robot control

Harris corners are especially useful for geometric structures such
as:

    - boxes
    - containers
    - conveyor landmarks
    - calibration targets
    - textured objects

For irregular waste, Harris is only one component of the complete
perception system.

A later system can combine:

    classical features
          +
    segmentation
          +
    object detection
          +
    depth
          +
    learned visual representations
          +
    robot control
"""
)


# ======================================================================
# STEP 28 — FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 72)
print("LESSON 09 COMPLETE")
print("=" * 72)

print(
    """
You learned:

1. Image features provide useful local visual information.

2. Corners contain intensity variation in two directions.

3. Image gradients describe local intensity changes.

4. The structure tensor summarizes local gradient structure.

5. Eigenvalues describe variation in principal directions.

6. Flat regions have two small eigenvalues.

7. Edges have one strong and one weak eigenvalue.

8. Corners have two strong eigenvalues.

9. Harris response is:

       R = det(M) - k(trace(M))²

10. Strong positive R indicates likely corners.

11. Negative R commonly corresponds to edges.

12. Non-maximum suppression converts a response map into sparse
    keypoints.

13. Padding strategy matters when implementing convolution.

14. Harris corners are local features, not complete object detectors.

The feature pipeline is:

    IMAGE
      ↓
    GRADIENT
      ↓
    STRUCTURE TENSOR
      ↓
    HARRIS RESPONSE
      ↓
    THRESHOLD
      ↓
    NON-MAXIMUM SUPPRESSION
      ↓
    KEYPOINTS

For robotics:

    CAMERA
      ↓
    KEYPOINTS
      ↓
    FEATURE MATCHING
      ↓
    MOTION ESTIMATION
      ↓
    ROBOT VISION

NEXT:
LESSON 10 — FEATURE DESCRIPTORS AND IMAGE MATCHING

We will use Harris points as the starting point and build local
descriptors and feature matching completely from scratch.
"""
)