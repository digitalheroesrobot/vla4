"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 07 — MORPHOLOGICAL IMAGE PROCESSING
======================================================================

COURSE RULE
-----------
This course is built from scratch.

NO:
    - pretrained models
    - pretrained weights
    - black-box morphology functions

We implement the main morphological operations ourselves using:

    - NumPy
    - Matplotlib

----------------------------------------------------------------------
WHAT YOU LEARN
----------------------------------------------------------------------

STEP 1 — WHY MORPHOLOGY?

After thresholding, a binary mask may contain:

    - isolated noise
    - small holes
    - broken regions
    - unwanted small objects
    - rough boundaries

Typical pipeline:

    image
      ↓
    threshold
      ↓
    binary mask
      ↓
    morphology
      ↓
    cleaner mask

----------------------------------------------------------------------
STEP 2 — BINARY IMAGE
----------------------------------------------------------------------

A binary image contains:

    0 = background
    1 = foreground

----------------------------------------------------------------------
STEP 3 — STRUCTURING ELEMENT
----------------------------------------------------------------------

Morphological operations use a small binary pattern called a:

    STRUCTURING ELEMENT

Example 3×3 square:

    1 1 1
    1 1 1
    1 1 1

----------------------------------------------------------------------
STEP 4 — EROSION
----------------------------------------------------------------------

Notation:

    A ⊖ B

A pixel remains foreground only when every active position of the
structuring element overlaps foreground.

Erosion generally:

    - shrinks objects
    - removes small objects
    - removes thin connections

----------------------------------------------------------------------
STEP 5 — DILATION
----------------------------------------------------------------------

Notation:

    A ⊕ B

A pixel becomes foreground if at least one active structuring-element
position overlaps foreground.

Dilation generally:

    - expands objects
    - fills small gaps
    - connects nearby regions

----------------------------------------------------------------------
STEP 6 — OPENING
----------------------------------------------------------------------

Opening:

    A ○ B = (A ⊖ B) ⊕ B

That is:

    erosion → dilation

Opening is useful for:

    - removing small isolated noise
    - removing small unwanted structures
    - smoothing boundaries

----------------------------------------------------------------------
STEP 7 — CLOSING
----------------------------------------------------------------------

Closing:

    A • B = (A ⊕ B) ⊖ B

That is:

    dilation → erosion

Closing is useful for:

    - filling small holes
    - closing small gaps
    - connecting nearby regions
    - smoothing boundaries

IMPORTANT BOUNDARY NOTE
-----------------------

Mathematically, closing has the property:

    A ⊆ Closing(A)

However, our implementation works on a finite image and uses:

    ZERO PADDING

outside the image.

Therefore, if a foreground object touches the image boundary,
the dilation/erosion sequence can interact with the artificial
zero-valued boundary.

This means:

    A ⊆ Closing(A)

is not necessarily true at the outermost image pixels.

This is a boundary-condition issue, not a failure of morphology.

Therefore our test checks the property on an interior region where
the structuring element has sufficient image support.

----------------------------------------------------------------------
STEP 8 — MORPHOLOGICAL GRADIENT
----------------------------------------------------------------------

A common definition is:

    Gradient = Dilation(A) - Erosion(A)

For binary images, this highlights boundaries.

----------------------------------------------------------------------
STEP 9 — BOUNDARY EXTRACTION
----------------------------------------------------------------------

Internal boundary:

    Boundary = A - Erosion(A)

----------------------------------------------------------------------
STEP 10 — WHITE TOP-HAT
----------------------------------------------------------------------

    WhiteTopHat = A - Opening(A)

Useful for highlighting small bright structures.

----------------------------------------------------------------------
STEP 11 — BLACK TOP-HAT
----------------------------------------------------------------------

    BlackTopHat = Closing(A) - A

Useful for highlighting small dark holes or structures.

----------------------------------------------------------------------
STEP 12 — REAL-WORLD OPENARM V2 EXAMPLE
----------------------------------------------------------------------

Consider:

    OpenArm V2
        +
    camera
        +
    waste sorting

A thresholded image may contain:

    - isolated noise
    - small holes
    - broken object regions
    - reflection artifacts
    - small disconnected foreground pixels

A classical pipeline can be:

    camera
       ↓
    RGB image
       ↓
    grayscale
       ↓
    threshold
       ↓
    binary mask
       ↓
    opening
       ↓
    remove small noise
       ↓
    closing
       ↓
    fill small gaps
       ↓
    clean object mask
       ↓
    connected components
       ↓
    object center / bounding box
       ↓
    grasp planning

This is one of the classical computer-vision pipelines that can
be used before a robot policy such as ACT.

----------------------------------------------------------------------
STEP 13 — KERNEL SIZE
----------------------------------------------------------------------

Kernel size controls the spatial scale of morphology.

Small kernel:

    3×3

produces relatively weak changes.

Large kernel:

    15×15

produces much stronger changes.

Therefore:

    structuring-element size
            =
    spatial scale of morphology

----------------------------------------------------------------------
STEP 14 — STRUCTURING ELEMENT SHAPE
----------------------------------------------------------------------

Possible shapes include:

    square
    cross
    horizontal line
    vertical line
    disk

The shape determines which geometric structures are preserved.

----------------------------------------------------------------------
STEP 15 — FROM-SCRATCH IMPLEMENTATION
----------------------------------------------------------------------

We do NOT use:

    cv2.erode()
    cv2.dilate()
    cv2.morphologyEx()

Instead, we explicitly inspect the local neighborhood.

EROSION:

    all active kernel positions must overlap foreground.

DILATION:

    at least one active kernel position must overlap foreground.

----------------------------------------------------------------------
STEP 16 — TESTING
----------------------------------------------------------------------

We test:

    - output shape
    - output type
    - binary property
    - erosion subset property
    - dilation superset property
    - opening subset property
    - closing extensive property away from boundaries
    - small-object removal
    - single-pixel dilation

----------------------------------------------------------------------
EXERCISES
----------------------------------------------------------------------

1. Compare 3×3, 5×5, 9×9 and 15×15 kernels.

2. Compare square and cross structuring elements.

3. Create a small isolated noise region.

4. Create a hole inside an object.

5. Create two objects connected by a thin bridge.

6. Implement a disk structuring element.

7. Compare opening and closing on a real or synthetic waste mask.

----------------------------------------------------------------------
NEXT LESSON
----------------------------------------------------------------------

LESSON 08 — GEOMETRIC IMAGE TRANSFORMATIONS

Topics:

    - coordinate systems
    - translation
    - rotation
    - scaling
    - affine transformation
    - interpolation
    - image warping
    - perspective transformation

======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# STEP 1 — CREATE CLEAN BINARY OBJECT
# ======================================================================

HEIGHT = 256
WIDTH = 384

yy, xx = np.ogrid[:HEIGHT, :WIDTH]

clean_mask = np.zeros(
    (HEIGHT, WIDTH),
    dtype=bool
)

# Large rectangular object
rectangle = (
    (xx >= 80)
    &
    (xx < 250)
    &
    (yy >= 60)
    &
    (yy < 190)
)

clean_mask[rectangle] = True

# Circular object
circle = (
    (xx - 310) ** 2
    +
    (yy - 125) ** 2
    <= 38 ** 2
)

clean_mask[circle] = True


# ======================================================================
# STEP 2 — ADD SMALL NOISE
# ======================================================================

rng = np.random.default_rng(42)

noisy_mask = clean_mask.copy()

noise_probability = 0.002

random_noise = (
    rng.random(noisy_mask.shape)
    < noise_probability
)

noisy_mask |= random_noise


# ======================================================================
# STEP 3 — ADD SMALL HOLES
# ======================================================================

hole_centers = [
    (100, 130),
    (140, 170),
    (165, 210),
    (120, 300),
]

for cy, cx in hole_centers:

    hole = (
        (xx - cx) ** 2
        +
        (yy - cy) ** 2
        <= 4 ** 2
    )

    noisy_mask[hole] = False


# ======================================================================
# STEP 4 — STRUCTURING ELEMENTS
# ======================================================================

def square_structuring_element(size):
    """
    Create a square binary structuring element.

    Example size=3:

        1 1 1
        1 1 1
        1 1 1
    """

    assert size > 0
    assert size % 2 == 1

    return np.ones(
        (size, size),
        dtype=bool
    )


def cross_structuring_element(size):
    """
    Create a cross-shaped structuring element.

    Example size=5:

        0 0 1 0 0
        0 0 1 0 0
        1 1 1 1 1
        0 0 1 0 0
        0 0 1 0 0
    """

    assert size > 0
    assert size % 2 == 1

    kernel = np.zeros(
        (size, size),
        dtype=bool
    )

    center = size // 2

    kernel[center, :] = True
    kernel[:, center] = True

    return kernel


def horizontal_structuring_element(size):
    """
    Create a horizontal line structuring element.
    """

    assert size > 0
    assert size % 2 == 1

    return np.ones(
        (1, size),
        dtype=bool
    )


def vertical_structuring_element(size):
    """
    Create a vertical line structuring element.
    """

    assert size > 0
    assert size % 2 == 1

    return np.ones(
        (size, 1),
        dtype=bool
    )


kernel = square_structuring_element(5)


# ======================================================================
# STEP 5 — PADDING
# ======================================================================

def morphological_pad(
    image,
    kernel,
    value=False
):
    """
    Pad binary image using constant-value padding.

    We use zero/False padding.

    IMPORTANT:
    This creates a finite-image boundary condition.

    Therefore mathematical morphology properties involving
    set inclusion should be tested away from the boundary.
    """

    kh, kw = kernel.shape

    pad_y = kh // 2
    pad_x = kw // 2

    return np.pad(
        image,
        (
            (pad_y, pad_y),
            (pad_x, pad_x)
        ),
        mode="constant",
        constant_values=value
    )


# ======================================================================
# STEP 6 — EROSION FROM SCRATCH
# ======================================================================

def erosion(
    image,
    kernel
):
    """
    Binary erosion.

    Mathematical idea:

        A ⊖ B

    A pixel remains foreground only when every active position
    of B overlaps foreground in A.
    """

    image = image.astype(bool)
    kernel = kernel.astype(bool)

    padded = morphological_pad(
        image,
        kernel,
        value=False
    )

    height, width = image.shape

    output = np.zeros(
        image.shape,
        dtype=bool
    )

    kernel_positions = np.argwhere(
        kernel
    )

    for y in range(height):

        for x in range(width):

            region = padded[
                y:y + kernel.shape[0],
                x:x + kernel.shape[1]
            ]

            valid = True

            for ky, kx in kernel_positions:

                if not region[ky, kx]:

                    valid = False
                    break

            output[y, x] = valid

    return output


# ======================================================================
# STEP 7 — DILATION FROM SCRATCH
# ======================================================================

def dilation(
    image,
    kernel
):
    """
    Binary dilation.

    Mathematical idea:

        A ⊕ B

    A pixel becomes foreground when at least one active position
    of B overlaps foreground.
    """

    image = image.astype(bool)
    kernel = kernel.astype(bool)

    padded = morphological_pad(
        image,
        kernel,
        value=False
    )

    height, width = image.shape

    output = np.zeros(
        image.shape,
        dtype=bool
    )

    kernel_positions = np.argwhere(
        kernel
    )

    for y in range(height):

        for x in range(width):

            region = padded[
                y:y + kernel.shape[0],
                x:x + kernel.shape[1]
            ]

            valid = False

            for ky, kx in kernel_positions:

                if region[ky, kx]:

                    valid = True
                    break

            output[y, x] = valid

    return output


# ======================================================================
# STEP 8 — OPENING
# ======================================================================

def opening(
    image,
    kernel
):
    """
    Opening:

        erosion → dilation
    """

    eroded = erosion(
        image,
        kernel
    )

    return dilation(
        eroded,
        kernel
    )


# ======================================================================
# STEP 9 — CLOSING
# ======================================================================

def closing(
    image,
    kernel
):
    """
    Closing:

        dilation → erosion
    """

    dilated = dilation(
        image,
        kernel
    )

    return erosion(
        dilated,
        kernel
    )


# ======================================================================
# STEP 10 — MORPHOLOGICAL GRADIENT
# ======================================================================

def morphological_gradient(
    image,
    kernel
):
    """
    Morphological gradient:

        dilation(A) - erosion(A)
    """

    dilated = dilation(
        image,
        kernel
    )

    eroded = erosion(
        image,
        kernel
    )

    return dilated & (~eroded)


# ======================================================================
# STEP 11 — BOUNDARY EXTRACTION
# ======================================================================

def extract_boundary(
    image,
    kernel
):
    """
    Internal boundary:

        A - erosion(A)
    """

    eroded = erosion(
        image,
        kernel
    )

    return image & (~eroded)


# ======================================================================
# STEP 12 — WHITE TOP-HAT
# ======================================================================

def white_top_hat(
    image,
    kernel
):
    """
    White top-hat:

        A - opening(A)
    """

    opened = opening(
        image,
        kernel
    )

    return image & (~opened)


# ======================================================================
# STEP 13 — BLACK TOP-HAT
# ======================================================================

def black_top_hat(
    image,
    kernel
):
    """
    Black top-hat:

        closing(A) - A
    """

    closed = closing(
        image,
        kernel
    )

    return closed & (~image)


# ======================================================================
# STEP 14 — APPLY OPERATIONS
# ======================================================================

eroded = erosion(
    noisy_mask,
    kernel
)

dilated = dilation(
    noisy_mask,
    kernel
)

opened = opening(
    noisy_mask,
    kernel
)

closed = closing(
    noisy_mask,
    kernel
)

gradient = morphological_gradient(
    noisy_mask,
    kernel
)

boundary = extract_boundary(
    noisy_mask,
    kernel
)

white_tophat = white_top_hat(
    noisy_mask,
    kernel
)

black_tophat = black_top_hat(
    noisy_mask,
    kernel
)


# ======================================================================
# STEP 15 — STATISTICS
# ======================================================================

def foreground_count(mask):
    """
    Count foreground pixels.
    """

    return int(
        np.sum(mask)
    )


print("=" * 72)
print("LESSON 07 — MORPHOLOGICAL IMAGE PROCESSING")
print("=" * 72)

print("\nSTRUCTURING ELEMENT")
print("-" * 72)
print(kernel.astype(int))

print("\nFOREGROUND PIXEL COUNTS")
print("-" * 72)

print(
    "Clean mask     :",
    foreground_count(clean_mask)
)

print(
    "Noisy mask     :",
    foreground_count(noisy_mask)
)

print(
    "Erosion        :",
    foreground_count(eroded)
)

print(
    "Dilation       :",
    foreground_count(dilated)
)

print(
    "Opening        :",
    foreground_count(opened)
)

print(
    "Closing        :",
    foreground_count(closed)
)

print(
    "Gradient       :",
    foreground_count(gradient)
)

print(
    "Boundary       :",
    foreground_count(boundary)
)

print(
    "White top-hat  :",
    foreground_count(white_tophat)
)

print(
    "Black top-hat  :",
    foreground_count(black_tophat)
)


# ======================================================================
# STEP 16 — TESTS
# ======================================================================

print("\nRUNNING TESTS")
print("-" * 72)


# ----------------------------------------------------------------------
# Shape tests
# ----------------------------------------------------------------------

assert eroded.shape == noisy_mask.shape
assert dilated.shape == noisy_mask.shape
assert opened.shape == noisy_mask.shape
assert closed.shape == noisy_mask.shape

print("Shape tests passed.")


# ----------------------------------------------------------------------
# Type tests
# ----------------------------------------------------------------------

assert eroded.dtype == np.bool_
assert dilated.dtype == np.bool_
assert opened.dtype == np.bool_
assert closed.dtype == np.bool_

print("Binary dtype tests passed.")


# ----------------------------------------------------------------------
# Binary-value tests
# ----------------------------------------------------------------------

assert np.all(
    np.isin(
        eroded,
        [False, True]
    )
)

assert np.all(
    np.isin(
        dilated,
        [False, True]
    )
)

print("Binary value tests passed.")


# ======================================================================
# STEP 17 — EROSION SUBSET PROPERTY
# ======================================================================
#
# Erosion cannot create foreground outside the original foreground:
#
#     A ⊖ B ⊆ A
#
# This is valid for our zero-padded implementation.

assert np.all(
    (~eroded) | noisy_mask
)

print(
    "Erosion subset test passed."
)


# ======================================================================
# STEP 18 — DILATION SUPERSET PROPERTY
# ======================================================================
#
# Because our kernels contain their center:
#
#     A ⊆ A ⊕ B
#
# should hold.

center_y = kernel.shape[0] // 2
center_x = kernel.shape[1] // 2

assert kernel[
    center_y,
    center_x
]

assert np.all(
    (~noisy_mask) | dilated
)

print(
    "Dilation superset test passed."
)


# ======================================================================
# STEP 19 — OPENING SUBSET PROPERTY
# ======================================================================
#
# Opening is anti-extensive:
#
#     Opening(A) ⊆ A
#
# Again, this holds for our implementation.

assert np.all(
    (~opened) | noisy_mask
)

print(
    "Opening subset test passed."
)


# ======================================================================
# STEP 20 — CLOSING EXTENSIVE PROPERTY
# ======================================================================
#
# The theoretical mathematical property is:
#
#     A ⊆ Closing(A)
#
# BUT:
#
# Our image is finite and we use zero padding.
#
# At the outer image boundary, a foreground pixel can be affected
# by the artificial zero-valued region outside the image.
#
# Therefore we test this property only on the interior.
#
# The margin must be at least the kernel radius.

radius_y = kernel.shape[0] // 2
radius_x = kernel.shape[1] // 2

interior_original = noisy_mask[
    radius_y:-radius_y,
    radius_x:-radius_x
]

interior_closed = closed[
    radius_y:-radius_y,
    radius_x:-radius_x
]

assert np.all(
    (~interior_original)
    |
    interior_closed
)

print(
    "Closing extensive-property test passed "
    "(interior region only)."
)


# ======================================================================
# STEP 21 — SMALL OBJECT EROSION TEST
# ======================================================================
#
# A 3×3 object should disappear under a 5×5 square erosion.

small_object = np.zeros(
    (25, 25),
    dtype=bool
)

small_object[
    11:14,
    11:14
] = True

small_kernel = square_structuring_element(
    5
)

small_eroded = erosion(
    small_object,
    small_kernel
)

assert (
    foreground_count(small_eroded)
    == 0
)

print(
    "Small-object erosion test passed."
)


# ======================================================================
# STEP 22 — SINGLE-PIXEL DILATION TEST
# ======================================================================
#
# A single center pixel with a 3×3 square kernel should become
# a 3×3 region.

single_pixel = np.zeros(
    (21, 21),
    dtype=bool
)

single_pixel[
    10,
    10
] = True

dilation_kernel = square_structuring_element(
    3
)

single_dilated = dilation(
    single_pixel,
    dilation_kernel
)

assert (
    foreground_count(single_dilated)
    == 9
)

print(
    "Single-pixel dilation test passed."
)


print("\nALL TESTS PASSED.")


# ======================================================================
# STEP 23 — VISUALIZE MAIN OPERATIONS
# ======================================================================

plt.figure(
    figsize=(16, 10)
)

plt.subplot(2, 3, 1)

plt.imshow(
    clean_mask,
    cmap="gray"
)

plt.title(
    "Clean Ground Truth"
)

plt.axis("off")


plt.subplot(2, 3, 2)

plt.imshow(
    noisy_mask,
    cmap="gray"
)

plt.title(
    "Noisy Binary Mask"
)

plt.axis("off")


plt.subplot(2, 3, 3)

plt.imshow(
    eroded,
    cmap="gray"
)

plt.title(
    "Erosion"
)

plt.axis("off")


plt.subplot(2, 3, 4)

plt.imshow(
    dilated,
    cmap="gray"
)

plt.title(
    "Dilation"
)

plt.axis("off")


plt.subplot(2, 3, 5)

plt.imshow(
    opened,
    cmap="gray"
)

plt.title(
    "Opening"
)

plt.axis("off")


plt.subplot(2, 3, 6)

plt.imshow(
    closed,
    cmap="gray"
)

plt.title(
    "Closing"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 24 — BOUNDARY VISUALIZATION
# ======================================================================

plt.figure(
    figsize=(16, 5)
)

plt.subplot(1, 3, 1)

plt.imshow(
    gradient,
    cmap="gray"
)

plt.title(
    "Morphological Gradient"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    boundary,
    cmap="gray"
)

plt.title(
    "Internal Boundary"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    noisy_mask,
    cmap="gray"
)

plt.title(
    "Original Mask"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 25 — TOP-HAT VISUALIZATION
# ======================================================================

plt.figure(
    figsize=(15, 5)
)

plt.subplot(1, 3, 1)

plt.imshow(
    noisy_mask,
    cmap="gray"
)

plt.title(
    "Original Mask"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    white_tophat,
    cmap="gray"
)

plt.title(
    "White Top-Hat"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    black_tophat,
    cmap="gray"
)

plt.title(
    "Black Top-Hat"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 26 — STRUCTURING ELEMENT COMPARISON
# ======================================================================

square_kernel = square_structuring_element(
    7
)

cross_kernel = cross_structuring_element(
    7
)

horizontal_kernel = horizontal_structuring_element(
    11
)

vertical_kernel = vertical_structuring_element(
    11
)

square_result = erosion(
    noisy_mask,
    square_kernel
)

cross_result = erosion(
    noisy_mask,
    cross_kernel
)

horizontal_result = erosion(
    noisy_mask,
    horizontal_kernel
)

vertical_result = erosion(
    noisy_mask,
    vertical_kernel
)


# ======================================================================
# STEP 27 — VISUALIZE STRUCTURING ELEMENTS
# ======================================================================

plt.figure(
    figsize=(12, 9)
)

plt.subplot(2, 2, 1)

plt.imshow(
    square_kernel,
    cmap="gray"
)

plt.title(
    "7×7 Square"
)

plt.axis("off")


plt.subplot(2, 2, 2)

plt.imshow(
    cross_kernel,
    cmap="gray"
)

plt.title(
    "7×7 Cross"
)

plt.axis("off")


plt.subplot(2, 2, 3)

plt.imshow(
    horizontal_kernel,
    cmap="gray"
)

plt.title(
    "Horizontal Line"
)

plt.axis("off")


plt.subplot(2, 2, 4)

plt.imshow(
    vertical_kernel,
    cmap="gray"
)

plt.title(
    "Vertical Line"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 28 — COMPARE KERNEL SHAPES
# ======================================================================

plt.figure(
    figsize=(16, 8)
)

plt.subplot(2, 2, 1)

plt.imshow(
    square_result,
    cmap="gray"
)

plt.title(
    "Erosion — Square Kernel"
)

plt.axis("off")


plt.subplot(2, 2, 2)

plt.imshow(
    cross_result,
    cmap="gray"
)

plt.title(
    "Erosion — Cross Kernel"
)

plt.axis("off")


plt.subplot(2, 2, 3)

plt.imshow(
    horizontal_result,
    cmap="gray"
)

plt.title(
    "Erosion — Horizontal Kernel"
)

plt.axis("off")


plt.subplot(2, 2, 4)

plt.imshow(
    vertical_result,
    cmap="gray"
)

plt.title(
    "Erosion — Vertical Kernel"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 29 — KERNEL SIZE EXPERIMENT
# ======================================================================

kernel_sizes = [
    3,
    5,
    9,
    15
]

erosion_results = []

for size in kernel_sizes:

    k = square_structuring_element(
        size
    )

    result = erosion(
        noisy_mask,
        k
    )

    erosion_results.append(
        result
    )


plt.figure(
    figsize=(16, 8)
)

for i, size in enumerate(
    kernel_sizes
):

    plt.subplot(
        2,
        2,
        i + 1
    )

    plt.imshow(
        erosion_results[i],
        cmap="gray"
    )

    plt.title(
        f"Erosion — Kernel {size}×{size}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 30 — KERNEL SIZE STATISTICS
# ======================================================================

print("\nKERNEL SIZE EXPERIMENT")
print("-" * 72)

for size, result in zip(
    kernel_sizes,
    erosion_results
):

    print(
        f"Kernel {size:2d}×{size:2d}"
        f" → foreground pixels:"
        f" {foreground_count(result)}"
    )


# ======================================================================
# STEP 31 — ROBOTICS INTERPRETATION
# ======================================================================
#
# OpenArm V2 waste sorting:
#
#     CAMERA
#        ↓
#     RGB IMAGE
#        ↓
#     GRAYSCALE
#        ↓
#     THRESHOLD
#        ↓
#     NOISY MASK
#        ↓
#     OPENING
#        ↓
#     REMOVE SMALL NOISE
#        ↓
#     CLOSING
#        ↓
#     FILL SMALL GAPS
#        ↓
#     CLEAN OBJECT MASK
#        ↓
#     CONNECTED COMPONENTS
#        ↓
#     OBJECT CENTER
#        ↓
#     GRASP LOCATION
#        ↓
#     ACT / ROBOT CONTROL
#
# This classical pipeline is useful because every step is
# interpretable.
#
# Later in the course, learned vision models will handle cases
# where thresholding and morphology are no longer sufficient.


# ======================================================================
# STEP 32 — FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 72)
print("LESSON 07 COMPLETE")
print("=" * 72)

print(
    """
You learned:

1. Morphology operates on binary image structure.
2. A structuring element defines the local neighborhood.
3. Erosion shrinks foreground regions.
4. Dilation expands foreground regions.
5. Opening = erosion followed by dilation.
6. Closing = dilation followed by erosion.
7. Morphological gradient highlights boundaries.
8. Boundary extraction can be based on erosion.
9. White top-hat detects structures removed by opening.
10. Black top-hat detects structures filled by closing.
11. Kernel size controls the spatial scale.
12. Kernel shape controls geometric behavior.
13. Finite-image boundary conditions matter when testing
    mathematical morphology properties.

The classical segmentation pipeline is now:

    IMAGE
      ↓
    GRAYSCALE
      ↓
    THRESHOLD
      ↓
    BINARY MASK
      ↓
    MORPHOLOGY
      ↓
    CLEAN MASK
      ↓
    OBJECT EXTRACTION

NEXT:
LESSON 08 — GEOMETRIC IMAGE TRANSFORMATIONS

We will move from changing pixel values to changing pixel
coordinates:

    translation
    rotation
    scaling
    affine transformation
    interpolation
    perspective transformation
"""
)