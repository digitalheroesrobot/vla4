"""
===============================================================================
COMPUTER VISION FROM SCRATCH
LESSON 01 — INTRODUCTION TO COMPUTER VISION
===============================================================================

COURSE RULE
-----------
This course builds Computer Vision from scratch.

We do NOT use:
    - pretrained models
    - pretrained weights
    - black-box vision models

We will progressively build the concepts ourselves.

PURPOSE
-------
Understand the most fundamental idea in Computer Vision:

    IMAGE → INFORMATION → DECISION

A computer does not "see" an image like a human.

An image is numerical data.

Computer Vision is the process of extracting useful information from
those numbers.

WHAT YOU LEARN
--------------
1. What Computer Vision is.
2. How an image becomes numerical data.
3. How pixels form an image.
4. How to inspect image data using NumPy.
5. How to visualize an image.
6. How to calculate simple visual measurements.
7. How a simple threshold can become a primitive perception system.
8. How this eventually connects to robot perception.

REAL-WORLD EXAMPLE
------------------
Imagine an OpenArm robot sorting waste.

A complete future system could be:

    Camera
       ↓
    Image
       ↓
    Object Detection
       ↓
    Object Segmentation
       ↓
    3-D Localization
       ↓
    Robot Coordinate Transform
       ↓
    Manipulation Policy
       ↓
    Robot Action

This first lesson studies only the beginning:

    Camera → Image → Numbers

===============================================================================
STEP 1 — WHAT IS AN IMAGE?
===============================================================================

A digital image is an array of numbers.

For a grayscale image:

    Image ∈ R^(H × W)

where:

    H = image height
    W = image width

Each number represents the intensity of one pixel.

For an 8-bit grayscale image:

    0   = black
    255 = white

For a color image:

    Image ∈ R^(H × W × C)

where:

    C = number of channels

For RGB:

    C = 3

The three channels are:

    Red
    Green
    Blue

For example:

    640 × 480 × 3

means:

    width  = 640
    height = 480
    channels = 3

===============================================================================
STEP 2 — COMPUTER VISION AS A DATA PIPELINE
===============================================================================

A useful mental model is:

    RAW IMAGE
        ↓
    NUMERICAL REPRESENTATION
        ↓
    PREPROCESSING
        ↓
    FEATURES
        ↓
    PERCEPTION
        ↓
    DECISION
        ↓
    ACTION

For example:

    Camera
       ↓
    RGB image
       ↓
    Detect red pixels
       ↓
    Estimate object region
       ↓
    "This may be a red object"
       ↓
    Robot decides whether to pick it

Later in this course, neural networks will replace many manually designed
steps.

But the fundamental pipeline remains.

===============================================================================
STEP 3 — WHY NUMPY?
===============================================================================

NumPy provides numerical arrays.

For Computer Vision, these are fundamental.

We can inspect:

    image.shape
    image.dtype
    image.min()
    image.max()
    image.mean()

Computer Vision is fundamentally numerical.

Before using complicated algorithms, we need to understand the data itself.

===============================================================================
STEP 4 — SYNTHETIC IMAGE
===============================================================================

We create an image ourselves.

The image contains:

    - black background
    - white rectangle
    - gray circle

This lets us study the pixels without requiring an external image file.

===============================================================================
STEP 5 — IMAGE STATISTICS
===============================================================================

We calculate:

    minimum pixel value
    maximum pixel value
    mean pixel value
    standard deviation

These statistics provide basic information about the image.

Mean intensity:

    average brightness

Standard deviation:

    amount of intensity variation

These concepts become useful later for:

    normalization
    preprocessing
    contrast analysis
    image quality analysis

===============================================================================
STEP 6 — IMAGE COORDINATES
===============================================================================

Images use two-dimensional coordinates.

Conceptually:

    x → horizontal
    y → vertical

But NumPy uses:

    image[y, x]

Therefore:

    image[row, column]

This distinction is extremely important.

===============================================================================
STEP 7 — SIMPLE VISUAL PERCEPTION
===============================================================================

We create a simple binary mask.

Rule:

    pixel > threshold

Then:

    True  → foreground
    False → background

This allows us to estimate:

    - foreground area
    - bounding box
    - centroid

This is a primitive form of object perception.

Later we will build much more sophisticated systems.

===============================================================================
STEP 8 — REAL-WORLD CONNECTION
===============================================================================

A simple threshold can work when:

    - lighting is controlled
    - object appearance is predictable
    - background is simple

But real environments contain:

    - shadows
    - reflections
    - different lighting
    - occlusion
    - clutter
    - multiple objects
    - different materials

Therefore, we need more advanced algorithms.

The course progression is:

    Pixels
       ↓
    Filtering
       ↓
    Edges
       ↓
    Features
       ↓
    Classical CV
       ↓
    CNN
       ↓
    Detection
       ↓
    Segmentation
       ↓
    Tracking
       ↓
    3-D Vision
       ↓
    Transformers
       ↓
    Vision-Language Models
       ↓
    Vision → Action

===============================================================================
STEP 9 — TESTING
===============================================================================

We verify:

    - image dimensions
    - image dtype
    - valid pixel range
    - foreground existence

Testing is important throughout this course.

We do not want Computer Vision systems that merely "look like they work."

We want measurable and reproducible results.

===============================================================================
EXERCISES
===============================================================================

Exercise 1
----------
Change the rectangle size.

Exercise 2
----------
Change the circle brightness.

Exercise 3
----------
Add another object.

Exercise 4
----------
Change the threshold.

Observe how the binary mask changes.

Exercise 5
----------
Create a grayscale gradient image.

Exercise 6
----------
Modify the program to detect multiple bright regions.

===============================================================================
NEXT LESSON
===============================================================================

LESSON 02 — DIGITAL IMAGES AND PIXEL REPRESENTATION

We will study:

    - image arrays
    - width
    - height
    - channels
    - dtype
    - RGB / BGR
    - pixel indexing
    - image memory
    - normalization

The goal is to become completely comfortable treating an image as data.

===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# STEP 1 — CREATE A SYNTHETIC GRAYSCALE IMAGE
# =============================================================================

HEIGHT = 256
WIDTH = 384

# Create a black image.
#
# Shape:
#
#     (HEIGHT, WIDTH)
#
# dtype:
#
#     uint8
#
# Pixel range:
#
#     0 ... 255
#
image = np.zeros(
    (HEIGHT, WIDTH),
    dtype=np.uint8,
)


# =============================================================================
# ADD A WHITE RECTANGLE
# =============================================================================

image[70:180, 70:220] = 255


# =============================================================================
# ADD A GRAY CIRCLE
# =============================================================================

yy, xx = np.ogrid[:HEIGHT, :WIDTH]

circle_center_x = 290
circle_center_y = 125
circle_radius = 50

circle_mask = (
    (xx - circle_center_x) ** 2
    + (yy - circle_center_y) ** 2
    <= circle_radius ** 2
)

image[circle_mask] = 160


# =============================================================================
# STEP 2 — DISPLAY BASIC IMAGE INFORMATION
# =============================================================================

print("=" * 72)
print("COMPUTER VISION FROM SCRATCH")
print("LESSON 01 — INTRODUCTION TO COMPUTER VISION")
print("=" * 72)

print("\nIMAGE INFORMATION")
print("-" * 72)

print(f"Shape              : {image.shape}")
print(f"Height             : {image.shape[0]}")
print(f"Width              : {image.shape[1]}")
print(f"Data type          : {image.dtype}")
print(f"Minimum pixel      : {image.min()}")
print(f"Maximum pixel      : {image.max()}")
print(f"Mean intensity     : {image.mean():.3f}")
print(f"Std. deviation     : {image.std():.3f}")


# =============================================================================
# STEP 3 — VERIFY BASIC IMAGE ASSUMPTIONS
# =============================================================================

assert image.shape == (HEIGHT, WIDTH)

assert image.dtype == np.uint8

assert image.min() >= 0

assert image.max() <= 255

assert image.max() > image.min()

print("\nBasic image tests: PASSED")


# =============================================================================
# STEP 4 — INSPECT INDIVIDUAL PIXELS
# =============================================================================

print("\nSELECTED PIXELS")
print("-" * 72)

coordinates = [
    (10, 10),
    (100, 100),
    (125, 290),
]

for y, x in coordinates:
    value = image[y, x]

    print(
        f"pixel[y={y}, x={x}] = {value}"
    )


# =============================================================================
# STEP 5 — SIMPLE THRESHOLDING
# =============================================================================

# Pixels above this value are considered foreground.

threshold = 100

binary_mask = image > threshold

foreground_pixels = int(
    binary_mask.sum()
)

foreground_ratio = (
    foreground_pixels
    / binary_mask.size
)

print("\nSIMPLE VISUAL PERCEPTION")
print("-" * 72)

print(f"Threshold          : {threshold}")

print(
    f"Foreground pixels  : {foreground_pixels}"
)

print(
    f"Foreground ratio   : {foreground_ratio:.4f}"
)


# =============================================================================
# STEP 6 — ESTIMATE A BOUNDING BOX
# =============================================================================

ys, xs = np.where(binary_mask)

if len(xs) > 0:

    x_min = int(xs.min())
    x_max = int(xs.max())

    y_min = int(ys.min())
    y_max = int(ys.max())

    bbox_width = (
        x_max - x_min + 1
    )

    bbox_height = (
        y_max - y_min + 1
    )

    print("\nFOREGROUND BOUNDING BOX")
    print("-" * 72)

    print(f"x_min              : {x_min}")
    print(f"y_min              : {y_min}")
    print(f"x_max              : {x_max}")
    print(f"y_max              : {y_max}")

    print(
        f"Bounding-box width  : {bbox_width}"
    )

    print(
        f"Bounding-box height : {bbox_height}"
    )


# =============================================================================
# STEP 7 — ESTIMATE THE FOREGROUND CENTROID
# =============================================================================

if foreground_pixels > 0:

    centroid_x = float(
        xs.mean()
    )

    centroid_y = float(
        ys.mean()
    )

    print("\nFOREGROUND CENTROID")
    print("-" * 72)

    print(
        f"Centroid x         : {centroid_x:.2f}"
    )

    print(
        f"Centroid y         : {centroid_y:.2f}"
    )


# =============================================================================
# STEP 8 — VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
)


# -----------------------------------------------------------------------------
# Original image
# -----------------------------------------------------------------------------

axes[0].imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255,
)

axes[0].set_title(
    "Synthetic Image"
)

axes[0].set_xlabel("x")

axes[0].set_ylabel("y")


# -----------------------------------------------------------------------------
# Binary mask
# -----------------------------------------------------------------------------

axes[1].imshow(
    binary_mask,
    cmap="gray",
)

axes[1].set_title(
    "Threshold Mask"
)

axes[1].set_xlabel("x")

axes[1].set_ylabel("y")


# -----------------------------------------------------------------------------
# Perception result
# -----------------------------------------------------------------------------

axes[2].imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255,
)

if foreground_pixels > 0:

    # Draw centroid.
    axes[2].plot(
        centroid_x,
        centroid_y,
        marker="x",
        markersize=12,
        markeredgewidth=3,
    )

    # Draw bounding box.
    axes[2].plot(
        [
            x_min,
            x_max,
            x_max,
            x_min,
            x_min,
        ],
        [
            y_min,
            y_min,
            y_max,
            y_max,
            y_min,
        ],
        linewidth=2,
    )

axes[2].set_title(
    "Simple Visual Perception"
)

axes[2].set_xlabel("x")

axes[2].set_ylabel("y")


plt.tight_layout()

plt.show()


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 72)

print("LESSON 01 COMPLETE")

print("=" * 72)

print(
    """
The computer does not directly understand an image.

It receives numerical data.

We performed:

    image
      ↓
    pixels
      ↓
    threshold
      ↓
    binary mask
      ↓
    bounding box
      ↓
    centroid

This is already a primitive Computer Vision system.

NEXT:
    Lesson 02 — Digital Images and Pixel Representation
"""
)