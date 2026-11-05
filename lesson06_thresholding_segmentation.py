"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 06 — THRESHOLDING AND BINARY IMAGE SEGMENTATION
======================================================================

COURSE RULE
-----------
This course is built from scratch.

NO:
    - pretrained models
    - pretrained weights
    - pretrained feature extractors
    - cv2.threshold() for the main implementation
    - cv2.adaptiveThreshold()
    - cv2.otsu-style black-box implementations

We use:
    - NumPy
    - Matplotlib

----------------------------------------------------------------------
WHAT YOU LEARN
----------------------------------------------------------------------

STEP 1 — What is segmentation?

Segmentation means separating an image into meaningful regions.

The simplest form is:

    foreground
        vs.
    background

For example:

        background
        background
        object
        object
        object

can become:

        0 0 0 0 0
        0 0 0 0 0
        0 1 1 1 0
        0 1 1 1 0
        0 0 0 0 0

where:

    0 = background
    1 = foreground

This is called a binary mask.

----------------------------------------------------------------------
STEP 2 — GLOBAL THRESHOLDING
----------------------------------------------------------------------

Given a grayscale image I:

        M(x,y) =
            1, if I(x,y) > T
            0, otherwise

where:

        T = threshold

Example:

    T = 128

then:

    intensity < 128 → background
    intensity > 128 → foreground

Global thresholding assumes one threshold can separate the
foreground from the background.

----------------------------------------------------------------------
STEP 3 — WHY THRESHOLDING WORKS

Suppose an image contains:

    background ≈ 30
    object     ≈ 200

The intensity distributions may look approximately like:

    frequency
       ^
       |
       |      object
       |       /\
       |      /  \
       |     /    \
       |    /
       |   /
       |  / background
       | /\
       +--------------------> intensity

There may be a valley between the two distributions.

A threshold near that valley can separate them.

----------------------------------------------------------------------
STEP 4 — PROBLEM WITH GLOBAL THRESHOLDING

Real images may have non-uniform illumination.

Example:

    left side  = dark
    right side = bright

The same object may have:

    dark pixels on one side
    bright pixels on the other side

One global threshold may fail.

This motivates adaptive/local thresholding.

----------------------------------------------------------------------
STEP 5 — ADAPTIVE THRESHOLDING
----------------------------------------------------------------------

Instead of using one global T:

        T = constant

we calculate a local threshold:

        T(x,y)

based on nearby pixels.

A simple method is:

        T(x,y) = local mean - C

Then:

        M(x,y) =
            1 if I(x,y) > T(x,y)
            0 otherwise

This is useful when illumination changes spatially.

----------------------------------------------------------------------
STEP 6 — LOCAL MEAN
----------------------------------------------------------------------

For a window W around a pixel:

        mean(x,y)
        =
        1/N Σ I(i,j)

Then:

        threshold(x,y)
        =
        mean(x,y) - C

This is a simple adaptive threshold.

----------------------------------------------------------------------
STEP 7 — OTSU THRESHOLDING
----------------------------------------------------------------------

Otsu's method automatically chooses a global threshold.

Instead of manually selecting T, we test possible thresholds.

For each threshold t:

    foreground:
        pixels <= t

    background:
        pixels > t

Otsu tries to maximize between-class variance.

----------------------------------------------------------------------
STEP 8 — OTSU MATHEMATICS
----------------------------------------------------------------------

Let:

        ω0(t) = probability of class 0
        ω1(t) = probability of class 1

and:

        μ0(t) = mean intensity of class 0
        μ1(t) = mean intensity of class 1

The between-class variance is:

        σ_b²(t)
        =
        ω0(t) ω1(t)
        [μ0(t) - μ1(t)]²

The optimal threshold is:

        t* = argmax σ_b²(t)

We will implement this calculation ourselves.

----------------------------------------------------------------------
STEP 9 — HISTOGRAM CONNECTION
----------------------------------------------------------------------

Otsu does not need to inspect every pixel for every threshold.

It can use the histogram.

This connects the previous lesson:

        IMAGE
          ↓
        HISTOGRAM
          ↓
        CLASS STATISTICS
          ↓
        OTSU
          ↓
        THRESHOLD
          ↓
        BINARY MASK

----------------------------------------------------------------------
STEP 10 — SEGMENTATION QUALITY
----------------------------------------------------------------------

If we know the ground-truth mask, we can measure segmentation.

Important metrics include:

    True Positive  (TP)
    True Negative  (TN)
    False Positive (FP)
    False Negative (FN)

From these we can calculate:

    Precision
    Recall
    IoU
    Dice coefficient

These metrics become extremely important later for:

    object detection
    semantic segmentation
    instance segmentation

----------------------------------------------------------------------
IoU
----------------------------------------------------------------------

Intersection over Union:

        IoU =
        intersection / union

For binary masks:

        IoU =
        TP / (TP + FP + FN)

----------------------------------------------------------------------
DICE
----------------------------------------------------------------------

Dice coefficient:

        Dice =
        2TP / (2TP + FP + FN)

Both metrics measure overlap between predicted and true masks.

----------------------------------------------------------------------
REAL-WORLD ROBOTICS EXAMPLE
----------------------------------------------------------------------

Consider an OpenArm V2 waste-sorting robot.

Camera:

        RGB image
             ↓
        grayscale
             ↓
        threshold
             ↓
        binary mask
             ↓
        object region
             ↓
        object location
             ↓
        robot grasp

For example:

        bright plastic cup
              vs.
        dark conveyor

may be easy to segment using a simple threshold.

But if lighting changes:

        bright side
        dark side
        shadows
        reflections

global thresholding may fail.

Adaptive thresholding can sometimes handle this better.

For more difficult waste scenes, later lessons will move toward:

    - contours
    - connected components
    - learned segmentation
    - instance segmentation
    - transformer-based segmentation

----------------------------------------------------------------------
LESSON EXPERIMENTS
----------------------------------------------------------------------

We will:

    1. Create a synthetic image.
    2. Create ground-truth segmentation.
    3. Apply global thresholds.
    4. Implement local adaptive thresholding.
    5. Implement Otsu thresholding.
    6. Compare the methods.
    7. Calculate IoU.
    8. Calculate Dice.
    9. Visualize errors.

----------------------------------------------------------------------
EXERCISES
----------------------------------------------------------------------

EXERCISE 1
----------
Change the global threshold:

    50
    100
    150
    200

Compare the masks.


EXERCISE 2
----------
Increase illumination variation.

Observe when global thresholding starts to fail.


EXERCISE 3
----------
Change adaptive window size:

    7
    15
    31
    51

Compare the results.


EXERCISE 4
----------
Change the adaptive constant C.

Try:

    0
    5
    10
    20


EXERCISE 5
----------
Implement Otsu using a slow pixel-based approach.

Then compare it with the histogram-based implementation.


EXERCISE 6
----------
Calculate:

    Precision
    Recall
    IoU
    Dice

for your segmentation.


EXERCISE 7
----------
Create an image containing two objects with different
intensities.

Try to separate both objects from the background.

----------------------------------------------------------------------
NEXT LESSON
----------------------------------------------------------------------

LESSON 07 — MORPHOLOGICAL IMAGE PROCESSING

We will implement from scratch:

    - erosion
    - dilation
    - opening
    - closing
    - morphological noise removal
    - hole filling
    - boundary extraction

======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# STEP 1 — CREATE A SYNTHETIC OBJECT
# ======================================================================

HEIGHT = 256
WIDTH = 384

yy, xx = np.ogrid[:HEIGHT, :WIDTH]

# Base background
clean_image = np.full(
    (HEIGHT, WIDTH),
    40.0,
    dtype=np.float32
)

# ----------------------------------------------------------------------
# Main rectangular object
# ----------------------------------------------------------------------

rectangle_mask = (
    (xx >= 70)
    &
    (xx < 250)
    &
    (yy >= 60)
    &
    (yy < 190)
)

clean_image[rectangle_mask] = 210.0


# ----------------------------------------------------------------------
# Circular object
# ----------------------------------------------------------------------

circle_center_x = 300
circle_center_y = 125
circle_radius = 45

circle_mask = (
    (xx - circle_center_x) ** 2
    +
    (yy - circle_center_y) ** 2
    <= circle_radius ** 2
)

clean_image[circle_mask] = 180.0


# Ground-truth foreground mask
ground_truth = (
    rectangle_mask
    |
    circle_mask
)


# ======================================================================
# STEP 2 — ADD NON-UNIFORM ILLUMINATION
# ======================================================================
#
# Real cameras often observe illumination that changes across
# the image.
#
# We create a smooth brightness gradient.
#
# Left side:
#     darker
#
# Right side:
#     brighter

x_gradient = np.linspace(
    -35,
    35,
    WIDTH,
    dtype=np.float32
)

illumination = np.tile(
    x_gradient,
    (HEIGHT, 1)
)

uneven_image = clean_image + illumination

uneven_image = np.clip(
    uneven_image,
    0,
    255
)


# ======================================================================
# STEP 3 — ADD GAUSSIAN NOISE
# ======================================================================

rng = np.random.default_rng(42)

noise_sigma = 8.0

noise = rng.normal(
    0.0,
    noise_sigma,
    size=uneven_image.shape
)

image = uneven_image + noise

image = np.clip(
    image,
    0,
    255
).astype(np.float32)


# ======================================================================
# STEP 4 — GLOBAL THRESHOLDING
# ======================================================================
#
# Formula:
#
#     M(x,y) =
#         1 if I(x,y) > T
#         0 otherwise

def global_threshold(image, threshold):
    """
    Apply a global binary threshold from scratch.
    """

    return image > threshold


# Test several thresholds.

threshold_80 = global_threshold(
    image,
    80
)

threshold_120 = global_threshold(
    image,
    120
)

threshold_160 = global_threshold(
    image,
    160
)

threshold_200 = global_threshold(
    image,
    200
)


# ======================================================================
# STEP 5 — MANUAL LOCAL MEAN
# ======================================================================
#
# Adaptive thresholding needs a local statistic.
#
# We calculate:
#
#     local mean
#
# using an explicit sliding window.
#
# This implementation is intentionally straightforward rather than
# optimized.

def local_mean(image, window_size):
    """
    Calculate a local mean image from scratch.

    Parameters
    ----------
    image:
        2-D grayscale image.

    window_size:
        Odd integer, e.g. 7, 15, 31.

    Returns
    -------
    local_means:
        Same spatial size as input.
    """

    assert window_size % 2 == 1

    radius = window_size // 2

    padded = np.pad(
        image,
        (
            (radius, radius),
            (radius, radius)
        ),
        mode="reflect"
    )

    height, width = image.shape

    means = np.zeros(
        image.shape,
        dtype=np.float32
    )

    for y in range(height):

        for x in range(width):

            region = padded[
                y:y + window_size,
                x:x + window_size
            ]

            means[y, x] = region.mean()

    return means


# ======================================================================
# STEP 6 — ADAPTIVE THRESHOLDING
# ======================================================================
#
# Formula:
#
#     T(x,y) = local_mean(x,y) - C
#
# then:
#
#     mask(x,y) =
#         1 if I(x,y) > T(x,y)
#         0 otherwise

def adaptive_threshold(
    image,
    window_size,
    C
):
    """
    Simple mean-based adaptive thresholding.
    """

    means = local_mean(
        image,
        window_size
    )

    thresholds = means - C

    mask = image > thresholds

    return mask, thresholds


adaptive_mask, local_thresholds = adaptive_threshold(
    image,
    window_size=21,
    C=10
)


# ======================================================================
# STEP 7 — MANUAL HISTOGRAM
# ======================================================================

def histogram_from_scratch(image_uint8):
    """
    Calculate a grayscale histogram with 256 bins.
    """

    histogram = np.zeros(
        256,
        dtype=np.int64
    )

    flat = image_uint8.reshape(-1)

    for pixel in flat:

        histogram[
            int(pixel)
        ] += 1

    return histogram


image_uint8 = np.clip(
    image,
    0,
    255
).astype(np.uint8)


histogram = histogram_from_scratch(
    image_uint8
)


# ======================================================================
# STEP 8 — OTSU THRESHOLDING FROM SCRATCH
# ======================================================================
#
# Otsu maximizes:
#
#     σ_b² =
#         ω0 ω1 (μ0 - μ1)²
#
# We calculate this for every possible threshold.

def otsu_threshold(image_uint8):
    """
    Calculate Otsu's global threshold from scratch.

    Uses the image histogram.

    Returns:
        best_threshold
        between_class_variances
    """

    histogram = histogram_from_scratch(
        image_uint8
    )

    total_pixels = image_uint8.size

    intensities = np.arange(
        256,
        dtype=np.float64
    )

    # Total intensity sum:
    #
    #     Σ k H(k)

    total_intensity = np.sum(
        intensities * histogram
    )

    weight_background = 0.0
    sum_background = 0.0

    best_threshold = 0
    best_variance = -1.0

    variances = np.zeros(
        256,
        dtype=np.float64
    )

    for threshold in range(256):

        # Add current histogram bin to class 0.
        weight_background += (
            histogram[threshold]
            / total_pixels
        )

        sum_background += (
            threshold
            * histogram[threshold]
            / total_pixels
        )

        weight_foreground = (
            1.0
            - weight_background
        )

        # Empty class.
        if weight_background <= 0:
            continue

        # Empty class.
        if weight_foreground <= 0:
            break

        mean_background = (
            sum_background
            / weight_background
        )

        sum_foreground = (
            total_intensity
            / total_pixels
            - sum_background
        )

        mean_foreground = (
            sum_foreground
            / weight_foreground
        )

        between_variance = (
            weight_background
            * weight_foreground
            * (
                mean_background
                -
                mean_foreground
            ) ** 2
        )

        variances[threshold] = (
            between_variance
        )

        if between_variance > best_variance:

            best_variance = (
                between_variance
            )

            best_threshold = threshold

    return (
        best_threshold,
        variances
    )


otsu_t, otsu_variances = otsu_threshold(
    image_uint8
)

otsu_mask = global_threshold(
    image,
    otsu_t
)


# ======================================================================
# STEP 9 — SEGMENTATION METRICS
# ======================================================================
#
# Given:
#
#     prediction
#     ground truth
#
# calculate:
#
#     TP = predicted foreground AND true foreground
#
#     TN = predicted background AND true background
#
#     FP = predicted foreground BUT true background
#
#     FN = predicted background BUT true foreground

def confusion_counts(
    prediction,
    ground_truth
):
    """
    Calculate TP, TN, FP, FN.
    """

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    TP = np.sum(
        prediction & ground_truth
    )

    TN = np.sum(
        (~prediction) & (~ground_truth)
    )

    FP = np.sum(
        prediction & (~ground_truth)
    )

    FN = np.sum(
        (~prediction) & ground_truth
    )

    return (
        int(TP),
        int(TN),
        int(FP),
        int(FN)
    )


def segmentation_metrics(
    prediction,
    ground_truth
):
    """
    Calculate common binary segmentation metrics.
    """

    TP, TN, FP, FN = confusion_counts(
        prediction,
        ground_truth
    )

    # Precision:
    #
    #     TP / (TP + FP)

    precision_denominator = TP + FP

    if precision_denominator > 0:

        precision = (
            TP
            /
            precision_denominator
        )

    else:

        precision = 0.0


    # Recall:
    #
    #     TP / (TP + FN)

    recall_denominator = TP + FN

    if recall_denominator > 0:

        recall = (
            TP
            /
            recall_denominator
        )

    else:

        recall = 0.0


    # IoU:
    #
    #     TP / (TP + FP + FN)

    iou_denominator = (
        TP + FP + FN
    )

    if iou_denominator > 0:

        iou = (
            TP
            /
            iou_denominator
        )

    else:

        iou = 0.0


    # Dice:
    #
    #     2TP / (2TP + FP + FN)

    dice_denominator = (
        2 * TP + FP + FN
    )

    if dice_denominator > 0:

        dice = (
            2 * TP
            /
            dice_denominator
        )

    else:

        dice = 0.0


    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "dice": dice
    }


# ======================================================================
# STEP 10 — EVALUATE ALL METHODS
# ======================================================================

metrics_80 = segmentation_metrics(
    threshold_80,
    ground_truth
)

metrics_120 = segmentation_metrics(
    threshold_120,
    ground_truth
)

metrics_160 = segmentation_metrics(
    threshold_160,
    ground_truth
)

metrics_200 = segmentation_metrics(
    threshold_200,
    ground_truth
)

metrics_adaptive = segmentation_metrics(
    adaptive_mask,
    ground_truth
)

metrics_otsu = segmentation_metrics(
    otsu_mask,
    ground_truth
)


# ======================================================================
# STEP 11 — PRINT RESULTS
# ======================================================================

print("=" * 72)
print("LESSON 06 — THRESHOLDING AND BINARY IMAGE SEGMENTATION")
print("=" * 72)

print("\nIMAGE")
print("-" * 72)

print("Shape:", image.shape)
print("Min intensity:", image.min())
print("Max intensity:", image.max())
print("Mean intensity:", image.mean())


print("\nOTSU RESULT")
print("-" * 72)

print(
    "Otsu threshold:",
    otsu_t
)


print("\nSEGMENTATION RESULTS")
print("-" * 72)


def print_metrics(
    name,
    metrics
):

    print(f"\n{name}")

    print(
        f"  TP        : {metrics['TP']}"
    )

    print(
        f"  TN        : {metrics['TN']}"
    )

    print(
        f"  FP        : {metrics['FP']}"
    )

    print(
        f"  FN        : {metrics['FN']}"
    )

    print(
        f"  Precision : {metrics['precision']:.4f}"
    )

    print(
        f"  Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"  IoU       : {metrics['iou']:.4f}"
    )

    print(
        f"  Dice      : {metrics['dice']:.4f}"
    )


print_metrics(
    "Global Threshold 80",
    metrics_80
)

print_metrics(
    "Global Threshold 120",
    metrics_120
)

print_metrics(
    "Global Threshold 160",
    metrics_160
)

print_metrics(
    "Global Threshold 200",
    metrics_200
)

print_metrics(
    "Adaptive Threshold",
    metrics_adaptive
)

print_metrics(
    "Otsu Threshold",
    metrics_otsu
)


# ======================================================================
# STEP 12 — TESTS
# ======================================================================

print("\nRUNNING TESTS")
print("-" * 72)

# ----------------------------------------------------------------------
# Global threshold tests
# ----------------------------------------------------------------------

assert threshold_80.shape == image.shape
assert threshold_120.shape == image.shape

assert threshold_80.dtype == np.bool_
assert threshold_120.dtype == np.bool_


# ----------------------------------------------------------------------
# Ground truth
# ----------------------------------------------------------------------

assert ground_truth.shape == image.shape
assert ground_truth.dtype == np.bool_


# ----------------------------------------------------------------------
# Adaptive threshold
# ----------------------------------------------------------------------

assert adaptive_mask.shape == image.shape
assert local_thresholds.shape == image.shape


# ----------------------------------------------------------------------
# Histogram
# ----------------------------------------------------------------------

assert histogram.shape == (256,)
assert histogram.sum() == image.size


# ----------------------------------------------------------------------
# Otsu
# ----------------------------------------------------------------------

assert 0 <= otsu_t <= 255
assert otsu_variances.shape == (256,)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

for metrics in [
    metrics_80,
    metrics_120,
    metrics_160,
    metrics_200,
    metrics_adaptive,
    metrics_otsu
]:

    assert metrics["TP"] >= 0
    assert metrics["TN"] >= 0
    assert metrics["FP"] >= 0
    assert metrics["FN"] >= 0

    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["iou"] <= 1.0
    assert 0.0 <= metrics["dice"] <= 1.0


# ----------------------------------------------------------------------
# Relationship between Dice and IoU
# ----------------------------------------------------------------------
#
# For nonnegative TP/FP/FN:
#
#     Dice >= IoU

assert metrics_otsu["dice"] >= metrics_otsu["iou"] - 1e-12

print("All tests passed.")


# ======================================================================
# STEP 13 — VISUALIZE INPUT AND GROUND TRUTH
# ======================================================================

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.imshow(
    clean_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Clean Image")
plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Uneven Illumination + Noise")
plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    ground_truth,
    cmap="gray"
)

plt.title("Ground Truth Mask")
plt.axis("off")


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 14 — COMPARE GLOBAL THRESHOLDS
# ======================================================================

plt.figure(figsize=(16, 8))


plt.subplot(2, 2, 1)

plt.imshow(
    threshold_80,
    cmap="gray"
)

plt.title(
    f"Threshold = 80\nIoU={metrics_80['iou']:.3f}"
)

plt.axis("off")


plt.subplot(2, 2, 2)

plt.imshow(
    threshold_120,
    cmap="gray"
)

plt.title(
    f"Threshold = 120\nIoU={metrics_120['iou']:.3f}"
)

plt.axis("off")


plt.subplot(2, 2, 3)

plt.imshow(
    threshold_160,
    cmap="gray"
)

plt.title(
    f"Threshold = 160\nIoU={metrics_160['iou']:.3f}"
)

plt.axis("off")


plt.subplot(2, 2, 4)

plt.imshow(
    threshold_200,
    cmap="gray"
)

plt.title(
    f"Threshold = 200\nIoU={metrics_200['iou']:.3f}"
)

plt.axis("off")


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 15 — COMPARE ADAPTIVE AND OTSU
# ======================================================================

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.imshow(
    adaptive_mask,
    cmap="gray"
)

plt.title(
    "Adaptive Threshold\n"
    f"IoU={metrics_adaptive['iou']:.3f}"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    otsu_mask,
    cmap="gray"
)

plt.title(
    f"Otsu Threshold={otsu_t}\n"
    f"IoU={metrics_otsu['iou']:.3f}"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    ground_truth,
    cmap="gray"
)

plt.title("Ground Truth")
plt.axis("off")


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 16 — VISUALIZE LOCAL THRESHOLD MAP
# ======================================================================
#
# Unlike global thresholding, adaptive thresholding generates
# a different threshold for each pixel.

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)

plt.imshow(
    local_thresholds,
    cmap="gray"
)

plt.title("Local Threshold Map")
plt.colorbar()
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    adaptive_mask,
    cmap="gray"
)

plt.title("Adaptive Binary Mask")
plt.axis("off")


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 17 — OTSU VARIANCE CURVE
# ======================================================================
#
# The best threshold corresponds to the maximum between-class
# variance.

plt.figure(figsize=(10, 5))

plt.plot(
    np.arange(256),
    otsu_variances
)

plt.axvline(
    otsu_t,
    linestyle="--",
    label=f"Otsu threshold = {otsu_t}"
)

plt.title(
    "Otsu Between-Class Variance"
)

plt.xlabel("Threshold")
plt.ylabel("Between-Class Variance")
plt.xlim(0, 255)
plt.legend()

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 18 — CONFUSION MAP
# ======================================================================
#
# We visualize:
#
#     True Positive
#     True Negative
#     False Positive
#     False Negative
#
# without assigning a specific color scheme, so that the underlying
# categories remain explicit in the numerical representation.

prediction = otsu_mask

TP_mask = prediction & ground_truth
TN_mask = (~prediction) & (~ground_truth)
FP_mask = prediction & (~ground_truth)
FN_mask = (~prediction) & ground_truth

confusion_map = np.zeros(
    image.shape,
    dtype=np.uint8
)

# Numeric labels:
#
#     0 = true negative
#     1 = true positive
#     2 = false positive
#     3 = false negative

confusion_map[TP_mask] = 1
confusion_map[FP_mask] = 2
confusion_map[FN_mask] = 3


plt.figure(figsize=(8, 6))

plt.imshow(
    confusion_map,
    cmap="viridis",
    vmin=0,
    vmax=3
)

plt.title(
    "Otsu Segmentation Error Map\n"
    "0=TN, 1=TP, 2=FP, 3=FN"
)

plt.colorbar(
    ticks=[0, 1, 2, 3]
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 19 — SIMPLE IoU VERIFICATION
# ======================================================================
#
# We manually calculate IoU again:
#
#     intersection = prediction ∩ ground_truth
#
#     union = prediction ∪ ground_truth
#
#     IoU = intersection / union

intersection = np.sum(
    prediction & ground_truth
)

union = np.sum(
    prediction | ground_truth
)

manual_iou = (
    intersection / union
    if union > 0
    else 0.0
)


print("\nMANUAL IoU VERIFICATION")
print("-" * 72)

print(
    "Intersection:",
    intersection
)

print(
    "Union:",
    union
)

print(
    "Manual IoU:",
    manual_iou
)

print(
    "Metric IoU:",
    metrics_otsu["iou"]
)


assert np.isclose(
    manual_iou,
    metrics_otsu["iou"]
)

print("IoU verification passed.")


# ======================================================================
# STEP 20 — REAL-WORLD INTERPRETATION
# ======================================================================
#
# A simple waste-sorting perception pipeline might look like:
#
#        CAMERA
#          |
#          v
#       RGB FRAME
#          |
#          v
#       GRAYSCALE
#          |
#          v
#     THRESHOLDING
#          |
#          v
#      BINARY MASK
#          |
#          v
#     MORPHOLOGY
#          |
#          v
#      COMPONENTS
#          |
#          v
#      OBJECT REGION
#          |
#          v
#      OBJECT CENTER
#          |
#          v
#       ROBOT ARM
#
# For a controlled industrial environment, this classical approach
# can sometimes work surprisingly well.
#
# Example:
#
#     black conveyor
#           +
#     brightly colored object
#
# may be very easy to segment.
#
# But for mixed household waste:
#
#     plastic
#     paper
#     metal
#     transparent objects
#     food
#     shadows
#     overlapping objects
#
# thresholding alone becomes insufficient.
#
# That is why the course gradually moves from:
#
#     classical CV
#
# toward:
#
#     CNNs
#     object detectors
#     segmentation networks
#     transformers
#
# The important point is NOT that thresholding is obsolete.
#
# The important point is understanding exactly what simpler
# perception algorithms can and cannot do.


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 72)
print("LESSON 06 COMPLETE")
print("=" * 72)

print(
    """
You learned:

1. Segmentation separates an image into regions.
2. A binary mask represents foreground and background.
3. Global thresholding uses one threshold for the entire image.
4. Global thresholding can fail under uneven illumination.
5. Adaptive thresholding calculates a local threshold.
6. Otsu automatically selects a global threshold.
7. Otsu maximizes between-class variance.
8. IoU measures overlap between predicted and true masks.
9. Dice is another important segmentation metric.
10. TP, TN, FP and FN explain segmentation errors.

The progression is:

    PIXELS
      ↓
    HISTOGRAM
      ↓
    THRESHOLD
      ↓
    BINARY MASK
      ↓
    SEGMENTATION
      ↓
    OBJECT REGION

And now we are ready for the next classical computer vision
building block:

    BINARY MASK
        ↓
    MORPHOLOGICAL OPERATIONS
        ↓
    CLEAN OBJECT MASK

NEXT:
LESSON 07 — MORPHOLOGICAL IMAGE PROCESSING
"""
)