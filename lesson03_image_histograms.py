"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 03 — IMAGE HISTOGRAMS AND INTENSITY DISTRIBUTIONS
======================================================================

COURSE RULE
-----------
This course is implemented from scratch.

NO:
    - pretrained models
    - pretrained weights
    - pretrained feature extractors
    - black-box implementations of the main learning algorithm

This lesson uses:
    - NumPy
    - Matplotlib

----------------------------------------------------------------------
WHAT YOU LEARN
----------------------------------------------------------------------

STEP 1 — What is an image histogram?
    A histogram describes how frequently each intensity value
    occurs in an image.

    For an 8-bit grayscale image:

        intensity ∈ {0, 1, 2, ..., 255}

    The histogram is:

        H(k) = number of pixels whose intensity is k

STEP 2 — Histogram mathematics

    Given an image I with N pixels:

        H(k) = Σ 1[I(x) = k]

    where:

        1[condition] = 1 if condition is true
                       0 otherwise

    Therefore:

        Σ H(k) = N

STEP 3 — Brightness

    A bright image tends to have its histogram shifted toward
    higher intensity values.

    A dark image tends to have its histogram shifted toward
    lower intensity values.

STEP 4 — Contrast

    Contrast describes how spread out the intensity values are.

    Low contrast:
        values concentrated in a narrow range.

    High contrast:
        values distributed over a wider range.

STEP 5 — Mean intensity

    The mean intensity is:

        μ = (1/N) Σ I(x)

    It provides a simple estimate of overall brightness.

STEP 6 — Variance and standard deviation

    Variance:

        σ² = (1/N) Σ (I(x) - μ)²

    Standard deviation:

        σ = sqrt(σ²)

    Larger standard deviation generally means greater intensity
    variation.

STEP 7 — Histogram calculation from scratch

    We will manually count intensity values.

    We will NOT use:

        cv2.calcHist()

    The purpose is to understand the underlying mathematics.

STEP 8 — Cumulative Distribution Function

    Define:

        C(k) = Σ H(i),  i = 0 ... k

    The normalized CDF is:

        C_norm(k) = C(k) / N

    The CDF will become important for histogram equalization.

STEP 9 — Histogram equalization

    The basic idea:

        input intensity
              ↓
        histogram
              ↓
        cumulative distribution
              ↓
        new intensity

    A simplified 8-bit transformation is:

        T(k) = floor(255 × CDF(k))

    In a practical implementation we also handle the minimum
    non-zero CDF so that the output uses the available intensity
    range more effectively.

STEP 10 — Why histograms matter in robotics

    For an OpenArm V2 waste-sorting robot, lighting can change:

        bright room
        dark room
        shadows
        reflections
        sunlight
        artificial lighting

    A perception system must understand how image intensity
    changes before attempting more complex vision algorithms.

----------------------------------------------------------------------
IMPORTANT CONCEPT
----------------------------------------------------------------------

A histogram DOES NOT preserve spatial information.

Consider:

    Image A:
        object on the left

    Image B:
        same object on the right

They can have nearly identical histograms.

Therefore:

    Histogram → tells us about intensity distribution

but:

    Histogram → does NOT tell us where pixels are located.

This distinction becomes very important later.

----------------------------------------------------------------------
REAL-WORLD ROBOTICS EXAMPLE
----------------------------------------------------------------------

Suppose an OpenArm waste-sorting robot sees the same plastic cup
under two lighting conditions.

Image A:

    strong illumination

Image B:

    weak illumination

The object itself has not changed, but the pixel intensities have.

A histogram lets us quantify this difference.

Later, histogram-based preprocessing can help make perception
more robust to certain illumination changes.

However:

    Histogram equalization is NOT a complete solution to
    lighting variation.

Modern systems may additionally use:

    - normalization
    - augmentation
    - adaptive preprocessing
    - learned representations
    - exposure control
    - HDR imaging

We study the classical mathematical foundation first.

----------------------------------------------------------------------
LESSON EXPERIMENTS
----------------------------------------------------------------------

We will create:

    1. Dark image
    2. Bright image
    3. Low-contrast image
    4. High-contrast image
    5. A synthetic grayscale image
    6. Manual histogram
    7. CDF
    8. Histogram-equalized image

Then we compare their statistics.

----------------------------------------------------------------------
TESTS
----------------------------------------------------------------------

The program verifies:

    - histogram has 256 bins
    - histogram count equals pixel count
    - CDF is monotonically increasing
    - final CDF equals number of pixels
    - normalized CDF ends at 1
    - histogram equalization stays in [0,255]

----------------------------------------------------------------------
EXERCISES
----------------------------------------------------------------------

EXERCISE 1
----------
Create a very dark image and a very bright image.

Compare their histograms.


EXERCISE 2
----------
Create an image where all pixels have intensity 128.

What does the histogram look like?

What happens to the standard deviation?


EXERCISE 3
----------
Create an image containing only four intensities:

    0
    64
    128
    255

Plot its histogram.


EXERCISE 4
----------
Implement histogram calculation using a Python loop.

Then compare it with the NumPy implementation.

This is intentionally inefficient.

The goal is to understand the algorithm.


EXERCISE 5
----------
Implement histogram equalization without using a library
histogram function.


EXERCISE 6
----------
Create two images with identical histograms but different
spatial arrangements.

Verify that their histograms are equal even though the images
look completely different.


EXERCISE 7
----------
Calculate:

    mean
    variance
    standard deviation

directly from the histogram instead of directly from the image.

Hint:

    μ = Σ k H(k) / N

----------------------------------------------------------------------
NEXT LESSON
----------------------------------------------------------------------

LESSON 04 — IMAGE FILTERING AND CONVOLUTION

We will implement:

    - kernels
    - convolution
    - averaging filters
    - Gaussian filtering
    - edge-preserving concepts
    - padding
    - stride
    - image smoothing

This will be one of the most important mathematical foundations
for both classical computer vision and CNNs.

======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# STEP 1 — CREATE A SYNTHETIC GRAYSCALE IMAGE
# ======================================================================

HEIGHT = 256
WIDTH = 384

image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

# Dark background
image[:, :] = 35

# Medium-gray rectangle
image[50:150, 50:180] = 100

# Bright rectangle
image[80:210, 210:350] = 220

# Small bright square
image[170:220, 80:140] = 255


# ======================================================================
# STEP 2 — MANUAL HISTOGRAM
# ======================================================================
#
# For every possible intensity k:
#
#     H(k) = number of pixels having intensity k
#
# Since this is an 8-bit image:
#
#     k = 0 ... 255
#
# We intentionally implement the counting ourselves.
#
# We do NOT use cv2.calcHist().

def histogram_from_scratch(gray_image):
    """
    Calculate an 8-bit grayscale histogram manually.

    Input:
        gray_image : H x W uint8 array

    Output:
        histogram : array with 256 bins
    """

    histogram = np.zeros(256, dtype=np.int64)

    flat = gray_image.reshape(-1)

    for pixel in flat:
        histogram[int(pixel)] += 1

    return histogram


histogram = histogram_from_scratch(image)


# ======================================================================
# STEP 3 — NUMPY HISTOGRAM FOR VERIFICATION
# ======================================================================
#
# NumPy provides bincount().
#
# We use it only as a numerical verification tool.
#
# The actual lesson implementation above is manual.

numpy_histogram = np.bincount(
    image.reshape(-1),
    minlength=256
)

assert np.array_equal(histogram, numpy_histogram)


# ======================================================================
# STEP 4 — IMAGE STATISTICS
# ======================================================================

num_pixels = image.size

mean_direct = image.mean()
std_direct = image.std()

# Mean calculated from histogram:
#
#     μ = Σ k H(k) / N

intensities = np.arange(256)

mean_from_histogram = (
    np.sum(intensities * histogram) / num_pixels
)

# Variance calculated from histogram:
#
#     σ² = Σ H(k)(k - μ)² / N

variance_from_histogram = (
    np.sum(
        histogram *
        (intensities - mean_from_histogram) ** 2
    )
    / num_pixels
)

std_from_histogram = np.sqrt(variance_from_histogram)


# ======================================================================
# STEP 5 — CUMULATIVE DISTRIBUTION FUNCTION
# ======================================================================
#
# CDF:
#
#     C(k) = Σ H(i)
#
# for i <= k
#
# We implement it manually rather than simply calling np.cumsum()
# so that the mathematical operation is explicit.

cdf = np.zeros(256, dtype=np.int64)

running_total = 0

for k in range(256):
    running_total += histogram[k]
    cdf[k] = running_total


# Normalized CDF:
#
#     CDF_norm(k) = C(k) / N

cdf_normalized = cdf.astype(np.float64) / num_pixels


# ======================================================================
# STEP 6 — HISTOGRAM EQUALIZATION
# ======================================================================
#
# A common transformation is:
#
#     T(k) = floor(255 × CDF_norm(k))
#
# However, if the histogram starts at a nonzero intensity,
# directly using the CDF wastes part of the available output range.
#
# We therefore use:
#
#     CDF_min = first non-zero CDF value
#
# and transform using:
#
#     T(k) =
#
#       floor(
#           (CDF(k) - CDF_min)
#           /
#           (N - CDF_min)
#           × 255
#       )
#
# for values where the denominator is nonzero.

nonzero_indices = np.where(histogram > 0)[0]

cdf_min = cdf[nonzero_indices[0]]

equalization_lookup = np.zeros(256, dtype=np.uint8)

denominator = num_pixels - cdf_min

if denominator > 0:

    for k in range(256):

        value = (
            (cdf[k] - cdf_min)
            / denominator
            * 255.0
        )

        value = np.clip(value, 0, 255)

        equalization_lookup[k] = np.uint8(np.floor(value))

else:

    # Special case:
    # every pixel has exactly the same intensity.

    equalization_lookup[:] = 0


# Apply lookup table to every pixel.
equalized_image = equalization_lookup[image]


# ======================================================================
# STEP 7 — HISTOGRAM OF EQUALIZED IMAGE
# ======================================================================

equalized_histogram = histogram_from_scratch(
    equalized_image
)


# ======================================================================
# STEP 8 — PRINT RESULTS
# ======================================================================

print("=" * 72)
print("LESSON 03 — IMAGE HISTOGRAMS AND INTENSITY DISTRIBUTIONS")
print("=" * 72)

print("\nIMAGE INFORMATION")
print("-" * 72)

print("Shape       :", image.shape)
print("Pixels      :", num_pixels)
print("Data type   :", image.dtype)
print("Min intensity:", image.min())
print("Max intensity:", image.max())


print("\nIMAGE STATISTICS")
print("-" * 72)

print("Mean from image     :", mean_direct)
print("Mean from histogram :", mean_from_histogram)

print("Std from image      :", std_direct)
print("Std from histogram  :", std_from_histogram)

print(
    "Variance from histogram:",
    variance_from_histogram
)


print("\nHISTOGRAM INFORMATION")
print("-" * 72)

print("Number of bins:", len(histogram))
print("Histogram sum :", histogram.sum())
print("Pixel count   :", num_pixels)

print("\nNon-zero histogram bins:")

nonzero_bins = np.where(histogram > 0)[0]

for k in nonzero_bins:
    print(
        f"Intensity {k:3d} -> "
        f"{histogram[k]:7d} pixels"
    )


print("\nCDF INFORMATION")
print("-" * 72)

print("Final CDF value       :", cdf[-1])
print("Number of pixels      :", num_pixels)
print("Final normalized CDF  :", cdf_normalized[-1])

print("Minimum non-zero CDF :", cdf_min)


# ======================================================================
# STEP 9 — TESTS
# ======================================================================

print("\nRUNNING TESTS")
print("-" * 72)

# Histogram must have one bin for each uint8 intensity.
assert histogram.shape == (256,)

# Every pixel must belong to exactly one histogram bin.
assert histogram.sum() == image.size

# Histogram calculated from scratch must match NumPy verification.
assert np.array_equal(histogram, numpy_histogram)

# CDF must be monotonically increasing.
assert np.all(np.diff(cdf) >= 0)

# Final CDF must equal total number of pixels.
assert cdf[-1] == image.size

# Normalized CDF must finish at one.
assert np.isclose(cdf_normalized[-1], 1.0)

# Mean calculated two different ways must agree.
assert np.isclose(mean_direct, mean_from_histogram)

# Standard deviation must agree.
assert np.isclose(std_direct, std_from_histogram)

# Equalized image must remain a valid uint8 image.
assert equalized_image.dtype == np.uint8

# Equalized intensity range must be valid.
assert equalized_image.min() >= 0
assert equalized_image.max() <= 255

# Equalized histogram must contain the same total number of pixels.
assert equalized_histogram.sum() == image.size

print("All tests passed.")


# ======================================================================
# STEP 10 — CREATE COMPARISON IMAGES
# ======================================================================
#
# These images demonstrate the difference between:
#
#     brightness
#     contrast
#
# and show why histogram analysis is useful.

dark_image = np.clip(
    image.astype(np.float32) * 0.35,
    0,
    255
).astype(np.uint8)

bright_image = np.clip(
    image.astype(np.float32) * 1.35,
    0,
    255
).astype(np.uint8)

low_contrast = np.clip(
    100 + (image.astype(np.float32) - 100) * 0.25,
    0,
    255
).astype(np.uint8)

high_contrast = np.clip(
    128 + (image.astype(np.float32) - 128) * 2.0,
    0,
    255
).astype(np.uint8)


# ======================================================================
# STEP 11 — HISTOGRAMS OF COMPARISON IMAGES
# ======================================================================

dark_histogram = histogram_from_scratch(dark_image)
bright_histogram = histogram_from_scratch(bright_image)
low_contrast_histogram = histogram_from_scratch(low_contrast)
high_contrast_histogram = histogram_from_scratch(high_contrast)


# ======================================================================
# STEP 12 — VISUALIZATION
# ======================================================================

fig, axes = plt.subplots(4, 2, figsize=(14, 16))


# ----------------------------------------------------------------------
# Original image
# ----------------------------------------------------------------------

axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")


# Original histogram
axes[0, 1].plot(histogram)
axes[0, 1].set_title("Original Histogram")
axes[0, 1].set_xlabel("Intensity")
axes[0, 1].set_ylabel("Pixel Count")
axes[0, 1].set_xlim(0, 255)


# ----------------------------------------------------------------------
# Dark image
# ----------------------------------------------------------------------

axes[1, 0].imshow(dark_image, cmap="gray", vmin=0, vmax=255)
axes[1, 0].set_title("Dark Image")
axes[1, 0].axis("off")


axes[1, 1].plot(dark_histogram)
axes[1, 1].set_title("Dark Image Histogram")
axes[1, 1].set_xlabel("Intensity")
axes[1, 1].set_ylabel("Pixel Count")
axes[1, 1].set_xlim(0, 255)


# ----------------------------------------------------------------------
# Bright image
# ----------------------------------------------------------------------

axes[2, 0].imshow(bright_image, cmap="gray", vmin=0, vmax=255)
axes[2, 0].set_title("Bright Image")
axes[2, 0].axis("off")


axes[2, 1].plot(bright_histogram)
axes[2, 1].set_title("Bright Image Histogram")
axes[2, 1].set_xlabel("Intensity")
axes[2, 1].set_ylabel("Pixel Count")
axes[2, 1].set_xlim(0, 255)


# ----------------------------------------------------------------------
# Equalized image
# ----------------------------------------------------------------------

axes[3, 0].imshow(equalized_image, cmap="gray", vmin=0, vmax=255)
axes[3, 0].set_title("Histogram Equalized")
axes[3, 0].axis("off")


axes[3, 1].plot(equalized_histogram)
axes[3, 1].set_title("Equalized Histogram")
axes[3, 1].set_xlabel("Intensity")
axes[3, 1].set_ylabel("Pixel Count")
axes[3, 1].set_xlim(0, 255)


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 13 — CDF VISUALIZATION
# ======================================================================

plt.figure(figsize=(10, 5))

plt.plot(
    intensities,
    cdf_normalized
)

plt.title("Cumulative Distribution Function")
plt.xlabel("Intensity")
plt.ylabel("Normalized CDF")
plt.xlim(0, 255)
plt.ylim(0, 1.05)
plt.grid(True)

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 14 — CONTRAST COMPARISON
# ======================================================================

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.imshow(
    low_contrast,
    cmap="gray",
    vmin=0,
    vmax=255
)
plt.title("Low Contrast")
plt.axis("off")


plt.subplot(2, 2, 2)
plt.plot(low_contrast_histogram)
plt.title("Low Contrast Histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixel Count")
plt.xlim(0, 255)


plt.subplot(2, 2, 3)
plt.imshow(
    high_contrast,
    cmap="gray",
    vmin=0,
    vmax=255
)
plt.title("High Contrast")
plt.axis("off")


plt.subplot(2, 2, 4)
plt.plot(high_contrast_histogram)
plt.title("High Contrast Histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixel Count")
plt.xlim(0, 255)

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 15 — HISTOGRAM DOES NOT CONTAIN SPATIAL INFORMATION
# ======================================================================
#
# We now construct two images with exactly the same intensity
# distribution but different spatial arrangements.
#
# Image A:
#     bright pixels form a rectangle.
#
# Image B:
#     the same number of bright pixels are randomly distributed.
#
# Their histograms will be identical even though their spatial
# structures are completely different.

spatial_a = np.zeros((200, 300), dtype=np.uint8)

spatial_a[50:150, 100:200] = 255


spatial_b = np.zeros((200, 300), dtype=np.uint8)

number_of_bright_pixels = np.sum(spatial_a == 255)

rng = np.random.default_rng(42)

random_indices = rng.choice(
    spatial_b.size,
    size=number_of_bright_pixels,
    replace=False
)

spatial_b.reshape(-1)[random_indices] = 255


hist_a = histogram_from_scratch(spatial_a)
hist_b = histogram_from_scratch(spatial_b)


print("\nSPATIAL INFORMATION EXPERIMENT")
print("-" * 72)

print("Histograms identical:", np.array_equal(hist_a, hist_b))

assert np.array_equal(hist_a, hist_b)


# ======================================================================
# STEP 16 — VISUALIZE THE SPATIAL EXPERIMENT
# ======================================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.imshow(
    spatial_a,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Image A — Structured")
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    spatial_b,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title("Image B — Random")
plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 17 — REAL-WORLD INTERPRETATION
# ======================================================================
#
# Consider an OpenArm V2 waste-sorting robot.
#
# Camera image:
#
#                  RGB camera
#                      |
#                      v
#                pixel values
#                      |
#                      v
#                 grayscale
#                      |
#                      v
#                  histogram
#                      |
#              +-------+-------+
#              |               |
#          brightness       contrast
#              |               |
#              +-------+-------+
#                      |
#                      v
#                preprocessing
#                      |
#                      v
#                perception
#
# Example:
#
# A plastic bottle may be illuminated by a strong overhead light.
#
# Its appearance can contain:
#
#     bright highlights
#     dark shadows
#     reflective regions
#
# The histogram can reveal that a large percentage of pixels
# are concentrated toward the bright end.
#
# But remember:
#
# Histogram information alone cannot tell the robot:
#
#     "The bottle is at x=320, y=180."
#
# Spatial information is required for localization.
#
# This distinction prepares us for:
#
#     filtering
#     edges
#     contours
#     feature detection
#     object detection
#     segmentation
#
# later in the course.


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 72)
print("LESSON 03 COMPLETE")
print("=" * 72)

print(
    """
You learned:

1. A histogram represents the distribution of image intensities.
2. H(k) counts how many pixels have intensity k.
3. The sum of all histogram bins equals the number of pixels.
4. Mean intensity can be calculated directly or from the histogram.
5. Standard deviation measures intensity variation.
6. Brightness changes shift the histogram.
7. Contrast changes the histogram's spread.
8. The CDF is the cumulative histogram.
9. Histogram equalization uses the CDF to redistribute intensities.
10. A histogram contains intensity information but not spatial information.

The key idea:

    IMAGE
      |
      +---- spatial information
      |
      +---- intensity information
                    |
                    v
                HISTOGRAM

A histogram tells us WHAT intensities exist
and HOW FREQUENTLY they occur.

It does NOT tell us WHERE those pixels are.

NEXT:
LESSON 04 — IMAGE FILTERING AND CONVOLUTION
"""
)