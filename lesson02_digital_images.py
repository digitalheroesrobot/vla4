"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 02 — DIGITAL IMAGES AND PIXEL REPRESENTATION
======================================================================

COURSE RULE
-----------
This entire course is built from scratch.

We do NOT use:
    - pretrained models
    - pretrained weights
    - pretrained feature extractors
    - black-box CV algorithms as the main implementation

We implement the mathematics ourselves whenever practical.

This lesson uses:
    - NumPy
    - Matplotlib

No OpenCV is required for the main lesson.

----------------------------------------------------------------------
WHAT YOU LEARN
----------------------------------------------------------------------

STEP 1 — What is a digital image?
    Understand that an image is numerical data.

STEP 2 — Pixels and coordinates
    Understand:
        image[y, x]

    Important:
        x = horizontal position = column
        y = vertical position   = row

    NumPy uses:
        array[row, column]

STEP 3 — Grayscale images
    Represent an image as:

        I ∈ R^(H × W)

    where:
        H = image height
        W = image width

STEP 4 — RGB images
    Represent a color image as:

        I ∈ R^(H × W × 3)

    where the three channels are:

        R = Red
        G = Green
        B = Blue

STEP 5 — Pixel intensity
    Typical 8-bit images use:

        0   → black
        255 → white

    Intermediate values represent different brightness levels.

STEP 6 — Data types
    Understand the difference between:

        uint8
        float32
        float64

    This becomes extremely important later when implementing:
        convolution
        gradients
        neural networks
        image normalization

STEP 7 — Image normalization
    Convert:

        [0, 255]

    into:

        [0, 1]

    using:

        normalized = image / 255

STEP 8 — RGB → grayscale
    Implement grayscale conversion manually.

    A common luminance approximation is:

        Gray =
            0.299 R +
            0.587 G +
            0.114 B

    The coefficients reflect human visual sensitivity.

STEP 9 — Channel manipulation
    Learn how to:
        - extract channels
        - modify channels
        - recombine channels

STEP 10 — Basic image statistics
    Calculate:
        - minimum
        - maximum
        - mean
        - standard deviation

    These statistics will become useful for:
        normalization
        preprocessing
        debugging
        dataset analysis

----------------------------------------------------------------------
MATHEMATICAL FOUNDATION
----------------------------------------------------------------------

A grayscale image can be represented as:

        I(y, x)

where:

        x ∈ {0, ..., W-1}
        y ∈ {0, ..., H-1}

For an RGB image:

        I(y, x, c)

where:

        c ∈ {R, G, B}

Therefore:

        I ∈ R^(H × W × 3)

For an 8-bit RGB image:

        I(y, x, c) ∈ {0, 1, ..., 255}

For normalized floating-point representation:

        I_norm = I / 255

and:

        I_norm ∈ [0, 1]

----------------------------------------------------------------------
IMPORTANT COORDINATE CONCEPT
----------------------------------------------------------------------

Computer vision normally uses:

        image[y, x]

NOT:

        image[x, y]

Example:

        image[20, 50]

means:

        y = 20
        x = 50

This is one of the most common sources of bugs in computer vision.

----------------------------------------------------------------------
REAL-WORLD ROBOTICS EXAMPLE
----------------------------------------------------------------------

Imagine an OpenArm V2 waste-sorting robot.

A camera produces:

        RGB image
             ↓
        H × W × 3
             ↓
        pixel values
             ↓
        visual processing
             ↓
        object detection
             ↓
        object location
             ↓
        robot action

For example, the robot may receive:

        640 × 480 × 3

This means:

        width  = 640 pixels
        height = 480 pixels
        channels = 3

Total pixel-channel values:

        640 × 480 × 3
        = 921,600 values

If each value uses uint8:

        921,600 bytes
        ≈ 0.88 MB

for one uncompressed RGB frame.

At 30 FPS:

        921,600 × 30
        = 27,648,000 bytes/sec

or approximately:

        26.4 MB/sec

before compression and transport overhead.

This is why understanding image representation is important
for real robot systems.

----------------------------------------------------------------------
EXPERIMENTS
----------------------------------------------------------------------

We will:

1. Create a synthetic RGB image.
2. Inspect its dimensions.
3. Inspect individual pixels.
4. Extract R/G/B channels.
5. Modify a channel.
6. Convert RGB → grayscale.
7. Normalize the image.
8. Calculate image statistics.
9. Verify numerical properties.
10. Visualize everything.

----------------------------------------------------------------------
TESTS
----------------------------------------------------------------------

The script verifies:

    - correct image dimensions
    - correct data type
    - valid uint8 range
    - correct normalized range
    - RGB → grayscale dimensions
    - channel extraction
    - grayscale conversion
    - coordinate access

----------------------------------------------------------------------
EXERCISES
----------------------------------------------------------------------

EXERCISE 1
----------
Change the image resolution.

Try:

    128 × 192
    480 × 640
    720 × 1280

Observe how the array shape changes.


EXERCISE 2
----------
Change the colors of the synthetic objects.


EXERCISE 3
----------
Create a fourth object using a different color.


EXERCISE 4
----------
Calculate the percentage of pixels belonging to each
dominant color region.


EXERCISE 5
----------
Implement grayscale conversion using only the red channel:

    Gray = R

Compare it with luminance-based grayscale.


EXERCISE 6
----------
Create an image with a horizontal brightness gradient.

For example:

    left side  → black
    right side → white

Do not use OpenCV.

Use NumPy only.


EXERCISE 7
----------
Create a vertical gradient.

Then combine horizontal and vertical gradients.


EXERCISE 8
----------
Calculate the histogram of grayscale intensity values
manually using NumPy.

Do not use cv2.calcHist.


----------------------------------------------------------------------
NEXT LESSON
----------------------------------------------------------------------

LESSON 03 — IMAGE HISTOGRAMS AND INTENSITY DISTRIBUTIONS

We will learn:

    - histogram mathematics
    - intensity distributions
    - contrast
    - brightness
    - histogram visualization
    - histogram equalization from scratch
    - cumulative distribution functions
    - why histograms matter for computer vision

======================================================================
"""


import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# STEP 1 — CREATE A SYNTHETIC RGB IMAGE
# ======================================================================

HEIGHT = 256
WIDTH = 384

# RGB image:
#
#     shape = (height, width, channels)
#
# channels:
#     0 = Red
#     1 = Green
#     2 = Blue

image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


# ======================================================================
# STEP 2 — DRAW SIMPLE OBJECTS USING NUMPY
# ======================================================================

# Red rectangle
image[50:120, 40:150, 0] = 255

# Green rectangle
image[140:220, 60:180, 1] = 255

# Blue rectangle
image[70:180, 220:330, 2] = 255


# ======================================================================
# STEP 3 — CREATE A YELLOW OBJECT
# ======================================================================
#
# Yellow in RGB is:
#
#     R = 255
#     G = 255
#     B = 0
#
# This demonstrates that a color is simply a combination
# of channel intensities.

image[150:220, 230:330, 0] = 255
image[150:220, 230:330, 1] = 255


# ======================================================================
# STEP 4 — INSPECT THE IMAGE
# ======================================================================

print("=" * 72)
print("LESSON 02 — DIGITAL IMAGES AND PIXEL REPRESENTATION")
print("=" * 72)

print("\nIMAGE INFORMATION")
print("-" * 72)

print("Shape :", image.shape)
print("Height:", image.shape[0])
print("Width :", image.shape[1])
print("Channels:", image.shape[2])
print("Dtype :", image.dtype)
print("Min   :", image.min())
print("Max   :", image.max())
print("Mean  :", image.mean())
print("Std   :", image.std())


# ======================================================================
# STEP 5 — PIXEL COORDINATES
# ======================================================================
#
# Remember:
#
#     image[y, x]
#
# For RGB:
#
#     image[y, x, channel]
#
# channel:
#
#     0 = R
#     1 = G
#     2 = B

print("\nPIXEL ACCESS")
print("-" * 72)

pixel_red = image[80, 80]

print("Pixel at (x=80, y=80):")
print(pixel_red)

print("\nIndividual channels:")

print("Red  :", image[80, 80, 0])
print("Green:", image[80, 80, 1])
print("Blue :", image[80, 80, 2])


# ======================================================================
# STEP 6 — EXTRACT RGB CHANNELS
# ======================================================================

red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

print("\nCHANNEL SHAPES")
print("-" * 72)

print("Red channel  :", red_channel.shape)
print("Green channel:", green_channel.shape)
print("Blue channel :", blue_channel.shape)


# ======================================================================
# STEP 7 — RGB → GRAYSCALE
# ======================================================================
#
# Human vision does not perceive R, G and B equally.
#
# A commonly used luminance approximation is:
#
#     Gray =
#         0.299 R +
#         0.587 G +
#         0.114 B
#
# Notice that floating-point arithmetic is used here.
#
# Therefore we explicitly convert the channels to float32.

rgb_float = image.astype(np.float32)

gray = (
    0.299 * rgb_float[:, :, 0]
    + 0.587 * rgb_float[:, :, 1]
    + 0.114 * rgb_float[:, :, 2]
)

print("\nGRAYSCALE IMAGE")
print("-" * 72)

print("Grayscale shape:", gray.shape)
print("Grayscale dtype:", gray.dtype)
print("Grayscale min  :", gray.min())
print("Grayscale max  :", gray.max())


# ======================================================================
# STEP 8 — NORMALIZATION
# ======================================================================
#
# Original range:
#
#     [0, 255]
#
# Normalized range:
#
#     [0, 1]
#
# Formula:
#
#     I_normalized = I / 255

normalized = rgb_float / 255.0

print("\nNORMALIZATION")
print("-" * 72)

print("Normalized dtype:", normalized.dtype)
print("Normalized min  :", normalized.min())
print("Normalized max  :", normalized.max())


# ======================================================================
# STEP 9 — MODIFY A CHANNEL
# ======================================================================
#
# Create a copy so the original image remains unchanged.
#
# Here we increase the red channel in the upper-left region.

modified_image = image.copy()

modified_image[20:100, 200:300, 0] = 255

print("\nCHANNEL MODIFICATION")
print("-" * 72)

print("Original image preserved:", image.shape == modified_image.shape)


# ======================================================================
# STEP 10 — BASIC IMAGE STATISTICS
# ======================================================================

print("\nIMAGE STATISTICS")
print("-" * 72)

print("RGB mean:", image.mean())
print("RGB std :", image.std())

print("\nPer-channel statistics:")

print(
    "Red   -> mean:",
    red_channel.mean(),
    "std:",
    red_channel.std()
)

print(
    "Green -> mean:",
    green_channel.mean(),
    "std:",
    green_channel.std()
)

print(
    "Blue  -> mean:",
    blue_channel.mean(),
    "std:",
    blue_channel.std()
)


# ======================================================================
# STEP 11 — MANUAL NUMERICAL TESTS
# ======================================================================

print("\nRUNNING TESTS")
print("-" * 72)

# Test image dimensions
assert image.ndim == 3
assert image.shape == (HEIGHT, WIDTH, 3)

# Test datatype
assert image.dtype == np.uint8

# Test uint8 range
assert image.min() >= 0
assert image.max() <= 255

# Test channel dimensions
assert red_channel.shape == (HEIGHT, WIDTH)
assert green_channel.shape == (HEIGHT, WIDTH)
assert blue_channel.shape == (HEIGHT, WIDTH)

# Test grayscale dimensions
assert gray.shape == (HEIGHT, WIDTH)

# Test grayscale range
assert gray.min() >= 0
assert gray.max() <= 255

# Test normalized range
assert normalized.min() >= 0.0
assert normalized.max() <= 1.0

# Test a known red pixel
assert image[80, 80, 0] == 255

# Test the red rectangle's green and blue values
assert image[80, 80, 1] == 0
assert image[80, 80, 2] == 0

print("All tests passed.")


# ======================================================================
# STEP 12 — VISUALIZATION
# ======================================================================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))


# Original RGB image
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original RGB")
axes[0, 0].axis("off")


# Red channel
axes[0, 1].imshow(red_channel, cmap="gray")
axes[0, 1].set_title("Red Channel")
axes[0, 1].axis("off")


# Green channel
axes[0, 2].imshow(green_channel, cmap="gray")
axes[0, 2].set_title("Green Channel")
axes[0, 2].axis("off")


# Blue channel
axes[0, 3].imshow(blue_channel, cmap="gray")
axes[0, 3].set_title("Blue Channel")
axes[0, 3].axis("off")


# Grayscale
axes[1, 0].imshow(gray, cmap="gray")
axes[1, 0].set_title("Grayscale")
axes[1, 0].axis("off")


# Normalized image
axes[1, 1].imshow(normalized)
axes[1, 1].set_title("Normalized RGB")
axes[1, 1].axis("off")


# Modified image
axes[1, 2].imshow(modified_image)
axes[1, 2].set_title("Modified Red Channel")
axes[1, 2].axis("off")


# Brightness map
axes[1, 3].imshow(gray, cmap="gray", vmin=0, vmax=255)
axes[1, 3].set_title("Pixel Intensity")
axes[1, 3].axis("off")


plt.tight_layout()
plt.show()


# ======================================================================
# STEP 13 — REAL-WORLD INTERPRETATION
# ======================================================================
#
# Imagine that the image comes from an OpenArm V2 waste-sorting robot.
#
# Camera:
#
#     640 × 480 RGB
#
# Each pixel contains:
#
#     [R, G, B]
#
# Example:
#
#     [255, 0, 0]
#
# means strong red.
#
#     [0, 255, 0]
#
# means strong green.
#
#     [0, 0, 255]
#
# means strong blue.
#
#     [255, 255, 255]
#
# means white.
#
#     [0, 0, 0]
#
# means black.
#
# The robot's perception system eventually transforms these
# numerical values into information such as:
#
#     "There is an object here."
#
#     "Its center is approximately here."
#
#     "Its visual appearance is consistent with a cup."
#
#     "The robot should grasp at this location."
#
# The first step toward all of this is understanding pixels.


print("\n" + "=" * 72)
print("LESSON 02 COMPLETE")
print("=" * 72)

print(
    """
You learned:

1. A digital image is numerical data.
2. RGB images have shape H × W × 3.
3. Grayscale images have shape H × W.
4. Pixels are accessed using image[y, x].
5. RGB channels can be extracted independently.
6. RGB can be converted to grayscale mathematically.
7. Images can be normalized from [0,255] to [0,1].
8. Image statistics can be calculated directly with NumPy.
9. Image channels can be modified numerically.
10. Computer vision begins with numerical image representation.

NEXT:
LESSON 03 — IMAGE HISTOGRAMS AND INTENSITY DISTRIBUTIONS
"""
)