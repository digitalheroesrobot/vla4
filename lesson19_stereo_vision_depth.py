"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 19 — STEREO VISION AND DEPTH ESTIMATION
======================================================================

COURSE RULE
-----------
No pretrained models.
No pretrained weights.
No external dataset required.

We generate a synthetic stereo camera scene and implement
the core stereo-matching mathematics using NumPy.

OpenCV is used for practical comparison with StereoBM.

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1 — STEREO CAMERA GEOMETRY
    Understand how two cameras observe the same scene.

STEP 2 — RECTIFIED STEREO IMAGES
    Understand why corresponding points should lie on the same row.

STEP 3 — DISPARITY
    Calculate horizontal displacement between corresponding pixels.

STEP 4 — BLOCK MATCHING
    Implement Sum of Absolute Differences (SAD).

STEP 5 — DISPARITY MAP
    Estimate disparity for every pixel.

STEP 6 — DEPTH ESTIMATION
    Convert disparity into metric depth.

STEP 7 — OPENCV STEREO MATCHING
    Compare our implementation with cv2.StereoBM.

STEP 8 — ROBOTICS APPLICATION
    Estimate object distance for an OpenArm V2 waste-sorting system.

======================================================================
MATHEMATICAL FOUNDATION
======================================================================

1. STEREO CAMERA GEOMETRY
-------------------------

A stereo camera consists of two cameras:

    Left camera                  Right camera
        L                              R
        |                              |
        |<--------- baseline B ------->|

The baseline B is the distance between camera centers.

For a rectified stereo pair:

    Z = (f * B) / d

where:

    Z = depth in meters
    f = focal length in pixels
    B = stereo baseline in meters
    d = disparity in pixels

2. DISPARITY
------------

If an object appears at:

    Left image:  x_left
    Right image: x_right

Then:

    d = x_left - x_right

For a rectified stereo pair, corresponding points have
approximately the same vertical coordinate.

3. DEPTH
--------

    Z = fB / d

Example:

    f = 600 pixels
    B = 0.12 meters
    d = 24 pixels

    Z = (600 * 0.12) / 24

    Z = 3 meters

Important:

    Larger disparity means closer object.
    Smaller disparity means farther object.

4. SUM OF ABSOLUTE DIFFERENCES (SAD)
-----------------------------------

For a pixel and its candidate match:

    SAD(d) = sum |L(x,y) - R(x-d,y)|

We calculate this over a local window.

The best disparity is:

    d_best = argmin_d SAD(d)

5. LIMITATIONS
--------------

Stereo matching becomes difficult when:

    - Surfaces have no texture.
    - Lighting differs between cameras.
    - Objects are reflective.
    - Objects are transparent.
    - Occlusions occur.
    - The camera pair is not rectified.
    - The baseline or camera calibration is inaccurate.

======================================================================
REAL-WORLD ROBOTICS APPLICATION
======================================================================

OPENARM V2 — WASTE SORTING

A stereo camera mounted above a conveyor belt can estimate
the distance of objects from the camera.

Potential pipeline:

    Left RGB image
          |
    Right RGB image
          |
    Stereo matching
          |
    Disparity map
          |
    Depth estimation
          |
    Object detection / segmentation
          |
    3D object localization
          |
    Grasp planning
          |
    ACT policy / robot controller

A stereo camera can help estimate the 3D position of a cup,
bottle, or other waste object.

IMPORTANT:

    Depth estimation alone does not provide a complete grasp pose.

    A real robot also needs:
        - Camera calibration
        - Stereo rectification
        - Camera-to-robot extrinsic calibration
        - Object localization
        - Coordinate-frame transformation

======================================================================
DEPENDENCIES
======================================================================

    pip install numpy matplotlib opencv-python

======================================================================
RUN
======================================================================

    python lesson19_stereo_vision_depth.py

======================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# CONFIGURATION
# =====================================================================

SEED = 42

np.random.seed(SEED)

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320

FOCAL_LENGTH_PX = 600.0
BASELINE_M = 0.12

MAX_DISPARITY = 16
BLOCK_SIZE = 9

EPSILON = 1e-8


# =====================================================================
# STEP 1 — SYNTHETIC SCENE GENERATION
# =====================================================================

def create_texture(height, width, seed=42):
    """
    Generate a textured grayscale image.

    Random texture makes stereo correspondence easier to demonstrate.
    """

    rng = np.random.default_rng(seed)

    image = rng.integers(
        30,
        220,
        size=(height, width),
        dtype=np.uint8
    )

    # Add spatial structure.
    image = cv2.GaussianBlur(
        image,
        (3, 3),
        0.6
    )

    # Add a few visual structures.
    image = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    cv2.rectangle(
        image,
        (30, 35),
        (95, 105),
        (230, 230, 230),
        2
    )

    cv2.circle(
        image,
        (180, 75),
        28,
        (20, 20, 20),
        3
    )

    cv2.rectangle(
        image,
        (210, 145),
        (285, 210),
        (240, 240, 240),
        -1
    )

    cv2.putText(
        image,
        "STEREO",
        (35, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def create_disparity_ground_truth(height, width):
    """
    Create a synthetic disparity map.

    Different regions represent objects at different depths.

    Disparity values:
        Background = 4 pixels
        Object A  = 8 pixels
        Object B  = 12 pixels
    """

    disparity = np.ones(
        (height, width),
        dtype=np.float32
    ) * 4.0

    disparity[35:115, 35:120] = 8.0

    disparity[115:205, 185:285] = 12.0

    disparity[65:145, 145:205] = 8.0

    return disparity


def create_stereo_pair(left_image, disparity_map):
    """
    Generate a right image from the left image.

    For a rectified stereo pair:

        x_left = x_right + disparity

    Thus:

        right(x) = left(x + disparity)

    This is a simplified synthetic renderer.

    It does not model realistic occlusion, lighting changes,
    or lens distortion.
    """

    height, width = left_image.shape

    right_image = np.zeros_like(left_image)

    x_coordinates = np.arange(width)

    for y in range(height):

        d = disparity_map[y].astype(np.int32)

        source_x = np.minimum(
            x_coordinates + d,
            width - 1
        )

        right_image[y] = left_image[y, source_x]

    return right_image


# =====================================================================
# STEP 2 — INTEGRAL IMAGE
# =====================================================================

def integral_image(image):
    """
    Compute a padded summed-area table.

    The integral image allows fast rectangular sum calculations.
    """

    image = np.asarray(
        image,
        dtype=np.float64
    )

    return np.pad(
        image.cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
        mode="constant"
    )


def box_sum_valid(image, radius):
    """
    Sum all square windows of size:

        (2*radius + 1) x (2*radius + 1)

    Returns only locations where the entire window fits.
    """

    integral = integral_image(image)

    y0 = 0
    x0 = 0

    y1 = 2 * radius + 1
    x1 = 2 * radius + 1

    result = (
        integral[y1:, x1:]
        - integral[y0:-y1, x1:]
        - integral[y1:, x0:-x1]
        + integral[y0:-y1, x0:-x1]
    )

    return result


# =====================================================================
# STEP 3 — SUM OF ABSOLUTE DIFFERENCES
# =====================================================================

def sad_cost_volume(
    left_image,
    right_image,
    max_disparity,
    block_size=9
):
    """
    Calculate SAD costs for candidate disparities.

    For each candidate disparity d:

        difference(x,y) =
            abs(left(x,y) - right(x-d,y))

    The local window sum is computed using an integral image.

    Returns:
        cost_volume: H x W x D
    """

    left = left_image.astype(np.float64)

    right = right_image.astype(np.float64)

    height, width = left.shape

    radius = block_size // 2

    cost_volume = np.full(
        (height, width, max_disparity),
        np.inf,
        dtype=np.float64
    )

    for disparity in range(max_disparity):

        if disparity >= width:
            break

        # Compare left[x] with right[x-disparity].
        difference = np.abs(
            left[:, disparity:] -
            right[:, :width - disparity]
        )

        # Calculate local SAD.
        costs = box_sum_valid(
            difference,
            radius
        )

        # Map valid costs back to original left-image coordinates.
        y_start = radius
        y_end = height - radius

        x_start = disparity + radius
        x_end = width - radius

        cost_volume[
            y_start:y_end,
            x_start:x_end,
            disparity
        ] = costs

    return cost_volume


# =====================================================================
# STEP 4 — DISPARITY ESTIMATION
# =====================================================================

def compute_disparity(
    left_image,
    right_image,
    max_disparity=16,
    block_size=9
):
    """
    Estimate disparity using winner-takes-all SAD matching.

    For each pixel:

        disparity = argmin(SAD cost)

    Returns:
        disparity map
        minimum cost map
    """

    cost_volume = sad_cost_volume(
        left_image,
        right_image,
        max_disparity,
        block_size
    )

    disparity = np.argmin(
        cost_volume,
        axis=2
    ).astype(np.float32)

    minimum_cost = np.min(
        cost_volume,
        axis=2
    )

    invalid = ~np.isfinite(minimum_cost)

    disparity[invalid] = 0.0

    return disparity, minimum_cost


# =====================================================================
# STEP 5 — DEPTH ESTIMATION
# =====================================================================

def disparity_to_depth(
    disparity,
    focal_length,
    baseline
):
    """
    Convert disparity into depth in meters.

        Z = f * B / d

    Invalid or zero disparities produce NaN.
    """

    disparity = np.asarray(
        disparity,
        dtype=np.float64
    )

    depth = np.full_like(
        disparity,
        np.nan
    )

    valid = disparity > EPSILON

    depth[valid] = (
        focal_length * baseline /
        disparity[valid]
    )

    return depth


# =====================================================================
# STEP 6 — DEPTH ERROR
# =====================================================================

def calculate_depth_error(estimated, ground_truth):
    """
    Calculate depth errors where both values are valid.
    """

    valid = (
        np.isfinite(estimated) &
        np.isfinite(ground_truth)
    )

    errors = np.abs(
        estimated[valid] -
        ground_truth[valid]
    )

    if len(errors) == 0:
        return np.nan, np.nan

    return (
        float(np.mean(errors)),
        float(np.median(errors))
    )


# =====================================================================
# STEP 7 — OPENCV STEREO MATCHING
# =====================================================================

def compute_opencv_disparity(left_image, right_image):
    """
    Practical OpenCV StereoBM implementation.

    StereoBM requires rectified grayscale stereo images.

    numDisparities must be divisible by 16.
    """

    stereo = cv2.StereoBM_create(
        numDisparities=16,
        blockSize=9
    )

    disparity_fixed = stereo.compute(
        left_image,
        right_image
    )

    # OpenCV StereoBM normally stores disparity scaled by 16.
    disparity = disparity_fixed.astype(
        np.float32
    ) / 16.0

    return disparity


# =====================================================================
# STEP 8 — STEREO RECTIFICATION CONCEPT
# =====================================================================

def draw_stereo_epipolar_lines(left, right):
    """
    Visualize horizontal correspondence rows.

    After stereo rectification, corresponding points should
    lie approximately on the same image row.
    """

    left_color = cv2.cvtColor(
        left,
        cv2.COLOR_GRAY2BGR
    )

    right_color = cv2.cvtColor(
        right,
        cv2.COLOR_GRAY2BGR
    )

    combined = np.hstack([
        left_color,
        right_color
    ])

    for y in range(20, IMAGE_HEIGHT, 40):

        cv2.line(
            combined,
            (0, y),
            (IMAGE_WIDTH * 2, y),
            (0, 255, 0),
            1
        )

    return combined


# =====================================================================
# STEP 9 — NUMERICAL TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)
    print("RUNNING LESSON 19 TESTS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # TEST 1 — Integral image
    # ---------------------------------------------------------------

    test = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float64)

    integral = integral_image(test)

    assert integral.shape == (4, 4)

    assert integral[-1, -1] == 45

    print("Test 1 passed: integral image.")

    # ---------------------------------------------------------------
    # TEST 2 — Box sum
    # ---------------------------------------------------------------

    box_test = np.ones(
        (5, 5),
        dtype=np.float64
    )

    sums = box_sum_valid(
        box_test,
        radius=1
    )

    assert sums.shape == (3, 3)

    assert np.allclose(sums, 9.0)

    print("Test 2 passed: integral-image box summation.")

    # ---------------------------------------------------------------
    # TEST 3 — Stereo pair generation
    # ---------------------------------------------------------------

    left = np.tile(
        np.arange(20, dtype=np.uint8),
        (10, 1)
    )

    disparity_gt = np.ones(
        (10, 20),
        dtype=np.float32
    ) * 3

    right = create_stereo_pair(
        left,
        disparity_gt
    )

    assert right[5, 5] == left[5, 8]

    print("Test 3 passed: synthetic stereo generation.")

    # ---------------------------------------------------------------
    # TEST 4 — Disparity estimation
    # ---------------------------------------------------------------

    rng = np.random.default_rng(123)

    left = rng.integers(
        0,
        256,
        size=(80, 120),
        dtype=np.uint8
    )

    known_disparity = 5

    right = np.zeros_like(left)

    right[:, :-known_disparity] = (
        left[:, known_disparity:]
    )

    estimated, _ = compute_disparity(
        left,
        right,
        max_disparity=10,
        block_size=7
    )

    region = estimated[
        10:-10,
        25:-10
    ]

    median_disparity = np.median(region)

    assert median_disparity == known_disparity

    print(
        "Test 4 passed: SAD stereo matching. "
        f"Estimated disparity = {median_disparity:.1f} px."
    )

    # ---------------------------------------------------------------
    # TEST 5 — Depth equation
    # ---------------------------------------------------------------

    disparity = np.array([
        10.0,
        20.0,
        30.0
    ])

    depth = disparity_to_depth(
        disparity,
        focal_length=600,
        baseline=0.12
    )

    expected = np.array([
        7.2,
        3.6,
        2.4
    ])

    assert np.allclose(depth, expected)

    print("Test 5 passed: disparity-to-depth conversion.")

    # ---------------------------------------------------------------
    # TEST 6 — Zero disparity
    # ---------------------------------------------------------------

    depth = disparity_to_depth(
        np.array([0.0, 5.0]),
        focal_length=600,
        baseline=0.12
    )

    assert np.isnan(depth[0])

    assert np.isclose(depth[1], 14.4)

    print("Test 6 passed: invalid disparity handling.")

    # ---------------------------------------------------------------
    # TEST 7 — Depth ordering
    # ---------------------------------------------------------------

    depths = disparity_to_depth(
        np.array([4.0, 8.0, 12.0]),
        focal_length=600,
        baseline=0.12
    )

    assert depths[0] > depths[1] > depths[2]

    print("Test 7 passed: larger disparity means closer depth.")

    print("\nALL LESSON 19 TESTS PASSED.")


# =====================================================================
# STEP 10 — MAIN SYNTHETIC EXPERIMENT
# =====================================================================

def run_synthetic_experiment():

    print("\n" + "=" * 70)

    print("SYNTHETIC STEREO CAMERA EXPERIMENT")

    print("=" * 70)

    # Generate scene.
    left = create_texture(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        seed=SEED
    )

    disparity_gt = create_disparity_ground_truth(
        IMAGE_HEIGHT,
        IMAGE_WIDTH
    )

    right = create_stereo_pair(
        left,
        disparity_gt
    )

    # Estimate disparity from scratch.
    disparity_numpy, cost = compute_disparity(
        left,
        right,
        max_disparity=MAX_DISPARITY,
        block_size=BLOCK_SIZE
    )

    # Convert disparity to depth.
    depth_numpy = disparity_to_depth(
        disparity_numpy,
        FOCAL_LENGTH_PX,
        BASELINE_M
    )

    depth_gt = disparity_to_depth(
        disparity_gt,
        FOCAL_LENGTH_PX,
        BASELINE_M
    )

    # Compare valid interior pixels.
    valid = (
        (disparity_gt > 0) &
        (disparity_numpy > 0) &
        (cost < np.inf)
    )

    disparity_error = np.abs(
        disparity_numpy[valid] -
        disparity_gt[valid]
    )

    mean_disparity_error = np.mean(
        disparity_error
    )

    mean_depth_error, median_depth_error = calculate_depth_error(
        depth_numpy,
        depth_gt
    )

    print(
        f"\nMean disparity error: "
        f"{mean_disparity_error:.3f} pixels"
    )

    print(
        f"Mean depth error: "
        f"{mean_depth_error:.4f} meters"
    )

    print(
        f"Median depth error: "
        f"{median_depth_error:.4f} meters"
    )

    # ---------------------------------------------------------------
    # OpenCV comparison
    # ---------------------------------------------------------------

    disparity_cv = compute_opencv_disparity(
        left,
        right
    )

    depth_cv = disparity_to_depth(
        disparity_cv,
        FOCAL_LENGTH_PX,
        BASELINE_M
    )

    # ---------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(17, 13)
    )

    axes[0, 0].imshow(
        left,
        cmap="gray"
    )

    axes[0, 0].set_title("Left Camera")

    axes[0, 1].imshow(
        right,
        cmap="gray"
    )

    axes[0, 1].set_title("Right Camera")

    axes[0, 2].imshow(
        draw_stereo_epipolar_lines(left, right)
    )

    axes[0, 2].set_title("Rectified Epipolar Rows")

    im1 = axes[1, 0].imshow(
        disparity_gt,
        cmap="turbo",
        vmin=0,
        vmax=MAX_DISPARITY
    )

    axes[1, 0].set_title("Ground-Truth Disparity")

    plt.colorbar(
        im1,
        ax=axes[1, 0]
    )

    im2 = axes[1, 1].imshow(
        disparity_numpy,
        cmap="turbo",
        vmin=0,
        vmax=MAX_DISPARITY
    )

    axes[1, 1].set_title("NumPy SAD Disparity")

    plt.colorbar(
        im2,
        ax=axes[1, 1]
    )

    im3 = axes[1, 2].imshow(
        disparity_cv,
        cmap="turbo",
        vmin=0,
        vmax=MAX_DISPARITY
    )

    axes[1, 2].set_title("OpenCV StereoBM Disparity")

    plt.colorbar(
        im3,
        ax=axes[1, 2]
    )

    im4 = axes[2, 0].imshow(
        depth_gt,
        cmap="viridis",
        vmin=0,
        vmax=FOCAL_LENGTH_PX * BASELINE_M / 4
    )

    axes[2, 0].set_title("Ground-Truth Depth (m)")

    plt.colorbar(
        im4,
        ax=axes[2, 0]
    )

    im5 = axes[2, 1].imshow(
        depth_numpy,
        cmap="viridis",
        vmin=0,
        vmax=FOCAL_LENGTH_PX * BASELINE_M / 4
    )

    axes[2, 1].set_title("NumPy Estimated Depth (m)")

    plt.colorbar(
        im5,
        ax=axes[2, 1]
    )

    im6 = axes[2, 2].imshow(
        np.abs(depth_numpy - depth_gt),
        cmap="inferno"
    )

    axes[2, 2].set_title("Absolute Depth Error (m)")

    plt.colorbar(
        im6,
        ax=axes[2, 2]
    )

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()

    plt.show()

    # ---------------------------------------------------------------
    # Example pixel depth calculations
    # ---------------------------------------------------------------

    print("\nDEPTH EXAMPLES")

    for d in [4, 8, 12]:

        z = FOCAL_LENGTH_PX * BASELINE_M / d

        print(
            f"Disparity = {d:2d} px "
            f"-> Depth = {z:.3f} m"
        )


# =====================================================================
# STEP 11 — ROBOTICS APPLICATION
# =====================================================================

def robotics_waste_sorting_demo():

    print("\n" + "=" * 70)

    print("OPENARM V2 — STEREO DEPTH FOR WASTE SORTING")

    print("=" * 70)

    # Example camera parameters.
    focal_length = 600.0

    baseline = 0.12

    # Simulated detected object centers and disparity.
    objects = [
        ("Plastic cup", 8.0),
        ("Bottle", 12.0),
        ("Paper waste", 5.0)
    ]

    for name, disparity in objects:

        depth = (
            focal_length * baseline /
            disparity
        )

        print(f"\nObject: {name}")

        print(
            f"Disparity: {disparity:.1f} pixels"
        )

        print(
            f"Estimated camera depth: {depth:.3f} meters"
        )

    print(
        "\nROBOTICS NOTE:"
        "\nDepth is measured relative to the camera coordinate frame."
        "\nTo command OpenArm, transform the 3D point into the robot "
        "base frame using calibrated camera-to-robot extrinsics."
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)

    print("COMPUTER VISION — LESSON 19")

    print("STEREO VISION AND DEPTH ESTIMATION")

    print("=" * 70)

    run_tests()

    run_synthetic_experiment()

    robotics_waste_sorting_demo()

    print("\nLESSON 19 COMPLETED.")


if __name__ == "__main__":

    main()