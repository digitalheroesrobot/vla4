"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 20 — STEREO MATCHING IMPROVEMENTS AND DEPTH REFINEMENT
======================================================================

COURSE RULE
-----------
No pretrained models.
No pretrained weights.
No external datasets required.

We implement the main stereo-refinement concepts using NumPy.

OpenCV is used for practical image filtering and comparison.

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1 — REVIEW OF STEREO MATCHING
    Understand why raw block matching produces errors.

STEP 2 — NORMALIZED SAD COST
    Compare matching costs independently of block area.

STEP 3 — UNIQUENESS CHECK
    Reject ambiguous matches when the best candidate is not
    sufficiently better than the second-best candidate.

STEP 4 — SUBPIXEL DISPARITY
    Estimate disparity between integer pixel positions using
    parabolic interpolation.

STEP 5 — LEFT-RIGHT CONSISTENCY
    Check whether left-to-right and right-to-left disparity
    estimates agree.

STEP 6 — MEDIAN FILTERING
    Remove isolated disparity noise.

STEP 7 — SPECKLE REMOVAL
    Remove small isolated disparity regions.

STEP 8 — CONFIDENCE ESTIMATION
    Estimate how reliable each disparity measurement is.

STEP 9 — DEPTH REFINEMENT
    Convert refined disparity into metric depth.

STEP 10 — OPENCV COMPARISON
    Compare with OpenCV StereoSGBM.

======================================================================
MATHEMATICAL FOUNDATION
======================================================================

1. NORMALIZED SAD
-----------------

For a block window W:

    SAD(d) = sum_W |L(x,y) - R(x-d,y)|

Normalized SAD:

    NSAD(d) = SAD(d) / number_of_pixels_in_W

This makes the cost independent of block size.

2. UNIQUENESS RATIO
-------------------

Let:

    C1 = best matching cost
    C2 = second-best matching cost

A reliable match should have a clearly better best cost.

We calculate:

    uniqueness = (C2 - C1) / (C1 + epsilon)

Reject matches when:

    uniqueness < threshold

We exclude the immediate neighboring disparities when
finding the second-best cost because neighboring disparity
candidates often represent the same matching minimum.

3. SUBPIXEL DISPARITY
---------------------

The true disparity may be:

    d = 7.35 pixels

rather than exactly 7 pixels.

Suppose the minimum matching cost occurs at integer disparity d.

Use the costs at:

    C(d-1), C(d), C(d+1)

Fit a parabola around the minimum.

The subpixel offset is:

    delta = 0.5 * (C(d-1) - C(d+1))
            --------------------------------
            C(d-1) - 2*C(d) + C(d+1)

Then:

    d_subpixel = d + delta

The offset is clipped to [-1, 1].

4. LEFT-RIGHT CONSISTENCY
-------------------------

For the left image:

    d_left(x,y) = x_left - x_right

For the right image:

    d_right(x,y) = x_left - x_right

At corresponding locations:

    x_right = x_left - d_left(x_left,y)

Therefore:

    d_left(x,y) approximately equals
    d_right(x - d_left(x,y), y)

Reject a pixel if:

    abs(d_left - sampled_d_right) > threshold

This detects many incorrect matches and occluded pixels.

5. MEDIAN FILTER
----------------

For each pixel:

    d_filtered(x,y) = median of neighboring disparities

Median filtering is useful because it removes isolated
outliers while preserving edges better than simple averaging.

However, large windows can erase small objects and thin structures.

6. SPECKLE REMOVAL
------------------

A speckle is a small isolated connected region of disparity.

We identify neighboring valid pixels whose disparity values
are sufficiently similar.

Small components are rejected.

7. DEPTH
--------

    Z = f * B / d

Depth uncertainty increases rapidly when disparity becomes small.

A useful approximation:

    sigma_Z approximately equals
        (f * B / d^2) * sigma_d

Thus, distant objects have less accurate depth estimates.

======================================================================
REAL-WORLD ROBOTICS APPLICATION
======================================================================

OPENARM V2 — WASTE SORTING

A raw disparity map may contain:

    - Incorrect cup boundaries
    - Noisy bottle surfaces
    - Invalid pixels
    - Background mismatches
    - Occlusion artifacts

A practical pipeline:

    Stereo RGB images
          |
    Rectification
          |
    Stereo matching
          |
    Left-right consistency
          |
    Uniqueness filtering
          |
    Median filtering
          |
    Speckle removal
          |
    Depth reconstruction
          |
    Object segmentation
          |
    3D localization
          |
    Robot grasp planning

IMPORTANT:

    These filters improve the disparity map, but they do not
    guarantee accurate 3D geometry.

    Reflective, transparent, textureless, or partially occluded
    objects remain difficult for passive stereo cameras.

======================================================================
DEPENDENCIES
======================================================================

    pip install numpy matplotlib opencv-python

======================================================================
RUN
======================================================================

    python lesson20_stereo_matching_refinement.py

======================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from collections import deque


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

UNIQUENESS_THRESHOLD = 0.08
LR_CONSISTENCY_THRESHOLD = 1.5

SPECKLE_DISPARITY_THRESHOLD = 1.5
SPECKLE_MIN_SIZE = 20

EPSILON = 1e-8


# =====================================================================
# STEP 1 — SYNTHETIC TEXTURED SCENE
# =====================================================================

def create_texture(height, width, seed=42):

    rng = np.random.default_rng(seed)

    image = rng.integers(
        20,
        230,
        size=(height, width),
        dtype=np.uint8
    )

    image = cv2.GaussianBlur(
        image,
        (3, 3),
        0.5
    )

    # Add structured features.

    image = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    cv2.rectangle(
        image,
        (30, 30),
        (110, 110),
        (240, 240, 240),
        3
    )

    cv2.circle(
        image,
        (190, 75),
        35,
        (10, 10, 10),
        3
    )

    cv2.rectangle(
        image,
        (210, 145),
        (290, 220),
        (245, 245, 245),
        -1
    )

    cv2.putText(
        image,
        "STEREO",
        (35, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def create_ground_truth_disparity(height, width):

    disparity = np.full(
        (height, width),
        4.0,
        dtype=np.float32
    )

    disparity[30:115, 30:115] = 8.0

    disparity[120:215, 185:290] = 12.0

    disparity[65:145, 145:205] = 8.0

    return disparity


def create_stereo_pair(left, disparity):

    """
    Simplified rectified stereo rendering.

    right(x) = left(x + disparity)

    This is a synthetic demonstration, not a physically
    complete renderer. It does not model realistic occlusions.
    """

    height, width = left.shape

    right = np.zeros_like(left)

    x = np.arange(width)

    for y in range(height):

        d = disparity[y].astype(int)

        source_x = np.minimum(
            x + d,
            width - 1
        )

        right[y] = left[y, source_x]

    return right


# =====================================================================
# STEP 2 — INTEGRAL IMAGE AND BLOCK SUM
# =====================================================================

def integral_image(image):

    image = np.asarray(
        image,
        dtype=np.float64
    )

    return np.pad(
        image.cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
        mode="constant"
    )


def box_sum_same_size(image, radius):

    """
    Calculate square-window sums.

    Output has the same height and width as the input.

    Border windows are zero-padded.
    """

    integral = integral_image(image)

    height, width = image.shape

    size = 2 * radius + 1

    y0 = np.maximum(
        np.arange(height) - radius,
        0
    )

    y1 = np.minimum(
        np.arange(height) + radius + 1,
        height
    )

    x0 = np.maximum(
        np.arange(width) - radius,
        0
    )

    x1 = np.minimum(
        np.arange(width) + radius + 1,
        width
    )

    result = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )

    return result


# =====================================================================
# STEP 3 — NORMALIZED SAD COST VOLUME
# =====================================================================

def compute_cost_volume(
    reference,
    target,
    max_disparity,
    block_size,
    direction="left"
):

    """
    Compute normalized SAD costs.

    direction="left":
        Reference is left image.
        Candidate match is target[x-d].

    direction="right":
        Reference is right image.
        Candidate match is target[x+d].

    Returns:
        H x W x D cost volume.
    """

    reference = reference.astype(np.float64)

    target = target.astype(np.float64)

    height, width = reference.shape

    radius = block_size // 2

    cost_volume = np.full(
        (height, width, max_disparity),
        np.inf,
        dtype=np.float64
    )

    for d in range(max_disparity):

        if direction == "left":

            if d >= width:
                continue

            difference = np.abs(
                reference[:, d:] -
                target[:, :width - d]
            )

            valid_x_start = d + radius

            valid_x_end = width - radius

        elif direction == "right":

            if d >= width:
                continue

            difference = np.abs(
                reference[:, :width - d] -
                target[:, d:]
            )

            valid_x_start = radius

            valid_x_end = width - d - radius

        else:

            raise ValueError(
                "direction must be 'left' or 'right'"
            )

        # Integral image over the overlapping region.
        costs = box_sum_same_size(
            difference,
            radius
        )

        # Normalize by the number of pixels in the full window.
        window_area = block_size * block_size

        costs = costs / window_area

        y_start = radius
        y_end = height - radius

        if valid_x_end <= valid_x_start:
            continue

        if direction == "left":

            cost_volume[
                y_start:y_end,
                valid_x_start:valid_x_end,
                d
            ] = costs[
                radius:-radius,
                radius:-radius
            ]

        else:

            cost_volume[
                y_start:y_end,
                valid_x_start:valid_x_end,
                d
            ] = costs[
                radius:-radius,
                radius:-radius
            ]

    return cost_volume


# =====================================================================
# STEP 4 — WINNER-TAKES-ALL + UNIQUENESS
# =====================================================================

def estimate_disparity(
    cost_volume,
    uniqueness_threshold=0.08
):

    """
    Estimate integer and subpixel disparity.

    Returns:
        disparity
        confidence
        valid mask
        best cost
    """

    height, width, num_disparities = cost_volume.shape

    safe_cost = np.where(
        np.isfinite(cost_volume),
        cost_volume,
        np.inf
    )

    best_disparity = np.argmin(
        safe_cost,
        axis=2
    )

    best_cost = np.min(
        safe_cost,
        axis=2
    )

    # ---------------------------------------------------------------
    # Find second-best cost outside immediate neighboring disparities.
    # ---------------------------------------------------------------

    second_cost_volume = safe_cost.copy()

    yy, xx = np.indices(
        (height, width)
    )

    second_cost_volume[
        yy,
        xx,
        best_disparity
    ] = np.inf

    if num_disparities > 1:

        for offset in [-1, 1]:

            neighbor = best_disparity + offset

            valid_neighbor = (
                (neighbor >= 0) &
                (neighbor < num_disparities)
            )

            rows = yy[valid_neighbor]

            cols = xx[valid_neighbor]

            disparities = neighbor[valid_neighbor]

            second_cost_volume[
                rows,
                cols,
                disparities
            ] = np.inf

    second_cost = np.min(
        second_cost_volume,
        axis=2
    )

    uniqueness = (
        second_cost - best_cost
    ) / (
        np.abs(best_cost) + EPSILON
    )

    valid = (
        np.isfinite(best_cost) &
        np.isfinite(second_cost) &
        (uniqueness >= uniqueness_threshold)
    )

    # ---------------------------------------------------------------
    # Subpixel parabolic interpolation.
    # ---------------------------------------------------------------

    disparity = best_disparity.astype(
        np.float32
    )

    for d in range(1, num_disparities - 1):

        mask = (
            (best_disparity == d) &
            np.isfinite(
                cost_volume[:, :, d - 1]
            ) &
            np.isfinite(
                cost_volume[:, :, d]
            ) &
            np.isfinite(
                cost_volume[:, :, d + 1]
            )
        )

        c_left = cost_volume[:, :, d - 1]

        c_center = cost_volume[:, :, d]

        c_right = cost_volume[:, :, d + 1]

        denominator = (
            c_left -
            2.0 * c_center +
            c_right
        )

        safe = mask & (
            denominator > EPSILON
        )

        offset = np.zeros(
            (height, width),
            dtype=np.float32
        )

        offset[safe] = (
            0.5 *
            (c_left[safe] - c_right[safe]) /
            denominator[safe]
        ).astype(np.float32)

        offset = np.clip(
            offset,
            -1.0,
            1.0
        )

        disparity[safe] += offset[safe]

    disparity[~valid] = np.nan

    confidence = np.zeros(
        (height, width),
        dtype=np.float32
    )

    confidence[valid] = np.clip(
        uniqueness[valid],
        0.0,
        1.0
    )

    return (
        disparity,
        confidence,
        valid,
        best_cost
    )


# =====================================================================
# STEP 5 — LEFT-RIGHT CONSISTENCY
# =====================================================================

def left_right_consistency(
    left_disparity,
    right_disparity,
    threshold=1.5
):

    """
    Compare left and right disparity estimates.

    Left disparity convention:

        d_left = x_left - x_right

    Corresponding right coordinate:

        x_right = x_left - d_left

    Right disparity is sampled at that corresponding location.
    """

    height, width = left_disparity.shape

    yy, xx = np.indices(
        (height, width)
    )

    valid_left = np.isfinite(
        left_disparity
    )

    right_x = np.rint(
        xx - np.nan_to_num(
            left_disparity,
            nan=0.0
        )
    ).astype(int)

    inside = (
        (right_x >= 0) &
        (right_x < width)
    )

    safe_x = np.clip(
        right_x,
        0,
        width - 1
    )

    sampled_right = right_disparity[
        yy,
        safe_x
    ]

    valid_right = np.isfinite(
        sampled_right
    )

    error = np.full(
        (height, width),
        np.inf,
        dtype=np.float32
    )

    valid = (
        valid_left &
        inside &
        valid_right
    )

    error[valid] = np.abs(
        left_disparity[valid] -
        sampled_right[valid]
    )

    consistent = (
        valid &
        (error <= threshold)
    )

    refined = left_disparity.copy()

    refined[~consistent] = np.nan

    return refined, consistent, error


# =====================================================================
# STEP 6 — MEDIAN FILTER FROM SCRATCH
# =====================================================================

def median_filter_disparity(
    disparity,
    kernel_size=5
):

    """
    Median filter that ignores invalid NaN values.

    This implementation uses NumPy sliding_window_view.

    It is intended for educational, moderate-sized images.
    OpenCV's medianBlur is generally faster for production use.
    """

    if kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be odd"
        )

    radius = kernel_size // 2

    padded = np.pad(
        disparity,
        radius,
        mode="constant",
        constant_values=np.nan
    )

    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (kernel_size, kernel_size)
    )

    windows = windows.reshape(
        disparity.shape[0],
        disparity.shape[1],
        -1
    )

    with np.errstate(
        all="ignore"
    ):

        filtered = np.nanmedian(
            windows,
            axis=2
        )

    return filtered.astype(
        np.float32
    )


# =====================================================================
# STEP 7 — SPECKLE REMOVAL
# =====================================================================

def remove_small_speckles(
    disparity,
    disparity_threshold=1.5,
    min_size=20
):

    """
    Remove small connected disparity regions.

    Two neighboring pixels belong to the same component if:

        1. Both disparities are valid.
        2. Their disparity difference is <= threshold.

    Uses 8-connectivity.

    This is an educational component implementation.
    """

    height, width = disparity.shape

    valid = np.isfinite(disparity)

    visited = np.zeros(
        (height, width),
        dtype=bool
    )

    result = disparity.copy()

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for y in range(height):

        for x in range(width):

            if not valid[y, x]:
                continue

            if visited[y, x]:
                continue

            queue = deque()

            queue.append((y, x))

            visited[y, x] = True

            component = []

            while queue:

                cy, cx = queue.popleft()

                component.append(
                    (cy, cx)
                )

                current_disparity = disparity[
                    cy,
                    cx
                ]

                for dy, dx in neighbors:

                    ny = cy + dy

                    nx = cx + dx

                    if (
                        ny < 0 or
                        ny >= height or
                        nx < 0 or
                        nx >= width
                    ):
                        continue

                    if visited[ny, nx]:
                        continue

                    if not valid[ny, nx]:
                        continue

                    difference = abs(
                        disparity[ny, nx] -
                        current_disparity
                    )

                    if difference <= disparity_threshold:

                        visited[ny, nx] = True

                        queue.append(
                            (ny, nx)
                        )

            if len(component) < min_size:

                for cy, cx in component:

                    result[cy, cx] = np.nan

    return result


# =====================================================================
# STEP 8 — DEPTH RECONSTRUCTION
# =====================================================================

def disparity_to_depth(
    disparity,
    focal_length,
    baseline
):

    disparity = np.asarray(
        disparity,
        dtype=np.float64
    )

    depth = np.full_like(
        disparity,
        np.nan
    )

    valid = (
        np.isfinite(disparity) &
        (disparity > EPSILON)
    )

    depth[valid] = (
        focal_length *
        baseline /
        disparity[valid]
    )

    return depth


def estimate_depth_uncertainty(
    disparity,
    disparity_uncertainty,
    focal_length,
    baseline
):

    """
    First-order uncertainty propagation.

        Z = fB/d

        sigma_Z approximately equals
            fB * sigma_d / d^2
    """

    disparity = np.asarray(
        disparity,
        dtype=np.float64
    )

    sigma_z = np.full_like(
        disparity,
        np.nan
    )

    valid = (
        np.isfinite(disparity) &
        (disparity > EPSILON)
    )

    sigma_z[valid] = (
        focal_length *
        baseline *
        disparity_uncertainty /
        (disparity[valid] ** 2)
    )

    return sigma_z


# =====================================================================
# STEP 9 — OPENCV PRACTICAL FILTERING
# =====================================================================

def opencv_median_filter(disparity):

    """
    OpenCV median filtering.

    OpenCV medianBlur does not support NaN values reliably.

    We replace invalid pixels with the median valid disparity,
    filter, then restore the original invalid mask.
    """

    valid = np.isfinite(disparity)

    if not np.any(valid):
        return disparity.copy()

    fill_value = np.median(
        disparity[valid]
    )

    filled = np.where(
        valid,
        disparity,
        fill_value
    ).astype(np.float32)

    filtered = cv2.medianBlur(
        filled,
        5
    )

    filtered[~valid] = np.nan

    return filtered


# =====================================================================
# STEP 10 — OPENCV STEREO SGBM
# =====================================================================

def opencv_stereo_sgbm(
    left,
    right,
    max_disparity=16
):

    """
    Practical Semi-Global Block Matching.

    StereoSGBM combines matching costs with smoothness penalties
    along multiple image paths.

    It is generally more robust than simple independent SAD
    block matching.

    Inputs must be rectified grayscale stereo images.
    """

    num_disparities = (
        max_disparity // 16 + 1
    ) * 16

    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=5,
        P1=8 * 5 * 5,
        P2=32 * 5 * 5,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity_fixed = matcher.compute(
        left,
        right
    )

    disparity = (
        disparity_fixed.astype(np.float32) /
        16.0
    )

    disparity[disparity <= 0] = np.nan

    return disparity


# =====================================================================
# STEP 11 — NUMERICAL TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print("RUNNING LESSON 20 TESTS")

    print("=" * 70)

    # ---------------------------------------------------------------
    # TEST 1 — Integral image
    # ---------------------------------------------------------------

    image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float64)

    integral = integral_image(image)

    assert integral.shape == (4, 4)

    assert integral[-1, -1] == 45

    print("Test 1 passed: integral image.")

    # ---------------------------------------------------------------
    # TEST 2 — Box sum
    # ---------------------------------------------------------------

    image = np.ones(
        (7, 7),
        dtype=np.float64
    )

    sums = box_sum_same_size(
        image,
        radius=1
    )

    assert sums.shape == image.shape

    assert sums[3, 3] == 9

    assert sums[0, 0] == 4

    print("Test 2 passed: same-size box summation.")

    # ---------------------------------------------------------------
    # TEST 3 — Normalized SAD cost
    # ---------------------------------------------------------------

    left = np.ones(
        (20, 30),
        dtype=np.uint8
    ) * 100

    right = left.copy()

    volume = compute_cost_volume(
        left,
        right,
        max_disparity=4,
        block_size=5
    )

    assert volume.shape == (20, 30, 4)

    assert volume[10, 10, 0] == 0

    print("Test 3 passed: normalized SAD cost volume.")

    # ---------------------------------------------------------------
    # TEST 4 — Subpixel interpolation
    # ---------------------------------------------------------------

    costs = np.full(
        (1, 1, 5),
        100.0
    )

    costs[0, 0, 1] = 9.0

    costs[0, 0, 2] = 1.0

    costs[0, 0, 3] = 4.0

    costs[0, 0, 4] = 100.0

    disparity, confidence, valid, best_cost = (
        estimate_disparity(
            costs,
            uniqueness_threshold=0.0
        )
    )

    assert valid[0, 0]

    assert disparity[0, 0] > 2.0

    assert disparity[0, 0] < 3.0

    print(
        "Test 4 passed: subpixel disparity interpolation."
    )

    print(
        f"  Estimated subpixel disparity: "
        f"{disparity[0, 0]:.4f}"
    )

    # ---------------------------------------------------------------
    # TEST 5 — Left-right consistency
    # ---------------------------------------------------------------

    left_disp = np.full(
        (5, 10),
        3.0,
        dtype=np.float32
    )

    right_disp = np.full(
        (5, 10),
        3.0,
        dtype=np.float32
    )

    refined, consistent, error = left_right_consistency(
        left_disp,
        right_disp,
        threshold=0.1
    )

    assert np.all(
        consistent[:, 3:]
    )

    assert np.allclose(
        refined[:, 3:],
        3.0
    )

    print("Test 5 passed: left-right consistency.")

    # ---------------------------------------------------------------
    # TEST 6 — Inconsistent disparity rejection
    # ---------------------------------------------------------------

    right_disp[:, 4] = 8.0

    refined, consistent, error = left_right_consistency(
        left_disp,
        right_disp,
        threshold=1.0
    )

    assert not consistent[2, 7]

    print("Test 6 passed: inconsistent matches rejected.")

    # ---------------------------------------------------------------
    # TEST 7 — Median filtering
    # ---------------------------------------------------------------

    disparity = np.ones(
        (9, 9),
        dtype=np.float32
    ) * 5.0

    disparity[4, 4] = 15.0

    filtered = median_filter_disparity(
        disparity,
        kernel_size=3
    )

    assert filtered[4, 4] == 5.0

    print("Test 7 passed: median filtering removes isolated noise.")

    # ---------------------------------------------------------------
    # TEST 8 — Speckle removal
    # ---------------------------------------------------------------

    disparity = np.full(
        (20, 20),
        np.nan,
        dtype=np.float32
    )

    disparity[2:12, 2:12] = 8.0

    disparity[16, 16] = 12.0

    cleaned = remove_small_speckles(
        disparity,
        disparity_threshold=1.0,
        min_size=10
    )

    assert np.isfinite(cleaned[5, 5])

    assert np.isnan(cleaned[16, 16])

    print("Test 8 passed: small disparity speckles removed.")

    # ---------------------------------------------------------------
    # TEST 9 — Depth estimation
    # ---------------------------------------------------------------

    disparity = np.array([
        [4.0, 8.0],
        [12.0, np.nan]
    ])

    depth = disparity_to_depth(
        disparity,
        focal_length=600,
        baseline=0.12
    )

    expected = np.array([
        [18.0, 9.0],
        [6.0, np.nan]
    ])

    assert np.allclose(
        depth[:1],
        expected[:1]
    )

    assert np.isclose(
        depth[1, 0],
        expected[1, 0]
    )

    assert np.isnan(
        depth[1, 1]
    )

    print("Test 9 passed: depth reconstruction.")

    # ---------------------------------------------------------------
    # TEST 10 — Depth uncertainty
    # ---------------------------------------------------------------

    disparity = np.array([
        4.0,
        8.0,
        16.0
    ])

    uncertainty = estimate_depth_uncertainty(
        disparity,
        disparity_uncertainty=0.5,
        focal_length=600,
        baseline=0.12
    )

    assert uncertainty[0] > uncertainty[1]

    assert uncertainty[1] > uncertainty[2]

    print("Test 10 passed: depth uncertainty propagation.")

    # ---------------------------------------------------------------
    # TEST 11 — OpenCV median filter
    # ---------------------------------------------------------------

    disparity = np.ones(
        (20, 20),
        dtype=np.float32
    ) * 5.0

    disparity[10, 10] = 20.0

    filtered_cv = opencv_median_filter(
        disparity
    )

    assert filtered_cv[10, 10] == 5.0

    print("Test 11 passed: OpenCV median filter.")

    print("\nALL LESSON 20 TESTS PASSED.")


# =====================================================================
# STEP 12 — COMPLETE STEREO EXPERIMENT
# =====================================================================

def run_stereo_experiment():

    print("\n" + "=" * 70)

    print("STEREO REFINEMENT EXPERIMENT")

    print("=" * 70)

    # ---------------------------------------------------------------
    # Generate stereo images
    # ---------------------------------------------------------------

    left = create_texture(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        SEED
    )

    ground_truth_disparity = create_ground_truth_disparity(
        IMAGE_HEIGHT,
        IMAGE_WIDTH
    )

    right = create_stereo_pair(
        left,
        ground_truth_disparity
    )

    # ---------------------------------------------------------------
    # Add slight image noise
    # ---------------------------------------------------------------

    rng = np.random.default_rng(SEED + 1)

    left_noisy = np.clip(
        left.astype(np.float32) +
        rng.normal(0, 2.0, left.shape),
        0,
        255
    ).astype(np.uint8)

    right_noisy = np.clip(
        right.astype(np.float32) +
        rng.normal(0, 2.0, right.shape),
        0,
        255
    ).astype(np.uint8)

    # ---------------------------------------------------------------
    # Estimate left disparity
    # ---------------------------------------------------------------

    left_cost = compute_cost_volume(
        left_noisy,
        right_noisy,
        MAX_DISPARITY,
        BLOCK_SIZE,
        direction="left"
    )

    (
        disparity_raw,
        confidence,
        valid_unique,
        best_cost
    ) = estimate_disparity(
        left_cost,
        UNIQUENESS_THRESHOLD
    )

    # ---------------------------------------------------------------
    # Estimate right disparity
    # ---------------------------------------------------------------

    right_cost = compute_cost_volume(
        right_noisy,
        left_noisy,
        MAX_DISPARITY,
        BLOCK_SIZE,
        direction="right"
    )

    (
        disparity_right,
        confidence_right,
        valid_right,
        best_cost_right
    ) = estimate_disparity(
        right_cost,
        UNIQUENESS_THRESHOLD
    )

    # ---------------------------------------------------------------
    # Left-right consistency
    # ---------------------------------------------------------------

    disparity_lr, consistent, lr_error = (
        left_right_consistency(
            disparity_raw,
            disparity_right,
            LR_CONSISTENCY_THRESHOLD
        )
    )

    # ---------------------------------------------------------------
    # Median filtering
    # ---------------------------------------------------------------

    disparity_median = median_filter_disparity(
        disparity_lr,
        kernel_size=5
    )

    # ---------------------------------------------------------------
    # Speckle removal
    # ---------------------------------------------------------------

    disparity_clean = remove_small_speckles(
        disparity_median,
        disparity_threshold=SPECKLE_DISPARITY_THRESHOLD,
        min_size=SPECKLE_MIN_SIZE
    )

    # ---------------------------------------------------------------
    # Depth estimation
    # ---------------------------------------------------------------

    depth_raw = disparity_to_depth(
        disparity_raw,
        FOCAL_LENGTH_PX,
        BASELINE_M
    )

    depth_clean = disparity_to_depth(
        disparity_clean,
        FOCAL_LENGTH_PX,
        BASELINE_M
    )

    depth_gt = disparity_to_depth(
        ground_truth_disparity,
        FOCAL_LENGTH_PX,
        BASELINE_M
    )

    # ---------------------------------------------------------------
    # OpenCV comparison
    # ---------------------------------------------------------------

    disparity_sgbm = opencv_stereo_sgbm(
        left_noisy,
        right_noisy,
        MAX_DISPARITY
    )

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    valid_gt = np.isfinite(
        ground_truth_disparity
    )

    valid_raw = (
        valid_gt &
        np.isfinite(disparity_raw)
    )

    valid_clean = (
        valid_gt &
        np.isfinite(disparity_clean)
    )

    raw_mae = np.mean(
        np.abs(
            disparity_raw[valid_raw] -
            ground_truth_disparity[valid_raw]
        )
    )

    clean_mae = np.mean(
        np.abs(
            disparity_clean[valid_clean] -
            ground_truth_disparity[valid_clean]
        )
    )

    print(
        f"\nRaw disparity MAE: {raw_mae:.4f} px"
    )

    print(
        f"Refined disparity MAE: {clean_mae:.4f} px"
    )

    print(
        f"Initial unique pixels: "
        f"{np.sum(valid_unique)}"
    )

    print(
        f"Left-right consistent pixels: "
        f"{np.sum(consistent)}"
    )

    print(
        f"Final valid pixels: "
        f"{np.sum(np.isfinite(disparity_clean))}"
    )

    # ---------------------------------------------------------------
    # Depth uncertainty example
    # ---------------------------------------------------------------

    depth_uncertainty = estimate_depth_uncertainty(
        disparity_clean,
        disparity_uncertainty=0.5,
        focal_length=FOCAL_LENGTH_PX,
        baseline=BASELINE_M
    )

    print(
        "\nExample depth uncertainty:"
    )

    for d in [4.0, 8.0, 12.0]:

        sigma_z = (
            FOCAL_LENGTH_PX *
            BASELINE_M *
            0.5 /
            (d ** 2)
        )

        print(
            f"Disparity {d:4.1f} px -> "
            f"Approx. depth uncertainty = {sigma_z:.4f} m"
        )

    # ---------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 14)
    )

    axes[0, 0].imshow(
        left_noisy,
        cmap="gray"
    )

    axes[0, 0].set_title(
        "Left Image"
    )

    axes[0, 1].imshow(
        right_noisy,
        cmap="gray"
    )

    axes[0, 1].set_title(
        "Right Image"
    )

    axes[0, 2].imshow(
        ground_truth_disparity,
        cmap="turbo",
        vmin=0,
        vmax=MAX_DISPARITY
    )

    axes[0, 2].set_title(
        "Ground-Truth Disparity"
    )

    axes[1, 0].imshow(
        disparity_raw,
        cmap="turbo",
        vmin=0,
        vmax=MAX_DISPARITY
    )

    axes[1, 0].set_title(
        "Raw Subpixel Disparity"
    )

    axes[1, 1].imshow(
        confidence,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    axes[1, 1].set_title(
        "Matching Confidence"
    )

    axes[1, 2].imshow(
        lr_error,
        cmap="inferno",
        vmin=0,
        vmax=5
    )

    axes[1, 2].set_title(
        "Left-Right Consistency Error"
    )

    axes[2, 0].imshow(
        disparity_clean,
        cmap="turbo",
        vmin=0,
        vmax=MAX_DISPARITY
    )

    axes[2, 0].set_title(
        "Refined Disparity"
    )

    axes[2, 1].imshow(
        disparity_sgbm,
        cmap="turbo",
        vmin=0,
        vmax=MAX_DISPARITY
    )

    axes[2, 1].set_title(
        "OpenCV StereoSGBM"
    )

    axes[2, 2].imshow(
        np.abs(depth_clean - depth_gt),
        cmap="inferno",
        vmin=0,
        vmax=2
    )

    axes[2, 2].set_title(
        "Refined Depth Error (m)"
    )

    for ax in axes.ravel():

        ax.axis("off")

    plt.tight_layout()

    plt.show()

    # ---------------------------------------------------------------
    # Depth uncertainty visualization
    # ---------------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.imshow(
        depth_uncertainty,
        cmap="magma"
    )

    plt.colorbar(
        label="Approximate depth uncertainty (m)"
    )

    plt.title(
        "Depth Uncertainty Map"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()


# =====================================================================
# STEP 13 — ROBOTICS APPLICATION
# =====================================================================

def robotics_waste_sorting_demo():

    print("\n" + "=" * 70)

    print("OPENARM V2 — REFINED WASTE OBJECT DEPTH")

    print("=" * 70)

    focal_length = FOCAL_LENGTH_PX

    baseline = BASELINE_M

    # Simulated disparity estimates after refinement.
    objects = [
        {
            "name": "Plastic cup",
            "disparity": 10.2,
            "uncertainty": 0.4
        },
        {
            "name": "Bottle",
            "disparity": 13.5,
            "uncertainty": 0.3
        },
        {
            "name": "Paper waste",
            "disparity": 6.8,
            "uncertainty": 0.5
        }
    ]

    for obj in objects:

        disparity = obj["disparity"]

        sigma_d = obj["uncertainty"]

        depth = (
            focal_length *
            baseline /
            disparity
        )

        sigma_z = (
            focal_length *
            baseline *
            sigma_d /
            (disparity ** 2)
        )

        print(
            f"\nObject: {obj['name']}"
        )

        print(
            f"Disparity: {disparity:.2f} px"
        )

        print(
            f"Estimated depth: {depth:.3f} m"
        )

        print(
            f"Approximate depth uncertainty: "
            f"+/- {sigma_z:.3f} m"
        )

    print(
        "\nROBOTICS INTERPRETATION:"
    )

    print(
        "1. Use the refined disparity map to estimate object depth."
    )

    print(
        "2. Combine depth with image coordinates to reconstruct 3D points."
    )

    print(
        "3. Transform camera coordinates into the robot base frame."
    )

    print(
        "4. Use the 3D object estimate for grasp planning."
    )

    print(
        "\nRemember: ACT predicts robot actions from observations. "
        "Stereo depth is a perception input, not a replacement "
        "for robot calibration or grasp planning."
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)

    print("COMPUTER VISION — LESSON 20")

    print("STEREO MATCHING IMPROVEMENTS AND DEPTH REFINEMENT")

    print("=" * 70)

    run_tests()

    run_stereo_experiment()

    robotics_waste_sorting_demo()

    print("\nLESSON 20 COMPLETED.")


if __name__ == "__main__":

    main()