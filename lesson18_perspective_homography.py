"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 18 — PERSPECTIVE TRANSFORMATION AND HOMOGRAPHY
======================================================================

COURSE RULE
-----------
We build the mathematical foundations ourselves.

We may use OpenCV where it is useful in real-world computer vision.

CORE IMPLEMENTATIONS:
    1. Homogeneous coordinates
    2. Perspective transformation
    3. Homography matrix estimation using DLT
    4. Homography normalization
    5. Point transformation
    6. Inverse perspective mapping
    7. Image warping using bilinear interpolation
    8. RANSAC homography estimation
    9. Comparison with OpenCV

No pretrained models.
No pretrained weights.
No external dataset required.

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1 — HOMOGENEOUS COORDINATES
    Understand why a 3x3 matrix can represent a 2D projective
    transformation.

STEP 2 — PERSPECTIVE PROJECTION
    Understand why parallel lines may converge in an image.

STEP 3 — HOMOGRAPHY
    Learn the mathematical relationship:

        p' = H p

    where:
        p  = original homogeneous coordinate
        p' = transformed homogeneous coordinate
        H  = 3x3 homography matrix

STEP 4 — DIRECT LINEAR TRANSFORMATION (DLT)
    Estimate a homography from at least four point correspondences.

STEP 5 — IMAGE WARPING
    Transform every pixel into a new image coordinate system.

STEP 6 — BILINEAR INTERPOLATION
    Estimate image intensity at non-integer pixel coordinates.

STEP 7 — RANSAC
    Estimate homography robustly when some point correspondences
    are incorrect.

STEP 8 — OPENCV COMPARISON
    Compare our mathematical implementation against:

        cv2.getPerspectiveTransform()
        cv2.findHomography()
        cv2.warpPerspective()

======================================================================
MATHEMATICAL FOUNDATION
======================================================================

1. HOMOGENEOUS COORDINATES
--------------------------

A 2D point:

    (x, y)

becomes:

    p = [x, y, 1]^T

A projective transformation is:

    p' = H p

where:

        [h11 h12 h13]
    H = [h21 h22 h23]
        [h31 h32 h33]

After multiplication:

    x' = h11*x + h12*y + h13
    y' = h21*x + h22*y + h23
    w' = h31*x + h32*y + h33

Convert back to Cartesian coordinates:

    u = x' / w'
    v = y' / w'

The homography has eight independent parameters because its
overall scale does not change the transformation.

Therefore, four point correspondences provide eight equations.

2. DLT
------

For each correspondence:

    (x, y) -> (u, v)

we construct two linear equations:

    [-x -y -1  0  0  0  ux uy u]
    [ 0  0  0 -x -y -1  vx vy v]

Stack all equations into matrix A:

    A h = 0

The solution is the right singular vector associated with
the smallest singular value of A.

We calculate it using:

    np.linalg.svd(A)

3. IMAGE WARPING
----------------

Forward mapping:

    source pixel -> destination pixel

can create holes because transformed coordinates are not
necessarily integers.

Instead, we use inverse mapping:

    destination pixel -> source pixel

For each destination pixel:

    p_source = inverse(H) * p_destination

Then interpolate the source image.

4. BILINEAR INTERPOLATION
-------------------------

For a floating-point coordinate (x, y), find four neighbors:

    I00, I10
    I01, I11

Interpolate horizontally:

    top    = I00*(1-dx) + I10*dx
    bottom = I01*(1-dx) + I11*dx

Interpolate vertically:

    value = top*(1-dy) + bottom*dy

======================================================================
REAL-WORLD ROBOTICS APPLICATION
======================================================================

OPENARM V2 — WASTE SORTING

Imagine a fixed overhead camera looking at a conveyor belt.

The camera sees a cup from an angle.

We know four reference points on the conveyor belt:

    Image coordinates:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4)

    Conveyor coordinates:
        (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)

We estimate:

    H_image_to_conveyor

Then an object detector provides the image center:

    (pixel_x, pixel_y)

We transform it into conveyor coordinates:

    (world_X, world_Y)

This provides a useful 2D localization estimate for robotic
pick-and-place.

IMPORTANT:
    Homography works for points on the same plane.

    It does not directly provide the full 3D position of
    an object above the conveyor.

    For 3D grasping, we need additional depth information,
    camera calibration, or geometric assumptions.

======================================================================
DEPENDENCIES
======================================================================

    pip install numpy matplotlib opencv-python

======================================================================
RUN
======================================================================

    python lesson18_perspective_homography.py

======================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass


# =====================================================================
# CONFIGURATION
# =====================================================================

SEED = 42

np.random.seed(SEED)

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

WARP_WIDTH = 500
WARP_HEIGHT = 350

RANSAC_ITERATIONS = 1000
RANSAC_THRESHOLD = 3.0


# =====================================================================
# STEP 1 — HOMOGENEOUS COORDINATES
# =====================================================================

def to_homogeneous(points):
    """
    Convert Nx2 Cartesian coordinates into Nx3 homogeneous coordinates.
    """

    points = np.asarray(points, dtype=np.float64)

    ones = np.ones((len(points), 1), dtype=np.float64)

    return np.hstack([points, ones])


def from_homogeneous(points):
    """
    Convert Nx3 homogeneous coordinates into Nx2 Cartesian coordinates.
    """

    points = np.asarray(points, dtype=np.float64)

    denominator = points[:, 2:3]

    denominator = np.where(
        np.abs(denominator) < 1e-12,
        np.nan,
        denominator
    )

    return points[:, :2] / denominator


# =====================================================================
# STEP 2 — APPLY HOMOGRAPHY
# =====================================================================

def transform_points(points, H):
    """
    Apply a 3x3 homography to Nx2 points.
    """

    points_h = to_homogeneous(points)

    transformed_h = (H @ points_h.T).T

    return from_homogeneous(transformed_h)


# =====================================================================
# STEP 3 — NORMALIZE POINTS FOR DLT
# =====================================================================

def normalize_points(points):
    """
    Normalize coordinates.

    The centroid becomes the origin.

    Mean distance from origin becomes sqrt(2).

    This improves numerical stability during DLT estimation.
    """

    points = np.asarray(points, dtype=np.float64)

    centroid = np.mean(points, axis=0)

    shifted = points - centroid

    distances = np.sqrt(np.sum(shifted ** 2, axis=1))

    mean_distance = np.mean(distances)

    if mean_distance < 1e-12:
        raise ValueError("Degenerate point configuration.")

    scale = np.sqrt(2.0) / mean_distance

    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0]
    ])

    normalized = transform_points(points, T)

    return normalized, T


# =====================================================================
# STEP 4 — DIRECT LINEAR TRANSFORMATION
# =====================================================================

def estimate_homography_dlt(source_points, destination_points):
    """
    Estimate homography using normalized DLT.

    Minimum:
        Four non-collinear correspondences.

    Returns:
        3x3 homography matrix.
    """

    source_points = np.asarray(
        source_points,
        dtype=np.float64
    )

    destination_points = np.asarray(
        destination_points,
        dtype=np.float64
    )

    if source_points.shape != destination_points.shape:
        raise ValueError("Point arrays must have identical shapes.")

    if source_points.ndim != 2 or source_points.shape[1] != 2:
        raise ValueError("Points must have shape Nx2.")

    if len(source_points) < 4:
        raise ValueError("At least four correspondences are required.")

    source_norm, T_source = normalize_points(source_points)

    destination_norm, T_destination = normalize_points(
        destination_points
    )

    A = []

    for (x, y), (u, v) in zip(
        source_norm,
        destination_norm
    ):

        A.append([
            -x, -y, -1,
            0, 0, 0,
            u * x, u * y, u
        ])

        A.append([
            0, 0, 0,
            -x, -y, -1,
            v * x, v * y, v
        ])

    A = np.asarray(A, dtype=np.float64)

    _, singular_values, Vt = np.linalg.svd(A)

    h = Vt[-1]

    H_normalized = h.reshape(3, 3)

    H = np.linalg.inv(T_destination) @ H_normalized @ T_source

    if abs(H[2, 2]) > 1e-12:
        H /= H[2, 2]
    else:
        H /= np.linalg.norm(H)

    return H


# =====================================================================
# STEP 5 — BILINEAR INTERPOLATION
# =====================================================================

def bilinear_sample(image, x, y):
    """
    Sample grayscale or RGB image at floating-point coordinates.

    Returns zero for coordinates outside the image.
    """

    height, width = image.shape[:2]

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = (
        (x >= 0) &
        (x <= width - 1) &
        (y >= 0) &
        (y <= height - 1)
    )

    x_safe = np.clip(x, 0, width - 1)
    y_safe = np.clip(y, 0, height - 1)

    x0 = np.floor(x_safe).astype(int)
    y0 = np.floor(y_safe).astype(int)

    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    dx = x_safe - x0
    dy = y_safe - y0

    if image.ndim == 2:

        I00 = image[y0, x0]
        I10 = image[y0, x1]

        I01 = image[y1, x0]
        I11 = image[y1, x1]

        top = I00 * (1 - dx) + I10 * dx

        bottom = I01 * (1 - dx) + I11 * dx

        result = top * (1 - dy) + bottom * dy

        return np.where(valid, result, 0.0)

    else:

        I00 = image[y0, x0]
        I10 = image[y0, x1]

        I01 = image[y1, x0]
        I11 = image[y1, x1]

        dx = dx[..., None]
        dy = dy[..., None]

        top = I00 * (1 - dx) + I10 * dx

        bottom = I01 * (1 - dx) + I11 * dx

        result = top * (1 - dy) + bottom * dy

        return np.where(valid[..., None], result, 0.0)


# =====================================================================
# STEP 6 — IMAGE WARPING FROM SCRATCH
# =====================================================================

def warp_perspective_from_scratch(
    image,
    H,
    output_width,
    output_height
):
    """
    Warp image using inverse mapping.

    H maps source coordinates to destination coordinates.

    Therefore, inverse(H) maps destination coordinates
    back to source coordinates.
    """

    H_inverse = np.linalg.inv(H)

    yy, xx = np.indices(
        (output_height, output_width),
        dtype=np.float64
    )

    destination_points = np.stack(
        [xx.ravel(), yy.ravel()],
        axis=1
    )

    source_points = transform_points(
        destination_points,
        H_inverse
    )

    source_x = source_points[:, 0].reshape(
        output_height,
        output_width
    )

    source_y = source_points[:, 1].reshape(
        output_height,
        output_width
    )

    warped = bilinear_sample(
        image,
        source_x,
        source_y
    )

    return warped


# =====================================================================
# STEP 7 — RANSAC HOMOGRAPHY
# =====================================================================

@dataclass
class RANSACResult:

    H: np.ndarray

    inliers: np.ndarray

    iterations: int

    mean_error: float


def reprojection_errors(source_points, destination_points, H):

    predicted = transform_points(source_points, H)

    errors = np.linalg.norm(
        predicted - destination_points,
        axis=1
    )

    errors[~np.isfinite(errors)] = np.inf

    return errors


def estimate_homography_ransac(
    source_points,
    destination_points,
    iterations=1000,
    threshold=3.0
):
    """
    Robust homography estimation.

    Each iteration:

        1. Randomly select four correspondences.
        2. Estimate homography.
        3. Calculate reprojection errors.
        4. Identify inliers.
        5. Keep the model with most inliers.
        6. Re-estimate using all best inliers.
    """

    source_points = np.asarray(source_points, dtype=np.float64)

    destination_points = np.asarray(
        destination_points,
        dtype=np.float64
    )

    if len(source_points) < 4:
        raise ValueError("RANSAC requires at least four points.")

    if source_points.shape != destination_points.shape:
        raise ValueError("Point arrays must have identical shapes.")

    best_inliers = None

    best_H = None

    best_count = 0

    for _ in range(iterations):

        indices = np.random.choice(
            len(source_points),
            size=4,
            replace=False
        )

        try:

            H_candidate = estimate_homography_dlt(
                source_points[indices],
                destination_points[indices]
            )

            errors = reprojection_errors(
                source_points,
                destination_points,
                H_candidate
            )

            inliers = errors < threshold

            count = np.sum(inliers)

            if count > best_count:

                best_count = count

                best_inliers = inliers

                best_H = H_candidate

        except (ValueError, np.linalg.LinAlgError):
            continue

    if best_H is None or best_count < 4:
        raise RuntimeError("RANSAC failed to find a valid homography.")

    H_refined = estimate_homography_dlt(
        source_points[best_inliers],
        destination_points[best_inliers]
    )

    final_errors = reprojection_errors(
        source_points,
        destination_points,
        H_refined
    )

    final_inliers = final_errors < threshold

    return RANSACResult(
        H=H_refined,
        inliers=final_inliers,
        iterations=iterations,
        mean_error=float(np.mean(final_errors[final_inliers]))
    )


# =====================================================================
# STEP 8 — SYNTHETIC DOCUMENT IMAGE
# =====================================================================

def create_document_image(width=500, height=350):

    image = np.ones(
        (height, width, 3),
        dtype=np.uint8
    ) * 245

    cv2.rectangle(
        image,
        (5, 5),
        (width - 6, height - 6),
        (30, 30, 30),
        4
    )

    cv2.putText(
        image,
        "COMPUTER VISION",
        (35, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        2
    )

    cv2.putText(
        image,
        "LESSON 18",
        (35, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 180),
        2
    )

    cv2.putText(
        image,
        "PERSPECTIVE TRANSFORMATION",
        (35, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        2
    )

    for i in range(4):

        y = 210 + i * 30

        cv2.line(
            image,
            (35, y),
            (width - 40, y),
            (130, 130, 130),
            2
        )

    cv2.rectangle(
        image,
        (365, 220),
        (450, 300),
        (30, 180, 60),
        -1
    )

    cv2.putText(
        image,
        "CV",
        (385, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    return image


# =====================================================================
# STEP 9 — CREATE PERSPECTIVE SCENE
# =====================================================================

def create_perspective_scene(document):

    height, width = document.shape[:2]

    scene = np.zeros(
        (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        dtype=np.uint8
    )

    scene[:] = (70, 75, 80)

    source = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    destination = np.array([
        [155, 65],
        [550, 115],
        [475, 420],
        [80, 335]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(
        source,
        destination
    )

    warped = cv2.warpPerspective(
        document,
        H,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    mask = cv2.warpPerspective(
        np.ones((height, width), dtype=np.uint8) * 255,
        H,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    scene[mask > 0] = warped[mask > 0]

    return scene, source, destination, H


# =====================================================================
# STEP 10 — NUMERICAL TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)
    print("RUNNING LESSON 18 TESTS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # TEST 1 — Homogeneous coordinates
    # ---------------------------------------------------------------

    points = np.array([
        [10, 20],
        [30, 40],
        [50, 60]
    ], dtype=np.float64)

    homogeneous = to_homogeneous(points)

    recovered = from_homogeneous(homogeneous)

    assert np.allclose(points, recovered)

    print("Test 1 passed: homogeneous coordinate conversion.")

    # ---------------------------------------------------------------
    # TEST 2 — Identity homography
    # ---------------------------------------------------------------

    H_identity = np.eye(3)

    transformed = transform_points(points, H_identity)

    assert np.allclose(points, transformed)

    print("Test 2 passed: identity transformation.")

    # ---------------------------------------------------------------
    # TEST 3 — Translation homography
    # ---------------------------------------------------------------

    H_translation = np.array([
        [1, 0, 15],
        [0, 1, 25],
        [0, 0, 1]
    ], dtype=np.float64)

    expected = points + np.array([15, 25])

    transformed = transform_points(
        points,
        H_translation
    )

    assert np.allclose(transformed, expected)

    print("Test 3 passed: translation transformation.")

    # ---------------------------------------------------------------
    # TEST 4 — DLT estimation
    # ---------------------------------------------------------------

    H_true = np.array([
        [1.2, 0.1, 30],
        [0.05, 1.1, 20],
        [0.0005, 0.0008, 1]
    ], dtype=np.float64)

    source = np.array([
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
        [30, 40],
        [70, 20],
        [60, 80]
    ], dtype=np.float64)

    destination = transform_points(
        source,
        H_true
    )

    H_estimated = estimate_homography_dlt(
        source,
        destination
    )

    predicted = transform_points(
        source,
        H_estimated
    )

    assert np.allclose(
        predicted,
        destination,
        atol=1e-6
    )

    print("Test 4 passed: DLT homography estimation.")

    # ---------------------------------------------------------------
    # TEST 5 — Bilinear interpolation
    # ---------------------------------------------------------------

    image = np.array([
        [0, 10],
        [20, 30]
    ], dtype=np.float64)

    value = bilinear_sample(
        image,
        np.array([0.5]),
        np.array([0.5])
    )

    assert np.allclose(value, [15.0])

    print("Test 5 passed: bilinear interpolation.")

    # ---------------------------------------------------------------
    # TEST 6 — Image warping
    # ---------------------------------------------------------------

    test_image = np.zeros((50, 50), dtype=np.uint8)

    test_image[20, 20] = 255

    H_shift = np.array([
        [1, 0, 5],
        [0, 1, 7],
        [0, 0, 1]
    ], dtype=np.float64)

    warped = warp_perspective_from_scratch(
        test_image,
        H_shift,
        50,
        50
    )

    assert warped[27, 25] > 200

    print("Test 6 passed: inverse-mapped image warping.")

    # ---------------------------------------------------------------
    # TEST 7 — OpenCV comparison
    # ---------------------------------------------------------------

    H_cv = cv2.getPerspectiveTransform(
        source[:4].astype(np.float32),
        destination[:4].astype(np.float32)
    )

    H_numpy = estimate_homography_dlt(
        source[:4],
        destination[:4]
    )

    test_points = np.array([
        [20, 20],
        [80, 30],
        [50, 90]
    ], dtype=np.float64)

    numpy_result = transform_points(
        test_points,
        H_numpy
    )

    cv_result = cv2.perspectiveTransform(
        test_points.astype(np.float32).reshape(-1, 1, 2),
        H_cv
    ).reshape(-1, 2)

    assert np.allclose(
        numpy_result,
        cv_result,
        atol=1e-3
    )

    print("Test 7 passed: NumPy DLT vs OpenCV.")

    # ---------------------------------------------------------------
    # TEST 8 — RANSAC with outliers
    # ---------------------------------------------------------------

    np.random.seed(123)

    inlier_source = np.random.uniform(
        0,
        200,
        size=(40, 2)
    )

    H_ground_truth = np.array([
        [1.1, 0.05, 15],
        [-0.02, 0.95, 25],
        [0.0003, -0.0002, 1]
    ])

    inlier_destination = transform_points(
        inlier_source,
        H_ground_truth
    )

    inlier_destination += np.random.normal(
        0,
        0.15,
        inlier_destination.shape
    )

    outlier_source = np.random.uniform(
        0,
        200,
        size=(10, 2)
    )

    outlier_destination = np.random.uniform(
        0,
        300,
        size=(10, 2)
    )

    all_source = np.vstack([
        inlier_source,
        outlier_source
    ])

    all_destination = np.vstack([
        inlier_destination,
        outlier_destination
    ])

    result = estimate_homography_ransac(
        all_source,
        all_destination,
        iterations=500,
        threshold=1.5
    )

    assert np.sum(result.inliers) >= 38

    print(
        "Test 8 passed: RANSAC rejected incorrect correspondences."
    )

    print(
        f"Detected inliers: {np.sum(result.inliers)}/{len(all_source)}"
    )

    # ---------------------------------------------------------------
    # TEST 9 — OpenCV RANSAC
    # ---------------------------------------------------------------

    H_cv_ransac, mask = cv2.findHomography(
        all_source.astype(np.float32),
        all_destination.astype(np.float32),
        cv2.RANSAC,
        1.5
    )

    assert H_cv_ransac is not None

    assert np.sum(mask) >= 38

    print("Test 9 passed: OpenCV RANSAC comparison.")

    print("\nALL LESSON 18 TESTS PASSED.")


# =====================================================================
# STEP 11 — VISUALIZATION
# =====================================================================

def visualize_results():

    document = create_document_image()

    scene, source, destination, H_true = create_perspective_scene(
        document
    )

    # Estimate H from four correspondences.

    H_estimated = estimate_homography_dlt(
        source,
        destination
    )

    # Recover front-facing document.

    recovered_numpy = warp_perspective_from_scratch(
        scene,
        H_estimated,
        WARP_WIDTH,
        WARP_HEIGHT
    )

    recovered_cv = cv2.warpPerspective(
        scene,
        H_estimated,
        (WARP_WIDTH, WARP_HEIGHT)
    )

    # Transform the four document corners.

    projected_corners = transform_points(
        source,
        H_estimated
    )

    # OpenCV comparison.

    H_cv = cv2.getPerspectiveTransform(
        source.astype(np.float32),
        destination.astype(np.float32)
    )

    recovered_cv_reference = cv2.warpPerspective(
        scene,
        np.linalg.inv(H_cv),
        (WARP_WIDTH, WARP_HEIGHT)
    )

    # ---------------------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 10)
    )

    axes[0, 0].imshow(
        cv2.cvtColor(document, cv2.COLOR_BGR2RGB)
    )

    axes[0, 0].set_title("Original Document")

    axes[0, 1].imshow(
        cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
    )

    axes[0, 1].scatter(
        destination[:, 0],
        destination[:, 1],
        c="red",
        s=50
    )

    axes[0, 1].set_title("Perspective Scene")

    axes[0, 2].imshow(
        cv2.cvtColor(recovered_numpy.astype(np.uint8), cv2.COLOR_BGR2RGB)
    )

    axes[0, 2].set_title("Recovered — NumPy From Scratch")

    axes[1, 0].imshow(
        cv2.cvtColor(recovered_cv, cv2.COLOR_BGR2RGB)
    )

    axes[1, 0].set_title("Recovered — OpenCV")

    axes[1, 1].imshow(
        cv2.cvtColor(recovered_cv_reference, cv2.COLOR_BGR2RGB)
    )

    axes[1, 1].set_title("OpenCV Reference")

    error = np.abs(
        recovered_numpy.astype(np.float64) -
        recovered_cv.astype(np.float64)
    )

    axes[1, 2].imshow(
        np.mean(error, axis=2),
        cmap="inferno"
    )

    axes[1, 2].set_title("Absolute Difference")

    for ax in axes.ravel():

        ax.axis("off")

    plt.tight_layout()

    plt.show()

    print("\nEstimated homography:")

    print(H_estimated)

    print("\nCorner reprojection error:")

    predicted = transform_points(
        source,
        H_estimated
    )

    errors = np.linalg.norm(
        predicted - destination,
        axis=1
    )

    print(errors)

    print(
        "\nMean corner error:",
        np.mean(errors)
    )


# =====================================================================
# STEP 12 — ROBOTICS APPLICATION
# =====================================================================

def robotics_conveyor_demo():

    """
    Demonstrate image-to-conveyor coordinate mapping.

    We assume the conveyor is a flat plane.

    The four reference points represent known conveyor coordinates.
    """

    print("\n" + "=" * 70)

    print("OPENARM V2 — CONVEYOR LOCALIZATION DEMO")

    print("=" * 70)

    # Four image reference points.

    image_points = np.array([
        [120, 100],
        [520, 120],
        [550, 400],
        [90, 380]
    ], dtype=np.float64)

    # Corresponding conveyor coordinates in meters.

    conveyor_points = np.array([
        [0.0, 0.0],
        [0.8, 0.0],
        [0.8, 0.5],
        [0.0, 0.5]
    ], dtype=np.float64)

    # Estimate image-to-conveyor homography.

    H_image_to_conveyor = estimate_homography_dlt(
        image_points,
        conveyor_points
    )

    # Simulated detected waste-object centers.

    detected_objects = np.array([
        [250, 230],
        [400, 280],
        [160, 300]
    ], dtype=np.float64)

    # Map pixels into conveyor coordinates.

    world_positions = transform_points(
        detected_objects,
        H_image_to_conveyor
    )

    object_names = [
        "Plastic cup",
        "Bottle",
        "Paper waste"
    ]

    for name, pixel, world in zip(
        object_names,
        detected_objects,
        world_positions
    ):

        print(f"\nObject: {name}")

        print(
            f"Image position: ({pixel[0]:.1f}, {pixel[1]:.1f}) pixels"
        )

        print(
            f"Conveyor position: "
            f"X={world[0]:.3f} m, Y={world[1]:.3f} m"
        )

    print(
        "\nNOTE: These coordinates are illustrative. "
        "A real robot requires measured reference points, "
        "correct coordinate-frame conventions, and validation."
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)

    print("COMPUTER VISION — LESSON 18")

    print("PERSPECTIVE TRANSFORMATION AND HOMOGRAPHY")

    print("=" * 70)

    run_tests()

    visualize_results()

    robotics_conveyor_demo()

    print("\nLESSON 18 COMPLETED.")


if __name__ == "__main__":

    main()