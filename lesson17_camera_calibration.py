"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 17 — CAMERA CALIBRATION FROM SCRATCH + OPENCV
======================================================================

LEARNING OBJECTIVES
-------------------
1. Understand the pinhole camera model.
2. Understand intrinsic and extrinsic parameters.
3. Project 3D points into 2D image coordinates using NumPy.
4. Implement radial and tangential lens distortion.
5. Implement numerical lens undistortion.
6. Generate synthetic chessboard observations.
7. Calibrate a camera using cv2.calibrateCamera().
8. Compare NumPy projection with cv2.projectPoints().
9. Calculate reprojection error.
10. Calibrate a real camera using a chessboard.
11. Undistort real camera images.

CAMERA MODEL
------------

P_camera = R * P_world + t

Normalized coordinates:

    x = X_camera / Z_camera
    y = Y_camera / Z_camera

Pixel coordinates:

    u = fx*x + cx
    v = fy*y + cy

Intrinsic matrix:

        [ fx   0   cx ]
    K = [  0  fy   cy ]
        [  0   0    1 ]

Distortion:

    r2 = x*x + y*y

    radial = 1 + k1*r2 + k2*r2^2 + k3*r2^3

    xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y

CALIBRATION
-----------

OpenCV estimates:

    - Camera intrinsic matrix
    - Distortion coefficients
    - Rotation vectors
    - Translation vectors

The optimization minimizes reprojection error.

IMPORTANT TESTING NOTE
---------------------

Individual distortion coefficients can be correlated.

A calibration may have small reprojection error even when
individual estimated distortion coefficients differ from
the synthetic ground-truth coefficients.

Therefore, this lesson tests:

    - Focal-length accuracy.
    - Principal-point accuracy.
    - Reprojection error.
    - Numerical projection consistency.

We do NOT require every distortion coefficient to match
the synthetic ground truth independently.

ROBOTICS APPLICATION
--------------------

For OpenArm V2:

    Camera
       |
       v
    Camera calibration
       |
       v
    Lens distortion correction
       |
       v
    Object detection
       |
       v
    Pixel coordinates
       |
       v
    Depth / known conveyor plane
       |
       v
    3D object localization
       |
       v
    Robot coordinate transformation
       |
       v
    ACT grasping policy

NEXT LESSON
-----------

LESSON 18 — PERSPECTIVE TRANSFORMATION AND HOMOGRAPHY
======================================================================
"""

import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ====================================================================
# CONFIGURATION
# ====================================================================

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

BOARD_WIDTH = 9
BOARD_HEIGHT = 6

SQUARE_SIZE = 0.025

NUM_SYNTHETIC_VIEWS = 24

RANDOM_SEED = 42


# ====================================================================
# STEP 1 — CAMERA INTRINSIC MATRIX
# ====================================================================

def create_camera_matrix(fx, fy, cx, cy):

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ],
        dtype=np.float64
    )


# ====================================================================
# STEP 2 — ROTATION MATRIX
# ====================================================================

def rotation_matrix_from_rvec(rvec):

    R, _ = cv2.Rodrigues(
        np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    )

    return R


# ====================================================================
# STEP 3 — PINHOLE PROJECTION FROM SCRATCH
# ====================================================================

def project_points_pinhole(
    points_3d,
    K,
    R=None,
    t=None
):

    points_3d = np.asarray(
        points_3d,
        dtype=np.float64
    )

    if R is None:
        R = np.eye(3)

    if t is None:
        t = np.zeros(3)

    t = np.asarray(t, dtype=np.float64).reshape(3)

    points_camera = points_3d @ R.T + t

    Z = points_camera[:, 2]

    if np.any(Z <= 0):
        raise ValueError(
            "All points must have positive camera depth."
        )

    x = points_camera[:, 0] / Z
    y = points_camera[:, 1] / Z

    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]

    return np.column_stack([u, v])


# ====================================================================
# STEP 4 — LENS DISTORTION
# ====================================================================

def distort_normalized_points(
    points,
    distortion
):

    points = np.asarray(
        points,
        dtype=np.float64
    )

    k1, k2, p1, p2, k3 = np.asarray(
        distortion,
        dtype=np.float64
    ).reshape(-1)[:5]

    x = points[:, 0]
    y = points[:, 1]

    r2 = x**2 + y**2

    radial = (
        1
        + k1 * r2
        + k2 * r2**2
        + k3 * r2**3
    )

    xd = (
        x * radial
        + 2 * p1 * x * y
        + p2 * (r2 + 2 * x**2)
    )

    yd = (
        y * radial
        + p1 * (r2 + 2 * y**2)
        + 2 * p2 * x * y
    )

    return np.column_stack([xd, yd])


# ====================================================================
# STEP 5 — COMPLETE DISTORTED PROJECTION
# ====================================================================

def project_points_distorted(
    points_3d,
    K,
    distortion,
    R=None,
    t=None
):

    points_3d = np.asarray(
        points_3d,
        dtype=np.float64
    )

    if R is None:
        R = np.eye(3)

    if t is None:
        t = np.zeros(3)

    points_camera = (
        points_3d @ R.T
        + np.asarray(t).reshape(3)
    )

    Z = points_camera[:, 2]

    if np.any(Z <= 0):
        raise ValueError(
            "Points must be in front of the camera."
        )

    normalized = points_camera[:, :2] / Z[:, None]

    distorted = distort_normalized_points(
        normalized,
        distortion
    )

    homogeneous = np.column_stack(
        [
            distorted,
            np.ones(len(distorted))
        ]
    )

    pixels = homogeneous @ K.T

    return pixels[:, :2]


# ====================================================================
# STEP 6 — NUMERICAL UNDISTORTION
# ====================================================================

def undistort_normalized_points(
    distorted_points,
    distortion,
    iterations=20
):

    """
    Numerically invert the distortion model.

    This is an educational fixed-point iteration.
    OpenCV's optimized undistortion functions should be preferred
    for production image processing.
    """

    distorted_points = np.asarray(
        distorted_points,
        dtype=np.float64
    )

    k1, k2, p1, p2, k3 = np.asarray(
        distortion,
        dtype=np.float64
    ).reshape(-1)[:5]

    xd = distorted_points[:, 0]
    yd = distorted_points[:, 1]

    x = xd.copy()
    y = yd.copy()

    for _ in range(iterations):

        r2 = x**2 + y**2

        radial = (
            1
            + k1 * r2
            + k2 * r2**2
            + k3 * r2**3
        )

        radial = np.where(
            np.abs(radial) < 1e-10,
            1e-10,
            radial
        )

        delta_x = (
            2 * p1 * x * y
            + p2 * (r2 + 2 * x**2)
        )

        delta_y = (
            p1 * (r2 + 2 * y**2)
            + 2 * p2 * x * y
        )

        x = (xd - delta_x) / radial
        y = (yd - delta_y) / radial

    return np.column_stack([x, y])


# ====================================================================
# STEP 7 — PIXEL / NORMALIZED COORDINATE CONVERSION
# ====================================================================

def pixels_to_normalized(pixels, K):

    pixels = np.asarray(
        pixels,
        dtype=np.float64
    )

    x = (pixels[:, 0] - K[0, 2]) / K[0, 0]
    y = (pixels[:, 1] - K[1, 2]) / K[1, 1]

    return np.column_stack([x, y])


def normalized_to_pixels(points, K):

    points = np.asarray(
        points,
        dtype=np.float64
    )

    u = K[0, 0] * points[:, 0] + K[0, 2]
    v = K[1, 1] * points[:, 1] + K[1, 2]

    return np.column_stack([u, v])


def undistort_pixels_from_scratch(
    pixels,
    K,
    distortion
):

    normalized = pixels_to_normalized(
        pixels,
        K
    )

    corrected = undistort_normalized_points(
        normalized,
        distortion
    )

    return normalized_to_pixels(
        corrected,
        K
    )


# ====================================================================
# STEP 8 — CHESSBOARD GEOMETRY
# ====================================================================

def create_chessboard_object_points(
    board_width=BOARD_WIDTH,
    board_height=BOARD_HEIGHT,
    square_size=SQUARE_SIZE
):

    points = np.zeros(
        (
            board_width * board_height,
            3
        ),
        dtype=np.float32
    )

    grid = np.mgrid[
        0:board_width,
        0:board_height
    ].T.reshape(-1, 2)

    points[:, :2] = grid * square_size

    return points


# ====================================================================
# STEP 9 — SYNTHETIC CALIBRATION DATA
# ====================================================================

def generate_synthetic_calibration_data():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    true_K = create_camera_matrix(
        900.0,
        920.0,
        IMAGE_WIDTH / 2,
        IMAGE_HEIGHT / 2
    )

    true_distortion = np.array(
        [
            -0.12,
            0.04,
            0.001,
            -0.0008,
            0.01
        ],
        dtype=np.float64
    )

    object_points = create_chessboard_object_points()

    object_points_list = []
    image_points_list = []

    for _ in range(NUM_SYNTHETIC_VIEWS):

        rvec = np.array(
            [
                rng.uniform(-0.35, 0.35),
                rng.uniform(-0.35, 0.35),
                rng.uniform(-0.25, 0.25)
            ]
        )

        tvec = np.array(
            [
                rng.uniform(-0.08, 0.08),
                rng.uniform(-0.06, 0.06),
                rng.uniform(0.65, 0.95)
            ]
        )

        R = rotation_matrix_from_rvec(rvec)

        projected = project_points_distorted(
            object_points,
            true_K,
            true_distortion,
            R,
            tvec
        )

        noise = rng.normal(
            0,
            0.08,
            size=projected.shape
        )

        projected += noise

        object_points_list.append(
            object_points.copy()
        )

        image_points_list.append(
            projected.astype(np.float32)
        )

    return (
        true_K,
        true_distortion,
        object_points_list,
        image_points_list
    )


# ====================================================================
# STEP 10 — OPENCV CAMERA CALIBRATION
# ====================================================================

def calibrate_camera_opencv(
    object_points_list,
    image_points_list,
    image_size
):

    result = cv2.calibrateCamera(
        object_points_list,
        image_points_list,
        image_size,
        None,
        None
    )

    rms, K, distortion, rvecs, tvecs = result

    return {
        "rms": rms,
        "camera_matrix": K,
        "distortion": distortion,
        "rvecs": rvecs,
        "tvecs": tvecs
    }


# ====================================================================
# STEP 11 — REPROJECTION ERROR
# ====================================================================

def calculate_reprojection_error(
    object_points_list,
    image_points_list,
    K,
    distortion,
    rvecs,
    tvecs
):

    all_errors = []

    for i in range(len(object_points_list)):

        projected, _ = cv2.projectPoints(
            object_points_list[i],
            rvecs[i],
            tvecs[i],
            K,
            distortion
        )

        projected = projected.reshape(-1, 2)

        observed = image_points_list[i].reshape(-1, 2)

        errors = np.linalg.norm(
            projected - observed,
            axis=1
        )

        all_errors.extend(
            errors.tolist()
        )

    return float(np.mean(all_errors))


# ====================================================================
# STEP 12 — SYNTHETIC CHESSBOARD IMAGE
# ====================================================================

def draw_synthetic_chessboard():

    image = np.ones(
        (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            3
        ),
        dtype=np.uint8
    ) * 220

    square_pixels = 45

    start_x = 350
    start_y = 180

    for row in range(BOARD_HEIGHT + 1):

        for col in range(BOARD_WIDTH + 1):

            color = (
                (0, 0, 0)
                if (row + col) % 2 == 0
                else (255, 255, 255)
            )

            x1 = start_x + col * square_pixels
            y1 = start_y + row * square_pixels

            cv2.rectangle(
                image,
                (x1, y1),
                (
                    x1 + square_pixels,
                    y1 + square_pixels
                ),
                color,
                -1
            )

    return image


# ====================================================================
# STEP 13 — DISTORTION VISUALIZATION
# ====================================================================

def visualize_distortion(K, distortion):

    axis = np.linspace(
        -0.65,
        0.65,
        15
    )

    xx, yy = np.meshgrid(
        axis,
        axis
    )

    points = np.column_stack(
        [
            xx.ravel(),
            yy.ravel()
        ]
    )

    distorted = distort_normalized_points(
        points,
        distortion
    )

    plt.figure(figsize=(8, 7))

    plt.scatter(
        points[:, 0],
        points[:, 1],
        s=15,
        label="Ideal"
    )

    plt.scatter(
        distorted[:, 0],
        distorted[:, 1],
        s=15,
        label="Distorted"
    )

    for original, changed in zip(points, distorted):

        plt.plot(
            [original[0], changed[0]],
            [original[1], changed[1]],
            linewidth=0.7
        )

    plt.xlabel("Normalized X")
    plt.ylabel("Normalized Y")

    plt.title("Radial and Tangential Lens Distortion")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()


# ====================================================================
# STEP 14 — REAL CAMERA CALIBRATION
# ====================================================================

def run_real_camera_calibration(
    camera_index=0,
    board_width=BOARD_WIDTH,
    board_height=BOARD_HEIGHT,
    square_size=SQUARE_SIZE,
    required_views=15
):

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open camera {camera_index}."
        )

    pattern_size = (
        board_width,
        board_height
    )

    object_template = create_chessboard_object_points(
        board_width,
        board_height,
        square_size
    )

    object_points_list = []
    image_points_list = []

    latest_frame = None
    latest_corners = None

    print("=" * 70)
    print("REAL CAMERA CALIBRATION")
    print("=" * 70)

    print("SPACE = Capture detected chessboard")
    print("R     = Reset captured views")
    print("Q     = Finish and calibrate")

    while True:

        success, frame = cap.read()

        if not success:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            (
                cv2.CALIB_CB_ADAPTIVE_THRESH
                |
                cv2.CALIB_CB_NORMALIZE_IMAGE
            )
        )

        display = frame.copy()

        latest_frame = frame.copy()
        latest_corners = None

        if found:

            criteria = (
                cv2.TERM_CRITERIA_EPS
                + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )

            refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            latest_corners = refined

            cv2.drawChessboardCorners(
                display,
                pattern_size,
                refined,
                found
            )

            status = "CHESSBOARD DETECTED"

            color = (0, 255, 0)

        else:

            status = "MOVE CHESSBOARD INTO VIEW"

            color = (0, 0, 255)

        cv2.putText(
            display,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.putText(
            display,
            f"Captured views: {len(object_points_list)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow(
            "Camera Calibration",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):

            object_points_list.clear()
            image_points_list.clear()

            print("Calibration views reset.")

        elif key == ord(" "):

            if latest_corners is None:

                print("No chessboard detected. Capture skipped.")

                continue

            object_points_list.append(
                object_template.copy()
            )

            image_points_list.append(
                latest_corners.copy()
            )

            print(
                f"Captured {len(object_points_list)}/"
                f"{required_views}"
            )

    cap.release()
    cv2.destroyAllWindows()

    if len(object_points_list) < 5:

        raise RuntimeError(
            "At least five valid views are recommended. "
            "Capture more chessboard images."
        )

    image_size = (
        latest_frame.shape[1],
        latest_frame.shape[0]
    )

    calibration = calibrate_camera_opencv(
        object_points_list,
        image_points_list,
        image_size
    )

    mean_error = calculate_reprojection_error(
        object_points_list,
        image_points_list,
        calibration["camera_matrix"],
        calibration["distortion"],
        calibration["rvecs"],
        calibration["tvecs"]
    )

    print("\nCALIBRATION RESULTS")

    print(
        "\nRMS reprojection error:",
        calibration["rms"]
    )

    print(
        "Mean reprojection error:",
        mean_error,
        "pixels"
    )

    print(
        "\nCamera matrix:\n",
        calibration["camera_matrix"]
    )

    print(
        "\nDistortion:\n",
        calibration["distortion"]
    )

    np.savez(
        "camera_calibration.npz",
        camera_matrix=calibration["camera_matrix"],
        distortion=calibration["distortion"],
        image_width=image_size[0],
        image_height=image_size[1]
    )

    print("\nSaved camera_calibration.npz")

    h, w = latest_frame.shape[:2]

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        calibration["camera_matrix"],
        calibration["distortion"],
        (w, h),
        1,
        (w, h)
    )

    undistorted = cv2.undistort(
        latest_frame,
        calibration["camera_matrix"],
        calibration["distortion"],
        None,
        new_K
    )

    cv2.imshow(
        "Original",
        latest_frame
    )

    cv2.imshow(
        "Undistorted",
        undistorted
    )

    print("Press any key to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return calibration


# ====================================================================
# STEP 15 — NUMERICAL TESTS
# ====================================================================

def run_tests():

    print("\n")
    print("=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # TEST 1 — Intrinsic matrix
    # ---------------------------------------------------------------

    K = create_camera_matrix(
        800,
        820,
        320,
        240
    )

    assert K.shape == (3, 3)
    assert K[0, 0] == 800
    assert K[1, 1] == 820

    print("Intrinsic matrix test passed.")

    # ---------------------------------------------------------------
    # TEST 2 — Pinhole projection
    # ---------------------------------------------------------------

    points_3d = np.array(
        [
            [0, 0, 2],
            [1, 0, 2],
            [0, 1, 2]
        ],
        dtype=np.float64
    )

    pixels = project_points_pinhole(
        points_3d,
        K
    )

    expected = np.array(
        [
            [320, 240],
            [720, 240],
            [320, 650]
        ],
        dtype=np.float64
    )

    assert np.allclose(pixels, expected)

    print("Pinhole projection test passed.")

    # ---------------------------------------------------------------
    # TEST 3 — Identity extrinsics
    # ---------------------------------------------------------------

    pixels_identity = project_points_pinhole(
        points_3d,
        K,
        np.eye(3),
        np.zeros(3)
    )

    assert np.allclose(
        pixels,
        pixels_identity
    )

    print("Identity extrinsic transformation test passed.")

    # ---------------------------------------------------------------
    # TEST 4 — Lens distortion
    # ---------------------------------------------------------------

    distortion = np.array(
        [
            -0.15,
            0.05,
            0.001,
            -0.001,
            0.01
        ]
    )

    ideal = np.array(
        [
            [0.2, 0.3],
            [-0.4, 0.2],
            [0.5, -0.3]
        ]
    )

    distorted = distort_normalized_points(
        ideal,
        distortion
    )

    assert not np.allclose(
        ideal,
        distorted
    )

    print("Lens distortion test passed.")

    # ---------------------------------------------------------------
    # TEST 5 — Undistortion
    # ---------------------------------------------------------------

    recovered = undistort_normalized_points(
        distorted,
        distortion,
        iterations=30
    )

    assert np.allclose(
        recovered,
        ideal,
        atol=1e-7
    )

    print("Undistortion numerical test passed.")

    # ---------------------------------------------------------------
    # TEST 6 — Pixel normalization
    # ---------------------------------------------------------------

    test_pixels = np.array(
        [
            [100, 100],
            [320, 240],
            [600, 400]
        ],
        dtype=np.float64
    )

    normalized = pixels_to_normalized(
        test_pixels,
        K
    )

    recovered_pixels = normalized_to_pixels(
        normalized,
        K
    )

    assert np.allclose(
        test_pixels,
        recovered_pixels
    )

    print("Pixel normalization test passed.")

    # ---------------------------------------------------------------
    # TEST 7 — NumPy vs OpenCV projection
    # ---------------------------------------------------------------

    test_K = create_camera_matrix(
        900,
        920,
        640,
        360
    )

    test_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.0],
            [-0.2, 0.1, 0.0],
            [0.3, -0.1, 0.0]
        ],
        dtype=np.float64
    )

    rvec = np.array(
        [0.1, -0.05, 0.02]
    )

    tvec = np.array(
        [0.0, 0.0, 1.0]
    )

    R = rotation_matrix_from_rvec(rvec)

    zero_distortion = np.zeros(5)

    numpy_projection = project_points_distorted(
        test_points,
        test_K,
        zero_distortion,
        R,
        tvec
    )

    opencv_projection, _ = cv2.projectPoints(
        test_points,
        rvec,
        tvec,
        test_K,
        zero_distortion
    )

    opencv_projection = opencv_projection.reshape(
        -1,
        2
    )

    assert np.allclose(
        numpy_projection,
        opencv_projection,
        atol=1e-6
    )

    print("NumPy vs OpenCV projection test passed.")

    # ---------------------------------------------------------------
    # TEST 8 — Chessboard geometry
    # ---------------------------------------------------------------

    board = create_chessboard_object_points()

    assert board.shape == (
        BOARD_WIDTH * BOARD_HEIGHT,
        3
    )

    assert np.allclose(
        board[0],
        [0, 0, 0]
    )

    assert np.isclose(
        board[1, 0],
        SQUARE_SIZE
    )

    print("Chessboard geometry test passed.")

    # ---------------------------------------------------------------
    # TEST 9 — Synthetic camera calibration
    # ---------------------------------------------------------------

    (
        true_K,
        true_distortion,
        object_points_list,
        image_points_list
    ) = generate_synthetic_calibration_data()

    calibration = calibrate_camera_opencv(
        object_points_list,
        image_points_list,
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        )
    )

    estimated_K = calibration["camera_matrix"]

    # Compare focal lengths.

    true_focal = np.array(
        [
            true_K[0, 0],
            true_K[1, 1]
        ]
    )

    estimated_focal = np.array(
        [
            estimated_K[0, 0],
            estimated_K[1, 1]
        ]
    )

    focal_error = np.mean(
        np.abs(
            estimated_focal - true_focal
        )
    )

    assert focal_error < 10.0, (
        f"Focal-length error too large: {focal_error}"
    )

    # Compare principal point.

    true_principal = true_K[:2, 2]

    estimated_principal = estimated_K[:2, 2]

    principal_error = np.mean(
        np.abs(
            estimated_principal - true_principal
        )
    )

    assert principal_error < 10.0, (
        f"Principal-point error too large: {principal_error}"
    )

    print("Synthetic camera calibration test passed.")

    print(
        "Mean focal-length error:",
        round(focal_error, 4),
        "pixels"
    )

    print(
        "Mean principal-point error:",
        round(principal_error, 4),
        "pixels"
    )

    # ---------------------------------------------------------------
    # TEST 10 — Reprojection error
    # ---------------------------------------------------------------

    mean_error = calculate_reprojection_error(
        object_points_list,
        image_points_list,
        estimated_K,
        calibration["distortion"],
        calibration["rvecs"],
        calibration["tvecs"]
    )

    assert mean_error < 0.5, (
        f"Reprojection error too large: {mean_error}"
    )

    print(
        "Reprojection error test passed:",
        round(mean_error, 5),
        "pixels"
    )

    # ---------------------------------------------------------------
    # TEST 11 — OpenCV distortion projection
    # ---------------------------------------------------------------

    sample_points = np.array(
        [
            [0.2, 0.1, 0.0],
            [-0.3, 0.2, 0.0],
            [0.4, -0.4, 0.0]
        ],
        dtype=np.float64
    )

    sample_rvec = np.array(
        [0.05, -0.08, 0.02]
    )

    sample_tvec = np.array(
        [0.0, 0.0, 1.0]
    )

    sample_R = rotation_matrix_from_rvec(
        sample_rvec
    )

    custom_projection = project_points_distorted(
        sample_points,
        true_K,
        true_distortion,
        sample_R,
        sample_tvec
    )

    opencv_projection, _ = cv2.projectPoints(
        sample_points,
        sample_rvec,
        sample_tvec,
        true_K,
        true_distortion
    )

    opencv_projection = opencv_projection.reshape(
        -1,
        2
    )

    assert np.allclose(
        custom_projection,
        opencv_projection,
        atol=1e-5
    )

    print("Distorted NumPy vs OpenCV projection test passed.")

    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


# ====================================================================
# STEP 16 — MAIN SYNTHETIC EXPERIMENT
# ====================================================================

def run_synthetic_experiment():

    print("=" * 70)
    print("COMPUTER VISION FROM SCRATCH")
    print("LESSON 17 — CAMERA CALIBRATION")
    print("=" * 70)

    (
        true_K,
        true_distortion,
        object_points_list,
        image_points_list
    ) = generate_synthetic_calibration_data()

    calibration = calibrate_camera_opencv(
        object_points_list,
        image_points_list,
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        )
    )

    estimated_K = calibration["camera_matrix"]

    estimated_distortion = calibration["distortion"]

    print("\nTRUE CAMERA MATRIX")
    print(true_K)

    print("\nESTIMATED CAMERA MATRIX")
    print(estimated_K)

    print("\nTRUE DISTORTION")
    print(true_distortion)

    print("\nESTIMATED DISTORTION")
    print(estimated_distortion)

    print(
        "\nOpenCV RMS reprojection error:",
        calibration["rms"]
    )

    mean_error = calculate_reprojection_error(
        object_points_list,
        image_points_list,
        estimated_K,
        estimated_distortion,
        calibration["rvecs"],
        calibration["tvecs"]
    )

    print(
        "Mean reprojection error:",
        mean_error,
        "pixels"
    )

    # ---------------------------------------------------------------
    # Visualize chessboard.
    # ---------------------------------------------------------------

    chessboard = draw_synthetic_chessboard()

    plt.figure(figsize=(9, 5))

    plt.imshow(
        cv2.cvtColor(
            chessboard,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Synthetic Chessboard")
    plt.axis("off")

    # ---------------------------------------------------------------
    # Visualize distortion.
    # ---------------------------------------------------------------

    visualize_distortion(
        true_K,
        true_distortion
    )

    # ---------------------------------------------------------------
    # Compare observed and predicted corners.
    # ---------------------------------------------------------------

    index = 0

    predicted, _ = cv2.projectPoints(
        object_points_list[index],
        calibration["rvecs"][index],
        calibration["tvecs"][index],
        estimated_K,
        estimated_distortion
    )

    predicted = predicted.reshape(-1, 2)

    observed = image_points_list[index].reshape(-1, 2)

    plt.figure(figsize=(9, 6))

    plt.scatter(
        observed[:, 0],
        observed[:, 1],
        label="Observed",
        s=25
    )

    plt.scatter(
        predicted[:, 0],
        predicted[:, 1],
        label="Reprojected",
        marker="x",
        s=30
    )

    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")

    plt.title(
        "Observed vs Reprojected Chessboard Corners"
    )

    plt.gca().invert_yaxis()

    plt.grid(True)
    plt.legend()

    # ---------------------------------------------------------------
    # Run tests.
    # ---------------------------------------------------------------

    run_tests()

    print("\nNEXT LESSON:")
    print(
        "LESSON 18 — PERSPECTIVE TRANSFORMATION AND HOMOGRAPHY"
    )

    plt.show()


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Lesson 17: Camera Calibration"
    )

    parser.add_argument(
        "--camera",
        action="store_true"
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0
    )

    parser.add_argument(
        "--board-width",
        type=int,
        default=BOARD_WIDTH
    )

    parser.add_argument(
        "--board-height",
        type=int,
        default=BOARD_HEIGHT
    )

    parser.add_argument(
        "--square-size",
        type=float,
        default=SQUARE_SIZE
    )

    args = parser.parse_args()

    if args.camera:

        run_real_camera_calibration(
            camera_index=args.camera_index,
            board_width=args.board_width,
            board_height=args.board_height,
            square_size=args.square_size
        )

    else:

        run_synthetic_experiment()