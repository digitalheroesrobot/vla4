"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 58 — RGB-D COMPUTER VISION
======================================================================

FILENAME
--------
lesson58_rgbd_computer_vision.py

LESSON OBJECTIVES
-----------------
1. Understand RGB-D camera output.
2. Understand camera intrinsic parameters.
3. Convert raw depth values into meters.
4. Back-project image pixels into 3-D camera coordinates.
5. Project 3-D points back into image coordinates.
6. Apply rigid transformations between coordinate frames.
7. Generate colored point clouds from RGB-D images.
8. Filter invalid depth measurements.
9. Perform voxel-grid point-cloud downsampling.
10. Estimate robust object depth from a segmentation mask or ROI.
11. Export a point cloud as a PLY file.
12. Visualize RGB, depth, and 3-D geometry.

======================================================================
1. WHAT IS AN RGB-D CAMERA?
======================================================================

An RGB-D camera provides:

    RGB image:
        I(u, v) = [R, G, B]

    Depth image:
        D(u, v) = distance measurement

For example:

    RGB:
        A bottle is visible in the image.

    Depth:
        The bottle surface is approximately 0.65 meters
        from the camera.

Combining these images allows us to estimate 3-D coordinates.

Common RGB-D camera technologies include:

    - Stereo depth cameras.
    - Structured-light cameras.
    - Time-of-Flight (ToF) cameras.
    - Active infrared depth cameras.

IMPORTANT:

    Depth data must be aligned with the RGB image before combining
    RGB colors with 3-D points.

======================================================================
2. CAMERA INTRINSIC PARAMETERS
======================================================================

A pinhole camera model describes how a 3-D point is projected
onto a 2-D image.

Camera coordinate point:

    P = [X, Y, Z]^T

Image coordinate:

    p = [u, v]^T

Intrinsic matrix:

              [ fx   0   cx ]
        K  =  [  0  fy   cy ]
              [  0   0    1 ]

Where:

    fx:
        Horizontal focal length in pixels.

    fy:
        Vertical focal length in pixels.

    cx:
        Principal point x-coordinate.

    cy:
        Principal point y-coordinate.

The projection equations are:

    u = fx * X / Z + cx

    v = fy * Y / Z + cy

These equations assume a pinhole camera model and that lens
distortion has already been corrected or is negligible.

======================================================================
3. BACK-PROJECTION: 2-D TO 3-D
======================================================================

If the depth image provides axial camera depth Z:

    X = (u - cx) * Z / fx

    Y = (v - cy) * Z / fy

    Z = depth(u, v)

Therefore:

    P_camera = [X, Y, Z]^T

Example:

    fx = 600 pixels
    fy = 600 pixels
    cx = 320 pixels
    cy = 240 pixels

    Pixel:
        u = 400
        v = 300

    Depth:
        Z = 0.8 meters

Then:

    X = (400 - 320) * 0.8 / 600
      = 0.1067 meters

    Y = (300 - 240) * 0.8 / 600
      = 0.0800 meters

    Z = 0.8 meters

The resulting point is:

    [0.1067, 0.0800, 0.8000] meters

Coordinate convention used in this lesson:

    X:
        Right.

    Y:
        Down.

    Z:
        Forward, away from the camera.

IMPORTANT:

    Some depth sensors report radial range rather than axial Z.
    This lesson assumes axial Z depth. Radial range must first be
    converted to axial depth using the camera ray direction.

======================================================================
4. DEPTH UNITS
======================================================================

Many depth cameras store depth in unsigned 16-bit integers.

Example:

    Raw depth = 850

If one unit represents one millimeter:

    Z = 850 * 0.001

    Z = 0.85 meters

For a floating-point depth image already measured in meters:

    depth_scale = 1.0

Never assume the depth unit without checking the camera or dataset
documentation.

======================================================================
5. 3-D COORDINATE TRANSFORMATIONS
======================================================================

Camera coordinates are not robot coordinates.

A robot typically needs object coordinates in its base frame.

Rigid transformation:

    P_robot = R * P_camera + t

Homogeneous transformation:

             [ R  t ]
    T      = [      ]
             [ 0  1 ]

    P_robot_h = T_camera_to_robot * P_camera_h

Where:

    R:
        3x3 rotation matrix.

    t:
        3x1 translation vector.

A valid rigid transformation must have:

    R^T R = I

    det(R) = +1

The transformation must be calibrated for the actual robot and
camera installation.

======================================================================
6. POINT CLOUD GENERATION
======================================================================

Each valid depth pixel becomes one 3-D point.

    RGB-D pixel
        |
        v
    Depth value
        |
        v
    Back-projection
        |
        v
    [X, Y, Z]
        |
        v
    Point cloud

A colored point cloud stores:

    X, Y, Z, R, G, B

A point cloud can represent:

    - Object geometry.
    - Surface shape.
    - Approximate object dimensions.
    - Object location in camera coordinates.

======================================================================
7. DEPTH FILTERING
======================================================================

Depth sensors contain noise.

Typical errors:

    - Invalid zero-depth measurements.
    - Reflective surfaces.
    - Transparent objects.
    - Motion artifacts.
    - Missing depth near object boundaries.

This lesson includes:

    - Invalid-depth filtering.
    - Range filtering.
    - Optional median filtering.
    - Voxel-grid downsampling.

Median filtering can reduce isolated depth noise, but it may
remove thin structures and smooth object boundaries.

======================================================================
8. VOXEL-GRID DOWNSAMPLING
======================================================================

A point cloud can contain hundreds of thousands of points.

Voxel filtering divides 3-D space into cubic cells.

For voxel size s:

    voxel_index = floor(point / s)

All points inside the same voxel are replaced by their average.

This reduces point count while approximately preserving geometry.

======================================================================
9. ROBOTICS APPLICATION — OPENARM V2
======================================================================

Waste-sorting pipeline:

    RGB camera
        |
        v
    Object detection / segmentation
        |
        v
    Object mask
        |
        v
    RGB-D depth image
        |
        v
    Masked depth extraction
        |
        v
    3-D point cloud
        |
        v
    Object position in camera frame
        |
        v
    Camera-to-robot transformation
        |
        v
    Grasp planning
        |
        v
    OpenArm V2

For a planar conveyor, a calibrated homography may be sufficient
for some localization tasks.

For objects with significant height or irregular geometry,
RGB-D point clouds provide more useful 3-D information.

======================================================================
10. INPUT FORMAT
======================================================================

Provide a registered RGB image and depth image:

    rgb.png
    depth.png

Supported depth inputs:

    - 16-bit PNG.
    - Single-channel image readable by OpenCV.
    - NumPy .npy array.

Example command:

    python lesson58_rgbd_computer_vision.py \
        --rgb rgb.png \
        --depth depth.png \
        --fx 600 \
        --fy 600 \
        --cx 320 \
        --cy 240 \
        --depth-scale 0.001

The intrinsic values above are examples only.
Use the actual camera calibration values for your camera.

Optional:

    --mask object_mask.png

or:

    --roi 100 80 200 250

The ROI format is:

    x y width height

======================================================================
11. OUTPUT
======================================================================

    runs/lesson58/
        rgbd_visualization.png
        point_cloud.ply
        rgbd_summary.json

======================================================================
12. TESTING
======================================================================

    python lesson58_rgbd_computer_vision.py --check-only

Tests cover:

    - Camera intrinsics.
    - Depth conversion.
    - Invalid depth handling.
    - Pixel back-projection.
    - 3-D projection.
    - Projection/back-projection consistency.
    - Rigid transformations.
    - Inverse transformations.
    - Point-cloud construction.
    - Voxel downsampling.
    - ROI selection.
    - Robust depth statistics.
    - PLY export.

======================================================================
NEXT LESSON
======================================================================

Lesson 59 — Point Clouds

We will study point-cloud geometry, coordinate transformations,
filtering, nearest-neighbor search, normals, and 3-D visualization.

======================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

from dataclasses import dataclass

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ======================================================================
# CONFIGURATION
# ======================================================================

DEFAULT_OUTPUT_DIR = os.path.join(
    "runs",
    "lesson58",
)


# ======================================================================
# CAMERA INTRINSICS
# ======================================================================

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        values = [
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        ]

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "Camera intrinsics must be finite."
            )

        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(
                "Focal lengths must be positive."
            )

    def matrix(self):
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


# ======================================================================
# DEPTH CONVERSION
# ======================================================================

def depth_to_meters(
    depth,
    depth_scale=0.001,
):
    """
    Convert raw depth values into meters.

    Example:

        uint16 depth = 850
        scale = 0.001

        result = 0.85 meters

    Invalid values are represented as zero.
    """

    depth = np.asarray(
        depth
    )

    if depth.ndim != 2:
        raise ValueError(
            "Depth image must be two-dimensional."
        )

    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError(
            "Depth scale must be finite and positive."
        )

    result = (
        depth.astype(np.float32)
        * float(depth_scale)
    )

    result[
        ~np.isfinite(result)
    ] = 0.0

    result[
        result < 0
    ] = 0.0

    return result


# ======================================================================
# DEPTH VALIDATION AND FILTERING
# ======================================================================

def valid_depth_mask(
    depth_m,
    min_depth=0.1,
    max_depth=5.0,
):
    depth_m = np.asarray(
        depth_m,
        dtype=np.float32,
    )

    if depth_m.ndim != 2:
        raise ValueError(
            "Depth image must be two-dimensional."
        )

    if (
        not np.isfinite(min_depth)
        or not np.isfinite(max_depth)
        or min_depth < 0
        or max_depth <= min_depth
    ):
        raise ValueError(
            "Invalid depth range."
        )

    return (
        np.isfinite(depth_m)
        & (depth_m >= min_depth)
        & (depth_m <= max_depth)
    )


def filter_depth(
    depth_m,
    min_depth=0.1,
    max_depth=5.0,
    median_kernel=0,
):
    """
    Apply range filtering and optional median filtering.

    Invalid measurements remain zero.

    median_kernel:
        0 = no median filtering
        3 = 3x3 median filter
        5 = 5x5 median filter
    """

    depth_m = np.asarray(
        depth_m,
        dtype=np.float32,
    )

    if depth_m.ndim != 2:
        raise ValueError(
            "Depth image must be two-dimensional."
        )

    valid = valid_depth_mask(
        depth_m,
        min_depth,
        max_depth,
    )

    result = np.where(
        valid,
        depth_m,
        0.0,
    ).astype(np.float32)

    if median_kernel:
        if median_kernel < 3 or median_kernel % 2 == 0:
            raise ValueError(
                "Median kernel must be an odd integer >= 3."
            )

        filtered = cv2.medianBlur(
            result,
            median_kernel,
        )

        # Preserve pixels that were originally invalid.
        filtered[~valid] = 0.0

        result = filtered.astype(
            np.float32
        )

    return result


# ======================================================================
# PIXEL BACK-PROJECTION
# ======================================================================

def backproject_pixel(
    u,
    v,
    z,
    intrinsics,
):
    """
    Convert one image pixel and axial depth into camera coordinates.

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth
    """

    u = float(u)
    v = float(v)
    z = float(z)

    if not np.all(
        np.isfinite([u, v, z])
    ):
        raise ValueError(
            "Pixel and depth values must be finite."
        )

    if z <= 0:
        raise ValueError(
            "Depth must be positive."
        )

    x = (
        (u - intrinsics.cx)
        * z
        / intrinsics.fx
    )

    y = (
        (v - intrinsics.cy)
        * z
        / intrinsics.fy
    )

    return np.array(
        [x, y, z],
        dtype=np.float64,
    )


def backproject_depth(
    depth_m,
    intrinsics,
    mask=None,
    min_depth=0.1,
    max_depth=5.0,
):
    """
    Back-project an entire depth image.

    Returns:

        points:
            H x W x 3 camera-coordinate array.

        valid:
            H x W boolean mask.

    Invalid points are filled with NaN.
    """

    depth_m = np.asarray(
        depth_m,
        dtype=np.float32,
    )

    if depth_m.ndim != 2:
        raise ValueError(
            "Depth image must be two-dimensional."
        )

    height, width = depth_m.shape

    valid = valid_depth_mask(
        depth_m,
        min_depth,
        max_depth,
    )

    if mask is not None:
        mask = np.asarray(mask)

        if mask.shape != depth_m.shape:
            raise ValueError(
                "Segmentation mask dimensions must match depth."
            )

        valid &= (
            mask > 0
        )

    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    z = depth_m

    x = (
        (u - intrinsics.cx)
        * z
        / intrinsics.fx
    )

    y = (
        (v - intrinsics.cy)
        * z
        / intrinsics.fy
    )

    points = np.stack(
        [
            x,
            y,
            z,
        ],
        axis=-1,
    ).astype(np.float32)

    points[~valid] = np.nan

    return points, valid


# ======================================================================
# 3-D PROJECTION
# ======================================================================

def project_points(
    points,
    intrinsics,
):
    """
    Project 3-D camera-coordinate points into image coordinates.

    Returns:

        pixels:
            (..., 2) array of [u, v].

        valid:
            Boolean array indicating positive finite depth.
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if points.shape[-1] != 3:
        raise ValueError(
            "Points must have a final dimension of 3."
        )

    z = points[..., 2]

    valid = (
        np.isfinite(points).all(axis=-1)
        & (z > 0)
    )

    safe_z = np.where(
        valid,
        z,
        1.0,
    )

    u = (
        intrinsics.fx
        * points[..., 0]
        / safe_z
        + intrinsics.cx
    )

    v = (
        intrinsics.fy
        * points[..., 1]
        / safe_z
        + intrinsics.cy
    )

    pixels = np.stack(
        [u, v],
        axis=-1,
    )

    pixels[~valid] = np.nan

    return pixels, valid


# ======================================================================
# RIGID TRANSFORMATIONS
# ======================================================================

def make_transform(
    rotation,
    translation,
):
    """
    Build a 4x4 homogeneous transformation matrix.

        [ R  t ]
        [ 0  1 ]
    """

    rotation = np.asarray(
        rotation,
        dtype=np.float64,
    )

    translation = np.asarray(
        translation,
        dtype=np.float64,
    )

    if rotation.shape != (3, 3):
        raise ValueError(
            "Rotation must have shape (3, 3)."
        )

    if translation.shape != (3,):
        raise ValueError(
            "Translation must have shape (3,)."
        )

    if not np.all(
        np.isfinite(rotation)
    ) or not np.all(
        np.isfinite(translation)
    ):
        raise ValueError(
            "Transformation values must be finite."
        )

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-5,
    ):
        raise ValueError(
            "Rotation matrix must be orthonormal."
        )

    if not np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Rotation matrix determinant must be +1."
        )

    transform = np.eye(
        4,
        dtype=np.float64,
    )

    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform


def transform_points(
    points,
    transform,
):
    """
    Apply a homogeneous transformation to 3-D points.

        P_target = T_target_source * P_source
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    transform = np.asarray(
        transform,
        dtype=np.float64,
    )

    if points.shape[-1] != 3:
        raise ValueError(
            "Points must have a final dimension of 3."
        )

    if transform.shape != (4, 4):
        raise ValueError(
            "Transform must have shape (4, 4)."
        )

    if not np.all(
        np.isfinite(transform)
    ):
        raise ValueError(
            "Transform must be finite."
        )

    flat = points.reshape(
        -1,
        3,
    )

    homogeneous = np.concatenate(
        [
            flat,
            np.ones(
                (len(flat), 1),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    transformed = (
        transform @ homogeneous.T
    ).T[:, :3]

    return transformed.reshape(
        points.shape
    )


def invert_transform(transform):
    """
    Invert a rigid homogeneous transformation.
    """

    transform = np.asarray(
        transform,
        dtype=np.float64,
    )

    if transform.shape != (4, 4):
        raise ValueError(
            "Transform must have shape (4, 4)."
        )

    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(
        4,
        dtype=np.float64,
    )

    inverse[:3, :3] = rotation.T

    inverse[:3, 3] = (
        -rotation.T @ translation
    )

    return inverse


# ======================================================================
# POINT CLOUD GENERATION
# ======================================================================

def build_point_cloud(
    rgb,
    depth_m,
    intrinsics,
    mask=None,
    min_depth=0.1,
    max_depth=5.0,
    stride=1,
):
    """
    Generate a colored point cloud.

    Returns:

        points:
            Nx3 float32 camera coordinates.

        colors:
            Nx3 uint8 RGB colors.
    """

    rgb = np.asarray(
        rgb
    )

    depth_m = np.asarray(
        depth_m,
        dtype=np.float32,
    )

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            "RGB image must have shape (H, W, 3)."
        )

    if rgb.shape[:2] != depth_m.shape:
        raise ValueError(
            "RGB and depth images must be aligned and have "
            "identical dimensions."
        )

    if stride < 1:
        raise ValueError(
            "Stride must be positive."
        )

    points_image, valid = backproject_depth(
        depth_m,
        intrinsics,
        mask=mask,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    sampled_points = points_image[
        ::stride,
        ::stride,
    ]

    sampled_rgb = rgb[
        ::stride,
        ::stride,
    ]

    sampled_valid = valid[
        ::stride,
        ::stride,
    ]

    points = sampled_points[
        sampled_valid
    ]

    colors = sampled_rgb[
        sampled_valid
    ]

    return (
        points.astype(np.float32),
        colors.astype(np.uint8),
    )


# ======================================================================
# VOXEL GRID DOWNSAMPLING
# ======================================================================

def voxel_downsample(
    points,
    colors=None,
    voxel_size=0.01,
):
    """
    Replace points inside each voxel with their mean position.

    If colors are provided, average colors in each voxel too.
    """

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "Points must have shape (N, 3)."
        )

    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError(
            "Voxel size must be positive."
        )

    valid = np.isfinite(
        points
    ).all(axis=1)

    points = points[
        valid
    ]

    if colors is not None:
        colors = np.asarray(
            colors
        )

        if colors.shape != (
            len(valid),
            3,
        ):
            raise ValueError(
                "Colors must have shape (N, 3)."
            )

        colors = colors[
            valid
        ]

    if len(points) == 0:
        if colors is None:
            return points

        return (
            points,
            colors,
        )

    voxel_indices = np.floor(
        points / voxel_size
    ).astype(np.int64)

    unique_voxels, inverse = np.unique(
        voxel_indices,
        axis=0,
        return_inverse=True,
    )

    number_of_voxels = len(
        unique_voxels
    )

    downsampled_points = np.zeros(
        (number_of_voxels, 3),
        dtype=np.float64,
    )

    counts = np.bincount(
        inverse,
        minlength=number_of_voxels,
    ).astype(np.float64)

    for dimension in range(3):
        downsampled_points[:, dimension] = (
            np.bincount(
                inverse,
                weights=points[:, dimension],
                minlength=number_of_voxels,
            )
            / counts
        )

    downsampled_points = downsampled_points.astype(
        np.float32
    )

    if colors is None:
        return downsampled_points

    downsampled_colors = np.zeros(
        (number_of_voxels, 3),
        dtype=np.float64,
    )

    for dimension in range(3):
        downsampled_colors[:, dimension] = (
            np.bincount(
                inverse,
                weights=colors[:, dimension].astype(np.float64),
                minlength=number_of_voxels,
            )
            / counts
        )

    downsampled_colors = np.clip(
        np.rint(downsampled_colors),
        0,
        255,
    ).astype(np.uint8)

    return (
        downsampled_points,
        downsampled_colors,
    )


# ======================================================================
# ROI AND OBJECT DEPTH STATISTICS
# ======================================================================

def make_roi_mask(
    image_shape,
    roi,
):
    """
    Create a binary mask from:

        x, y, width, height
    """

    height, width = image_shape[:2]

    x, y, roi_width, roi_height = map(
        int,
        roi,
    )

    if (
        x < 0
        or y < 0
        or roi_width <= 0
        or roi_height <= 0
        or x + roi_width > width
        or y + roi_height > height
    ):
        raise ValueError(
            "ROI is outside the image."
        )

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    mask[
        y:y + roi_height,
        x:x + roi_width,
    ] = 255

    return mask


def robust_depth_statistics(
    depth_m,
    mask,
    min_depth=0.1,
    max_depth=5.0,
):
    """
    Calculate robust depth statistics for a selected object region.

    Median is preferred over mean when depth contains outliers.
    """

    depth_m = np.asarray(
        depth_m,
        dtype=np.float32,
    )

    mask = np.asarray(
        mask
    )

    if depth_m.ndim != 2 or mask.shape != depth_m.shape:
        raise ValueError(
            "Depth and mask must have identical 2-D dimensions."
        )

    valid = (
        valid_depth_mask(
            depth_m,
            min_depth,
            max_depth,
        )
        & (mask > 0)
    )

    values = depth_m[
        valid
    ]

    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


# ======================================================================
# PLY EXPORT
# ======================================================================

def write_ply(
    path,
    points,
    colors=None,
):
    """
    Write an ASCII PLY point cloud.

    Coordinates are stored in meters.

    Colors are stored as uint8 RGB values.
    """

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "Points must have shape (N, 3)."
        )

    valid = np.isfinite(
        points
    ).all(axis=1)

    points = points[
        valid
    ]

    if colors is not None:
        colors = np.asarray(
            colors
        )

        if colors.shape != (
            len(valid),
            3,
        ):
            raise ValueError(
                "Colors must have shape (N, 3)."
            )

        colors = colors[
            valid
        ]

    os.makedirs(
        os.path.dirname(os.path.abspath(path)),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "ply\n"
        )

        file.write(
            "format ascii 1.0\n"
        )

        file.write(
            f"element vertex {len(points)}\n"
        )

        file.write(
            "property float x\n"
        )

        file.write(
            "property float y\n"
        )

        file.write(
            "property float z\n"
        )

        if colors is not None:
            file.write(
                "property uchar red\n"
            )

            file.write(
                "property uchar green\n"
            )

            file.write(
                "property uchar blue\n"
            )

        file.write(
            "end_header\n"
        )

        if colors is None:
            for point in points:
                file.write(
                    f"{point[0]:.6f} "
                    f"{point[1]:.6f} "
                    f"{point[2]:.6f}\n"
                )

        else:
            for point, color in zip(
                points,
                colors,
            ):
                file.write(
                    f"{point[0]:.6f} "
                    f"{point[1]:.6f} "
                    f"{point[2]:.6f} "
                    f"{int(color[0])} "
                    f"{int(color[1])} "
                    f"{int(color[2])}\n"
                )


# ======================================================================
# FILE INPUT
# ======================================================================

def load_rgb_image(path):
    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Cannot load RGB image: {path}"
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def load_depth_image(path):
    extension = os.path.splitext(
        path
    )[1].lower()

    if extension == ".npy":
        depth = np.load(
            path
        )
    else:
        depth = cv2.imread(
            path,
            cv2.IMREAD_UNCHANGED,
        )

    if depth is None:
        raise RuntimeError(
            f"Cannot load depth image: {path}"
        )

    if depth.ndim != 2:
        raise ValueError(
            "Depth input must be a single-channel image."
        )

    return depth


def load_mask_image(path):
    mask = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise RuntimeError(
            f"Cannot load mask image: {path}"
        )

    return (
        mask > 0
    ).astype(np.uint8) * 255


def load_extrinsics(path):
    """
    Load a 4x4 camera-to-robot transformation matrix from JSON.

    Example JSON:

    {
        "camera_to_robot": [
            [1, 0, 0, 0.2],
            [0, 1, 0, 0.1],
            [0, 0, 1, 0.5],
            [0, 0, 0, 1]
        ]
    }

    The values above are only an example, not real calibration.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    matrix = np.asarray(
        data["camera_to_robot"],
        dtype=np.float64,
    )

    if matrix.shape != (4, 4):
        raise ValueError(
            "camera_to_robot must be a 4x4 matrix."
        )

    if not np.allclose(
        matrix[3],
        [0, 0, 0, 1],
        atol=1e-6,
    ):
        raise ValueError(
            "Invalid homogeneous transformation last row."
        )

    # Validate the rigid rotation component.
    make_transform(
        matrix[:3, :3],
        matrix[:3, 3],
    )

    return matrix


# ======================================================================
# VISUALIZATION
# ======================================================================

def visualize_rgbd(
    rgb,
    depth_m,
    points,
    colors,
    output_path,
    max_points=20000,
):
    """
    Create a three-panel visualization:

        1. RGB image.
        2. Depth image.
        3. Colored 3-D point cloud.
    """

    if max_points < 1:
        raise ValueError(
            "max_points must be positive."
        )

    fig = plt.figure(
        figsize=(17, 6),
    )

    ax1 = fig.add_subplot(
        1,
        3,
        1,
    )

    ax1.imshow(
        rgb
    )

    ax1.set_title(
        "RGB image"
    )

    ax1.axis(
        "off"
    )

    ax2 = fig.add_subplot(
        1,
        3,
        2,
    )

    depth_display = np.ma.masked_where(
        depth_m <= 0,
        depth_m,
    )

    image = ax2.imshow(
        depth_display,
        cmap="viridis",
    )

    ax2.set_title(
        "Depth (meters)"
    )

    fig.colorbar(
        image,
        ax=ax2,
        fraction=0.046,
        pad=0.04,
    )

    ax3 = fig.add_subplot(
        1,
        3,
        3,
        projection="3d",
    )

    if len(points) > 0:
        count = min(
            len(points),
            max_points,
        )

        rng = np.random.default_rng(
            7
        )

        indices = rng.choice(
            len(points),
            size=count,
            replace=False,
        )

        sampled_points = points[
            indices
        ]

        sampled_colors = colors[
            indices
        ] / 255.0

        ax3.scatter(
            sampled_points[:, 0],
            sampled_points[:, 1],
            sampled_points[:, 2],
            c=sampled_colors,
            s=1,
            depthshade=False,
        )

    ax3.set_title(
        "Colored point cloud"
    )

    ax3.set_xlabel(
        "X (m)"
    )

    ax3.set_ylabel(
        "Y (m)"
    )

    ax3.set_zlabel(
        "Z (m)"
    )

    ax3.invert_yaxis()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ======================================================================
# MAIN RGB-D PROCESSING
# ======================================================================

def process_rgbd(args):
    start_time = time.perf_counter()

    rgb = load_rgb_image(
        args.rgb
    )

    raw_depth = load_depth_image(
        args.depth
    )

    if rgb.shape[:2] != raw_depth.shape:
        raise ValueError(
            "RGB and depth dimensions differ. "
            "Register or align the depth image to RGB first. "
            "This script does not silently resize depth."
        )

    depth_m = depth_to_meters(
        raw_depth,
        depth_scale=args.depth_scale,
    )

    depth_m = filter_depth(
        depth_m,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        median_kernel=args.median_kernel,
    )

    selection_mask = None

    if args.mask is not None:
        selection_mask = load_mask_image(
            args.mask
        )

        if selection_mask.shape != depth_m.shape:
            raise ValueError(
                "Segmentation mask dimensions must match RGB/depth."
            )

    elif args.roi is not None:
        selection_mask = make_roi_mask(
            depth_m.shape,
            args.roi,
        )

    intrinsics = CameraIntrinsics(
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
    )

    points, colors = build_point_cloud(
        rgb,
        depth_m,
        intrinsics,
        mask=selection_mask,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        stride=args.stride,
    )

    if len(points) == 0:
        raise RuntimeError(
            "No valid 3-D points were generated. "
            "Check the depth scale, depth range, and selection mask."
        )

    original_point_count = len(
        points
    )

    if args.voxel_size > 0:
        points, colors = voxel_downsample(
            points,
            colors,
            voxel_size=args.voxel_size,
        )

    camera_points = points.copy()

    robot_points = None

    if args.extrinsics is not None:
        transform = load_extrinsics(
            args.extrinsics
        )

        robot_points = transform_points(
            camera_points,
            transform,
        )

    depth_mask = (
        selection_mask
        if selection_mask is not None
        else np.ones(
            depth_m.shape,
            dtype=np.uint8,
        )
    )

    depth_stats = robust_depth_statistics(
        depth_m,
        depth_mask,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    ply_path = os.path.join(
        args.output_dir,
        "point_cloud.ply",
    )

    write_ply(
        ply_path,
        points,
        colors,
    )

    visualization_path = os.path.join(
        args.output_dir,
        "rgbd_visualization.png",
    )

    visualize_rgbd(
        rgb,
        depth_m,
        points,
        colors,
        visualization_path,
        max_points=args.max_visualization_points,
    )

    cloud_center = np.mean(
        points,
        axis=0,
    )

    cloud_min = np.min(
        points,
        axis=0,
    )

    cloud_max = np.max(
        points,
        axis=0,
    )

    summary = {
        "lesson": "Lesson 58 - RGB-D Computer Vision",
        "rgb_file": args.rgb,
        "depth_file": args.depth,
        "depth_scale": float(args.depth_scale),
        "depth_units_after_conversion": "meters",
        "camera_intrinsics": {
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.cx),
            "cy": float(intrinsics.cy),
        },
        "frame_width": int(rgb.shape[1]),
        "frame_height": int(rgb.shape[0]),
        "depth_statistics": depth_stats,
        "point_count_before_downsampling": int(
            original_point_count
        ),
        "point_count_after_downsampling": int(
            len(points)
        ),
        "voxel_size_meters": float(
            args.voxel_size
        ),
        "camera_point_cloud": {
            "centroid_xyz_meters": cloud_center.tolist(),
            "minimum_xyz_meters": cloud_min.tolist(),
            "maximum_xyz_meters": cloud_max.tolist(),
        },
        "robot_point_cloud": (
            {
                "centroid_xyz_meters": np.mean(
                    robot_points,
                    axis=0,
                ).tolist(),
                "minimum_xyz_meters": np.min(
                    robot_points,
                    axis=0,
                ).tolist(),
                "maximum_xyz_meters": np.max(
                    robot_points,
                    axis=0,
                ).tolist(),
            }
            if robot_points is not None
            else None
        ),
        "point_cloud_file": ply_path,
        "visualization_file": visualization_path,
        "processing_seconds": float(
            time.perf_counter() - start_time
        ),
    }

    summary_path = os.path.join(
        args.output_dir,
        "rgbd_summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\n" + "=" * 68)
    print("RGB-D COMPUTER VISION SUMMARY")
    print("=" * 68)

    print(
        f"Image dimensions: {rgb.shape[1]} x {rgb.shape[0]}"
    )

    print(
        f"Valid depth pixels: {depth_stats['count']}"
    )

    print(
        f"Median depth: {depth_stats['median']}"
    )

    print(
        f"Points before downsampling: {original_point_count}"
    )

    print(
        f"Points after downsampling: {len(points)}"
    )

    print(
        "Point-cloud centroid in camera coordinates:"
    )

    print(
        cloud_center
    )

    if robot_points is not None:
        print(
            "Point-cloud centroid in robot coordinates:"
        )

        print(
            np.mean(robot_points, axis=0)
        )

    print(
        f"\nPoint cloud: {ply_path}"
    )

    print(
        f"Visualization: {visualization_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print("=" * 68)


# ======================================================================
# NUMERICAL TESTS
# ======================================================================

def run_tests():
    print("\nRunning Lesson 58 tests...")
    print("=" * 68)

    intrinsics = CameraIntrinsics(
        fx=600,
        fy=600,
        cx=320,
        cy=240,
    )

    # --------------------------------------------------------------
    # Test 1: intrinsic matrix
    # --------------------------------------------------------------

    expected = np.array(
        [
            [600, 0, 320],
            [0, 600, 240],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    assert np.allclose(
        intrinsics.matrix(),
        expected,
    )

    print("Test 01 passed: camera intrinsic matrix.")

    # --------------------------------------------------------------
    # Test 2: invalid focal length
    # --------------------------------------------------------------

    try:
        CameraIntrinsics(
            fx=0,
            fy=600,
            cx=320,
            cy=240,
        )

        raise AssertionError(
            "Invalid focal length accepted."
        )

    except ValueError:
        pass

    print("Test 02 passed: focal-length validation.")

    # --------------------------------------------------------------
    # Test 3: depth conversion
    # --------------------------------------------------------------

    raw = np.array(
        [
            [1000, 2000],
            [500, 0],
        ],
        dtype=np.uint16,
    )

    meters = depth_to_meters(
        raw,
        0.001,
    )

    assert np.allclose(
        meters,
        [
            [1.0, 2.0],
            [0.5, 0.0],
        ],
    )

    print("Test 03 passed: depth-to-meter conversion.")

    # --------------------------------------------------------------
    # Test 4: invalid depth scale
    # --------------------------------------------------------------

    try:
        depth_to_meters(
            raw,
            0,
        )

        raise AssertionError(
            "Invalid depth scale accepted."
        )

    except ValueError:
        pass

    print("Test 04 passed: depth-scale validation.")

    # --------------------------------------------------------------
    # Test 5: valid depth mask
    # --------------------------------------------------------------

    depth = np.array(
        [
            [0.0, 0.5],
            [1.0, 6.0],
        ],
        dtype=np.float32,
    )

    valid = valid_depth_mask(
        depth,
        min_depth=0.1,
        max_depth=5.0,
    )

    assert np.array_equal(
        valid,
        [
            [False, True],
            [True, False],
        ],
    )

    print("Test 05 passed: valid-depth filtering.")

    # --------------------------------------------------------------
    # Test 6: depth range filtering
    # --------------------------------------------------------------

    filtered = filter_depth(
        depth,
        min_depth=0.1,
        max_depth=5.0,
    )

    assert np.allclose(
        filtered,
        [
            [0.0, 0.5],
            [1.0, 0.0],
        ],
    )

    print("Test 06 passed: depth range filtering.")

    # --------------------------------------------------------------
    # Test 7: back-project principal point
    # --------------------------------------------------------------

    point = backproject_pixel(
        320,
        240,
        2.0,
        intrinsics,
    )

    assert np.allclose(
        point,
        [0, 0, 2],
    )

    print("Test 07 passed: principal-point back-projection.")

    # --------------------------------------------------------------
    # Test 8: back-project off-center pixel
    # --------------------------------------------------------------

    point = backproject_pixel(
        380,
        300,
        2.0,
        intrinsics,
    )

    assert np.allclose(
        point,
        [0.2, 0.2, 2.0],
    )

    print("Test 08 passed: off-center back-projection.")

    # --------------------------------------------------------------
    # Test 9: back-projection rejects zero depth
    # --------------------------------------------------------------

    try:
        backproject_pixel(
            320,
            240,
            0,
            intrinsics,
        )

        raise AssertionError(
            "Zero depth accepted."
        )

    except ValueError:
        pass

    print("Test 09 passed: invalid pixel depth rejection.")

    # --------------------------------------------------------------
    # Test 10: back-project full depth image
    # --------------------------------------------------------------

    depth = np.ones(
        (4, 5),
        dtype=np.float32,
    )

    points, valid = backproject_depth(
        depth,
        intrinsics,
        min_depth=0.1,
        max_depth=2.0,
    )

    assert points.shape == (4, 5, 3)
    assert valid.shape == (4, 5)
    assert np.all(valid)

    print("Test 10 passed: full-image back-projection.")

    # --------------------------------------------------------------
    # Test 11: full-image center pixel
    # --------------------------------------------------------------

    test_intrinsics = CameraIntrinsics(
        fx=2,
        fy=2,
        cx=2,
        cy=1,
    )

    depth = np.ones(
        (3, 5),
        dtype=np.float32,
    )

    points, valid = backproject_depth(
        depth,
        test_intrinsics,
    )

    assert np.allclose(
        points[1, 2],
        [0, 0, 1],
    )

    print("Test 11 passed: image center back-projection.")

    # --------------------------------------------------------------
    # Test 12: back-projection with selection mask
    # --------------------------------------------------------------

    selection = np.zeros(
        (3, 5),
        dtype=np.uint8,
    )

    selection[1, 2] = 255

    points, valid = backproject_depth(
        depth,
        test_intrinsics,
        mask=selection,
    )

    assert np.count_nonzero(valid) == 1

    print("Test 12 passed: masked back-projection.")

    # --------------------------------------------------------------
    # Test 13: projection
    # --------------------------------------------------------------

    point = np.array(
        [0.2, 0.2, 2.0],
    )

    pixel, valid = project_points(
        point,
        intrinsics,
    )

    assert bool(valid)
    assert np.allclose(
        pixel,
        [380, 300],
    )

    print("Test 13 passed: 3-D point projection.")

    # --------------------------------------------------------------
    # Test 14: projection/back-projection consistency
    # --------------------------------------------------------------

    original_point = np.array(
        [0.12, -0.08, 1.5],
    )

    pixel, valid = project_points(
        original_point,
        intrinsics,
    )

    recovered = backproject_pixel(
        pixel[0],
        pixel[1],
        original_point[2],
        intrinsics,
    )

    assert np.allclose(
        recovered,
        original_point,
        atol=1e-10,
    )

    print("Test 14 passed: projection/back-projection consistency.")

    # --------------------------------------------------------------
    # Test 15: invalid negative-depth projection
    # --------------------------------------------------------------

    pixels, valid = project_points(
        np.array(
            [
                [0, 0, -1],
                [0, 0, 1],
            ],
            dtype=np.float64,
        ),
        intrinsics,
    )

    assert np.array_equal(
        valid,
        [False, True],
    )

    assert np.isnan(
        pixels[0]
    ).all()

    print("Test 15 passed: invalid projection handling.")

    # --------------------------------------------------------------
    # Test 16: identity transformation
    # --------------------------------------------------------------

    identity = make_transform(
        np.eye(3),
        np.zeros(3),
    )

    points = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.float64,
    )

    transformed = transform_points(
        points,
        identity,
    )

    assert np.allclose(
        transformed,
        points,
    )

    print("Test 16 passed: identity transformation.")

    # --------------------------------------------------------------
    # Test 17: translation
    # --------------------------------------------------------------

    translation = np.array(
        [1, 2, 3],
    )

    transform = make_transform(
        np.eye(3),
        translation,
    )

    point = np.array(
        [0.5, 0.5, 0.5],
    )

    result = transform_points(
        point,
        transform,
    )

    assert np.allclose(
        result,
        [1.5, 2.5, 3.5],
    )

    print("Test 17 passed: rigid translation.")

    # --------------------------------------------------------------
    # Test 18: 90-degree rotation
    # --------------------------------------------------------------

    rotation = np.array(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    transform = make_transform(
        rotation,
        np.zeros(3),
    )

    result = transform_points(
        np.array([1, 0, 0]),
        transform,
    )

    assert np.allclose(
        result,
        [0, 1, 0],
    )

    print("Test 18 passed: 90-degree rotation.")

    # --------------------------------------------------------------
    # Test 19: inverse transformation
    # --------------------------------------------------------------

    transform = make_transform(
        rotation,
        np.array([0.5, 0.2, 0.1]),
    )

    inverse = invert_transform(
        transform
    )

    point = np.array(
        [0.2, 0.3, 0.4],
    )

    transformed = transform_points(
        point,
        transform,
    )

    recovered = transform_points(
        transformed,
        inverse,
    )

    assert np.allclose(
        recovered,
        point,
    )

    print("Test 19 passed: inverse transformation.")

    # --------------------------------------------------------------
    # Test 20: point-cloud generation
    # --------------------------------------------------------------

    rgb = np.zeros(
        (4, 5, 3),
        dtype=np.uint8,
    )

    rgb[:, :, 0] = 255

    depth = np.ones(
        (4, 5),
        dtype=np.float32,
    )

    test_intrinsics = CameraIntrinsics(
        fx=2,
        fy=2,
        cx=2,
        cy=1,
    )

    cloud, colors = build_point_cloud(
        rgb,
        depth,
        test_intrinsics,
        min_depth=0.1,
        max_depth=2.0,
    )

    assert cloud.shape == (20, 3)
    assert colors.shape == (20, 3)
    assert np.all(colors[:, 0] == 255)

    print("Test 20 passed: colored point-cloud generation.")

    # --------------------------------------------------------------
    # Test 21: voxel downsampling
    # --------------------------------------------------------------

    points = np.array(
        [
            [0.001, 0.001, 0.001],
            [0.002, 0.002, 0.002],
            [0.101, 0.101, 0.101],
        ],
        dtype=np.float32,
    )

    downsampled = voxel_downsample(
        points,
        voxel_size=0.05,
    )

    assert len(downsampled) == 2

    print("Test 21 passed: voxel-grid downsampling.")

    # --------------------------------------------------------------
    # Test 22: voxel color averaging
    # --------------------------------------------------------------

    points = np.array(
        [
            [0.001, 0.001, 0.001],
            [0.002, 0.002, 0.002],
        ],
        dtype=np.float32,
    )

    colors = np.array(
        [
            [100, 100, 100],
            [200, 200, 200],
        ],
        dtype=np.uint8,
    )

    down_points, down_colors = voxel_downsample(
        points,
        colors,
        voxel_size=0.05,
    )

    assert len(down_points) == 1
    assert np.allclose(
        down_colors[0],
        [150, 150, 150],
    )

    print("Test 22 passed: voxel color averaging.")

    # --------------------------------------------------------------
    # Test 23: ROI mask
    # --------------------------------------------------------------

    roi_mask = make_roi_mask(
        (20, 30),
        (5, 4, 10, 8),
    )

    assert mask_area(roi_mask) == 80

    print("Test 23 passed: ROI mask generation.")

    # --------------------------------------------------------------
    # Test 24: robust depth statistics
    # --------------------------------------------------------------

    depth = np.ones(
        (10, 10),
        dtype=np.float32,
    )

    depth[5, 5] = 4.0

    mask = np.ones(
        (10, 10),
        dtype=np.uint8,
    ) * 255

    statistics = robust_depth_statistics(
        depth,
        mask,
        min_depth=0.1,
        max_depth=5.0,
    )

    assert statistics["count"] == 100
    assert np.isclose(
        statistics["median"],
        1.0,
    )

    print("Test 24 passed: robust depth statistics.")

    # --------------------------------------------------------------
    # Test 25: empty depth statistics
    # --------------------------------------------------------------

    statistics = robust_depth_statistics(
        np.zeros((10, 10), dtype=np.float32),
        mask,
    )

    assert statistics["count"] == 0
    assert statistics["median"] is None

    print("Test 25 passed: empty depth statistics.")

    # --------------------------------------------------------------
    # Test 26: PLY export
    # --------------------------------------------------------------

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(
            directory,
            "test.ply",
        )

        points = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            dtype=np.float32,
        )

        colors = np.array(
            [
                [255, 0, 0],
                [0, 255, 0],
            ],
            dtype=np.uint8,
        )

        write_ply(
            path,
            points,
            colors,
        )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            content = file.read()

        assert "element vertex 2" in content
        assert "property uchar red" in content
        assert "255 0 0" in content

    print("Test 26 passed: PLY export.")

    # --------------------------------------------------------------
    # Test 27: invalid rigid rotation
    # --------------------------------------------------------------

    try:
        make_transform(
            np.zeros((3, 3)),
            np.zeros(3),
        )

        raise AssertionError(
            "Invalid rotation was accepted."
        )

    except ValueError:
        pass

    print("Test 27 passed: rigid-rotation validation.")

    # --------------------------------------------------------------
    # Test 28: RGB/depth shape mismatch
    # --------------------------------------------------------------

    try:
        build_point_cloud(
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.ones((8, 8), dtype=np.float32),
            intrinsics,
        )

        raise AssertionError(
            "Misaligned RGB-D images were accepted."
        )

    except ValueError:
        pass

    print("Test 28 passed: RGB-D alignment validation.")

    print("=" * 68)
    print("All 28 tests passed.")


# ======================================================================
# COMMAND LINE
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Lesson 58 - RGB-D Computer Vision"
        )
    )

    parser.add_argument(
        "--rgb",
        type=str,
        default=None,
        help="Path to RGB image.",
    )

    parser.add_argument(
        "--depth",
        type=str,
        default=None,
        help="Path to depth image or .npy file.",
    )

    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Optional object segmentation mask.",
    )

    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=None,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
    )

    parser.add_argument(
        "--fx",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--fy",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--cx",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--cy",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-depth",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--median-kernel",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Voxel size in meters; 0 disables downsampling.",
    )

    parser.add_argument(
        "--extrinsics",
        type=str,
        default=None,
        help="JSON camera-to-robot 4x4 transformation.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--max-visualization-points",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    return parser.parse_args()


# ======================================================================
# MAIN
# ======================================================================

def main():
    args = parse_args()

    if args.check_only:
        run_tests()
        return

    required = {
        "--rgb": args.rgb,
        "--depth": args.depth,
        "--fx": args.fx,
        "--fy": args.fy,
        "--cx": args.cx,
        "--cy": args.cy,
    }

    missing = [
        name
        for name, value in required.items()
        if value is None
    ]

    if missing:
        raise ValueError(
            "Missing required arguments: "
            + ", ".join(missing)
        )

    if args.mask is not None and args.roi is not None:
        raise ValueError(
            "Use either --mask or --roi, not both."
        )

    if args.stride < 1:
        raise ValueError(
            "--stride must be positive."
        )

    if args.voxel_size < 0:
        raise ValueError(
            "--voxel-size cannot be negative."
        )

    if args.max_visualization_points < 1:
        raise ValueError(
            "--max-visualization-points must be positive."
        )

    if (
        args.min_depth < 0
        or args.max_depth <= args.min_depth
    ):
        raise ValueError(
            "Invalid depth range."
        )

    run_tests()

    process_rgbd(args)


if __name__ == "__main__":
    main()