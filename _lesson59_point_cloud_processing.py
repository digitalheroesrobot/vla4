"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 59 — POINT CLOUD PROCESSING
======================================================================

FILENAME
--------
lesson59_point_cloud_processing.py

LESSON OBJECTIVES
-----------------
1. Understand 3-D point-cloud representations.
2. Load and save ASCII PLY point clouds.
3. Calculate centroids and geometric bounds.
4. Transform point clouds between coordinate frames.
5. Implement nearest-neighbor search.
6. Understand KD-tree acceleration.
7. Implement statistical outlier removal.
8. Implement radius-based outlier removal.
9. Implement voxel-grid downsampling.
10. Estimate surface normals using PCA.
11. Estimate local surface curvature.
12. Estimate a plane using RANSAC.
13. Estimate axis-aligned and oriented bounding boxes.
14. Visualize real point-cloud geometry.
15. Build a basic preprocessing pipeline for robotic perception.

======================================================================
1. WHAT IS A POINT CLOUD?
======================================================================

A point cloud is a collection of 3-D points.

    P = {p1, p2, p3, ..., pN}

Each point:

    pi = [xi, yi, zi]

A colored point cloud additionally stores:

    pi = [xi, yi, zi, Ri, Gi, Bi]

Point clouds can be generated using:

    - RGB-D cameras.
    - Stereo cameras.
    - LiDAR.
    - Photogrammetry.
    - 3-D scanners.

In Lesson 58, we generated a point cloud from RGB-D images.

Now we will process that point cloud.

======================================================================
2. POINT-CLOUD GEOMETRY
======================================================================

Centroid:

    C = (1/N) * sum(pi)

Axis-aligned bounding box:

    minimum = min(P, axis=0)

    maximum = max(P, axis=0)

Dimensions:

    size = maximum - minimum

These properties provide a simple geometric description of
an object.

======================================================================
3. NEAREST-NEIGHBOR SEARCH
======================================================================

For a query point q, find the closest point:

    nearest(q) = argmin ||q - pi||_2

Euclidean distance:

    d(q, pi) = sqrt(
        (qx-pix)^2 +
        (qy-piy)^2 +
        (qz-piz)^2
    )

Brute-force search:

    O(N) per query.

KD-tree:

    Spatially partitions points to accelerate nearest-neighbor
    queries.

We implement a brute-force reference algorithm and use SciPy's
cKDTree for practical processing.

======================================================================
4. STATISTICAL OUTLIER REMOVAL
======================================================================

Depth sensors sometimes generate isolated points.

For each point:

    1. Find its k nearest neighbors.
    2. Calculate the average neighbor distance.

        di = mean(distance to k neighbors)

Calculate global statistics:

    mu = mean(di)

    sigma = std(di)

Reject points satisfying:

    di > mu + alpha * sigma

Where:

    alpha = standard-deviation multiplier.

This method is useful for removing isolated measurements.

It can also remove legitimate sparse geometry, so its parameters
must be selected carefully.

======================================================================
5. RADIUS OUTLIER REMOVAL
======================================================================

For each point, count the number of neighbors within radius r.

    count_i = number of points within radius r

Reject if:

    count_i < minimum_neighbors

This is useful when the expected point density is reasonably
consistent.

======================================================================
6. VOXEL-GRID DOWNSAMPLING
======================================================================

Divide 3-D space into cubic voxels.

For voxel size s:

    voxel_index = floor(point / s)

Points in the same voxel are replaced by their average.

Advantages:

    - Lower memory usage.
    - Faster geometric processing.
    - Reduced redundant measurements.

Disadvantage:

    Small geometric details can disappear.

======================================================================
7. SURFACE NORMAL ESTIMATION
======================================================================

A surface normal describes the local orientation of a surface.

For each point:

    1. Find its neighboring points.
    2. Calculate their centroid.
    3. Calculate the covariance matrix.
    4. Perform eigenvalue decomposition.
    5. Select the eigenvector associated with the smallest
       eigenvalue.

Covariance:

    C = (1/N) * sum((pi - mean)(pi - mean)^T)

Eigenvalues:

    lambda0 <= lambda1 <= lambda2

The eigenvector associated with lambda0 approximates the
surface normal.

Why?

    A locally planar surface has little variation in its
    perpendicular direction.

IMPORTANT:

    The sign of an estimated normal is ambiguous.

    Both n and -n represent the same surface orientation.

    Normal orientation requires an additional viewpoint or
    orientation-consistency procedure.

======================================================================
8. CURVATURE ESTIMATION
======================================================================

A simple local surface variation measure:

    curvature = lambda0 / (lambda0 + lambda1 + lambda2)

Interpretation:

    Small curvature:
        Locally planar region.

    Larger curvature:
        More variation in the local neighborhood.

This is a local geometric descriptor, not a complete
differential-geometric curvature estimate.

======================================================================
9. RANSAC PLANE ESTIMATION
======================================================================

A plane is described by:

    ax + by + cz + d = 0

For a point p:

    distance = |a*x + b*y + c*z + d|

RANSAC:

    1. Randomly select three points.
    2. Estimate a plane.
    3. Count points within a distance threshold.
    4. Repeat.
    5. Keep the plane with the largest consensus set.

Applications:

    - Conveyor surface detection.
    - Tabletop detection.
    - Ground-plane removal.
    - Robot workspace segmentation.

======================================================================
10. BOUNDING BOXES
======================================================================

Axis-aligned bounding box:

    The box is aligned with global X, Y, and Z axes.

Oriented bounding box:

    The box is aligned with the principal directions of the
    point cloud.

We estimate an oriented box using PCA.

PCA-based OBB is an approximation. For irregular or symmetric
objects, its orientation may be ambiguous.

======================================================================
11. ROBOTICS APPLICATION — OPENARM V2
======================================================================

Example perception pipeline:

    RGB-D camera
        |
        v
    Object segmentation
        |
        v
    Object point cloud
        |
        v
    Invalid point removal
        |
        v
    Statistical outlier removal
        |
        v
    Voxel downsampling
        |
        v
    Surface normal estimation
        |
        v
    Bounding box / grasp geometry
        |
        v
    Camera-to-robot transformation
        |
        v
    Grasp planning
        |
        v
    OpenArm V2

For waste sorting:

    Bottle:
        Surface geometry and orientation.

    Can:
        Cylindrical shape estimation.

    Irregular waste:
        Local surface normals and geometric grasp candidates.

This lesson does not yet implement object classification or
grasp planning. Those require additional perception and
robot-coordinate reasoning.

======================================================================
12. INPUT
======================================================================

Use the PLY file from Lesson 58:

    runs/lesson58/point_cloud.ply

Run:

    python lesson59_point_cloud_processing.py \
        --ply runs/lesson58/point_cloud.ply

Optional parameters:

    --neighbors 20
    --std-ratio 2.0
    --voxel-size 0.01
    --normal-neighbors 20
    --plane-threshold 0.01

All distance parameters are measured in meters.

======================================================================
13. OUTPUT
======================================================================

    runs/lesson59/
        point_cloud_processing.png
        filtered_point_cloud.ply
        processing_summary.json

======================================================================
14. TESTS
======================================================================

    python lesson59_point_cloud_processing.py --check-only

Tests cover:

    - Centroid.
    - Bounding boxes.
    - Rigid transformations.
    - Inverse transformations.
    - Brute-force nearest neighbors.
    - KD-tree nearest neighbors.
    - Statistical outlier removal.
    - Radius outlier removal.
    - Voxel downsampling.
    - Surface normal estimation.
    - Curvature.
    - Plane estimation.
    - PLY export and loading.

======================================================================
NEXT LESSON
======================================================================

Lesson 60 — 3-D Object Detection

We will move from point-cloud geometry to detecting objects
in 3-D space.

======================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import cKDTree


# ======================================================================
# POINT CLOUD VALIDATION
# ======================================================================

def validate_points(points):
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "Point cloud must have shape (N, 3)."
        )

    return points


def remove_invalid_points(
    points,
    colors=None,
):
    points = validate_points(
        points
    )

    valid = np.isfinite(
        points
    ).all(axis=1)

    filtered_points = points[
        valid
    ]

    if colors is None:
        return filtered_points

    colors = np.asarray(
        colors
    )

    if colors.shape != (
        len(points),
        3,
    ):
        raise ValueError(
            "Colors must have shape (N, 3)."
        )

    filtered_colors = colors[
        valid
    ]

    return (
        filtered_points,
        filtered_colors,
    )


# ======================================================================
# BASIC GEOMETRY
# ======================================================================

def calculate_centroid(points):
    points = validate_points(
        points
    )

    points = points[
        np.isfinite(points).all(axis=1)
    ]

    if len(points) == 0:
        raise ValueError(
            "Cannot calculate centroid of an empty cloud."
        )

    return np.mean(
        points,
        axis=0,
    )


def calculate_aabb(points):
    """
    Axis-aligned bounding box.

    Returns:
        minimum
        maximum
        dimensions
        center
    """

    points = validate_points(
        points
    )

    points = points[
        np.isfinite(points).all(axis=1)
    ]

    if len(points) == 0:
        raise ValueError(
            "Cannot calculate bounding box of an empty cloud."
        )

    minimum = np.min(
        points,
        axis=0,
    )

    maximum = np.max(
        points,
        axis=0,
    )

    dimensions = maximum - minimum

    center = (
        minimum + maximum
    ) / 2.0

    return {
        "minimum": minimum,
        "maximum": maximum,
        "dimensions": dimensions,
        "center": center,
    }


def calculate_obb(points):
    """
    Estimate an oriented bounding box using PCA.

    The PCA axes are sorted by descending variance.

    This is a PCA-based approximation, not a minimum-volume
    oriented bounding box solver.
    """

    points = validate_points(
        points
    )

    points = points[
        np.isfinite(points).all(axis=1)
    ]

    if len(points) < 3:
        raise ValueError(
            "At least three points are required."
        )

    centroid = calculate_centroid(
        points
    )

    centered = points - centroid

    covariance = (
        centered.T @ centered
    ) / len(points)

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[
        order
    ]

    axes = eigenvectors[
        :, order
    ]

    # Ensure a right-handed coordinate frame.
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1

    local_points = (
        centered @ axes
    )

    local_min = np.min(
        local_points,
        axis=0,
    )

    local_max = np.max(
        local_points,
        axis=0,
    )

    dimensions = (
        local_max - local_min
    )

    local_center = (
        local_min + local_max
    ) / 2.0

    world_center = (
        centroid + axes @ local_center
    )

    return {
        "center": world_center,
        "axes": axes,
        "dimensions": dimensions,
        "eigenvalues": eigenvalues,
        "local_minimum": local_min,
        "local_maximum": local_max,
    }


# ======================================================================
# RIGID TRANSFORMATIONS
# ======================================================================

def make_transform(
    rotation,
    translation,
):
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
            "Rotation must be 3x3."
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
            "Transformation must be finite."
        )

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-5,
    ):
        raise ValueError(
            "Rotation must be orthonormal."
        )

    if not np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Rotation determinant must be +1."
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

    result = (
        transform @ homogeneous.T
    ).T[:, :3]

    return result.reshape(
        points.shape
    )


def invert_transform(transform):
    transform = np.asarray(
        transform,
        dtype=np.float64,
    )

    if transform.shape != (4, 4):
        raise ValueError(
            "Transform must be 4x4."
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
# NEAREST-NEIGHBOR SEARCH
# ======================================================================

def brute_force_nearest_neighbors(
    points,
    queries,
    k=1,
):
    """
    Reference nearest-neighbor implementation.

    Complexity:
        O(number_of_queries * number_of_points)

    This is useful for understanding the operation and testing
    a faster spatial index.
    """

    points = validate_points(
        points
    )

    queries = validate_points(
        queries
    )

    if len(points) == 0:
        raise ValueError(
            "Cannot search an empty point cloud."
        )

    if k < 1 or k > len(points):
        raise ValueError(
            "Invalid k."
        )

    distances = np.zeros(
        (len(queries), k),
        dtype=np.float64,
    )

    indices = np.zeros(
        (len(queries), k),
        dtype=np.int64,
    )

    for i, query in enumerate(
        queries
    ):
        squared = np.sum(
            (points - query) ** 2,
            axis=1,
        )

        nearest = np.argsort(
            squared
        )[:k]

        indices[i] = nearest

        distances[i] = np.sqrt(
            squared[nearest]
        )

    if k == 1:
        return (
            distances[:, 0],
            indices[:, 0],
        )

    return (
        distances,
        indices,
    )


def kdtree_nearest_neighbors(
    points,
    queries,
    k=1,
):
    """
    Efficient nearest-neighbor search using SciPy cKDTree.
    """

    points = validate_points(
        points
    )

    queries = validate_points(
        queries
    )

    if len(points) == 0:
        raise ValueError(
            "Cannot search an empty point cloud."
        )

    if k < 1 or k > len(points):
        raise ValueError(
            "Invalid k."
        )

    tree = cKDTree(
        points
    )

    distances, indices = tree.query(
        queries,
        k=k,
    )

    return (
        distances,
        indices,
    )


# ======================================================================
# STATISTICAL OUTLIER REMOVAL
# ======================================================================

def statistical_outlier_removal(
    points,
    neighbors=20,
    std_ratio=2.0,
):
    """
    Remove points with unusually large average neighbor distances.

    Returns:

        filtered_points
        inlier_mask
        average_neighbor_distances
        distance_threshold
    """

    points = validate_points(
        points
    )

    if not np.isfinite(std_ratio) or std_ratio < 0:
        raise ValueError(
            "std_ratio must be finite and non-negative."
        )

    valid = np.isfinite(
        points
    ).all(axis=1)

    clean = points[
        valid
    ]

    if len(clean) <= neighbors:
        raise ValueError(
            "Point cloud must contain more points than neighbors."
        )

    if neighbors < 1:
        raise ValueError(
            "neighbors must be positive."
        )

    tree = cKDTree(
        clean
    )

    distances, _ = tree.query(
        clean,
        k=neighbors + 1,
    )

    # First returned neighbor is the point itself.
    mean_distances = np.mean(
        distances[:, 1:],
        axis=1,
    )

    global_mean = float(
        np.mean(mean_distances)
    )

    global_std = float(
        np.std(mean_distances)
    )

    threshold = (
        global_mean
        + std_ratio * global_std
    )

    local_inliers = (
        mean_distances <= threshold
    )

    filtered = clean[
        local_inliers
    ]

    full_mask = np.zeros(
        len(points),
        dtype=bool,
    )

    full_mask[
        np.flatnonzero(valid)
    ] = local_inliers

    full_distances = np.full(
        len(points),
        np.nan,
        dtype=np.float64,
    )

    full_distances[
        np.flatnonzero(valid)
    ] = mean_distances

    return (
        filtered,
        full_mask,
        full_distances,
        threshold,
    )


# ======================================================================
# RADIUS OUTLIER REMOVAL
# ======================================================================

def radius_outlier_removal(
    points,
    radius=0.05,
    min_neighbors=5,
):
    """
    Keep points having at least min_neighbors other points
    within the specified radius.

    The point itself is not counted.
    """

    points = validate_points(
        points
    )

    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(
            "Radius must be positive."
        )

    if min_neighbors < 1:
        raise ValueError(
            "min_neighbors must be positive."
        )

    valid = np.isfinite(
        points
    ).all(axis=1)

    clean = points[
        valid
    ]

    tree = cKDTree(
        clean
    )

    neighbor_lists = tree.query_ball_point(
        clean,
        r=radius,
    )

    counts = np.array(
        [
            max(0, len(indices) - 1)
            for indices in neighbor_lists
        ],
        dtype=np.int64,
    )

    local_mask = (
        counts >= min_neighbors
    )

    full_mask = np.zeros(
        len(points),
        dtype=bool,
    )

    full_mask[
        np.flatnonzero(valid)
    ] = local_mask

    return (
        clean[local_mask],
        full_mask,
        counts,
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
    Average points inside each occupied voxel.

    Returns downsampled points and optional averaged colors.
    """

    points = validate_points(
        points
    )

    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError(
            "Voxel size must be positive."
        )

    valid = np.isfinite(
        points
    ).all(axis=1)

    clean = points[
        valid
    ]

    if colors is not None:
        colors = np.asarray(
            colors
        )

        if colors.shape != (
            len(points),
            3,
        ):
            raise ValueError(
                "Colors must have shape (N, 3)."
            )

        clean_colors = colors[
            valid
        ]

    if len(clean) == 0:
        if colors is None:
            return clean

        return (
            clean,
            clean_colors,
        )

    voxel_indices = np.floor(
        clean / voxel_size
    ).astype(np.int64)

    _, inverse = np.unique(
        voxel_indices,
        axis=0,
        return_inverse=True,
    )

    number_of_voxels = (
        inverse.max() + 1
    )

    counts = np.bincount(
        inverse,
        minlength=number_of_voxels,
    )

    downsampled = np.zeros(
        (number_of_voxels, 3),
        dtype=np.float64,
    )

    for dimension in range(3):
        downsampled[:, dimension] = (
            np.bincount(
                inverse,
                weights=clean[:, dimension],
                minlength=number_of_voxels,
            )
            / counts
        )

    downsampled = downsampled.astype(
        np.float32
    )

    if colors is None:
        return downsampled

    downsampled_colors = np.zeros(
        (number_of_voxels, 3),
        dtype=np.float64,
    )

    for dimension in range(3):
        downsampled_colors[:, dimension] = (
            np.bincount(
                inverse,
                weights=clean_colors[:, dimension].astype(float),
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
        downsampled,
        downsampled_colors,
    )


# ======================================================================
# SURFACE NORMAL ESTIMATION
# ======================================================================

def estimate_normals(
    points,
    neighbors=20,
):
    """
    Estimate local surface normals using PCA.

    For each point:

        1. Find k nearest neighbors.
        2. Calculate neighborhood centroid.
        3. Calculate covariance.
        4. Calculate eigenvectors.
        5. Select eigenvector with smallest eigenvalue.

    Returns:

        normals:
            Nx3 unit vectors.

        curvature:
            N scalar values.

        valid:
            Boolean mask.

    Normal direction is not automatically oriented toward a
    particular camera viewpoint.
    """

    points = validate_points(
        points
    )

    if neighbors < 3:
        raise ValueError(
            "At least 3 neighbors are required."
        )

    valid = np.isfinite(
        points
    ).all(axis=1)

    clean = points[
        valid
    ]

    if len(clean) < 4:
        raise ValueError(
            "At least 4 valid points are required."
        )

    k = min(
        neighbors + 1,
        len(clean),
    )

    tree = cKDTree(
        clean
    )

    _, indices = tree.query(
        clean,
        k=k,
    )

    normals = np.zeros(
        (len(clean), 3),
        dtype=np.float64,
    )

    curvature = np.zeros(
        len(clean),
        dtype=np.float64,
    )

    for i in range(len(clean)):
        neighborhood = clean[
            indices[i, 1:]
        ]

        if len(neighborhood) < 3:
            normals[i] = np.nan
            curvature[i] = np.nan
            continue

        center = np.mean(
            neighborhood,
            axis=0,
        )

        centered = (
            neighborhood - center
        )

        covariance = (
            centered.T @ centered
        ) / len(centered)

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        eigenvalues = np.maximum(
            eigenvalues,
            0.0,
        )

        normal = eigenvectors[:, 0]

        norm = np.linalg.norm(
            normal
        )

        if norm < 1e-12:
            normals[i] = np.nan
            curvature[i] = np.nan
            continue

        normals[i] = (
            normal / norm
        )

        total = np.sum(
            eigenvalues
        )

        curvature[i] = (
            eigenvalues[0] / total
            if total > 1e-12
            else 0.0
        )

    full_normals = np.full(
        (len(points), 3),
        np.nan,
        dtype=np.float64,
    )

    full_curvature = np.full(
        len(points),
        np.nan,
        dtype=np.float64,
    )

    full_normals[
        np.flatnonzero(valid)
    ] = normals

    full_curvature[
        np.flatnonzero(valid)
    ] = curvature

    return (
        full_normals,
        full_curvature,
        valid,
    )


# ======================================================================
# RANSAC PLANE ESTIMATION
# ======================================================================

def estimate_plane_from_three_points(
    points,
):
    """
    Estimate a plane from three non-collinear points.

    Returns normalized [a, b, c, d] coefficients.
    """

    points = validate_points(
        points
    )

    if len(points) != 3:
        raise ValueError(
            "Exactly three points are required."
        )

    p1, p2, p3 = points

    normal = np.cross(
        p2 - p1,
        p3 - p1,
    )

    norm = np.linalg.norm(
        normal
    )

    if norm < 1e-12:
        return None

    normal = normal / norm

    d = -np.dot(
        normal,
        p1,
    )

    return np.array(
        [
            normal[0],
            normal[1],
            normal[2],
            d,
        ],
        dtype=np.float64,
    )


def ransac_plane(
    points,
    threshold=0.01,
    iterations=500,
    seed=42,
):
    """
    Estimate a dominant plane using RANSAC.

    Returns:
        plane coefficients [a, b, c, d]
        inlier mask
        inlier RMSE
    """

    points = validate_points(
        points
    )

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            "Threshold must be positive."
        )

    if iterations < 1:
        raise ValueError(
            "Iterations must be positive."
        )

    valid = np.isfinite(
        points
    ).all(axis=1)

    clean = points[
        valid
    ]

    if len(clean) < 3:
        raise ValueError(
            "At least three valid points are required."
        )

    rng = np.random.default_rng(
        seed
    )

    best_plane = None
    best_inliers = None
    best_count = 0

    for _ in range(iterations):
        sample_indices = rng.choice(
            len(clean),
            size=3,
            replace=False,
        )

        plane = estimate_plane_from_three_points(
            clean[sample_indices]
        )

        if plane is None:
            continue

        distances = np.abs(
            clean @ plane[:3]
            + plane[3]
        )

        inliers = (
            distances <= threshold
        )

        count = int(
            np.count_nonzero(inliers)
        )

        if count > best_count:
            best_count = count
            best_plane = plane
            best_inliers = inliers

    if best_plane is None:
        raise RuntimeError(
            "RANSAC could not find a valid plane."
        )

    # Refine using PCA on the largest consensus set.
    inlier_points = clean[
        best_inliers
    ]

    center = np.mean(
        inlier_points,
        axis=0,
    )

    centered = (
        inlier_points - center
    )

    covariance = (
        centered.T @ centered
    ) / len(centered)

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    normal = eigenvectors[:, 0]

    normal /= np.linalg.norm(
        normal
    )

    d = -np.dot(
        normal,
        center,
    )

    refined_plane = np.array(
        [
            normal[0],
            normal[1],
            normal[2],
            d,
        ]
    )

    distances = np.abs(
        clean @ refined_plane[:3]
        + refined_plane[3]
    )

    refined_inliers = (
        distances <= threshold
    )

    rmse = float(
        np.sqrt(
            np.mean(
                distances[refined_inliers] ** 2
            )
        )
    )

    full_mask = np.zeros(
        len(points),
        dtype=bool,
    )

    full_mask[
        np.flatnonzero(valid)
    ] = refined_inliers

    return (
        refined_plane,
        full_mask,
        rmse,
    )


# ======================================================================
# ASCII PLY READER
# ======================================================================

def load_ply(path):
    """
    Load an ASCII PLY point cloud.

    Supports vertex properties:
        x, y, z

    Optional color properties:
        red, green, blue

    This reader intentionally supports ASCII PLY only.
    Binary PLY files require a binary parser or a dedicated library.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        first_line = file.readline().strip()

        if first_line != "ply":
            raise ValueError(
                "Invalid PLY file."
            )

        format_line = None
        vertex_count = None
        vertex_properties = []
        in_vertex_element = False

        while True:
            line = file.readline()

            if not line:
                raise ValueError(
                    "PLY header is incomplete."
                )

            line = line.strip()

            if line.startswith("format "):
                format_line = line

            elif line.startswith("element "):
                parts = line.split()

                in_vertex_element = (
                    parts[1] == "vertex"
                )

                if in_vertex_element:
                    vertex_count = int(
                        parts[2]
                    )

            elif line.startswith("property ") and in_vertex_element:
                parts = line.split()

                if len(parts) != 3:
                    raise ValueError(
                        "Unsupported PLY vertex property."
                    )

                vertex_properties.append(
                    parts[2]
                )

            elif line == "end_header":
                break

        if format_line != "format ascii 1.0":
            raise ValueError(
                "Only ASCII PLY format 1.0 is supported."
            )

        if vertex_count is None:
            raise ValueError(
                "PLY file does not contain a vertex element."
            )

        required = [
            "x",
            "y",
            "z",
        ]

        for name in required:
            if name not in vertex_properties:
                raise ValueError(
                    f"Missing PLY property: {name}"
                )

        rows = []

        for _ in range(vertex_count):
            line = file.readline()

            if not line:
                raise ValueError(
                    "PLY vertex data is incomplete."
                )

            values = line.split()

            if len(values) < len(vertex_properties):
                raise ValueError(
                    "PLY vertex row is incomplete."
                )

            rows.append(
                [
                    float(value)
                    for value in values[:len(vertex_properties)]
                ]
            )

    data = np.asarray(
        rows,
        dtype=np.float64,
    )

    indices = {
        name: i
        for i, name in enumerate(vertex_properties)
    }

    points = data[
        :,
        [
            indices["x"],
            indices["y"],
            indices["z"],
        ],
    ]

    color_names = [
        "red",
        "green",
        "blue",
    ]

    if all(
        name in indices
        for name in color_names
    ):
        colors = data[
            :,
            [
                indices["red"],
                indices["green"],
                indices["blue"],
            ],
        ]

        colors = np.clip(
            np.rint(colors),
            0,
            255,
        ).astype(np.uint8)

    else:
        colors = np.full(
            (len(points), 3),
            180,
            dtype=np.uint8,
        )

    return remove_invalid_points(
        points,
        colors,
    )


# ======================================================================
# ASCII PLY WRITER
# ======================================================================

def write_ply(
    path,
    points,
    colors=None,
):
    points = validate_points(
        points
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
        os.path.dirname(
            os.path.abspath(path)
        ),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write("ply\n")
        file.write("format ascii 1.0\n")

        file.write(
            f"element vertex {len(points)}\n"
        )

        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")

        if colors is not None:
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")

        file.write("end_header\n")

        if colors is None:
            for point in points:
                file.write(
                    f"{point[0]:.7f} "
                    f"{point[1]:.7f} "
                    f"{point[2]:.7f}\n"
                )

        else:
            for point, color in zip(
                points,
                colors,
            ):
                file.write(
                    f"{point[0]:.7f} "
                    f"{point[1]:.7f} "
                    f"{point[2]:.7f} "
                    f"{int(color[0])} "
                    f"{int(color[1])} "
                    f"{int(color[2])}\n"
                )


# ======================================================================
# VISUALIZATION
# ======================================================================

def draw_box(
    ax,
    center,
    axes,
    dimensions,
):
    """
    Draw an oriented 3-D box using its local axes and dimensions.
    """

    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )

    local_corners = (
        signs * dimensions / 2.0
    )

    corners = (
        center
        + local_corners @ axes.T
    )

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for start, end in edges:
        ax.plot(
            [
                corners[start, 0],
                corners[end, 0],
            ],
            [
                corners[start, 1],
                corners[end, 1],
            ],
            [
                corners[start, 2],
                corners[end, 2],
            ],
        )


def visualize_processing(
    original_points,
    original_colors,
    filtered_points,
    filtered_colors,
    normals,
    obb,
    output_path,
    max_points=20000,
):
    rng = np.random.default_rng(
        42
    )

    fig = plt.figure(
        figsize=(17, 6),
    )

    # --------------------------------------------------------------
    # Original point cloud
    # --------------------------------------------------------------

    ax1 = fig.add_subplot(
        1,
        3,
        1,
        projection="3d",
    )

    if len(original_points):
        count = min(
            len(original_points),
            max_points,
        )

        indices = rng.choice(
            len(original_points),
            size=count,
            replace=False,
        )

        ax1.scatter(
            original_points[indices, 0],
            original_points[indices, 1],
            original_points[indices, 2],
            c=original_colors[indices] / 255.0,
            s=1,
            depthshade=False,
        )

    ax1.set_title(
        "Original point cloud"
    )

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")

    # --------------------------------------------------------------
    # Filtered point cloud
    # --------------------------------------------------------------

    ax2 = fig.add_subplot(
        1,
        3,
        2,
        projection="3d",
    )

    if len(filtered_points):
        count = min(
            len(filtered_points),
            max_points,
        )

        indices = rng.choice(
            len(filtered_points),
            size=count,
            replace=False,
        )

        ax2.scatter(
            filtered_points[indices, 0],
            filtered_points[indices, 1],
            filtered_points[indices, 2],
            c=filtered_colors[indices] / 255.0,
            s=1,
            depthshade=False,
        )

    ax2.set_title(
        "Filtered point cloud"
    )

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")

    # --------------------------------------------------------------
    # Geometry and normals
    # --------------------------------------------------------------

    ax3 = fig.add_subplot(
        1,
        3,
        3,
        projection="3d",
    )

    if len(filtered_points):
        count = min(
            len(filtered_points),
            max_points,
        )

        indices = rng.choice(
            len(filtered_points),
            size=count,
            replace=False,
        )

        ax3.scatter(
            filtered_points[indices, 0],
            filtered_points[indices, 1],
            filtered_points[indices, 2],
            c=filtered_colors[indices] / 255.0,
            s=1,
            depthshade=False,
        )

        valid_normals = (
            np.isfinite(normals).all(axis=1)
        )

        normal_indices = np.flatnonzero(
            valid_normals
        )

        if len(normal_indices):
            normal_count = min(
                len(normal_indices),
                300,
            )

            selected = rng.choice(
                normal_indices,
                size=normal_count,
                replace=False,
            )

            ax3.quiver(
                filtered_points[selected, 0],
                filtered_points[selected, 1],
                filtered_points[selected, 2],
                normals[selected, 0],
                normals[selected, 1],
                normals[selected, 2],
                length=0.025,
                normalize=True,
            )

        draw_box(
            ax3,
            obb["center"],
            obb["axes"],
            obb["dimensions"],
        )

    ax3.set_title(
        "Surface normals + PCA bounding box"
    )

    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")

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
# PROCESSING PIPELINE
# ======================================================================

def process_point_cloud(args):
    start = time.perf_counter()

    points, colors = load_ply(
        args.ply
    )

    if len(points) <= args.neighbors:
        raise ValueError(
            "Input point cloud has too few points for the selected "
            "statistical neighbor count."
        )

    print("\n" + "=" * 68)
    print("POINT CLOUD PROCESSING")
    print("=" * 68)

    print(
        f"Input points: {len(points)}"
    )

    # --------------------------------------------------------------
    # 1. Original geometry
    # --------------------------------------------------------------

    original_centroid = calculate_centroid(
        points
    )

    original_aabb = calculate_aabb(
        points
    )

    # --------------------------------------------------------------
    # 2. Statistical outlier removal
    # --------------------------------------------------------------

    statistical_points, statistical_mask, distances, threshold = (
        statistical_outlier_removal(
            points,
            neighbors=args.neighbors,
            std_ratio=args.std_ratio,
        )
    )

    statistical_colors = colors[
        statistical_mask
    ]

    print(
        f"After statistical filtering: "
        f"{len(statistical_points)}"
    )

    # --------------------------------------------------------------
    # 3. Radius outlier removal
    # --------------------------------------------------------------

    if args.radius > 0:
        filtered_points, radius_mask, counts = (
            radius_outlier_removal(
                statistical_points,
                radius=args.radius,
                min_neighbors=args.min_radius_neighbors,
            )
        )

        filtered_colors = statistical_colors[
            radius_mask
        ]

    else:
        filtered_points = statistical_points
        filtered_colors = statistical_colors
        radius_mask = np.ones(
            len(filtered_points),
            dtype=bool,
        )

    print(
        f"After radius filtering: {len(filtered_points)}"
    )

    if len(filtered_points) <= args.normal_neighbors:
        raise ValueError(
            "Too few points remain for normal estimation. "
            "Reduce filtering or the normal-neighbor count."
        )

    # --------------------------------------------------------------
    # 4. Voxel downsampling
    # --------------------------------------------------------------

    if args.voxel_size > 0:
        downsampled_points, downsampled_colors = voxel_downsample(
            filtered_points,
            filtered_colors,
            voxel_size=args.voxel_size,
        )

    else:
        downsampled_points = filtered_points
        downsampled_colors = filtered_colors

    print(
        f"After voxel downsampling: {len(downsampled_points)}"
    )

    if len(downsampled_points) <= args.normal_neighbors:
        raise ValueError(
            "Too few points remain for normal estimation after "
            "voxel downsampling. Reduce voxel size or normal neighbors."
        )

    # --------------------------------------------------------------
    # 5. Surface normals
    # --------------------------------------------------------------

    normals, curvature, normal_valid = estimate_normals(
        downsampled_points,
        neighbors=args.normal_neighbors,
    )

    valid_curvature = curvature[
        np.isfinite(curvature)
    ]

    # --------------------------------------------------------------
    # 6. Geometry
    # --------------------------------------------------------------

    centroid = calculate_centroid(
        downsampled_points
    )

    aabb = calculate_aabb(
        downsampled_points
    )

    obb = calculate_obb(
        downsampled_points
    )

    # --------------------------------------------------------------
    # 7. Optional plane detection
    # --------------------------------------------------------------

    plane_summary = None

    if args.detect_plane:
        if len(downsampled_points) < 3:
            raise ValueError(
                "At least three points are required for plane detection."
            )

        plane, inliers, rmse = ransac_plane(
            downsampled_points,
            threshold=args.plane_threshold,
            iterations=args.plane_iterations,
        )

        plane_summary = {
            "coefficients_abcd": plane.tolist(),
            "inlier_count": int(
                np.count_nonzero(inliers)
            ),
            "inlier_ratio": float(
                np.mean(inliers)
            ),
            "rmse_meters": rmse,
        }

        print(
            f"RANSAC plane inliers: {np.count_nonzero(inliers)}"
        )

        print(
            f"RANSAC plane: {plane}"
        )

    # --------------------------------------------------------------
    # 8. Save results
    # --------------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    output_ply = os.path.join(
        args.output_dir,
        "filtered_point_cloud.ply",
    )

    write_ply(
        output_ply,
        downsampled_points,
        downsampled_colors,
    )

    visualization_path = os.path.join(
        args.output_dir,
        "point_cloud_processing.png",
    )

    visualize_processing(
        points,
        colors,
        downsampled_points,
        downsampled_colors,
        normals,
        obb,
        visualization_path,
        max_points=args.max_visualization_points,
    )

    summary = {
        "lesson": "Lesson 59 - Point Clouds",
        "input_file": args.ply,
        "original_point_count": int(len(points)),
        "statistical_filter": {
            "neighbors": args.neighbors,
            "std_ratio": args.std_ratio,
            "distance_threshold": float(threshold),
            "remaining_points": int(
                len(statistical_points)
            ),
        },
        "radius_filter": {
            "radius_meters": args.radius,
            "minimum_neighbors": args.min_radius_neighbors,
            "remaining_points": int(
                len(filtered_points)
            ),
        },
        "voxel_filter": {
            "voxel_size_meters": args.voxel_size,
            "remaining_points": int(
                len(downsampled_points)
            ),
        },
        "geometry": {
            "centroid_xyz": centroid.tolist(),
            "aabb": {
                "minimum": aabb["minimum"].tolist(),
                "maximum": aabb["maximum"].tolist(),
                "dimensions": aabb["dimensions"].tolist(),
                "center": aabb["center"].tolist(),
            },
            "pca_obb": {
                "center": obb["center"].tolist(),
                "dimensions": obb["dimensions"].tolist(),
                "axes": obb["axes"].tolist(),
                "eigenvalues": obb["eigenvalues"].tolist(),
            },
        },
        "surface_normals": {
            "valid_normal_count": int(
                np.count_nonzero(
                    np.isfinite(normals).all(axis=1)
                )
            ),
            "mean_curvature": (
                float(np.mean(valid_curvature))
                if len(valid_curvature)
                else None
            ),
            "max_curvature": (
                float(np.max(valid_curvature))
                if len(valid_curvature)
                else None
            ),
        },
        "plane_detection": plane_summary,
        "output_ply": output_ply,
        "visualization": visualization_path,
        "processing_seconds": float(
            time.perf_counter() - start
        ),
    }

    summary_path = os.path.join(
        args.output_dir,
        "processing_summary.json",
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

    print("\nGEOMETRY")
    print("-" * 68)

    print(
        "Centroid:",
        centroid,
    )

    print(
        "AABB dimensions:",
        aabb["dimensions"],
    )

    print(
        "PCA OBB dimensions:",
        obb["dimensions"],
    )

    print(
        "Mean curvature:",
        np.mean(valid_curvature)
        if len(valid_curvature)
        else None,
    )

    print("\nOUTPUT")
    print("-" * 68)

    print(
        output_ply
    )

    print(
        visualization_path
    )

    print(
        summary_path
    )

    print("=" * 68)


# ======================================================================
# TEST UTILITIES
# ======================================================================

def make_plane_grid(
    size=10,
    spacing=0.1,
):
    x, y = np.meshgrid(
        np.arange(size) * spacing,
        np.arange(size) * spacing,
    )

    z = np.zeros_like(
        x
    )

    return np.column_stack(
        [
            x.ravel(),
            y.ravel(),
            z.ravel(),
        ]
    )


# ======================================================================
# TESTS
# ======================================================================

def run_tests():
    print("\nRunning Lesson 59 tests...")
    print("=" * 68)

    rng = np.random.default_rng(
        123
    )

    # --------------------------------------------------------------
    # Test 1: point validation
    # --------------------------------------------------------------

    points = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=float,
    )

    assert validate_points(
        points
    ).shape == (2, 3)

    print("Test 01 passed: point-cloud validation.")

    # --------------------------------------------------------------
    # Test 2: invalid point removal
    # --------------------------------------------------------------

    points = np.array(
        [
            [1, 2, 3],
            [np.nan, 0, 0],
            [4, 5, 6],
        ]
    )

    filtered = remove_invalid_points(
        points
    )

    assert len(filtered) == 2

    print("Test 02 passed: invalid point removal.")

    # --------------------------------------------------------------
    # Test 3: centroid
    # --------------------------------------------------------------

    points = np.array(
        [
            [0, 0, 0],
            [2, 2, 2],
        ]
    )

    assert np.allclose(
        calculate_centroid(points),
        [1, 1, 1],
    )

    print("Test 03 passed: centroid calculation.")

    # --------------------------------------------------------------
    # Test 4: AABB
    # --------------------------------------------------------------

    points = np.array(
        [
            [0, 1, 2],
            [2, 4, 6],
            [1, 2, 3],
        ]
    )

    box = calculate_aabb(
        points
    )

    assert np.allclose(
        box["minimum"],
        [0, 1, 2],
    )

    assert np.allclose(
        box["maximum"],
        [2, 4, 6],
    )

    assert np.allclose(
        box["dimensions"],
        [2, 3, 4],
    )

    print("Test 04 passed: axis-aligned bounding box.")

    # --------------------------------------------------------------
    # Test 5: OBB dimensions for an axis-aligned cuboid
    # --------------------------------------------------------------

    x, y, z = np.meshgrid(
        [0, 1],
        [0, 2],
        [0, 3],
    )

    cuboid = np.column_stack(
        [
            x.ravel(),
            y.ravel(),
            z.ravel(),
        ]
    )

    obb = calculate_obb(
        cuboid
    )

    assert np.allclose(
        np.sort(obb["dimensions"]),
        [1, 2, 3],
    )

    print("Test 05 passed: PCA oriented bounding box.")

    # --------------------------------------------------------------
    # Test 6: transformation validation
    # --------------------------------------------------------------

    identity = make_transform(
        np.eye(3),
        np.zeros(3),
    )

    assert np.allclose(
        identity,
        np.eye(4),
    )

    print("Test 06 passed: homogeneous transformation.")

    # --------------------------------------------------------------
    # Test 7: translation
    # --------------------------------------------------------------

    transform = make_transform(
        np.eye(3),
        [1, 2, 3],
    )

    result = transform_points(
        np.array(
            [0, 0, 0],
        ),
        transform,
    )

    assert np.allclose(
        result,
        [1, 2, 3],
    )

    print("Test 07 passed: translation.")

    # --------------------------------------------------------------
    # Test 8: rotation
    # --------------------------------------------------------------

    rotation = np.array(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )

    transform = make_transform(
        rotation,
        [0, 0, 0],
    )

    result = transform_points(
        np.array([1, 0, 0]),
        transform,
    )

    assert np.allclose(
        result,
        [0, 1, 0],
    )

    print("Test 08 passed: rotation.")

    # --------------------------------------------------------------
    # Test 9: inverse transformation
    # --------------------------------------------------------------

    transform = make_transform(
        rotation,
        [0.5, 0.2, 0.1],
    )

    inverse = invert_transform(
        transform
    )

    point = np.array(
        [0.3, 0.4, 0.5],
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

    print("Test 09 passed: inverse transformation.")

    # --------------------------------------------------------------
    # Test 10: brute-force nearest neighbor
    # --------------------------------------------------------------

    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 2, 0],
        ],
        dtype=float,
    )

    queries = np.array(
        [
            [0.1, 0, 0],
        ]
    )

    distances, indices = brute_force_nearest_neighbors(
        points,
        queries,
        k=1,
    )

    assert indices[0] == 0
    assert np.isclose(
        distances[0],
        0.1,
    )

    print("Test 10 passed: brute-force nearest neighbor.")

    # --------------------------------------------------------------
    # Test 11: KD-tree nearest neighbor
    # --------------------------------------------------------------

    distances, indices = kdtree_nearest_neighbors(
        points,
        queries,
        k=1,
    )

    assert indices[0] == 0
    assert np.isclose(
        distances[0],
        0.1,
    )

    print("Test 11 passed: KD-tree nearest neighbor.")

    # --------------------------------------------------------------
    # Test 12: KD-tree vs brute force
    # --------------------------------------------------------------

    points = rng.normal(
        size=(100, 3),
    )

    queries = rng.normal(
        size=(15, 3),
    )

    d1, i1 = brute_force_nearest_neighbors(
        points,
        queries,
        k=3,
    )

    d2, i2 = kdtree_nearest_neighbors(
        points,
        queries,
        k=3,
    )

    assert np.allclose(
        d1,
        d2,
    )

    assert np.allclose(
        np.sort(d1, axis=1),
        np.sort(d2, axis=1),
    )

    print("Test 12 passed: KD-tree/brute-force comparison.")

    # --------------------------------------------------------------
    # Test 13: statistical outlier removal
    # --------------------------------------------------------------

    cluster = rng.normal(
        loc=0,
        scale=0.01,
        size=(150, 3),
    )

    outliers = np.array(
        [
            [3, 3, 3],
            [-3, -3, -3],
        ],
        dtype=float,
    )

    points = np.vstack(
        [
            cluster,
            outliers,
        ]
    )

    filtered, mask, distances, threshold = (
        statistical_outlier_removal(
            points,
            neighbors=10,
            std_ratio=1.0,
        )
    )

    assert len(filtered) < len(points)
    assert not mask[-1]
    assert not mask[-2]

    print("Test 13 passed: statistical outlier removal.")

    # --------------------------------------------------------------
    # Test 14: radius outlier removal
    # --------------------------------------------------------------

    cluster = rng.normal(
        loc=0,
        scale=0.01,
        size=(100, 3),
    )

    points = np.vstack(
        [
            cluster,
            [2, 2, 2],
        ]
    )

    filtered, mask, counts = radius_outlier_removal(
        points,
        radius=0.08,
        min_neighbors=3,
    )

    assert not mask[-1]
    assert len(filtered) == 100

    print("Test 14 passed: radius outlier removal.")

    # --------------------------------------------------------------
    # Test 15: voxel downsampling
    # --------------------------------------------------------------

    points = np.array(
        [
            [0.001, 0.001, 0.001],
            [0.002, 0.002, 0.002],
            [0.101, 0.101, 0.101],
        ]
    )

    filtered = voxel_downsample(
        points,
        voxel_size=0.05,
    )

    assert len(filtered) == 2

    print("Test 15 passed: voxel-grid downsampling.")

    # --------------------------------------------------------------
    # Test 16: voxel color averaging
    # --------------------------------------------------------------

    points = np.array(
        [
            [0.001, 0.001, 0.001],
            [0.002, 0.002, 0.002],
        ]
    )

    colors = np.array(
        [
            [100, 100, 100],
            [200, 200, 200],
        ],
        dtype=np.uint8,
    )

    filtered, filtered_colors = voxel_downsample(
        points,
        colors,
        voxel_size=0.05,
    )

    assert len(filtered) == 1

    assert np.allclose(
        filtered_colors[0],
        [150, 150, 150],
    )

    print("Test 16 passed: voxel color averaging.")

    # --------------------------------------------------------------
    # Test 17: planar surface normal
    # --------------------------------------------------------------

    plane = make_plane_grid(
        size=10,
        spacing=0.1,
    )

    normals, curvature, valid = estimate_normals(
        plane,
        neighbors=10,
    )

    assert np.all(
        valid
    )

    assert np.nanmean(
        np.abs(normals[:, 2])
    ) > 0.99

    print("Test 17 passed: planar surface normals.")

    # --------------------------------------------------------------
    # Test 18: normal vector unit length
    # --------------------------------------------------------------

    lengths = np.linalg.norm(
        normals,
        axis=1,
    )

    assert np.allclose(
        lengths,
        1.0,
        atol=1e-5,
    )

    print("Test 18 passed: normal unit length.")

    # --------------------------------------------------------------
    # Test 19: planar curvature
    # --------------------------------------------------------------

    assert np.nanmean(
        curvature
    ) < 1e-5

    print("Test 19 passed: planar curvature.")

    # --------------------------------------------------------------
    # Test 20: plane from three points
    # --------------------------------------------------------------

    sample = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=float,
    )

    plane_coefficients = estimate_plane_from_three_points(
        sample
    )

    assert np.allclose(
        abs(plane_coefficients[2]),
        1.0,
    )

    assert np.isclose(
        plane_coefficients[3],
        0.0,
    )

    print("Test 20 passed: plane from three points.")

    # --------------------------------------------------------------
    # Test 21: collinear plane samples
    # --------------------------------------------------------------

    sample = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
        ],
        dtype=float,
    )

    result = estimate_plane_from_three_points(
        sample
    )

    assert result is None

    print("Test 21 passed: degenerate plane rejection.")

    # --------------------------------------------------------------
    # Test 22: RANSAC plane
    # --------------------------------------------------------------

    x = rng.uniform(
        -1,
        1,
        300,
    )

    y = rng.uniform(
        -1,
        1,
        300,
    )

    z = rng.normal(
        0,
        0.001,
        300,
    )

    plane_points = np.column_stack(
        [
            x,
            y,
            z,
        ]
    )

    outliers = rng.uniform(
        -1,
        1,
        size=(30, 3),
    )

    points = np.vstack(
        [
            plane_points,
            outliers,
        ]
    )

    coefficients, inliers, rmse = ransac_plane(
        points,
        threshold=0.005,
        iterations=300,
        seed=123,
    )

    assert np.count_nonzero(inliers) >= 290

    assert rmse < 0.002

    assert abs(
        coefficients[2]
    ) > 0.99

    print("Test 22 passed: RANSAC plane estimation.")

    # --------------------------------------------------------------
    # Test 23: PLY write and read
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
            dtype=float,
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

        loaded_points, loaded_colors = load_ply(
            path
        )

        assert np.allclose(
            loaded_points,
            points,
        )

        assert np.array_equal(
            loaded_colors,
            colors,
        )

    print("Test 23 passed: PLY write/read.")

    # --------------------------------------------------------------
    # Test 24: PLY without colors
    # --------------------------------------------------------------

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(
            directory,
            "test_no_color.ply",
        )

        points = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
            ],
            dtype=float,
        )

        write_ply(
            path,
            points,
        )

        loaded_points, loaded_colors = load_ply(
            path
        )

        assert np.allclose(
            loaded_points,
            points,
        )

        assert loaded_colors.shape == (2, 3)

    print("Test 24 passed: PLY without colors.")

    # --------------------------------------------------------------
    # Test 25: OBB dimensions after rotation
    # --------------------------------------------------------------

    angle = np.deg2rad(
        35
    )

    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )

    x, y, z = np.meshgrid(
        np.linspace(-1, 1, 8),
        np.linspace(-0.5, 0.5, 8),
        np.linspace(-0.25, 0.25, 8),
    )

    box_points = np.column_stack(
        [
            x.ravel(),
            y.ravel(),
            z.ravel(),
        ]
    )

    rotated = (
        box_points @ rotation.T
    )

    obb = calculate_obb(
        rotated
    )

    assert np.allclose(
        np.sort(obb["dimensions"]),
        [0.5, 1.0, 2.0],
        atol=1e-5,
    )

    print("Test 25 passed: rotated OBB geometry.")

    # --------------------------------------------------------------
    # Test 26: nearest-neighbor k validation
    # --------------------------------------------------------------

    try:
        kdtree_nearest_neighbors(
            np.zeros((3, 3)),
            np.zeros((1, 3)),
            k=4,
        )

        raise AssertionError(
            "Invalid k accepted."
        )

    except ValueError:
        pass

    print("Test 26 passed: nearest-neighbor validation.")

    print("=" * 68)
    print("All 26 tests passed.")


# ======================================================================
# COMMAND LINE
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Lesson 59 - Point Cloud Processing"
    )

    parser.add_argument(
        "--ply",
        type=str,
        default=None,
        help="Input ASCII PLY file from Lesson 58.",
    )

    parser.add_argument(
        "--neighbors",
        type=int,
        default=20,
        help="Neighbors for statistical outlier removal.",
    )

    parser.add_argument(
        "--std-ratio",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=0.05,
        help="Radius outlier distance in meters; 0 disables it.",
    )

    parser.add_argument(
        "--min-radius-neighbors",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Voxel size in meters; 0 disables downsampling.",
    )

    parser.add_argument(
        "--normal-neighbors",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--detect-plane",
        action="store_true",
    )

    parser.add_argument(
        "--plane-threshold",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--plane-iterations",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--max-visualization-points",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/lesson59",
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

    if args.ply is None:
        print(
            "\nNo input point cloud specified."
        )

        print(
            "\nFirst generate a point cloud using Lesson 58:"
        )

        print(
            "python lesson58_rgbd_computer_vision.py "
            "--rgb rgb.png --depth depth.png "
            "--fx 600 --fy 600 --cx 320 --cy 240"
        )

        print(
            "\nThen process its PLY output:"
        )

        print(
            "python lesson59_point_cloud_processing.py "
            "--ply runs/lesson58/point_cloud.ply"
        )

        print(
            "\nRun numerical tests:"
        )

        print(
            "python lesson59_point_cloud_processing.py --check-only"
        )

        return

    if not os.path.isfile(
        args.ply
    ):
        raise FileNotFoundError(
            f"Point-cloud file not found: {args.ply}"
        )

    if args.neighbors < 1:
        raise ValueError(
            "--neighbors must be positive."
        )

    if args.normal_neighbors < 3:
        raise ValueError(
            "--normal-neighbors must be at least 3."
        )

    if args.std_ratio < 0:
        raise ValueError(
            "--std-ratio cannot be negative."
        )

    if args.radius < 0:
        raise ValueError(
            "--radius cannot be negative."
        )

    if args.min_radius_neighbors < 1:
        raise ValueError(
            "--min-radius-neighbors must be positive."
        )

    if args.voxel_size < 0:
        raise ValueError(
            "--voxel-size cannot be negative."
        )

    if args.max_visualization_points < 1:
        raise ValueError(
            "--max-visualization-points must be positive."
        )

    if args.plane_threshold <= 0:
        raise ValueError(
            "--plane-threshold must be positive."
        )

    if args.plane_iterations < 1:
        raise ValueError(
            "--plane-iterations must be positive."
        )

    run_tests()

    process_point_cloud(
        args
    )


if __name__ == "__main__":
    main()