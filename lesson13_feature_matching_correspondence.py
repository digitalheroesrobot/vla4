"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 13 — FEATURE MATCHING AND DESCRIPTOR-BASED CORRESPONDENCE
======================================================================

COURSE RULE
-----------
All main algorithms are implemented from scratch.

NO:
    - cv2.ORB_create()
    - cv2.SIFT_create()
    - cv2.BFMatcher()
    - cv2.findHomography()
    - pretrained models or weights

Libraries:
    - NumPy
    - Matplotlib
    - Python standard library

======================================================================
WHAT YOU WILL LEARN
======================================================================

1. What is a feature correspondence?
2. Why do we need local descriptors?
3. Harris corner detection.
4. Patch-based feature descriptors.
5. Descriptor normalization.
6. Euclidean descriptor distance.
7. Brute-force nearest-neighbor matching.
8. Lowe's ratio test.
9. Mutual nearest-neighbor matching.
10. RANSAC homography estimation.
11. Geometric verification of feature matches.
12. Applications in visual odometry and robotics.

======================================================================
1. FEATURE CORRESPONDENCE
======================================================================

Given two images:

    Image A
    Image B

We want to identify points that represent the same physical
location or visual structure in both images.

A correspondence is:

    (x_A, y_A) <----> (x_B, y_B)

For example, a corner on a cup in Image A should correspond
to the same physical corner on the cup in Image B.

A feature pipeline:

    Image
      |
      v
    Keypoint Detection
      |
      v
    Descriptor Extraction
      |
      v
    Descriptor Matching
      |
      v
    Geometric Verification
      |
      v
    Reliable Correspondences

======================================================================
2. KEYPOINTS
======================================================================

A keypoint is an image location that is distinctive enough
to be detected and matched.

Examples:
    - Corners
    - Junctions
    - Textured regions
    - Distinctive object boundaries

We use the Harris corner response.

    R = det(M) - k * trace(M)^2

where M is the local structure tensor.

    M = [ Sxx  Sxy ]
        [ Sxy  Syy ]

    det(M) = Sxx*Syy - Sxy^2

    trace(M) = Sxx + Syy

Large positive Harris response indicates a likely corner.

======================================================================
3. PATCH DESCRIPTORS
======================================================================

For every keypoint, extract a local image patch.

Example:

    PATCH_SIZE = 15 x 15

Flatten the patch into a vector:

    descriptor = [p1, p2, ..., p225]

Normalize it:

    d = (d - mean(d)) / (std(d) + epsilon)

This reduces sensitivity to simple brightness changes.

IMPORTANT:
    A raw patch descriptor is not rotation invariant.
    It is not scale invariant.
    It can fail under viewpoint changes or occlusion.

This is a simple educational descriptor, not SIFT or ORB.

======================================================================
4. DESCRIPTOR DISTANCE
======================================================================

For descriptors A and B:

    d(A,B) = sqrt(sum((A_i - B_i)^2))

Smaller distance means greater similarity.

For each descriptor in Image A, find the nearest descriptors
in Image B.

This is brute-force matching.

Computational complexity:

    O(N_A * N_B * D)

where:
    N_A = number of descriptors in Image A
    N_B = number of descriptors in Image B
    D   = descriptor dimension

======================================================================
5. LOWE'S RATIO TEST
======================================================================

The nearest match is not always reliable.

Let:

    d1 = distance to nearest neighbor
    d2 = distance to second-nearest neighbor

Ratio:

    r = d1 / (d2 + epsilon)

Accept if:

    r < ratio_threshold

A lower ratio indicates that the best match is more distinctive
relative to the second-best match.

======================================================================
6. MUTUAL MATCHING
======================================================================

A match is mutual if:

    A's nearest neighbor is B
    AND
    B's nearest neighbor is A

This removes some ambiguous matches.

======================================================================
7. HOMOGRAPHY
======================================================================

A homography describes a projective transformation between
two image planes.

    [x']       [h11 h12 h13] [x]
    [y']  ~    [h21 h22 h23] [y]
    [ 1]       [h31 h32 h33] [1]

The symbol "~" means equality up to scale.

For a point (x,y):

    denominator = h31*x + h32*y + h33

    x' = (h11*x + h12*y + h13) / denominator

    y' = (h21*x + h22*y + h23) / denominator

A homography has 8 independent degrees of freedom because
its 9 elements are defined only up to scale.

At least four point correspondences are needed to estimate it,
assuming the points are in a non-degenerate configuration.

======================================================================
8. RANSAC
======================================================================

RANSAC means Random Sample Consensus.

It estimates a model despite incorrect matches.

Algorithm:

    1. Randomly select four correspondences.
    2. Estimate a homography.
    3. Transform all source points.
    4. Calculate geometric reprojection errors.
    5. Count inliers.
    6. Repeat.
    7. Keep the model with the strongest consensus.
    8. Refit using all inliers.

Reprojection error:

    error = sqrt(
        (x_predicted - x_observed)^2
        +
        (y_predicted - y_observed)^2
    )

A match is an inlier when:

    error < reprojection_threshold

======================================================================
9. LIMITATIONS
======================================================================

Our patch descriptors have limitations:

    - Sensitive to rotation.
    - Sensitive to scale changes.
    - Sensitive to perspective changes.
    - Sensitive to occlusion.
    - May confuse repeated patterns.

RANSAC can reject geometrically inconsistent matches, but it
cannot repair a descriptor that consistently matches the wrong
physical object.

A single homography is appropriate for planar scenes or pure
camera rotation under suitable conditions. It is not a general
3-D scene motion model.

======================================================================
REAL-WORLD ROBOTICS EXAMPLE
======================================================================

OPENARM V2 — VISUAL LOCALIZATION

Imagine a camera mounted above a waste-sorting table.

At time t:

    Image A

At time t+1:

    Image B

The robot or camera may have moved.

Feature correspondence can help estimate image motion.

A simplified pipeline:

    Camera A
       |
       v
    Harris Keypoints
       |
       v
    Patch Descriptors
       |
       v
    Feature Matching
       |
       v
    Ratio Test
       |
       v
    RANSAC
       |
       v
    Geometric Transformation

This is a foundation for:

    - Visual odometry
    - Image registration
    - Camera motion estimation
    - SLAM
    - Object tracking

======================================================================
EXERCISES
======================================================================

1. Change the descriptor patch radius.
2. Change the Harris threshold.
3. Change the ratio-test threshold.
4. Add random incorrect correspondences.
5. Change the camera translation.
6. Add rotation to Image B.
7. Compare raw matching with RANSAC-filtered matching.

======================================================================
NEXT LESSON
======================================================================

LESSON 14 — OPTICAL FLOW FROM SCRATCH

We will estimate apparent pixel motion between consecutive
images using the brightness-constancy assumption.

======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ====================================================================
# CONFIGURATION
# ====================================================================

IMAGE_HEIGHT = 180
IMAGE_WIDTH = 260

PATCH_RADIUS = 6
MAX_KEYPOINTS = 100

HARRIS_K = 0.04
HARRIS_THRESHOLD = 0.015
NMS_RADIUS = 5

RATIO_THRESHOLD = 0.85
REPROJECTION_THRESHOLD = 3.0

RANSAC_ITERATIONS = 500

RANDOM_SEED = 42

SHIFT_X = 12
SHIFT_Y = 8


# ====================================================================
# STEP 1 — SYNTHETIC IMAGE
# ====================================================================

def create_scene():

    image = np.full(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        25.0,
        dtype=np.float32
    )

    # Distinctive rectangles
    image[20:60, 20:65] = 180
    image[25:45, 30:55] = 240

    image[80:125, 35:85] = 150
    image[90:110, 45:70] = 235

    # Circle
    yy, xx = np.mgrid[
        0:IMAGE_HEIGHT,
        0:IMAGE_WIDTH
    ]

    circle = (
        (xx - 145) ** 2
        +
        (yy - 48) ** 2
    ) < 20 ** 2

    image[circle] = 210

    # Asymmetric object
    image[100:145, 150:195] = 190
    image[105:115, 155:180] = 245
    image[125:137, 175:190] = 65

    # Diagonal line
    for i in range(30, 80):

        x = 205 + i // 3
        y = i

        if (
            0 <= x < IMAGE_WIDTH
            and
            0 <= y < IMAGE_HEIGHT
        ):

            image[y, x] = 255

    # Additional small distinctive structures
    image[35:43, 205:220] = 100
    image[43:51, 205:220] = 240

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    image += rng.normal(
        0,
        1.5,
        image.shape
    )

    return np.clip(
        image,
        0,
        255
    ).astype(np.float32)


# ====================================================================
# STEP 2 — TRANSLATE IMAGE WITHOUT WRAPPING
# ====================================================================

def translate_image(image, dx, dy):

    height, width = image.shape

    translated = np.zeros_like(
        image
    )

    source_x_start = max(0, -dx)
    source_x_end = min(width, width - dx)

    source_y_start = max(0, -dy)
    source_y_end = min(height, height - dy)

    destination_x_start = max(0, dx)
    destination_x_end = min(width, width + dx)

    destination_y_start = max(0, dy)
    destination_y_end = min(height, height + dy)

    translated[
        destination_y_start:destination_y_end,
        destination_x_start:destination_x_end
    ] = image[
        source_y_start:source_y_end,
        source_x_start:source_x_end
    ]

    return translated


# ====================================================================
# STEP 3 — CONVOLUTION
# ====================================================================

def convolve2d(image, kernel):

    image = image.astype(
        np.float32
    )

    kernel = kernel.astype(
        np.float32
    )

    kh, kw = kernel.shape

    pad_y = kh // 2
    pad_x = kw // 2

    padded = np.pad(
        image,
        (
            (pad_y, pad_y),
            (pad_x, pad_x)
        ),
        mode="reflect"
    )

    output = np.zeros_like(
        image
    )

    flipped = kernel[::-1, ::-1]

    for y in range(image.shape[0]):

        for x in range(image.shape[1]):

            patch = padded[
                y:y + kh,
                x:x + kw
            ]

            output[y, x] = np.sum(
                patch * flipped
            )

    return output


# ====================================================================
# STEP 4 — FAST BOX FILTER USING INTEGRAL IMAGE
# ====================================================================

def box_filter(image, radius=1):

    """
    Compute a local box sum using an integral image.

    This is still implemented directly with NumPy.

    The integral image allows each rectangular sum to be calculated
    using four array accesses.
    """

    size = 2 * radius + 1

    padded = np.pad(
        image,
        (
            (radius, radius),
            (radius, radius)
        ),
        mode="reflect"
    )

    integral = np.pad(
        padded,
        (
            (1, 0),
            (1, 0)
        ),
        mode="constant"
    )

    integral = np.cumsum(
        np.cumsum(
            integral,
            axis=0
        ),
        axis=1
    )

    height, width = image.shape

    y1 = np.arange(
        0,
        height
    )

    x1 = np.arange(
        0,
        width
    )

    y2 = y1 + size
    x2 = x1 + size

    output = (
        integral[np.ix_(y2, x2)]
        -
        integral[np.ix_(y1, x2)]
        -
        integral[np.ix_(y2, x1)]
        +
        integral[np.ix_(y1, x1)]
    )

    return output.astype(
        np.float32
    )


# ====================================================================
# STEP 5 — HARRIS CORNER DETECTOR
# ====================================================================

def harris_response(image):

    sobel_x = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],
        dtype=np.float32
    )

    sobel_y = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],
        dtype=np.float32
    )

    gx = convolve2d(
        image,
        sobel_x
    )

    gy = convolve2d(
        image,
        sobel_y
    )

    ixx = gx * gx
    iyy = gy * gy
    ixy = gx * gy

    sxx = box_filter(
        ixx,
        radius=2
    )

    syy = box_filter(
        iyy,
        radius=2
    )

    sxy = box_filter(
        ixy,
        radius=2
    )

    determinant = (
        sxx * syy
        -
        sxy * sxy
    )

    trace = (
        sxx + syy
    )

    response = (
        determinant
        -
        HARRIS_K * trace ** 2
    )

    return response


# ====================================================================
# STEP 6 — NON-MAXIMUM SUPPRESSION
# ====================================================================

def non_maximum_suppression(
    response,
    threshold,
    radius=5,
    max_points=100
):

    height, width = response.shape

    candidates = np.argwhere(
        response > threshold
    )

    if len(candidates) == 0:

        return np.empty(
            (0, 2),
            dtype=np.int32
        )

    scores = response[
        candidates[:, 0],
        candidates[:, 1]
    ]

    order = np.argsort(
        scores
    )[::-1]

    selected = []

    suppressed = np.zeros(
        (height, width),
        dtype=bool
    )

    for index in order:

        y, x = candidates[index]

        if suppressed[y, x]:
            continue

        selected.append(
            (x, y)
        )

        y_min = max(
            0,
            y - radius
        )

        y_max = min(
            height,
            y + radius + 1
        )

        x_min = max(
            0,
            x - radius
        )

        x_max = min(
            width,
            x + radius + 1
        )

        suppressed[
            y_min:y_max,
            x_min:x_max
        ] = True

        if len(selected) >= max_points:
            break

    return np.asarray(
        selected,
        dtype=np.int32
    )


# ====================================================================
# STEP 7 — PATCH DESCRIPTORS
# ====================================================================

def extract_descriptors(
    image,
    keypoints,
    patch_radius=6,
    epsilon=1e-8
):

    descriptors = []
    valid_keypoints = []

    height, width = image.shape

    for x, y in keypoints:

        x = int(x)
        y = int(y)

        if (
            x - patch_radius < 0
            or
            x + patch_radius >= width
            or
            y - patch_radius < 0
            or
            y + patch_radius >= height
        ):

            continue

        patch = image[
            y - patch_radius:y + patch_radius + 1,
            x - patch_radius:x + patch_radius + 1
        ]

        descriptor = patch.flatten().astype(
            np.float32
        )

        descriptor = (
            descriptor - np.mean(descriptor)
        )

        standard_deviation = np.std(
            descriptor
        )

        if standard_deviation < epsilon:
            continue

        descriptor /= (
            standard_deviation + epsilon
        )

        descriptors.append(
            descriptor
        )

        valid_keypoints.append(
            (x, y)
        )

    if len(descriptors) == 0:

        return (
            np.empty((0, (2 * patch_radius + 1) ** 2)),
            np.empty((0, 2), dtype=np.int32)
        )

    return (
        np.asarray(
            descriptors,
            dtype=np.float32
        ),

        np.asarray(
            valid_keypoints,
            dtype=np.int32
        )
    )


# ====================================================================
# STEP 8 — BRUTE-FORCE DESCRIPTOR MATCHING
# ====================================================================

def brute_force_matching(
    descriptors_a,
    descriptors_b
):

    """
    Find the two nearest descriptors in B for each descriptor in A.
    """

    if (
        len(descriptors_a) == 0
        or
        len(descriptors_b) < 2
    ):

        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 2), dtype=np.float32)
        )

    distance_matrix = np.sqrt(
        np.maximum(
            np.sum(
                (
                    descriptors_a[:, None, :]
                    -
                    descriptors_b[None, :, :]
                ) ** 2,
                axis=2
            ),
            0.0
        )
    )

    nearest_indices = np.argsort(
        distance_matrix,
        axis=1
    )[:, :2]

    nearest_distances = np.take_along_axis(
        distance_matrix,
        nearest_indices,
        axis=1
    )

    return (
        nearest_indices,
        nearest_distances
    )


# ====================================================================
# STEP 9 — RATIO TEST
# ====================================================================

def ratio_test(
    nearest_indices,
    nearest_distances,
    threshold=0.85
):

    matches = []

    ratios = []

    for i in range(
        len(nearest_indices)
    ):

        d1 = nearest_distances[i, 0]
        d2 = nearest_distances[i, 1]

        ratio = d1 / (
            d2 + 1e-8
        )

        ratios.append(
            ratio
        )

        if ratio < threshold:

            matches.append(
                (
                    i,
                    int(nearest_indices[i, 0])
                )
            )

    return (
        np.asarray(
            matches,
            dtype=np.int32
        ).reshape(-1, 2),

        np.asarray(
            ratios,
            dtype=np.float32
        )
    )


# ====================================================================
# STEP 10 — MUTUAL NEAREST-NEIGHBOR MATCHING
# ====================================================================

def mutual_matching(
    descriptors_a,
    descriptors_b
):

    if (
        len(descriptors_a) == 0
        or
        len(descriptors_b) == 0
    ):

        return np.empty(
            (0, 2),
            dtype=np.int32
        )

    distance_matrix = np.sqrt(
        np.maximum(
            np.sum(
                (
                    descriptors_a[:, None, :]
                    -
                    descriptors_b[None, :, :]
                ) ** 2,
                axis=2
            ),
            0.0
        )
    )

    nearest_b_for_a = np.argmin(
        distance_matrix,
        axis=1
    )

    nearest_a_for_b = np.argmin(
        distance_matrix,
        axis=0
    )

    matches = []

    for index_a, index_b in enumerate(
        nearest_b_for_a
    ):

        if nearest_a_for_b[index_b] == index_a:

            matches.append(
                (index_a, index_b)
            )

    return np.asarray(
        matches,
        dtype=np.int32
    ).reshape(-1, 2)


# ====================================================================
# STEP 11 — HOMOGRAPHY ESTIMATION USING DLT
# ====================================================================

def estimate_homography(
    points_a,
    points_b
):

    """
    Estimate H using the Direct Linear Transform.

    points_a and points_b have shape (N, 2).

    At least four non-degenerate correspondences are required.
    """

    if len(points_a) < 4:

        raise ValueError(
            "At least four correspondences are required."
        )

    rows = []

    for (x, y), (u, v) in zip(
        points_a,
        points_b
    ):

        rows.append(
            [
                -x, -y, -1,
                0, 0, 0,
                u * x, u * y, u
            ]
        )

        rows.append(
            [
                0, 0, 0,
                -x, -y, -1,
                v * x, v * y, v
            ]
        )

    A = np.asarray(
        rows,
        dtype=np.float64
    )

    # Solve Ah = 0 using the right singular vector associated
    # with the smallest singular value.

    _, singular_values, vt = np.linalg.svd(
        A
    )

    if (
        len(singular_values) < 8
        or
        singular_values[-2] < 1e-10
    ):

        raise ValueError(
            "Degenerate homography configuration."
        )

    h = vt[-1]

    H = h.reshape(
        3,
        3
    )

    if abs(H[2, 2]) > 1e-10:

        H /= H[2, 2]

    else:

        norm = np.linalg.norm(H)

        if norm < 1e-10:
            raise ValueError("Invalid homography.")

        H /= norm

    if not np.all(
        np.isfinite(H)
    ):

        raise ValueError(
            "Homography contains invalid values."
        )

    return H


# ====================================================================
# STEP 12 — APPLY HOMOGRAPHY
# ====================================================================

def transform_points(points, H):

    homogeneous = np.column_stack(
        [
            points,
            np.ones(
                len(points)
            )
        ]
    )

    transformed = (
        H @ homogeneous.T
    ).T

    denominator = transformed[:, 2]

    output = np.full(
        (len(points), 2),
        np.nan,
        dtype=np.float64
    )

    valid = np.abs(
        denominator
    ) > 1e-10

    output[valid] = (
        transformed[valid, :2]
        /
        denominator[valid, None]
    )

    return output


# ====================================================================
# STEP 13 — REPROJECTION ERRORS
# ====================================================================

def reprojection_errors(
    points_a,
    points_b,
    H
):

    predicted = transform_points(
        points_a,
        H
    )

    errors = np.linalg.norm(
        predicted - points_b,
        axis=1
    )

    errors[~np.isfinite(errors)] = np.inf

    return errors


# ====================================================================
# STEP 14 — RANSAC HOMOGRAPHY
# ====================================================================

def ransac_homography(
    points_a,
    points_b,
    iterations=500,
    threshold=3.0,
    seed=42
):

    """
    Estimate a homography robustly.

    The model with the largest inlier count is retained.
    """

    if len(points_a) != len(points_b):

        raise ValueError(
            "Correspondence arrays must have equal length."
        )

    if len(points_a) < 4:

        raise ValueError(
            "RANSAC requires at least four correspondences."
        )

    rng = np.random.default_rng(
        seed
    )

    best_H = None

    best_inliers = None

    best_count = 0

    for _ in range(iterations):

        sample_indices = rng.choice(
            len(points_a),
            size=4,
            replace=False
        )

        try:

            H = estimate_homography(
                points_a[sample_indices],
                points_b[sample_indices]
            )

        except (
            ValueError,
            np.linalg.LinAlgError
        ):

            continue

        errors = reprojection_errors(
            points_a,
            points_b,
            H
        )

        inliers = errors < threshold

        count = int(
            np.sum(inliers)
        )

        if count > best_count:

            best_count = count

            best_H = H

            best_inliers = inliers

    if best_H is None or best_count < 4:

        raise RuntimeError(
            "RANSAC could not find a valid homography."
        )

    # Refit using all best inliers.

    try:

        refined_H = estimate_homography(
            points_a[best_inliers],
            points_b[best_inliers]
        )

        refined_errors = reprojection_errors(
            points_a,
            points_b,
            refined_H
        )

        refined_inliers = (
            refined_errors < threshold
        )

        if np.sum(refined_inliers) >= best_count:

            best_H = refined_H

            best_inliers = refined_inliers

    except (
        ValueError,
        np.linalg.LinAlgError
    ):

        pass

    return best_H, best_inliers


# ====================================================================
# STEP 15 — MATCHING VISUALIZATION
# ====================================================================

def visualize_matches(
    image_a,
    image_b,
    keypoints_a,
    keypoints_b,
    matches,
    inliers=None,
    title="Feature Matches"
):

    height = max(
        image_a.shape[0],
        image_b.shape[0]
    )

    width_a = image_a.shape[1]
    width_b = image_b.shape[1]

    canvas = np.zeros(
        (height, width_a + width_b),
        dtype=np.float32
    )

    canvas[
        :image_a.shape[0],
        :width_a
    ] = image_a

    canvas[
        :image_b.shape[0],
        width_a:
    ] = image_b

    fig, ax = plt.subplots(
        figsize=(16, 8)
    )

    ax.imshow(
        canvas,
        cmap="gray"
    )

    for index, (index_a, index_b) in enumerate(
        matches
    ):

        x1, y1 = keypoints_a[index_a]

        x2, y2 = keypoints_b[index_b]

        x2 += width_a

        if inliers is not None and inliers[index]:

            linewidth = 1.5
            alpha = 0.9

        else:

            linewidth = 0.7
            alpha = 0.25

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=linewidth,
            alpha=alpha
        )

        ax.scatter(
            [x1, x2],
            [y1, y2],
            s=12,
            alpha=alpha
        )

    ax.set_title(
        title
    )

    ax.axis("off")

    plt.tight_layout()


# ====================================================================
# STEP 16 — NUMERICAL TESTS
# ====================================================================

def run_tests():

    print("\nRUNNING TESTS")
    print("-" * 70)

    # ---------------------------------------------------------------
    # Test 1: Descriptor dimension
    # ---------------------------------------------------------------

    test_image = np.arange(
        400,
        dtype=np.float32
    ).reshape(
        20,
        20
    )

    test_keypoints = np.array(
        [
            [10, 10]
        ],
        dtype=np.int32
    )

    descriptors, valid_points = extract_descriptors(
        test_image,
        test_keypoints,
        patch_radius=2
    )

    assert descriptors.shape == (1, 25)

    print("Descriptor dimension test passed.")

    # ---------------------------------------------------------------
    # Test 2: Descriptor normalization
    # ---------------------------------------------------------------

    assert np.isclose(
        np.mean(descriptors[0]),
        0.0,
        atol=1e-5
    )

    assert np.isclose(
        np.std(descriptors[0]),
        1.0,
        atol=1e-5
    )

    print("Descriptor normalization test passed.")

    # ---------------------------------------------------------------
    # Test 3: Matching identity
    # ---------------------------------------------------------------

    test_descriptors = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2]
        ],
        dtype=np.float32
    )

    indices, distances = brute_force_matching(
        test_descriptors,
        test_descriptors
    )

    assert indices.shape == (3, 2)

    assert np.allclose(
        distances[:, 0],
        0.0
    )

    print("Nearest-neighbor identity test passed.")

    # ---------------------------------------------------------------
    # Test 4: Ratio test
    # ---------------------------------------------------------------

    test_indices = np.array(
        [
            [0, 1],
            [1, 0]
        ]
    )

    test_distances = np.array(
        [
            [1.0, 5.0],
            [4.0, 4.1]
        ]
    )

    matches, ratios = ratio_test(
        test_indices,
        test_distances,
        threshold=0.8
    )

    assert len(matches) == 1
    assert matches[0, 0] == 0

    print("Ratio-test test passed.")

    # ---------------------------------------------------------------
    # Test 5: Mutual matching
    # ---------------------------------------------------------------

    a = np.array(
        [
            [0.0, 0.0],
            [10.0, 10.0]
        ]
    )

    b = np.array(
        [
            [0.1, 0.1],
            [10.1, 10.1]
        ]
    )

    mutual = mutual_matching(
        a,
        b
    )

    assert len(mutual) == 2

    print("Mutual matching test passed.")

    # ---------------------------------------------------------------
    # Test 6: Homography translation
    # ---------------------------------------------------------------

    points_a = np.array(
        [
            [10, 10],
            [100, 10],
            [10, 100],
            [100, 100],
            [50, 60],
            [140, 80]
        ],
        dtype=np.float64
    )

    dx = 12
    dy = 8

    points_b = points_a + np.array(
        [dx, dy]
    )

    H = estimate_homography(
        points_a,
        points_b
    )

    transformed = transform_points(
        points_a,
        H
    )

    assert np.allclose(
        transformed,
        points_b,
        atol=1e-5
    )

    print("Homography translation test passed.")

    # ---------------------------------------------------------------
    # Test 7: RANSAC with outliers
    # ---------------------------------------------------------------

    rng = np.random.default_rng(
        123
    )

    inlier_a = rng.uniform(
        10,
        100,
        size=(20, 2)
    )

    inlier_b = inlier_a + np.array(
        [5.0, 7.0]
    )

    outlier_a = rng.uniform(
        10,
        100,
        size=(8, 2)
    )

    outlier_b = rng.uniform(
        10,
        100,
        size=(8, 2)
    )

    all_a = np.vstack(
        [
            inlier_a,
            outlier_a
        ]
    )

    all_b = np.vstack(
        [
            inlier_b,
            outlier_b
        ]
    )

    H_ransac, inlier_mask = ransac_homography(
        all_a,
        all_b,
        iterations=300,
        threshold=0.1,
        seed=123
    )

    assert np.sum(inlier_mask) >= 18

    transformed = transform_points(
        all_a[inlier_mask],
        H_ransac
    )

    assert np.mean(
        np.linalg.norm(
            transformed - all_b[inlier_mask],
            axis=1
        )
    ) < 0.1

    print("RANSAC outlier-rejection test passed.")

    print("-" * 70)
    print("ALL TESTS PASSED!")


# ====================================================================
# STEP 17 — MAIN EXPERIMENT
# ====================================================================

def main():

    print("=" * 70)
    print("COMPUTER VISION FROM SCRATCH")
    print("LESSON 13 — FEATURE MATCHING")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Create two images
    # ---------------------------------------------------------------

    image_a = create_scene()

    image_b = translate_image(
        image_a,
        SHIFT_X,
        SHIFT_Y
    )

    print("\nIMAGE INFORMATION")
    print("-" * 70)

    print("Image A:", image_a.shape)
    print("Image B:", image_b.shape)

    print(
        "Known translation:",
        (SHIFT_X, SHIFT_Y)
    )

    # ---------------------------------------------------------------
    # Harris detection
    # ---------------------------------------------------------------

    response_a = harris_response(
        image_a
    )

    response_b = harris_response(
        image_b
    )

    threshold_a = (
        np.max(response_a)
        *
        HARRIS_THRESHOLD
    )

    threshold_b = (
        np.max(response_b)
        *
        HARRIS_THRESHOLD
    )

    keypoints_a = non_maximum_suppression(
        response_a,
        threshold_a,
        radius=NMS_RADIUS,
        max_points=MAX_KEYPOINTS
    )

    keypoints_b = non_maximum_suppression(
        response_b,
        threshold_b,
        radius=NMS_RADIUS,
        max_points=MAX_KEYPOINTS
    )

    print("\nKEYPOINT DETECTION")
    print("-" * 70)

    print("Keypoints A:", len(keypoints_a))
    print("Keypoints B:", len(keypoints_b))

    # ---------------------------------------------------------------
    # Extract descriptors
    # ---------------------------------------------------------------

    descriptors_a, keypoints_a = extract_descriptors(
        image_a,
        keypoints_a,
        patch_radius=PATCH_RADIUS
    )

    descriptors_b, keypoints_b = extract_descriptors(
        image_b,
        keypoints_b,
        patch_radius=PATCH_RADIUS
    )

    print("\nDESCRIPTOR EXTRACTION")
    print("-" * 70)

    print("Descriptors A:", descriptors_a.shape)
    print("Descriptors B:", descriptors_b.shape)

    # ---------------------------------------------------------------
    # Brute-force matching
    # ---------------------------------------------------------------

    nearest_indices, nearest_distances = brute_force_matching(
        descriptors_a,
        descriptors_b
    )

    matches, ratios = ratio_test(
        nearest_indices,
        nearest_distances,
        threshold=RATIO_THRESHOLD
    )

    print("\nDESCRIPTOR MATCHING")
    print("-" * 70)

    print(
        "Ratio-test matches:",
        len(matches)
    )

    # ---------------------------------------------------------------
    # Mutual matching
    # ---------------------------------------------------------------

    mutual_matches = mutual_matching(
        descriptors_a,
        descriptors_b
    )

    print(
        "Mutual matches:",
        len(mutual_matches)
    )

    # ---------------------------------------------------------------
    # Use mutual matches for geometric verification
    # ---------------------------------------------------------------

    if len(mutual_matches) >= 4:

        points_a = keypoints_a[
            mutual_matches[:, 0]
        ].astype(np.float64)

        points_b = keypoints_b[
            mutual_matches[:, 1]
        ].astype(np.float64)

        H, inlier_mask = ransac_homography(
            points_a,
            points_b,
            iterations=RANSAC_ITERATIONS,
            threshold=REPROJECTION_THRESHOLD,
            seed=RANDOM_SEED
        )

        inlier_count = int(
            np.sum(inlier_mask)
        )

        print("\nRANSAC HOMOGRAPHY")
        print("-" * 70)

        print("Estimated homography:")
        print(H)

        print(
            "Geometric inliers:",
            inlier_count
        )

        print(
            "Total mutual matches:",
            len(mutual_matches)
        )

        print(
            "Inlier ratio:",
            round(
                inlier_count / len(mutual_matches),
                4
            )
        )

        estimated_translation = (
            H[0, 2],
            H[1, 2]
        )

        print(
            "Estimated translation:",
            estimated_translation
        )

    else:

        H = None

        inlier_mask = None

        print(
            "\nNot enough mutual matches for homography estimation."
        )

    # ---------------------------------------------------------------
    # Visualization 1 — Input images
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5)
    )

    axes[0].imshow(
        image_a,
        cmap="gray"
    )

    axes[0].set_title(
        "Image A"
    )

    axes[1].imshow(
        image_b,
        cmap="gray"
    )

    axes[1].set_title(
        "Image B — Translated"
    )

    for ax in axes:

        ax.axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 2 — Harris responses
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5)
    )

    axes[0].imshow(
        response_a,
        cmap="viridis"
    )

    axes[0].scatter(
        keypoints_a[:, 0],
        keypoints_a[:, 1],
        s=12
    )

    axes[0].set_title(
        "Harris Response — Image A"
    )

    axes[1].imshow(
        response_b,
        cmap="viridis"
    )

    axes[1].scatter(
        keypoints_b[:, 0],
        keypoints_b[:, 1],
        s=12
    )

    axes[1].set_title(
        "Harris Response — Image B"
    )

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 3 — Ratio-test matches
    # ---------------------------------------------------------------

    visualize_matches(
        image_a,
        image_b,
        keypoints_a,
        keypoints_b,
        matches,
        title="Descriptor Matches After Ratio Test"
    )

    # ---------------------------------------------------------------
    # Visualization 4 — RANSAC-filtered mutual matches
    # ---------------------------------------------------------------

    if len(mutual_matches) >= 4:

        visualize_matches(
            image_a,
            image_b,
            keypoints_a,
            keypoints_b,
            mutual_matches,
            inliers=inlier_mask,
            title="Mutual Matches — RANSAC Inliers and Outliers"
        )

    # ---------------------------------------------------------------
    # Visualization 5 — Descriptor distance
    # ---------------------------------------------------------------

    if len(nearest_distances) > 0:

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.hist(
            nearest_distances[:, 0],
            bins=25
        )

        ax.set_title(
            "Nearest-Neighbor Descriptor Distances"
        )

        ax.set_xlabel(
            "Euclidean Distance"
        )

        ax.set_ylabel(
            "Number of Matches"
        )

        ax.grid(
            alpha=0.3
        )

        plt.tight_layout()

    # ---------------------------------------------------------------
    # Numerical tests
    # ---------------------------------------------------------------

    run_tests()

    # ---------------------------------------------------------------
    # Robotics interpretation
    # ---------------------------------------------------------------

    print("\nOPENARM V2 — ROBOTICS INTERPRETATION")
    print("-" * 70)

    print(
        "Feature correspondence connects visual observations"
    )

    print(
        "from different camera frames."
    )

    print("\nPotential applications:")

    print("  1. Camera motion estimation")
    print("  2. Visual odometry")
    print("  3. Image registration")
    print("  4. Visual localization")
    print("  5. SLAM initialization")

    print(
        "\nFor OpenArm waste sorting, these methods can help"
    )

    print(
        "understand how the camera or scene has moved."
    )

    print(
        "\nA homography is not a complete 3-D motion model."
    )

    print(
        "Depth, camera calibration, and 3-D geometry are needed"
    )

    print(
        "for accurate robot-world localization."
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("\nLESSON 13 SUMMARY")
    print("-" * 70)

    print("1. Implemented Harris keypoint detection.")
    print("2. Extracted normalized local patch descriptors.")
    print("3. Implemented brute-force descriptor matching.")
    print("4. Implemented Lowe's ratio test.")
    print("5. Implemented mutual nearest-neighbor matching.")
    print("6. Estimated homography using DLT.")
    print("7. Implemented RANSAC geometric verification.")
    print("8. Visualized feature correspondences.")

    print("\nNEXT LESSON:")
    print("LESSON 14 — OPTICAL FLOW FROM SCRATCH")

    plt.show()


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()