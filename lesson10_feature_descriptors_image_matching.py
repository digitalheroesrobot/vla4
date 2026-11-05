"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 10 — FEATURE DESCRIPTORS AND IMAGE MATCHING
======================================================================

COURSE RULE
-----------

This course is built from scratch.

NO:
    - pretrained models
    - pretrained weights
    - SIFT
    - SURF
    - ORB
    - cv2.BFMatcher()
    - cv2.FlannBasedMatcher()
    - black-box feature descriptors
    - black-box feature matching

We implement the mathematics using:

    - NumPy
    - Matplotlib

----------------------------------------------------------------------
WHAT YOU LEARN
----------------------------------------------------------------------

LESSON 09 gave us:

    IMAGE
      ↓
    GRADIENT
      ↓
    HARRIS RESPONSE
      ↓
    KEYPOINTS

But a keypoint coordinate alone is not enough.

Suppose we detect:

    keypoint A = (120, 80)

in image 1.

How do we know which point in image 2 corresponds to it?

We need a description of the local image around the keypoint.

That is the purpose of a:

    FEATURE DESCRIPTOR

Then we need:

    FEATURE MATCHING

----------------------------------------------------------------------
STEP 1 — KEYPOINT VS DESCRIPTOR
----------------------------------------------------------------------

A keypoint tells us:

    WHERE?

Example:

    (x, y) = (120, 80)

A descriptor tells us:

    WHAT DOES THE LOCAL REGION LOOK LIKE?

Therefore:

    KEYPOINT
        =
    location

    DESCRIPTOR
        =
    local visual signature

A complete feature can therefore be thought of as:

    FEATURE = (KEYPOINT, DESCRIPTOR)

----------------------------------------------------------------------
STEP 2 — LOCAL IMAGE PATCH
----------------------------------------------------------------------

The simplest descriptor is just a local image patch.

For a keypoint:

    (x, y)

we extract:

    image[y-r:y+r+1,
          x-r:x+r+1]

If:

    r = 7

then:

    patch size = 15 × 15

and the descriptor contains:

    225 values

Flattening gives:

    descriptor ∈ R^225

This is not a sophisticated descriptor, but it is an excellent
way to understand the fundamental concept.

----------------------------------------------------------------------
STEP 3 — WHY NORMALIZE THE PATCH?
----------------------------------------------------------------------

Suppose two images contain the same object.

Image A:

    [50, 100, 150]

Image B:

    [70, 120, 170]

The structure is similar, but brightness changed.

A raw descriptor may report a large difference.

We can reduce brightness sensitivity using:

    mean = (1/N) Σ xi

    standard deviation:

        σ = sqrt(
                (1/N) Σ (xi - mean)²
            )

Then:

    normalized_descriptor:

        di = (xi - mean) / σ

This removes:

    - average brightness
    - much of the local contrast scale

----------------------------------------------------------------------
STEP 4 — DESCRIPTOR DISTANCE
----------------------------------------------------------------------

Once we have descriptors, we need a similarity measure.

----------------------------------------------------------------------
EUCLIDEAN DISTANCE
----------------------------------------------------------------------

For descriptors A and B:

    d(A,B)
      =
    sqrt(
        Σ (Ai - Bi)²
    )

Smaller:

    d

means more similar.

----------------------------------------------------------------------
SSD
----------------------------------------------------------------------

Squared Euclidean distance:

    SSD(A,B)
      =
    Σ (Ai - Bi)²

This avoids the square root.

Smaller:

    SSD

means more similar.

----------------------------------------------------------------------
STEP 5 — NORMALIZED CROSS CORRELATION
----------------------------------------------------------------------

Another useful similarity measure is:

    NCC(A,B)
      =
    Σ (Ai-μA)(Bi-μB)
    -----------------
    sqrt(
        Σ(Ai-μA)²
        Σ(Bi-μB)²
    )

Interpretation:

    NCC ≈ +1
        → very similar

    NCC ≈ 0
        → weak relationship

    NCC ≈ -1
        → opposite pattern

NCC is particularly useful because it is less sensitive to
brightness offsets.

----------------------------------------------------------------------
STEP 6 — BRUTE-FORCE FEATURE MATCHING
----------------------------------------------------------------------

Suppose image A has:

    N descriptors

and image B has:

    M descriptors

The simplest strategy is:

    for every descriptor in A:

        compare with every descriptor in B

        choose the closest one

Complexity:

    O(N × M × D)

where:

    N = number of features in image A
    M = number of features in image B
    D = descriptor dimension

For our 15×15 patch:

    D = 225

This is slow for very large feature sets, but excellent for
learning.

Later, approximate nearest-neighbor methods can make matching
much faster.

----------------------------------------------------------------------
STEP 7 — RATIO TEST
----------------------------------------------------------------------

Nearest-neighbor matching alone can produce ambiguous matches.

Suppose a descriptor has:

    best distance  = d1
    second distance = d2

If:

    d1 / d2

is small, the best match is much better than the second-best.

A common rule is:

    d1 / d2 < ratio_threshold

For example:

    0.75

This rejects ambiguous matches.

----------------------------------------------------------------------
STEP 8 — CROSS-CHECK / MUTUAL MATCHING
----------------------------------------------------------------------

Suppose:

    A1 → B5

But:

    B5 → A8

Then the match is inconsistent.

A mutual match requires:

    A1 → B5

AND:

    B5 → A1

This can reduce incorrect correspondences.

----------------------------------------------------------------------
STEP 9 — SYNTHETIC IMAGE TRANSFORMATION
----------------------------------------------------------------------

We will create:

    image1

and then generate:

    image2

by translating image1.

Known translation:

    dx = 18 pixels
    dy = 12 pixels

Therefore:

    x2 ≈ x1 + 18
    y2 ≈ y1 + 12

This gives us ground truth.

We can therefore evaluate whether our feature matching system
recovers the known motion.

This is an important computer-vision development technique:

    create a known transformation
        ↓
    run algorithm
        ↓
    compare estimate with ground truth

----------------------------------------------------------------------
STEP 10 — BRIGHTNESS CHANGE
----------------------------------------------------------------------

We will also change image brightness:

    image2 = image1 * scale + offset

This lets us compare:

    raw patch descriptors

against:

    normalized patch descriptors

The expectation is:

    normalized descriptors
        →
    more robust to brightness changes

----------------------------------------------------------------------
STEP 11 — ROBUST MATCHING PIPELINE
----------------------------------------------------------------------

Our pipeline becomes:

    IMAGE 1
       ↓
    HARRIS KEYPOINTS
       ↓
    PATCH DESCRIPTORS
       ↓
    IMAGE 2
       ↓
    HARRIS KEYPOINTS
       ↓
    PATCH DESCRIPTORS
       ↓
    DESCRIPTOR DISTANCE
       ↓
    NEAREST NEIGHBOR
       ↓
    RATIO TEST
       ↓
    CROSS CHECK
       ↓
    MATCHES
       ↓
    MOTION ESTIMATION

----------------------------------------------------------------------
STEP 12 — MOTION ESTIMATION
----------------------------------------------------------------------

Once we have corresponding points:

    p1 = (x1, y1)
    p2 = (x2, y2)

we can estimate translation:

    dx = x2 - x1
    dy = y2 - y1

If many matches are correct:

    median(dx)

and:

    median(dy)

should approach the true translation.

We use the median because it is more robust to a few bad matches
than the ordinary mean.

----------------------------------------------------------------------
STEP 13 — REAL-WORLD ROBOTICS EXAMPLE
----------------------------------------------------------------------

For OpenArm V2:

    camera
       ↓
    image
       ↓
    Harris keypoints
       ↓
    local descriptors
       ↓
    feature matching
       ↓
    motion estimation
       ↓
    camera/object movement
       ↓
    geometric reasoning
       ↓
    robot control

Possible applications:

    - tracking a waste object
    - tracking conveyor movement
    - estimating camera motion
    - visual odometry
    - localization
    - matching calibration targets
    - detecting movement between frames

For example:

    frame t
       ↓
    feature points

    frame t+1
       ↓
    feature points

    match
       ↓
    displacement

    displacement
       ↓
    object/camera motion

----------------------------------------------------------------------
STEP 14 — IMPORTANT LIMITATIONS
----------------------------------------------------------------------

Raw image patches are simple but weak descriptors.

They can be sensitive to:

    - rotation
    - scale
    - large viewpoint changes
    - occlusion
    - deformation
    - illumination
    - noise

More advanced descriptors solve some of these problems.

Examples studied later:

    - gradient orientation descriptors
    - ORB-style binary descriptors
    - learned descriptors
    - deep feature embeddings

But we first need to understand the fundamental idea.

----------------------------------------------------------------------
STEP 15 — FROM-SCRATCH IMPLEMENTATION
----------------------------------------------------------------------

We implement:

    1. synthetic image
    2. translated image
    3. Harris keypoints
    4. patch extraction
    5. raw patch descriptors
    6. normalized descriptors
    7. SSD
    8. Euclidean distance
    9. NCC
   10. brute-force matching
   11. ratio test
   12. cross-check
   13. translation estimation
   14. visualization
   15. numerical tests

----------------------------------------------------------------------
EXERCISES
----------------------------------------------------------------------

1. Change the translation.

2. Add Gaussian noise.

3. Change brightness.

4. Change contrast.

5. Rotate image2.

6. Change patch size.

7. Change the Harris threshold.

8. Change the ratio threshold.

9. Compare raw descriptors against normalized descriptors.

10. Replace SSD with NCC matching.

11. Add a second object to image2.

12. Create false features and study outliers.

----------------------------------------------------------------------
NEXT LESSON
----------------------------------------------------------------------

LESSON 11 — CONTOURS AND CONNECTED COMPONENTS

We will build:

    binary mask
        ↓
    connected components
        ↓
    contours
        ↓
    bounding boxes
        ↓
    object centers
        ↓
    object regions

This will move the course from local feature geometry toward
actual object-region analysis.

======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# STEP 1 — SYNTHETIC IMAGE
# ======================================================================

HEIGHT = 320
WIDTH = 480

image1 = np.zeros(
    (HEIGHT, WIDTH),
    dtype=np.float32
)

image1[:] = 30.0


# ----------------------------------------------------------------------
# Large rectangle
# ----------------------------------------------------------------------

image1[
    50:180,
    60:220
] = 210.0


# ----------------------------------------------------------------------
# Small square
# ----------------------------------------------------------------------

image1[
    210:280,
    80:150
] = 180.0


# ----------------------------------------------------------------------
# Second rectangle
# ----------------------------------------------------------------------

image1[
    70:220,
    280:410
] = 200.0


# ----------------------------------------------------------------------
# Thin horizontal structure
# ----------------------------------------------------------------------

image1[
    240:255,
    200:390
] = 230.0


# ----------------------------------------------------------------------
# Circle
# ----------------------------------------------------------------------

yy, xx = np.ogrid[
    :HEIGHT,
    :WIDTH
]

circle = (
    (xx - 390) ** 2
    +
    (yy - 270) ** 2
    <= 35 ** 2
)

image1[circle] = 160.0


# ----------------------------------------------------------------------
# Diagonal structure
# ----------------------------------------------------------------------

for y in range(30, 180):

    x = int(
        230 + 0.6 * y
    )

    if 0 <= x < WIDTH:

        image1[
            y,
            x
        ] = 255.0


# ======================================================================
# STEP 2 — CREATE TRANSLATED IMAGE
# ======================================================================

SHIFT_X = 18
SHIFT_Y = 12


def translate_image(
    image,
    shift_x,
    shift_y
):
    """
    Translate an image using zero-filled boundaries.

    Positive shift_x:
        moves image to the right.

    Positive shift_y:
        moves image downward.
    """

    height, width = image.shape

    output = np.zeros_like(
        image
    )

    source_x0 = max(
        0,
        -shift_x
    )

    source_x1 = min(
        width,
        width - shift_x
    )

    source_y0 = max(
        0,
        -shift_y
    )

    source_y1 = min(
        height,
        height - shift_y
    )

    target_x0 = max(
        0,
        shift_x
    )

    target_x1 = target_x0 + (
        source_x1 - source_x0
    )

    target_y0 = max(
        0,
        shift_y
    )

    target_y1 = target_y0 + (
        source_y1 - source_y0
    )

    if (
        source_x1 > source_x0
        and
        source_y1 > source_y0
    ):

        output[
            target_y0:target_y1,
            target_x0:target_x1
        ] = image[
            source_y0:source_y1,
            source_x0:source_x1
        ]

    return output


image2 = translate_image(
    image1,
    SHIFT_X,
    SHIFT_Y
)


# ----------------------------------------------------------------------
# Add brightness and contrast change.
# ----------------------------------------------------------------------

image2 = (
    image2 * 1.05
    +
    5.0
)

image2 = np.clip(
    image2,
    0.0,
    255.0
).astype(
    np.float32
)


# ======================================================================
# STEP 3 — FROM-SCRATCH CONVOLUTION
# ======================================================================

def zero_pad(
    image,
    kernel_shape
):
    """
    Zero-pad an image.
    """

    kh, kw = kernel_shape

    pad_y = kh // 2
    pad_x = kw // 2

    return np.pad(
        image,
        (
            (pad_y, pad_y),
            (pad_x, pad_x)
        ),
        mode="constant",
        constant_values=0.0
    )


def convolve2d(
    image,
    kernel
):
    """
    Basic 2-D convolution from scratch.
    """

    image = image.astype(
        np.float32
    )

    kernel = kernel.astype(
        np.float32
    )

    flipped = np.flip(
        kernel
    )

    padded = zero_pad(
        image,
        kernel.shape
    )

    height, width = image.shape

    output = np.zeros(
        image.shape,
        dtype=np.float32
    )

    kh, kw = kernel.shape

    for y in range(height):

        for x in range(width):

            region = padded[
                y:y + kh,
                x:x + kw
            ]

            output[y, x] = np.sum(
                region * flipped
            )

    return output


# ======================================================================
# STEP 4 — HARRIS CORNER DETECTOR
# ======================================================================

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


def box_filter(
    image,
    window_size
):
    """
    Local average using a box filter.
    """

    assert (
        window_size % 2 == 1
    )

    kernel = np.ones(
        (
            window_size,
            window_size
        ),
        dtype=np.float32
    )

    kernel /= (
        window_size
        *
        window_size
    )

    return convolve2d(
        image,
        kernel
    )


def harris_response_map(
    image,
    window_size=7,
    k=0.04
):
    """
    Compute the Harris response map from scratch.
    """

    Ix = convolve2d(
        image,
        sobel_x
    )

    Iy = convolve2d(
        image,
        sobel_y
    )

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy

    Sxx = box_filter(
        Ix2,
        window_size
    )

    Syy = box_filter(
        Iy2,
        window_size
    )

    Sxy = box_filter(
        IxIy,
        window_size
    )

    determinant = (
        Sxx * Syy
        -
        Sxy * Sxy
    )

    trace = (
        Sxx
        +
        Syy
    )

    response = (
        determinant
        -
        k * trace ** 2
    )

    return response


def non_maximum_suppression(
    response,
    candidate_mask,
    radius
):
    """
    From-scratch local non-maximum suppression.
    """

    height, width = response.shape

    output = np.zeros(
        response.shape,
        dtype=bool
    )

    for y in range(height):

        for x in range(width):

            if not candidate_mask[y, x]:
                continue

            y0 = max(
                0,
                y - radius
            )

            y1 = min(
                height,
                y + radius + 1
            )

            x0 = max(
                0,
                x - radius
            )

            x1 = min(
                width,
                x + radius + 1
            )

            region = response[
                y0:y1,
                x0:x1
            ]

            if (
                response[y, x]
                >=
                np.max(region)
            ):

                output[y, x] = True

    return output


def detect_harris_keypoints(
    image,
    window_size=7,
    k=0.04,
    threshold=0.05,
    nms_radius=6,
    max_keypoints=80
):
    """
    Full Harris keypoint detector.

    Returns:

        keypoints
            shape = (N, 2)

            columns:
                x
                y

        scores
            Harris response at each keypoint

        response
            full response map
    """

    response = harris_response_map(
        image,
        window_size,
        k
    )

    max_abs = np.max(
        np.abs(response)
    )

    if max_abs > 0:

        normalized = (
            response
            /
            max_abs
        )

    else:

        normalized = np.zeros_like(
            response
        )

    candidates = (
        normalized
        >
        threshold
    )

    suppressed = (
        non_maximum_suppression(
            response,
            candidates,
            nms_radius
        )
    )

    y, x = np.where(
        suppressed
    )

    scores = response[
        y,
        x
    ]

    if len(scores) > 0:

        order = np.argsort(
            scores
        )[::-1]

        order = order[
            :max_keypoints
        ]

        x = x[
            order
        ]

        y = y[
            order
        ]

        scores = scores[
            order
        ]

    keypoints = np.column_stack(
        (
            x,
            y
        )
    ).astype(
        np.int32
    )

    return (
        keypoints,
        scores,
        response
    )


# ======================================================================
# STEP 5 — DETECT KEYPOINTS IN BOTH IMAGES
# ======================================================================

keypoints1, scores1, response1 = (
    detect_harris_keypoints(
        image1,
        window_size=7,
        k=0.04,
        threshold=0.05,
        nms_radius=6,
        max_keypoints=80
    )
)

keypoints2, scores2, response2 = (
    detect_harris_keypoints(
        image2,
        window_size=7,
        k=0.04,
        threshold=0.05,
        nms_radius=6,
        max_keypoints=80
    )
)


print("=" * 72)
print(
    "LESSON 10 — FEATURE DESCRIPTORS AND IMAGE MATCHING"
)
print("=" * 72)

print("\nKEYPOINTS")
print("-" * 72)

print(
    "Image 1 keypoints:",
    len(keypoints1)
)

print(
    "Image 2 keypoints:",
    len(keypoints2)
)


# ======================================================================
# STEP 6 — PATCH EXTRACTION
# ======================================================================

PATCH_RADIUS = 7


def extract_patch(
    image,
    x,
    y,
    radius
):
    """
    Extract a local square patch.

    Patch size:

        (2*radius + 1)
        ×
        (2*radius + 1)

    Keypoints too close to the boundary are rejected.
    """

    size = (
        2 * radius
        +
        1
    )

    height, width = image.shape

    if (
        x - radius < 0
        or
        x + radius >= width
        or
        y - radius < 0
        or
        y + radius >= height
    ):
        return None

    patch = image[
        y - radius:y + radius + 1,
        x - radius:x + radius + 1
    ]

    assert patch.shape == (
        size,
        size
    )

    return patch.copy()


# ======================================================================
# STEP 7 — PATCH NORMALIZATION
# ======================================================================

def normalize_descriptor(
    patch
):
    """
    Convert a patch into a zero-mean, unit-standard-deviation
    descriptor.

        d = (x - mean) / std

    If std is extremely small, return None because the patch does not
    contain enough variation to make a useful normalized descriptor.
    """

    vector = patch.astype(
        np.float32
    ).reshape(
        -1
    )

    mean = np.mean(
        vector
    )

    std = np.std(
        vector
    )

    if std < 1e-8:

        return None

    descriptor = (
        vector - mean
    ) / std

    return descriptor.astype(
        np.float32
    )


def raw_descriptor(
    patch
):
    """
    Flatten the raw image patch.
    """

    return patch.astype(
        np.float32
    ).reshape(
        -1
    )


def build_descriptors(
    image,
    keypoints,
    radius,
    normalize=True
):
    """
    Extract descriptors for valid keypoints.

    Invalid boundary keypoints and constant patches are skipped.
    """

    descriptors = []
    valid_keypoints = []

    for x, y in keypoints:

        patch = extract_patch(
            image,
            int(x),
            int(y),
            radius
        )

        if patch is None:
            continue

        if normalize:

            descriptor = (
                normalize_descriptor(
                    patch
                )
            )

            if descriptor is None:
                continue

        else:

            descriptor = raw_descriptor(
                patch
            )

        descriptors.append(
            descriptor
        )

        valid_keypoints.append(
            (
                int(x),
                int(y)
            )
        )

    if len(descriptors) == 0:

        return (
            np.empty(
                (
                    0,
                    (2 * radius + 1) ** 2
                ),
                dtype=np.float32
            ),
            np.empty(
                (
                    0,
                    2
                ),
                dtype=np.int32
            )
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


# ======================================================================
# STEP 8 — BUILD RAW AND NORMALIZED DESCRIPTORS
# ======================================================================

raw_desc1, valid_points1_raw = (
    build_descriptors(
        image1,
        keypoints1,
        PATCH_RADIUS,
        normalize=False
    )
)

raw_desc2, valid_points2_raw = (
    build_descriptors(
        image2,
        keypoints2,
        PATCH_RADIUS,
        normalize=False
    )
)


norm_desc1, valid_points1 = (
    build_descriptors(
        image1,
        keypoints1,
        PATCH_RADIUS,
        normalize=True
    )
)

norm_desc2, valid_points2 = (
    build_descriptors(
        image2,
        keypoints2,
        PATCH_RADIUS,
        normalize=True
    )
)


print("\nDESCRIPTORS")
print("-" * 72)

print(
    "Patch size:",
    f"{2 * PATCH_RADIUS + 1} x "
    f"{2 * PATCH_RADIUS + 1}"
)

print(
    "Descriptor dimension:",
    norm_desc1.shape[1]
)

print(
    "Image 1 normalized descriptors:",
    len(norm_desc1)
)

print(
    "Image 2 normalized descriptors:",
    len(norm_desc2)
)


# ======================================================================
# STEP 9 — DISTANCE FUNCTIONS
# ======================================================================

def ssd_distance(
    descriptor_a,
    descriptor_b
):
    """
    Sum of Squared Differences:

        SSD = Σ(a-b)²
    """

    difference = (
        descriptor_a
        -
        descriptor_b
    )

    return float(
        np.sum(
            difference ** 2
        )
    )


def euclidean_distance(
    descriptor_a,
    descriptor_b
):
    """
    Euclidean distance:

        sqrt(Σ(a-b)²)
    """

    return float(
        np.sqrt(
            ssd_distance(
                descriptor_a,
                descriptor_b
            )
        )
    )


def normalized_cross_correlation(
    descriptor_a,
    descriptor_b
):
    """
    Normalized cross correlation.

        NCC =
            Σ(a-mean_a)(b-mean_b)
            ----------------------
            sqrt(
                Σ(a-mean_a)²
                Σ(b-mean_b)²
            )

    Range:

        approximately [-1, +1]
    """

    a = descriptor_a.astype(
        np.float64
    )

    b = descriptor_b.astype(
        np.float64
    )

    a_centered = (
        a - np.mean(a)
    )

    b_centered = (
        b - np.mean(b)
    )

    denominator = (
        np.linalg.norm(a_centered)
        *
        np.linalg.norm(b_centered)
    )

    if denominator < 1e-12:

        return 0.0

    numerator = np.dot(
        a_centered,
        b_centered
    )

    value = (
        numerator
        /
        denominator
    )

    return float(
        np.clip(
            value,
            -1.0,
            1.0
        )
    )


# ======================================================================
# STEP 10 — BRUTE-FORCE NEAREST-NEIGHBOR MATCHING
# ======================================================================

def brute_force_matches(
    descriptors_a,
    descriptors_b,
    metric="ssd"
):
    """
    For every descriptor in A, find the nearest descriptor in B.

    Returns:

        best_indices
        best_distances
        second_distances

    metric options:

        "ssd"
        "euclidean"
        "ncc"

    For NCC, higher similarity is better, so we internally convert
    similarity into:

        distance = 1 - NCC
    """

    if len(descriptors_a) == 0:
        return (
            np.empty(
                (0,),
                dtype=np.int32
            ),
            np.empty(
                (0,),
                dtype=np.float32
            ),
            np.empty(
                (0,),
                dtype=np.float32
            )
        )

    if len(descriptors_b) == 0:
        return (
            np.empty(
                (0,),
                dtype=np.int32
            ),
            np.empty(
                (0,),
                dtype=np.float32
            ),
            np.empty(
                (0,),
                dtype=np.float32
            )
        )

    best_indices = []
    best_distances = []
    second_distances = []

    for descriptor_a in descriptors_a:

        distances = []

        for descriptor_b in descriptors_b:

            if metric == "ssd":

                distance = (
                    ssd_distance(
                        descriptor_a,
                        descriptor_b
                    )
                )

            elif metric == "euclidean":

                distance = (
                    euclidean_distance(
                        descriptor_a,
                        descriptor_b
                    )
                )

            elif metric == "ncc":

                similarity = (
                    normalized_cross_correlation(
                        descriptor_a,
                        descriptor_b
                    )
                )

                distance = (
                    1.0 - similarity
                )

            else:

                raise ValueError(
                    "Unknown metric: "
                    + str(metric)
                )

            distances.append(
                distance
            )

        distances = np.asarray(
            distances,
            dtype=np.float32
        )

        order = np.argsort(
            distances
        )

        best = order[0]

        best_indices.append(
            best
        )

        best_distances.append(
            distances[best]
        )

        if len(order) >= 2:

            second_distances.append(
                distances[
                    order[1]
                ]
            )

        else:

            second_distances.append(
                np.inf
            )

    return (
        np.asarray(
            best_indices,
            dtype=np.int32
        ),
        np.asarray(
            best_distances,
            dtype=np.float32
        ),
        np.asarray(
            second_distances,
            dtype=np.float32
        )
    )


# ======================================================================
# STEP 11 — NORMALIZED DESCRIPTOR MATCHING
# ======================================================================

best_indices, best_distances, second_distances = (
    brute_force_matches(
        norm_desc1,
        norm_desc2,
        metric="ssd"
    )
)


# ======================================================================
# STEP 12 — RATIO TEST
# ======================================================================

RATIO_THRESHOLD = 0.80


def ratio_test(
    best_distances,
    second_distances,
    threshold
):
    """
    Lowe-style ratio idea.

        ratio = best / second_best

    Smaller is better.

    A small ratio means the best match is much better than the
    second-best candidate.
    """

    ratios = (
        best_distances
        /
        np.maximum(
            second_distances,
            1e-12
        )
    )

    accepted = (
        ratios
        <
        threshold
    )

    return (
        ratios,
        accepted
    )


ratios, ratio_accepted = (
    ratio_test(
        best_distances,
        second_distances,
        RATIO_THRESHOLD
    )
)


# ======================================================================
# STEP 13 — CROSS CHECK
# ======================================================================

reverse_best_indices, _, _ = (
    brute_force_matches(
        norm_desc2,
        norm_desc1,
        metric="ssd"
    )
)


cross_checked = np.zeros(
    len(best_indices),
    dtype=bool
)

for i in range(
    len(best_indices)
):

    j = best_indices[i]

    if (
        j >= 0
        and
        j < len(reverse_best_indices)
    ):

        if reverse_best_indices[j] == i:

            cross_checked[i] = True


# Final matches must satisfy both tests.

final_match_mask = (
    ratio_accepted
    &
    cross_checked
)


match_indices_a = np.where(
    final_match_mask
)[0]

match_indices_b = (
    best_indices[
        match_indices_a
    ]
)


print("\nMATCHING")
print("-" * 72)

print(
    "Initial nearest-neighbor matches:",
    len(best_indices)
)

print(
    "Ratio-test matches:",
    int(
        np.sum(
            ratio_accepted
        )
    )
)

print(
    "Cross-checked matches:",
    int(
        np.sum(
            cross_checked
        )
    )
)

print(
    "Final matches:",
    len(match_indices_a)
)


# ======================================================================
# STEP 14 — MATCH DISPLACEMENT
# ======================================================================

if len(match_indices_a) > 0:

    matched_points_a = (
        valid_points1[
            match_indices_a
        ].astype(
            np.float32
        )
    )

    matched_points_b = (
        valid_points2[
            match_indices_b
        ].astype(
            np.float32
        )
    )

    displacements = (
        matched_points_b
        -
        matched_points_a
    )

    estimated_dx = np.median(
        displacements[:, 0]
    )

    estimated_dy = np.median(
        displacements[:, 1]
    )

    displacement_errors = (
        np.linalg.norm(
            displacements
            -
            np.array(
                [
                    SHIFT_X,
                    SHIFT_Y
                ],
                dtype=np.float32
            ),
            axis=1
        )
    )

else:

    matched_points_a = np.empty(
        (0, 2),
        dtype=np.float32
    )

    matched_points_b = np.empty(
        (0, 2),
        dtype=np.float32
    )

    displacements = np.empty(
        (0, 2),
        dtype=np.float32
    )

    estimated_dx = np.nan
    estimated_dy = np.nan

    displacement_errors = np.empty(
        (0,),
        dtype=np.float32
    )


print("\nMOTION ESTIMATION")
print("-" * 72)

print(
    "Ground-truth dx:",
    SHIFT_X
)

print(
    "Ground-truth dy:",
    SHIFT_Y
)

print(
    "Estimated dx:",
    estimated_dx
)

print(
    "Estimated dy:",
    estimated_dy
)


# ======================================================================
# STEP 15 — PRINT MATCH DETAILS
# ======================================================================

print("\nTOP MATCHES")
print("-" * 72)

for i in range(
    min(
        20,
        len(match_indices_a)
    )
):

    idx_a = match_indices_a[i]
    idx_b = match_indices_b[i]

    x1, y1 = valid_points1[
        idx_a
    ]

    x2, y2 = valid_points2[
        idx_b
    ]

    dx = x2 - x1
    dy = y2 - y1

    print(
        f"{i + 1:2d}: "
        f"({x1:3d},{y1:3d}) "
        f"-> "
        f"({x2:3d},{y2:3d}) "
        f"delta=({dx:3d},{dy:3d}) "
        f"ratio={ratios[idx_a]:.3f}"
    )


# ======================================================================
# STEP 16 — VISUALIZE BOTH IMAGES
# ======================================================================

plt.figure(
    figsize=(15, 6)
)

plt.subplot(1, 2, 1)

plt.imshow(
    image1,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.scatter(
    valid_points1[:, 0],
    valid_points1[:, 1],
    s=35,
    facecolors="none",
    edgecolors="red"
)

plt.title(
    f"Image 1 — {len(valid_points1)} keypoints"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    image2,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.scatter(
    valid_points2[:, 0],
    valid_points2[:, 1],
    s=35,
    facecolors="none",
    edgecolors="red"
)

plt.title(
    f"Image 2 — {len(valid_points2)} keypoints"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 17 — VISUALIZE FEATURE MATCHES
# ======================================================================

canvas_height = HEIGHT
canvas_width = WIDTH * 2

canvas = np.zeros(
    (
        canvas_height,
        canvas_width
    ),
    dtype=np.float32
)

canvas[
    :,
    :WIDTH
] = image1

canvas[
    :,
    WIDTH:
] = image2


plt.figure(
    figsize=(16, 8)
)

plt.imshow(
    canvas,
    cmap="gray",
    vmin=0,
    vmax=255
)

for i in range(
    len(match_indices_a)
):

    idx_a = match_indices_a[i]
    idx_b = match_indices_b[i]

    x1, y1 = valid_points1[
        idx_a
    ]

    x2, y2 = valid_points2[
        idx_b
    ]

    x2_canvas = (
        x2
        +
        WIDTH
    )

    plt.plot(
        [
            x1,
            x2_canvas
        ],
        [
            y1,
            y2
        ],
        linewidth=0.8
    )

    plt.scatter(
        [x1, x2_canvas],
        [y1, y2],
        s=20
    )

plt.axvline(
    WIDTH - 0.5,
    linewidth=2
)

plt.title(
    f"Feature Matches — {len(match_indices_a)} final matches"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 18 — DISPLACEMENT VISUALIZATION
# ======================================================================

plt.figure(
    figsize=(10, 7)
)

plt.imshow(
    image1,
    cmap="gray",
    vmin=0,
    vmax=255
)

if len(matched_points_a) > 0:

    plt.quiver(
        matched_points_a[:, 0],
        matched_points_a[:, 1],
        displacements[:, 0],
        displacements[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1
    )

    plt.scatter(
        matched_points_a[:, 0],
        matched_points_a[:, 1],
        s=30
    )

plt.title(
    "Estimated Feature Displacements"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 19 — DESCRIPTOR VISUALIZATION
# ======================================================================

if len(valid_points1) > 0:

    descriptor_index = 0

    x, y = valid_points1[
        descriptor_index
    ]

    patch = extract_patch(
        image1,
        int(x),
        int(y),
        PATCH_RADIUS
    )

    descriptor = norm_desc1[
        descriptor_index
    ]

    plt.figure(
        figsize=(12, 5)
    )

    plt.subplot(1, 2, 1)

    plt.imshow(
        patch,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    plt.scatter(
        [PATCH_RADIUS],
        [PATCH_RADIUS],
        s=80,
        facecolors="none",
        edgecolors="red"
    )

    plt.title(
        f"Local Patch at ({x},{y})"
    )

    plt.axis("off")


    plt.subplot(1, 2, 2)

    plt.plot(
        descriptor
    )

    plt.xlabel(
        "Descriptor Dimension"
    )

    plt.ylabel(
        "Normalized Value"
    )

    plt.title(
        "Normalized Patch Descriptor"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ======================================================================
# STEP 20 — BRIGHTNESS ROBUSTNESS EXPERIMENT
# ======================================================================
#
# Compare a patch with:
#
#     original
#
# and:
#
#     brighter = original * scale + offset
#
# Raw SSD will increase.
#
# Normalized descriptors should remain much more similar.

if len(valid_points1) > 0:

    x, y = valid_points1[0]

    patch_original = extract_patch(
        image1,
        int(x),
        int(y),
        PATCH_RADIUS
    )

    patch_bright = (
        patch_original * 1.5
        +
        30.0
    )

    patch_bright = np.clip(
        patch_bright,
        0,
        255
    ).astype(
        np.float32
    )

    raw_a = raw_descriptor(
        patch_original
    )

    raw_b = raw_descriptor(
        patch_bright
    )

    norm_a = normalize_descriptor(
        patch_original
    )

    norm_b = normalize_descriptor(
        patch_bright
    )

    raw_ssd = ssd_distance(
        raw_a,
        raw_b
    )

    normalized_ssd = ssd_distance(
        norm_a,
        norm_b
    )

    normalized_ncc = (
        normalized_cross_correlation(
            norm_a,
            norm_b
        )
    )

    print("\nBRIGHTNESS ROBUSTNESS")
    print("-" * 72)

    print(
        "Raw SSD:",
        raw_ssd
    )

    print(
        "Normalized SSD:",
        normalized_ssd
    )

    print(
        "Normalized NCC:",
        normalized_ncc
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.subplot(1, 2, 1)

    plt.imshow(
        patch_original,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    plt.title(
        "Original Patch"
    )

    plt.axis("off")


    plt.subplot(1, 2, 2)

    plt.imshow(
        patch_bright,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    plt.title(
        "Brightness-Changed Patch"
    )

    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ======================================================================
# STEP 21 — NUMERICAL TESTS
# ======================================================================

print("\nRUNNING TESTS")
print("-" * 72)


# ----------------------------------------------------------------------
# Test 1 — image shape
# ----------------------------------------------------------------------

assert image1.shape == (
    HEIGHT,
    WIDTH
)

assert image2.shape == (
    HEIGHT,
    WIDTH
)

print(
    "Image shape test passed."
)


# ----------------------------------------------------------------------
# Test 2 — translation preserves image shape
# ----------------------------------------------------------------------

translated_test = translate_image(
    image1,
    SHIFT_X,
    SHIFT_Y
)

assert translated_test.shape == image1.shape

print(
    "Translation shape test passed."
)


# ----------------------------------------------------------------------
# Test 3 — keypoint coordinates are valid
# ----------------------------------------------------------------------

assert np.all(
    keypoints1[:, 0] >= 0
)

assert np.all(
    keypoints1[:, 0] < WIDTH
)

assert np.all(
    keypoints1[:, 1] >= 0
)

assert np.all(
    keypoints1[:, 1] < HEIGHT
)

assert np.all(
    keypoints2[:, 0] >= 0
)

assert np.all(
    keypoints2[:, 0] < WIDTH
)

assert np.all(
    keypoints2[:, 1] >= 0
)

assert np.all(
    keypoints2[:, 1] < HEIGHT
)

print(
    "Keypoint-coordinate bounds test passed."
)


# ----------------------------------------------------------------------
# Test 4 — patch dimensions
# ----------------------------------------------------------------------

if len(valid_points1) > 0:

    x, y = valid_points1[0]

    test_patch = extract_patch(
        image1,
        int(x),
        int(y),
        PATCH_RADIUS
    )

    expected_size = (
        2 * PATCH_RADIUS
        +
        1
    )

    assert test_patch.shape == (
        expected_size,
        expected_size
    )

print(
    "Patch dimension test passed."
)


# ----------------------------------------------------------------------
# Test 5 — descriptor dimension
# ----------------------------------------------------------------------

expected_descriptor_dimension = (
    2 * PATCH_RADIUS
    +
    1
) ** 2

assert (
    norm_desc1.shape[1]
    ==
    expected_descriptor_dimension
)

print(
    "Descriptor dimension test passed."
)


# ----------------------------------------------------------------------
# Test 6 — normalized descriptors have zero mean
# ----------------------------------------------------------------------

if len(norm_desc1) > 0:

    descriptor_means = np.mean(
        norm_desc1,
        axis=1
    )

    assert np.allclose(
        descriptor_means,
        0.0,
        atol=1e-5
    )

print(
    "Descriptor zero-mean test passed."
)


# ----------------------------------------------------------------------
# Test 7 — normalized descriptors have unit standard deviation
# ----------------------------------------------------------------------

if len(norm_desc1) > 0:

    descriptor_stds = np.std(
        norm_desc1,
        axis=1
    )

    assert np.allclose(
        descriptor_stds,
        1.0,
        atol=1e-5
    )

print(
    "Descriptor unit-standard-deviation test passed."
)


# ----------------------------------------------------------------------
# Test 8 — SSD identity
# ----------------------------------------------------------------------

if len(norm_desc1) > 0:

    value = ssd_distance(
        norm_desc1[0],
        norm_desc1[0]
    )

    assert np.isclose(
        value,
        0.0,
        atol=1e-7
    )

print(
    "SSD identity test passed."
)


# ----------------------------------------------------------------------
# Test 9 — Euclidean identity
# ----------------------------------------------------------------------

if len(norm_desc1) > 0:

    value = euclidean_distance(
        norm_desc1[0],
        norm_desc1[0]
    )

    assert np.isclose(
        value,
        0.0,
        atol=1e-7
    )

print(
    "Euclidean identity test passed."
)


# ----------------------------------------------------------------------
# Test 10 — NCC identity
# ----------------------------------------------------------------------

if len(norm_desc1) > 0:

    value = normalized_cross_correlation(
        norm_desc1[0],
        norm_desc1[0]
    )

    assert np.isclose(
        value,
        1.0,
        atol=1e-6
    )

print(
    "NCC identity test passed."
)


# ----------------------------------------------------------------------
# Test 11 — NCC range
# ----------------------------------------------------------------------

if len(norm_desc1) > 0 and len(norm_desc2) > 0:

    value = normalized_cross_correlation(
        norm_desc1[0],
        norm_desc2[0]
    )

    assert (
        value >= -1.000001
    )

    assert (
        value <= 1.000001
    )

print(
    "NCC range test passed."
)


# ----------------------------------------------------------------------
# Test 12 — match indices are valid
# ----------------------------------------------------------------------

if len(best_indices) > 0:

    assert np.all(
        best_indices >= 0
    )

    assert np.all(
        best_indices < len(norm_desc2)
    )

print(
    "Match-index bounds test passed."
)


# ----------------------------------------------------------------------
# Test 13 — ratio values
# ----------------------------------------------------------------------

if len(ratios) > 0:

    finite_ratios = ratios[
        np.isfinite(ratios)
    ]

    assert np.all(
        finite_ratios >= 0
    )

print(
    "Ratio-test values passed."
)


# ----------------------------------------------------------------------
# Test 14 — final match indices
# ----------------------------------------------------------------------

if len(match_indices_a) > 0:

    assert np.all(
        match_indices_a >= 0
    )

    assert np.all(
        match_indices_a
        <
        len(valid_points1)
    )

    assert np.all(
        match_indices_b >= 0
    )

    assert np.all(
        match_indices_b
        <
        len(valid_points2)
    )

print(
    "Final match-index bounds test passed."
)


# ----------------------------------------------------------------------
# Test 15 — known translation
# ----------------------------------------------------------------------
#
# The synthetic image was deliberately translated by:
#
#     SHIFT_X = 18
#     SHIFT_Y = 12
#
# Because feature detection and matching can legitimately reject some
# points, we do not require every match to be perfect.
#
# Instead, if enough final matches exist, require the median estimate
# to be close to the known transformation.

if len(displacements) >= 3:

    assert np.isclose(
        estimated_dx,
        SHIFT_X,
        atol=5.0
    )

    assert np.isclose(
        estimated_dy,
        SHIFT_Y,
        atol=5.0
    )

    print(
        "Known-translation consistency test passed."
    )

else:

    print(
        "Known-translation test skipped: "
        "fewer than 3 robust matches."
    )


# ----------------------------------------------------------------------
# Test 16 — normalized descriptors improve brightness robustness
# ----------------------------------------------------------------------

if len(valid_points1) > 0:

    original_patch = extract_patch(
        image1,
        int(valid_points1[0, 0]),
        int(valid_points1[0, 1]),
        PATCH_RADIUS
    )

    brighter_patch = (
        original_patch * 1.5
        +
        30.0
    )

    brighter_patch = np.clip(
        brighter_patch,
        0,
        255
    ).astype(
        np.float32
    )

    raw_a = raw_descriptor(
        original_patch
    )

    raw_b = raw_descriptor(
        brighter_patch
    )

    normalized_a = normalize_descriptor(
        original_patch
    )

    normalized_b = normalize_descriptor(
        brighter_patch
    )

    raw_distance = ssd_distance(
        raw_a,
        raw_b
    )

    normalized_distance = ssd_distance(
        normalized_a,
        normalized_b
    )

    assert (
        normalized_distance
        <
        raw_distance
    )

    print(
        "Brightness-normalization robustness test passed."
    )


# ----------------------------------------------------------------------
# Test 17 — exact translated pixel
# ----------------------------------------------------------------------
#
# Verify the translation function itself using a simple unique point.

test_image = np.zeros(
    (20, 20),
    dtype=np.float32
)

test_image[
    5,
    7
] = 100.0

test_translated = translate_image(
    test_image,
    3,
    4
)

assert np.isclose(
    test_translated[
        9,
        10
    ],
    100.0
)

print(
    "Translation-coordinate test passed."
)


print("\nALL TESTS PASSED.")


# ======================================================================
# STEP 22 — COMPARE SSD AND NCC
# ======================================================================

if len(norm_desc1) > 0 and len(norm_desc2) > 0:

    ssd_indices, ssd_distances, _ = (
        brute_force_matches(
            norm_desc1,
            norm_desc2,
            metric="ssd"
        )
    )

    ncc_indices, ncc_distances, _ = (
        brute_force_matches(
            norm_desc1,
            norm_desc2,
            metric="ncc"
        )
    )

    print("\nSSD VS NCC")
    print("-" * 72)

    print(
        "Number of descriptors:",
        len(norm_desc1)
    )

    print(
        "SSD best-distance mean:",
        np.mean(
            ssd_distances
        )
    )

    print(
        "NCC converted-distance mean:",
        np.mean(
            ncc_distances
        )
    )


# ======================================================================
# STEP 23 — DESCRIPTOR DISTANCE DISTRIBUTION
# ======================================================================

if len(best_distances) > 0:

    plt.figure(
        figsize=(9, 5)
    )

    plt.hist(
        best_distances,
        bins=20
    )

    plt.xlabel(
        "Best SSD Distance"
    )

    plt.ylabel(
        "Number of Matches"
    )

    plt.title(
        "Nearest-Neighbor Descriptor Distances"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ======================================================================
# STEP 24 — RATIO DISTRIBUTION
# ======================================================================

if len(ratios) > 0:

    finite_ratios = ratios[
        np.isfinite(ratios)
    ]

    if len(finite_ratios) > 0:

        plt.figure(
            figsize=(9, 5)
        )

        plt.hist(
            finite_ratios,
            bins=20
        )

        plt.axvline(
            RATIO_THRESHOLD,
            linewidth=2
        )

        plt.xlabel(
            "Best / Second-Best Distance"
        )

        plt.ylabel(
            "Number of Features"
        )

        plt.title(
            "Ratio-Test Distribution"
        )

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()
        plt.show()


# ======================================================================
# STEP 25 — MATCH DISPLACEMENT HISTOGRAM
# ======================================================================

if len(displacements) > 0:

    plt.figure(
        figsize=(12, 5)
    )

    plt.subplot(1, 2, 1)

    plt.hist(
        displacements[:, 0],
        bins=15
    )

    plt.axvline(
        SHIFT_X,
        linewidth=2
    )

    plt.xlabel(
        "dx"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Horizontal Displacement"
    )


    plt.subplot(1, 2, 2)

    plt.hist(
        displacements[:, 1],
        bins=15
    )

    plt.axvline(
        SHIFT_Y,
        linewidth=2
    )

    plt.xlabel(
        "dy"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Vertical Displacement"
    )

    plt.tight_layout()
    plt.show()


# ======================================================================
# STEP 26 — ROBOTICS INTERPRETATION
# ======================================================================

print(
    """
======================================================================
ROBOTICS INTERPRETATION
======================================================================

The computer-vision pipeline has now grown:

    CAMERA
       ↓
    IMAGE
       ↓
    GRADIENT
       ↓
    HARRIS KEYPOINTS
       ↓
    LOCAL DESCRIPTORS
       ↓
    FEATURE MATCHING
       ↓
    CORRESPONDENCES
       ↓
    DISPLACEMENT
       ↓
    MOTION ESTIMATION
       ↓
    ROBOT VISION

For OpenArm V2 waste sorting, imagine:

    FRAME t
       ↓
    detect visual features

    FRAME t+1
       ↓
    detect visual features

    match features
       ↓
    estimate displacement
       ↓
    estimate object/camera motion
       ↓
    update robot perception

This is one of the conceptual foundations of:

    - visual odometry
    - SLAM
    - object tracking
    - camera motion estimation
    - image registration

The important idea is:

    KEYPOINT
        tells us WHERE

    DESCRIPTOR
        tells us WHAT THE LOCAL REGION LOOKS LIKE

    MATCH
        tells us WHICH POINT CORRESPONDS TO WHICH

    DISPLACEMENT
        tells us HOW THE IMAGE STRUCTURE MOVED
"""
)


# ======================================================================
# STEP 27 — FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 72)
print("LESSON 10 COMPLETE")
print("=" * 72)

print(
    """
You learned:

1. A keypoint provides a local image location.

2. A descriptor provides a numerical representation of its local
   appearance.

3. A raw image patch can be used as a simple descriptor.

4. Descriptor normalization can reduce brightness sensitivity.

5. SSD measures squared descriptor difference.

6. Euclidean distance measures geometric descriptor distance.

7. NCC measures normalized similarity.

8. Brute-force matching compares every descriptor against every
   candidate descriptor.

9. The ratio test rejects ambiguous matches.

10. Cross-checking keeps mutually consistent matches.

11. Corresponding feature coordinates can estimate image motion.

12. The median displacement provides a robust translation estimate.

The complete pipeline is:

    IMAGE
      ↓
    HARRIS KEYPOINTS
      ↓
    PATCH DESCRIPTORS
      ↓
    DESCRIPTOR DISTANCE
      ↓
    NEAREST NEIGHBOR
      ↓
    RATIO TEST
      ↓
    CROSS CHECK
      ↓
    FEATURE CORRESPONDENCES
      ↓
    MOTION ESTIMATION

For robotics:

    CAMERA
      ↓
    KEYPOINTS
      ↓
    DESCRIPTORS
      ↓
    MATCHING
      ↓
    MOTION
      ↓
    VISUAL ODOMETRY / TRACKING / LOCALIZATION

IMPORTANT:

Raw patch descriptors are intentionally simple.

They are NOT robust to:

    - large rotation
    - large scale changes
    - severe viewpoint changes
    - strong deformation
    - occlusion

That limitation motivates more advanced feature descriptors.

NEXT:

LESSON 11 — CONTOURS AND CONNECTED COMPONENTS

We will move from individual local features toward object regions:

    BINARY MASK
        ↓
    CONNECTED COMPONENTS
        ↓
    CONTOURS
        ↓
    BOUNDING BOXES
        ↓
    CENTROIDS
        ↓
    OBJECT REGIONS

This will be particularly useful for classical robot perception
and waste-object localization.
"""
)