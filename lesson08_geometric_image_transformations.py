"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 08 — GEOMETRIC IMAGE TRANSFORMATIONS
======================================================================

COURSE RULE
-----------
This course is built from scratch.

NO:
    - pretrained models
    - pretrained weights
    - black-box geometric transformation functions

We implement the important geometric operations using:

    - NumPy
    - Matplotlib

OpenCV is intentionally NOT used for the main transformation
implementation.

----------------------------------------------------------------------
WHAT YOU LEARN
----------------------------------------------------------------------

STEP 1 — WHAT IS A GEOMETRIC TRANSFORMATION?

A geometric transformation changes WHERE pixels are located.

Unlike filtering, which changes pixel values, geometric
transformations modify image coordinates.

Examples:

    translation
    rotation
    scaling
    shearing
    affine transformation
    perspective transformation

General idea:

    input coordinate
          ↓
    mathematical transformation
          ↓
    output coordinate

----------------------------------------------------------------------
STEP 2 — IMAGE COORDINATES
----------------------------------------------------------------------

An image uses pixel coordinates:

    (x, y)

where:

    x = horizontal position
    y = vertical position

For an image:

    height = H
    width  = W

valid coordinates are approximately:

    0 <= x < W
    0 <= y < H

and:

    0 <= y < H

IMPORTANT:

NumPy indexing is:

    image[y, x]

not:

    image[x, y]

Therefore:

    coordinate = (x, y)

but:

    array access = image[y, x]

This distinction is extremely important in computer vision.

----------------------------------------------------------------------
STEP 3 — TRANSLATION
----------------------------------------------------------------------

Translation moves an image.

    x' = x + tx
    y' = y + ty

Matrix form using homogeneous coordinates:

    [x']   [1  0  tx] [x]
    [y'] = [0  1  ty] [y]
    [1 ]   [0  0  1 ] [1]

Example:

    tx = 50
    ty = 20

moves the image:

    50 pixels right
    20 pixels down

----------------------------------------------------------------------
STEP 4 — SCALING
----------------------------------------------------------------------

Scaling changes size.

    x' = sx x
    y' = sy y

Matrix:

    [x']   [sx  0   0] [x]
    [y'] = [0   sy  0] [y]
    [1 ]   [0   0   1] [1]

If:

    sx = sy = 2

the image becomes twice as large.

----------------------------------------------------------------------
STEP 5 — ROTATION
----------------------------------------------------------------------

Rotation around the origin:

    x' = cos(theta)x - sin(theta)y

    y' = sin(theta)x + cos(theta)y

Matrix:

    [x']   [ cosθ  -sinθ ] [x]
    [y'] = [ sinθ   cosθ ] [y]

However, rotating around the image origin usually moves the object
outside the image.

Therefore practical image rotation is usually performed around
the image center.

----------------------------------------------------------------------
STEP 6 — ROTATION AROUND IMAGE CENTER
----------------------------------------------------------------------

Let:

    cx = image center x
    cy = image center y

First translate:

    (x, y)
       ↓
    (x-cx, y-cy)

Then rotate:

    xr = cosθ(x-cx) - sinθ(y-cy)

    yr = sinθ(x-cx) + cosθ(y-cy)

Then translate back:

    x' = xr + cx

    y' = yr + cy

This is one of the most important coordinate transformations
in computer vision.

----------------------------------------------------------------------
STEP 7 — SHEARING
----------------------------------------------------------------------

Horizontal shear:

    x' = x + kx y
    y' = y

Vertical shear:

    x' = x
    y' = y + ky x

Shearing changes the angle of structures without simply rotating
them.

----------------------------------------------------------------------
STEP 8 — AFFINE TRANSFORMATION
----------------------------------------------------------------------

An affine transformation combines:

    translation
    rotation
    scaling
    shearing

General form:

    x' = a x + b y + tx

    y' = c x + d y + ty

Matrix:

    [x']   [a b tx] [x]
    [y'] = [c d ty] [y]
    [1 ]   [0 0  1] [1]

Important property:

    straight lines remain straight

and:

    parallel lines remain parallel

----------------------------------------------------------------------
STEP 9 — HOMOGENEOUS COORDINATES
----------------------------------------------------------------------

Normal coordinates:

    (x, y)

are extended to:

    (x, y, 1)

This allows translation to be represented using matrix
multiplication.

This is fundamental in:

    computer vision
    robotics
    camera geometry
    3-D reconstruction
    SLAM
    robot calibration

----------------------------------------------------------------------
STEP 10 — FORWARD VS INVERSE MAPPING
----------------------------------------------------------------------

There are two approaches.

FORWARD MAPPING:

    source pixel
        ↓
    calculate destination
        ↓
    write pixel

Problem:

    - holes can appear
    - multiple source pixels may map to one destination

INVERSE MAPPING:

    destination pixel
        ↓
    calculate source coordinate
        ↓
    sample source image

This is generally more convenient for image warping.

We will use inverse mapping.

----------------------------------------------------------------------
STEP 11 — NEAREST-NEIGHBOR INTERPOLATION
----------------------------------------------------------------------

Suppose the source coordinate is:

    x = 10.7
    y = 20.2

Nearest-neighbor chooses:

    x = 11
    y = 20

Advantages:

    - very simple
    - fast
    - preserves exact pixel values

Disadvantages:

    - blocky results
    - jagged edges

----------------------------------------------------------------------
STEP 12 — BILINEAR INTERPOLATION
----------------------------------------------------------------------

Suppose:

    x = 10.3
    y = 20.7

The point lies between four pixels:

    Q11 ----- Q21
      |         |
      |    P    |
      |         |
    Q12 ----- Q22

Bilinear interpolation performs two 1-D interpolations and then
one interpolation between those results.

Mathematically:

    I(x,y)
      =
    (1-a)(1-b)Q11
    + a(1-b)Q21
    + (1-a)b Q12
    + ab Q22

where:

    a = fractional x
    b = fractional y

Bilinear interpolation is smoother than nearest neighbor.

----------------------------------------------------------------------
STEP 13 — PERSPECTIVE TRANSFORMATION
----------------------------------------------------------------------

An affine transformation cannot model perspective convergence.

For example:

    railroad tracks
    road lanes
    rectangular objects viewed at an angle

can appear to converge.

Perspective transformation uses a 3×3 homography matrix:

    [x']   [h11 h12 h13] [x]
    [y'] = [h21 h22 h23] [y]
    [w ]   [h31 h32 h33] [1]

Then:

    x_out = x' / w
    y_out = y' / w

This division is the key difference from affine transformation.

----------------------------------------------------------------------
STEP 14 — HOMOGRAPHY
----------------------------------------------------------------------

A homography maps one planar image surface to another.

It is fundamental for:

    - document scanning
    - bird's-eye views
    - camera calibration
    - planar object tracking
    - image stitching
    - robot workspace mapping

Later in this course we will derive homography estimation from
corresponding points.

----------------------------------------------------------------------
STEP 15 — REAL-WORLD OPENARM EXAMPLE
----------------------------------------------------------------------

Imagine an OpenArm V2 waste-sorting robot.

The camera sees the conveyor at an angle.

The robot works in a physical workspace.

Image coordinates:

    pixel (u, v)

must eventually be related to:

    robot workspace coordinates
    (X, Y, Z)

Before 3-D geometry, we can use planar transformations.

For example:

    camera image
         ↓
    perspective correction
         ↓
    top-down conveyor image
         ↓
    object location
         ↓
    robot coordinate mapping
         ↓
    grasp planning

This is one reason geometric computer vision is extremely important
for robotics.

----------------------------------------------------------------------
STEP 16 — TESTING
----------------------------------------------------------------------

We test:

    - identity transformation
    - translation
    - scaling
    - rotation
    - affine transformation
    - interpolation
    - coordinate bounds
    - homogeneous-coordinate consistency

----------------------------------------------------------------------
EXERCISES
----------------------------------------------------------------------

1. Change the translation.

2. Rotate by:

       15°
       45°
       90°
       180°

3. Compare nearest-neighbor and bilinear interpolation.

4. Change the scaling factors.

5. Create horizontal and vertical shear.

6. Create your own affine matrix.

7. Modify the perspective transformation.

8. Try to transform a synthetic conveyor image into a
   bird's-eye-view representation.

----------------------------------------------------------------------
NEXT LESSON
----------------------------------------------------------------------

LESSON 09 — IMAGE FEATURES AND CORNERS

We will build:

    - image gradients
    - corner intuition
    - Harris corner detector
    - structure tensor
    - eigenvalue interpretation
    - feature visualization

======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# STEP 1 — CREATE A SYNTHETIC TEST IMAGE
# ======================================================================

HEIGHT = 300
WIDTH = 420

image = np.zeros(
    (HEIGHT, WIDTH),
    dtype=np.float32
)

# Background
image[:] = 30.0

# Large rectangle
image[
    60:190,
    80:250
] = 210.0

# Circle
yy, xx = np.ogrid[
    :HEIGHT,
    :WIDTH
]

circle = (
    (xx - 320) ** 2
    +
    (yy - 150) ** 2
    <= 45 ** 2
)

image[circle] = 180.0

# Diagonal object
for y in range(30, 140):

    x = int(
        280 + 0.45 * y
    )

    if 0 <= x < WIDTH:

        image[
            y,
            x
        ] = 240.0


# ======================================================================
# STEP 2 — HOMOGENEOUS COORDINATE UTILITIES
# ======================================================================

def make_homogeneous_points(
    x,
    y
):
    """
    Convert x,y coordinates into homogeneous coordinates.

        (x, y)
            ↓
        (x, y, 1)
    """

    return np.array(
        [
            x,
            y,
            1.0
        ],
        dtype=np.float64
    )


def apply_homography(
    H,
    x,
    y
):
    """
    Apply a 3×3 homography to one coordinate.

    H
    ·
    [x, y, 1]^T

    followed by homogeneous normalization.
    """

    point = make_homogeneous_points(
        x,
        y
    )

    transformed = H @ point

    w = transformed[2]

    if abs(w) < 1e-12:
        raise ValueError(
            "Homogeneous coordinate w is too close to zero."
        )

    return (
        transformed[0] / w,
        transformed[1] / w
    )


# ======================================================================
# STEP 3 — TRANSLATION MATRIX
# ======================================================================

def translation_matrix(
    tx,
    ty
):
    """
    Create homogeneous translation matrix.

        [1 0 tx]
        [0 1 ty]
        [0 0  1]
    """

    return np.array(
        [
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0]
        ],
        dtype=np.float64
    )


# ======================================================================
# STEP 4 — SCALING MATRIX
# ======================================================================

def scaling_matrix(
    sx,
    sy
):
    """
    Create homogeneous scaling matrix.
    """

    return np.array(
        [
            [sx, 0.0, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, 1.0]
        ],
        dtype=np.float64
    )


# ======================================================================
# STEP 5 — ROTATION MATRIX
# ======================================================================

def rotation_matrix(
    angle_degrees
):
    """
    Create a rotation matrix around the origin.
    """

    theta = np.deg2rad(
        angle_degrees
    )

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ],
        dtype=np.float64
    )


# ======================================================================
# STEP 6 — SHEAR MATRIX
# ======================================================================

def shear_matrix(
    shear_x=0.0,
    shear_y=0.0
):
    """
    Create an affine shear matrix.

    x' = x + shear_x * y

    y' = y + shear_y * x
    """

    return np.array(
        [
            [1.0, shear_x, 0.0],
            [shear_y, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        dtype=np.float64
    )


# ======================================================================
# STEP 7 — CENTERED ROTATION
# ======================================================================

def centered_rotation_matrix(
    angle_degrees,
    center_x,
    center_y
):
    """
    Rotate around an arbitrary center.

    Operation:

        translate center to origin
        rotate
        translate back
    """

    to_origin = translation_matrix(
        -center_x,
        -center_y
    )

    rotation = rotation_matrix(
        angle_degrees
    )

    back = translation_matrix(
        center_x,
        center_y
    )

    return (
        back
        @ rotation
        @ to_origin
    )


# ======================================================================
# STEP 8 — AFFINE MATRIX
# ======================================================================

def affine_matrix(
    a,
    b,
    c,
    d,
    tx,
    ty
):
    """
    General affine transformation:

        x' = a*x + b*y + tx
        y' = c*x + d*y + ty
    """

    return np.array(
        [
            [a, b, tx],
            [c, d, ty],
            [0.0, 0.0, 1.0]
        ],
        dtype=np.float64
    )


# ======================================================================
# STEP 9 — IMAGE SAMPLING
# ======================================================================

def nearest_neighbor_sample(
    image,
    x,
    y,
    fill_value=0.0
):
    """
    Sample an image using nearest-neighbor interpolation.
    """

    height, width = image.shape

    xi = int(
        np.round(x)
    )

    yi = int(
        np.round(y)
    )

    if (
        xi < 0
        or xi >= width
        or yi < 0
        or yi >= height
    ):
        return fill_value

    return float(
        image[yi, xi]
    )


def bilinear_sample(
    image,
    x,
    y,
    fill_value=0.0
):
    """
    Bilinear interpolation from scratch.

    Four neighboring pixels:

        Q11 ----- Q21
          |         |
          |    P    |
          |         |
        Q12 ----- Q22
    """

    height, width = image.shape

    x0 = int(
        np.floor(x)
    )

    y0 = int(
        np.floor(y)
    )

    x1 = x0 + 1
    y1 = y0 + 1

    # If the required neighborhood is outside the image,
    # return the fill value.
    if (
        x0 < 0
        or y0 < 0
        or x1 >= width
        or y1 >= height
    ):
        return fill_value

    dx = x - x0
    dy = y - y0

    q11 = image[
        y0,
        x0
    ]

    q21 = image[
        y0,
        x1
    ]

    q12 = image[
        y1,
        x0
    ]

    q22 = image[
        y1,
        x1
    ]

    value = (
        (1.0 - dx)
        * (1.0 - dy)
        * q11

        +

        dx
        * (1.0 - dy)
        * q21

        +

        (1.0 - dx)
        * dy
        * q12

        +

        dx
        * dy
        * q22
    )

    return float(
        value
    )


# ======================================================================
# STEP 10 — GENERIC IMAGE WARP
# ======================================================================

def warp_image(
    image,
    H,
    interpolation="bilinear",
    fill_value=0.0
):
    """
    Warp an image using inverse mapping.

    For every output pixel:

        output coordinate
              ↓
        inverse transformation
              ↓
        source coordinate
              ↓
        interpolation
              ↓
        output pixel

    H is assumed to map source → destination.

    Therefore we use:

        H_inverse

    for destination → source mapping.
    """

    height, width = image.shape

    H_inverse = np.linalg.inv(
        H
    )

    output = np.full(
        image.shape,
        fill_value,
        dtype=np.float32
    )

    for y_out in range(height):

        for x_out in range(width):

            destination_point = np.array(
                [
                    x_out,
                    y_out,
                    1.0
                ],
                dtype=np.float64
            )

            source_h = (
                H_inverse
                @ destination_point
            )

            w = source_h[2]

            if abs(w) < 1e-12:
                continue

            x_src = (
                source_h[0]
                / w
            )

            y_src = (
                source_h[1]
                / w
            )

            if interpolation == "nearest":

                output[
                    y_out,
                    x_out
                ] = nearest_neighbor_sample(
                    image,
                    x_src,
                    y_src,
                    fill_value
                )

            elif interpolation == "bilinear":

                output[
                    y_out,
                    x_out
                ] = bilinear_sample(
                    image,
                    x_src,
                    y_src,
                    fill_value
                )

            else:

                raise ValueError(
                    "Unknown interpolation method: "
                    f"{interpolation}"
                )

    return output


# ======================================================================
# STEP 11 — CREATE TRANSFORMATIONS
# ======================================================================

translation = translation_matrix(
    tx=50,
    ty=30
)

scaling = scaling_matrix(
    sx=1.20,
    sy=0.80
)

rotation = centered_rotation_matrix(
    angle_degrees=25,
    center_x=WIDTH / 2,
    center_y=HEIGHT / 2
)

shear = shear_matrix(
    shear_x=0.25,
    shear_y=0.0
)

affine = affine_matrix(
    a=1.05,
    b=0.20,
    c=-0.10,
    d=0.90,
    tx=30,
    ty=20
)


# ======================================================================
# STEP 12 — APPLY TRANSFORMATIONS
# ======================================================================

translated = warp_image(
    image,
    translation,
    interpolation="bilinear"
)

scaled = warp_image(
    image,
    scaling,
    interpolation="bilinear"
)

rotated = warp_image(
    image,
    rotation,
    interpolation="bilinear"
)

sheared = warp_image(
    image,
    shear,
    interpolation="bilinear"
)

affine_result = warp_image(
    image,
    affine,
    interpolation="bilinear"
)


# ======================================================================
# STEP 13 — DISPLAY BASIC TRANSFORMATIONS
# ======================================================================

print("=" * 72)
print("LESSON 08 — GEOMETRIC IMAGE TRANSFORMATIONS")
print("=" * 72)

print("\nIMAGE")
print("-" * 72)

print(
    "Shape:",
    image.shape
)

print(
    "dtype:",
    image.dtype
)

print(
    "min:",
    image.min()
)

print(
    "max:",
    image.max()
)


plt.figure(
    figsize=(16, 10)
)

plt.subplot(2, 3, 1)

plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Original"
)

plt.axis("off")


plt.subplot(2, 3, 2)

plt.imshow(
    translated,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Translation"
)

plt.axis("off")


plt.subplot(2, 3, 3)

plt.imshow(
    scaled,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Scaling"
)

plt.axis("off")


plt.subplot(2, 3, 4)

plt.imshow(
    rotated,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Rotation"
)

plt.axis("off")


plt.subplot(2, 3, 5)

plt.imshow(
    sheared,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Shearing"
)

plt.axis("off")


plt.subplot(2, 3, 6)

plt.imshow(
    affine_result,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Affine Transformation"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 14 — NEAREST VS BILINEAR
# ======================================================================

nearest_rotated = warp_image(
    image,
    rotation,
    interpolation="nearest"
)

bilinear_rotated = warp_image(
    image,
    rotation,
    interpolation="bilinear"
)


plt.figure(
    figsize=(12, 5)
)

plt.subplot(1, 2, 1)

plt.imshow(
    nearest_rotated,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Nearest Neighbor"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    bilinear_rotated,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Bilinear Interpolation"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 15 — SHOW COORDINATE TRANSFORMATION
# ======================================================================

print("\nCOORDINATE TRANSFORMATION EXAMPLES")
print("-" * 72)

test_points = [
    (0, 0),
    (100, 50),
    (200, 100),
    (300, 200),
]

for x, y in test_points:

    tx, ty = apply_homography(
        translation,
        x,
        y
    )

    print(
        f"Translation: "
        f"({x:6.1f}, {y:6.1f})"
        f" -> "
        f"({tx:6.1f}, {ty:6.1f})"
    )


# ======================================================================
# STEP 16 — PERSPECTIVE / HOMOGRAPHY
# ======================================================================
#
# Construct a synthetic perspective transformation.
#
# This matrix is intentionally simple.
#
#     x' = ...
#     y' = ...
#     w  = ...
#
# Then:
#
#     x_out = x'/w
#     y_out = y'/w

perspective = np.array(
    [
        [1.0, 0.15, 0.0],
        [0.05, 1.0, 0.0],
        [0.0012, 0.0005, 1.0]
    ],
    dtype=np.float64
)

perspective_result = warp_image(
    image,
    perspective,
    interpolation="bilinear"
)


# ======================================================================
# STEP 17 — DISPLAY PERSPECTIVE TRANSFORMATION
# ======================================================================

plt.figure(
    figsize=(10, 5)
)

plt.subplot(1, 2, 1)

plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Original"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    perspective_result,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Perspective / Homography"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 18 — VISUALIZE IMAGE COORDINATES
# ======================================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)

points_x = [
    0,
    WIDTH - 1,
    0,
    WIDTH - 1,
    WIDTH // 2
]

points_y = [
    0,
    0,
    HEIGHT - 1,
    HEIGHT - 1,
    HEIGHT // 2
]

plt.scatter(
    points_x,
    points_y,
    s=80
)

for x, y in zip(
    points_x,
    points_y
):

    plt.text(
        x + 5,
        y + 5,
        f"({x},{y})"
    )

plt.title(
    "Image Coordinate System"
)

plt.xlabel(
    "x"
)

plt.ylabel(
    "y"

)

plt.show()


# ======================================================================
# STEP 19 — NUMERICAL TESTS
# ======================================================================

print("\nRUNNING TESTS")
print("-" * 72)


# ----------------------------------------------------------------------
# Test 1 — identity transformation
# ----------------------------------------------------------------------

identity = np.eye(
    3,
    dtype=np.float64
)

identity_result = warp_image(
    image,
    identity,
    interpolation="nearest"
)

identity_error = np.max(
    np.abs(
        identity_result
        -
        image
    )
)

print(
    "Identity max error:",
    identity_error
)

assert identity_error == 0.0

print(
    "Identity transformation test passed."
)


# ----------------------------------------------------------------------
# Test 2 — homogeneous coordinates
# ----------------------------------------------------------------------

point = make_homogeneous_points(
    10,
    20
)

assert np.allclose(
    point,
    np.array(
        [10.0, 20.0, 1.0]
    )
)

print(
    "Homogeneous-coordinate test passed."
)


# ----------------------------------------------------------------------
# Test 3 — translation
# ----------------------------------------------------------------------

T = translation_matrix(
    50,
    30
)

x2, y2 = apply_homography(
    T,
    10,
    20
)

assert np.allclose(
    [x2, y2],
    [60, 50]
)

print(
    "Translation test passed."
)


# ----------------------------------------------------------------------
# Test 4 — scaling
# ----------------------------------------------------------------------

S = scaling_matrix(
    2.0,
    3.0
)

x2, y2 = apply_homography(
    S,
    10,
    20
)

assert np.allclose(
    [x2, y2],
    [20, 60]
)

print(
    "Scaling test passed."
)


# ----------------------------------------------------------------------
# Test 5 — rotation by 90 degrees
# ----------------------------------------------------------------------

R90 = rotation_matrix(
    90
)

x2, y2 = apply_homography(
    R90,
    1,
    0
)

assert np.allclose(
    [x2, y2],
    [0, 1],
    atol=1e-10
)

print(
    "90-degree rotation test passed."
)


# ----------------------------------------------------------------------
# Test 6 — rotation by 180 degrees
# ----------------------------------------------------------------------

R180 = rotation_matrix(
    180
)

x2, y2 = apply_homography(
    R180,
    1,
    0
)

assert np.allclose(
    [x2, y2],
    [-1, 0],
    atol=1e-10
)

print(
    "180-degree rotation test passed."
)


# ----------------------------------------------------------------------
# Test 7 — affine transformation
# ----------------------------------------------------------------------

A = affine_matrix(
    2,
    0,
    0,
    3,
    10,
    20
)

x2, y2 = apply_homography(
    A,
    5,
    7
)

assert np.allclose(
    [x2, y2],
    [20, 41]
)

print(
    "Affine transformation test passed."
)


# ----------------------------------------------------------------------
# Test 8 — nearest-neighbor interpolation
# ----------------------------------------------------------------------

small_image = np.array(
    [
        [0, 10],
        [20, 30]
    ],
    dtype=np.float32
)

value = nearest_neighbor_sample(
    small_image,
    0.2,
    0.1
)

assert value == 0.0

print(
    "Nearest-neighbor test passed."
)


# ----------------------------------------------------------------------
# Test 9 — bilinear interpolation
# ----------------------------------------------------------------------
#
# At the center of:
#
#     0   10
#     20  30
#
# expected value:
#
#     (0 + 10 + 20 + 30) / 4
#       =
#     15

value = bilinear_sample(
    small_image,
    0.5,
    0.5
)

assert np.isclose(
    value,
    15.0
)

print(
    "Bilinear interpolation test passed."
)


# ----------------------------------------------------------------------
# Test 10 — inverse consistency
# ----------------------------------------------------------------------

random_point = np.array(
    [
        100.0,
        80.0,
        1.0
    ]
)

transformed_point = (
    translation
    @ random_point
)

recovered_point = (
    np.linalg.inv(
        translation
    )
    @ transformed_point
)

recovered_point /= recovered_point[2]

assert np.allclose(
    recovered_point,
    random_point
)

print(
    "Forward/inverse transformation test passed."
)


# ----------------------------------------------------------------------
# Test 11 — perspective identity
# ----------------------------------------------------------------------

identity_perspective = np.eye(
    3,
    dtype=np.float64
)

x2, y2 = apply_homography(
    identity_perspective,
    123,
    77
)

assert np.allclose(
    [x2, y2],
    [123, 77]
)

print(
    "Homography normalization test passed."
)


# ----------------------------------------------------------------------
# Test 12 — output dimensions
# ----------------------------------------------------------------------

assert translated.shape == image.shape
assert scaled.shape == image.shape
assert rotated.shape == image.shape
assert sheared.shape == image.shape
assert affine_result.shape == image.shape
assert perspective_result.shape == image.shape

print(
    "Output-shape tests passed."
)


print("\nALL TESTS PASSED.")


# ======================================================================
# STEP 20 — COMBINE TRANSFORMATIONS
# ======================================================================
#
# Matrix multiplication allows multiple transformations to be
# combined into one transformation.
#
# IMPORTANT:
#
# Matrix multiplication order matters.
#
# In general:
#
#     A @ B != B @ A
#
# Therefore:
#
#     rotate → translate
#
# is different from:
#
#     translate → rotate

combined = (
    translation
    @
    rotation
    @
    scaling
)

combined_result = warp_image(
    image,
    combined,
    interpolation="bilinear"
)


# ======================================================================
# STEP 21 — VISUALIZE COMBINED TRANSFORMATION
# ======================================================================

plt.figure(
    figsize=(10, 5)
)

plt.subplot(1, 2, 1)

plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Original"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    combined_result,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Combined Transformation"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 22 — TRANSFORMATION PIPELINE VISUALIZATION
# ======================================================================

plt.figure(
    figsize=(16, 8)
)

plt.subplot(2, 3, 1)

plt.imshow(
    image,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Original"
)

plt.axis("off")


plt.subplot(2, 3, 2)

plt.imshow(
    translated,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Translation"
)

plt.axis("off")


plt.subplot(2, 3, 3)

plt.imshow(
    rotated,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Rotation"
)

plt.axis("off")


plt.subplot(2, 3, 4)

plt.imshow(
    scaled,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Scaling"
)

plt.axis("off")


plt.subplot(2, 3, 5)

plt.imshow(
    affine_result,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Affine"
)

plt.axis("off")


plt.subplot(2, 3, 6)

plt.imshow(
    perspective_result,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.title(
    "Perspective"
)

plt.axis("off")

plt.tight_layout()
plt.show()


# ======================================================================
# STEP 23 — ROBOTICS INTERPRETATION
# ======================================================================
#
# One of the most important applications is converting image
# coordinates into workspace coordinates.
#
# Simplified planar example:
#
#     camera image
#
#          (u,v)
#             |
#             | homography
#             v
#
#     conveyor plane
#
#          (X,Y)
#
# Then:
#
#     object detector
#           ↓
#     object pixel center
#           ↓
#     geometric transformation
#           ↓
#     conveyor coordinate
#           ↓
#     robot coordinate
#
# For OpenArm V2:
#
#     camera
#       ↓
#     segmentation
#       ↓
#     object center
#       ↓
#     perspective correction
#       ↓
#     workspace position
#       ↓
#     grasp planning
#       ↓
#     ACT / robot controller
#
# This is one of the bridges between classical computer vision
# and robotics.


# ======================================================================
# STEP 24 — FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 72)
print("LESSON 08 COMPLETE")
print("=" * 72)

print(
    """
You learned:

1. Image coordinates use (x, y).
2. NumPy images use image[y, x].
3. Translation moves coordinates.
4. Scaling changes spatial size.
5. Rotation changes orientation.
6. Shearing changes geometric angles.
7. Affine transformations combine these operations.
8. Homogeneous coordinates allow translation to be represented
   by matrix multiplication.
9. Inverse mapping is useful for image warping.
10. Nearest-neighbor interpolation is simple but blocky.
11. Bilinear interpolation produces smoother values.
12. Perspective transformation requires homogeneous normalization.
13. Homography is fundamental for planar geometric mapping.
14. Transformation order matters.

The geometric vision pipeline is now:

    IMAGE
      ↓
    COORDINATES
      ↓
    TRANSFORMATION
      ↓
    WARPING
      ↓
    INTERPOLATION
      ↓
    TRANSFORMED IMAGE

For robotics:

    CAMERA PIXEL
        ↓
    HOMOGRAPHY
        ↓
    WORKSPACE COORDINATE
        ↓
    ROBOT MOTION

NEXT:
LESSON 09 — IMAGE FEATURES AND CORNERS

We will implement:

    image gradients
    structure tensor
    Harris corner detector
    eigenvalue interpretation
    corner visualization

All from scratch.
"""
)