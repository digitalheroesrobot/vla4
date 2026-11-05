"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 15 — PYRAMIDAL OPTICAL FLOW
======================================================================

IMPLEMENTATION RULES
--------------------
- No pretrained models or weights.
- No OpenCV optical-flow functions.
- Implement image pyramids, warping, interpolation, and
  Lucas-Kanade estimation using NumPy.
- Matplotlib is used only for visualization.

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1 — Small-motion limitation of Lucas-Kanade.
STEP 2 — Gaussian filtering.
STEP 3 — Gaussian image pyramids.
STEP 4 — Bilinear interpolation.
STEP 5 — Flow-field upsampling.
STEP 6 — Image warping.
STEP 7 — Iterative Lucas-Kanade.
STEP 8 — Coarse-to-fine optical flow.
STEP 9 — Motion visualization.
STEP 10 — Numerical verification.

======================================================================
MATHEMATICAL FOUNDATION
======================================================================

Brightness constancy:

    I(x,y,t) = I(x+u,y+v,t+1)

First-order Taylor approximation:

    Ix*u + Iy*v + It = 0

Lucas-Kanade local least squares:

    A * [u,v].T = b

    [ Ix1  Iy1 ] [u]   [ -It1 ]
    [ Ix2  Iy2 ] [v] = [ -It2 ]
       ...                  ...

Solution:

    flow = argmin ||A*flow - b||^2

Pyramidal estimation:

    1. Construct progressively smaller images.
    2. Estimate flow at the coarsest level.
    3. Upsample flow to the next level.
    4. Scale displacement values to the new resolution.
    5. Warp the second image.
    6. Estimate residual flow.
    7. Refine until reaching full resolution.

IMPORTANT FLOW SCALING:

    If image resolution doubles:

        u_new = 2 * u_old
        v_new = 2 * v_old

    The spatial flow field must also be interpolated.

======================================================================
IMPORTANT IMPLEMENTATION DETAIL — BOUNDARY INTERPOLATION
======================================================================

Bilinear interpolation must handle the last image row and column.

For a coordinate exactly on the final pixel:

    x0 = width - 1
    x1 = min(x0 + 1, width - 1)

The same applies to y.

This prevents valid boundary pixels from incorrectly becoming zero.

======================================================================
REAL-WORLD ROBOTICS APPLICATION
======================================================================

OPENARM V2 — WASTE SORTING

A camera observes a waste object moving on a conveyor.

Frame 1:
    Object center = (100, 120)

Frame 2:
    Object center = (125, 130)

Displacement:
    (25, 10) pixels

Pyramidal optical flow estimates larger pixel displacement
using coarse-to-fine refinement.

Potential applications:

    - Conveyor motion estimation.
    - Moving-object tracking.
    - Camera-motion estimation.
    - Visual tracking initialization.
    - Motion-aware grasp timing.

Optical flow measures apparent image motion.
It does not directly measure physical 3-D velocity.

======================================================================
EXERCISES
======================================================================

1. Change translation to (30, 15).
2. Increase pyramid levels.
3. Change Lucas-Kanade window size.
4. Add Gaussian noise.
5. Compare single-scale and pyramidal estimates.
6. Test two objects moving in different directions.

======================================================================
NEXT LESSON
======================================================================

LESSON 16 — BACKGROUND SUBTRACTION FROM SCRATCH
======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ====================================================================
# CONFIGURATION
# ====================================================================

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 256

SHIFT_X = 20.0
SHIFT_Y = 12.0

PYRAMID_LEVELS = 4
WINDOW_RADIUS = 3
REFINEMENT_ITERATIONS = 4

EIGENVALUE_THRESHOLD = 1e-4

RANDOM_SEED = 42


# ====================================================================
# STEP 1 — SYNTHETIC IMAGE
# ====================================================================

def create_texture():

    rng = np.random.default_rng(RANDOM_SEED)

    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH

    image = rng.normal(
        100,
        18,
        size=(height, width)
    ).astype(np.float32)

    yy, xx = np.mgrid[
        0:height,
        0:width
    ]

    circle = (
        (xx - 65) ** 2
        +
        (yy - 65) ** 2
    ) < 24 ** 2

    image[circle] = (
        180
        +
        30 * np.sin(xx[circle] / 3)
    )

    image[105:150, 115:165] = 210
    image[115:140, 125:155] = 245

    diagonal = np.abs(
        yy - 0.45 * xx - 25
    ) < 3

    image[diagonal] = 35

    for _ in range(15):

        cx = rng.integers(
            15,
            width - 15
        )

        cy = rng.integers(
            15,
            height - 15
        )

        image[
            cy - 2:cy + 3,
            cx - 2:cx + 3
        ] = rng.integers(20, 250)

    return np.clip(
        image,
        0,
        255
    ).astype(np.float32)


# ====================================================================
# STEP 2 — BILINEAR INTERPOLATION
# ====================================================================

def bilinear_sample(image, x, y):

    """
    Bilinear interpolation with correct boundary handling.

    Coordinates outside the image return zero.

    Coordinates exactly on the last row or column are valid.
    Their interpolation neighbors are clamped to the boundary.
    """

    height, width = image.shape

    x, y = np.broadcast_arrays(
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )

    result = np.zeros(
        x.shape,
        dtype=np.float32
    )

    valid = (
        (x >= 0)
        &
        (x <= width - 1)
        &
        (y >= 0)
        &
        (y <= height - 1)
    )

    if not np.any(valid):
        return result

    xv = x[valid]
    yv = y[valid]

    x0 = np.floor(xv).astype(int)
    y0 = np.floor(yv).astype(int)

    # CRITICAL FIX:
    # Clamp the second interpolation coordinate.
    # This preserves values at the image boundary.

    x1 = np.minimum(
        x0 + 1,
        width - 1
    )

    y1 = np.minimum(
        y0 + 1,
        height - 1
    )

    dx = xv - x0
    dy = yv - y0

    top = (
        image[y0, x0] * (1 - dx)
        +
        image[y0, x1] * dx
    )

    bottom = (
        image[y1, x0] * (1 - dx)
        +
        image[y1, x1] * dx
    )

    result[valid] = (
        top * (1 - dy)
        +
        bottom * dy
    )

    return result


# ====================================================================
# STEP 3 — IMAGE TRANSLATION
# ====================================================================

def translate_image(image, dx, dy):

    height, width = image.shape

    yy, xx = np.mgrid[
        0:height,
        0:width
    ]

    source_x = (
        xx.astype(np.float32) - dx
    )

    source_y = (
        yy.astype(np.float32) - dy
    )

    return bilinear_sample(
        image,
        source_x,
        source_y
    )


# ====================================================================
# STEP 4 — GAUSSIAN KERNEL
# ====================================================================

def gaussian_kernel(size=5, sigma=1.0):

    radius = size // 2

    coordinates = np.arange(
        -radius,
        radius + 1
    )

    xx, yy = np.meshgrid(
        coordinates,
        coordinates
    )

    kernel = np.exp(
        -(xx ** 2 + yy ** 2)
        /
        (2 * sigma ** 2)
    )

    kernel /= np.sum(kernel)

    return kernel.astype(np.float32)


# ====================================================================
# STEP 5 — 2D CONVOLUTION
# ====================================================================

def convolve2d(image, kernel):

    height, width = image.shape

    kh, kw = kernel.shape

    ph = kh // 2
    pw = kw // 2

    padded = np.pad(
        image,
        ((ph, ph), (pw, pw)),
        mode="reflect"
    )

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    for y in range(height):

        for x in range(width):

            region = padded[
                y:y + kh,
                x:x + kw
            ]

            output[y, x] = np.sum(
                region * kernel
            )

    return output


# ====================================================================
# STEP 6 — GAUSSIAN PYRAMID
# ====================================================================

def build_gaussian_pyramid(image, levels):

    pyramid = [
        image.astype(np.float32)
    ]

    kernel = gaussian_kernel(
        size=5,
        sigma=1.0
    )

    for _ in range(1, levels):

        current = pyramid[-1]

        if (
            current.shape[0] < 8
            or
            current.shape[1] < 8
        ):
            break

        smoothed = convolve2d(
            current,
            kernel
        )

        downsampled = smoothed[
            ::2,
            ::2
        ]

        pyramid.append(
            downsampled.astype(np.float32)
        )

    return pyramid


# ====================================================================
# STEP 7 — BILINEAR IMAGE RESIZING
# ====================================================================

def resize_bilinear(image, target_shape):

    """
    Resize an image using bilinear interpolation.

    Unlike optical-flow upsampling, this function does not
    scale displacement values.
    """

    target_height, target_width = target_shape

    source_height, source_width = image.shape

    yy, xx = np.mgrid[
        0:target_height,
        0:target_width
    ]

    # Align image endpoints.

    if target_width > 1:

        source_x = (
            xx * (source_width - 1)
            /
            (target_width - 1)
        )

    else:

        source_x = np.zeros_like(
            xx,
            dtype=np.float32
        )

    if target_height > 1:

        source_y = (
            yy * (source_height - 1)
            /
            (target_height - 1)
        )

    else:

        source_y = np.zeros_like(
            yy,
            dtype=np.float32
        )

    return bilinear_sample(
        image,
        source_x,
        source_y
    )


# ====================================================================
# STEP 8 — FLOW UPSAMPLING
# ====================================================================

def upsample_flow(u, v, target_shape):

    """
    Upsample a flow field and scale displacement values.

    Example:

        Low-resolution flow:
            u = 2 pixels
            v = 3 pixels

        Resolution doubles:

            u = 4 pixels
            v = 6 pixels

    This is essential because flow values are measured in pixels.
    """

    target_height, target_width = target_shape

    source_height, source_width = u.shape

    upsampled_u = resize_bilinear(
        u,
        target_shape
    )

    upsampled_v = resize_bilinear(
        v,
        target_shape
    )

    scale_x = (
        target_width / source_width
    )

    scale_y = (
        target_height / source_height
    )

    upsampled_u *= scale_x
    upsampled_v *= scale_y

    return (
        upsampled_u,
        upsampled_v
    )


# ====================================================================
# STEP 9 — SPATIAL IMAGE GRADIENTS
# ====================================================================

def spatial_gradients(image):

    padded = np.pad(
        image,
        1,
        mode="reflect"
    )

    ix = (
        padded[1:-1, 2:]
        -
        padded[1:-1, :-2]
    ) / 2.0

    iy = (
        padded[2:, 1:-1]
        -
        padded[:-2, 1:-1]
    ) / 2.0

    return (
        ix.astype(np.float32),
        iy.astype(np.float32)
    )


# ====================================================================
# STEP 10 — IMAGE WARPING
# ====================================================================

def warp_image(image, u, v):

    """
    Warp image2 toward image1.

    Convention:

        image1(x,y) approximately equals
        image2(x+u, y+v)

    Positive u means content moved right.
    Positive v means content moved down.
    """

    height, width = image.shape

    yy, xx = np.mgrid[
        0:height,
        0:width
    ]

    source_x = (
        xx.astype(np.float32) + u
    )

    source_y = (
        yy.astype(np.float32) + v
    )

    return bilinear_sample(
        image,
        source_x,
        source_y
    )


# ====================================================================
# STEP 11 — ITERATIVE LUCAS-KANADE
# ====================================================================

def lucas_kanade_increment(
    image1,
    image2_warped,
    window_radius=3,
    eigenvalue_threshold=1e-4
):

    """
    Estimate residual displacement between two aligned images.

    Returns:
        du, dv, valid
    """

    ix1, iy1 = spatial_gradients(
        image1
    )

    ix2, iy2 = spatial_gradients(
        image2_warped
    )

    ix = (
        ix1 + ix2
    ) / 2.0

    iy = (
        iy1 + iy2
    ) / 2.0

    it = (
        image2_warped - image1
    )

    height, width = image1.shape

    du = np.zeros(
        (height, width),
        dtype=np.float32
    )

    dv = np.zeros_like(du)

    valid = np.zeros(
        (height, width),
        dtype=bool
    )

    for y in range(
        window_radius,
        height - window_radius
    ):

        for x in range(
            window_radius,
            width - window_radius
        ):

            y0 = y - window_radius
            y1 = y + window_radius + 1

            x0 = x - window_radius
            x1 = x + window_radius + 1

            local_ix = ix[
                y0:y1,
                x0:x1
            ].ravel()

            local_iy = iy[
                y0:y1,
                x0:x1
            ].ravel()

            local_it = it[
                y0:y1,
                x0:x1
            ].ravel()

            A = np.column_stack(
                [
                    local_ix,
                    local_iy
                ]
            )

            b = -local_it

            G = A.T @ A

            eigenvalues = np.linalg.eigvalsh(
                G
            )

            if eigenvalues[0] < eigenvalue_threshold:
                continue

            solution, _, rank, _ = np.linalg.lstsq(
                A,
                b,
                rcond=None
            )

            if rank < 2:
                continue

            if not np.all(
                np.isfinite(solution)
            ):
                continue

            du[y, x] = solution[0]
            dv[y, x] = solution[1]

            valid[y, x] = True

    return (
        du,
        dv,
        valid
    )


# ====================================================================
# STEP 12 — PYRAMIDAL LUCAS-KANADE
# ====================================================================

def pyramidal_lucas_kanade(
    image1,
    image2,
    levels=4,
    window_radius=3,
    iterations=4,
    eigenvalue_threshold=1e-4
):

    """
    Coarse-to-fine optical flow.

    At each pyramid level:

        1. Upsample previous flow.
        2. Warp the second image.
        3. Estimate residual flow.
        4. Update the flow.
        5. Repeat.
    """

    if image1.shape != image2.shape:

        raise ValueError(
            "Images must have equal dimensions."
        )

    pyramid1 = build_gaussian_pyramid(
        image1,
        levels
    )

    pyramid2 = build_gaussian_pyramid(
        image2,
        levels
    )

    actual_levels = min(
        len(pyramid1),
        len(pyramid2)
    )

    u = None
    v = None

    for level in reversed(
        range(actual_levels)
    ):

        frame1 = pyramid1[level]
        frame2 = pyramid2[level]

        height, width = frame1.shape

        if u is None:

            u = np.zeros(
                (height, width),
                dtype=np.float32
            )

            v = np.zeros_like(u)

        else:

            u, v = upsample_flow(
                u,
                v,
                frame1.shape
            )

        for _ in range(iterations):

            warped = warp_image(
                frame2,
                u,
                v
            )

            du, dv, valid = lucas_kanade_increment(
                frame1,
                warped,
                window_radius,
                eigenvalue_threshold
            )

            u[valid] += du[valid]
            v[valid] += dv[valid]

    return (
        u,
        v
    )


# ====================================================================
# STEP 13 — FLOW METRICS
# ====================================================================

def flow_magnitude(u, v):

    return np.sqrt(
        u ** 2 + v ** 2
    )


def flow_error(
    u,
    v,
    ground_truth_u,
    ground_truth_v,
    mask
):

    error = np.sqrt(
        (u - ground_truth_u) ** 2
        +
        (v - ground_truth_v) ** 2
    )

    if not np.any(mask):
        return float("nan")

    return float(
        np.mean(error[mask])
    )


# ====================================================================
# STEP 14 — NUMERICAL TESTS
# ====================================================================

def run_tests():

    print("\nRUNNING TESTS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # TEST 1 — Gaussian kernel normalization
    # ---------------------------------------------------------------

    kernel = gaussian_kernel()

    assert np.isclose(
        np.sum(kernel),
        1.0
    )

    print("Gaussian normalization test passed.")

    # ---------------------------------------------------------------
    # TEST 2 — Pyramid dimensions
    # ---------------------------------------------------------------

    test_image = np.zeros(
        (64, 96),
        dtype=np.float32
    )

    pyramid = build_gaussian_pyramid(
        test_image,
        4
    )

    assert pyramid[0].shape == (64, 96)
    assert pyramid[1].shape == (32, 48)
    assert pyramid[2].shape == (16, 24)
    assert pyramid[3].shape == (8, 12)

    print("Gaussian pyramid dimensions test passed.")

    # ---------------------------------------------------------------
    # TEST 3 — Flow upsampling
    # ---------------------------------------------------------------

    small_u = np.ones(
        (8, 8),
        dtype=np.float32
    ) * 2

    small_v = np.ones(
        (8, 8),
        dtype=np.float32
    ) * 3

    large_u, large_v = upsample_flow(
        small_u,
        small_v,
        (16, 16)
    )

    assert large_u.shape == (16, 16)
    assert large_v.shape == (16, 16)

    # Constant flow must remain constant everywhere,
    # including the last row and column.

    assert np.allclose(
        large_u,
        4.0
    )

    assert np.allclose(
        large_v,
        6.0
    )

    print("Flow upsampling and displacement scaling test passed.")

    # ---------------------------------------------------------------
    # TEST 4 — Bilinear boundary handling
    # ---------------------------------------------------------------

    boundary_image = np.ones(
        (8, 8),
        dtype=np.float32
    ) * 7.0

    boundary_samples = bilinear_sample(
        boundary_image,
        np.array([0.0, 7.0, 7.0]),
        np.array([0.0, 0.0, 7.0])
    )

    assert np.allclose(
        boundary_samples,
        [7.0, 7.0, 7.0]
    )

    print("Bilinear boundary interpolation test passed.")

    # ---------------------------------------------------------------
    # TEST 5 — Image warping
    # ---------------------------------------------------------------

    image = np.zeros(
        (32, 32),
        dtype=np.float32
    )

    image[10:15, 10:15] = 255

    u = np.ones_like(image) * 2
    v = np.zeros_like(image)

    warped = warp_image(
        image,
        u,
        v
    )

    assert warped.shape == image.shape

    assert np.isfinite(
        warped
    ).all()

    assert warped[10, 8] == image[10, 10]

    print("Image-warping test passed.")

    # ---------------------------------------------------------------
    # TEST 6 — Large translation
    # ---------------------------------------------------------------

    rng = np.random.default_rng(123)

    test_texture = rng.normal(
        100,
        35,
        size=(128, 160)
    ).astype(np.float32)

    test_texture[30:75, 25:70] += 80
    test_texture[80:110, 90:130] -= 40

    test_texture = np.clip(
        test_texture,
        0,
        255
    )

    dx = 12.0
    dy = 8.0

    moved_texture = translate_image(
        test_texture,
        dx,
        dy
    )

    estimated_u, estimated_v = pyramidal_lucas_kanade(
        test_texture,
        moved_texture,
        levels=4,
        window_radius=3,
        iterations=5
    )

    yy, xx = np.mgrid[
        0:128,
        0:160
    ]

    evaluation_mask = (
        (xx > 20)
        &
        (xx < 125)
        &
        (yy > 20)
        &
        (yy < 95)
    )

    median_u = np.median(
        estimated_u[evaluation_mask]
    )

    median_v = np.median(
        estimated_v[evaluation_mask]
    )

    assert median_u > 0
    assert median_v > 0

    assert abs(median_u - dx) < 5.0
    assert abs(median_v - dy) < 5.0

    print(
        "Large-translation test passed:",
        f"estimated=({median_u:.2f}, {median_v:.2f}),",
        f"expected=({dx:.2f}, {dy:.2f})"
    )

    print("=" * 70)
    print("ALL TESTS PASSED!")


# ====================================================================
# STEP 15 — VISUALIZATION
# ====================================================================

def visualize_pyramid(pyramid):

    count = len(pyramid)

    fig, axes = plt.subplots(
        1,
        count,
        figsize=(4 * count, 4)
    )

    if count == 1:
        axes = [axes]

    for index, (ax, image) in enumerate(
        zip(axes, pyramid)
    ):

        ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        ax.set_title(
            f"Level {index}\n{image.shape}"
        )

        ax.axis("off")

    fig.suptitle(
        "Gaussian Image Pyramid"
    )

    plt.tight_layout()


def visualize_flow(image, u, v, title):

    magnitude = flow_magnitude(
        u,
        v
    )

    height, width = image.shape

    step = 8

    yy, xx = np.mgrid[
        0:height:step,
        0:width:step
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    im = axes[0].imshow(
        magnitude,
        cmap="viridis"
    )

    axes[0].set_title(
        "Flow Magnitude"
    )

    plt.colorbar(
        im,
        ax=axes[0]
    )

    axes[1].imshow(
        image,
        cmap="gray"
    )

    axes[1].quiver(
        xx,
        yy,
        u[yy, xx],
        v[yy, xx],
        angles="xy",
        scale_units="xy",
        scale=1
    )

    axes[1].set_title(
        "Flow Vectors"
    )

    axes[1].set_xlim(
        0,
        width
    )

    axes[1].set_ylim(
        height,
        0
    )

    fig.suptitle(title)

    plt.tight_layout()


# ====================================================================
# STEP 16 — MAIN EXPERIMENT
# ====================================================================

def main():

    print("=" * 70)
    print("COMPUTER VISION FROM SCRATCH")
    print("LESSON 15 — PYRAMIDAL OPTICAL FLOW")
    print("=" * 70)

    image1 = create_texture()

    image2 = translate_image(
        image1,
        SHIFT_X,
        SHIFT_Y
    )

    print("\nINPUT")
    print("-" * 70)

    print("Image shape:", image1.shape)

    print(
        "Ground-truth translation:",
        (SHIFT_X, SHIFT_Y)
    )

    # ---------------------------------------------------------------
    # Build Gaussian pyramids.
    # ---------------------------------------------------------------

    pyramid1 = build_gaussian_pyramid(
        image1,
        PYRAMID_LEVELS
    )

    print("\nPYRAMID SHAPES")

    for level, image in enumerate(pyramid1):

        print(
            f"Level {level}: {image.shape}"
        )

    # ---------------------------------------------------------------
    # Pyramidal optical flow.
    # ---------------------------------------------------------------

    print("\nESTIMATING PYRAMIDAL OPTICAL FLOW")

    u, v = pyramidal_lucas_kanade(
        image1,
        image2,
        levels=PYRAMID_LEVELS,
        window_radius=WINDOW_RADIUS,
        iterations=REFINEMENT_ITERATIONS,
        eigenvalue_threshold=EIGENVALUE_THRESHOLD
    )

    # ---------------------------------------------------------------
    # Single-scale comparison.
    # ---------------------------------------------------------------

    print("\nCOMPARING SINGLE-SCALE AND PYRAMIDAL FLOW")

    basic_du, basic_dv, basic_valid = lucas_kanade_increment(
        image1,
        image2,
        window_radius=WINDOW_RADIUS,
        eigenvalue_threshold=EIGENVALUE_THRESHOLD
    )

    basic_u = basic_du
    basic_v = basic_dv

    # ---------------------------------------------------------------
    # Evaluation region.
    # ---------------------------------------------------------------

    yy, xx = np.mgrid[
        0:IMAGE_HEIGHT,
        0:IMAGE_WIDTH
    ]

    mask = (
        (xx > 30)
        &
        (xx < IMAGE_WIDTH - 35)
        &
        (yy > 30)
        &
        (yy < IMAGE_HEIGHT - 35)
    )

    # ---------------------------------------------------------------
    # Error calculations.
    # ---------------------------------------------------------------

    pyramid_error = flow_error(
        u,
        v,
        SHIFT_X,
        SHIFT_Y,
        mask
    )

    basic_error = flow_error(
        basic_u,
        basic_v,
        SHIFT_X,
        SHIFT_Y,
        mask & basic_valid
    )

    print("\nRESULTS")
    print("-" * 70)

    print(
        "Ground truth:",
        (SHIFT_X, SHIFT_Y)
    )

    print(
        "Pyramidal median flow:",
        (
            round(float(np.median(u[mask])), 3),
            round(float(np.median(v[mask])), 3)
        )
    )

    print(
        "Pyramidal mean endpoint error:",
        round(pyramid_error, 4)
    )

    print(
        "Single-scale mean endpoint error:",
        round(basic_error, 4)
    )

    # ---------------------------------------------------------------
    # Visualizations.
    # ---------------------------------------------------------------

    visualize_pyramid(
        pyramid1
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    axes[0].imshow(
        image1,
        cmap="gray"
    )

    axes[0].set_title(
        "Frame 1"
    )

    axes[1].imshow(
        image2,
        cmap="gray"
    )

    axes[1].set_title(
        "Frame 2"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    visualize_flow(
        image1,
        basic_u,
        basic_v,
        "Single-Scale Lucas-Kanade"
    )

    visualize_flow(
        image1,
        u,
        v,
        "Pyramidal Lucas-Kanade"
    )

    # ---------------------------------------------------------------
    # Numerical tests.
    # ---------------------------------------------------------------

    run_tests()

    # ---------------------------------------------------------------
    # Robotics interpretation.
    # ---------------------------------------------------------------

    print("\nOPENARM V2 ROBOTICS APPLICATION")
    print("-" * 70)

    print(
        "Pyramidal optical flow estimates larger apparent"
    )

    print(
        "pixel displacement between consecutive camera frames."
    )

    print(
        "For waste sorting, this can help estimate conveyor"
    )

    print(
        "motion and track moving objects before grasping."
    )

    print(
        "Physical velocity estimation requires camera calibration"
    )

    print(
        "and depth information."
    )

    print("\nNEXT LESSON:")
    print("LESSON 16 — BACKGROUND SUBTRACTION FROM SCRATCH")

    plt.show()


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()