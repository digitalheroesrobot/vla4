"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 14 — OPTICAL FLOW
======================================================================

COURSE RULE
-----------
Implement the main optical-flow mathematics from scratch.

NO:
    - cv2.calcOpticalFlowPyrLK()
    - cv2.calcOpticalFlowFarneback()
    - pretrained models
    - pretrained weights

Libraries:
    - NumPy
    - Matplotlib

======================================================================
WHAT YOU WILL LEARN
======================================================================

1. What is optical flow?
2. Why consecutive images contain motion information.
3. Brightness-constancy assumption.
4. Image spatial gradients.
5. Temporal image gradient.
6. Optical-flow constraint equation.
7. The aperture problem.
8. Lucas-Kanade optical flow.
9. Local least-squares estimation.
10. Flow magnitude and direction.
11. Motion-vector visualization.
12. Numerical validation using known synthetic motion.

======================================================================
1. WHAT IS OPTICAL FLOW?
======================================================================

Optical flow estimates apparent pixel motion between two images.

Let:

    I1(x,y) = first image
    I2(x,y) = second image

A pixel moves approximately from:

    (x,y) -> (x+u, y+v)

where:

    u = horizontal displacement
    v = vertical displacement

The optical-flow vector is:

    F(x,y) = [u(x,y), v(x,y)]

Optical flow is useful for understanding how visual information
changes over time.

======================================================================
2. BRIGHTNESS CONSTANCY
======================================================================

The classical optical-flow assumption is:

    I(x,y,t) = I(x+u, y+v, t+1)

For small motion, use a first-order Taylor expansion:

    I(x+u, y+v, t+1)
        ≈ I(x,y,t)
        + Ix*u
        + Iy*v
        + It

Therefore:

    Ix*u + Iy*v + It = 0

This is the optical-flow constraint equation.

Where:

    Ix = spatial image gradient in x
    Iy = spatial image gradient in y
    It = temporal image gradient

======================================================================
3. THE APERTURE PROBLEM
======================================================================

The optical-flow constraint provides one equation:

    Ix*u + Iy*v = -It

But there are two unknowns:

    u, v

Therefore, one pixel alone cannot determine both motion
components reliably.

This is called the aperture problem.

======================================================================
4. LUCAS-KANADE METHOD
======================================================================

Lucas-Kanade assumes that nearby pixels have approximately
the same motion.

For a local window containing N pixels:

    Ix1*u + Iy1*v = -It1
    Ix2*u + Iy2*v = -It2
    ...
    IxN*u + IyN*v = -ItN

Matrix form:

    A * flow = b

where:

        [ Ix1  Iy1 ]
        [ Ix2  Iy2 ]
    A = [   ...   ]

        [ IxN  IyN ]

        [ -It1 ]
        [ -It2 ]
    b = [  ... ]
        [ -ItN ]

Least-squares solution:

    flow = (A.T * A)^(-1) * A.T * b

Numerically, use:

    np.linalg.lstsq(A, b)

This is more stable than explicitly calculating the matrix inverse.

======================================================================
5. LOCAL WINDOW
======================================================================

The algorithm calculates motion using a neighborhood around
each pixel.

A larger window:
    - Provides more measurements.
    - Can improve numerical stability.
    - May mix different objects with different motion.

A smaller window:
    - Preserves local motion changes.
    - May be less stable.
    - Can suffer from insufficient texture.

======================================================================
6. FLOW QUALITY
======================================================================

The structure tensor is:

    G = A.T * A

Its eigenvalues describe how well motion can be estimated.

If both eigenvalues are small:
    The region has little texture.

If one eigenvalue is large and the other is small:
    The region has an aperture-problem ambiguity.

If both are sufficiently large:
    The region contains useful two-dimensional texture.

======================================================================
7. LIMITATIONS
======================================================================

Classical Lucas-Kanade assumes:

    - Small motion between frames.
    - Approximately constant brightness.
    - Similar motion inside the local window.
    - Enough local image texture.

It may struggle with:

    - Large displacement.
    - Motion blur.
    - Occlusion.
    - Strong lighting changes.
    - Deforming objects.
    - Multiple motions inside one window.

Pyramidal Lucas-Kanade will be studied in a later lesson.

======================================================================
REAL-WORLD ROBOTICS EXAMPLE
======================================================================

OPENARM V2 — WASTE-SORTING ROBOT

A camera observes a moving waste object.

Frame t:
    Object at (x1, y1)

Frame t+1:
    Object at (x2, y2)

Optical flow estimates local image displacement.

Potential uses:

    - Detect object movement on a conveyor.
    - Estimate relative object motion.
    - Detect unexpected scene motion.
    - Track objects between frames.
    - Support visual servoing.

Important:

    Optical flow estimates IMAGE motion.

It does not directly provide:
    - 3-D object velocity.
    - Robot joint velocity.
    - World-coordinate displacement.

Camera calibration and scene geometry are needed for those tasks.

======================================================================
EXERCISES
======================================================================

1. Change the synthetic translation.
2. Increase the optical-flow window size.
3. Reduce the window size.
4. Add stronger image noise.
5. Add brightness changes between frames.
6. Compare estimated flow with known ground-truth displacement.
7. Test a scene containing two independently moving objects.

======================================================================
NEXT LESSON
======================================================================

LESSON 15 — PYRAMIDAL OPTICAL FLOW

We will handle larger displacements by constructing image
pyramids and estimating motion from coarse to fine.

======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ====================================================================
# CONFIGURATION
# ====================================================================

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 220

WINDOW_RADIUS = 4

SHIFT_X = 2.0
SHIFT_Y = 1.0

FLOW_STEP = 6

EIGENVALUE_THRESHOLD = 1e-4

RANDOM_SEED = 42


# ====================================================================
# STEP 1 — CREATE SYNTHETIC TEXTURED IMAGE
# ====================================================================

def create_texture():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    image = rng.normal(
        100,
        20,
        size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    ).astype(np.float32)

    yy, xx = np.mgrid[
        0:IMAGE_HEIGHT,
        0:IMAGE_WIDTH
    ]

    # Add geometric structures.

    circle = (
        (xx - 65) ** 2
        +
        (yy - 55) ** 2
    ) < 22 ** 2

    image[circle] = 220

    image[95:135, 120:170] = 180

    image[105:125, 130:160] = 245

    image[25:45, 165:195] = 45

    image = np.clip(
        image,
        0,
        255
    )

    return image.astype(
        np.float32
    )


# ====================================================================
# STEP 2 — TRANSLATE IMAGE USING BILINEAR SAMPLING
# ====================================================================

def bilinear_sample(image, x, y):

    """
    Sample image at floating-point coordinates using bilinear
    interpolation.
    """

    height, width = image.shape

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))

    x1 = x0 + 1
    y1 = y0 + 1

    if (
        x0 < 0
        or
        y0 < 0
        or
        x1 >= width
        or
        y1 >= height
    ):

        return 0.0

    dx = x - x0
    dy = y - y0

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

    return (
        top * (1 - dy)
        +
        bottom * dy
    )


def translate_image(image, dx, dy):

    """
    Generate a translated image.

    Positive dx moves content right.
    Positive dy moves content down.
    """

    height, width = image.shape

    translated = np.zeros_like(
        image
    )

    for y in range(height):

        for x in range(width):

            source_x = x - dx
            source_y = y - dy

            translated[y, x] = bilinear_sample(
                image,
                source_x,
                source_y
            )

    return translated


# ====================================================================
# STEP 3 — IMAGE GRADIENTS
# ====================================================================

def spatial_gradients(image):

    """
    Central-difference spatial derivatives.

    Ix = horizontal derivative
    Iy = vertical derivative
    """

    image = image.astype(
        np.float32
    )

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

    return ix, iy


# ====================================================================
# STEP 4 — TEMPORAL GRADIENT
# ====================================================================

def temporal_gradient(image1, image2):

    return (
        image2.astype(np.float32)
        -
        image1.astype(np.float32)
    )


# ====================================================================
# STEP 5 — LUCAS-KANADE OPTICAL FLOW
# ====================================================================

def lucas_kanade(
    image1,
    image2,
    window_radius=4,
    eigenvalue_threshold=1e-4
):

    """
    Dense local Lucas-Kanade optical flow.

    For every pixel:

        Ix*u + Iy*v = -It

    Build a local least-squares system and solve for (u,v).

    Pixels without enough two-dimensional texture are marked invalid.
    """

    image1 = image1.astype(
        np.float32
    )

    image2 = image2.astype(
        np.float32
    )

    if image1.shape != image2.shape:

        raise ValueError(
            "Input images must have the same dimensions."
        )

    ix1, iy1 = spatial_gradients(
        image1
    )

    ix2, iy2 = spatial_gradients(
        image2
    )

    ix = (
        ix1 + ix2
    ) / 2.0

    iy = (
        iy1 + iy2
    ) / 2.0

    it = temporal_gradient(
        image1,
        image2
    )

    height, width = image1.shape

    u = np.zeros(
        (height, width),
        dtype=np.float32
    )

    v = np.zeros(
        (height, width),
        dtype=np.float32
    )

    valid = np.zeros(
        (height, width),
        dtype=bool
    )

    min_eigenvalue = np.zeros(
        (height, width),
        dtype=np.float32
    )

    # ---------------------------------------------------------------
    # Solve local least-squares systems.
    # ---------------------------------------------------------------

    for y in range(
        window_radius,
        height - window_radius
    ):

        for x in range(
            window_radius,
            width - window_radius
        ):

            x0 = x - window_radius
            x1 = x + window_radius + 1

            y0 = y - window_radius
            y1 = y + window_radius + 1

            local_ix = ix[
                y0:y1,
                x0:x1
            ].flatten()

            local_iy = iy[
                y0:y1,
                x0:x1
            ].flatten()

            local_it = it[
                y0:y1,
                x0:x1
            ].flatten()

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

            minimum_eigenvalue = float(
                eigenvalues[0]
            )

            min_eigenvalue[y, x] = (
                minimum_eigenvalue
            )

            # Require sufficient texture in both directions.

            if minimum_eigenvalue < eigenvalue_threshold:
                continue

            solution, _, rank, _ = np.linalg.lstsq(
                A,
                b,
                rcond=None
            )

            if rank < 2:
                continue

            flow_u, flow_v = solution

            if not np.all(
                np.isfinite(solution)
            ):
                continue

            u[y, x] = flow_u
            v[y, x] = flow_v

            valid[y, x] = True

    return (
        u,
        v,
        valid,
        min_eigenvalue
    )


# ====================================================================
# STEP 6 — FLOW MAGNITUDE AND ANGLE
# ====================================================================

def flow_magnitude(u, v):

    return np.sqrt(
        u ** 2 + v ** 2
    )


def flow_angle(u, v):

    return np.arctan2(
        v,
        u
    )


# ====================================================================
# STEP 7 — FLOW COLOR REPRESENTATION
# ====================================================================

def flow_to_rgb(u, v, valid):

    """
    Simple visualization:

        Hue        = direction
        Saturation = confidence / validity
        Value      = normalized magnitude

    This is an educational visualization rather than a standard
    Middlebury optical-flow color wheel.
    """

    magnitude = flow_magnitude(
        u,
        v
    )

    angle = flow_angle(
        u,
        v
    )

    hue = (
        angle + np.pi
    ) / (2 * np.pi)

    max_magnitude = np.max(
        magnitude[valid]
    ) if np.any(valid) else 1.0

    value = magnitude / (
        max_magnitude + 1e-8
    )

    saturation = valid.astype(
        np.float32
    )

    # Convert HSV to RGB manually.

    h6 = hue * 6.0

    sector = np.floor(
        h6
    ).astype(int) % 6

    fraction = h6 - np.floor(h6)

    p = value * (1 - saturation)

    q = value * (1 - fraction * saturation)

    t = value * (1 - (1 - fraction) * saturation)

    rgb = np.zeros(
        (*u.shape, 3),
        dtype=np.float32
    )

    conditions = [
        sector == 0,
        sector == 1,
        sector == 2,
        sector == 3,
        sector == 4,
        sector == 5
    ]

    values = [
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q)
    ]

    for condition, channels in zip(
        conditions,
        values
    ):

        for channel in range(3):

            rgb[..., channel][condition] = (
                channels[channel][condition]
            )

    return np.clip(
        rgb,
        0,
        1
    )


# ====================================================================
# STEP 8 — SAMPLE FLOW VECTORS
# ====================================================================

def sample_flow_vectors(
    u,
    v,
    valid,
    step=6
):

    height, width = u.shape

    ys, xs = np.mgrid[
        0:height:step,
        0:width:step
    ]

    sampled_u = u[
        ys,
        xs
    ]

    sampled_v = v[
        ys,
        xs
    ]

    sampled_valid = valid[
        ys,
        xs
    ]

    return (
        xs[sampled_valid],
        ys[sampled_valid],
        sampled_u[sampled_valid],
        sampled_v[sampled_valid]
    )


# ====================================================================
# STEP 9 — NUMERICAL TESTS
# ====================================================================

def run_tests():

    print("\nRUNNING TESTS")
    print("-" * 70)

    # ---------------------------------------------------------------
    # Test 1: Gradient shape
    # ---------------------------------------------------------------

    test_image = np.arange(
        100,
        dtype=np.float32
    ).reshape(
        10,
        10
    )

    gx, gy = spatial_gradients(
        test_image
    )

    assert gx.shape == test_image.shape
    assert gy.shape == test_image.shape

    print("Spatial-gradient shape test passed.")

    # ---------------------------------------------------------------
    # Test 2: Horizontal ramp
    # ---------------------------------------------------------------

    ramp_x = np.tile(
        np.arange(20, dtype=np.float32),
        (20, 1)
    )

    gx, gy = spatial_gradients(
        ramp_x
    )

    assert np.allclose(
        gx[2:-2, 2:-2],
        1.0
    )

    assert np.allclose(
        gy[2:-2, 2:-2],
        0.0
    )

    print("Horizontal-gradient test passed.")

    # ---------------------------------------------------------------
    # Test 3: Vertical ramp
    # ---------------------------------------------------------------

    ramp_y = np.tile(
        np.arange(20, dtype=np.float32)[:, None],
        (1, 20)
    )

    gx, gy = spatial_gradients(
        ramp_y
    )

    assert np.allclose(
        gx[2:-2, 2:-2],
        0.0
    )

    assert np.allclose(
        gy[2:-2, 2:-2],
        1.0
    )

    print("Vertical-gradient test passed.")

    # ---------------------------------------------------------------
    # Test 4: Flow magnitude
    # ---------------------------------------------------------------

    u_test = np.array(
        [[3.0, 0.0]],
        dtype=np.float32
    )

    v_test = np.array(
        [[4.0, 5.0]],
        dtype=np.float32
    )

    magnitude = flow_magnitude(
        u_test,
        v_test
    )

    assert np.allclose(
        magnitude,
        [[5.0, 5.0]]
    )

    print("Flow-magnitude test passed.")

    # ---------------------------------------------------------------
    # Test 5: Translation direction and approximate displacement
    # ---------------------------------------------------------------

    image1 = create_texture()

    image2 = translate_image(
        image1,
        1.0,
        1.0
    )

    u, v, valid, _ = lucas_kanade(
        image1,
        image2,
        window_radius=4,
        eigenvalue_threshold=1e-4
    )

    interior = (
        valid
        &
        (np.indices(valid.shape)[0] > 10)
        &
        (np.indices(valid.shape)[0] < IMAGE_HEIGHT - 10)
        &
        (np.indices(valid.shape)[1] > 10)
        &
        (np.indices(valid.shape)[1] < IMAGE_WIDTH - 10)
    )

    if np.sum(interior) > 0:

        median_u = np.median(
            u[interior]
        )

        median_v = np.median(
            v[interior]
        )

        assert median_u > 0
        assert median_v > 0

        print(
            "Translation direction test passed:",
            round(float(median_u), 3),
            round(float(median_v), 3)
        )

    else:

        raise AssertionError(
            "No valid optical-flow pixels found."
        )

    print("-" * 70)
    print("ALL TESTS PASSED!")


# ====================================================================
# STEP 10 — MAIN EXPERIMENT
# ====================================================================

def main():

    print("=" * 70)
    print("COMPUTER VISION FROM SCRATCH")
    print("LESSON 14 — OPTICAL FLOW")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Generate two consecutive frames.
    # ---------------------------------------------------------------

    image1 = create_texture()

    image2 = translate_image(
        image1,
        SHIFT_X,
        SHIFT_Y
    )

    print("\nINPUT INFORMATION")
    print("-" * 70)

    print("Frame shape:", image1.shape)

    print(
        "Ground-truth translation:",
        (SHIFT_X, SHIFT_Y)
    )

    # ---------------------------------------------------------------
    # Calculate optical flow.
    # ---------------------------------------------------------------

    print("\nCALCULATING LUCAS-KANADE OPTICAL FLOW")
    print("-" * 70)

    u, v, valid, min_eigenvalue = lucas_kanade(
        image1,
        image2,
        window_radius=WINDOW_RADIUS,
        eigenvalue_threshold=EIGENVALUE_THRESHOLD
    )

    magnitude = flow_magnitude(
        u,
        v
    )

    angle = flow_angle(
        u,
        v
    )

    flow_rgb = flow_to_rgb(
        u,
        v,
        valid
    )

    print(
        "Valid flow pixels:",
        int(np.sum(valid))
    )

    print(
        "Valid pixel ratio:",
        round(float(np.mean(valid)), 4)
    )

    if np.any(valid):

        median_u = float(
            np.median(u[valid])
        )

        median_v = float(
            np.median(v[valid])
        )

        median_magnitude = float(
            np.median(magnitude[valid])
        )

        print(
            "Median horizontal flow:",
            round(median_u, 4)
        )

        print(
            "Median vertical flow:",
            round(median_v, 4)
        )

        print(
            "Median flow magnitude:",
            round(median_magnitude, 4)
        )

    # ---------------------------------------------------------------
    # Sample flow vectors for quiver plot.
    # ---------------------------------------------------------------

    xs, ys, us, vs = sample_flow_vectors(
        u,
        v,
        valid,
        step=FLOW_STEP
    )

    # ---------------------------------------------------------------
    # Visualization 1 — Input frames.
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5)
    )

    axes[0].imshow(
        image1,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0].set_title(
        "Frame 1"
    )

    axes[1].imshow(
        image2,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1].set_title(
        "Frame 2 — Translated"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 2 — Flow components.
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5)
    )

    u_display = np.where(
        valid,
        u,
        np.nan
    )

    v_display = np.where(
        valid,
        v,
        np.nan
    )

    magnitude_display = np.where(
        valid,
        magnitude,
        np.nan
    )

    im0 = axes[0].imshow(
        u_display,
        cmap="coolwarm"
    )

    axes[0].set_title(
        "Horizontal Flow (u)"
    )

    plt.colorbar(
        im0,
        ax=axes[0]
    )

    im1 = axes[1].imshow(
        v_display,
        cmap="coolwarm"
    )

    axes[1].set_title(
        "Vertical Flow (v)"
    )

    plt.colorbar(
        im1,
        ax=axes[1]
    )

    im2 = axes[2].imshow(
        magnitude_display,
        cmap="viridis"
    )

    axes[2].set_title(
        "Flow Magnitude"
    )

    plt.colorbar(
        im2,
        ax=axes[2]
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 3 — Optical-flow vectors.
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    ax.imshow(
        image1,
        cmap="gray"
    )

    ax.quiver(
        xs,
        ys,
        us,
        vs,
        angles="xy",
        scale_units="xy",
        scale=1
    )

    ax.set_title(
        "Lucas-Kanade Optical Flow Vectors"
    )

    ax.set_xlim(
        0,
        IMAGE_WIDTH
    )

    ax.set_ylim(
        IMAGE_HEIGHT,
        0
    )

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 4 — Flow direction and magnitude.
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.imshow(
        flow_rgb
    )

    ax.set_title(
        "Optical Flow Visualization"
    )

    ax.axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 5 — Minimum eigenvalue.
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    eigen_display = np.where(
        valid,
        min_eigenvalue,
        np.nan
    )

    im = ax.imshow(
        eigen_display,
        cmap="viridis"
    )

    ax.set_title(
        "Minimum Structure-Tensor Eigenvalue"
    )

    plt.colorbar(
        im,
        ax=ax
    )

    ax.axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 6 — Flow angle.
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    angle_display = np.where(
        valid,
        angle,
        np.nan
    )

    im = ax.imshow(
        angle_display,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi
    )

    ax.set_title(
        "Optical Flow Direction"
    )

    plt.colorbar(
        im,
        ax=ax
    )

    ax.axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Numerical tests.
    # ---------------------------------------------------------------

    run_tests()

    # ---------------------------------------------------------------
    # Robotics interpretation.
    # ---------------------------------------------------------------

    print("\nOPENARM V2 — ROBOTICS INTERPRETATION")
    print("-" * 70)

    print(
        "Optical flow estimates apparent motion between camera frames."
    )

    print(
        "For waste sorting, it can help detect moving objects,"
    )

    print(
        "conveyor motion, and changes in the visual scene."
    )

    print(
        "\nIt is an image-space measurement, not direct 3-D velocity."
    )

    print(
        "Depth and camera calibration are needed to estimate"
    )

    print(
        "physical motion in the robot coordinate system."
    )

    # ---------------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------------

    print("\nLESSON 14 SUMMARY")
    print("-" * 70)

    print("1. Generated two synthetic frames.")
    print("2. Implemented spatial image gradients.")
    print("3. Implemented temporal image gradients.")
    print("4. Implemented the optical-flow constraint.")
    print("5. Solved local Lucas-Kanade least-squares systems.")
    print("6. Calculated flow magnitude and direction.")
    print("7. Visualized motion vectors.")
    print("8. Validated flow direction against known motion.")

    print("\nNEXT LESSON:")
    print("LESSON 15 — PYRAMIDAL OPTICAL FLOW")

    plt.show()


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()