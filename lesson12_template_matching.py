"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 12 — TEMPLATE MATCHING
======================================================================

COURSE RULE
-----------
Everything in this lesson is implemented from scratch.

NO:
    - cv2.matchTemplate()
    - pretrained models
    - pretrained weights
    - black-box matching algorithms

Libraries:
    - NumPy
    - Matplotlib

======================================================================
WHAT YOU WILL LEARN
======================================================================

1. What is template matching?
2. Sliding-window search.
3. Sum of Squared Differences (SSD).
4. Mean Squared Error (MSE).
5. Normalized Cross-Correlation (NCC).
6. Similarity maps.
7. Finding the best matching location.
8. Comparing matching methods.
9. Robustness to brightness changes.
10. Limitations of classical template matching.

======================================================================
1. TEMPLATE MATCHING
======================================================================

Template matching searches for a small image inside a larger image.

Let:

    I = source image
    T = template image

The template has dimensions:

    T_height x T_width

We slide the template over every valid position in I.

At each position, extract an image patch:

    P(x,y)

Compare the patch with the template.

The best matching position is the location with the highest
similarity or lowest matching error.

======================================================================
2. SLIDING WINDOW
======================================================================

For an image of size:

    H x W

and template size:

    h x w

the number of valid positions is:

    (H - h + 1) x (W - w + 1)

At every position:

    patch = image[y:y+h, x:x+w]

The matching algorithm evaluates this patch.

======================================================================
3. SUM OF SQUARED DIFFERENCES (SSD)
======================================================================

SSD measures the total squared pixel difference.

    SSD(x,y) = sum((P(x,y) - T)^2)

Lower is better.

Perfect match:

    SSD = 0

SSD is sensitive to brightness and contrast changes.

======================================================================
4. MEAN SQUARED ERROR (MSE)
======================================================================

MSE is the average squared pixel difference.

    MSE(x,y) = SSD(x,y) / N

where:

    N = h * w

Lower is better.

SSD and MSE produce the same best location for a fixed-size template.

======================================================================
5. NORMALIZED CROSS-CORRELATION (NCC)
======================================================================

NCC compares the relationship between pixel intensities.

First subtract the mean:

    P_centered = P - mean(P)

    T_centered = T - mean(T)

Then:

    NCC = sum(P_centered * T_centered)
          --------------------------------
          sqrt(sum(P_centered^2) * sum(T_centered^2))

Range:

    -1 <= NCC <= 1

Interpretation:

    NCC =  1  : perfect positive similarity
    NCC =  0  : no linear correlation
    NCC = -1  : perfect inverse relationship

Higher is better.

NCC is more robust to additive brightness changes and positive
linear contrast scaling than raw SSD.

A constant template has zero variance, so NCC is undefined.
We handle that case explicitly.

======================================================================
6. SIMILARITY MAP
======================================================================

The matching algorithm produces a 2-D score map.

Each map location represents one possible template position.

For SSD:

    best location = argmin(score_map)

For NCC:

    best location = argmax(score_map)

The best coordinate is the top-left corner of the matched region.

======================================================================
7. IMPORTANT LIMITATIONS
======================================================================

Classical template matching assumes that the object has a similar:

    - Scale
    - Rotation
    - Appearance
    - Shape
    - Viewpoint

It is not naturally invariant to rotation or scale.

It can also fail when:

    - Objects are partially occluded.
    - Lighting changes significantly.
    - The object deforms.
    - The camera viewpoint changes.
    - Multiple similar objects appear.

Later lessons will explore feature matching and more robust methods.

======================================================================
REAL-WORLD ROBOTICS EXAMPLE
======================================================================

OPENARM V2 — WASTE SORTING

Imagine that a robot must locate a known cup in a camera image.

A simple pipeline:

    Camera Image
        |
        v
    Grayscale Conversion
        |
        v
    Template Matching
        |
        v
    Similarity Map
        |
        v
    Best Matching Location
        |
        v
    Bounding Box
        |
        v
    Object Localization
        |
        v
    Grasp Planning / ACT

Template matching can be useful when:

    - Object appearance is relatively consistent.
    - The camera position is fixed.
    - The target has a known visual pattern.

It is not a complete waste-classification solution.

For varied waste objects, a learned detector or segmentation model
will eventually be more suitable.

======================================================================
EXERCISES
======================================================================

1. Change the template location.
2. Change the template size.
3. Compare SSD and NCC.
4. Add stronger brightness changes.
5. Add Gaussian noise.
6. Search for multiple copies of the same object.
7. Try matching a template after rotating it.

======================================================================
NEXT LESSON
======================================================================

LESSON 13 — FEATURE MATCHING AND DESCRIPTOR-BASED CORRESPONDENCE

We will connect template matching with local image features and
learn how to match corresponding points between images.

======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ====================================================================
# CONFIGURATION
# ====================================================================

IMAGE_HEIGHT = 220
IMAGE_WIDTH = 340

TEMPLATE_HEIGHT = 45
TEMPLATE_WIDTH = 55

RANDOM_SEED = 42

BRIGHTNESS_OFFSET = 25.0
CONTRAST_FACTOR = 1.15

NOISE_STD = 2.0


# ====================================================================
# STEP 1 — CREATE A SYNTHETIC OBJECT
# ====================================================================

def create_object(height, width):

    """
    Create an asymmetric synthetic object.

    Asymmetry makes the template easier to locate and helps
    demonstrate why object appearance matters.
    """

    obj = np.full(
        (height, width),
        35.0,
        dtype=np.float32
    )

    yy, xx = np.mgrid[
        0:height,
        0:width
    ]

    # Main body
    body = (
        (xx >= 8)
        &
        (xx <= width - 9)
        &
        (yy >= 8)
        &
        (yy <= height - 9)
    )

    obj[body] = 170.0

    # Bright upper-left region
    obj[
        10:20,
        12:25
    ] = 245.0

    # Dark lower-right region
    obj[
        28:38,
        35:48
    ] = 65.0

    # Small vertical feature
    obj[
        14:34,
        width // 2 - 2:width // 2 + 2
    ] = 110.0

    # Asymmetric diagonal detail
    diagonal = (
        (xx >= 12)
        &
        (xx <= width - 12)
        &
        (np.abs(yy - (0.55 * xx + 5)) <= 1.5)
    )

    obj[diagonal] = 220.0

    return obj


# ====================================================================
# STEP 2 — CREATE A LARGER SCENE
# ====================================================================

def create_scene():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    image = np.full(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        25.0,
        dtype=np.float32
    )

    # ---------------------------------------------------------------
    # Target object location
    # ---------------------------------------------------------------

    target_x = 215
    target_y = 115

    target = create_object(
        TEMPLATE_HEIGHT,
        TEMPLATE_WIDTH
    )

    image[
        target_y:target_y + TEMPLATE_HEIGHT,
        target_x:target_x + TEMPLATE_WIDTH
    ] = target

    # ---------------------------------------------------------------
    # Add distractor objects
    # ---------------------------------------------------------------

    distractor1 = create_object(
        TEMPLATE_HEIGHT,
        TEMPLATE_WIDTH
    )

    distractor1 = np.clip(
        distractor1 * 0.45,
        0,
        255
    )

    image[
        25:25 + TEMPLATE_HEIGHT,
        35:35 + TEMPLATE_WIDTH
    ] = distractor1

    distractor2 = create_object(
        TEMPLATE_HEIGHT,
        TEMPLATE_WIDTH
    )

    distractor2 = np.clip(
        distractor2 + 45,
        0,
        255
    )

    image[
        145:145 + TEMPLATE_HEIGHT,
        115:115 + TEMPLATE_WIDTH
    ] = distractor2

    # ---------------------------------------------------------------
    # Add background texture
    # ---------------------------------------------------------------

    texture = rng.normal(
        0,
        1.5,
        image.shape
    )

    image += texture

    return np.clip(
        image,
        0,
        255
    ).astype(np.float32), target, (target_x, target_y)


# ====================================================================
# STEP 3 — SLIDING-WINDOW SSD
# ====================================================================

def ssd_matching(image, template):

    """
    Calculate SSD at every valid image position.

    Output shape:

        (H - h + 1, W - w + 1)
    """

    image_height, image_width = image.shape

    template_height, template_width = template.shape

    output_height = (
        image_height - template_height + 1
    )

    output_width = (
        image_width - template_width + 1
    )

    if output_height <= 0 or output_width <= 0:
        raise ValueError(
            "Template must not be larger than image."
        )

    score_map = np.zeros(
        (output_height, output_width),
        dtype=np.float32
    )

    # ---------------------------------------------------------------
    # Sliding-window search
    # ---------------------------------------------------------------

    for y in range(output_height):

        for x in range(output_width):

            patch = image[
                y:y + template_height,
                x:x + template_width
            ]

            difference = patch - template

            score_map[y, x] = np.sum(
                difference ** 2
            )

    return score_map


# ====================================================================
# STEP 4 — SLIDING-WINDOW MSE
# ====================================================================

def mse_matching(image, template):

    ssd_map = ssd_matching(
        image,
        template
    )

    number_of_pixels = template.size

    return ssd_map / number_of_pixels


# ====================================================================
# STEP 5 — NORMALIZED CROSS-CORRELATION
# ====================================================================

def ncc_matching(image, template, epsilon=1e-8):

    """
    Compute normalized cross-correlation for every valid patch.

    Constant patches or templates have undefined NCC.

    We assign -1 to invalid positions because higher NCC is better.
    """

    image_height, image_width = image.shape

    template_height, template_width = template.shape

    output_height = (
        image_height - template_height + 1
    )

    output_width = (
        image_width - template_width + 1
    )

    if output_height <= 0 or output_width <= 0:
        raise ValueError(
            "Template must not be larger than image."
        )

    template_centered = (
        template - np.mean(template)
    )

    template_norm = np.sqrt(
        np.sum(template_centered ** 2)
    )

    score_map = np.full(
        (output_height, output_width),
        -1.0,
        dtype=np.float32
    )

    if template_norm < epsilon:

        return score_map

    # ---------------------------------------------------------------
    # Sliding-window NCC
    # ---------------------------------------------------------------

    for y in range(output_height):

        for x in range(output_width):

            patch = image[
                y:y + template_height,
                x:x + template_width
            ]

            patch_centered = (
                patch - np.mean(patch)
            )

            patch_norm = np.sqrt(
                np.sum(patch_centered ** 2)
            )

            if patch_norm < epsilon:
                continue

            numerator = np.sum(
                patch_centered * template_centered
            )

            denominator = (
                patch_norm * template_norm
            )

            score = numerator / denominator

            score_map[y, x] = np.clip(
                score,
                -1.0,
                1.0
            )

    return score_map


# ====================================================================
# STEP 6 — FIND BEST MATCH
# ====================================================================

def find_best_match(score_map, method="ssd"):

    if method.lower() in ("ssd", "mse"):

        index = np.argmin(
            score_map
        )

    elif method.lower() == "ncc":

        index = np.argmax(
            score_map
        )

    else:

        raise ValueError(
            "Method must be 'ssd', 'mse', or 'ncc'."
        )

    y, x = np.unravel_index(
        index,
        score_map.shape
    )

    return int(x), int(y), float(score_map[y, x])


# ====================================================================
# STEP 7 — MATCHING METRICS
# ====================================================================

def localization_error(
    predicted_position,
    ground_truth_position
):

    predicted = np.asarray(
        predicted_position,
        dtype=np.float32
    )

    ground_truth = np.asarray(
        ground_truth_position,
        dtype=np.float32
    )

    return float(
        np.linalg.norm(
            predicted - ground_truth
        )
    )


# ====================================================================
# STEP 8 — CREATE A BRIGHTNESS-CHANGED IMAGE
# ====================================================================

def change_brightness_contrast(
    image,
    contrast=1.0,
    brightness=0.0
):

    changed = (
        image * contrast
        +
        brightness
    )

    return np.clip(
        changed,
        0,
        255
    ).astype(np.float32)


# ====================================================================
# STEP 9 — TESTS
# ====================================================================

def run_tests():

    print("\nRUNNING TESTS")
    print("-" * 70)

    # ---------------------------------------------------------------
    # Test 1: SSD identity
    # ---------------------------------------------------------------

    template = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ],
        dtype=np.float32
    )

    score_map = ssd_matching(
        template,
        template
    )

    assert score_map.shape == (1, 1)
    assert np.isclose(score_map[0, 0], 0.0)

    print("SSD identity test passed.")

    # ---------------------------------------------------------------
    # Test 2: MSE identity
    # ---------------------------------------------------------------

    mse_map = mse_matching(
        template,
        template
    )

    assert np.isclose(
        mse_map[0, 0],
        0.0
    )

    print("MSE identity test passed.")

    # ---------------------------------------------------------------
    # Test 3: NCC identity
    # ---------------------------------------------------------------

    ncc_map = ncc_matching(
        template,
        template
    )

    assert np.isclose(
        ncc_map[0, 0],
        1.0
    )

    print("NCC identity test passed.")

    # ---------------------------------------------------------------
    # Test 4: Exact location
    # ---------------------------------------------------------------

    test_image = np.zeros(
        (20, 25),
        dtype=np.float32
    )

    test_template = np.array(
        [
            [10, 20, 30],
            [40, 80, 60],
            [70, 90, 50]
        ],
        dtype=np.float32
    )

    expected_x = 13
    expected_y = 7

    test_image[
        expected_y:expected_y + 3,
        expected_x:expected_x + 3
    ] = test_template

    test_map = ssd_matching(
        test_image,
        test_template
    )

    found_x, found_y, found_score = find_best_match(
        test_map,
        method="ssd"
    )

    assert found_x == expected_x
    assert found_y == expected_y
    assert np.isclose(found_score, 0.0)

    print("Exact localization test passed.")

    # ---------------------------------------------------------------
    # Test 5: NCC brightness robustness
    # ---------------------------------------------------------------

    changed_template = (
        test_template * 1.5 + 20
    )

    ncc_changed = ncc_matching(
        changed_template,
        test_template
    )

    assert np.isclose(
        ncc_changed[0, 0],
        1.0,
        atol=1e-6
    )

    print("NCC brightness robustness test passed.")

    # ---------------------------------------------------------------
    # Test 6: Output shape
    # ---------------------------------------------------------------

    large_image = np.zeros(
        (30, 40),
        dtype=np.float32
    )

    small_template = np.zeros(
        (5, 7),
        dtype=np.float32
    )

    result = ssd_matching(
        large_image,
        small_template
    )

    assert result.shape == (26, 34)

    print("Output shape test passed.")

    # ---------------------------------------------------------------
    # Test 7: Constant-template NCC
    # ---------------------------------------------------------------

    constant = np.ones(
        (4, 4),
        dtype=np.float32
    )

    constant_result = ncc_matching(
        large_image,
        constant
    )

    assert np.all(
        constant_result == -1.0
    )

    print("Constant-template NCC test passed.")

    print("-" * 70)
    print("ALL TESTS PASSED!")


# ====================================================================
# STEP 10 — MAIN EXPERIMENT
# ====================================================================

def main():

    print("=" * 70)
    print("COMPUTER VISION FROM SCRATCH")
    print("LESSON 12 — TEMPLATE MATCHING")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Create scene and template
    # ---------------------------------------------------------------

    image, template, ground_truth = create_scene()

    target_x, target_y = ground_truth

    print("\nSCENE INFORMATION")
    print("-" * 70)

    print("Image shape:", image.shape)
    print("Template shape:", template.shape)

    print(
        "Ground-truth position:",
        ground_truth
    )

    # ---------------------------------------------------------------
    # Run SSD
    # ---------------------------------------------------------------

    print("\nSSD MATCHING")
    print("-" * 70)

    ssd_map = ssd_matching(
        image,
        template
    )

    ssd_x, ssd_y, ssd_score = find_best_match(
        ssd_map,
        method="ssd"
    )

    ssd_error = localization_error(
        (ssd_x, ssd_y),
        ground_truth
    )

    print("Predicted position:", (ssd_x, ssd_y))
    print("SSD score:", round(ssd_score, 2))
    print("Localization error:", round(ssd_error, 3))

    # ---------------------------------------------------------------
    # Run MSE
    # ---------------------------------------------------------------

    mse_map = mse_matching(
        image,
        template
    )

    mse_x, mse_y, mse_score = find_best_match(
        mse_map,
        method="mse"
    )

    print("\nMSE MATCHING")
    print("-" * 70)

    print("Predicted position:", (mse_x, mse_y))
    print("MSE score:", round(mse_score, 4))

    # ---------------------------------------------------------------
    # Run NCC
    # ---------------------------------------------------------------

    ncc_map = ncc_matching(
        image,
        template
    )

    ncc_x, ncc_y, ncc_score = find_best_match(
        ncc_map,
        method="ncc"
    )

    ncc_error = localization_error(
        (ncc_x, ncc_y),
        ground_truth
    )

    print("\nNCC MATCHING")
    print("-" * 70)

    print("Predicted position:", (ncc_x, ncc_y))
    print("NCC score:", round(ncc_score, 6))
    print("Localization error:", round(ncc_error, 3))

    # ---------------------------------------------------------------
    # Brightness robustness experiment
    # ---------------------------------------------------------------

    changed_image = change_brightness_contrast(
        image,
        contrast=CONTRAST_FACTOR,
        brightness=BRIGHTNESS_OFFSET
    )

    changed_ssd_map = ssd_matching(
        changed_image,
        template
    )

    changed_ncc_map = ncc_matching(
        changed_image,
        template
    )

    changed_ssd_position = find_best_match(
        changed_ssd_map,
        method="ssd"
    )

    changed_ncc_position = find_best_match(
        changed_ncc_map,
        method="ncc"
    )

    print("\nBRIGHTNESS / CONTRAST ROBUSTNESS")
    print("-" * 70)

    print(
        "SSD position after brightness change:",
        changed_ssd_position[:2]
    )

    print(
        "NCC position after brightness change:",
        changed_ncc_position[:2]
    )

    print(
        "SSD score after brightness change:",
        round(changed_ssd_position[2], 2)
    )

    print(
        "NCC score after brightness change:",
        round(changed_ncc_position[2], 6)
    )

    # ---------------------------------------------------------------
    # Add Gaussian noise experiment
    # ---------------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED + 1
    )

    noisy_image = changed_image + rng.normal(
        0,
        NOISE_STD,
        changed_image.shape
    )

    noisy_image = np.clip(
        noisy_image,
        0,
        255
    )

    noisy_ncc_map = ncc_matching(
        noisy_image,
        template
    )

    noisy_x, noisy_y, noisy_score = find_best_match(
        noisy_ncc_map,
        method="ncc"
    )

    print("\nNOISY IMAGE MATCHING")
    print("-" * 70)

    print(
        "Predicted position:",
        (noisy_x, noisy_y)
    )

    print(
        "NCC score:",
        round(noisy_score, 6)
    )

    print(
        "Localization error:",
        round(
            localization_error(
                (noisy_x, noisy_y),
                ground_truth
            ),
            3
        )
    )

    # ---------------------------------------------------------------
    # Visualization 1 — Scene and template
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0].set_title(
        "Source Image"
    )

    axes[1].imshow(
        template,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1].set_title(
        "Template"
    )

    axes[2].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[2].add_patch(
        plt.Rectangle(
            (target_x, target_y),
            TEMPLATE_WIDTH,
            TEMPLATE_HEIGHT,
            fill=False,
            linewidth=2
        )
    )

    axes[2].set_title(
        "Ground-Truth Location"
    )

    for ax in axes:

        ax.axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 2 — SSD and NCC score maps
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    ssd_display = np.log1p(
        ssd_map
    )

    axes[0].imshow(
        ssd_display,
        cmap="viridis"
    )

    axes[0].scatter(
        ssd_x,
        ssd_y,
        marker="x",
        s=100
    )

    axes[0].set_title(
        "SSD Score Map (log scale, lower is better)"
    )

    axes[1].imshow(
        ncc_map,
        cmap="viridis",
        vmin=-1,
        vmax=1
    )

    axes[1].scatter(
        ncc_x,
        ncc_y,
        marker="x",
        s=100
    )

    axes[1].set_title(
        "NCC Similarity Map (higher is better)"
    )

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 3 — Predicted bounding boxes
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0].add_patch(
        plt.Rectangle(
            (ssd_x, ssd_y),
            TEMPLATE_WIDTH,
            TEMPLATE_HEIGHT,
            fill=False,
            linewidth=2
        )
    )

    axes[0].set_title(
        f"SSD Prediction: ({ssd_x}, {ssd_y})"
    )

    axes[1].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1].add_patch(
        plt.Rectangle(
            (ncc_x, ncc_y),
            TEMPLATE_WIDTH,
            TEMPLATE_HEIGHT,
            fill=False,
            linewidth=2
        )
    )

    axes[1].set_title(
        f"NCC Prediction: ({ncc_x}, {ncc_y})"
    )

    for ax in axes:

        ax.axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 4 — Brightness robustness
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[1].imshow(
        changed_image,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1].set_title(
        "Brightness + Contrast Changed"
    )

    axes[2].imshow(
        noisy_image,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[2].set_title(
        "Brightness + Contrast + Noise"
    )

    for ax in axes:

        ax.axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 5 — NCC under appearance changes
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5)
    )

    axes[0].imshow(
        changed_ncc_map,
        cmap="viridis",
        vmin=-1,
        vmax=1
    )

    axes[0].scatter(
        changed_ncc_position[0],
        changed_ncc_position[1],
        marker="x",
        s=100
    )

    axes[0].set_title(
        "NCC After Brightness / Contrast Change"
    )

    axes[1].imshow(
        noisy_ncc_map,
        cmap="viridis",
        vmin=-1,
        vmax=1
    )

    axes[1].scatter(
        noisy_x,
        noisy_y,
        marker="x",
        s=100
    )

    axes[1].set_title(
        "NCC After Noise"
    )

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Numerical tests
    # ---------------------------------------------------------------

    run_tests()

    # ---------------------------------------------------------------
    # Robotics interpretation
    # ---------------------------------------------------------------

    print("\nOPENARM V2 — WASTE-SORTING APPLICATION")
    print("-" * 70)

    print(
        "Template matching can locate a known object appearance"
    )

    print(
        "when camera viewpoint, scale, and object appearance"
    )

    print(
        "remain reasonably consistent."
    )

    print("\nExample output:")

    print(
        "  Object bounding box:",
        (
            ncc_x,
            ncc_y,
            ncc_x + TEMPLATE_WIDTH,
            ncc_y + TEMPLATE_HEIGHT
        )
    )

    print(
        "  Object image center:",
        (
            ncc_x + TEMPLATE_WIDTH / 2,
            ncc_y + TEMPLATE_HEIGHT / 2
        )
    )

    print(
        "\nThese image coordinates can be passed to a later"
    )

    print(
        "camera-calibration and robot-localization pipeline."
    )

    print(
        "\nThey are not directly robot-world coordinates."
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("\nLESSON 12 SUMMARY")
    print("-" * 70)

    print("1. Implemented sliding-window template matching.")
    print("2. Implemented SSD.")
    print("3. Implemented MSE.")
    print("4. Implemented normalized cross-correlation.")
    print("5. Generated similarity maps.")
    print("6. Located the best matching position.")
    print("7. Tested brightness and contrast robustness.")
    print("8. Tested matching under image noise.")

    print("\nNEXT LESSON:")
    print("LESSON 13 — FEATURE MATCHING AND DESCRIPTOR-BASED CORRESPONDENCE")

    plt.show()


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()