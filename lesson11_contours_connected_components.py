"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 11 — CONTOURS AND CONNECTED COMPONENTS
======================================================================

COURSE RULE
-----------
This course is implemented from scratch.

We do NOT use:
    - cv2.findContours()
    - cv2.connectedComponents()
    - cv2.connectedComponentsWithStats()
    - pretrained models
    - pretrained weights
    - black-box object detection algorithms

We implement the core algorithms using:
    - NumPy
    - Matplotlib
    - Python collections.deque

----------------------------------------------------------------------
WHAT YOU WILL LEARN
----------------------------------------------------------------------

1. What is a binary image?
2. What is a connected component?
3. Difference between 4-connectivity and 8-connectivity.
4. Connected-component labeling using Breadth-First Search (BFS).
5. Object area calculation.
6. Bounding-box extraction.
7. Object centroid calculation.
8. Boundary and contour extraction.
9. Basic object filtering.
10. How these techniques support robotic object detection.

----------------------------------------------------------------------
1. CONNECTED COMPONENTS
----------------------------------------------------------------------

A binary image contains foreground and background pixels:

    foreground = 1
    background = 0

A connected component is a group of foreground pixels connected
to one another according to a neighborhood definition.

For a pixel at (y, x), its 4-connected neighbors are:

                  (y-1, x)

    (y, x-1)      (y, x)      (y, x+1)

                  (y+1, x)

8-connectivity additionally includes diagonal neighbors:

    (y-1,x-1)  (y-1,x)  (y-1,x+1)

    (y,  x-1)  (y,  x)  (y,  x+1)

    (y+1,x-1)  (y+1,x)  (y+1,x+1)

4-connectivity:
    Pixels connect only horizontally and vertically.

8-connectivity:
    Pixels also connect diagonally.

----------------------------------------------------------------------
2. CONNECTED-COMPONENT LABELING
----------------------------------------------------------------------

We assign a unique integer label to every connected component.

    0 = background
    1 = first object
    2 = second object
    3 = third object
    ...

Algorithm:

    1. Scan every pixel.
    2. If the pixel is foreground and unlabeled:
       - Create a new component label.
       - Start a BFS queue.
    3. Visit connected foreground neighbors.
    4. Assign the same label to all pixels in that component.
    5. Continue until every pixel has been examined.

BFS uses a queue:

    queue.append(start_pixel)

    while queue:
        current = queue.popleft()

        for neighbor in neighbors(current):
            if neighbor is foreground and unlabeled:
                label[neighbor] = current_label
                queue.append(neighbor)

----------------------------------------------------------------------
3. OBJECT AREA
----------------------------------------------------------------------

The area of a connected component is its number of foreground pixels.

    Area = number of pixels belonging to the component

For component k:

    A_k = sum(label_image == k)

This is a pixel-area measurement.

To convert to physical area, camera calibration and the distance
between the camera and object are required.

----------------------------------------------------------------------
4. BOUNDING BOX
----------------------------------------------------------------------

A bounding box is the smallest axis-aligned rectangle containing
an object.

For all object pixels:

    x_min = minimum x coordinate
    x_max = maximum x coordinate

    y_min = minimum y coordinate
    y_max = maximum y coordinate

Bounding-box width:

    width = x_max - x_min + 1

Bounding-box height:

    height = y_max - y_min + 1

----------------------------------------------------------------------
5. OBJECT CENTROID
----------------------------------------------------------------------

The centroid is the average position of all object pixels.

    x_center = (1/N) * sum(x_i)

    y_center = (1/N) * sum(y_i)

where N is the number of foreground pixels.

The centroid gives a simple estimate of the object's image location.

It is NOT necessarily the object's center of mass or grasp point.

----------------------------------------------------------------------
6. CONTOURS AND BOUNDARIES
----------------------------------------------------------------------

A contour represents the boundary of an object.

A simple boundary pixel is a foreground pixel that has at least
one background pixel in its 8-neighborhood.

Mathematically:

    B(x,y) = 1 if:

        I(x,y) = 1

    and at least one neighboring pixel is 0.

Otherwise:

    B(x,y) = 0

This lesson extracts boundary pixels directly.

IMPORTANT:
    The extracted boundary is an unordered set of boundary pixels.
    It is useful for visualization and boundary analysis.

    It is not yet a topologically ordered polygon contour.

A later advanced implementation can use boundary-following
algorithms such as Moore-neighbor tracing.

----------------------------------------------------------------------
7. OBJECT FILTERING
----------------------------------------------------------------------

Not every connected component is a real object.

Small components may be:
    - Image noise
    - Dust
    - Isolated segmentation errors
    - Small reflections

We can filter components using:

    minimum_area <= object_area <= maximum_area

Other useful filters include:
    - Bounding-box width
    - Bounding-box height
    - Aspect ratio
    - Solidity
    - Circularity

----------------------------------------------------------------------
REAL-WORLD ROBOTICS EXAMPLE
----------------------------------------------------------------------

OPENARM V2 — WASTE-SORTING ROBOT

A simple classical vision pipeline:

    RGB Camera
        |
        v
    Grayscale Conversion
        |
        v
    Thresholding
        |
        v
    Binary Segmentation
        |
        v
    Connected Components
        |
        v
    Object Area and Bounding Box
        |
        v
    Object Centroid
        |
        v
    Candidate Object Regions
        |
        v
    Grasp Planning / ACT Policy
        |
        v
    OpenArm V2

Example:

    Object 1:
        centroid = (x1, y1)
        area = A1
        bounding box = B1

    Object 2:
        centroid = (x2, y2)
        area = A2
        bounding box = B2

These measurements can help a robot identify candidate objects
before running a more advanced perception or manipulation model.

A 2-D centroid alone is NOT a 3-D grasp position.

For robotic grasping, additional information is needed:
    - Camera calibration
    - Depth information
    - Camera-to-robot transformation
    - Object geometry
    - Gripper constraints

----------------------------------------------------------------------
EXERCISES
----------------------------------------------------------------------

1. Change connectivity from 8 to 4.
   Observe how diagonal pixels are treated.

2. Change MIN_OBJECT_AREA.
   Observe which objects disappear.

3. Add more synthetic objects.
   Compare their area and bounding boxes.

4. Add two touching objects.
   Observe how connected-component labeling merges them.

5. Try separating touching objects using morphology or watershed
   segmentation in a future lesson.

----------------------------------------------------------------------
NEXT LESSON
----------------------------------------------------------------------

LESSON 12 — TEMPLATE MATCHING FROM SCRATCH

You will learn how to locate a known image pattern inside a larger
image using sliding windows, SSD, normalized cross-correlation,
and similarity maps.

======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from collections import deque


# ====================================================================
# CONFIGURATION
# ====================================================================

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 360

CONNECTIVITY = 8

MIN_OBJECT_AREA = 80
MAX_OBJECT_AREA = 10000

RANDOM_SEED = 42


# ====================================================================
# STEP 1 — CREATE A SYNTHETIC GRAYSCALE IMAGE
# ====================================================================

def create_synthetic_scene():

    image = np.full(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        25,
        dtype=np.float32
    )

    # ---------------------------------------------------------------
    # Object 1: rectangle
    # ---------------------------------------------------------------

    image[35:85, 35:100] = 220

    # ---------------------------------------------------------------
    # Object 2: circle
    # ---------------------------------------------------------------

    yy, xx = np.mgrid[
        0:IMAGE_HEIGHT,
        0:IMAGE_WIDTH
    ]

    circle = (
        (xx - 165) ** 2
        +
        (yy - 65) ** 2
    ) <= 24 ** 2

    image[circle] = 190

    # ---------------------------------------------------------------
    # Object 3: triangle
    # ---------------------------------------------------------------

    triangle = (
        (yy >= 115)
        &
        (yy <= 185)
        &
        (xx >= 40)
        &
        (xx <= 115)
        &
        (xx >= 40 + (yy - 115) * 0.35)
        &
        (xx <= 115 - (yy - 115) * 0.35)
    )

    image[triangle] = 240

    # ---------------------------------------------------------------
    # Object 4: small rectangle
    # ---------------------------------------------------------------

    image[125:160, 160:205] = 150

    # ---------------------------------------------------------------
    # Object 5: irregular polygon-like region
    # ---------------------------------------------------------------

    irregular = (
        (xx >= 250)
        &
        (xx <= 320)
        &
        (yy >= 120)
        &
        (yy <= 185)
        &
        (
            (xx - 285) ** 2 / 35 ** 2
            +
            (yy - 152) ** 2 / 32 ** 2
            <= 1.0
        )
    )

    image[irregular] = 205

    # ---------------------------------------------------------------
    # Object 6: small square
    # ---------------------------------------------------------------

    image[25:43, 270:288] = 175

    # ---------------------------------------------------------------
    # Add Gaussian noise
    # ---------------------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    noise = rng.normal(
        loc=0.0,
        scale=3.0,
        size=image.shape
    )

    image = image + noise

    image = np.clip(
        image,
        0,
        255
    )

    return image.astype(np.uint8)


# ====================================================================
# STEP 2 — THRESHOLDING
# ====================================================================

def threshold_image(image, threshold):

    binary = image > threshold

    return binary.astype(np.uint8)


# ====================================================================
# STEP 3 — DEFINE NEIGHBORHOODS
# ====================================================================

def get_neighbors(y, x, height, width, connectivity=8):

    if connectivity == 4:

        offsets = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

    elif connectivity == 8:

        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),

            (0, -1),
            (0, 1),

            (1, -1),
            (1, 0),
            (1, 1)
        ]

    else:

        raise ValueError(
            "Connectivity must be 4 or 8."
        )

    neighbors = []

    for dy, dx in offsets:

        ny = y + dy
        nx = x + dx

        if (
            0 <= ny < height
            and
            0 <= nx < width
        ):

            neighbors.append(
                (ny, nx)
            )

    return neighbors


# ====================================================================
# STEP 4 — CONNECTED COMPONENT LABELING USING BFS
# ====================================================================

def connected_components(
    binary_image,
    connectivity=8
):

    """
    Label every connected foreground component.

    Input:
        binary_image:
            2-D binary image containing 0 and 1.

        connectivity:
            4 or 8.

    Returns:
        label_image:
            Integer image. Zero is background.

        components:
            List containing component statistics.
    """

    height, width = binary_image.shape

    labels = np.zeros(
        (height, width),
        dtype=np.int32
    )

    current_label = 0

    components = []

    # ---------------------------------------------------------------
    # Scan the entire image
    # ---------------------------------------------------------------

    for y in range(height):

        for x in range(width):

            # Ignore background and previously labeled pixels.

            if binary_image[y, x] == 0:
                continue

            if labels[y, x] != 0:
                continue

            # -------------------------------------------------------
            # A new component has been discovered.
            # -------------------------------------------------------

            current_label += 1

            queue = deque()

            queue.append(
                (y, x)
            )

            labels[y, x] = current_label

            pixel_coordinates = []

            # -------------------------------------------------------
            # Breadth-First Search
            # -------------------------------------------------------

            while queue:

                cy, cx = queue.popleft()

                pixel_coordinates.append(
                    (cy, cx)
                )

                neighbors = get_neighbors(
                    cy,
                    cx,
                    height,
                    width,
                    connectivity
                )

                for ny, nx in neighbors:

                    if binary_image[ny, nx] == 0:
                        continue

                    if labels[ny, nx] != 0:
                        continue

                    labels[ny, nx] = current_label

                    queue.append(
                        (ny, nx)
                    )

            # -------------------------------------------------------
            # Calculate component statistics
            # -------------------------------------------------------

            coordinates = np.asarray(
                pixel_coordinates,
                dtype=np.int32
            )

            ys = coordinates[:, 0]
            xs = coordinates[:, 1]

            area = len(coordinates)

            x_min = int(np.min(xs))
            x_max = int(np.max(xs))

            y_min = int(np.min(ys))
            y_max = int(np.max(ys))

            centroid_x = float(np.mean(xs))
            centroid_y = float(np.mean(ys))

            bbox_width = x_max - x_min + 1
            bbox_height = y_max - y_min + 1

            aspect_ratio = (
                bbox_width / max(bbox_height, 1)
            )

            components.append(
                {
                    "label": current_label,
                    "area": area,

                    "x_min": x_min,
                    "y_min": y_min,

                    "x_max": x_max,
                    "y_max": y_max,

                    "width": bbox_width,
                    "height": bbox_height,

                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,

                    "aspect_ratio": aspect_ratio,

                    "coordinates": coordinates
                }
            )

    return labels, components


# ====================================================================
# STEP 5 — EXTRACT BOUNDARY PIXELS
# ====================================================================

def extract_boundary(binary_image):

    """
    Extract boundary pixels using 8-neighborhood inspection.

    A foreground pixel is a boundary pixel if at least one
    neighboring pixel is background.

    This returns a boundary mask, not an ordered polygon.
    """

    binary = binary_image.astype(bool)

    padded = np.pad(
        binary,
        pad_width=1,
        mode="constant",
        constant_values=False
    )

    height, width = binary.shape

    neighbor_background = np.zeros(
        (height, width),
        dtype=bool
    )

    # ---------------------------------------------------------------
    # Examine all 8 neighboring positions.
    # ---------------------------------------------------------------

    for dy in range(-1, 2):

        for dx in range(-1, 2):

            if dy == 0 and dx == 0:
                continue

            neighbor = padded[
                1 + dy:1 + dy + height,
                1 + dx:1 + dx + width
            ]

            neighbor_background |= ~neighbor

    boundary = binary & neighbor_background

    return boundary.astype(np.uint8)


# ====================================================================
# STEP 6 — FILTER COMPONENTS
# ====================================================================

def filter_components(
    components,
    min_area=MIN_OBJECT_AREA,
    max_area=MAX_OBJECT_AREA
):

    filtered = []

    for component in components:

        if component["area"] < min_area:
            continue

        if component["area"] > max_area:
            continue

        filtered.append(component)

    return filtered


# ====================================================================
# STEP 7 — CREATE A COLOR LABEL IMAGE
# ====================================================================

def colorize_labels(label_image):

    height, width = label_image.shape

    rng = np.random.default_rng(123)

    number_of_labels = int(
        np.max(label_image)
    )

    colors = rng.integers(
        low=40,
        high=256,
        size=(number_of_labels + 1, 3),
        dtype=np.uint8
    )

    colors[0] = [0, 0, 0]

    color_image = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for label_id in range(
        1,
        number_of_labels + 1
    ):

        color_image[
            label_image == label_id
        ] = colors[label_id]

    return color_image


# ====================================================================
# STEP 8 — PRINT COMPONENT STATISTICS
# ====================================================================

def print_component_statistics(components):

    print("\nCONNECTED COMPONENT STATISTICS")
    print("-" * 90)

    header = (
        f"{'ID':>4}"
        f"{'AREA':>10}"
        f"{'CENTER X':>12}"
        f"{'CENTER Y':>12}"
        f"{'WIDTH':>10}"
        f"{'HEIGHT':>10}"
        f"{'ASPECT':>10}"
    )

    print(header)
    print("-" * 90)

    for component in components:

        print(
            f"{component['label']:>4}"
            f"{component['area']:>10}"
            f"{component['centroid_x']:>12.2f}"
            f"{component['centroid_y']:>12.2f}"
            f"{component['width']:>10}"
            f"{component['height']:>10}"
            f"{component['aspect_ratio']:>10.2f}"
        )

    print("-" * 90)


# ====================================================================
# STEP 9 — TEST CONNECTED COMPONENTS
# ====================================================================

def run_tests():

    print("\nRUNNING TESTS")
    print("-" * 70)

    # ---------------------------------------------------------------
    # Test 1: One connected component
    # ---------------------------------------------------------------

    test_image = np.zeros(
        (10, 10),
        dtype=np.uint8
    )

    test_image[2:5, 3:7] = 1

    labels, components = connected_components(
        test_image,
        connectivity=4
    )

    assert len(components) == 1
    assert components[0]["area"] == 12

    print("Single-component test passed.")

    # ---------------------------------------------------------------
    # Test 2: Two separate components
    # ---------------------------------------------------------------

    test_image = np.zeros(
        (12, 12),
        dtype=np.uint8
    )

    test_image[1:3, 1:3] = 1
    test_image[8:10, 8:10] = 1

    labels, components = connected_components(
        test_image,
        connectivity=4
    )

    assert len(components) == 2

    print("Two-component test passed.")

    # ---------------------------------------------------------------
    # Test 3: 4-connectivity versus 8-connectivity
    # ---------------------------------------------------------------

    diagonal_image = np.zeros(
        (5, 5),
        dtype=np.uint8
    )

    diagonal_image[1, 1] = 1
    diagonal_image[2, 2] = 1

    _, components_4 = connected_components(
        diagonal_image,
        connectivity=4
    )

    _, components_8 = connected_components(
        diagonal_image,
        connectivity=8
    )

    assert len(components_4) == 2
    assert len(components_8) == 1

    print("4-connectivity / 8-connectivity test passed.")

    # ---------------------------------------------------------------
    # Test 4: Bounding box
    # ---------------------------------------------------------------

    test_image = np.zeros(
        (10, 10),
        dtype=np.uint8
    )

    test_image[2:6, 3:8] = 1

    _, components = connected_components(
        test_image,
        connectivity=8
    )

    component = components[0]

    assert component["x_min"] == 3
    assert component["x_max"] == 7

    assert component["y_min"] == 2
    assert component["y_max"] == 5

    assert component["width"] == 5
    assert component["height"] == 4

    print("Bounding-box test passed.")

    # ---------------------------------------------------------------
    # Test 5: Centroid
    # ---------------------------------------------------------------

    expected_x = (3 + 7) / 2
    expected_y = (2 + 5) / 2

    assert np.isclose(
        component["centroid_x"],
        expected_x
    )

    assert np.isclose(
        component["centroid_y"],
        expected_y
    )

    print("Centroid test passed.")

    # ---------------------------------------------------------------
    # Test 6: Boundary extraction
    # ---------------------------------------------------------------

    test_image = np.zeros(
        (9, 9),
        dtype=np.uint8
    )

    test_image[2:7, 2:7] = 1

    boundary = extract_boundary(test_image)

    assert boundary.shape == test_image.shape

    assert np.all(
        boundary[test_image == 0] == 0
    )

    assert np.all(
        boundary[2, 2:7] == 1
    )

    assert np.all(
        boundary[6, 2:7] == 1
    )

    assert np.all(
        boundary[2:7, 2] == 1
    )

    assert np.all(
        boundary[2:7, 6] == 1
    )

    # The center of the square is not a boundary pixel.

    assert boundary[4, 4] == 0

    print("Boundary extraction test passed.")

    # ---------------------------------------------------------------
    # Test 7: Component filtering
    # ---------------------------------------------------------------

    test_components = [
        {"area": 10},
        {"area": 100},
        {"area": 500}
    ]

    filtered = filter_components(
        test_components,
        min_area=50,
        max_area=200
    )

    assert len(filtered) == 1
    assert filtered[0]["area"] == 100

    print("Component filtering test passed.")

    print("-" * 70)
    print("ALL TESTS PASSED!")


# ====================================================================
# STEP 10 — MAIN EXPERIMENT
# ====================================================================

def main():

    print("=" * 70)
    print("COMPUTER VISION FROM SCRATCH")
    print("LESSON 11 — CONTOURS AND CONNECTED COMPONENTS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Create image
    # ---------------------------------------------------------------

    image = create_synthetic_scene()

    print("\nIMAGE INFORMATION")
    print("-" * 70)

    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)
    print("Minimum intensity:", image.min())
    print("Maximum intensity:", image.max())
    print("Mean intensity:", round(float(image.mean()), 2))

    # ---------------------------------------------------------------
    # Threshold image
    # ---------------------------------------------------------------

    threshold = 100

    binary = threshold_image(
        image,
        threshold
    )

    print("\nSEGMENTATION")
    print("-" * 70)

    print("Threshold:", threshold)
    print("Foreground pixels:", int(np.sum(binary)))
    print(
        "Foreground ratio:",
        round(float(np.mean(binary)), 4)
    )

    # ---------------------------------------------------------------
    # Connected components
    # ---------------------------------------------------------------

    labels, components = connected_components(
        binary,
        connectivity=CONNECTIVITY
    )

    print("\nCONNECTED COMPONENTS")
    print("-" * 70)

    print("Connectivity:", CONNECTIVITY)
    print("Total components:", len(components))

    print_component_statistics(
        components
    )

    # ---------------------------------------------------------------
    # Filter small components
    # ---------------------------------------------------------------

    filtered_components = filter_components(
        components,
        min_area=MIN_OBJECT_AREA,
        max_area=MAX_OBJECT_AREA
    )

    print("\nFILTERING")
    print("-" * 70)

    print("Minimum area:", MIN_OBJECT_AREA)
    print("Maximum area:", MAX_OBJECT_AREA)

    print(
        "Components before filtering:",
        len(components)
    )

    print(
        "Components after filtering:",
        len(filtered_components)
    )

    # ---------------------------------------------------------------
    # Extract boundaries
    # ---------------------------------------------------------------

    boundary = extract_boundary(
        binary
    )

    print("\nBOUNDARY INFORMATION")
    print("-" * 70)

    print(
        "Boundary pixels:",
        int(np.sum(boundary))
    )

    print(
        "Boundary ratio:",
        round(float(np.mean(boundary)), 4)
    )

    # ---------------------------------------------------------------
    # Build color label image
    # ---------------------------------------------------------------

    color_labels = colorize_labels(
        labels
    )

    # ---------------------------------------------------------------
    # Build filtered label image
    # ---------------------------------------------------------------

    filtered_mask = np.zeros_like(
        binary
    )

    for component in filtered_components:

        filtered_mask[
            labels == component["label"]
        ] = 1

    filtered_boundary = extract_boundary(
        filtered_mask
    )

    # ---------------------------------------------------------------
    # Visualization 1 — Main segmentation pipeline
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 10)
    )

    axes[0, 0].imshow(
        image,
        cmap="gray"
    )

    axes[0, 0].set_title(
        "1. Original Grayscale Image"
    )

    axes[0, 1].imshow(
        binary,
        cmap="gray"
    )

    axes[0, 1].set_title(
        f"2. Binary Segmentation (T={threshold})"
    )

    axes[0, 2].imshow(
        color_labels
    )

    axes[0, 2].set_title(
        "3. Connected-Component Labels"
    )

    axes[1, 0].imshow(
        boundary,
        cmap="gray"
    )

    axes[1, 0].set_title(
        "4. Extracted Boundary Pixels"
    )

    axes[1, 1].imshow(
        filtered_mask,
        cmap="gray"
    )

    axes[1, 1].set_title(
        "5. Filtered Components"
    )

    axes[1, 2].imshow(
        filtered_boundary,
        cmap="gray"
    )

    axes[1, 2].set_title(
        "6. Boundaries After Filtering"
    )

    for ax in axes.flat:

        ax.axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 2 — Bounding boxes and centroids
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    ax.imshow(
        image,
        cmap="gray"
    )

    ax.set_title(
        "Connected Components — Bounding Boxes and Centroids"
    )

    for component in filtered_components:

        x_min = component["x_min"]
        y_min = component["y_min"]

        width = component["width"]
        height = component["height"]

        center_x = component["centroid_x"]
        center_y = component["centroid_y"]

        rectangle_x = [
            x_min,
            x_min + width - 1,
            x_min + width - 1,
            x_min,
            x_min
        ]

        rectangle_y = [
            y_min,
            y_min,
            y_min + height - 1,
            y_min + height - 1,
            y_min
        ]

        ax.plot(
            rectangle_x,
            rectangle_y,
            linewidth=2
        )

        ax.scatter(
            center_x,
            center_y,
            marker="x",
            s=100
        )

        ax.text(
            x_min,
            max(y_min - 5, 0),
            f"ID {component['label']} | A={component['area']}",
            fontsize=8,
            color="yellow",
            bbox=dict(
                facecolor="black",
                alpha=0.6
            )
        )

    ax.set_xlim(
        0,
        IMAGE_WIDTH
    )

    ax.set_ylim(
        IMAGE_HEIGHT,
        0
    )

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    ax.grid(
        alpha=0.2
    )

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 3 — Compare 4-connectivity and 8-connectivity
    # ---------------------------------------------------------------

    diagonal_test = np.zeros(
        (12, 12),
        dtype=np.uint8
    )

    for i in range(2, 9):

        diagonal_test[i, i] = 1

    labels_4, components_4 = connected_components(
        diagonal_test,
        connectivity=4
    )

    labels_8, components_8 = connected_components(
        diagonal_test,
        connectivity=8
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4)
    )

    axes[0].imshow(
        diagonal_test,
        cmap="gray",
        interpolation="nearest"
    )

    axes[0].set_title(
        "Diagonal Pixels"
    )

    axes[1].imshow(
        labels_4,
        cmap="tab10",
        interpolation="nearest"
    )

    axes[1].set_title(
        f"4-Connectivity: {len(components_4)} Components"
    )

    axes[2].imshow(
        labels_8,
        cmap="tab10",
        interpolation="nearest"
    )

    axes[2].set_title(
        f"8-Connectivity: {len(components_8)} Component"
    )

    for ax in axes:

        ax.set_xticks(
            range(12)
        )

        ax.set_yticks(
            range(12)
        )

        ax.grid(
            alpha=0.3
        )

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 4 — Component area comparison
    # ---------------------------------------------------------------

    if len(components) > 0:

        component_ids = [
            component["label"]
            for component in components
        ]

        areas = [
            component["area"]
            for component in components
        ]

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.bar(
            component_ids,
            areas
        )

        ax.set_title(
            "Connected Component Area Comparison"
        )

        ax.set_xlabel(
            "Component ID"
        )

        ax.set_ylabel(
            "Area (pixels)"
        )

        ax.grid(
            axis="y",
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

    print("\nOPENARM V2 — WASTE SORTING INTERPRETATION")
    print("-" * 70)

    print(
        "A classical vision system can extract candidate object regions."
    )

    print(
        "Each region provides:"
    )

    print(
        "  - Component ID"
    )

    print(
        "  - Pixel area"
    )

    print(
        "  - Bounding box"
    )

    print(
        "  - Image centroid"
    )

    print(
        "  - Boundary pixels"
    )

    print(
        "\nThese features can support object localization before"
    )

    print(
        "a downstream ACT policy or grasp-planning system."
    )

    print(
        "\nRemember: image coordinates are not robot coordinates."
    )

    print(
        "Camera calibration and depth/geometry are needed for 3-D grasping."
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("\nLESSON 11 SUMMARY")
    print("-" * 70)

    print("1. Implemented 4-connectivity and 8-connectivity.")
    print("2. Implemented BFS connected-component labeling.")
    print("3. Calculated component area.")
    print("4. Calculated bounding boxes.")
    print("5. Calculated object centroids.")
    print("6. Extracted boundary pixels.")
    print("7. Filtered components by area.")
    print("8. Visualized object regions and statistics.")

    print("\nNEXT LESSON:")
    print("LESSON 12 — TEMPLATE MATCHING FROM SCRATCH")

    plt.show()


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    main()