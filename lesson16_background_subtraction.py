"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 16 — BACKGROUND SUBTRACTION
======================================================================

COURSE APPROACH
---------------
We now combine two approaches:

1. FROM SCRATCH:
   Understand and implement the fundamental mathematics.

2. OPENCV:
   Learn practical computer-vision techniques used in production.

We will use:
    - NumPy
    - OpenCV (cv2)
    - Matplotlib

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1 — Background subtraction fundamentals
STEP 2 — Frame differencing
STEP 3 — Binary foreground segmentation
STEP 4 — Running-average background modeling
STEP 5 — Background adaptation
STEP 6 — OpenCV MOG2
STEP 7 — OpenCV KNN
STEP 8 — Morphological noise removal
STEP 9 — Contour extraction
STEP 10 — Bounding boxes and object centers
STEP 11 — Real-time camera processing
STEP 12 — Numerical tests

======================================================================
1. WHAT IS BACKGROUND SUBTRACTION?
======================================================================

Background subtraction separates moving foreground objects
from a relatively stationary scene.

Example:

    Frame 1:
        Empty conveyor belt.

    Frame 2:
        Empty conveyor belt.

    Frame 3:
        A plastic bottle appears.

    Frame 4:
        The bottle moves across the conveyor.

We want to distinguish:

    Background:
        Conveyor, walls, stationary equipment.

    Foreground:
        Moving bottle, cup, paper, waste object.

The basic mathematical operation is:

    D(x,y) = |I(x,y) - B(x,y)|

Where:

    I = current image
    B = background image
    D = absolute difference

Threshold:

    M(x,y) = 1 if D(x,y) > T
             0 otherwise

M is the foreground mask.

======================================================================
2. FRAME DIFFERENCING
======================================================================

Compare two consecutive frames:

    D(x,y) = |I_t(x,y) - I_(t-1)(x,y)|

Advantages:
    - Very simple.
    - Fast.
    - No background model required.

Disadvantages:
    - Often detects only changing object boundaries.
    - Stationary objects disappear from the difference image.
    - Camera motion causes widespread false detections.

======================================================================
3. RUNNING-AVERAGE BACKGROUND MODEL
======================================================================

Instead of storing a fixed background image, update it over time.

    B_t = (1-alpha)*B_(t-1) + alpha*I_t

Where:

    alpha = learning rate

Small alpha:
    Background changes slowly.

Large alpha:
    Background adapts quickly.

Foreground mask:

    M_t = |I_t - B_(t-1)| > threshold

Important:
    Detect foreground against the previous background before
    updating the background model.

Otherwise, a moving object may immediately be absorbed
into the background.

======================================================================
4. OPENCV MOG2
======================================================================

OpenCV provides:

    cv2.createBackgroundSubtractorMOG2()

MOG2 uses a Gaussian-mixture background model.

It can model multiple background appearances at a pixel.

Useful when:
    - Lighting changes.
    - Shadows appear.
    - Background pixels have multiple recurring appearances.

Important parameters:

    history:
        Number of previous frames used by the model.

    varThreshold:
        Threshold controlling foreground classification.

    detectShadows:
        Whether shadow detection is enabled.

Shadow pixels are generally represented by value 127.
Foreground pixels are generally represented by value 255.

======================================================================
5. OPENCV KNN
======================================================================

OpenCV also provides:

    cv2.createBackgroundSubtractorKNN()

KNN models background using samples from previous observations.

It is another practical option for dynamic backgrounds.

MOG2 and KNN have different assumptions and computational
characteristics. Neither is universally better.

======================================================================
6. MORPHOLOGICAL CLEANUP
======================================================================

Raw foreground masks frequently contain noise.

We use:

    cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

Opening:
    Erosion followed by dilation.
    Removes small isolated foreground noise.

    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

Closing:
    Dilation followed by erosion.
    Fills small holes and connects nearby regions.

======================================================================
7. CONTOUR EXTRACTION
======================================================================

OpenCV:

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

For each contour:

    cv2.boundingRect(contour)

    cv2.contourArea(contour)

Bounding box:

    x, y, width, height

Object center:

    center_x = x + width/2
    center_y = y + height/2

This provides a basic image-space object location.

======================================================================
8. REAL-WORLD OPENARM V2 APPLICATION
======================================================================

WASTE-SORTING PIPELINE

    Camera
       |
       v
    RGB frame
       |
       v
    Background subtraction
       |
       v
    Foreground mask
       |
       v
    Morphological cleanup
       |
       v
    Contour extraction
       |
       v
    Bounding boxes and object centers
       |
       v
    Object classification
       |
       v
    ACT policy
       |
       v
    OpenArm V2 grasping

IMPORTANT:

Background subtraction is NOT an object classifier.

It cannot reliably tell whether an object is:

    - Plastic bottle
    - Aluminum can
    - Paper cup
    - Food waste

It identifies foreground regions.

A separate classification or detection model is needed
to identify waste categories.

Also, background subtraction alone is insufficient when
the camera or robot moves significantly.

======================================================================
9. PRACTICAL CAMERA MODE
======================================================================

Run:

    python lesson16_background_subtraction.py --camera

The script opens your default camera.

Press Q to quit.

For a real conveyor system, use a fixed camera and ensure
the background model has enough time to learn the empty scene.

======================================================================
EXERCISES
======================================================================

1. Change the foreground threshold.
2. Change the running-average learning rate.
3. Compare MOG2 and KNN.
4. Disable shadow detection.
5. Change morphological kernel size.
6. Test with a real webcam.
7. Test moving objects with different sizes.
8. Test a moving camera and observe false detections.

======================================================================
NEXT LESSON
======================================================================

LESSON 17 — CAMERA CALIBRATION FROM SCRATCH + OPENCV

We will learn camera intrinsics, distortion coefficients,
chessboard calibration, and pixel-to-camera geometry.
======================================================================
"""

import argparse
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ====================================================================
# CONFIGURATION
# ====================================================================

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 400

FOREGROUND_THRESHOLD = 25

RUNNING_AVERAGE_ALPHA = 0.02

MIN_CONTOUR_AREA = 150

MORPH_KERNEL_SIZE = 5

RANDOM_SEED = 42


# ====================================================================
# STEP 1 — SYNTHETIC BACKGROUND
# ====================================================================

def create_background():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    background = np.zeros(
        (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        dtype=np.uint8
    )

    # Conveyor-belt background.

    background[:] = (65, 65, 65)

    # Add subtle texture.

    noise = rng.normal(
        0,
        4,
        size=background.shape
    )

    background = np.clip(
        background.astype(np.float32) + noise,
        0,
        255
    ).astype(np.uint8)

    # Conveyor boundaries.

    cv2.line(
        background,
        (0, 70),
        (IMAGE_WIDTH, 70),
        (100, 100, 100),
        3
    )

    cv2.line(
        background,
        (0, 330),
        (IMAGE_WIDTH, 330),
        (100, 100, 100),
        3
    )

    # Stationary equipment.

    cv2.rectangle(
        background,
        (15, 10),
        (90, 55),
        (45, 45, 45),
        -1
    )

    return background


# ====================================================================
# STEP 2 — SYNTHETIC MOVING OBJECTS
# ====================================================================

def create_frame(background, frame_index):

    """
    Create a synthetic conveyor frame.

    Objects move horizontally across the scene.
    """

    frame = background.copy()

    # Moving plastic-bottle-like object.

    bottle_x = 50 + frame_index * 5

    bottle_x %= IMAGE_WIDTH - 70

    cv2.rectangle(
        frame,
        (bottle_x, 145),
        (bottle_x + 35, 220),
        (30, 180, 220),
        -1
    )

    cv2.rectangle(
        frame,
        (bottle_x + 10, 125),
        (bottle_x + 25, 145),
        (30, 180, 220),
        -1
    )

    # Moving can-like object.

    can_x = 400 - frame_index * 3

    can_x %= IMAGE_WIDTH - 45

    cv2.ellipse(
        frame,
        (can_x + 15, 265),
        (18, 12),
        0,
        0,
        360,
        (200, 200, 200),
        -1
    )

    cv2.rectangle(
        frame,
        (can_x - 3, 265),
        (can_x + 33, 295),
        (200, 200, 200),
        -1
    )

    cv2.ellipse(
        frame,
        (can_x + 15, 295),
        (18, 12),
        0,
        0,
        360,
        (160, 160, 160),
        -1
    )

    return frame


# ====================================================================
# STEP 3 — GRAYSCALE CONVERSION
# ====================================================================

def to_grayscale(frame):

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


# ====================================================================
# STEP 4 — FRAME DIFFERENCING FROM SCRATCH
# ====================================================================

def frame_difference(
    previous_frame,
    current_frame,
    threshold=25
):

    """
    Calculate absolute difference between consecutive frames.

        D = |I_t - I_(t-1)|

    Threshold the difference to produce a binary mask.
    """

    difference = cv2.absdiff(
        current_frame,
        previous_frame
    )

    _, mask = cv2.threshold(
        difference,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return (
        difference,
        mask
    )


# ====================================================================
# STEP 5 — RUNNING-AVERAGE BACKGROUND MODEL
# ====================================================================

class RunningAverageBackground:

    """
    Simple background model implemented with NumPy.

        B_t = (1-alpha)*B_(t-1) + alpha*I_t

    Foreground is detected against the previous background.

    This implementation intentionally separates:
        1. Foreground detection.
        2. Background model update.
    """

    def __init__(
        self,
        initial_frame,
        alpha=0.02,
        threshold=25
    ):

        self.background = initial_frame.astype(
            np.float32
        )

        self.alpha = alpha
        self.threshold = threshold

    def apply(self, frame):

        frame_float = frame.astype(
            np.float32
        )

        # Detect foreground against existing model.

        difference = np.abs(
            frame_float - self.background
        )

        difference_gray = np.mean(
            difference,
            axis=2
        )

        mask = (
            difference_gray > self.threshold
        ).astype(
            np.uint8
        ) * 255

        # Update the background after detection.

        self.background = (
            (1 - self.alpha) * self.background
            +
            self.alpha * frame_float
        )

        return mask

    def get_background(self):

        return np.clip(
            self.background,
            0,
            255
        ).astype(np.uint8)


# ====================================================================
# STEP 6 — OPENCV BACKGROUND SUBTRACTION
# ====================================================================

def create_mog2():

    return cv2.createBackgroundSubtractorMOG2(
        history=200,
        varThreshold=25,
        detectShadows=True
    )


def create_knn():

    return cv2.createBackgroundSubtractorKNN(
        history=200,
        dist2Threshold=400,
        detectShadows=True
    )


# ====================================================================
# STEP 7 — MORPHOLOGICAL CLEANUP
# ====================================================================

def clean_foreground_mask(
    mask,
    kernel_size=5
):

    """
    Remove isolated noise and fill small holes.

    OpenCV implementation:

        Opening = erosion -> dilation
        Closing = dilation -> erosion
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel
    )

    return closed


# ====================================================================
# STEP 8 — CONTOUR EXTRACTION
# ====================================================================

def extract_objects(
    mask,
    minimum_area=150
):

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < minimum_area:
            continue

        x, y, width, height = cv2.boundingRect(
            contour
        )

        center_x = x + width / 2.0
        center_y = y + height / 2.0

        objects.append(
            {
                "contour": contour,
                "area": float(area),
                "bbox": (
                    x,
                    y,
                    width,
                    height
                ),
                "center": (
                    center_x,
                    center_y
                )
            }
        )

    return objects


# ====================================================================
# STEP 9 — DRAW DETECTED OBJECTS
# ====================================================================

def draw_objects(frame, objects):

    output = frame.copy()

    for index, obj in enumerate(objects):

        x, y, width, height = obj["bbox"]

        center_x, center_y = obj["center"]

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        cv2.circle(
            output,
            (
                int(center_x),
                int(center_y)
            ),
            4,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            output,
            f"Object {index + 1}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return output


# ====================================================================
# STEP 10 — SYNTHETIC EXPERIMENT
# ====================================================================

def run_synthetic_experiment():

    print("=" * 70)
    print("COMPUTER VISION FROM SCRATCH")
    print("LESSON 16 — BACKGROUND SUBTRACTION")
    print("=" * 70)

    background = create_background()

    # Initialize algorithms.

    running_average = RunningAverageBackground(
        background,
        alpha=RUNNING_AVERAGE_ALPHA,
        threshold=FOREGROUND_THRESHOLD
    )

    mog2 = create_mog2()

    knn = create_knn()

    # Train MOG2 and KNN on an empty background.

    for _ in range(30):

        mog2.apply(
            background,
            learningRate=0.1
        )

        knn.apply(
            background,
            learningRate=0.1
        )

    previous_gray = to_grayscale(
        background
    )

    results = {}

    # Select representative frames.

    selected_indices = [
        1,
        5,
        10,
        20,
        30
    ]

    for frame_index in range(1, 31):

        frame = create_frame(
            background,
            frame_index
        )

        gray = to_grayscale(
            frame
        )

        # -----------------------------------------------------------
        # Frame differencing.
        # -----------------------------------------------------------

        difference, frame_mask = frame_difference(
            previous_gray,
            gray,
            threshold=FOREGROUND_THRESHOLD
        )

        previous_gray = gray.copy()

        # -----------------------------------------------------------
        # Running-average model.
        # -----------------------------------------------------------

        running_mask = running_average.apply(
            frame
        )

        # -----------------------------------------------------------
        # OpenCV MOG2.
        # -----------------------------------------------------------

        mog2_mask = mog2.apply(
            frame,
            learningRate=0
        )

        # -----------------------------------------------------------
        # OpenCV KNN.
        # -----------------------------------------------------------

        knn_mask = knn.apply(
            frame,
            learningRate=0
        )

        if frame_index in selected_indices:

            results[frame_index] = {
                "frame": frame,
                "difference": difference,
                "frame_mask": frame_mask,
                "running_mask": running_mask,
                "mog2_mask": mog2_mask,
                "knn_mask": knn_mask
            }

    # ---------------------------------------------------------------
    # Analyze final selected frame.
    # ---------------------------------------------------------------

    final = results[
        selected_indices[-1]
    ]

    # MOG2 uses 127 for shadow pixels and 255 for foreground.
    # Convert only definite foreground to a binary mask.

    mog2_binary = np.where(
        final["mog2_mask"] == 255,
        255,
        0
    ).astype(np.uint8)

    cleaned_mask = clean_foreground_mask(
        mog2_binary,
        MORPH_KERNEL_SIZE
    )

    objects = extract_objects(
        cleaned_mask,
        MIN_CONTOUR_AREA
    )

    annotated = draw_objects(
        final["frame"],
        objects
    )

    # ---------------------------------------------------------------
    # Console output.
    # ---------------------------------------------------------------

    print("\nDETECTION RESULTS")
    print("-" * 70)

    print(
        "Frame:",
        selected_indices[-1]
    )

    print(
        "Detected foreground objects:",
        len(objects)
    )

    for index, obj in enumerate(objects):

        print(
            f"Object {index + 1}:",
            "Area =", round(obj["area"], 1),
            "Bounding box =", obj["bbox"],
            "Center =", tuple(
                round(value, 1)
                for value in obj["center"]
            )
        )

    # ---------------------------------------------------------------
    # Visualization 1 — Input and background.
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5)
    )

    axes[0].imshow(
        cv2.cvtColor(
            background,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[0].set_title(
        "Background"
    )

    axes[1].imshow(
        cv2.cvtColor(
            final["frame"],
            cv2.COLOR_BGR2RGB
        )
    )

    axes[1].set_title(
        "Current Frame"
    )

    axes[2].imshow(
        cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[2].set_title(
        "Detected Objects"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 2 — Compare algorithms.
    # ---------------------------------------------------------------

    images = [
        ("Frame Difference", final["frame_mask"]),
        ("Running Average", final["running_mask"]),
        ("OpenCV MOG2", final["mog2_mask"]),
        ("OpenCV KNN", final["knn_mask"]),
        ("MOG2 Cleaned", cleaned_mask)
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    for ax, (title, image) in zip(
        axes.flat,
        images
    ):

        ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        ax.set_title(title)
        ax.axis("off")

    axes.flat[-1].axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 3 — Motion sequence.
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(selected_indices),
        figsize=(18, 4)
    )

    for ax, frame_index in zip(
        axes,
        selected_indices
    ):

        frame = results[
            frame_index
        ]["frame"]

        ax.imshow(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
        )

        ax.set_title(
            f"Frame {frame_index}"
        )

        ax.axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Visualization 4 — Background model.
    # ---------------------------------------------------------------

    learned_background = running_average.get_background()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].imshow(
        cv2.cvtColor(
            background,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[0].set_title(
        "Initial Background"
    )

    axes[1].imshow(
        cv2.cvtColor(
            learned_background,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[1].set_title(
        "Running-Average Background"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    # ---------------------------------------------------------------
    # Run numerical tests.
    # ---------------------------------------------------------------

    run_tests()

    print("\nNEXT LESSON:")
    print(
        "LESSON 17 — CAMERA CALIBRATION FROM SCRATCH + OPENCV"
    )

    plt.show()


# ====================================================================
# STEP 11 — REAL-TIME CAMERA MODE
# ====================================================================

def run_camera(camera_index=0):

    print("=" * 70)
    print("REAL-TIME BACKGROUND SUBTRACTION")
    print("=" * 70)

    cap = cv2.VideoCapture(
        camera_index
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open camera {camera_index}."
        )

    mog2 = create_mog2()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE
        )
    )

    print("Camera opened.")
    print("Press Q to quit.")

    frame_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # Background subtraction.

        raw_mask = mog2.apply(
            frame
        )

        # Remove shadow pixels.

        binary_mask = np.where(
            raw_mask == 255,
            255,
            0
        ).astype(np.uint8)

        # Morphological cleanup.

        cleaned = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Object extraction.

        objects = extract_objects(
            cleaned,
            MIN_CONTOUR_AREA
        )

        annotated = draw_objects(
            frame,
            objects
        )

        # Display.

        cv2.imshow(
            "Camera",
            annotated
        )

        cv2.imshow(
            "Foreground Mask",
            cleaned
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()

    print(
        "Processed frames:",
        frame_count
    )


# ====================================================================
# STEP 12 — NUMERICAL TESTS
# ====================================================================

def run_tests():

    print("\nRUNNING TESTS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # TEST 1 — Frame difference
    # ---------------------------------------------------------------

    first = np.zeros(
        (20, 20),
        dtype=np.uint8
    )

    second = first.copy()

    second[5:10, 5:10] = 200

    difference, mask = frame_difference(
        first,
        second,
        threshold=25
    )

    assert np.all(
        mask[5:10, 5:10] == 255
    )

    assert np.all(
        mask[:5, :] == 0
    )

    print("Frame-difference test passed.")

    # ---------------------------------------------------------------
    # TEST 2 — Running-average background initialization
    # ---------------------------------------------------------------

    initial = np.ones(
        (20, 20, 3),
        dtype=np.uint8
    ) * 100

    model = RunningAverageBackground(
        initial,
        alpha=0.1,
        threshold=20
    )

    assert np.allclose(
        model.get_background(),
        initial
    )

    print("Background initialization test passed.")

    # ---------------------------------------------------------------
    # TEST 3 — Foreground detection
    # ---------------------------------------------------------------

    changed = initial.copy()

    changed[5:10, 5:10] = 200

    foreground = model.apply(
        changed
    )

    assert np.all(
        foreground[5:10, 5:10] == 255
    )

    assert np.all(
        foreground[:5, :] == 0
    )

    print("Running-average foreground detection test passed.")

    # ---------------------------------------------------------------
    # TEST 4 — Background adaptation
    # ---------------------------------------------------------------

    fixed_object = np.ones(
        (20, 20, 3),
        dtype=np.uint8
    ) * 200

    initial_background = np.ones(
        (20, 20, 3),
        dtype=np.uint8
    ) * 100

    adaptive_model = RunningAverageBackground(
        initial_background,
        alpha=0.5,
        threshold=20
    )

    first_mask = adaptive_model.apply(
        fixed_object
    )

    assert np.all(
        first_mask == 255
    )

    adaptive_model.apply(
        fixed_object
    )

    adapted_background = adaptive_model.get_background()

    assert np.mean(
        adapted_background
    ) > 170

    print("Background adaptation test passed.")

    # ---------------------------------------------------------------
    # TEST 5 — Morphological cleanup
    # ---------------------------------------------------------------

    noisy_mask = np.zeros(
        (40, 40),
        dtype=np.uint8
    )

    noisy_mask[10:30, 10:30] = 255

    noisy_mask[2, 2] = 255

    cleaned = clean_foreground_mask(
        noisy_mask,
        kernel_size=5
    )

    assert cleaned[20, 20] == 255

    print("Morphological cleanup test passed.")

    # ---------------------------------------------------------------
    # TEST 6 — Contour extraction
    # ---------------------------------------------------------------

    test_mask = np.zeros(
        (100, 100),
        dtype=np.uint8
    )

    cv2.rectangle(
        test_mask,
        (20, 30),
        (60, 70),
        255,
        -1
    )

    objects = extract_objects(
        test_mask,
        minimum_area=100
    )

    assert len(objects) == 1

    x, y, width, height = objects[0]["bbox"]

    assert x == 20
    assert y == 30

    assert width == 41
    assert height == 41

    print("Contour extraction and bounding-box test passed.")

    # ---------------------------------------------------------------
    # TEST 7 — Object center
    # ---------------------------------------------------------------

    center_x, center_y = objects[0]["center"]

    assert np.isclose(
        center_x,
        40.5
    )

    assert np.isclose(
        center_y,
        50.5
    )

    print("Object-center calculation test passed.")

    print("=" * 70)
    print("ALL TESTS PASSED!")


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Lesson 16: Background Subtraction"
    )

    parser.add_argument(
        "--camera",
        action="store_true",
        help="Run real-time background subtraction using a camera."
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index. Default: 0."
    )

    args = parser.parse_args()

    if args.camera:

        run_camera(
            args.camera_index
        )

    else:

        run_synthetic_experiment()