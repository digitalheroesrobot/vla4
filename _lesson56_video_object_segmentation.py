"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 56 — VIDEO OBJECT SEGMENTATION
======================================================================

FILENAME
--------
lesson56_video_object_segmentation.py

LESSON OBJECTIVES
-----------------
1. Understand the difference between object detection, tracking,
   and video object segmentation.
2. Initialize an object mask using OpenCV GrabCut.
3. Estimate dense motion using Farneback optical flow.
4. Propagate a segmentation mask between video frames.
5. Refine propagated masks using GrabCut.
6. Apply morphological operations to clean masks.
7. Visualize segmentation overlays and object boundaries.
8. Save segmented video and segmentation statistics.
9. Understand common temporal segmentation failure cases.

======================================================================
1. OBJECT DETECTION VS TRACKING VS SEGMENTATION
======================================================================

OBJECT DETECTION:

    Input image
        |
        v
    [Bounding box]
        |
        v
    Object location

OBJECT TRACKING:

    Frame 1 -> Object ID 1
    Frame 2 -> Object ID 1
    Frame 3 -> Object ID 1

    The tracker maintains object identity over time.

IMAGE SEGMENTATION:

    Input image
        |
        v
    Pixel classification
        |
        v
    Foreground mask

VIDEO OBJECT SEGMENTATION:

    Frame 1 -> Object mask
    Frame 2 -> Propagated object mask
    Frame 3 -> Propagated object mask
    Frame 4 -> Propagated object mask

The goal is to maintain the same object's pixel-level segmentation
throughout a video.

======================================================================
2. GRABCUT INITIALIZATION
======================================================================

GrabCut is a graph-cut-based foreground/background segmentation
algorithm.

It models foreground and background color distributions using
Gaussian Mixture Models.

Each pixel receives one of four labels:

    GC_BGD:
        Definite background.

    GC_FGD:
        Definite foreground.

    GC_PR_BGD:
        Probable background.

    GC_PR_FGD:
        Probable foreground.

For the first frame, the user selects a rectangle around the object.

GrabCut estimates the object mask inside that rectangle.

IMPORTANT:

    GrabCut does not understand object semantics.

    It separates foreground and background using appearance,
    spatial constraints, and graph optimization.

======================================================================
3. FARNEBACK DENSE OPTICAL FLOW
======================================================================

Optical flow estimates apparent pixel movement between frames.

Let:

    I_t(x, y)

be the previous grayscale frame.

Let:

    I_t+1(x, y)

be the current grayscale frame.

The optical flow field is:

    F(x, y) = [u(x, y), v(x, y)]

where:

    u = horizontal displacement
    v = vertical displacement

The approximate brightness constancy assumption is:

    I_t(x, y) ~= I_t+1(x + u, y + v)

Farneback estimates dense optical flow using polynomial
expansion and coarse-to-fine processing.

OpenCV implementation:

    cv2.calcOpticalFlowFarneback()

======================================================================
4. MASK PROPAGATION
======================================================================

Suppose an object moves three pixels to the right.

Previous mask:

    M_t(x, y)

Current mask:

    M_t+1(x, y)

A forward flow field estimates where previous pixels move.

For practical image warping, we need a source coordinate for each
destination pixel.

Using an approximate inverse flow:

    source_x = destination_x - u
    source_y = destination_y - v

Then:

    M_t+1(x, y) =
        M_t(source_x, source_y)

This is implemented using:

    cv2.remap()

IMPORTANT:

    This inverse mapping is approximate because the forward flow
    field is not necessarily an exact inverse flow field.

    Occlusions, disocclusions, motion boundaries, and large
    displacements can produce incorrect mask propagation.

======================================================================
5. TEMPORAL MASK REFINEMENT
======================================================================

Optical flow alone is not enough.

A mask can gradually drift because of:

    - Occlusion.
    - Motion blur.
    - Illumination changes.
    - Similar foreground/background colors.
    - Camera movement.
    - Optical-flow errors.
    - Object deformation.

We refine the propagated mask using GrabCut.

The propagated mask initializes probable foreground/background
labels.

GrabCut then estimates a new foreground mask using the current
frame's appearance.

Morphological operations clean small artifacts:

    Opening:
        Remove small foreground noise.

    Closing:
        Fill small holes and gaps.

======================================================================
6. LIMITATIONS
======================================================================

This lesson is an educational video object segmentation pipeline.

It is NOT:

    - A pretrained segmentation model.
    - A semantic segmentation network.
    - An instance segmentation network.
    - SAM or SAM 2.
    - A general-purpose multi-object segmentation system.

It segments one manually initialized object.

For difficult scenes, consider learned segmentation models,
multiple-object masks, and more advanced temporal memory methods.

======================================================================
7. ROBOTICS APPLICATION — OPENARM V2 WASTE SORTING
======================================================================

Example:

    Camera
       |
       v
    Initial bottle selection
       |
       v
    GrabCut foreground mask
       |
       v
    Optical flow
       |
       v
    Mask propagation
       |
       v
    Temporal mask refinement
       |
       v
    Object contour
       |
       v
    Center / grasp-region estimation
       |
       v
    Camera-to-robot coordinate mapping
       |
       v
    OpenArm V2 grasping

The segmentation mask can help estimate:

    - Object silhouette.
    - Object area.
    - Object centroid.
    - Visible grasp regions.
    - Object orientation in the image plane.

A 2-D mask alone does not provide physical 3-D pose.
Depth and camera-to-robot calibration are needed for that.

======================================================================
8. INPUT
======================================================================

Use a real video:

    python lesson56_video_object_segmentation.py \
        --video conveyor_video.mp4

The program opens the first frame.

Select an object with the mouse and press ENTER or SPACE.

Controls:

    q:
        Quit.

    p:
        Pause / resume.

    r:
        Reinitialize the object mask on the current frame.

    +:
        Increase GrabCut refinement iterations.

    -:
        Decrease GrabCut refinement iterations.

======================================================================
9. OUTPUT
======================================================================

    runs/lesson56/
        video_object_segmentation.mp4
        segmentation_summary.json

======================================================================
10. TESTING
======================================================================

    python lesson56_video_object_segmentation.py --check-only

Tests cover:

    - Binary mask conversion.
    - Morphological cleanup.
    - Mask area.
    - Mask centroid.
    - Bounding box extraction.
    - Constant-flow mask propagation.
    - Zero-flow propagation.
    - Empty-mask handling.
    - Mask statistics.
    - Input validation.

======================================================================
NEXT LESSON
======================================================================

Lesson 57 — SAM 2 Video Tracking

We will study promptable segmentation, temporal memory,
and object-mask propagation using the SAM 2 architecture.

======================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import time

from dataclasses import dataclass

import cv2
import numpy as np


# ======================================================================
# CONFIGURATION
# ======================================================================

DEFAULT_OUTPUT_DIR = os.path.join(
    "runs",
    "lesson56",
)

DEFAULT_GRABCUT_ITERATIONS = 5

DEFAULT_FLOW_PARAMETERS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 21,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


# ======================================================================
# VALIDATION AND BASIC MASK OPERATIONS
# ======================================================================

def validate_mask(mask):
    """
    Validate a 2-D segmentation mask.
    """

    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(
            "Segmentation mask must be a 2-D array."
        )

    return mask


def to_binary_mask(mask):
    """
    Convert a mask to uint8 values:

        foreground = 255
        background = 0
    """

    mask = validate_mask(mask)

    return (
        mask > 0
    ).astype(np.uint8) * 255


def mask_area(mask):
    """
    Return foreground area in pixels.
    """

    binary = to_binary_mask(mask)

    return int(
        np.count_nonzero(binary)
    )


def mask_centroid(mask):
    """
    Return foreground centroid:

        (cx, cy)

    Returns None for an empty mask.
    """

    binary = to_binary_mask(mask)

    moments = cv2.moments(
        binary,
        binaryImage=True,
    )

    if moments["m00"] <= 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    return (
        float(cx),
        float(cy),
    )


def mask_bounding_box(mask):
    """
    Return:

        (x, y, width, height)

    Returns None for an empty mask.
    """

    binary = to_binary_mask(mask)

    points = cv2.findNonZero(binary)

    if points is None:
        return None

    x, y, width, height = cv2.boundingRect(
        points
    )

    return (
        int(x),
        int(y),
        int(width),
        int(height),
    )


# ======================================================================
# MORPHOLOGICAL CLEANUP
# ======================================================================

def clean_mask(
    mask,
    kernel_size=5,
    opening_iterations=1,
    closing_iterations=1,
):
    """
    Remove small noise and fill small holes.

    Opening:
        Erosion followed by dilation.

    Closing:
        Dilation followed by erosion.
    """

    binary = to_binary_mask(mask)

    if kernel_size < 1:
        raise ValueError(
            "Kernel size must be positive."
        )

    if kernel_size % 2 == 0:
        kernel_size += 1

    if opening_iterations < 0:
        raise ValueError(
            "Opening iterations cannot be negative."
        )

    if closing_iterations < 0:
        raise ValueError(
            "Closing iterations cannot be negative."
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    result = binary.copy()

    if opening_iterations > 0:
        result = cv2.morphologyEx(
            result,
            cv2.MORPH_OPEN,
            kernel,
            iterations=opening_iterations,
        )

    if closing_iterations > 0:
        result = cv2.morphologyEx(
            result,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=closing_iterations,
        )

    return result


# ======================================================================
# OPTICAL FLOW
# ======================================================================

def calculate_dense_optical_flow(
    previous_gray,
    current_gray,
    parameters=None,
):
    """
    Calculate Farneback dense optical flow.

    Returns:

        flow[..., 0] = horizontal displacement
        flow[..., 1] = vertical displacement
    """

    previous_gray = np.asarray(
        previous_gray,
        dtype=np.uint8,
    )

    current_gray = np.asarray(
        current_gray,
        dtype=np.uint8,
    )

    if previous_gray.ndim != 2:
        raise ValueError(
            "Previous grayscale image must be 2-D."
        )

    if current_gray.ndim != 2:
        raise ValueError(
            "Current grayscale image must be 2-D."
        )

    if previous_gray.shape != current_gray.shape:
        raise ValueError(
            "Optical-flow images must have identical dimensions."
        )

    if parameters is None:
        parameters = DEFAULT_FLOW_PARAMETERS

    return cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        **parameters,
    )


def warp_mask_with_flow(
    previous_mask,
    flow,
    threshold=127,
):
    """
    Propagate a previous mask using dense optical flow.

    Approximate inverse mapping:

        source_x = destination_x - flow_x
        source_y = destination_y - flow_y

    This is an approximation because the supplied flow is a
    forward flow field, not a mathematically exact inverse field.

    cv2.remap() uses destination-to-source mapping.
    """

    previous_mask = to_binary_mask(
        previous_mask
    )

    flow = np.asarray(
        flow,
        dtype=np.float32,
    )

    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(
            "Flow must have shape (H, W, 2)."
        )

    height, width = previous_mask.shape

    if flow.shape[:2] != (height, width):
        raise ValueError(
            "Flow and mask dimensions must match."
        )

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]

    warped = cv2.remap(
        previous_mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return (
        warped >= threshold
    ).astype(np.uint8) * 255


# ======================================================================
# GRABCUT INITIALIZATION
# ======================================================================

def initialize_grabcut_mask(
    frame,
    rectangle,
    iterations=DEFAULT_GRABCUT_ITERATIONS,
):
    """
    Initialize foreground segmentation using a selected rectangle.

    rectangle:

        (x, y, width, height)

    GrabCut uses the rectangle as an initial foreground region.
    """

    if frame is None:
        raise ValueError(
            "Input frame cannot be None."
        )

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            "Expected a three-channel BGR image."
        )

    if iterations < 1:
        raise ValueError(
            "GrabCut iterations must be positive."
        )

    height, width = frame.shape[:2]

    x, y, rect_width, rect_height = map(
        int,
        rectangle,
    )

    if (
        rect_width <= 0
        or rect_height <= 0
        or x < 0
        or y < 0
        or x + rect_width > width
        or y + rect_height > height
    ):
        raise ValueError(
            "Selected rectangle is outside the image."
        )

    if rect_width < 2 or rect_height < 2:
        raise ValueError(
            "Selected rectangle is too small."
        )

    grabcut_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    background_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    cv2.grabCut(
        frame,
        grabcut_mask,
        (
            x,
            y,
            rect_width,
            rect_height,
        ),
        background_model,
        foreground_model,
        iterations,
        cv2.GC_INIT_WITH_RECT,
    )

    binary = np.where(
        (
            grabcut_mask == cv2.GC_FGD
        ) | (
            grabcut_mask == cv2.GC_PR_FGD
        ),
        255,
        0,
    ).astype(np.uint8)

    return clean_mask(binary)


# ======================================================================
# GRABCUT REFINEMENT
# ======================================================================

def refine_mask_with_grabcut(
    frame,
    initial_mask,
    iterations=2,
):
    """
    Refine a propagated mask using GrabCut.

    Pixels inside the propagated mask are probable foreground.
    Pixels outside are probable background.

    The outer image boundary is marked as definite background.

    This is not a semantic model. GrabCut uses appearance and
    graph-based optimization to refine the mask.
    """

    initial_mask = to_binary_mask(
        initial_mask
    )

    height, width = initial_mask.shape

    if frame.shape[:2] != (height, width):
        raise ValueError(
            "Frame and initial mask dimensions must match."
        )

    if iterations < 1:
        raise ValueError(
            "Iterations must be positive."
        )

    if mask_area(initial_mask) == 0:
        return initial_mask

    grabcut_mask = np.where(
        initial_mask > 0,
        cv2.GC_PR_FGD,
        cv2.GC_PR_BGD,
    ).astype(np.uint8)

    # Pixels along the image border are fixed background.
    grabcut_mask[0, :] = cv2.GC_BGD
    grabcut_mask[-1, :] = cv2.GC_BGD
    grabcut_mask[:, 0] = cv2.GC_BGD
    grabcut_mask[:, -1] = cv2.GC_BGD

    background_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    try:
        cv2.grabCut(
            frame,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            iterations,
            cv2.GC_INIT_WITH_MASK,
        )

    except cv2.error:
        # GrabCut may fail when the foreground/background models
        # cannot be estimated reliably.
        return clean_mask(initial_mask)

    result = np.where(
        (
            grabcut_mask == cv2.GC_FGD
        ) | (
            grabcut_mask == cv2.GC_PR_FGD
        ),
        255,
        0,
    ).astype(np.uint8)

    return clean_mask(result)


# ======================================================================
# VISUALIZATION
# ======================================================================

def create_overlay(
    frame,
    mask,
    alpha=0.45,
):
    """
    Draw the segmentation mask over the original frame.
    """

    binary = to_binary_mask(mask)

    if frame.shape[:2] != binary.shape:
        raise ValueError(
            "Frame and mask dimensions must match."
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "Alpha must be between 0 and 1."
        )

    overlay = frame.copy()

    foreground = binary > 0

    mask_color = np.zeros_like(
        frame
    )

    mask_color[:, :, 1] = 255

    overlay[foreground] = cv2.addWeighted(
        frame[foreground],
        1.0 - alpha,
        mask_color[foreground],
        alpha,
        0,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 255),
        2,
    )

    centroid = mask_centroid(binary)

    if centroid is not None:
        center = (
            int(round(centroid[0])),
            int(round(centroid[1])),
        )

        cv2.circle(
            overlay,
            center,
            5,
            (0, 0, 255),
            -1,
        )

    return overlay


# ======================================================================
# SEGMENTATION STATE
# ======================================================================

@dataclass
class SegmentationState:
    previous_gray: np.ndarray | None = None
    mask: np.ndarray | None = None
    frame_index: int = 0
    initialized: bool = False

    def reset(self):
        self.previous_gray = None
        self.mask = None
        self.initialized = False


# ======================================================================
# INTERACTIVE INITIALIZATION
# ======================================================================

def select_initial_object(frame):
    """
    Let the user select an initial object rectangle.

    ENTER or SPACE confirms.
    ESC cancels.
    """

    window_name = "Select object for segmentation"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    rectangle = cv2.selectROI(
        window_name,
        frame,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(
        window_name
    )

    x, y, width, height = map(
        int,
        rectangle,
    )

    if width <= 0 or height <= 0:
        return None

    return (
        x,
        y,
        width,
        height,
    )


# ======================================================================
# VIDEO PROCESSING
# ======================================================================

def process_video(args):
    """
    Run video object segmentation.

    Workflow:

        1. Read first frame.
        2. Select object ROI.
        3. Initialize GrabCut mask.
        4. Estimate dense optical flow.
        5. Warp previous mask.
        6. Refine with GrabCut.
        7. Visualize and save.
    """

    capture = cv2.VideoCapture(
        args.video
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {args.video}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    original_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    original_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if original_width <= 0 or original_height <= 0:
        capture.release()

        raise RuntimeError(
            "Invalid video dimensions."
        )

    scale = 1.0

    if (
        args.max_width > 0
        and original_width > args.max_width
    ):
        scale = args.max_width / original_width

    width = int(
        round(original_width * scale)
    )

    height = int(
        round(original_height * scale)
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        args.output_dir,
        "video_object_segmentation.mp4",
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()

        raise RuntimeError(
            "Could not create output video."
        )

    state = SegmentationState()

    total_frames = 0
    processed_frames = 0

    areas = []
    centroids = []

    start_time = time.perf_counter()

    frame = None

    try:
        ret, frame = capture.read()

        if not ret:
            raise RuntimeError(
                "Could not read the first video frame."
            )

        if scale != 1.0:
            frame = cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_AREA,
            )

        rectangle = select_initial_object(
            frame
        )

        if rectangle is None:
            raise RuntimeError(
                "No valid object rectangle was selected."
            )

        mask = initialize_grabcut_mask(
            frame,
            rectangle,
            iterations=args.initial_iterations,
        )

        if mask_area(mask) == 0:
            raise RuntimeError(
                "GrabCut did not produce a foreground mask. "
                "Try selecting a tighter object rectangle."
            )

        state.mask = mask

        state.previous_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        state.initialized = True

        # ----------------------------------------------------------
        # Process first frame
        # ----------------------------------------------------------

        overlay = create_overlay(
            frame,
            state.mask,
        )

        cv2.putText(
            overlay,
            "INITIAL MASK",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        writer.write(overlay)

        total_frames += 1
        processed_frames += 1

        areas.append(
            mask_area(state.mask)
        )

        centroids.append(
            mask_centroid(state.mask)
        )

        cv2.imshow(
            "Video Object Segmentation",
            overlay,
        )

        print(
            "\nSegmentation initialized."
        )

        print(
            f"Initial mask area: {mask_area(state.mask)} pixels"
        )

        print(
            "\nControls: q=quit, p=pause, r=reinitialize, "
            "+/-=GrabCut iterations"
        )

        grabcut_iterations = args.refinement_iterations

        paused = False

        # ----------------------------------------------------------
        # Main video loop
        # ----------------------------------------------------------

        while True:

            if not paused:
                ret, frame = capture.read()

                if not ret:
                    break

                if scale != 1.0:
                    frame = cv2.resize(
                        frame,
                        (width, height),
                        interpolation=cv2.INTER_AREA,
                    )

                current_gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY,
                )

                # --------------------------------------------------
                # Dense optical flow
                # --------------------------------------------------

                flow = calculate_dense_optical_flow(
                    state.previous_gray,
                    current_gray,
                )

                # --------------------------------------------------
                # Propagate previous mask
                # --------------------------------------------------

                propagated_mask = warp_mask_with_flow(
                    state.mask,
                    flow,
                )

                # --------------------------------------------------
                # Morphological cleanup
                # --------------------------------------------------

                propagated_mask = clean_mask(
                    propagated_mask,
                    kernel_size=args.kernel_size,
                    opening_iterations=1,
                    closing_iterations=1,
                )

                # --------------------------------------------------
                # GrabCut refinement
                # --------------------------------------------------

                if (
                    mask_area(propagated_mask)
                    >= args.minimum_mask_area
                ):
                    refined_mask = refine_mask_with_grabcut(
                        frame,
                        propagated_mask,
                        iterations=grabcut_iterations,
                    )
                else:
                    # If propagation loses the mask completely,
                    # retain the last valid mask rather than
                    # silently declaring the object absent.
                    refined_mask = state.mask.copy()

                # --------------------------------------------------
                # Update state
                # --------------------------------------------------

                state.mask = refined_mask

                state.previous_gray = current_gray

                state.frame_index += 1

                total_frames += 1
                processed_frames += 1

                area = mask_area(state.mask)

                centroid = mask_centroid(
                    state.mask
                )

                areas.append(area)
                centroids.append(centroid)

                # --------------------------------------------------
                # Visualization
                # --------------------------------------------------

                overlay = create_overlay(
                    frame,
                    state.mask,
                )

                cv2.putText(
                    overlay,
                    f"Frame: {state.frame_index}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    overlay,
                    f"Mask area: {area} px",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                if centroid is not None:
                    cv2.putText(
                        overlay,
                        (
                            f"Centroid: "
                            f"({centroid[0]:.1f}, "
                            f"{centroid[1]:.1f})"
                        ),
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

                writer.write(
                    overlay
                )

                if not args.no_display:
                    cv2.imshow(
                        "Video Object Segmentation",
                        overlay,
                    )

            # ------------------------------------------------------
            # Keyboard controls
            # ------------------------------------------------------

            if args.no_display:
                if not paused:
                    continue

                key = cv2.waitKey(30) & 0xFF

            else:
                key = cv2.waitKey(
                    30 if paused else 1
                ) & 0xFF

            if key == ord("q") or key == 27:
                break

            elif key == ord("p"):
                paused = not paused

            elif key in (
                ord("+"),
                ord("="),
            ):
                grabcut_iterations += 1

                print(
                    f"GrabCut iterations: {grabcut_iterations}"
                )

            elif key in (
                ord("-"),
                ord("_"),
            ):
                grabcut_iterations = max(
                    1,
                    grabcut_iterations - 1,
                )

                print(
                    f"GrabCut iterations: {grabcut_iterations}"
                )

            elif key == ord("r"):
                if args.no_display:
                    continue

                print(
                    "\nReinitialize segmentation."
                )

                rectangle = select_initial_object(
                    frame
                )

                if rectangle is not None:
                    state.mask = initialize_grabcut_mask(
                        frame,
                        rectangle,
                        iterations=args.initial_iterations,
                    )

                    state.previous_gray = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2GRAY,
                    )

                    print(
                        "Segmentation reinitialized."
                    )

    finally:
        capture.release()
        writer.release()

        cv2.destroyAllWindows()

    elapsed = (
        time.perf_counter() - start_time
    )

    processing_fps = (
        processed_frames / elapsed
        if elapsed > 0
        else 0.0
    )

    valid_centroids = [
        list(point)
        for point in centroids
        if point is not None
    ]

    summary = {
        "lesson": "Lesson 56 - Video Object Segmentation",
        "input_video": args.video,
        "output_video": output_path,
        "frames_processed": int(processed_frames),
        "input_fps": float(fps),
        "processing_fps": float(processing_fps),
        "processing_seconds": float(elapsed),
        "frame_width": int(width),
        "frame_height": int(height),
        "mean_mask_area": (
            float(np.mean(areas))
            if areas
            else 0.0
        ),
        "minimum_mask_area": (
            int(np.min(areas))
            if areas
            else 0
        ),
        "maximum_mask_area": (
            int(np.max(areas))
            if areas
            else 0
        ),
        "centroid_history": valid_centroids,
    }

    summary_path = os.path.join(
        args.output_dir,
        "segmentation_summary.json",
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

    print("\n" + "=" * 65)
    print("VIDEO OBJECT SEGMENTATION SUMMARY")
    print("=" * 65)

    print(
        f"Frames processed: {processed_frames}"
    )

    print(
        f"Input FPS: {fps:.2f}"
    )

    print(
        f"Processing FPS: {processing_fps:.2f}"
    )

    print(
        f"Mean mask area: {summary['mean_mask_area']:.1f}"
    )

    print(
        f"Output video: {output_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print("=" * 65)


# ======================================================================
# NUMERICAL TESTS
# ======================================================================

def run_tests():
    print("\nRunning Lesson 56 tests...")
    print("=" * 65)

    # --------------------------------------------------------------
    # Test 1: binary mask conversion
    # --------------------------------------------------------------

    mask = np.array(
        [
            [0, 1],
            [255, 0],
        ],
        dtype=np.uint8,
    )

    binary = to_binary_mask(mask)

    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset(
        {0, 255}
    )

    assert binary[0, 0] == 0
    assert binary[0, 1] == 255

    print("Test 01 passed: binary mask conversion.")

    # --------------------------------------------------------------
    # Test 2: mask area
    # --------------------------------------------------------------

    mask = np.zeros(
        (10, 10),
        dtype=np.uint8,
    )

    mask[2:5, 3:7] = 255

    assert mask_area(mask) == 12

    print("Test 02 passed: mask area.")

    # --------------------------------------------------------------
    # Test 3: empty mask area
    # --------------------------------------------------------------

    empty = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    assert mask_area(empty) == 0

    print("Test 03 passed: empty mask area.")

    # --------------------------------------------------------------
    # Test 4: centroid
    # --------------------------------------------------------------

    mask = np.zeros(
        (10, 10),
        dtype=np.uint8,
    )

    mask[2:4, 4:6] = 255

    centroid = mask_centroid(mask)

    assert np.allclose(
        centroid,
        [4.5, 2.5],
    )

    print("Test 04 passed: mask centroid.")

    # --------------------------------------------------------------
    # Test 5: empty mask centroid
    # --------------------------------------------------------------

    assert mask_centroid(empty) is None

    print("Test 05 passed: empty mask centroid.")

    # --------------------------------------------------------------
    # Test 6: bounding box
    # --------------------------------------------------------------

    mask = np.zeros(
        (20, 30),
        dtype=np.uint8,
    )

    mask[4:9, 7:15] = 255

    bbox = mask_bounding_box(mask)

    assert bbox == (7, 4, 8, 5)

    print("Test 06 passed: bounding box extraction.")

    # --------------------------------------------------------------
    # Test 7: empty bounding box
    # --------------------------------------------------------------

    assert mask_bounding_box(empty) is None

    print("Test 07 passed: empty bounding box.")

    # --------------------------------------------------------------
    # Test 8: binary mask cleanup
    # --------------------------------------------------------------

    mask = np.zeros(
        (50, 50),
        dtype=np.uint8,
    )

    mask[10:40, 10:40] = 255

    cleaned = clean_mask(
        mask,
        kernel_size=3,
    )

    assert cleaned.shape == mask.shape
    assert cleaned.dtype == np.uint8

    print("Test 08 passed: morphological cleanup.")

    # --------------------------------------------------------------
    # Test 9: zero-flow mask propagation
    # --------------------------------------------------------------

    mask = np.zeros(
        (30, 30),
        dtype=np.uint8,
    )

    mask[10:20, 10:20] = 255

    flow = np.zeros(
        (30, 30, 2),
        dtype=np.float32,
    )

    warped = warp_mask_with_flow(
        mask,
        flow,
    )

    assert np.array_equal(
        warped,
        mask,
    )

    print("Test 09 passed: zero-flow propagation.")

    # --------------------------------------------------------------
    # Test 10: horizontal flow propagation
    # --------------------------------------------------------------

    mask = np.zeros(
        (40, 40),
        dtype=np.uint8,
    )

    mask[10:20, 10:20] = 255

    flow = np.zeros(
        (40, 40, 2),
        dtype=np.float32,
    )

    flow[..., 0] = 3.0

    warped = warp_mask_with_flow(
        mask,
        flow,
    )

    bbox = mask_bounding_box(warped)

    assert bbox == (13, 10, 10, 10)

    print("Test 10 passed: horizontal flow propagation.")

    # --------------------------------------------------------------
    # Test 11: vertical flow propagation
    # --------------------------------------------------------------

    flow = np.zeros(
        (40, 40, 2),
        dtype=np.float32,
    )

    flow[..., 1] = 4.0

    warped = warp_mask_with_flow(
        mask,
        flow,
    )

    bbox = mask_bounding_box(warped)

    assert bbox == (10, 14, 10, 10)

    print("Test 11 passed: vertical flow propagation.")

    # --------------------------------------------------------------
    # Test 12: diagonal flow propagation
    # --------------------------------------------------------------

    flow = np.zeros(
        (40, 40, 2),
        dtype=np.float32,
    )

    flow[..., 0] = 2.0
    flow[..., 1] = 3.0

    warped = warp_mask_with_flow(
        mask,
        flow,
    )

    bbox = mask_bounding_box(warped)

    assert bbox == (12, 13, 10, 10)

    print("Test 12 passed: diagonal flow propagation.")

    # --------------------------------------------------------------
    # Test 13: empty mask propagation
    # --------------------------------------------------------------

    empty = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    flow = np.zeros(
        (20, 20, 2),
        dtype=np.float32,
    )

    warped = warp_mask_with_flow(
        empty,
        flow,
    )

    assert mask_area(warped) == 0

    print("Test 13 passed: empty mask propagation.")

    # --------------------------------------------------------------
    # Test 14: flow dimension validation
    # --------------------------------------------------------------

    try:
        warp_mask_with_flow(
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.float32),
        )

        raise AssertionError(
            "Invalid flow shape was accepted."
        )

    except ValueError:
        pass

    print("Test 14 passed: flow dimension validation.")

    # --------------------------------------------------------------
    # Test 15: mismatched flow dimensions
    # --------------------------------------------------------------

    try:
        warp_mask_with_flow(
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((12, 12, 2), dtype=np.float32),
        )

        raise AssertionError(
            "Mismatched dimensions were accepted."
        )

    except ValueError:
        pass

    print("Test 15 passed: mismatched dimensions rejection.")

    # --------------------------------------------------------------
    # Test 16: invalid mask dimensionality
    # --------------------------------------------------------------

    try:
        to_binary_mask(
            np.zeros((10, 10, 3), dtype=np.uint8)
        )

        raise AssertionError(
            "Three-dimensional mask was accepted."
        )

    except ValueError:
        pass

    print("Test 16 passed: mask dimensionality validation.")

    # --------------------------------------------------------------
    # Test 17: mask centroid translation
    # --------------------------------------------------------------

    first = np.zeros(
        (50, 50),
        dtype=np.uint8,
    )

    first[10:20, 10:20] = 255

    second = np.zeros(
        (50, 50),
        dtype=np.uint8,
    )

    second[10:20, 15:25] = 255

    c1 = mask_centroid(first)
    c2 = mask_centroid(second)

    assert np.isclose(
        c2[0] - c1[0],
        5.0,
    )

    print("Test 17 passed: centroid translation.")

    # --------------------------------------------------------------
    # Test 18: overlay dimensions
    # --------------------------------------------------------------

    frame = np.zeros(
        (40, 40, 3),
        dtype=np.uint8,
    )

    mask = np.zeros(
        (40, 40),
        dtype=np.uint8,
    )

    mask[10:20, 10:20] = 255

    overlay = create_overlay(
        frame,
        mask,
    )

    assert overlay.shape == frame.shape
    assert overlay.dtype == np.uint8

    print("Test 18 passed: segmentation overlay.")

    # --------------------------------------------------------------
    # Test 19: overlay rejects invalid alpha
    # --------------------------------------------------------------

    try:
        create_overlay(
            frame,
            mask,
            alpha=1.5,
        )

        raise AssertionError(
            "Invalid alpha was accepted."
        )

    except ValueError:
        pass

    print("Test 19 passed: overlay alpha validation.")

    # --------------------------------------------------------------
    # Test 20: mask cleanup preserves large object
    # --------------------------------------------------------------

    mask = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    mask[20:80, 20:80] = 255

    cleaned = clean_mask(
        mask,
        kernel_size=5,
    )

    assert mask_area(cleaned) > 3000

    print("Test 20 passed: large-object preservation.")

    # --------------------------------------------------------------
    # Test 21: invalid rectangle
    # --------------------------------------------------------------

    frame = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    try:
        initialize_grabcut_mask(
            frame,
            (90, 90, 30, 30),
        )

        raise AssertionError(
            "Invalid rectangle was accepted."
        )

    except ValueError:
        pass

    print("Test 21 passed: invalid ROI rejection.")

    # --------------------------------------------------------------
    # Test 22: valid rectangle dimensions
    # --------------------------------------------------------------

    try:
        initialize_grabcut_mask(
            frame,
            (10, 10, 0, 20),
        )

        raise AssertionError(
            "Zero-width rectangle was accepted."
        )

    except ValueError:
        pass

    print("Test 22 passed: degenerate ROI rejection.")

    # --------------------------------------------------------------
    # Test 23: mismatched grayscale image dimensions
    # --------------------------------------------------------------

    try:
        calculate_dense_optical_flow(
            np.zeros((20, 20), dtype=np.uint8),
            np.zeros((30, 30), dtype=np.uint8),
        )

        raise AssertionError(
            "Mismatched flow inputs were accepted."
        )

    except ValueError:
        pass

    print("Test 23 passed: optical-flow input validation.")

    # --------------------------------------------------------------
    # Test 24: segmentation state reset
    # --------------------------------------------------------------

    state = SegmentationState(
        previous_gray=np.zeros((10, 10), dtype=np.uint8),
        mask=np.ones((10, 10), dtype=np.uint8),
        frame_index=10,
        initialized=True,
    )

    state.reset()

    assert state.previous_gray is None
    assert state.mask is None
    assert state.initialized is False

    print("Test 24 passed: segmentation state reset.")

    print("=" * 65)
    print("All 24 tests passed.")


# ======================================================================
# COMMAND LINE
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Lesson 56 - Video Object Segmentation"
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a real input video.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=960,
        help="Resize videos wider than this. Use 0 to disable.",
    )

    parser.add_argument(
        "--initial-iterations",
        type=int,
        default=DEFAULT_GRABCUT_ITERATIONS,
    )

    parser.add_argument(
        "--refinement-iterations",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--minimum-mask-area",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display.",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run numerical tests only.",
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

    if args.video is None:
        raise ValueError(
            "Please provide a real video using --video."
        )

    if args.max_width < 0:
        raise ValueError(
            "--max-width cannot be negative."
        )

    if args.minimum_mask_area < 0:
        raise ValueError(
            "--minimum-mask-area cannot be negative."
        )

    run_tests()

    process_video(args)


if __name__ == "__main__":
    main()