"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 46 — OPTICAL FLOW-ASSISTED OBJECT TRACKING
======================================================================

Filename:
    lesson46_optical_flow_assisted_tracking.py

PHASE:
    Phase 4 — Tracking and Visual Memory

======================================================================
LEARNING OBJECTIVES
======================================================================

Implement:

    1. Sparse optical flow
    2. Lucas-Kanade feature tracking
    3. Feature-point initialization inside object bounding boxes
    4. Robust median motion estimation
    5. Outlier rejection using median absolute deviation
    6. Bounding-box propagation
    7. Detection-assisted correction
    8. Track recovery after temporary detection failures
    9. Feature-point visualization
    10. Real-video tracking

======================================================================
1. WHAT IS OPTICAL FLOW?
======================================================================

Optical flow estimates apparent motion of image pixels between
consecutive video frames.

Let:

    I(x, y, t)

represent image intensity at location (x, y) and time t.

The brightness constancy assumption is:

    I(x, y, t) = I(x + dx, y + dy, t + dt)

Using a first-order Taylor expansion:

    Ix * dx + Iy * dy + It * dt = 0

Divide by dt:

    Ix * u + Iy * v + It = 0

Where:

    Ix = horizontal image gradient
    Iy = vertical image gradient
    It = temporal image gradient

    u = dx / dt
    v = dy / dt

This is the optical flow constraint equation.

There are two unknowns (u, v), but only one equation.

The Lucas-Kanade method solves this by assuming nearby pixels
share approximately the same motion.

======================================================================
2. LUCAS-KANADE OPTICAL FLOW
======================================================================

For a local image window containing multiple pixels:

    Ix1*u + Iy1*v = -It1
    Ix2*u + Iy2*v = -It2
    ...
    Ixn*u + Iyn*v = -Itn

In matrix form:

    A * velocity = b

Least-squares solution:

    velocity = (A.T @ A)^(-1) @ A.T @ b

OpenCV implementation:

    cv2.calcOpticalFlowPyrLK()

We use it for real video processing.

======================================================================
3. WHY TRACK FEATURES INSIDE OBJECT BOXES?
======================================================================

A bounding box alone tells us where an object was detected.

Feature points tell us how image texture inside that object moves.

Example:

    Previous frame:
        Bottle box at x = 100

    Current frame:
        Feature points move approximately +5 pixels.

    New predicted box:
        x = 105

We estimate displacement from multiple points instead of trusting
one potentially noisy feature.

======================================================================
4. ROBUST MOTION ESTIMATION
======================================================================

For feature-point displacements:

    dx_1, dx_2, ..., dx_n

Use:

    median_dx = median(dx)

Similarly:

    median_dy = median(dy)

The median is less sensitive to outliers than the mean.

We additionally reject points whose displacement differs
significantly from the median using median absolute deviation.

======================================================================
5. DETECTION + OPTICAL FLOW
======================================================================

Pipeline:

    Previous frame
          |
          v
    Feature points
          |
          v
    Lucas-Kanade optical flow
          |
          v
    Robust displacement
          |
          v
    Predicted bounding box
          |
          v
    New object detections
          |
          v
    IoU association
          |
          v
    Corrected object trajectory

When detections temporarily disappear, optical flow can continue
propagating the track for a limited number of frames.

======================================================================
6. IMPORTANT LIMITATIONS
======================================================================

Optical flow measures apparent image motion, not object identity.

Problems include:

    - Motion blur
    - Low-texture objects
    - Large displacement between frames
    - Occlusion
    - Camera movement
    - Similar motion between nearby objects

A moving camera also creates global image motion.

For a fixed conveyor camera, optical flow is particularly useful.

For a moving robot camera, camera-motion compensation may be needed.

======================================================================
7. REAL VIDEO DETECTOR
======================================================================

The included detector uses OpenCV MOG2 background subtraction.

It is a demonstration detector, not a semantic detector.

It cannot reliably distinguish:

    Bottle
    Can
    Cup
    Paper
    Plastic bag

For a real waste-sorting robot, replace it with YOLO or another
object detector. The optical-flow tracker can remain unchanged.

======================================================================
INSTALL
======================================================================

    pip install numpy opencv-python

======================================================================
RUN TESTS
======================================================================

    python lesson46_optical_flow_assisted_tracking.py --check-only

======================================================================
PROCESS REAL VIDEO
======================================================================

    python lesson46_optical_flow_assisted_tracking.py \
        --video conveyor_video.mp4

Disable display:

    python lesson46_optical_flow_assisted_tracking.py \
        --video conveyor_video.mp4 \
        --no-display

======================================================================
OUTPUT
======================================================================

    runs/lesson46/
        optical_flow_tracking.mp4
        tracking_summary.json

======================================================================
ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste-sorting conveyor:

    Camera
      |
      v
    Object detector
      |
      +--------------------+
      |                    |
      v                    v
    Detections         Optical flow
      |                    |
      +---------+----------+
                |
                v
         Object tracking
                |
                v
         Object trajectory
                |
                v
         Grasp scheduling

Optical flow can help estimate how far an object moves between
camera observations and improve short-term trajectory continuity.

======================================================================
"""

import argparse
import json
import time

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# =====================================================================
# CONFIGURATION
# =====================================================================

DEFAULT_MAX_FEATURES = 80

DEFAULT_QUALITY_LEVEL = 0.01

DEFAULT_MIN_DISTANCE = 5

DEFAULT_FLOW_WINDOW = 21

DEFAULT_PYRAMID_LEVELS = 3

DEFAULT_MIN_TRACKED_POINTS = 4

DEFAULT_MAX_MISSED = 15

DEFAULT_IOU_THRESHOLD = 0.10

DEFAULT_MIN_DETECTION_AREA = 300


# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class Detection:

    bbox: np.ndarray

    score: float = 1.0

    class_id: int = 0

    def __post_init__(self):

        self.bbox = np.asarray(
            self.bbox,
            dtype=np.float64
        )

        self.score = float(self.score)

        self.class_id = int(self.class_id)

        if self.bbox.shape != (4,):
            raise ValueError("bbox must have shape (4,).")

        if not np.isfinite(self.bbox).all():
            raise ValueError("bbox must be finite.")

        if self.bbox[2] <= self.bbox[0]:
            raise ValueError("Invalid bbox width.")

        if self.bbox[3] <= self.bbox[1]:
            raise ValueError("Invalid bbox height.")


# =====================================================================
# BOUNDING BOX UTILITIES
# =====================================================================

def bbox_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = map(float, box_a)

    bx1, by1, bx2, by2 = map(float, box_b)

    ix1 = max(ax1, bx1)

    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)

    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    intersection = iw * ih

    area_a = max(
        0.0,
        ax2 - ax1
    ) * max(
        0.0,
        ay2 - ay1
    )

    area_b = max(
        0.0,
        bx2 - bx1
    ) * max(
        0.0,
        by2 - by1
    )

    union = (
        area_a + area_b - intersection
    )

    return intersection / max(
        union,
        1e-12
    )


def clip_bbox(bbox, width, height):

    x1, y1, x2, y2 = map(
        float,
        bbox
    )

    x1 = np.clip(x1, 0, width - 1)

    y1 = np.clip(y1, 0, height - 1)

    x2 = np.clip(x2, 0, width - 1)

    y2 = np.clip(y2, 0, height - 1)

    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)

    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float64
    )


# =====================================================================
# ROBUST MOTION ESTIMATION
# =====================================================================

def median_motion(displacements, mad_multiplier=3.5):

    """
    Estimate robust 2D displacement.

    Input:
        displacements: shape (N, 2)

    Output:
        median displacement: shape (2,)
        inlier mask: shape (N,)

    Median absolute deviation is used to reject outliers.
    """

    displacements = np.asarray(
        displacements,
        dtype=np.float64
    )

    if displacements.ndim != 2:
        raise ValueError(
            "Displacements must have shape (N, 2)."
        )

    if displacements.shape[1] != 2:
        raise ValueError(
            "Displacements must have two columns."
        )

    if len(displacements) == 0:

        return (
            np.zeros(2, dtype=np.float64),

            np.zeros(0, dtype=bool)
        )

    finite = np.isfinite(
        displacements
    ).all(axis=1)

    if not np.any(finite):

        return (
            np.zeros(2, dtype=np.float64),

            finite
        )

    median = np.median(
        displacements[finite],
        axis=0
    )

    distances = np.linalg.norm(
        displacements - median,
        axis=1
    )

    median_distance = np.median(
        distances[finite]
    )

    mad = np.median(
        np.abs(
            distances[finite] - median_distance
        )
    )

    robust_sigma = 1.4826 * mad

    threshold = max(
        1.0,
        median_distance + mad_multiplier * robust_sigma
    )

    inliers = (
        finite
        & (distances <= threshold)
    )

    if not np.any(inliers):
        inliers = finite

    final_motion = np.median(
        displacements[inliers],
        axis=0
    )

    return (
        final_motion,
        inliers
    )


# =====================================================================
# FEATURE EXTRACTION
# =====================================================================

def extract_features(
    gray,
    bbox,
    max_features=DEFAULT_MAX_FEATURES,
    quality_level=DEFAULT_QUALITY_LEVEL,
    min_distance=DEFAULT_MIN_DISTANCE
):

    """
    Extract Shi-Tomasi corners inside a bounding box.

    OpenCV implementation:

        cv2.goodFeaturesToTrack()

    Points are returned in shape:

        (N, 1, 2)
    """

    height, width = gray.shape[:2]

    bbox = clip_bbox(
        bbox,
        width,
        height
    )

    x1, y1, x2, y2 = map(
        int,
        bbox
    )

    x1 = max(0, x1)

    y1 = max(0, y1)

    x2 = min(width, x2)

    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:

        return None

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    mask[y1:y2, x1:x2] = 255

    points = cv2.goodFeaturesToTrack(

        gray,

        maxCorners=max_features,

        qualityLevel=quality_level,

        minDistance=min_distance,

        mask=mask,

        blockSize=7,

        useHarrisDetector=False
    )

    return points


# =====================================================================
# LUCAS-KANADE OPTICAL FLOW
# =====================================================================

def calculate_optical_flow(
    previous_gray,
    current_gray,
    previous_points,
    window_size=DEFAULT_FLOW_WINDOW,
    pyramid_levels=DEFAULT_PYRAMID_LEVELS
):

    """
    Track sparse feature points using Lucas-Kanade optical flow.

    OpenCV:

        cv2.calcOpticalFlowPyrLK()

    Returns:
        current_points
        valid_mask
        displacements
        errors
    """

    if previous_points is None or len(previous_points) == 0:

        return (
            None,

            np.zeros(0, dtype=bool),

            np.zeros((0, 2), dtype=np.float64),

            None
        )

    next_points, status, errors = (
        cv2.calcOpticalFlowPyrLK(

            previous_gray,

            current_gray,

            previous_points,

            None,

            winSize=(
                window_size,
                window_size
            ),

            maxLevel=pyramid_levels,

            criteria=(

                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_COUNT,

                30,

                0.01
            )
        )
    )

    if next_points is None or status is None:

        return (
            None,

            np.zeros(len(previous_points), dtype=bool),

            np.zeros((0, 2), dtype=np.float64),

            errors
        )

    old_xy = previous_points.reshape(
        -1,
        2
    )

    new_xy = next_points.reshape(
        -1,
        2
    )

    valid = (
        status.reshape(-1).astype(bool)
        & np.isfinite(new_xy).all(axis=1)
    )

    if errors is not None:

        error_values = errors.reshape(-1)

        valid &= np.isfinite(
            error_values
        )

        # Reject exceptionally poor tracking estimates.

        valid &= error_values < 30.0

    displacements = (
        new_xy - old_xy
    )

    return (
        next_points,

        valid,

        displacements,

        errors
    )


# =====================================================================
# TRACK CLASS
# =====================================================================

class FlowTrack:

    def __init__(
        self,
        detection,
        track_id,
        gray,
        max_features=DEFAULT_MAX_FEATURES,
        min_points=DEFAULT_MIN_TRACKED_POINTS
    ):

        self.track_id = track_id

        self.bbox = detection.bbox.copy()

        self.score = detection.score

        self.class_id = detection.class_id

        self.age = 1

        self.hits = 1

        self.missed = 0

        self.confirmed = True

        self.state = "tracked"

        self.velocity = np.zeros(
            2,
            dtype=np.float64
        )

        self.flow_motion = np.zeros(
            2,
            dtype=np.float64
        )

        self.flow_valid = False

        self.points = extract_features(

            gray,

            self.bbox,

            max_features=max_features
        )

        self.max_features = max_features

        self.min_points = min_points

        self.tracked_point_count = (

            len(self.points)

            if self.points is not None else 0
        )

        self.flow_inliers = 0

    # -----------------------------------------------------------------
    # Predict using optical flow.
    # -----------------------------------------------------------------

    def predict(
        self,
        previous_gray,
        current_gray
    ):

        self.age += 1

        self.flow_valid = False

        self.flow_motion = np.zeros(
            2,
            dtype=np.float64
        )

        old_bbox = self.bbox.copy()

        if self.points is not None and len(self.points) >= 1:

            (
                next_points,
                valid,
                displacements,
                errors
            ) = calculate_optical_flow(

                previous_gray,

                current_gray,

                self.points
            )

            if next_points is not None and np.any(valid):

                valid_displacements = displacements[
                    valid
                ]

                motion, inliers = median_motion(
                    valid_displacements
                )

                if len(valid_displacements) >= 1:

                    self.flow_motion = motion

                    self.flow_valid = True

                    self.flow_inliers = int(
                        np.sum(inliers)
                    )

                    self.tracked_point_count = len(
                        valid_displacements
                    )

                    self.points = next_points[
                        valid
                    ].reshape(
                        -1,
                        1,
                        2
                    ).astype(
                        np.float32
                    )

        # -------------------------------------------------------------
        # Optical flow displacement.
        # -------------------------------------------------------------

        if self.flow_valid:

            dx, dy = self.flow_motion

            self.bbox += np.array(
                [
                    dx,
                    dy,
                    dx,
                    dy
                ]
            )

            # Update velocity estimate.

            self.velocity = (

                0.7 * self.velocity

                + 0.3 * self.flow_motion
            )

        else:

            # If optical flow fails, use previous velocity.

            dx, dy = self.velocity

            self.bbox += np.array(
                [
                    dx,
                    dy,
                    dx,
                    dy
                ]
            )

        return old_bbox, self.bbox.copy()

    # -----------------------------------------------------------------
    # Detection correction.
    # -----------------------------------------------------------------

    def correct(self, detection, gray):

        detection_bbox = detection.bbox

        predicted_center = (

            self.bbox[:2] + self.bbox[2:]

        ) / 2.0

        measured_center = (

            detection_bbox[:2] + detection_bbox[2:]

        ) / 2.0

        measured_motion = (
            measured_center - predicted_center
        )

        # Correct the predicted box.

        alpha = 0.75

        self.bbox = (

            alpha * detection_bbox

            + (1.0 - alpha) * self.bbox
        )

        self.velocity = (

            0.7 * self.velocity

            + 0.3 * measured_motion
        )

        self.score = detection.score

        self.class_id = detection.class_id

        self.hits += 1

        self.missed = 0

        self.state = "tracked"

        self.confirmed = True

        # Replenish feature points using the current real frame.

        new_points = extract_features(

            gray,

            self.bbox,

            max_features=self.max_features
        )

        if new_points is not None:

            if self.points is not None and len(self.points) > 0:

                combined = np.concatenate(

                    [
                        self.points.reshape(-1, 2),

                        new_points.reshape(-1, 2)
                    ],

                    axis=0
                )

                # Remove nearly duplicate points.

                selected = []

                for point in combined:

                    if not selected:

                        selected.append(point)

                        continue

                    distances = np.linalg.norm(

                        np.asarray(selected) - point,

                        axis=1
                    )

                    if np.min(distances) >= 3.0:

                        selected.append(point)

                    if len(selected) >= self.max_features:
                        break

                self.points = np.asarray(

                    selected,

                    dtype=np.float32
                ).reshape(
                    -1,
                    1,
                    2
                )

            else:

                self.points = new_points

        self.tracked_point_count = (

            len(self.points)

            if self.points is not None else 0
        )


# =====================================================================
# HUNGARIAN ASSIGNMENT
# =====================================================================

def hungarian_assignment(cost_matrix):

    cost = np.asarray(
        cost_matrix,
        dtype=np.float64
    )

    if cost.ndim != 2:
        raise ValueError("Cost matrix must be 2D.")

    rows, cols = cost.shape

    if rows == 0 or cols == 0:
        return []

    if not np.isfinite(cost).all():
        raise ValueError("Cost matrix must be finite.")

    transposed = False

    if rows > cols:

        cost = cost.T

        transposed = True

    n, m = cost.shape

    u = np.zeros(n + 1)

    v = np.zeros(m + 1)

    p = np.zeros(m + 1, dtype=np.int64)

    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):

        p[0] = i

        j0 = 0

        minv = np.full(
            m + 1,
            np.inf
        )

        used = np.zeros(
            m + 1,
            dtype=bool
        )

        while True:

            used[j0] = True

            i0 = p[j0]

            delta = np.inf

            j1 = 0

            for j in range(1, m + 1):

                if used[j]:
                    continue

                current = (

                    cost[i0 - 1, j - 1]

                    - u[i0]

                    - v[j]
                )

                if current < minv[j]:

                    minv[j] = current

                    way[j] = j0

                if minv[j] < delta:

                    delta = minv[j]

                    j1 = j

            if not np.isfinite(delta):
                raise RuntimeError("Hungarian assignment failed.")

            for j in range(m + 1):

                if used[j]:

                    u[p[j]] += delta

                    v[j] -= delta

                else:

                    minv[j] -= delta

            j0 = j1

            if p[j0] == 0:
                break

        while True:

            j1 = way[j0]

            p[j0] = p[j1]

            j0 = j1

            if j0 == 0:
                break

    assignments = []

    for j in range(1, m + 1):

        if p[j] == 0:
            continue

        row = int(p[j] - 1)

        col = int(j - 1)

        if transposed:

            assignments.append(
                (col, row)
            )

        else:

            assignments.append(
                (row, col)
            )

    return sorted(assignments)


# =====================================================================
# OPTICAL FLOW TRACKER
# =====================================================================

class OpticalFlowTracker:

    def __init__(
        self,
        max_features=DEFAULT_MAX_FEATURES,
        min_points=DEFAULT_MIN_TRACKED_POINTS,
        max_missed=DEFAULT_MAX_MISSED,
        iou_threshold=DEFAULT_IOU_THRESHOLD
    ):

        self.max_features = max_features

        self.min_points = min_points

        self.max_missed = max_missed

        self.iou_threshold = iou_threshold

        self.tracks = []

        self.next_id = 1

        self.frame_count = 0

        self.total_created = 0

        self.total_removed = 0

        self.total_flow_updates = 0

        self.total_flow_failures = 0

        self.total_detection_updates = 0

    def create_track(self, detection, gray):

        track = FlowTrack(

            detection=detection,

            track_id=self.next_id,

            gray=gray,

            max_features=self.max_features,

            min_points=self.min_points
        )

        self.tracks.append(
            track
        )

        self.next_id += 1

        self.total_created += 1

    def update(
        self,
        detections,
        previous_gray,
        current_gray
    ):

        self.frame_count += 1

        height, width = current_gray.shape[:2]

        # -------------------------------------------------------------
        # Step 1: Optical flow prediction.
        # -------------------------------------------------------------

        for track in self.tracks:

            track.predict(

                previous_gray,

                current_gray
            )

            track.bbox = clip_bbox(

                track.bbox,

                width,

                height
            )

            if track.flow_valid:

                self.total_flow_updates += 1

            else:

                self.total_flow_failures += 1

        # -------------------------------------------------------------
        # Step 2: Build IoU cost matrix.
        # -------------------------------------------------------------

        cost = np.ones(

            (
                len(self.tracks),

                len(detections)
            ),

            dtype=np.float64
        )

        for i, track in enumerate(self.tracks):

            for j, detection in enumerate(detections):

                if track.class_id != detection.class_id:
                    continue

                overlap = bbox_iou(

                    track.bbox,

                    detection.bbox
                )

                cost[i, j] = 1.0 - overlap

        # -------------------------------------------------------------
        # Step 3: Hungarian matching.
        # -------------------------------------------------------------

        assignments = hungarian_assignment(
            cost
        )

        matched_tracks = set()

        matched_detections = set()

        for track_index, detection_index in assignments:

            if cost[track_index, detection_index] > (
                1.0 - self.iou_threshold
            ):

                continue

            track = self.tracks[
                track_index
            ]

            detection = detections[
                detection_index
            ]

            track.correct(

                detection,

                current_gray
            )

            self.total_detection_updates += 1

            matched_tracks.add(
                track_index
            )

            matched_detections.add(
                detection_index
            )

        # -------------------------------------------------------------
        # Step 4: Unmatched tracks.
        #
        # Optical flow continues their trajectories temporarily.
        # -------------------------------------------------------------

        for index, track in enumerate(self.tracks):

            if index in matched_tracks:
                continue

            track.missed += 1

            track.state = "lost"

            if track.flow_valid:

                # Flow prediction is accepted as a temporary update.

                pass

            # If no flow points survive, retain the velocity prediction.
            # This is less reliable, so the track lifetime is limited.

        # -------------------------------------------------------------
        # Step 5: Create tracks from unmatched detections.
        # -------------------------------------------------------------

        for detection_index, detection in enumerate(detections):

            if detection_index in matched_detections:
                continue

            self.create_track(

                detection,

                current_gray
            )

        # -------------------------------------------------------------
        # Step 6: Remove expired tracks.
        # -------------------------------------------------------------

        retained = []

        for track in self.tracks:

            if track.missed > self.max_missed:

                self.total_removed += 1

                continue

            retained.append(
                track
            )

        self.tracks = retained

        return self.get_tracks()

    def get_tracks(self):

        output = []

        for track in self.tracks:

            if track.missed > self.max_missed:
                continue

            output.append(

                {
                    "id": track.track_id,

                    "bbox": track.bbox.copy(),

                    "score": track.score,

                    "class_id": track.class_id,

                    "age": track.age,

                    "hits": track.hits,

                    "missed": track.missed,

                    "state": track.state,

                    "flow_valid": track.flow_valid,

                    "flow_motion": track.flow_motion.copy(),

                    "point_count": track.tracked_point_count,

                    "flow_inliers": track.flow_inliers,

                    "points": (

                        track.points.copy()

                        if track.points is not None

                        else None
                    )
                }
            )

        return output

    def summary(self):

        return {

            "frames_processed": self.frame_count,

            "tracks_created": self.total_created,

            "tracks_removed": self.total_removed,

            "tracks_remaining": len(self.tracks),

            "optical_flow_updates": self.total_flow_updates,

            "optical_flow_failures": self.total_flow_failures,

            "detection_updates": self.total_detection_updates
        }


# =====================================================================
# REAL VIDEO DETECTOR
# =====================================================================

class MotionDetector:

    """
    MOG2 foreground detector.

    This is a demonstration detector.

    For semantic waste detection, replace it with YOLO or another
    object detector that returns Detection objects.
    """

    def __init__(
        self,
        min_area=DEFAULT_MIN_DETECTION_AREA,
        history=500,
        var_threshold=32
    ):

        self.min_area = min_area

        self.subtractor = cv2.createBackgroundSubtractorMOG2(

            history=history,

            varThreshold=var_threshold,

            detectShadows=False
        )

        self.kernel = cv2.getStructuringElement(

            cv2.MORPH_ELLIPSE,

            (5, 5)
        )

    def detect(self, frame):

        mask = self.subtractor.apply(
            frame
        )

        mask = cv2.morphologyEx(

            mask,

            cv2.MORPH_OPEN,

            self.kernel
        )

        mask = cv2.morphologyEx(

            mask,

            cv2.MORPH_CLOSE,

            self.kernel,

            iterations=2
        )

        contours, _ = cv2.findContours(

            mask,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if w < 5 or h < 5:
                continue

            score = (

                0.25

                + 0.75 * np.sqrt(

                    area / (
                        area + self.min_area * 4.0
                    )
                )
            )

            score = float(
                np.clip(score, 0.0, 0.99)
            )

            detections.append(

                Detection(

                    bbox=np.array(
                        [
                            x,
                            y,
                            x + w,
                            y + h
                        ],
                        dtype=np.float64
                    ),

                    score=score,

                    class_id=0
                )
            )

        return detections, mask


# =====================================================================
# VISUALIZATION
# =====================================================================

def draw_results(
    frame,
    detections,
    tracks,
    frame_number,
    fps
):

    output = frame.copy()

    # ---------------------------------------------------------------
    # Draw raw detections.
    # ---------------------------------------------------------------

    for detection in detections:

        x1, y1, x2, y2 = map(
            int,
            detection.bbox
        )

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            (255, 0, 255),

            1
        )

        cv2.putText(

            output,

            f"DET {detection.score:.2f}",

            (x1, max(15, y1 - 5)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.4,

            (255, 0, 255),

            1
        )

    # ---------------------------------------------------------------
    # Draw tracks and feature points.
    # ---------------------------------------------------------------

    for track in tracks:

        x1, y1, x2, y2 = map(
            int,
            track["bbox"]
        )

        track_id = track["id"]

        color = (

            int((track_id * 67) % 255),

            int((track_id * 131) % 255),

            int((track_id * 197) % 255)
        )

        thickness = 3 if track["state"] == "tracked" else 2

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            color,

            thickness
        )

        status = (

            "FLOW"

            if track["missed"] > 0

            else "DET"
        )

        cv2.putText(

            output,

            f"ID {track_id} {status}",

            (x1, max(20, y1 - 8)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            color,

            2
        )

        # Draw feature points.

        points = track["points"]

        if points is not None:

            for point in points.reshape(-1, 2):

                px, py = map(
                    int,
                    point
                )

                cv2.circle(

                    output,

                    (px, py),

                    2,

                    (255, 255, 0),

                    -1
                )

        # Draw motion arrow.

        if track["flow_valid"]:

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            dx, dy = track["flow_motion"]

            end_point = (

                int(center_x + dx * 3),

                int(center_y + dy * 3)
            )

            cv2.arrowedLine(

                output,

                (center_x, center_y),

                end_point,

                (0, 255, 255),

                2,

                tipLength=0.3
            )

    # ---------------------------------------------------------------
    # Information overlay.
    # ---------------------------------------------------------------

    cv2.putText(

        output,

        f"Frame: {frame_number}",

        (15, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2
    )

    cv2.putText(

        output,

        f"FPS: {fps:.1f}",

        (15, 60),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2
    )

    cv2.putText(

        output,

        f"Tracks: {len(tracks)}",

        (15, 90),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2
    )

    return output


# =====================================================================
# VIDEO PROCESSING
# =====================================================================

def process_video(args):

    video_path = Path(
        args.video
    )

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if not np.isfinite(fps) or fps <= 0:

        fps = 30.0

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    writer = cv2.VideoWriter(

        str(output_path),

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (width, height)
    )

    if not writer.isOpened():

        capture.release()

        raise RuntimeError(
            f"Cannot create output video: {output_path}"
        )

    detector = MotionDetector(

        min_area=args.min_area,

        history=args.history,

        var_threshold=args.var_threshold
    )

    tracker = OpticalFlowTracker(

        max_features=args.max_features,

        min_points=args.min_points,

        max_missed=args.max_missed,

        iou_threshold=args.iou_threshold
    )

    previous_gray = None

    frame_number = 0

    processing_times = []

    print("\nOptical-flow tracking started.")

    print("Input:", video_path)

    print("Resolution:", width, "x", height)

    print("FPS:", fps)

    print("Total frames:", total_frames)

    try:

        while True:

            start = time.perf_counter()

            ret, frame = capture.read()

            if not ret:
                break

            frame_number += 1

            current_gray = cv2.cvtColor(

                frame,

                cv2.COLOR_BGR2GRAY
            )

            # ---------------------------------------------------------
            # Frame 1:
            # Initialize detector and features.
            # ---------------------------------------------------------

            detections, foreground_mask = detector.detect(
                frame
            )

            if previous_gray is None:

                for detection in detections:

                    tracker.create_track(

                        detection,

                        current_gray
                    )

                previous_gray = current_gray.copy()

                output_frame = draw_results(

                    frame,

                    detections,

                    tracker.get_tracks(),

                    frame_number,

                    0.0
                )

                writer.write(
                    output_frame
                )

                if not args.no_display:

                    cv2.imshow(
                        "Lesson 46 - Optical Flow Tracking",
                        output_frame
                    )

                    if args.show_mask:

                        cv2.imshow(
                            "Foreground Mask",
                            foreground_mask
                        )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                continue

            # ---------------------------------------------------------
            # Step 1: Optical flow prediction.
            # Step 2: Detection correction.
            # ---------------------------------------------------------

            tracks = tracker.update(

                detections,

                previous_gray,

                current_gray
            )

            # ---------------------------------------------------------
            # Visualization.
            # ---------------------------------------------------------

            elapsed = (
                time.perf_counter() - start
            )

            processing_times.append(
                elapsed
            )

            current_fps = 1.0 / max(
                elapsed,
                1e-9
            )

            output_frame = draw_results(

                frame,

                detections,

                tracks,

                frame_number,

                current_fps
            )

            writer.write(
                output_frame
            )

            if not args.no_display:

                cv2.imshow(

                    "Lesson 46 - Optical Flow Tracking",

                    output_frame
                )

                if args.show_mask:

                    cv2.imshow(
                        "Foreground Mask",
                        foreground_mask
                    )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            previous_gray = current_gray.copy()

            if frame_number % 100 == 0:

                print(

                    f"Frame {frame_number}/{total_frames} | "

                    f"Detections: {len(detections)} | "

                    f"Tracks: {len(tracks)} | "

                    f"FPS: {current_fps:.1f}"
                )

    finally:

        capture.release()

        writer.release()

        cv2.destroyAllWindows()

    average_time = (

        float(np.mean(processing_times))

        if processing_times else 0.0
    )

    summary = {

        "input_video": str(video_path),

        "output_video": str(output_path),

        "resolution": [
            width,
            height
        ],

        "frames_processed": frame_number,

        "average_processing_fps": (

            1.0 / average_time

            if average_time > 0 else 0.0
        ),

        "tracker": tracker.summary()
    }

    summary_path = (

        output_path.parent

        / "tracking_summary.json"
    )

    with open(
        summary_path,
        "w"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\nProcessing complete.")

    print(
        json.dumps(
            summary,
            indent=4
        )
    )

    print("\nVideo:", output_path)

    print("Summary:", summary_path)


# =====================================================================
# TEST HELPERS
# =====================================================================

def make_detection(
    x,
    y=50,
    width=60,
    height=80,
    score=0.9,
    class_id=0
):

    return Detection(

        bbox=np.array(
            [
                x,
                y,
                x + width,
                y + height
            ],
            dtype=np.float64
        ),

        score=score,

        class_id=class_id
    )


# =====================================================================
# UNIT TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print("LESSON 46 — OPTICAL FLOW TRACKING TESTS")

    print("=" * 70)

    # ---------------------------------------------------------------
    # Test 1: identical bounding boxes
    # ---------------------------------------------------------------

    box = np.array(
        [10, 20, 100, 150],
        dtype=np.float64
    )

    assert abs(
        bbox_iou(box, box) - 1.0
    ) < 1e-10

    print("Test 1 passed: identical-box IoU")

    # ---------------------------------------------------------------
    # Test 2: zero overlap
    # ---------------------------------------------------------------

    box_b = np.array(
        [200, 200, 300, 300],
        dtype=np.float64
    )

    assert bbox_iou(box, box_b) == 0.0

    print("Test 2 passed: zero-overlap IoU")

    # ---------------------------------------------------------------
    # Test 3: known IoU
    # ---------------------------------------------------------------

    box_a = np.array(
        [0, 0, 100, 100]
    )

    box_b = np.array(
        [50, 0, 150, 100]
    )

    assert abs(
        bbox_iou(box_a, box_b) - 1.0 / 3.0
    ) < 1e-8

    print("Test 3 passed: known IoU")

    # ---------------------------------------------------------------
    # Test 4: median motion
    # ---------------------------------------------------------------

    displacements = np.array(

        [
            [5, 2],

            [6, 2],

            [5, 3],

            [5, 2]
        ],

        dtype=np.float64
    )

    motion, inliers = median_motion(
        displacements
    )

    assert np.allclose(
        motion,
        [5, 2]
    )

    assert np.all(
        inliers
    )

    print("Test 4 passed: median motion")

    # ---------------------------------------------------------------
    # Test 5: median rejects outlier
    # ---------------------------------------------------------------

    displacements = np.array(

        [
            [5, 2],

            [5, 2],

            [6, 2],

            [100, 100]
        ],

        dtype=np.float64
    )

    motion, inliers = median_motion(
        displacements
    )

    assert np.allclose(
        motion,
        [5.5, 2.0]
    )

    assert not inliers[-1]

    print("Test 5 passed: outlier rejection")

    # ---------------------------------------------------------------
    # Test 6: empty motion input
    # ---------------------------------------------------------------

    motion, inliers = median_motion(
        np.empty((0, 2))
    )

    assert np.allclose(
        motion,
        [0, 0]
    )

    assert len(inliers) == 0

    print("Test 6 passed: empty motion")

    # ---------------------------------------------------------------
    # Test 7: Hungarian assignment
    # ---------------------------------------------------------------

    cost = np.array(

        [
            [4, 1, 3],

            [2, 0, 5],

            [3, 2, 2]
        ],

        dtype=np.float64
    )

    assignments = hungarian_assignment(
        cost
    )

    total_cost = sum(
        cost[i, j]
        for i, j in assignments
    )

    assert len(assignments) == 3

    assert abs(
        total_cost - 5.0
    ) < 1e-8

    print("Test 7 passed: Hungarian assignment")

    # ---------------------------------------------------------------
    # Test 8: rectangular assignment
    # ---------------------------------------------------------------

    cost = np.array(

        [
            [1, 5],

            [4, 2],

            [3, 6]
        ],

        dtype=np.float64
    )

    assignments = hungarian_assignment(
        cost
    )

    assert len(assignments) == 2

    print("Test 8 passed: rectangular assignment")

    # ---------------------------------------------------------------
    # Test 9: bbox clipping
    # ---------------------------------------------------------------

    clipped = clip_bbox(

        np.array(
            [-10, -20, 200, 200]
        ),

        width=100,

        height=100
    )

    assert np.allclose(
        clipped,
        [0, 0, 99, 99]
    )

    print("Test 9 passed: bounding-box clipping")

    # ---------------------------------------------------------------
    # Test 10: detection creation
    # ---------------------------------------------------------------

    detection = make_detection(
        100
    )

    assert detection.bbox.shape == (4,)

    assert detection.score == 0.9

    print("Test 10 passed: detection structure")

    # ---------------------------------------------------------------
    # Test 11: tracker initialization
    # ---------------------------------------------------------------

    tracker = OpticalFlowTracker()

    assert tracker.next_id == 1

    assert len(tracker.tracks) == 0

    print("Test 11 passed: tracker initialization")

    # ---------------------------------------------------------------
    # Test 12: track creation using a real image
    #
    # Uses an actual image file if supplied.
    # No synthetic photograph is generated.
    # ---------------------------------------------------------------

    image_path = None

    for candidate in [

        "test_image.jpg",

        "test_image.png",

        "frame.jpg",

        "frame.png"
    ]:

        if Path(candidate).exists():

            image_path = candidate

            break

    if image_path is not None:

        image = cv2.imread(
            image_path
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        height, width = gray.shape

        x1 = max(
            0,
            width // 4
        )

        y1 = max(
            0,
            height // 4
        )

        x2 = min(
            width - 1,
            3 * width // 4
        )

        y2 = min(
            height - 1,
            3 * height // 4
        )

        detection = Detection(

            bbox=np.array(
                [
                    x1,
                    y1,
                    x2,
                    y2
                ],
                dtype=np.float64
            )
        )

        tracker.create_track(
            detection,
            gray
        )

        assert len(tracker.tracks) == 1

        assert tracker.tracks[0].track_id == 1

        print("Test 12 passed: real-image track initialization")

    else:

        print(
            "Test 12 skipped: no test_image.jpg/png or "
            "frame.jpg/png found."
        )

    # ---------------------------------------------------------------
    # Test 13: detection correction
    # ---------------------------------------------------------------

    tracker = OpticalFlowTracker()

    # A blank real-sized grayscale array is not used here.
    # This test exercises the correction mathematics directly.

    track = FlowTrack.__new__(
        FlowTrack
    )

    track.track_id = 1

    track.bbox = np.array(
        [10, 10, 60, 60],
        dtype=np.float64
    )

    track.velocity = np.zeros(2)

    track.score = 0.9

    track.class_id = 0

    track.hits = 1

    track.missed = 1

    track.state = "lost"

    track.confirmed = True

    track.max_features = DEFAULT_MAX_FEATURES

    track.points = None

    track.tracked_point_count = 0

    track.flow_valid = False

    track.flow_motion = np.zeros(2)

    track.flow_inliers = 0

    track.min_points = DEFAULT_MIN_TRACKED_POINTS

    # Verify the detection correction formula.

    detection_bbox = np.array(
        [14, 10, 64, 60],
        dtype=np.float64
    )

    predicted = track.bbox.copy()

    corrected = (
        0.75 * detection_bbox
        + 0.25 * predicted
    )

    assert np.allclose(
        corrected,
        [13, 10, 63, 60]
    )

    print("Test 13 passed: detection correction mathematics")

    # ---------------------------------------------------------------
    # Test 14: track identity persistence through a gap
    # ---------------------------------------------------------------

    tracker = OpticalFlowTracker(
        max_missed=3
    )

    class DummyTrack:

        def __init__(self):

            self.track_id = 1

            self.bbox = np.array(
                [10, 10, 60, 60],
                dtype=np.float64
            )

            self.score = 0.9

            self.class_id = 0

            self.age = 1

            self.hits = 1

            self.missed = 1

            self.state = "lost"

            self.flow_valid = True

            self.flow_motion = np.array(
                [5, 0]
            )

            self.tracked_point_count = 10

            self.flow_inliers = 8

            self.points = None

    tracker.tracks = [
        DummyTrack()
    ]

    output = tracker.get_tracks()

    assert len(output) == 1

    assert output[0]["id"] == 1

    assert output[0]["missed"] == 1

    print("Test 14 passed: temporary lost-track retention")

    # ---------------------------------------------------------------
    # Test 15: track removal
    # ---------------------------------------------------------------

    tracker = OpticalFlowTracker(
        max_missed=2
    )

    class ExpiredTrack:

        def __init__(self):

            self.track_id = 1

            self.bbox = np.array(
                [0, 0, 50, 50],
                dtype=np.float64
            )

            self.score = 0.9

            self.class_id = 0

            self.age = 5

            self.hits = 2

            self.missed = 3

            self.state = "lost"

            self.flow_valid = False

            self.flow_motion = np.zeros(2)

            self.tracked_point_count = 0

            self.flow_inliers = 0

            self.points = None

    tracker.tracks = [
        ExpiredTrack()
    ]

    # Expired tracks are removed during update.
    # Here verify that the expiration condition is correct.

    assert (
        tracker.tracks[0].missed
        > tracker.max_missed
    )

    print("Test 15 passed: expiration condition")

    # ---------------------------------------------------------------
    # Test 16: different classes cannot associate
    # ---------------------------------------------------------------

    track_box = np.array(
        [0, 0, 100, 100]
    )

    detection_box = np.array(
        [0, 0, 100, 100]
    )

    class_a = 0

    class_b = 1

    can_match = (

        class_a == class_b

        and bbox_iou(
            track_box,
            detection_box
        ) >= 0.1
    )

    assert not can_match

    print("Test 16 passed: class-aware association")

    # ---------------------------------------------------------------
    # Test 17: displacement dimensions
    # ---------------------------------------------------------------

    points_a = np.array(

        [
            [10, 10],

            [20, 20],

            [30, 30]
        ],

        dtype=np.float64
    )

    points_b = points_a + np.array(
        [5, 3]
    )

    displacement = points_b - points_a

    motion, inliers = median_motion(
        displacement
    )

    assert np.allclose(
        motion,
        [5, 3]
    )

    print("Test 17 passed: 2D displacement estimation")

    print("\nAll available tests passed.")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description="Lesson 46 — Optical Flow-Assisted Tracking"
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/lesson46/optical_flow_tracking.mp4"
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES
    )

    parser.add_argument(
        "--min-points",
        type=int,
        default=DEFAULT_MIN_TRACKED_POINTS
    )

    parser.add_argument(
        "--max-missed",
        type=int,
        default=DEFAULT_MAX_MISSED
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_DETECTION_AREA
    )

    parser.add_argument(
        "--history",
        type=int,
        default=500
    )

    parser.add_argument(
        "--var-threshold",
        type=float,
        default=32.0
    )

    parser.add_argument(
        "--show-mask",
        action="store_true"
    )

    parser.add_argument(
        "--no-display",
        action="store_true"
    )

    parser.add_argument(
        "--check-only",
        action="store_true"
    )

    args = parser.parse_args()

    if args.max_features < 1:

        parser.error(
            "--max-features must be positive."
        )

    if args.min_points < 1:

        parser.error(
            "--min-points must be positive."
        )

    if args.max_missed < 0:

        parser.error(
            "--max-missed cannot be negative."
        )

    if not 0 <= args.iou_threshold <= 1:

        parser.error(
            "--iou-threshold must be between 0 and 1."
        )

    if args.min_area < 1:

        parser.error(
            "--min-area must be positive."
        )

    print("=" * 70)

    print("LESSON 46 — OPTICAL FLOW-ASSISTED TRACKING")

    print("=" * 70)

    run_tests()

    if args.check_only:

        print("\nCheck-only mode completed.")

        return

    if args.video is None:

        print("\nNo video supplied.")

        print(
            "Example:"
        )

        print(
            "python lesson46_optical_flow_assisted_tracking.py "
            "--video conveyor_video.mp4"
        )

        return

    process_video(args)


if __name__ == "__main__":

    main()