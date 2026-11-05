"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 44 — BYTETRACK: LOW-CONFIDENCE OBJECT TRACKING
======================================================================

Filename:
    lesson44_bytetrack_low_confidence_tracking.py

PHASE:
    Phase 4 — Tracking and Visual Memory

======================================================================
LEARNING OBJECTIVES
======================================================================

Implement a ByteTrack-inspired multi-object tracker.

You will learn:

    1. High-confidence and low-confidence detection splitting
    2. Two-stage data association
    3. IoU-based matching
    4. Hungarian assignment
    5. Kalman motion prediction
    6. Track activation and confirmation
    7. Lost-track recovery
    8. Track expiration
    9. Why low-confidence detections are useful
    10. How to connect a real detector to ByteTrack

======================================================================
1. THE PROBLEM
======================================================================

Imagine a conveyor carrying waste objects.

Frame 1:
    Bottle A is detected with confidence 0.95.

Frame 2:
    Bottle A is partially occluded.
    Detector confidence falls to 0.28.

Frame 3:
    Bottle A becomes visible again.
    Detector confidence returns to 0.91.

A conventional tracker might discard Frame 2 because
the confidence is too low.

It may lose Bottle A's identity.

ByteTrack uses low-confidence detections to recover existing tracks.

======================================================================
2. TWO-STAGE ASSOCIATION
======================================================================

Split detections:

    HIGH:
        score >= HIGH_THRESHOLD

    LOW:
        LOW_THRESHOLD <= score < HIGH_THRESHOLD

Stage 1:
    Associate existing tracks with HIGH detections.

Stage 2:
    Associate remaining active tracks with LOW detections.

New tracks:
    Created from unmatched HIGH detections only.

This is the central ByteTrack idea.

======================================================================
3. IOU ASSOCIATION
======================================================================

For two bounding boxes:

    IoU = Intersection / Union

Matching cost:

    cost = 1 - IoU

Hungarian assignment minimizes total association cost.

======================================================================
4. KALMAN FILTER
======================================================================

State:

    [cx, cy, width, height,
     vx, vy, v_width, v_height]

Measurement:

    [cx, cy, width, height]

The filter predicts object positions between frames.

======================================================================
5. TRACK STATES
======================================================================

    NEW:
        Just created from a high-confidence detection.

    TRACKED:
        Successfully associated with a detection.

    LOST:
        Not matched in the current frame.

    REMOVED:
        Lost for longer than the configured lifetime.

Unconfirmed tracks must be matched again before being
reported as stable object identities.

======================================================================
6. IMPORTANT IMPLEMENTATION DISTINCTION
======================================================================

Original ByteTrack receives confidence scores from an object detector.

This standalone lesson uses OpenCV background subtraction for
the real-video example.

The video detector generates a heuristic score from the foreground
region area. It is NOT a neural detector confidence and does not
provide semantic object classification.

For a real robot, replace MotionDetector.detect() with:

    YOLO / Faster R-CNN / FCOS / another object detector

Return:

    [
        {
            "bbox": [x1, y1, x2, y2],
            "score": 0.92,
            "class_id": 0
        }
    ]

The ByteTrack association stages can then use actual detector scores.

======================================================================
7. ROBOTICS APPLICATION
======================================================================

Waste-sorting conveyor:

    Camera
      |
      v
    Object detector
      |
      v
    Detection confidence scores
      |
      v
    ByteTrack
      |
      v
    Persistent object IDs
      |
      v
    Object localization
      |
      v
    Robot grasp scheduling

Example:

    Frame 1: ID 1, bottle, score 0.95
    Frame 2: ID 1, bottle, score 0.31
    Frame 3: ID 1, bottle, score 0.92

The low-confidence observation can help maintain ID 1.

======================================================================
INSTALL
======================================================================

    pip install numpy opencv-python

======================================================================
RUN TESTS
======================================================================

    python lesson44_bytetrack_low_confidence_tracking.py --check-only

======================================================================
PROCESS A REAL VIDEO
======================================================================

    python lesson44_bytetrack_low_confidence_tracking.py \
        --video conveyor_video.mp4

Disable display:

    python lesson44_bytetrack_low_confidence_tracking.py \
        --video conveyor_video.mp4 \
        --no-display

Adjust thresholds:

    python lesson44_bytetrack_low_confidence_tracking.py \
        --video conveyor_video.mp4 \
        --track-threshold 0.60 \
        --low-threshold 0.10

======================================================================
OUTPUT
======================================================================

    runs/lesson44/
        bytetrack_output.mp4
        bytetrack_summary.json

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

DEFAULT_HIGH_THRESHOLD = 0.50

DEFAULT_LOW_THRESHOLD = 0.10

DEFAULT_MATCH_THRESHOLD = 0.80

DEFAULT_SECOND_MATCH_THRESHOLD = 0.50

DEFAULT_NEW_TRACK_THRESHOLD = 0.60

DEFAULT_MAX_TIME_LOST = 30

DEFAULT_N_INIT = 2


# =====================================================================
# DETECTION STRUCTURE
# =====================================================================

@dataclass
class Detection:

    bbox: np.ndarray

    score: float

    class_id: int = 0

    def __post_init__(self):

        self.bbox = np.asarray(
            self.bbox,
            dtype=np.float64
        )

        self.score = float(self.score)

        self.class_id = int(self.class_id)

        if self.bbox.shape != (4,):
            raise ValueError(
                "bbox must have shape (4,)."
            )

        if not np.isfinite(self.bbox).all():
            raise ValueError(
                "bbox must contain finite values."
            )

        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                "score must be between 0 and 1."
            )

        if self.bbox[2] <= self.bbox[0]:
            raise ValueError(
                "Bounding-box width must be positive."
            )

        if self.bbox[3] <= self.bbox[1]:
            raise ValueError(
                "Bounding-box height must be positive."
            )


# =====================================================================
# BOUNDING BOX UTILITIES
# =====================================================================

def bbox_to_measurement(bbox):

    x1, y1, x2, y2 = map(
        float,
        bbox
    )

    width = max(
        1.0,
        x2 - x1
    )

    height = max(
        1.0,
        y2 - y1
    )

    cx = (x1 + x2) / 2.0

    cy = (y1 + y2) / 2.0

    return np.array(
        [
            cx,
            cy,
            width,
            height
        ],
        dtype=np.float64
    )


def measurement_to_bbox(measurement):

    cx, cy, width, height = map(
        float,
        measurement
    )

    width = max(
        1.0,
        width
    )

    height = max(
        1.0,
        height
    )

    return np.array(
        [
            cx - width / 2.0,
            cy - height / 2.0,
            cx + width / 2.0,
            cy + height / 2.0
        ],
        dtype=np.float64
    )


# =====================================================================
# IOU
# =====================================================================

def bbox_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = map(
        float,
        box_a
    )

    bx1, by1, bx2, by2 = map(
        float,
        box_b
    )

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


# =====================================================================
# HUNGARIAN ASSIGNMENT
# =====================================================================

def hungarian_assignment(cost_matrix):

    """
    Rectangular Hungarian algorithm.

    Returns:
        List of (row, column) assignments.
    """

    cost = np.asarray(
        cost_matrix,
        dtype=np.float64
    )

    if cost.ndim != 2:
        raise ValueError(
            "Cost matrix must be two-dimensional."
        )

    rows, cols = cost.shape

    if rows == 0 or cols == 0:
        return []

    if not np.isfinite(cost).all():
        raise ValueError(
            "Cost matrix must be finite."
        )

    transposed = False

    if rows > cols:

        cost = cost.T

        transposed = True

    n, m = cost.shape

    u = np.zeros(
        n + 1,
        dtype=np.float64
    )

    v = np.zeros(
        m + 1,
        dtype=np.float64
    )

    p = np.zeros(
        m + 1,
        dtype=np.int64
    )

    way = np.zeros(
        m + 1,
        dtype=np.int64
    )

    for i in range(1, n + 1):

        p[0] = i

        j0 = 0

        minv = np.full(
            m + 1,
            np.inf,
            dtype=np.float64
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

                raise RuntimeError(
                    "Hungarian assignment failed."
                )

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

        row = int(
            p[j] - 1
        )

        col = int(
            j - 1
        )

        if transposed:

            assignments.append(
                (col, row)
            )

        else:

            assignments.append(
                (row, col)
            )

    assignments.sort()

    return assignments


# =====================================================================
# KALMAN FILTER
# =====================================================================

class KalmanFilter:

    """
    State:

        [cx, cy, width, height,
         vx, vy, v_width, v_height]

    Measurement:

        [cx, cy, width, height]
    """

    def __init__(self, measurement):

        self.x = np.zeros(
            8,
            dtype=np.float64
        )

        self.x[:4] = measurement

        self.P = np.eye(
            8,
            dtype=np.float64
        )

        self.P[:4, :4] *= 10.0

        self.P[4:, 4:] *= 100.0

        self.F = np.eye(
            8,
            dtype=np.float64
        )

        self.F[:4, 4:] = np.eye(4)

        self.H = np.zeros(
            (4, 8),
            dtype=np.float64
        )

        self.H[:4, :4] = np.eye(4)

        self.Q = np.diag(
            [
                0.1,
                0.1,
                0.1,
                0.1,
                0.01,
                0.01,
                0.01,
                0.01
            ]
        )

        self.R = np.diag(
            [
                1.0,
                1.0,
                4.0,
                4.0
            ]
        )

        self.I = np.eye(
            8,
            dtype=np.float64
        )

    def predict(self, dt=1.0):

        dt = float(dt)

        if dt <= 0:
            raise ValueError(
                "dt must be positive."
            )

        self.F = np.eye(
            8,
            dtype=np.float64
        )

        self.F[:4, 4:] = np.eye(4) * dt

        self.x = self.F @ self.x

        self.x[2:] = np.maximum(
            self.x[2:],
            1.0
        )

        self.P = (
            self.F @ self.P @ self.F.T
            + self.Q
        )

        return self.x.copy()

    def update(self, measurement):

        z = np.asarray(
            measurement,
            dtype=np.float64
        )

        if z.shape != (4,):
            raise ValueError(
                "Measurement must have shape (4,)."
            )

        innovation = (
            z - self.H @ self.x
        )

        S = (
            self.H @ self.P @ self.H.T
            + self.R
        )

        K = np.linalg.solve(
            S,
            self.H @ self.P
        ).T

        self.x = (
            self.x + K @ innovation
        )

        A = (
            self.I - K @ self.H
        )

        self.P = (
            A @ self.P @ A.T
            + K @ self.R @ K.T
        )

        self.P = (
            self.P + self.P.T
        ) / 2.0

        self.x[2:] = np.maximum(
            self.x[2:],
            1.0
        )

        return self.x.copy()

    def get_bbox(self):

        return measurement_to_bbox(
            self.x[:4]
        )


# =====================================================================
# TRACK
# =====================================================================

class Track:

    def __init__(
        self,
        detection,
        track_id,
        n_init=DEFAULT_N_INIT
    ):

        self.track_id = track_id

        measurement = bbox_to_measurement(
            detection.bbox
        )

        self.kalman = KalmanFilter(
            measurement
        )

        self.score = detection.score

        self.class_id = detection.class_id

        self.age = 1

        self.hits = 1

        self.time_since_update = 0

        self.n_init = n_init

        self.confirmed = (
            n_init <= 1
        )

        self.state = "new"

        self.last_score = detection.score

    def predict(self, dt=1.0):

        self.kalman.predict(dt)

        self.age += 1

        self.time_since_update += 1

        if self.time_since_update > 0:

            self.state = "lost"

    def update(self, detection):

        measurement = bbox_to_measurement(
            detection.bbox
        )

        self.kalman.update(
            measurement
        )

        self.score = detection.score

        self.last_score = detection.score

        self.class_id = detection.class_id

        self.hits += 1

        self.time_since_update = 0

        if self.hits >= self.n_init:

            self.confirmed = True

        self.state = (
            "tracked"
            if self.confirmed
            else "new"
        )

    def get_bbox(self):

        return self.kalman.get_bbox()


# =====================================================================
# ASSOCIATION
# =====================================================================

def build_iou_cost(tracks, detections):

    cost = np.ones(
        (
            len(tracks),
            len(detections)
        ),
        dtype=np.float64
    )

    for i, track in enumerate(tracks):

        predicted_box = track.get_bbox()

        for j, detection in enumerate(detections):

            # Do not associate different semantic classes.

            if track.class_id != detection.class_id:
                continue

            overlap = bbox_iou(
                predicted_box,
                detection.bbox
            )

            cost[i, j] = 1.0 - overlap

    return cost


def associate(
    tracks,
    detections,
    match_threshold
):

    if not tracks or not detections:

        return (
            [],
            set(range(len(tracks))),
            set(range(len(detections)))
        )

    cost = build_iou_cost(
        tracks,
        detections
    )

    assignments = hungarian_assignment(
        cost
    )

    matches = []

    unmatched_tracks = set(
        range(len(tracks))
    )

    unmatched_detections = set(
        range(len(detections))
    )

    for track_index, detection_index in assignments:

        if cost[track_index, detection_index] > match_threshold:

            continue

        matches.append(
            (track_index, detection_index)
        )

        unmatched_tracks.discard(
            track_index
        )

        unmatched_detections.discard(
            detection_index
        )

    return (
        matches,
        unmatched_tracks,
        unmatched_detections
    )


# =====================================================================
# BYTETRACK
# =====================================================================

class ByteTrack:

    def __init__(
        self,
        track_threshold=DEFAULT_HIGH_THRESHOLD,
        low_threshold=DEFAULT_LOW_THRESHOLD,
        match_threshold=DEFAULT_MATCH_THRESHOLD,
        second_match_threshold=DEFAULT_SECOND_MATCH_THRESHOLD,
        new_track_threshold=DEFAULT_NEW_TRACK_THRESHOLD,
        max_time_lost=DEFAULT_MAX_TIME_LOST,
        n_init=DEFAULT_N_INIT
    ):

        if not 0 <= low_threshold < track_threshold <= 1:

            raise ValueError(
                "Require 0 <= low_threshold < "
                "track_threshold <= 1."
            )

        self.track_threshold = track_threshold

        self.low_threshold = low_threshold

        self.match_threshold = match_threshold

        self.second_match_threshold = second_match_threshold

        self.new_track_threshold = new_track_threshold

        self.max_time_lost = max_time_lost

        self.n_init = n_init

        self.tracks = []

        self.next_id = 1

        self.frame_count = 0

        self.total_created = 0

        self.total_removed = 0

        self.total_high_matches = 0

        self.total_low_matches = 0

        self.total_unmatched_low = 0

    def create_track(self, detection):

        track = Track(

            detection=detection,

            track_id=self.next_id,

            n_init=self.n_init
        )

        self.tracks.append(
            track
        )

        self.next_id += 1

        self.total_created += 1

    def update(self, detections, dt=1.0):

        self.frame_count += 1

        # -------------------------------------------------------------
        # Step 1: Split detections into high and low confidence.
        # -------------------------------------------------------------

        high_detections = [

            detection

            for detection in detections

            if detection.score >= self.track_threshold
        ]

        low_detections = [

            detection

            for detection in detections

            if (
                self.low_threshold
                <= detection.score
                < self.track_threshold
            )
        ]

        # -------------------------------------------------------------
        # Step 2: Predict existing tracks.
        # -------------------------------------------------------------

        for track in self.tracks:

            track.predict(dt)

        # -------------------------------------------------------------
        # Step 3: First association.
        #
        # Existing tracked and lost tracks are associated
        # with high-confidence detections.
        # -------------------------------------------------------------

        stage1_tracks = [

            track

            for track in self.tracks

            if track.confirmed
        ]

        matches1, unmatched1_tracks, unmatched1_detections = (
            associate(

                stage1_tracks,

                high_detections,

                self.match_threshold
            )
        )

        self.total_high_matches += len(
            matches1
        )

        for track_index, detection_index in matches1:

            stage1_tracks[track_index].update(
                high_detections[detection_index]
            )

        matched_track_objects = {

            id(stage1_tracks[track_index])

            for track_index, _ in matches1
        }

        # -------------------------------------------------------------
        # Step 4: Second association.
        #
        # Only unmatched tracks that were active immediately
        # before this frame are considered here.
        #
        # Lost tracks from earlier frames are not matched
        # against low-confidence detections.
        # -------------------------------------------------------------

        second_stage_tracks = [

            stage1_tracks[index]

            for index in unmatched1_tracks

            if stage1_tracks[index].time_since_update == 1
        ]

        unmatched_low_matches = []

        if second_stage_tracks and low_detections:

            matches2, unmatched2_tracks, unmatched2_detections = (
                associate(

                    second_stage_tracks,

                    low_detections,

                    self.second_match_threshold
                )
            )

            self.total_low_matches += len(
                matches2
            )

            for track_index, detection_index in matches2:

                second_stage_tracks[track_index].update(
                    low_detections[detection_index]
                )

                matched_track_objects.add(
                    id(second_stage_tracks[track_index])
                )

            unmatched_low_matches = [
                low_detections[index]
                for index in unmatched2_detections
            ]

        else:

            unmatched_low_matches = list(
                low_detections
            )

        self.total_unmatched_low += len(
            unmatched_low_matches
        )

        # -------------------------------------------------------------
        # Step 5: Handle unconfirmed tracks.
        #
        # Unconfirmed tracks are matched against remaining
        # high-confidence detections.
        # -------------------------------------------------------------

        unconfirmed_tracks = [

            track

            for track in self.tracks

            if not track.confirmed
        ]

        remaining_high = [

            high_detections[index]

            for index in unmatched1_detections
        ]

        if unconfirmed_tracks and remaining_high:

            matches3, unmatched3_tracks, unmatched3_detections = (
                associate(

                    unconfirmed_tracks,

                    remaining_high,

                    self.match_threshold
                )
            )

            for track_index, detection_index in matches3:

                unconfirmed_tracks[track_index].update(
                    remaining_high[detection_index]
                )

                matched_track_objects.add(
                    id(unconfirmed_tracks[track_index])
                )

            # Unconfirmed tracks that are not matched are removed.

            for track_index in unmatched3_tracks:

                unconfirmed_tracks[track_index].state = "removed"

            remaining_high = [

                remaining_high[index]

                for index in unmatched3_detections
            ]

        else:

            # Unconfirmed tracks must be confirmed quickly.

            for track in unconfirmed_tracks:

                if id(track) not in matched_track_objects:

                    track.state = "removed"

        # -------------------------------------------------------------
        # Step 6: Create new tracks.
        #
        # Only sufficiently confident detections can create
        # new object identities.
        # -------------------------------------------------------------

        for detection in remaining_high:

            if detection.score < self.new_track_threshold:
                continue

            self.create_track(
                detection
            )

        # -------------------------------------------------------------
        # Step 7: Track expiration.
        # -------------------------------------------------------------

        retained = []

        for track in self.tracks:

            if track.state == "removed":

                self.total_removed += 1

                continue

            if track.time_since_update > self.max_time_lost:

                track.state = "removed"

                self.total_removed += 1

                continue

            retained.append(
                track
            )

        self.tracks = retained

        return self.get_active_tracks()

    def get_active_tracks(self):

        output = []

        for track in self.tracks:

            if not track.confirmed:
                continue

            if track.time_since_update != 0:
                continue

            output.append(
                {
                    "id": track.track_id,

                    "bbox": track.get_bbox(),

                    "score": track.score,

                    "class_id": track.class_id,

                    "hits": track.hits,

                    "age": track.age,

                    "state": track.state
                }
            )

        return output

    def summary(self):

        return {

            "frames_processed": self.frame_count,

            "tracks_created": self.total_created,

            "tracks_removed": self.total_removed,

            "tracks_remaining": len(self.tracks),

            "high_confidence_matches": self.total_high_matches,

            "low_confidence_matches": self.total_low_matches,

            "unmatched_low_detections": self.total_unmatched_low,

            "next_track_id": self.next_id
        }


# =====================================================================
# REAL VIDEO DETECTOR
# =====================================================================

class MotionDetector:

    """
    Educational real-video detector.

    Uses background subtraction to find moving regions.

    Its score is a heuristic based on region area.

    It is NOT a semantic object detector.
    """

    def __init__(
        self,
        min_area=300,
        history=500,
        var_threshold=32
    ):

        self.min_area = min_area

        self.subtractor = (
            cv2.createBackgroundSubtractorMOG2(

                history=history,

                varThreshold=var_threshold,

                detectShadows=False
            )
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

            # ---------------------------------------------------------
            # Heuristic score.
            #
            # Larger foreground regions generally receive higher
            # scores. This is only a demonstration of score splitting.
            #
            # Replace with actual detector confidence in production.
            # ---------------------------------------------------------

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

        detections.sort(

            key=lambda detection: detection.score,

            reverse=True
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

    # Draw high-confidence detections in yellow.

    for detection in detections:

        x1, y1, x2, y2 = map(
            int,
            detection.bbox
        )

        if detection.score >= DEFAULT_HIGH_THRESHOLD:

            color = (0, 255, 255)

        else:

            color = (255, 0, 255)

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            color,

            1
        )

        cv2.putText(

            output,

            f"{detection.score:.2f}",

            (x1, max(15, y1 - 5)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            color,

            1
        )

    # Draw confirmed tracks.

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

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            color,

            3
        )

        cv2.putText(

            output,

            f"ID {track_id} score={track['score']:.2f}",

            (x1, max(20, y1 - 8)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            color,

            2
        )

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
            f"Cannot create video: {output_path}"
        )

    detector = MotionDetector(

        min_area=args.min_area,

        history=args.history,

        var_threshold=args.var_threshold
    )

    tracker = ByteTrack(

        track_threshold=args.track_threshold,

        low_threshold=args.low_threshold,

        match_threshold=args.match_threshold,

        second_match_threshold=args.second_match_threshold,

        new_track_threshold=args.new_track_threshold,

        max_time_lost=args.max_time_lost,

        n_init=args.n_init
    )

    frame_number = 0

    processing_times = []

    print("\nByteTrack processing started.")

    print("Input:", video_path)

    print("Resolution:", width, "x", height)

    print("Video FPS:", fps)

    print("Total frames:", total_frames)

    print(
        "High threshold:",
        args.track_threshold
    )

    print(
        "Low threshold:",
        args.low_threshold
    )

    try:

        while True:

            start = time.perf_counter()

            ret, frame = capture.read()

            if not ret:
                break

            frame_number += 1

            # ---------------------------------------------------------
            # Step 1: Detection
            # ---------------------------------------------------------

            detections, foreground_mask = detector.detect(
                frame
            )

            # ---------------------------------------------------------
            # Step 2: ByteTrack
            # ---------------------------------------------------------

            active_tracks = tracker.update(
                detections
            )

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

            # ---------------------------------------------------------
            # Step 3: Visualization
            # ---------------------------------------------------------

            output_frame = draw_results(

                frame,

                detections,

                active_tracks,

                frame_number,

                current_fps
            )

            writer.write(
                output_frame
            )

            if not args.no_display:

                cv2.imshow(
                    "Lesson 44 - ByteTrack",
                    output_frame
                )

                if args.show_mask:

                    cv2.imshow(
                        "Foreground Mask",
                        foreground_mask
                    )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            if frame_number % 100 == 0:

                high_count = sum(

                    d.score >= args.track_threshold

                    for d in detections
                )

                low_count = sum(

                    args.low_threshold <= d.score < args.track_threshold

                    for d in detections
                )

                print(

                    f"Frame {frame_number}/{total_frames} | "

                    f"High: {high_count} | "

                    f"Low: {low_count} | "

                    f"Tracks: {len(active_tracks)} | "

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

        "resolution": [width, height],

        "frames_processed": frame_number,

        "average_processing_fps": (

            1.0 / average_time

            if average_time > 0 else 0.0
        ),

        "track_threshold": args.track_threshold,

        "low_threshold": args.low_threshold,

        "tracker": tracker.summary()
    }

    summary_path = (

        output_path.parent

        / "bytetrack_summary.json"
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
    score=0.9,
    y=0,
    width=50,
    height=50,
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

    print("LESSON 44 — BYTETRACK TESTS")

    print("=" * 70)

    # ---------------------------------------------------------------
    # Test 1: bounding-box conversion
    # ---------------------------------------------------------------

    box = np.array(
        [10, 20, 110, 220],
        dtype=np.float64
    )

    measurement = bbox_to_measurement(
        box
    )

    reconstructed = measurement_to_bbox(
        measurement
    )

    assert np.allclose(
        box,
        reconstructed
    )

    print("Test 1 passed: bounding-box conversion")

    # ---------------------------------------------------------------
    # Test 2: IoU
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

    print("Test 2 passed: IoU calculation")

    # ---------------------------------------------------------------
    # Test 3: identical boxes
    # ---------------------------------------------------------------

    assert abs(
        bbox_iou(box_a, box_a) - 1.0
    ) < 1e-8

    print("Test 3 passed: identical-box IoU")

    # ---------------------------------------------------------------
    # Test 4: no overlap
    # ---------------------------------------------------------------

    box_c = np.array(
        [200, 200, 300, 300]
    )

    assert bbox_iou(box_a, box_c) == 0.0

    print("Test 4 passed: zero-overlap IoU")

    # ---------------------------------------------------------------
    # Test 5: Hungarian assignment
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

    print("Test 5 passed: Hungarian assignment")

    # ---------------------------------------------------------------
    # Test 6: Kalman prediction
    # ---------------------------------------------------------------

    kalman = KalmanFilter(
        np.array(
            [100, 100, 50, 50],
            dtype=np.float64
        )
    )

    kalman.x[4] = 5.0

    kalman.x[5] = 3.0

    kalman.predict()

    assert abs(
        kalman.x[0] - 105
    ) < 1e-8

    assert abs(
        kalman.x[1] - 103
    ) < 1e-8

    print("Test 6 passed: Kalman prediction")

    # ---------------------------------------------------------------
    # Test 7: Kalman update
    # ---------------------------------------------------------------

    result = kalman.update(
        np.array(
            [108, 105, 50, 50],
            dtype=np.float64
        )
    )

    assert np.isfinite(
        result
    ).all()

    print("Test 7 passed: Kalman update")

    # ---------------------------------------------------------------
    # Test 8: high/low detection splitting
    # ---------------------------------------------------------------

    tracker = ByteTrack(
        track_threshold=0.5,
        low_threshold=0.1,
        n_init=1
    )

    detections = [

        make_detection(0, 0.9),

        make_detection(100, 0.4),

        make_detection(200, 0.05)
    ]

    high = [

        d for d in detections

        if d.score >= tracker.track_threshold
    ]

    low = [

        d for d in detections

        if tracker.low_threshold <= d.score < tracker.track_threshold
    ]

    assert len(high) == 1

    assert len(low) == 1

    print("Test 8 passed: confidence splitting")

    # ---------------------------------------------------------------
    # Test 9: only high-confidence detections create tracks
    # ---------------------------------------------------------------

    tracker = ByteTrack(
        track_threshold=0.5,
        low_threshold=0.1,
        new_track_threshold=0.6,
        n_init=1
    )

    tracker.update(
        [
            make_detection(0, 0.4)
        ]
    )

    assert len(tracker.tracks) == 0

    tracker.update(
        [
            make_detection(0, 0.9)
        ]
    )

    assert len(tracker.tracks) == 1

    print("Test 9 passed: high-confidence track creation")

    # ---------------------------------------------------------------
    # Test 10: low-confidence recovery
    # ---------------------------------------------------------------

    tracker = ByteTrack(
        track_threshold=0.5,
        low_threshold=0.1,
        match_threshold=0.8,
        second_match_threshold=0.5,
        new_track_threshold=0.6,
        n_init=1
    )

    tracker.update(
        [
            make_detection(0, 0.95)
        ]
    )

    original_id = tracker.tracks[0].track_id

    tracker.update(
        [
            make_detection(2, 0.30)
        ]
    )

    assert len(tracker.tracks) == 1

    assert tracker.tracks[0].track_id == original_id

    assert tracker.tracks[0].time_since_update == 0

    assert tracker.total_low_matches == 1

    print("Test 10 passed: low-confidence track recovery")

    # ---------------------------------------------------------------
    # Test 11: low-confidence detections do not create tracks
    # ---------------------------------------------------------------

    tracker = ByteTrack(
        track_threshold=0.5,
        low_threshold=0.1,
        n_init=1
    )

    tracker.update(
        [
            make_detection(0, 0.3)
        ]
    )

    assert len(tracker.tracks) == 0

    print("Test 11 passed: low detections cannot create tracks")

    # ---------------------------------------------------------------
    # Test 12: high-confidence identity persistence
    # ---------------------------------------------------------------

    tracker = ByteTrack(
        track_threshold=0.5,
        low_threshold=0.1,
        n_init=1
    )

    tracker.update(
        [
            make_detection(0, 0.9)
        ]
    )

    track_id = tracker.tracks[0].track_id

    tracker.update(
        [
            make_detection(3, 0.95)
        ]
    )

    assert len(tracker.tracks) == 1

    assert tracker.tracks[0].track_id == track_id

    assert tracker.tracks[0].hits == 2

    print("Test 12 passed: identity persistence")

    # ---------------------------------------------------------------
    # Test 13: two objects
    # ---------------------------------------------------------------

    tracker = ByteTrack(
        track_threshold=0.5,
        low_threshold=0.1,
        n_init=1
    )

    tracker.update(
        [
            make_detection(0, 0.9),

            make_detection(200, 0.9)
        ]
    )

    assert len(tracker.tracks) == 2

    ids = {
        track.track_id
        for track in tracker.tracks
    }

    assert len(ids) == 2

    print("Test 13 passed: multiple object identities")

    # ---------------------------------------------------------------
    # Test 14: different classes cannot match
    # ---------------------------------------------------------------

    tracks = [

        Track(
            make_detection(
                0,
                0.9,
                class_id=0
            ),
            track_id=1,
            n_init=1
        )
    ]

    detections = [

        make_detection(
            0,
            0.9,
            class_id=1
        )
    ]

    matches, unmatched_tracks, unmatched_detections = associate(

        tracks,

        detections,

        match_threshold=0.8
    )

    assert len(matches) == 0

    assert len(unmatched_tracks) == 1

    assert len(unmatched_detections) == 1

    print("Test 14 passed: class-aware association")

    # ---------------------------------------------------------------
    # Test 15: track expiration
    # ---------------------------------------------------------------

    tracker = ByteTrack(
        track_threshold=0.5,
        low_threshold=0.1,
        max_time_lost=2,
        n_init=1
    )

    tracker.update(
        [
            make_detection(0, 0.9)
        ]
    )

    tracker.update([])

    assert len(tracker.tracks) == 1

    tracker.update([])

    assert len(tracker.tracks) == 1

    tracker.update([])

    assert len(tracker.tracks) == 0

    print("Test 15 passed: track expiration")

    # ---------------------------------------------------------------
    # Test 16: unmatched low detections
    # ---------------------------------------------------------------

    tracker = ByteTrack(
        track_threshold=0.5,
        low_threshold=0.1,
        n_init=1
    )

    tracker.update(
        [
            make_detection(0, 0.9)
        ]
    )

    tracker.update(
        [
            make_detection(500, 0.3)
        ]
    )

    assert tracker.total_unmatched_low == 1

    assert len(tracker.tracks) == 1

    print("Test 16 passed: unmatched low detections")

    # ---------------------------------------------------------------
    # Test 17: confidence threshold validation
    # ---------------------------------------------------------------

    raised = False

    try:

        ByteTrack(
            track_threshold=0.1,
            low_threshold=0.5
        )

    except ValueError:

        raised = True

    assert raised

    print("Test 17 passed: threshold validation")

    print("\nAll 17 tests passed.")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description="Lesson 44 — ByteTrack"
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/lesson44/bytetrack_output.mp4"
    )

    parser.add_argument(
        "--track-threshold",
        type=float,
        default=DEFAULT_HIGH_THRESHOLD
    )

    parser.add_argument(
        "--low-threshold",
        type=float,
        default=DEFAULT_LOW_THRESHOLD
    )

    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD
    )

    parser.add_argument(
        "--second-match-threshold",
        type=float,
        default=DEFAULT_SECOND_MATCH_THRESHOLD
    )

    parser.add_argument(
        "--new-track-threshold",
        type=float,
        default=DEFAULT_NEW_TRACK_THRESHOLD
    )

    parser.add_argument(
        "--max-time-lost",
        type=int,
        default=DEFAULT_MAX_TIME_LOST
    )

    parser.add_argument(
        "--n-init",
        type=int,
        default=DEFAULT_N_INIT
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=300
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

    if not 0 <= args.low_threshold < args.track_threshold <= 1:

        parser.error(
            "Require 0 <= low-threshold < track-threshold <= 1."
        )

    if not 0 <= args.new_track_threshold <= 1:

        parser.error(
            "new-track-threshold must be between 0 and 1."
        )

    if args.max_time_lost < 0:

        parser.error(
            "max-time-lost cannot be negative."
        )

    if args.n_init < 1:

        parser.error(
            "n-init must be at least 1."
        )

    if args.min_area < 1:

        parser.error(
            "min-area must be positive."
        )

    print("=" * 70)

    print("LESSON 44 — BYTETRACK")

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
            "python lesson44_bytetrack_low_confidence_tracking.py "
            "--video conveyor_video.mp4"
        )

        return

    process_video(args)


if __name__ == "__main__":

    main()