"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 51 — LONG-TERM OBJECT TRACKING AND OCCLUSION HANDLING
======================================================================

Filename:
    lesson51_long_term_tracking_occlusion.py

======================================================================
LEARNING OBJECTIVES
======================================================================

1. Understand why short-term trackers lose object identities.
2. Implement a constant-velocity Kalman filter.
3. Predict object positions when detections disappear.
4. Maintain ACTIVE, LOST, and REMOVED track states.
5. Keep lost tracks in memory for a configurable time.
6. Recover lost tracks using appearance similarity.
7. Maintain an appearance gallery for each track.
8. Combine motion and appearance for data association.
9. Use the Hungarian algorithm for global assignment.
10. Evaluate track recovery with numerical tests.

======================================================================
1. THE OCCLUSION PROBLEM
======================================================================

Imagine two objects moving along a conveyor.

Frame 1:
    Object A -> ID 1
    Object B -> ID 2

Frame 2:
    Object A is visible.
    Object B is visible.

Frame 3:
    Object A is partially hidden behind Object B.

Frame 4:
    Object A is completely hidden.

Frame 5:
    Object A reappears.

A naive tracker may do this:

    Before occlusion: Object A -> ID 1
    After occlusion:  Object A -> ID 3

This is an identity switch.

A long-term tracker should instead preserve ID 1.

======================================================================
2. TRACK LIFECYCLE
======================================================================

ACTIVE:
    The object is currently being detected.

LOST:
    The object was previously detected but is currently missing.

REMOVED:
    The object has been missing longer than the allowed retention
    period and is no longer considered recoverable.

Transitions:

    ACTIVE -> LOST
    LOST   -> ACTIVE
    LOST   -> REMOVED

A new detection can also create a new ACTIVE track.

======================================================================
3. CONSTANT-VELOCITY KALMAN FILTER
======================================================================

We represent a bounding box using:

    z = [cx, cy, width, height]

The state vector is:

    x = [cx, cy, width, height,
         vx, vy, vwidth, vheight]

The motion model assumes approximately constant velocity:

    x(k+1) = F x(k) + noise

For time interval dt:

    F = [[I, dt*I],
         [0,    I ]]

The observation model is:

    z(k) = H x(k) + measurement_noise

    H = [I, 0]

Prediction:

    x_pred = F x

    P_pred = F P F.T + Q

Correction:

    y = z - H x_pred

    S = H P_pred H.T + R

    K = P_pred H.T inv(S)

    x = x_pred + K y

    P = (I - K H) P_pred

We use a numerically stable Joseph-form covariance update:

    P = (I-KH) P_pred (I-KH).T + K R K.T

======================================================================
4. MOTION GATING
======================================================================

A detection far from the predicted position should not be assigned
to the track merely because its appearance looks similar.

The innovation covariance is:

    S = H P H.T + R

The Mahalanobis distance is:

    d^2 = (z - Hx).T inv(S) (z - Hx)

A detection is rejected if its distance exceeds a threshold.

This is called motion gating.

======================================================================
5. APPEARANCE GALLERY
======================================================================

One appearance vector may not represent an object under all
viewing conditions.

We maintain several historical embeddings:

    Gallery = [e1, e2, e3, ..., en]

For a new detection embedding e:

    appearance_distance =
        min(cosine_distance(e, gallery_embedding))

Using the minimum distance allows a track to match one of its
previous appearance states.

The gallery is bounded to avoid unlimited memory growth.

IMPORTANT:
    Appearance matching is only as good as the embedding model.
    This lesson uses a handcrafted HSV histogram descriptor so the
    script can run without a pretrained or separately trained CNN.

    Replace this descriptor with the trained Lesson 49 CNN embedding
    when available.

======================================================================
6. ASSOCIATION COST
======================================================================

For track i and detection j:

    cost(i,j) =
        w_motion * normalized_motion_distance
        + w_appearance * appearance_distance
        + w_iou * (1 - IoU)

Impossible associations receive infinite cost.

The Hungarian algorithm finds a globally consistent assignment.

======================================================================
7. OCCLUSION RECOVERY
======================================================================

When a track misses a detection:

    1. Predict its next bounding box.
    2. Mark it LOST.
    3. Preserve its appearance gallery.
    4. Continue predicting for a limited number of frames.
    5. Compare new detections against lost-track galleries.
    6. Reactivate a matching track with its original ID.

======================================================================
8. IMPORTANT LIMITATIONS
======================================================================

- A constant-velocity model cannot predict abrupt motion perfectly.
- Similar-looking objects may be confused.
- Long occlusions increase uncertainty.
- A stationary camera is assumed by the MOG2 demonstration detector.
- MOG2 is not semantic object detection.
- A robust production tracker should use detector confidence,
  class labels, camera calibration, and a learned Re-ID model.

======================================================================
9. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    Overhead camera
          |
          v
    YOLO / instance segmentation
          |
          v
    Long-term tracker
          |
          v
    Object identity + trajectory
          |
          v
    Camera-to-robot coordinate mapping
          |
          v
    Grasp planning
          |
          v
    ACT policy

If a cup is temporarily hidden behind another waste item, the
tracker can preserve its identity and trajectory rather than
immediately assigning it a new ID.

======================================================================
DEPENDENCIES
======================================================================

    pip install numpy opencv-python scipy matplotlib

======================================================================
RUN
======================================================================

    python lesson51_long_term_tracking_occlusion.py --check-only

    python lesson51_long_term_tracking_occlusion.py \
        --video conveyor_video.mp4

    python lesson51_long_term_tracking_occlusion.py \
        --video conveyor_video.mp4 \
        --max-lost-frames 90

    python lesson51_long_term_tracking_occlusion.py \
        --video conveyor_video.mp4 \
        --no-display

======================================================================
OUTPUT
======================================================================

    runs/lesson51/
        long_term_tracking.mp4
        tracking_summary.json
        track_lifecycle.png
======================================================================
"""

import argparse
import json
import time

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment


# =====================================================================
# CONFIGURATION
# =====================================================================

STATE_DIM = 8

MEASUREMENT_DIM = 4

DEFAULT_MAX_LOST_FRAMES = 60

DEFAULT_MAX_MAHALANOBIS = 16.0

DEFAULT_MAX_APPEARANCE_DISTANCE = 0.45

DEFAULT_MAX_ASSOCIATION_COST = 0.90

DEFAULT_GALLERY_SIZE = 20

DEFAULT_MIN_HITS = 2


# =====================================================================
# TRACK STATES
# =====================================================================

class TrackState(Enum):

    ACTIVE = "active"

    LOST = "lost"

    REMOVED = "removed"


# =====================================================================
# BOUNDING BOX CONVERSIONS
# =====================================================================

def xyxy_to_measurement(box):

    """
    Convert:

        [x1, y1, x2, y2]

    into:

        [cx, cy, width, height]
    """

    x1, y1, x2, y2 = np.asarray(
        box,
        dtype=np.float64
    )

    width = max(
        1.0,
        x2 - x1
    )

    height = max(
        1.0,
        y2 - y1
    )

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0

    return np.array(

        [
            center_x,
            center_y,
            width,
            height
        ],

        dtype=np.float64
    )


def measurement_to_xyxy(measurement):

    cx, cy, width, height = np.asarray(
        measurement,
        dtype=np.float64
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


def box_center(box):

    x1, y1, x2, y2 = np.asarray(
        box,
        dtype=np.float64
    )

    return np.array(

        [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0
        ],

        dtype=np.float64
    )


def box_area(box):

    x1, y1, x2, y2 = np.asarray(
        box,
        dtype=np.float64
    )

    return (

        max(0.0, x2 - x1)

        * max(0.0, y2 - y1)
    )


def box_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b

    x1 = max(
        ax1,
        bx1
    )

    y1 = max(
        ay1,
        by1
    )

    x2 = min(
        ax2,
        bx2
    )

    y2 = min(
        ay2,
        by2
    )

    intersection = (

        max(0.0, x2 - x1)

        * max(0.0, y2 - y1)
    )

    union = (

        box_area(box_a)

        + box_area(box_b)

        - intersection
    )

    return intersection / max(
        union,
        1e-12
    )


# =====================================================================
# KALMAN FILTER
# =====================================================================

class KalmanBoxFilter:

    """
    Eight-dimensional constant-velocity Kalman filter.

    State:

        [cx, cy, w, h, vx, vy, vw, vh]

    Measurement:

        [cx, cy, w, h]

    The time step is one video frame unless explicitly changed.
    """

    def __init__(
        self,
        initial_box,
        dt=1.0
    ):

        self.dt = float(dt)

        self.F = np.eye(
            STATE_DIM,
            dtype=np.float64
        )

        self.F[:4, 4:] = (
            np.eye(4) * self.dt
        )

        self.H = np.zeros(

            (
                MEASUREMENT_DIM,
                STATE_DIM
            ),

            dtype=np.float64
        )

        self.H[:4, :4] = np.eye(4)

        measurement = xyxy_to_measurement(
            initial_box
        )

        self.x = np.zeros(
            STATE_DIM,
            dtype=np.float64
        )

        self.x[:4] = measurement

        # Initial uncertainty:
        # position is moderately certain, velocity is uncertain.

        self.P = np.diag(

            [

                10.0,
                10.0,
                10.0,
                10.0,

                100.0,
                100.0,
                100.0,
                100.0
            ]

        ).astype(
            np.float64
        )

        self.Q = np.diag(

            [

                1.0,
                1.0,
                2.0,
                2.0,

                0.5,
                0.5,
                1.0,
                1.0
            ]

        ).astype(
            np.float64
        )

        self.R = np.diag(

            [

                4.0,
                4.0,
                9.0,
                9.0
            ]

        ).astype(
            np.float64
        )

        self.last_innovation = np.zeros(
            MEASUREMENT_DIM
        )

        self.last_innovation_covariance = (

            self.H @ self.P @ self.H.T

            + self.R
        )

    @property
    def box(self):

        return measurement_to_xyxy(
            self.x[:4]
        )

    @property
    def center(self):

        return self.x[:2].copy()

    @property
    def velocity(self):

        return self.x[4:].copy()

    def predict(self):

        self.x = self.F @ self.x

        self.P = (

            self.F @ self.P @ self.F.T

            + self.Q
        )

        # Width and height must remain positive.

        self.x[2] = max(
            1.0,
            self.x[2]
        )

        self.x[3] = max(
            1.0,
            self.x[3]
        )

        self.P = (
            self.P + self.P.T
        ) / 2.0

        return self.box

    def innovation(self, measurement):

        z = np.asarray(
            measurement,
            dtype=np.float64
        )

        predicted_measurement = (
            self.H @ self.x
        )

        y = (
            z - predicted_measurement
        )

        S = (

            self.H @ self.P @ self.H.T

            + self.R
        )

        return y, S

    def mahalanobis_distance(self, measurement):

        y, S = self.innovation(
            measurement
        )

        try:

            solved = np.linalg.solve(
                S,
                y
            )

        except np.linalg.LinAlgError:

            solved = np.linalg.pinv(
                S
            ) @ y

        return float(
            y.T @ solved
        )

    def update(self, measurement):

        z = np.asarray(
            measurement,
            dtype=np.float64
        )

        y, S = self.innovation(z)

        self.last_innovation = y.copy()

        self.last_innovation_covariance = S.copy()

        try:

            PHt = (
                self.P @ self.H.T
            )

            K = np.linalg.solve(

                S,

                PHt.T
            ).T

        except np.linalg.LinAlgError:

            K = (

                self.P

                @ self.H.T

                @ np.linalg.pinv(S)
            )

        self.x = (
            self.x + K @ y
        )

        # Joseph-form covariance update.

        identity = np.eye(
            STATE_DIM
        )

        A = (
            identity - K @ self.H
        )

        self.P = (

            A @ self.P @ A.T

            + K @ self.R @ K.T
        )

        self.P = (
            self.P + self.P.T
        ) / 2.0

        self.x[2] = max(
            1.0,
            self.x[2]
        )

        self.x[3] = max(
            1.0,
            self.x[3]
        )

        return self.box


# =====================================================================
# APPEARANCE DESCRIPTOR
# =====================================================================

class HSVAppearanceDescriptor:

    """
    Handcrafted appearance descriptor.

    It uses HSV histograms from real image crops.

    This is not a learned Re-ID model.

    Advantages:
        - No pretrained weights.
        - No training checkpoint required.
        - Fast to compute.
        - Useful for understanding appearance matching.

    Limitations:
        - Sensitive to lighting and camera color changes.
        - Does not understand object shape semantically.
        - Similar-colored objects can have similar descriptors.
    """

    def __init__(

        self,

        bins=(8, 8, 4),

        gallery_size=DEFAULT_GALLERY_SIZE
    ):

        self.bins = bins

        self.gallery_size = gallery_size

    def extract(self, frame, box):

        height, width = frame.shape[:2]

        x1, y1, x2, y2 = np.asarray(
            box,
            dtype=np.float64
        ).astype(
            int
        )

        x1 = int(
            np.clip(x1, 0, width - 1)
        )

        y1 = int(
            np.clip(y1, 0, height - 1)
        )

        x2 = int(
            np.clip(x2, 0, width)
        )

        y2 = int(
            np.clip(y2, 0, height)
        )

        if x2 <= x1 or y2 <= y1:

            return np.zeros(
                np.prod(self.bins),
                dtype=np.float64
            )

        crop = frame[
            y1:y2,
            x1:x2
        ]

        hsv = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2HSV
        )

        histogram = cv2.calcHist(

            [hsv],

            [0, 1, 2],

            None,

            list(self.bins),

            [0, 180, 0, 256, 0, 256]
        )

        histogram = histogram.flatten().astype(
            np.float64
        )

        norm = np.linalg.norm(
            histogram
        )

        if norm > 1e-12:

            histogram /= norm

        return histogram


def cosine_distance(a, b):

    a = np.asarray(
        a,
        dtype=np.float64
    )

    b = np.asarray(
        b,
        dtype=np.float64
    )

    denominator = (

        np.linalg.norm(a)

        * np.linalg.norm(b)
    )

    if denominator < 1e-12:

        return 1.0

    similarity = (
        np.dot(a, b) / denominator
    )

    return float(
        1.0 - np.clip(similarity, -1.0, 1.0)
    )


# =====================================================================
# TRACK DATA STRUCTURE
# =====================================================================

@dataclass
class Track:

    track_id: int

    kalman: KalmanBoxFilter

    state: TrackState

    first_frame: int

    last_frame: int

    hits: int = 1

    missed_frames: int = 0

    age: int = 1

    appearance_gallery: list = field(
        default_factory=list
    )

    trajectory: list = field(
        default_factory=list
    )

    lifecycle: list = field(
        default_factory=list
    )

    last_detection_frame: int = 0

    def __post_init__(self):

        self.trajectory.append(
            self.kalman.center.tolist()
        )

        self.lifecycle.append(

            {

                "frame": self.first_frame,

                "state": self.state.value
            }
        )

    @property
    def bbox(self):

        return self.kalman.box

    def add_appearance(

        self,

        embedding,

        gallery_size=DEFAULT_GALLERY_SIZE
    ):

        embedding = np.asarray(
            embedding,
            dtype=np.float64
        ).copy()

        self.appearance_gallery.append(
            embedding
        )

        if len(self.appearance_gallery) > gallery_size:

            self.appearance_gallery = (

                self.appearance_gallery[-gallery_size:]
            )

    def appearance_distance(self, embedding):

        if not self.appearance_gallery:

            return 1.0

        distances = [

            cosine_distance(

                embedding,

                gallery_embedding
            )

            for gallery_embedding in self.appearance_gallery
        ]

        return float(
            min(distances)
        )

    def set_state(
        self,
        new_state,
        frame_index
    ):

        if self.state != new_state:

            self.state = new_state

            self.lifecycle.append(

                {

                    "frame": frame_index,

                    "state": new_state.value
                }
            )


# =====================================================================
# DETECTION
# =====================================================================

@dataclass
class Detection:

    bbox: np.ndarray

    appearance: np.ndarray

    confidence: float = 1.0

    class_id: int = 0

    @property
    def measurement(self):

        return xyxy_to_measurement(
            self.bbox
        )


# =====================================================================
# ASSOCIATION COST
# =====================================================================

def association_cost(

    track,

    detection,

    max_mahalanobis=DEFAULT_MAX_MAHALANOBIS,

    max_appearance_distance=DEFAULT_MAX_APPEARANCE_DISTANCE,

    motion_weight=0.45,

    appearance_weight=0.40,

    iou_weight=0.15
):

    """
    Combine motion, appearance, and overlap.

    Returns infinity if motion or appearance gating rejects
    the association.
    """

    measurement = detection.measurement

    mahalanobis = track.kalman.mahalanobis_distance(
        measurement
    )

    if mahalanobis > max_mahalanobis:

        return np.inf

    appearance = track.appearance_distance(
        detection.appearance
    )

    if appearance > max_appearance_distance:

        return np.inf

    overlap = box_iou(

        track.bbox,

        detection.bbox
    )

    normalized_motion = min(

        1.0,

        mahalanobis / max_mahalanobis
    )

    return (

        motion_weight * normalized_motion

        + appearance_weight * appearance

        + iou_weight * (1.0 - overlap)
    )


# =====================================================================
# LONG-TERM TRACKER
# =====================================================================

class LongTermTracker:

    """
    Multi-object tracker with occlusion recovery.

    Track lifecycle:

        ACTIVE -> LOST -> ACTIVE

        ACTIVE -> LOST -> REMOVED

    Lost tracks remain available for re-identification until their
    missed-frame count exceeds max_lost_frames.

    Matching is performed in two stages:

        Stage 1:
            Match detections to ACTIVE tracks.

        Stage 2:
            Match remaining detections to LOST tracks.

    This gives recently active tracks priority over lost tracks.
    """

    def __init__(

        self,

        max_lost_frames=DEFAULT_MAX_LOST_FRAMES,

        max_mahalanobis=DEFAULT_MAX_MAHALANOBIS,

        max_appearance_distance=DEFAULT_MAX_APPEARANCE_DISTANCE,

        max_association_cost=DEFAULT_MAX_ASSOCIATION_COST,

        min_hits=DEFAULT_MIN_HITS,

        gallery_size=DEFAULT_GALLERY_SIZE
    ):

        self.max_lost_frames = max_lost_frames

        self.max_mahalanobis = max_mahalanobis

        self.max_appearance_distance = max_appearance_distance

        self.max_association_cost = max_association_cost

        self.min_hits = min_hits

        self.gallery_size = gallery_size

        self.tracks = {}

        self.next_track_id = 1

        self.frame_index = 0

        self.statistics = {

            "created": 0,

            "matched": 0,

            "lost": 0,

            "reactivated": 0,

            "removed": 0,

            "identity_switches_avoided": 0
        }

        self.history = []

    def _new_track(self, detection):

        track_id = self.next_track_id

        self.next_track_id += 1

        kalman = KalmanBoxFilter(
            detection.bbox
        )

        track = Track(

            track_id=track_id,

            kalman=kalman,

            state=TrackState.ACTIVE,

            first_frame=self.frame_index,

            last_frame=self.frame_index,

            hits=1,

            last_detection_frame=self.frame_index
        )

        track.add_appearance(

            detection.appearance,

            self.gallery_size
        )

        self.tracks[track_id] = track

        self.statistics["created"] += 1

        return track

    def _predict_all(self):

        for track in self.tracks.values():

            if track.state == TrackState.REMOVED:
                continue

            track.kalman.predict()

            track.age += 1

    def _build_cost_matrix(
        self,
        track_list,
        detection_list
    ):

        cost = np.full(

            (
                len(track_list),

                len(detection_list)
            ),

            np.inf,

            dtype=np.float64
        )

        for i, track in enumerate(track_list):

            for j, detection in enumerate(detection_list):

                cost[i, j] = association_cost(

                    track,

                    detection,

                    max_mahalanobis=self.max_mahalanobis,

                    max_appearance_distance=self.max_appearance_distance
                )

        return cost

    def _assign(
        self,
        track_list,
        detection_list
    ):

        if not track_list or not detection_list:

            return [], set(), set()

        cost = self._build_cost_matrix(

            track_list,

            detection_list
        )

        rows, cols = linear_sum_assignment(

            np.where(

                np.isfinite(cost),

                cost,

                1e6
            )
        )

        matches = []

        matched_tracks = set()

        matched_detections = set()

        for row, col in zip(rows, cols):

            if not np.isfinite(cost[row, col]):
                continue

            if cost[row, col] > self.max_association_cost:
                continue

            matches.append(
                (row, col)
            )

            matched_tracks.add(
                row
            )

            matched_detections.add(
                col
            )

        return (

            matches,

            matched_tracks,

            matched_detections
        )

    def _update_track(
        self,
        track,
        detection,
        was_lost
    ):

        track.kalman.update(
            detection.measurement
        )

        track.add_appearance(

            detection.appearance,

            self.gallery_size
        )

        track.hits += 1

        track.missed_frames = 0

        track.last_frame = self.frame_index

        track.last_detection_frame = self.frame_index

        track.trajectory.append(
            track.kalman.center.tolist()
        )

        if was_lost:

            track.set_state(

                TrackState.ACTIVE,

                self.frame_index
            )

            self.statistics["reactivated"] += 1

            self.statistics["identity_switches_avoided"] += 1

        else:

            track.set_state(

                TrackState.ACTIVE,

                self.frame_index
            )

        self.statistics["matched"] += 1

    def update(self, detections):

        self.frame_index += 1

        # Step 1: predict every non-removed track.

        self._predict_all()

        active_tracks = [

            track

            for track in self.tracks.values()

            if track.state == TrackState.ACTIVE
        ]

        lost_tracks = [

            track

            for track in self.tracks.values()

            if track.state == TrackState.LOST
        ]

        # Step 2: associate detections with ACTIVE tracks.

        active_matches, active_rows, active_cols = self._assign(

            active_tracks,

            detections
        )

        for row, col in active_matches:

            self._update_track(

                active_tracks[row],

                detections[col],

                was_lost=False
            )

        unmatched_detections = [

            index

            for index in range(len(detections))

            if index not in active_cols
        ]

        # Step 3: associate remaining detections with LOST tracks.

        remaining_detections = [

            detections[index]

            for index in unmatched_detections
        ]

        lost_matches, lost_rows, lost_cols = self._assign(

            lost_tracks,

            remaining_detections
        )

        for row, col in lost_matches:

            detection_index = unmatched_detections[col]

            self._update_track(

                lost_tracks[row],

                detections[detection_index],

                was_lost=True
            )

        recovered_detection_indices = {

            unmatched_detections[col]

            for _, col in lost_matches
        }

        # Step 4: unmatched active tracks become LOST.

        for row, track in enumerate(active_tracks):

            if row not in active_rows:

                track.missed_frames += 1

                track.set_state(

                    TrackState.LOST,

                    self.frame_index
                )

                self.statistics["lost"] += 1

        # Step 5: unmatched lost tracks continue aging.

        for row, track in enumerate(lost_tracks):

            if row not in lost_rows:

                track.missed_frames += 1

        # Step 6: remove tracks that exceed the retention window.

        for track in self.tracks.values():

            if track.state == TrackState.LOST:

                if track.missed_frames > self.max_lost_frames:

                    track.set_state(

                        TrackState.REMOVED,

                        self.frame_index
                    )

                    self.statistics["removed"] += 1

        # Step 7: create tracks for unmatched detections.

        for index, detection in enumerate(detections):

            if index in active_cols:
                continue

            if index in recovered_detection_indices:
                continue

            self._new_track(
                detection
            )

        # Step 8: record lifecycle statistics.

        self.history.append(

            {

                "frame": self.frame_index,

                "active": sum(

                    track.state == TrackState.ACTIVE

                    for track in self.tracks.values()
                ),

                "lost": sum(

                    track.state == TrackState.LOST

                    for track in self.tracks.values()
                ),

                "removed": sum(

                    track.state == TrackState.REMOVED

                    for track in self.tracks.values()
                )
            }
        )

        return [

            track

            for track in self.tracks.values()

            if track.state == TrackState.ACTIVE

            and track.hits >= self.min_hits
        ]

    def summary(self):

        return {

            "frames": self.frame_index,

            "track_count": len(
                self.tracks
            ),

            "statistics": self.statistics.copy(),

            "tracks": [

                {

                    "track_id": track.track_id,

                    "state": track.state.value,

                    "hits": track.hits,

                    "age": track.age,

                    "missed_frames": track.missed_frames,

                    "first_frame": track.first_frame,

                    "last_frame": track.last_detection_frame,

                    "trajectory": track.trajectory,

                    "lifecycle": track.lifecycle
                }

                for track in self.tracks.values()
            ]
        }


# =====================================================================
# BACKGROUND-SUBTRACTION DETECTOR
# =====================================================================

class MotionDetector:

    """
    Demonstration detector for a mostly stationary camera.

    Replace this class with YOLO/Mask R-CNN/FCOS detections when
    connecting the tracker to a semantic object detector.
    """

    def __init__(
        self,
        min_area=500
    ):

        self.min_area = min_area

        self.background = cv2.createBackgroundSubtractorMOG2(

            history=500,

            varThreshold=32,

            detectShadows=False
        )

        self.kernel = cv2.getStructuringElement(

            cv2.MORPH_ELLIPSE,

            (5, 5)
        )

    def detect(self, frame):

        mask = self.background.apply(
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

            bbox = np.array(

                [
                    x,
                    y,
                    x + w,
                    y + h
                ],

                dtype=np.float64
            )

            detections.append(
                (bbox, 1.0, 0)
            )

        return detections, mask


# =====================================================================
# DRAWING
# =====================================================================

def track_color(track_id):

    return (

        int(track_id * 67 % 255),

        int(track_id * 131 % 255),

        int(track_id * 197 % 255)
    )


def draw_tracks(
    frame,
    tracks,
    all_tracks,
    frame_index
):

    output = frame.copy()

    # Draw lost tracks as dashed predictions.

    for track in all_tracks.values():

        if track.state != TrackState.LOST:
            continue

        x1, y1, x2, y2 = track.bbox.astype(
            int
        )

        color = (0, 165, 255)

        # Dashed rectangle.

        dash = 10

        for x in range(x1, x2, dash * 2):

            cv2.line(

                output,

                (x, y1),

                (min(x + dash, x2), y1),

                color,

                2
            )

            cv2.line(

                output,

                (x, y2),

                (min(x + dash, x2), y2),

                color,

                2
            )

        for y in range(y1, y2, dash * 2):

            cv2.line(

                output,

                (x1, y),

                (x1, min(y + dash, y2)),

                color,

                2
            )

            cv2.line(

                output,

                (x2, y),

                (x2, min(y + dash, y2)),

                color,

                2
            )

        cv2.putText(

            output,

            f"LOST ID {track.track_id}",

            (x1, max(20, y1 - 6)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            color,

            2
        )

    # Draw active tracks.

    for track in tracks:

        x1, y1, x2, y2 = track.bbox.astype(
            int
        )

        color = track_color(
            track.track_id
        )

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            color,

            2
        )

        label = (

            f"ID {track.track_id} "

            f"hits={track.hits}"
        )

        cv2.putText(

            output,

            label,

            (x1, max(20, y1 - 7)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            color,

            2
        )

        trajectory = np.asarray(

            track.trajectory,

            dtype=np.int32
        )

        if len(trajectory) >= 2:

            cv2.polylines(

                output,

                [trajectory.reshape(-1, 1, 2)],

                False,

                color,

                2
            )

    active_count = sum(

        track.state == TrackState.ACTIVE

        for track in all_tracks.values()
    )

    lost_count = sum(

        track.state == TrackState.LOST

        for track in all_tracks.values()
    )

    removed_count = sum(

        track.state == TrackState.REMOVED

        for track in all_tracks.values()
    )

    cv2.putText(

        output,

        f"Frame {frame_index} | Active {active_count} | "
        f"Lost {lost_count} | Removed {removed_count}",

        (15, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )

    return output


# =====================================================================
# TRACK LIFECYCLE PLOT
# =====================================================================

def save_lifecycle_plot(
    history,
    output_path
):

    if not history:
        return

    frames = [
        item["frame"]
        for item in history
    ]

    active = [
        item["active"]
        for item in history
    ]

    lost = [
        item["lost"]
        for item in history
    ]

    removed = [
        item["removed"]
        for item in history
    ]

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(

        frames,

        active,

        label="Active tracks"
    )

    plt.plot(

        frames,

        lost,

        label="Lost tracks"
    )

    plt.plot(

        frames,

        removed,

        label="Removed tracks"
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Number of tracks"
    )

    plt.title(
        "Long-Term Tracking Lifecycle"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        output_path,

        dpi=150
    )

    plt.close()


# =====================================================================
# VIDEO PROCESSING
# =====================================================================

def process_video(args):

    video_path = Path(
        args.video
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

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

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if not np.isfinite(fps) or fps <= 0:

        fps = 30.0

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True
    )

    output_video = (
        output_dir / "long_term_tracking.mp4"
    )

    output_json = (
        output_dir / "tracking_summary.json"
    )

    output_plot = (
        output_dir / "track_lifecycle.png"
    )

    writer = cv2.VideoWriter(

        str(output_video),

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (width, height)
    )

    if not writer.isOpened():

        capture.release()

        raise RuntimeError(
            "Could not create output video."
        )

    detector = MotionDetector(
        min_area=args.min_area
    )

    descriptor = HSVAppearanceDescriptor(

        bins=(8, 8, 4),

        gallery_size=args.gallery_size
    )

    tracker = LongTermTracker(

        max_lost_frames=args.max_lost_frames,

        max_mahalanobis=args.max_mahalanobis,

        max_appearance_distance=args.max_appearance_distance,

        max_association_cost=args.max_association_cost,

        min_hits=args.min_hits,

        gallery_size=args.gallery_size
    )

    processing_times = []

    try:

        while True:

            start = time.perf_counter()

            ret, frame = capture.read()

            if not ret:
                break

            raw_detections, mask = detector.detect(
                frame
            )

            detections = []

            for bbox, confidence, class_id in raw_detections:

                appearance = descriptor.extract(

                    frame,

                    bbox
                )

                detections.append(

                    Detection(

                        bbox=bbox,

                        appearance=appearance,

                        confidence=confidence,

                        class_id=class_id
                    )
                )

            visible_tracks = tracker.update(
                detections
            )

            output = draw_tracks(

                frame,

                visible_tracks,

                tracker.tracks,

                tracker.frame_index
            )

            if args.show_mask:

                mask_bgr = cv2.cvtColor(

                    mask,

                    cv2.COLOR_GRAY2BGR
                )

                mask_bgr = cv2.resize(

                    mask_bgr,

                    (width, height)
                )

                output = np.concatenate(

                    [
                        output,

                        mask_bgr
                    ],

                    axis=1
                )

            # The output writer expects the original frame size.
            # When showing the mask, resize the combined image back.

            if output.shape[1] != width:

                output = cv2.resize(

                    output,

                    (width, height)
                )

            writer.write(
                output
            )

            processing_times.append(

                time.perf_counter() - start
            )

            if not args.no_display:

                cv2.imshow(
                    "Lesson 51 - Long-Term Tracking",
                    output
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            if tracker.frame_index % 100 == 0:

                print(

                    f"Frame {tracker.frame_index} | "

                    f"Active: "

                    f"{sum(t.state == TrackState.ACTIVE for t in tracker.tracks.values())} | "

                    f"Lost: "

                    f"{sum(t.state == TrackState.LOST for t in tracker.tracks.values())} | "

                    f"Removed: "

                    f"{sum(t.state == TrackState.REMOVED for t in tracker.tracks.values())}"
                )

    finally:

        capture.release()

        writer.release()

        cv2.destroyAllWindows()

    summary = tracker.summary()

    summary["input_video"] = str(
        video_path
    )

    summary["output_video"] = str(
        output_video
    )

    summary["fps"] = fps

    summary["average_processing_fps"] = (

        1.0 / np.mean(processing_times)

        if processing_times and np.mean(processing_times) > 0

        else 0.0
    )

    with open(

        output_json,

        "w"

    ) as file:

        json.dump(

            summary,

            file,

            indent=4
        )

    save_lifecycle_plot(

        tracker.history,

        output_plot
    )

    print("\nTracking completed.")

    print(
        json.dumps(
            summary["statistics"],
            indent=4
        )
    )

    print("\nOutput video:", output_video)

    print("Summary:", output_json)

    print("Lifecycle plot:", output_plot)


# =====================================================================
# NUMERICAL TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print(
        "LESSON 51 — LONG-TERM TRACKING TESTS"
    )

    print("=" * 70)

    # ---------------------------------------------------------------
    # Test 1: bounding-box conversion.
    # ---------------------------------------------------------------

    box = np.array(
        [10, 20, 50, 80],
        dtype=np.float64
    )

    measurement = xyxy_to_measurement(
        box
    )

    reconstructed = measurement_to_xyxy(
        measurement
    )

    assert np.allclose(
        box,
        reconstructed
    )

    print("Test 1 passed: bounding-box conversion")

    # ---------------------------------------------------------------
    # Test 2: box IoU.
    # ---------------------------------------------------------------

    assert abs(
        box_iou(box, box) - 1.0
    ) < 1e-8

    print("Test 2 passed: identical-box IoU")

    # ---------------------------------------------------------------
    # Test 3: Kalman initialization.
    # ---------------------------------------------------------------

    kalman = KalmanBoxFilter(
        box
    )

    assert np.allclose(
        kalman.box,
        box
    )

    assert kalman.x.shape == (
        STATE_DIM,
    )

    print("Test 3 passed: Kalman initialization")

    # ---------------------------------------------------------------
    # Test 4: prediction follows constant velocity.
    # ---------------------------------------------------------------

    kalman = KalmanBoxFilter(
        box
    )

    kalman.x[4] = 5.0

    kalman.predict()

    assert abs(
        kalman.x[0] - 35.0
    ) < 1e-8

    print("Test 4 passed: constant-velocity prediction")

    # ---------------------------------------------------------------
    # Test 5: measurement correction.
    # ---------------------------------------------------------------

    kalman = KalmanBoxFilter(
        box
    )

    predicted_before = kalman.x[0]

    measurement = np.array(
        [35, 50, 40, 60],
        dtype=np.float64
    )

    kalman.update(
        measurement
    )

    assert abs(
        kalman.x[0] - 30.0
    ) > 1e-8

    assert kalman.x[0] > predicted_before

    print("Test 5 passed: Kalman measurement correction")

    # ---------------------------------------------------------------
    # Test 6: covariance symmetry.
    # ---------------------------------------------------------------

    kalman = KalmanBoxFilter(
        box
    )

    kalman.predict()

    kalman.update(
        xyxy_to_measurement(box)
    )

    assert np.allclose(
        kalman.P,
        kalman.P.T,
        atol=1e-8
    )

    print("Test 6 passed: covariance symmetry")

    # ---------------------------------------------------------------
    # Test 7: covariance positive semidefinite.
    # ---------------------------------------------------------------

    eigenvalues = np.linalg.eigvalsh(
        kalman.P
    )

    assert np.min(
        eigenvalues
    ) > -1e-8

    print("Test 7 passed: covariance positive semidefinite")

    # ---------------------------------------------------------------
    # Test 8: zero Mahalanobis distance.
    # ---------------------------------------------------------------

    kalman = KalmanBoxFilter(
        box
    )

    distance = kalman.mahalanobis_distance(

        xyxy_to_measurement(box)
    )

    assert distance < 1e-8

    print("Test 8 passed: zero innovation distance")

    # ---------------------------------------------------------------
    # Test 9: appearance distance.
    # ---------------------------------------------------------------

    appearance_a = np.array(
        [1.0, 0.0, 0.0]
    )

    appearance_b = np.array(
        [1.0, 0.0, 0.0]
    )

    assert abs(
        cosine_distance(
            appearance_a,
            appearance_b
        )
    ) < 1e-8

    print("Test 9 passed: identical appearance")

    # ---------------------------------------------------------------
    # Test 10: orthogonal appearance.
    # ---------------------------------------------------------------

    appearance_b = np.array(
        [0.0, 1.0, 0.0]
    )

    assert abs(
        cosine_distance(
            appearance_a,
            appearance_b
        ) - 1.0
    ) < 1e-8

    print("Test 10 passed: orthogonal appearance")

    # ---------------------------------------------------------------
    # Test 11: appearance gallery.
    # ---------------------------------------------------------------

    track = Track(

        track_id=1,

        kalman=KalmanBoxFilter(box),

        state=TrackState.ACTIVE,

        first_frame=1,

        last_frame=1
    )

    track.add_appearance(
        appearance_a,
        gallery_size=2
    )

    track.add_appearance(
        appearance_b,
        gallery_size=2
    )

    track.add_appearance(

        np.array(
            [0.0, 0.0, 1.0]
        ),

        gallery_size=2
    )

    assert len(
        track.appearance_gallery
    ) == 2

    print("Test 11 passed: bounded appearance gallery")

    # ---------------------------------------------------------------
    # Test 12: gallery minimum-distance matching.
    # ---------------------------------------------------------------

    track = Track(

        track_id=1,

        kalman=KalmanBoxFilter(box),

        state=TrackState.ACTIVE,

        first_frame=1,

        last_frame=1
    )

    track.add_appearance(
        appearance_a
    )

    track.add_appearance(
        appearance_b
    )

    distance = track.appearance_distance(
        appearance_a
    )

    assert distance < 1e-8

    print("Test 12 passed: gallery appearance matching")

    # ---------------------------------------------------------------
    # Test 13: track becomes LOST.
    # ---------------------------------------------------------------

    tracker = LongTermTracker(
        min_hits=1
    )

    appearance = np.zeros(
        256
    )

    appearance[0] = 1.0

    detection = Detection(

        bbox=box,

        appearance=appearance
    )

    tracker.update(
        [detection]
    )

    tracker.update(
        []
    )

    track = tracker.tracks[1]

    assert track.state == TrackState.LOST

    assert track.missed_frames == 1

    print("Test 13 passed: ACTIVE to LOST transition")

    # ---------------------------------------------------------------
    # Test 14: lost track is reactivated.
    # ---------------------------------------------------------------

    tracker = LongTermTracker(
        min_hits=1
    )

    tracker.update(
        [detection]
    )

    original_id = 1

    tracker.update(
        []
    )

    tracker.update(
        [detection]
    )

    track = tracker.tracks[original_id]

    assert track.state == TrackState.ACTIVE

    assert track.hits == 2

    assert tracker.statistics["reactivated"] == 1

    print("Test 14 passed: LOST to ACTIVE recovery")

    # ---------------------------------------------------------------
    # Test 15: recovery preserves original ID.
    # ---------------------------------------------------------------

    tracker = LongTermTracker(
        min_hits=1
    )

    tracker.update(
        [detection]
    )

    original_id = list(
        tracker.tracks.keys()
    )[0]

    tracker.update(
        []
    )

    visible = tracker.update(
        [detection]
    )

    assert len(visible) == 1

    assert visible[0].track_id == original_id

    print("Test 15 passed: original identity preserved")

    # ---------------------------------------------------------------
    # Test 16: track removed after retention window.
    # ---------------------------------------------------------------

    tracker = LongTermTracker(

        max_lost_frames=2,

        min_hits=1
    )

    tracker.update(
        [detection]
    )

    tracker.update([])
    tracker.update([])
    tracker.update([])

    assert tracker.tracks[1].state == TrackState.REMOVED

    print("Test 16 passed: LOST to REMOVED transition")

    # ---------------------------------------------------------------
    # Test 17: removed track cannot be reactivated.
    # ---------------------------------------------------------------

    tracker = LongTermTracker(

        max_lost_frames=1,

        min_hits=1
    )

    tracker.update(
        [detection]
    )

    tracker.update([])
    tracker.update([])

    assert tracker.tracks[1].state == TrackState.REMOVED

    visible = tracker.update(
        [detection]
    )

    assert len(visible) == 1

    assert visible[0].track_id == 2

    print("Test 17 passed: removed identity is not reused")

    # ---------------------------------------------------------------
    # Test 18: new object receives a different ID.
    # ---------------------------------------------------------------

    tracker = LongTermTracker(
        min_hits=1
    )

    tracker.update(
        [detection]
    )

    second_detection = Detection(

        bbox=np.array(
            [200, 200, 240, 260],
            dtype=np.float64
        ),

        appearance=appearance.copy()
    )

    visible = tracker.update(

        [
            second_detection
        ]
    )

    assert len(visible) == 2

    assert visible[0].track_id != visible[1].track_id

    print("Test 18 passed: separate object receives separate ID")

    # ---------------------------------------------------------------
    # Test 19: motion gating rejects distant detection.
    # ---------------------------------------------------------------

    tracker = LongTermTracker(
        min_hits=1
    )

    tracker.update(
        [detection]
    )

    far_detection = Detection(

        bbox=np.array(
            [1000, 1000, 1040, 1060],
            dtype=np.float64
        ),

        appearance=appearance.copy()
    )

    visible = tracker.update(
        [far_detection]
    )

    assert len(visible) == 1

    assert visible[0].track_id == 2

    print("Test 19 passed: distant detection rejected")

    # ---------------------------------------------------------------
    # Test 20: trajectory records observations.
    # ---------------------------------------------------------------

    tracker = LongTermTracker(
        min_hits=1
    )

    tracker.update(
        [detection]
    )

    tracker.update(
        [detection]
    )

    track = tracker.tracks[1]

    assert len(
        track.trajectory
    ) == 2

    print("Test 20 passed: trajectory history")

    # ---------------------------------------------------------------
    # Test 21: lifecycle history records transitions.
    # ---------------------------------------------------------------

    tracker = LongTermTracker(
        min_hits=1
    )

    tracker.update(
        [detection]
    )

    tracker.update([])

    tracker.update(
        [detection]
    )

    lifecycle = tracker.tracks[1].lifecycle

    states = [

        item["state"]

        for item in lifecycle
    ]

    assert states == [
        "active",
        "lost",
        "active"
    ]

    print("Test 21 passed: lifecycle history")

    # ---------------------------------------------------------------
    # Test 22: HSV descriptor dimension.
    # ---------------------------------------------------------------

    descriptor = HSVAppearanceDescriptor(
        bins=(8, 8, 4)
    )

    image = np.random.default_rng(
        42
    ).integers(

        0,

        256,

        size=(100, 120, 3),

        dtype=np.uint8
    )

    feature = descriptor.extract(

        image,

        [10, 10, 60, 60]
    )

    assert feature.shape == (
        8 * 8 * 4,
    )

    print("Test 22 passed: HSV descriptor dimensions")

    # ---------------------------------------------------------------
    # Test 23: HSV descriptor normalization.
    # ---------------------------------------------------------------

    assert abs(
        np.linalg.norm(feature) - 1.0
    ) < 1e-8

    print("Test 23 passed: HSV descriptor normalization")

    # ---------------------------------------------------------------
    # Test 24: association cost is finite for a matching object.
    # ---------------------------------------------------------------

    tracker = LongTermTracker(
        min_hits=1
    )

    tracker.update(
        [detection]
    )

    track = tracker.tracks[1]

    matching_detection = Detection(

        bbox=box.copy(),

        appearance=appearance.copy()
    )

    cost = association_cost(

        track,

        matching_detection
    )

    assert np.isfinite(cost)

    assert cost < 0.1

    print("Test 24 passed: matching association cost")

    # ---------------------------------------------------------------
    # Test 25: association cost rejects appearance mismatch.
    # ---------------------------------------------------------------

    mismatched = Detection(

        bbox=box.copy(),

        appearance=np.eye(
            1,
            256,
            1
        ).flatten()
    )

    cost = association_cost(

        track,

        mismatched,

        max_appearance_distance=0.2
    )

    assert not np.isfinite(cost)

    print("Test 25 passed: appearance gating")

    # ---------------------------------------------------------------
    # Test 26: summary contains statistics.
    # ---------------------------------------------------------------

    tracker = LongTermTracker(
        min_hits=1
    )

    tracker.update(
        [detection]
    )

    summary = tracker.summary()

    assert summary["frames"] == 1

    assert summary["statistics"]["created"] == 1

    assert len(summary["tracks"]) == 1

    print("Test 26 passed: tracking summary")

    print("\nAll 26 tests passed.")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(

            "Lesson 51 — Long-Term Object Tracking "
            "and Occlusion Handling"
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/lesson51"
    )

    parser.add_argument(
        "--max-lost-frames",
        type=int,
        default=DEFAULT_MAX_LOST_FRAMES
    )

    parser.add_argument(
        "--max-mahalanobis",
        type=float,
        default=DEFAULT_MAX_MAHALANOBIS
    )

    parser.add_argument(
        "--max-appearance-distance",
        type=float,
        default=DEFAULT_MAX_APPEARANCE_DISTANCE
    )

    parser.add_argument(
        "--max-association-cost",
        type=float,
        default=DEFAULT_MAX_ASSOCIATION_COST
    )

    parser.add_argument(
        "--gallery-size",
        type=int,
        default=DEFAULT_GALLERY_SIZE
    )

    parser.add_argument(
        "--min-hits",
        type=int,
        default=DEFAULT_MIN_HITS
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=500
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

    if args.max_lost_frames < 0:

        parser.error(
            "--max-lost-frames cannot be negative."
        )

    if args.gallery_size < 1:

        parser.error(
            "--gallery-size must be at least 1."
        )

    if args.min_hits < 1:

        parser.error(
            "--min-hits must be at least 1."
        )

    if args.check_only:

        run_tests()

        return

    if args.video is None:

        parser.error(
            "Provide a video using --video."
        )

    process_video(args)


if __name__ == "__main__":

    main()