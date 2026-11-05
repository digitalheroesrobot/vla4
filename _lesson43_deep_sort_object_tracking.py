"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 43 — DEEP SORT-INSPIRED OBJECT TRACKING
======================================================================

Filename:
    lesson43_deep_sort_object_tracking.py

PHASE:
    Phase 4 — Tracking and Visual Memory

======================================================================
LEARNING OBJECTIVES
======================================================================

Implement appearance-assisted multi-object tracking.

Components:

    1. Kalman filter with an 8-dimensional state
    2. Bounding-box motion prediction
    3. Appearance feature extraction from real image crops
    4. Cosine appearance distance
    5. Appearance feature galleries
    6. Mahalanobis motion gating
    7. Hungarian assignment
    8. Matching cascade
    9. IoU fallback matching
    10. Track lifecycle management

IMPORTANT IMPLEMENTATION NOTE:

Original Deep SORT uses a learned CNN appearance embedding.

This lesson does not use pretrained weights or an untrained CNN.

Instead, it uses a handcrafted HSV color descriptor extracted
from real video detection crops.

This allows us to understand and test the appearance-association
architecture without claiming that random neural-network weights
produce meaningful identity embeddings.

The implementation is Deep SORT-inspired, not a reproduction
of the original Deep SORT implementation.

======================================================================
1. WHY APPEARANCE FEATURES?
======================================================================

SORT relies primarily on motion and bounding-box overlap.

Consider two objects:

    Object A: red bottle
    Object B: blue bottle

They cross paths.

Their bounding boxes may overlap, making motion-based
association ambiguous.

Appearance descriptors provide additional information:

    Track A -> red appearance descriptor
    Track B -> blue appearance descriptor

New detections:

    Detection 1 -> blue appearance descriptor
    Detection 2 -> red appearance descriptor

The tracker can use appearance similarity to preserve identities.

======================================================================
2. APPEARANCE DESCRIPTOR
======================================================================

For each detection crop, calculate:

    Global HSV histogram:
        Hue:        16 bins
        Saturation:  8 bins
        Value:       8 bins

    Spatial hue histograms:
        Divide crop into a 2 x 2 grid.
        Each region uses 8 hue bins.

Total feature dimensions:

    16 + 8 + 8 + 4 * 8 = 64

Normalize the descriptor using L2 normalization.

This is a handcrafted visual descriptor, not a learned
semantic representation.

======================================================================
3. COSINE DISTANCE
======================================================================

For normalized vectors a and b:

    cosine_similarity = a.T @ b

    cosine_distance = 1 - cosine_similarity

Lower distance means more similar appearance.

Each track stores a gallery of recent descriptors.

We use the nearest-neighbor distance:

    distance(track, detection) =
        minimum cosine distance between the detection feature
        and all features in the track gallery.

======================================================================
4. KALMAN STATE
======================================================================

State:

    x = [
        cx,
        cy,
        aspect_ratio,
        height,
        vx,
        vy,
        v_aspect_ratio,
        v_height
    ]

Measurement:

    z = [
        cx,
        cy,
        aspect_ratio,
        height
    ]

The model assumes approximately constant velocity.

======================================================================
5. MAHALANOBIS MOTION GATING
======================================================================

Appearance similarity alone is not sufficient.

A blue bottle detected on the opposite side of the image
should not suddenly become an existing track.

We calculate:

    d² = (z - z_pred).T S^-1 (z - z_pred)

Where:

    z       = detection measurement
    z_pred  = predicted measurement
    S       = innovation covariance

We reject physically implausible matches.

For a four-dimensional measurement, a commonly used
95% chi-square gating threshold is approximately:

    9.4877

======================================================================
6. MATCHING CASCADE
======================================================================

Confirmed tracks are matched in order of recency.

    First:
        Tracks missed for 1 frame.

    Then:
        Tracks missed for 2 frames.

    Then:
        Tracks missed for 3 frames.

This prioritizes recently observed tracks.

For unmatched tentative tracks, we use IoU matching.

======================================================================
7. REAL VIDEO PIPELINE
======================================================================

Real video
    |
    v
Background subtraction
    |
    v
Bounding-box detections
    |
    v
Real image crops
    |
    v
HSV appearance descriptors
    |
    +----------------------+
    |                      |
    v                      v
Kalman prediction      Appearance gallery
    |                      |
    v                      v
Motion gating         Cosine distance
    |                      |
    +-----------+----------+
                |
                v
        Hungarian assignment
                |
                v
          Track management
                |
                v
         Persistent IDs

Background subtraction requires a mostly stationary camera.

It detects moving regions, not semantic object categories.

======================================================================
8. ROBOTICS APPLICATION
======================================================================

Waste-sorting conveyor:

    Camera sees:
        Red bottle
        Blue bottle
        Aluminum can

    Tracker creates:
        ID 1
        ID 2
        ID 3

    Objects cross.

    Motion prediction becomes ambiguous.

    Appearance matching helps preserve identity.

A real robot system should combine this with:

    - A semantic detector or segmentation model
    - Camera calibration
    - Object localization
    - Conveyor velocity estimation
    - Robot-base coordinate transformation
    - Grasp planning

======================================================================
RUN
======================================================================

Install:

    pip install numpy opencv-python

Tests:

    python lesson43_deep_sort_object_tracking.py --check-only

Real video:

    python lesson43_deep_sort_object_tracking.py \
        --video conveyor_video.mp4

Save output:

    python lesson43_deep_sort_object_tracking.py \
        --video conveyor_video.mp4 \
        --output runs/lesson43/deep_sort_output.mp4

Disable display:

    python lesson43_deep_sort_object_tracking.py \
        --video conveyor_video.mp4 \
        --no-display

Adjust appearance threshold:

    python lesson43_deep_sort_object_tracking.py \
        --video conveyor_video.mp4 \
        --max-cosine-distance 0.40

======================================================================
OUTPUT
======================================================================

runs/lesson43/
    deep_sort_output.mp4
    deep_sort_summary.json

======================================================================
"""

import argparse
import json
import time

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# =====================================================================
# CONFIGURATION
# =====================================================================

CHI2INV95 = 9.4877

DEFAULT_MAX_AGE = 30

DEFAULT_N_INIT = 3

DEFAULT_MAX_COSINE_DISTANCE = 0.40

DEFAULT_MAX_FEATURES = 50

DEFAULT_IOU_THRESHOLD = 0.20


# =====================================================================
# BOUNDING BOX CONVERSIONS
# =====================================================================

def bbox_to_measurement(box):
    """
    Convert xyxy bounding box into Deep SORT measurement:

        [cx, cy, aspect_ratio, height]
    """

    x1, y1, x2, y2 = map(float, box)

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    aspect_ratio = width / height

    return np.array(
        [cx, cy, aspect_ratio, height],
        dtype=np.float64
    )


def measurement_to_bbox(measurement):
    """
    Convert:

        [cx, cy, aspect_ratio, height]

    Into:

        [x1, y1, x2, y2]
    """

    cx, cy, aspect_ratio, height = map(
        float,
        measurement
    )

    height = max(1.0, height)
    aspect_ratio = max(1e-6, aspect_ratio)

    width = aspect_ratio * height

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

    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(
        0.0, ay2 - ay1
    )

    area_b = max(0.0, bx2 - bx1) * max(
        0.0, by2 - by1
    )

    union = area_a + area_b - intersection

    return intersection / max(union, 1e-12)


# =====================================================================
# HUNGARIAN ASSIGNMENT
# =====================================================================

def hungarian_assignment(cost_matrix):
    """
    Rectangular Hungarian assignment using the
    shortest augmenting path algorithm.

    Returns:
        List of (row, column) assignments.

    Supports:
        N x M matrices
        N < M
        N > M
        Empty matrices
    """

    cost = np.asarray(
        cost_matrix,
        dtype=np.float64
    )

    if cost.ndim != 2:
        raise ValueError(
            "Cost matrix must be 2-dimensional."
        )

    original_rows, original_cols = cost.shape

    if original_rows == 0 or original_cols == 0:
        return []

    if not np.isfinite(cost).all():
        raise ValueError(
            "Cost matrix must contain finite values."
        )

    transposed = False

    if original_rows > original_cols:
        cost = cost.T
        transposed = True

    n, m = cost.shape

    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)

    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)

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

        row = int(p[j] - 1)
        col = int(j - 1)

        if transposed:
            assignments.append((col, row))
        else:
            assignments.append((row, col))

    assignments.sort()

    return assignments


# =====================================================================
# APPEARANCE FEATURE EXTRACTION
# =====================================================================

class AppearanceExtractor:
    """
    Extract a 64-dimensional appearance descriptor
    from an actual image crop.

    Global HSV histogram:

        H: 16 bins
        S:  8 bins
        V:  8 bins

    Spatial H histogram:

        2 x 2 grid
        8 bins per region

    Total:

        16 + 8 + 8 + 4*8 = 64

    This is a handcrafted descriptor, not a neural
    appearance embedding.
    """

    FEATURE_DIM = 64

    def __init__(self):

        self.h_bins = 16
        self.s_bins = 8
        self.v_bins = 8
        self.spatial_bins = 8

    @staticmethod
    def normalize(feature):

        feature = np.asarray(
            feature,
            dtype=np.float32
        )

        norm = np.linalg.norm(feature)

        if norm < 1e-12:
            return feature

        return feature / norm

    def extract(self, image, bbox):

        height, width = image.shape[:2]

        x1, y1, x2, y2 = map(
            int,
            bbox
        )

        x1 = int(np.clip(x1, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        x2 = int(np.clip(x2, 0, width))
        y2 = int(np.clip(y2, 0, height))

        if x2 <= x1 or y2 <= y1:
            return np.zeros(
                self.FEATURE_DIM,
                dtype=np.float32
            )

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros(
                self.FEATURE_DIM,
                dtype=np.float32
            )

        hsv = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2HSV
        )

        # -------------------------------------------------------------
        # Global HSV histogram
        # -------------------------------------------------------------

        hist_h = cv2.calcHist(
            [hsv],
            [0],
            None,
            [self.h_bins],
            [0, 180]
        ).flatten()

        hist_s = cv2.calcHist(
            [hsv],
            [1],
            None,
            [self.s_bins],
            [0, 256]
        ).flatten()

        hist_v = cv2.calcHist(
            [hsv],
            [2],
            None,
            [self.v_bins],
            [0, 256]
        ).flatten()

        # -------------------------------------------------------------
        # Spatial hue histograms
        # -------------------------------------------------------------

        crop_h, crop_w = hsv.shape[:2]

        mid_x = crop_w // 2
        mid_y = crop_h // 2

        regions = [

            hsv[:mid_y, :mid_x],

            hsv[:mid_y, mid_x:],

            hsv[mid_y:, :mid_x],

            hsv[mid_y:, mid_x:]
        ]

        spatial_features = []

        for region in regions:

            if region.size == 0:

                hist = np.zeros(
                    self.spatial_bins,
                    dtype=np.float32
                )

            else:

                hist = cv2.calcHist(

                    [region],

                    [0],

                    None,

                    [self.spatial_bins],

                    [0, 180]
                ).flatten()

            spatial_features.extend(
                hist
            )

        feature = np.concatenate(
            [
                hist_h,
                hist_s,
                hist_v,
                np.asarray(
                    spatial_features,
                    dtype=np.float32
                )
            ]
        )

        feature = self.normalize(
            feature
        )

        return feature.astype(
            np.float32
        )


# =====================================================================
# COSINE DISTANCE
# =====================================================================

def cosine_distance(feature_a, feature_b):

    a = np.asarray(
        feature_a,
        dtype=np.float64
    )

    b = np.asarray(
        feature_b,
        dtype=np.float64
    )

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0

    similarity = np.dot(
        a,
        b
    ) / (
        norm_a * norm_b
    )

    similarity = np.clip(
        similarity,
        -1.0,
        1.0
    )

    return float(
        1.0 - similarity
    )


def nearest_gallery_distance(
    gallery,
    query_feature
):

    if len(gallery) == 0:
        return 1.0

    distances = [

        cosine_distance(
            feature,
            query_feature
        )

        for feature in gallery
    ]

    return float(
        min(distances)
    )


# =====================================================================
# DEEP SORT KALMAN FILTER
# =====================================================================

class DeepSORTKalmanFilter:

    """
    State:

        [cx, cy, aspect_ratio, height,
         vx, vy, v_aspect_ratio, v_height]

    Measurement:

        [cx, cy, aspect_ratio, height]
    """

    def __init__(self, measurement):

        z = np.asarray(
            measurement,
            dtype=np.float64
        )

        if z.shape != (4,):
            raise ValueError(
                "Measurement must have shape (4,)."
            )

        self.x = np.zeros(
            8,
            dtype=np.float64
        )

        self.x[:4] = z

        self.P = np.eye(
            8,
            dtype=np.float64
        )

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

        self.R = np.diag(
            [
                1.0,
                1.0,
                0.01,
                1.0
            ]
        )

        self.Q = np.diag(
            [
                0.1,
                0.1,
                0.001,
                0.1,
                0.01,
                0.01,
                0.0001,
                0.01
            ]
        )

        self.P[:4, :4] *= 10.0
        self.P[4:, 4:] *= 100.0

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

        self.x[2] = max(
            1e-6,
            self.x[2]
        )

        self.x[3] = max(
            1.0,
            self.x[3]
        )

        self.P = (
            self.F @ self.P @ self.F.T
            + self.Q
        )

        return self.x.copy()

    def project(self):

        mean = self.H @ self.x

        covariance = (
            self.H @ self.P @ self.H.T
            + self.R
        )

        return mean, covariance

    def update(self, measurement):

        z = np.asarray(
            measurement,
            dtype=np.float64
        )

        if z.shape != (4,):
            raise ValueError(
                "Measurement must have shape (4,)."
            )

        mean, covariance = self.project()

        innovation = z - mean

        K = np.linalg.solve(
            covariance,
            self.H @ self.P
        ).T

        self.x = self.x + K @ innovation

        A = self.I - K @ self.H

        # Joseph covariance update.

        self.P = (
            A @ self.P @ A.T
            + K @ self.R @ K.T
        )

        self.P = (
            self.P + self.P.T
        ) / 2.0

        self.x[2] = max(
            1e-6,
            self.x[2]
        )

        self.x[3] = max(
            1.0,
            self.x[3]
        )

        return self.x.copy()

    def gating_distance(self, measurement):

        z = np.asarray(
            measurement,
            dtype=np.float64
        )

        mean, covariance = self.project()

        difference = z - mean

        solved = np.linalg.solve(
            covariance,
            difference
        )

        return float(
            difference.T @ solved
        )

    def get_bbox(self):

        return measurement_to_bbox(
            self.x[:4]
        )

    def get_velocity(self):

        return self.x[4:].copy()


# =====================================================================
# DETECTION STRUCTURE
# =====================================================================

@dataclass
class Detection:

    bbox: np.ndarray

    feature: np.ndarray

    def __post_init__(self):

        self.bbox = np.asarray(
            self.bbox,
            dtype=np.float64
        )

        self.feature = np.asarray(
            self.feature,
            dtype=np.float32
        )

        if self.bbox.shape != (4,):
            raise ValueError(
                "Detection bbox must have shape (4,)."
            )

        if self.feature.ndim != 1:
            raise ValueError(
                "Detection feature must be one-dimensional."
            )

        if not np.isfinite(self.bbox).all():
            raise ValueError(
                "Detection bbox must be finite."
            )

        if not np.isfinite(self.feature).all():
            raise ValueError(
                "Detection feature must be finite."
            )


# =====================================================================
# TRACK
# =====================================================================

class DeepSORTTrack:

    def __init__(
        self,
        detection,
        track_id,
        max_features=DEFAULT_MAX_FEATURES,
        n_init=DEFAULT_N_INIT
    ):

        self.track_id = track_id

        measurement = bbox_to_measurement(
            detection.bbox
        )

        self.kalman = DeepSORTKalmanFilter(
            measurement
        )

        self.features = deque(
            maxlen=max_features
        )

        self.features.append(
            detection.feature.copy()
        )

        self.age = 1

        self.hits = 1

        self.time_since_update = 0

        self.n_init = n_init

        self.confirmed = (
            n_init <= 1
        )

        self.state = "tentative"

        self.last_detection = detection

    def predict(self, dt=1.0):

        self.kalman.predict(dt)

        self.age += 1

        self.time_since_update += 1

        if self.time_since_update > 0:

            self.state = (
                "lost"
                if self.confirmed
                else "tentative"
            )

    def update(self, detection):

        measurement = bbox_to_measurement(
            detection.bbox
        )

        self.kalman.update(
            measurement
        )

        self.features.append(
            detection.feature.copy()
        )

        self.hits += 1

        self.time_since_update = 0

        self.last_detection = detection

        if self.hits >= self.n_init:

            self.confirmed = True

        self.state = (
            "confirmed"
            if self.confirmed
            else "tentative"
        )

    def get_bbox(self):

        return self.kalman.get_bbox()

    def get_velocity(self):

        return self.kalman.get_velocity()


# =====================================================================
# APPEARANCE COST MATRIX
# =====================================================================

def build_appearance_cost_matrix(
    tracks,
    detections,
    max_cosine_distance,
    gating_threshold=CHI2INV95
):

    cost = np.full(
        (
            len(tracks),
            len(detections)
        ),
        1e6,
        dtype=np.float64
    )

    for i, track in enumerate(tracks):

        for j, detection in enumerate(detections):

            measurement = bbox_to_measurement(
                detection.bbox
            )

            # ---------------------------------------------------------
            # Motion gating
            # ---------------------------------------------------------

            motion_distance = (
                track.kalman.gating_distance(
                    measurement
                )
            )

            if motion_distance > gating_threshold:

                continue

            # ---------------------------------------------------------
            # Appearance distance
            # ---------------------------------------------------------

            appearance_distance = (
                nearest_gallery_distance(

                    track.features,

                    detection.feature
                )
            )

            if appearance_distance > max_cosine_distance:

                continue

            cost[i, j] = appearance_distance

    return cost


# =====================================================================
# MATCHING HELPERS
# =====================================================================

def assign_with_threshold(
    cost_matrix,
    threshold
):

    assignments = hungarian_assignment(
        cost_matrix
    )

    matches = []

    unmatched_rows = set(
        range(cost_matrix.shape[0])
    )

    unmatched_columns = set(
        range(cost_matrix.shape[1])
    )

    for row, column in assignments:

        if cost_matrix[row, column] > threshold:

            continue

        matches.append(
            (row, column)
        )

        unmatched_rows.discard(row)

        unmatched_columns.discard(column)

    return (
        matches,
        unmatched_rows,
        unmatched_columns
    )


def iou_matching(
    tracks,
    detections,
    threshold
):

    cost = np.full(
        (
            len(tracks),
            len(detections)
        ),
        1e6,
        dtype=np.float64
    )

    for i, track in enumerate(tracks):

        for j, detection in enumerate(detections):

            similarity = bbox_iou(
                track.get_bbox(),
                detection.bbox
            )

            if similarity >= threshold:

                cost[i, j] = 1.0 - similarity

    return assign_with_threshold(
        cost,
        1.0 - threshold
    )


# =====================================================================
# DEEP SORT TRACKER
# =====================================================================

class DeepSORT:

    def __init__(
        self,
        max_age=DEFAULT_MAX_AGE,
        n_init=DEFAULT_N_INIT,
        max_cosine_distance=DEFAULT_MAX_COSINE_DISTANCE,
        max_features=DEFAULT_MAX_FEATURES,
        iou_threshold=DEFAULT_IOU_THRESHOLD
    ):

        self.max_age = max_age

        self.n_init = n_init

        self.max_cosine_distance = max_cosine_distance

        self.max_features = max_features

        self.iou_threshold = iou_threshold

        self.tracks = []

        self.next_id = 1

        self.frame_count = 0

        self.total_created = 0

        self.total_deleted = 0

        self.total_appearance_matches = 0

        self.total_iou_matches = 0

    def create_track(self, detection):

        track = DeepSORTTrack(

            detection=detection,

            track_id=self.next_id,

            max_features=self.max_features,

            n_init=self.n_init
        )

        self.tracks.append(
            track
        )

        self.next_id += 1

        self.total_created += 1

    def matching_cascade(
        self,
        detections
    ):

        """
        Match confirmed tracks by recency.

        Level 1:
            Tracks missed for one frame.

        Level 2:
            Tracks missed for two frames.

        Continue until max_age.
        """

        unmatched_detections = set(
            range(len(detections))
        )

        matches = []

        confirmed_indices = [

            index

            for index, track in enumerate(self.tracks)

            if track.confirmed
        ]

        for level in range(
            1,
            self.max_age + 1
        ):

            track_indices = [

                index

                for index in confirmed_indices

                if self.tracks[index].time_since_update == level
            ]

            if not track_indices:
                continue

            detection_indices = sorted(
                unmatched_detections
            )

            if not detection_indices:
                break

            selected_tracks = [
                self.tracks[index]
                for index in track_indices
            ]

            selected_detections = [
                detections[index]
                for index in detection_indices
            ]

            cost = build_appearance_cost_matrix(

                selected_tracks,

                selected_detections,

                self.max_cosine_distance
            )

            level_matches, _, unmatched_columns = (
                assign_with_threshold(
                    cost,
                    self.max_cosine_distance
                )
            )

            for local_track, local_detection in level_matches:

                original_track = track_indices[
                    local_track
                ]

                original_detection = detection_indices[
                    local_detection
                ]

                matches.append(
                    (
                        original_track,
                        original_detection
                    )
                )

                unmatched_detections.discard(
                    original_detection
                )

            self.total_appearance_matches += len(
                level_matches
            )

        return matches, unmatched_detections

    def update(
        self,
        detections,
        dt=1.0
    ):

        self.frame_count += 1

        # -------------------------------------------------------------
        # Step 1: Predict all tracks.
        # -------------------------------------------------------------

        for track in self.tracks:

            track.predict(dt)

        # -------------------------------------------------------------
        # Step 2: Appearance matching cascade.
        # -------------------------------------------------------------

        matches, unmatched_detections = (
            self.matching_cascade(
                detections
            )
        )

        matched_tracks = {
            track_index
            for track_index, _ in matches
        }

        # -------------------------------------------------------------
        # Step 3: IoU fallback matching.
        #
        # Use for:
        #   - Unconfirmed tracks.
        #   - Confirmed tracks missed for one frame.
        #
        # This helps tracks that do not yet have a reliable
        # appearance gallery.
        # -------------------------------------------------------------

        iou_track_indices = [

            index

            for index, track in enumerate(self.tracks)

            if index not in matched_tracks

            and (
                not track.confirmed
                or track.time_since_update == 1
            )
        ]

        iou_detection_indices = sorted(
            unmatched_detections
        )

        if iou_track_indices and iou_detection_indices:

            selected_tracks = [
                self.tracks[index]
                for index in iou_track_indices
            ]

            selected_detections = [
                detections[index]
                for index in iou_detection_indices
            ]

            iou_matches, _, unmatched_columns = iou_matching(

                selected_tracks,

                selected_detections,

                self.iou_threshold
            )

            for local_track, local_detection in iou_matches:

                original_track = iou_track_indices[
                    local_track
                ]

                original_detection = iou_detection_indices[
                    local_detection
                ]

                matches.append(
                    (
                        original_track,
                        original_detection
                    )
                )

                matched_tracks.add(
                    original_track
                )

                unmatched_detections.discard(
                    original_detection
                )

            self.total_iou_matches += len(
                iou_matches
            )

        # -------------------------------------------------------------
        # Step 4: Update matched tracks.
        # -------------------------------------------------------------

        for track_index, detection_index in matches:

            self.tracks[track_index].update(
                detections[detection_index]
            )

        # -------------------------------------------------------------
        # Step 5: Create tracks for unmatched detections.
        # -------------------------------------------------------------

        for detection_index in sorted(
            unmatched_detections
        ):

            self.create_track(
                detections[detection_index]
            )

        # -------------------------------------------------------------
        # Step 6: Delete stale tracks.
        # -------------------------------------------------------------

        retained_tracks = []

        for track in self.tracks:

            if track.time_since_update > self.max_age:

                self.total_deleted += 1

            else:

                retained_tracks.append(
                    track
                )

        self.tracks = retained_tracks

        return self.get_active_tracks()

    def get_active_tracks(self):

        results = []

        for track in self.tracks:

            if not track.confirmed:
                continue

            if track.time_since_update != 0:
                continue

            results.append(
                {
                    "id": track.track_id,

                    "bbox": track.get_bbox(),

                    "velocity": track.get_velocity(),

                    "hits": track.hits,

                    "age": track.age,

                    "state": track.state,

                    "gallery_size": len(track.features)
                }
            )

        return results

    def get_summary(self):

        return {

            "frames_processed": self.frame_count,

            "tracks_created": self.total_created,

            "tracks_deleted": self.total_deleted,

            "tracks_remaining": len(self.tracks),

            "appearance_matches": self.total_appearance_matches,

            "iou_fallback_matches": self.total_iou_matches,

            "next_track_id": self.next_id
        }


# =====================================================================
# REAL VIDEO DETECTOR
# =====================================================================

class BackgroundMotionDetector:

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

            detections.append(
                np.array(
                    [
                        x,
                        y,
                        x + w,
                        y + h
                    ],
                    dtype=np.float64
                )
            )

        detections.sort(

            key=lambda box: (

                (box[2] - box[0])

                * (box[3] - box[1])
            ),

            reverse=True
        )

        return detections, mask


# =====================================================================
# VIDEO DETECTIONS WITH APPEARANCE FEATURES
# =====================================================================

def extract_detections(
    frame,
    boxes,
    extractor
):

    detections = []

    for box in boxes:

        feature = extractor.extract(
            frame,
            box
        )

        detections.append(
            Detection(
                bbox=box,
                feature=feature
            )
        )

    return detections


# =====================================================================
# VISUALIZATION
# =====================================================================

def draw_tracking_results(
    frame,
    raw_boxes,
    tracks,
    frame_number,
    processing_fps
):

    output = frame.copy()

    # Raw detections: gray.

    for box in raw_boxes:

        x1, y1, x2, y2 = map(
            int,
            box
        )

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            (120, 120, 120),

            1
        )

    # Confirmed tracks.

    for track in tracks:

        x1, y1, x2, y2 = map(
            int,
            track["bbox"]
        )

        track_id = track["id"]

        vx = track["velocity"][0]
        vy = track["velocity"][1]

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

            2
        )

        label = (

            f"ID {track_id} "

            f"vx={vx:.1f} "

            f"vy={vy:.1f}"
        )

        cv2.putText(

            output,

            label,

            (x1, max(20, y1 - 8)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            color,

            2
        )

        cx = int(
            (x1 + x2) / 2
        )

        cy = int(
            (y1 + y2) / 2
        )

        cv2.circle(

            output,

            (cx, cy),

            4,

            color,

            -1
        )

    cv2.putText(

        output,

        f"Frame: {frame_number}",

        (15, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 255),

        2
    )

    cv2.putText(

        output,

        f"FPS: {processing_fps:.1f}",

        (15, 60),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 255),

        2
    )

    cv2.putText(

        output,

        f"Confirmed tracks: {len(tracks)}",

        (15, 90),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 255),

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

    detector = BackgroundMotionDetector(

        min_area=args.min_area,

        history=args.history,

        var_threshold=args.var_threshold
    )

    extractor = AppearanceExtractor()

    tracker = DeepSORT(

        max_age=args.max_age,

        n_init=args.n_init,

        max_cosine_distance=args.max_cosine_distance,

        max_features=args.max_features,

        iou_threshold=args.iou_threshold
    )

    frame_number = 0

    processing_times = []

    print("\nDeep SORT-inspired tracking started.")

    print("Video:", video_path)

    print("Resolution:", width, "x", height)

    print("Video FPS:", fps)

    print("Total frames:", total_frames)

    print(
        "Appearance descriptor dimensions:",
        extractor.FEATURE_DIM
    )

    try:

        while True:

            start = time.perf_counter()

            ret, frame = capture.read()

            if not ret:
                break

            frame_number += 1

            # ---------------------------------------------------------
            # Step 1: Motion detection
            # ---------------------------------------------------------

            raw_boxes, foreground = detector.detect(
                frame
            )

            # ---------------------------------------------------------
            # Step 2: Appearance feature extraction
            # ---------------------------------------------------------

            detections = extract_detections(

                frame,

                raw_boxes,

                extractor
            )

            # ---------------------------------------------------------
            # Step 3: Deep SORT tracking
            # ---------------------------------------------------------

            active_tracks = tracker.update(
                detections
            )

            elapsed = (
                time.perf_counter() - start
            )

            processing_fps = 1.0 / max(
                elapsed,
                1e-9
            )

            processing_times.append(
                elapsed
            )

            # ---------------------------------------------------------
            # Step 4: Visualization
            # ---------------------------------------------------------

            output_frame = draw_tracking_results(

                frame,

                raw_boxes,

                active_tracks,

                frame_number,

                processing_fps
            )

            writer.write(
                output_frame
            )

            if args.display:

                cv2.imshow(

                    "Lesson 43 - Deep SORT",

                    output_frame
                )

                if args.show_mask:

                    cv2.imshow(
                        "Foreground Mask",
                        foreground
                    )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            if frame_number % 100 == 0:

                print(

                    f"Frame {frame_number}/{total_frames} | "

                    f"Detections: {len(detections)} | "

                    f"Tracks: {len(active_tracks)} | "

                    f"FPS: {processing_fps:.1f}"
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

        "appearance_feature_dimensions": extractor.FEATURE_DIM,

        "tracker": tracker.get_summary()
    }

    summary_path = (

        output_path.parent

        / "deep_sort_summary.json"
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

    print("\nProcessing completed.")

    print(
        json.dumps(
            summary,
            indent=4
        )
    )

    print("\nOutput video:", output_path)

    print("Summary:", summary_path)


# =====================================================================
# NUMERICAL TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print("LESSON 43 — DEEP SORT TESTS")

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
    # Test 2: identical cosine features
    # ---------------------------------------------------------------

    feature = np.array(
        [1, 2, 3, 4],
        dtype=np.float64
    )

    assert abs(
        cosine_distance(feature, feature)
    ) < 1e-8

    print("Test 2 passed: identical appearance distance")

    # ---------------------------------------------------------------
    # Test 3: orthogonal features
    # ---------------------------------------------------------------

    feature_a = np.array(
        [1, 0],
        dtype=np.float64
    )

    feature_b = np.array(
        [0, 1],
        dtype=np.float64
    )

    assert abs(
        cosine_distance(feature_a, feature_b) - 1.0
    ) < 1e-8

    print("Test 3 passed: orthogonal appearance distance")

    # ---------------------------------------------------------------
    # Test 4: gallery nearest-neighbor distance
    # ---------------------------------------------------------------

    gallery = [

        np.array([1, 0, 0], dtype=np.float64),

        np.array([0, 1, 0], dtype=np.float64)
    ]

    query = np.array(
        [0.99, 0.01, 0],
        dtype=np.float64
    )

    distance = nearest_gallery_distance(
        gallery,
        query
    )

    assert distance < 0.02

    print("Test 4 passed: nearest gallery distance")

    # ---------------------------------------------------------------
    # Test 5: Kalman prediction
    # ---------------------------------------------------------------

    kalman = DeepSORTKalmanFilter(
        np.array(
            [100, 100, 1.0, 50],
            dtype=np.float64
        )
    )

    kalman.x[4] = 5.0
    kalman.x[5] = 2.0

    kalman.predict()

    assert abs(
        kalman.x[0] - 105.0
    ) < 1e-8

    assert abs(
        kalman.x[1] - 102.0
    ) < 1e-8

    print("Test 5 passed: Kalman prediction")

    # ---------------------------------------------------------------
    # Test 6: Kalman update
    # ---------------------------------------------------------------

    updated = kalman.update(
        np.array(
            [108, 104, 1.0, 50],
            dtype=np.float64
        )
    )

    assert np.isfinite(
        updated
    ).all()

    print("Test 6 passed: Kalman update")

    # ---------------------------------------------------------------
    # Test 7: covariance symmetry
    # ---------------------------------------------------------------

    assert np.allclose(
        kalman.P,
        kalman.P.T,
        atol=1e-8
    )

    print("Test 7 passed: covariance symmetry")

    # ---------------------------------------------------------------
    # Test 8: Mahalanobis gating
    # ---------------------------------------------------------------

    close_measurement = np.array(
        [109, 105, 1.0, 50],
        dtype=np.float64
    )

    far_measurement = np.array(
        [10000, 10000, 1.0, 50],
        dtype=np.float64
    )

    close_distance = kalman.gating_distance(
        close_measurement
    )

    far_distance = kalman.gating_distance(
        far_measurement
    )

    assert close_distance < far_distance

    assert far_distance > CHI2INV95

    print("Test 8 passed: Mahalanobis motion gating")

    # ---------------------------------------------------------------
    # Test 9: Hungarian assignment
    # ---------------------------------------------------------------

    matrix = np.array(
        [
            [4, 1, 3],
            [2, 0, 5],
            [3, 2, 2]
        ],
        dtype=np.float64
    )

    assignments = hungarian_assignment(
        matrix
    )

    total_cost = sum(
        matrix[i, j]
        for i, j in assignments
    )

    assert len(assignments) == 3

    assert abs(
        total_cost - 5.0
    ) < 1e-8

    print("Test 9 passed: Hungarian assignment")

    # ---------------------------------------------------------------
    # Test 10: assignment threshold
    # ---------------------------------------------------------------

    matrix = np.array(
        [
            [0.1, 0.9],
            [0.8, 0.2]
        ],
        dtype=np.float64
    )

    matches, unmatched_rows, unmatched_cols = (
        assign_with_threshold(
            matrix,
            0.3
        )
    )

    assert len(matches) == 2

    assert len(unmatched_rows) == 0

    assert len(unmatched_cols) == 0

    print("Test 10 passed: thresholded assignment")

    # ---------------------------------------------------------------
    # Test 11: appearance cost matrix
    # ---------------------------------------------------------------

    dummy_feature = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32
    )

    dummy_detection = Detection(

        bbox=np.array(
            [0, 0, 50, 50],
            dtype=np.float64
        ),

        feature=dummy_feature
    )

    dummy_track = DeepSORTTrack(

        dummy_detection,

        track_id=1,

        n_init=1
    )

    dummy_track.predict()

    cost = build_appearance_cost_matrix(

        [dummy_track],

        [dummy_detection],

        max_cosine_distance=0.1
    )

    assert cost.shape == (1, 1)

    assert cost[0, 0] < 1e-8

    print("Test 11 passed: appearance cost matrix")

    # ---------------------------------------------------------------
    # Test 12: track initialization
    # ---------------------------------------------------------------

    tracker = DeepSORT(
        max_age=5,
        n_init=1,
        max_cosine_distance=0.2
    )

    tracker.update(
        [
            Detection(
                np.array([0, 0, 50, 50]),
                np.array([1.0, 0.0, 0.0])
            )
        ]
    )

    assert len(tracker.tracks) == 1

    assert tracker.tracks[0].track_id == 1

    print("Test 12 passed: track initialization")

    # ---------------------------------------------------------------
    # Test 13: appearance-based identity persistence
    # ---------------------------------------------------------------

    tracker.update(
        [
            Detection(
                np.array([3, 0, 53, 50]),
                np.array([0.99, 0.01, 0.0])
            )
        ]
    )

    assert len(tracker.tracks) == 1

    assert tracker.tracks[0].track_id == 1

    assert tracker.tracks[0].hits == 2

    print("Test 13 passed: appearance identity persistence")

    # ---------------------------------------------------------------
    # Test 14: two distinct appearances
    # ---------------------------------------------------------------

    tracker = DeepSORT(
        max_age=5,
        n_init=1,
        max_cosine_distance=0.2
    )

    tracker.update(
        [
            Detection(
                np.array([0, 0, 50, 50]),
                np.array([1.0, 0.0, 0.0])
            ),

            Detection(
                np.array([200, 0, 250, 50]),
                np.array([0.0, 1.0, 0.0])
            )
        ]
    )

    assert len(tracker.tracks) == 2

    assert tracker.tracks[0].track_id != (
        tracker.tracks[1].track_id
    )

    print("Test 14 passed: two distinct appearance tracks")

    # ---------------------------------------------------------------
    # Test 15: appearance gallery update
    # ---------------------------------------------------------------

    track = tracker.tracks[0]

    old_size = len(
        track.features
    )

    tracker.update(
        [
            Detection(
                np.array([5, 0, 55, 50]),
                np.array([1.0, 0.0, 0.0])
            ),

            Detection(
                np.array([205, 0, 255, 50]),
                np.array([0.0, 1.0, 0.0])
            )
        ]
    )

    assert len(
        tracker.tracks[0].features
    ) == old_size + 1

    print("Test 15 passed: appearance gallery update")

    # ---------------------------------------------------------------
    # Test 16: track confirmation
    # ---------------------------------------------------------------

    tracker = DeepSORT(
        max_age=5,
        n_init=3,
        max_cosine_distance=0.2
    )

    detection = Detection(
        np.array([0, 0, 50, 50]),
        np.array([1.0, 0.0, 0.0])
    )

    tracker.update([detection])

    assert not tracker.tracks[0].confirmed

    tracker.update([detection])

    assert not tracker.tracks[0].confirmed

    tracker.update([detection])

    assert tracker.tracks[0].confirmed

    print("Test 16 passed: track confirmation")

    # ---------------------------------------------------------------
    # Test 17: track deletion
    # ---------------------------------------------------------------

    tracker = DeepSORT(
        max_age=2,
        n_init=1
    )

    tracker.update([detection])

    tracker.update([])

    assert len(tracker.tracks) == 1

    tracker.update([])

    assert len(tracker.tracks) == 1

    tracker.update([])

    assert len(tracker.tracks) == 0

    print("Test 17 passed: stale-track deletion")

    print("\nAll 17 tests passed.\n")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description="Lesson 43: Deep SORT-inspired tracking"
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/lesson43/deep_sort_output.mp4"
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=300
    )

    parser.add_argument(
        "--max-age",
        type=int,
        default=30
    )

    parser.add_argument(
        "--n-init",
        type=int,
        default=3
    )

    parser.add_argument(
        "--max-cosine-distance",
        type=float,
        default=DEFAULT_MAX_COSINE_DISTANCE
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD
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

    if args.min_area < 1:
        parser.error("--min-area must be positive")

    if args.max_age < 0:
        parser.error("--max-age cannot be negative")

    if args.n_init < 1:
        parser.error("--n-init must be at least 1")

    if not 0.0 <= args.max_cosine_distance <= 2.0:
        parser.error(
            "--max-cosine-distance must be between 0 and 2"
        )

    if args.max_features < 1:
        parser.error("--max-features must be positive")

    if not 0.0 <= args.iou_threshold <= 1.0:
        parser.error(
            "--iou-threshold must be between 0 and 1"
        )

    print("=" * 70)

    print("LESSON 43 — DEEP SORT-INSPIRED TRACKING")

    print("=" * 70)

    run_tests()

    if args.check_only:

        print("Check-only mode completed.")

        return

    if args.video is None:

        print("\nNo video supplied.")

        print(
            "Example:"
        )

        print(
            "python lesson43_deep_sort_object_tracking.py "
            "--video conveyor_video.mp4"
        )

        return

    args.display = not args.no_display

    process_video(args)


if __name__ == "__main__":

    main()