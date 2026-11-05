"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 48 — LONG-TERM OBJECT TRACKING AND TRACKLET MANAGEMENT
======================================================================

Filename:
    lesson48_long_term_object_tracking.py

PHASE:
    Phase 4 — Tracking and Visual Memory

======================================================================
LEARNING OBJECTIVES
======================================================================

1. Understand the difference between detections, tracks and tracklets.
2. Implement a track lifecycle:
       TENTATIVE -> CONFIRMED -> LOST -> REMOVED
3. Preserve object identity during temporary occlusion.
4. Recover lost objects using appearance descriptors.
5. Combine motion, IoU and appearance for data association.
6. Implement a two-stage association strategy.
7. Maintain a history of object trajectories.
8. Evaluate tracking behavior with numerical tests.
9. Process real video.

======================================================================
1. DETECTION, TRACK AND TRACKLET
======================================================================

DETECTION:
    An object detected in one individual frame.

TRACK:
    A continuously maintained estimate of an object's identity,
    position and motion.

TRACKLET:
    A temporal sequence of observations belonging to one object.

For example:

    Frame 1: Bottle detected -> ID 1
    Frame 2: Bottle detected -> ID 1
    Frame 3: Bottle detected -> ID 1
    Frame 4: Bottle occluded
    Frame 5: Bottle occluded
    Frame 6: Bottle detected -> ID 1

The tracker should preserve ID 1 if it can confidently associate
the returning detection with the original tracklet.

======================================================================
2. TRACK LIFECYCLE
======================================================================

TENTATIVE:
    Newly detected object.
    Not enough observations to trust its identity yet.

CONFIRMED:
    Object has been detected consistently.
    It is considered a valid track.

LOST:
    Previously confirmed object is temporarily not detected.
    Its identity and appearance information are retained.

REMOVED:
    Object has been missing for too long.
    Its active track is deleted.

State transitions:

    TENTATIVE ------> CONFIRMED
        |
        +-----------> REMOVED

    CONFIRMED ------> LOST

    LOST ------------> CONFIRMED
        |
        +------------> REMOVED

======================================================================
3. WHY TRACKLETS MATTER
======================================================================

A simple tracker often does this:

    Detection disappears
          |
          v
    Delete track
          |
          v
    Detection returns
          |
          v
    Create new ID

This causes ID switches.

A tracklet-based tracker instead does:

    Detection disappears
          |
          v
    Preserve tracklet
          |
          v
    Search for matching detection
          |
          v
    Recover previous ID

This is particularly important for waste sorting.

A bottle may temporarily disappear behind a conveyor object or
another item before becoming visible again.

======================================================================
4. APPEARANCE DESCRIPTORS
======================================================================

We represent an object's appearance using an HSV histogram.

For an image crop:

    H = Hue
    S = Saturation
    V = Value

We calculate a normalized histogram over H and S.

This provides a compact appearance descriptor.

Histogram comparison:

    d = BhattacharyyaDistance(histogram_A, histogram_B)

OpenCV:

    cv2.compareHist(
        histogram_A,
        histogram_B,
        cv2.HISTCMP_BHATTACHARYYA
    )

Interpretation:

    d = 0:
        Identical histograms.

    d approaches 1:
        Increasing appearance difference.

IMPORTANT:
    HSV histograms are not learned identity embeddings.

They are useful for this educational tracker but can confuse
objects with similar colors.

======================================================================
5. DATA ASSOCIATION
======================================================================

For each track and detection, calculate:

    IoU cost:
        1 - IoU(track_bbox, detection_bbox)

    Appearance cost:
        Bhattacharyya distance

    Motion cost:
        Normalized center distance

Combined cost:

    C = w_iou * IoU_cost
      + w_app * Appearance_cost
      + w_motion * Motion_cost

Association is solved using the Hungarian algorithm.

======================================================================
6. TWO-STAGE ASSOCIATION
======================================================================

STAGE 1:
    Match current detections to active tracks.

STAGE 2:
    Match remaining detections to lost tracks.

This prioritizes continuous tracks before attempting recovery.

A lost track is recovered only if it passes the association
thresholds.

======================================================================
7. TRACK MEMORY
======================================================================

Each track stores:

    track_id
    bounding_box
    class_id
    appearance_descriptor
    age
    hits
    missed_frames
    lifecycle_state
    trajectory
    last_seen_frame

A lost track retains its appearance descriptor and last known
position for possible recovery.

======================================================================
8. IMPORTANT LIMITATIONS
======================================================================

This implementation uses:

    - MOG2 foreground detection
    - HSV histogram appearance
    - Bounding-box overlap
    - Center-distance motion gating
    - Hungarian assignment

It is NOT a complete Deep SORT or Re-ID neural network.

MOG2 works best with a mostly stationary camera and a reasonably
stable background.

For moving robot cameras, combine this lesson with Lesson 47's
camera-motion compensation.

For real waste sorting, replace MOG2 with YOLO or an instance
segmentation detector. The tracklet manager can remain.

======================================================================
9. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    Camera
       |
       v
    YOLO / Instance Segmentation
       |
       v
    Detection boxes and classes
       |
       v
    Tracklet Manager
       |
       +---- Active tracks
       |
       +---- Lost tracks
       |
       +---- Appearance memory
       |
       v
    Stable object IDs
       |
       v
    Object localization
       |
       v
    Grasp planning
       |
       v
    ACT manipulation policy

Stable object IDs help prevent the robot from treating the same
temporarily occluded item as a completely new object.

======================================================================
INSTALL
======================================================================

    pip install numpy opencv-python scipy matplotlib

======================================================================
RUN
======================================================================

    python lesson48_long_term_object_tracking.py --check-only

    python lesson48_long_term_object_tracking.py \
        --video conveyor_video.mp4

    python lesson48_long_term_object_tracking.py \
        --video conveyor_video.mp4 \
        --no-display

======================================================================
OUTPUT
======================================================================

    runs/lesson48/
        long_term_tracking.mp4
        tracking_summary.json
        tracklets.json
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

from scipy.optimize import linear_sum_assignment


# =====================================================================
# CONFIGURATION
# =====================================================================

HISTOGRAM_H_BINS = 24

HISTOGRAM_S_BINS = 16

MIN_CROP_SIZE = 4

DEFAULT_MAX_LOST_FRAMES = 45

DEFAULT_MIN_HITS = 3

DEFAULT_MAX_CENTER_DISTANCE = 120.0

DEFAULT_APPEARANCE_THRESHOLD = 0.55

DEFAULT_IOU_THRESHOLD = 0.05


# =====================================================================
# TRACK LIFECYCLE
# =====================================================================

class TrackState(Enum):

    TENTATIVE = "tentative"

    CONFIRMED = "confirmed"

    LOST = "lost"

    REMOVED = "removed"


# =====================================================================
# BASIC GEOMETRY
# =====================================================================

def bbox_center(bbox):

    x1, y1, x2, y2 = np.asarray(
        bbox,
        dtype=np.float64
    )

    return np.array(
        [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0
        ],
        dtype=np.float64
    )


def bbox_area(bbox):

    x1, y1, x2, y2 = np.asarray(
        bbox,
        dtype=np.float64
    )

    return max(
        0.0,
        x2 - x1
    ) * max(
        0.0,
        y2 - y1
    )


def intersection_over_union(box_a, box_b):

    ax1, ay1, ax2, ay2 = np.asarray(
        box_a,
        dtype=np.float64
    )

    bx1, by1, bx2, by2 = np.asarray(
        box_b,
        dtype=np.float64
    )

    x1 = max(ax1, bx1)

    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)

    y2 = min(ay2, by2)

    intersection = (

        max(0.0, x2 - x1)

        * max(0.0, y2 - y1)
    )

    union = (

        bbox_area(box_a)

        + bbox_area(box_b)

        - intersection
    )

    return intersection / max(
        union,
        1e-12
    )


def center_distance(box_a, box_b):

    return float(
        np.linalg.norm(
            bbox_center(box_a) - bbox_center(box_b)
        )
    )


def clip_bbox(bbox, width, height):

    x1, y1, x2, y2 = np.asarray(
        bbox,
        dtype=np.float64
    )

    return np.array(

        [
            np.clip(x1, 0, width - 1),

            np.clip(y1, 0, height - 1),

            np.clip(x2, 0, width - 1),

            np.clip(y2, 0, height - 1)
        ],

        dtype=np.float64
    )


# =====================================================================
# APPEARANCE DESCRIPTOR
# =====================================================================

def extract_hsv_histogram(
    frame,
    bbox,
    h_bins=HISTOGRAM_H_BINS,
    s_bins=HISTOGRAM_S_BINS
):

    """
    Extract normalized HSV appearance histogram.

    We use H and S channels.

    The V channel is omitted because illumination changes can
    significantly affect brightness.

    Returns:
        A normalized 1D float32 histogram.
    """

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = clip_bbox(
        bbox,
        width,
        height
    ).astype(int)

    if x2 <= x1 or y2 <= y1:

        return np.zeros(
            h_bins * s_bins,
            dtype=np.float32
        )

    if x2 - x1 < MIN_CROP_SIZE:

        return np.zeros(
            h_bins * s_bins,
            dtype=np.float32
        )

    if y2 - y1 < MIN_CROP_SIZE:

        return np.zeros(
            h_bins * s_bins,
            dtype=np.float32
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

        [0, 1],

        None,

        [h_bins, s_bins],

        [0, 180, 0, 256]
    )

    histogram = histogram.astype(
        np.float32
    )

    total = float(
        histogram.sum()
    )

    if total > 0:

        histogram /= total

    return histogram.flatten()


def appearance_distance(hist_a, hist_b):

    """
    Bhattacharyya histogram distance.

    0 = identical distributions.
    Larger values = greater appearance difference.
    """

    hist_a = np.asarray(
        hist_a,
        dtype=np.float32
    )

    hist_b = np.asarray(
        hist_b,
        dtype=np.float32
    )

    if hist_a.shape != hist_b.shape:

        raise ValueError(
            "Appearance descriptors must have matching shapes."
        )

    if hist_a.size == 0:

        return 1.0

    if hist_a.sum() <= 0 or hist_b.sum() <= 0:

        return 1.0

    distance = cv2.compareHist(

        hist_a.reshape(-1, 1),

        hist_b.reshape(-1, 1),

        cv2.HISTCMP_BHATTACHARYYA
    )

    return float(
        np.clip(distance, 0.0, 1.0)
    )


def update_appearance(
    old_histogram,
    new_histogram,
    alpha=0.15
):

    """
    Exponential moving average of appearance descriptors.

        h_new = (1-alpha)*h_old + alpha*h_observation

    A smaller alpha preserves long-term appearance memory.
    """

    old_histogram = np.asarray(
        old_histogram,
        dtype=np.float32
    )

    new_histogram = np.asarray(
        new_histogram,
        dtype=np.float32
    )

    if old_histogram.shape != new_histogram.shape:

        raise ValueError(
            "Histogram dimensions do not match."
        )

    updated = (

        (1.0 - alpha) * old_histogram

        + alpha * new_histogram
    )

    total = float(
        updated.sum()
    )

    if total > 0:

        updated /= total

    return updated.astype(
        np.float32
    )


# =====================================================================
# DETECTION DATA
# =====================================================================

@dataclass
class Detection:

    bbox: np.ndarray

    appearance: np.ndarray

    class_id: int = 0

    confidence: float = 1.0

    def __post_init__(self):

        self.bbox = np.asarray(
            self.bbox,
            dtype=np.float64
        ).reshape(4)

        self.appearance = np.asarray(
            self.appearance,
            dtype=np.float32
        ).flatten()


# =====================================================================
# TRACKLET DATA
# =====================================================================

@dataclass
class Tracklet:

    track_id: int

    bbox: np.ndarray

    appearance: np.ndarray

    class_id: int

    frame_index: int

    min_hits: int = DEFAULT_MIN_HITS

    state: TrackState = TrackState.TENTATIVE

    age: int = 1

    hits: int = 1

    missed_frames: int = 0

    consecutive_hits: int = 1

    first_seen_frame: int = 0

    last_seen_frame: int = 0

    trajectory: list = field(
        default_factory=list
    )

    confidence_history: list = field(
        default_factory=list
    )

    def __post_init__(self):

        self.bbox = np.asarray(
            self.bbox,
            dtype=np.float64
        ).reshape(4)

        self.appearance = np.asarray(
            self.appearance,
            dtype=np.float32
        ).flatten()

        center = bbox_center(
            self.bbox
        )

        self.trajectory.append(
            center.tolist()
        )

        self.first_seen_frame = self.frame_index

        self.last_seen_frame = self.frame_index

    @property
    def center(self):

        return bbox_center(
            self.bbox
        )

    @property
    def is_confirmed(self):

        return self.state in (

            TrackState.CONFIRMED,

            TrackState.LOST
        )

    def mark_missed(self):

        self.age += 1

        self.missed_frames += 1

        self.consecutive_hits = 0

        if self.state == TrackState.TENTATIVE:

            self.state = TrackState.REMOVED

        elif self.state == TrackState.CONFIRMED:

            self.state = TrackState.LOST

    def update(
        self,
        detection,
        frame_index,
        appearance_alpha=0.15
    ):

        self.bbox = detection.bbox.copy()

        self.appearance = update_appearance(

            self.appearance,

            detection.appearance,

            alpha=appearance_alpha
        )

        self.age += 1

        self.hits += 1

        self.missed_frames = 0

        self.consecutive_hits += 1

        self.last_seen_frame = frame_index

        self.trajectory.append(
            self.center.tolist()
        )

        self.confidence_history.append(
            detection.confidence
        )

        # Tentative tracks become confirmed after enough
        # consecutive detections.

        if (

            self.state == TrackState.TENTATIVE

            and self.consecutive_hits >= self.min_hits
        ):

            self.state = TrackState.CONFIRMED

        elif self.state == TrackState.LOST:

            # Recover the original identity.

            self.state = TrackState.CONFIRMED

    def to_dict(self):

        return {

            "track_id": self.track_id,

            "class_id": self.class_id,

            "state": self.state.value,

            "age": self.age,

            "hits": self.hits,

            "missed_frames": self.missed_frames,

            "first_seen_frame": self.first_seen_frame,

            "last_seen_frame": self.last_seen_frame,

            "trajectory": self.trajectory,

            "average_confidence": (

                float(
                    np.mean(self.confidence_history)
                )

                if self.confidence_history else None
            )
        }


# =====================================================================
# ASSOCIATION COST
# =====================================================================

@dataclass
class AssociationWeights:

    iou: float = 0.40

    appearance: float = 0.40

    motion: float = 0.20

    def normalize(self):

        total = (
            self.iou
            + self.appearance
            + self.motion
        )

        if total <= 0:

            raise ValueError(
                "Association weights must have positive sum."
            )

        self.iou /= total

        self.appearance /= total

        self.motion /= total


def association_cost(
    track,
    detection,
    max_center_distance,
    appearance_threshold,
    weights
):

    """
    Calculate combined association cost.

    Invalid associations receive infinity.

    The center-distance gate prevents a visually similar object
    far away from stealing an existing track ID.
    """

    if track.class_id != detection.class_id:

        return np.inf

    distance = center_distance(

        track.bbox,

        detection.bbox
    )

    if distance > max_center_distance:

        return np.inf

    iou_value = intersection_over_union(

        track.bbox,

        detection.bbox
    )

    app_distance = appearance_distance(

        track.appearance,

        detection.appearance
    )

    # Appearance gating.
    #
    # Lost tracks rely more heavily on appearance because
    # they may have been missing for several frames.

    if (

        track.state == TrackState.LOST

        and app_distance > appearance_threshold
    ):

        return np.inf

    # For active tracks, accept either reasonable overlap or
    # reasonable appearance similarity.

    if (

        track.state == TrackState.CONFIRMED

        and iou_value < 0.01

        and app_distance > appearance_threshold
    ):

        return np.inf

    normalized_motion = min(

        distance / max(
            max_center_distance,
            1e-9
        ),

        1.0
    )

    cost = (

        weights.iou * (1.0 - iou_value)

        + weights.appearance * app_distance

        + weights.motion * normalized_motion
    )

    return float(cost)


# =====================================================================
# TRACKLET MANAGER
# =====================================================================

class TrackletManager:

    """
    Long-term object identity manager.

    Association stages:

        Stage 1:
            Match detections with confirmed tracks.

        Stage 2:
            Match remaining detections with lost tracks.

        Stage 3:
            Match remaining detections with tentative tracks.

        Stage 4:
            Create new tracklets for unmatched detections.

    The lost-track stage is the key addition in this lesson.
    """

    def __init__(

        self,

        min_hits=DEFAULT_MIN_HITS,

        max_lost_frames=DEFAULT_MAX_LOST_FRAMES,

        max_center_distance=DEFAULT_MAX_CENTER_DISTANCE,

        appearance_threshold=DEFAULT_APPEARANCE_THRESHOLD,

        max_association_cost=0.75,

        appearance_alpha=0.15,

        weights=None
    ):

        self.min_hits = min_hits

        self.max_lost_frames = max_lost_frames

        self.max_center_distance = max_center_distance

        self.appearance_threshold = appearance_threshold

        self.max_association_cost = max_association_cost

        self.appearance_alpha = appearance_alpha

        self.weights = weights or AssociationWeights()

        self.weights.normalize()

        self.tracklets = {}

        self.next_track_id = 1

        self.frame_index = 0

        self.events = {

            "created": 0,

            "confirmed": 0,

            "lost": 0,

            "recovered": 0,

            "removed": 0,

            "matches": 0
        }

    def _new_tracklet(self, detection):

        tracklet = Tracklet(

            track_id=self.next_track_id,

            bbox=detection.bbox.copy(),

            appearance=detection.appearance.copy(),

            class_id=detection.class_id,

            frame_index=self.frame_index,

            min_hits=self.min_hits
        )

        self.tracklets[
            self.next_track_id
        ] = tracklet

        self.next_track_id += 1

        self.events["created"] += 1

        return tracklet

    def _associate(
        self,
        track_ids,
        detection_indices,
        detections
    ):

        """
        Hungarian assignment for a selected track group
        and remaining detection indices.
        """

        if not track_ids or not detection_indices:

            return [], list(track_ids), list(detection_indices)

        cost_matrix = np.full(

            (
                len(track_ids),

                len(detection_indices)
            ),

            np.inf,

            dtype=np.float64
        )

        for row, track_id in enumerate(track_ids):

            track = self.tracklets[track_id]

            for col, detection_index in enumerate(
                detection_indices
            ):

                detection = detections[detection_index]

                cost_matrix[row, col] = association_cost(

                    track,

                    detection,

                    self.max_center_distance,

                    self.appearance_threshold,

                    self.weights
                )

        # Hungarian assignment requires finite values.
        # Replace invalid values with a very large cost.

        solver_cost = np.where(

            np.isfinite(cost_matrix),

            cost_matrix,

            1e6
        )

        rows, cols = linear_sum_assignment(
            solver_cost
        )

        matches = []

        matched_track_ids = set()

        matched_detection_indices = set()

        for row, col in zip(rows, cols):

            original_cost = cost_matrix[row, col]

            if not np.isfinite(original_cost):
                continue

            if original_cost > self.max_association_cost:
                continue

            track_id = track_ids[row]

            detection_index = detection_indices[col]

            matches.append(
                (
                    track_id,
                    detection_index,
                    float(original_cost)
                )
            )

            matched_track_ids.add(
                track_id
            )

            matched_detection_indices.add(
                detection_index
            )

        unmatched_tracks = [

            track_id

            for track_id in track_ids

            if track_id not in matched_track_ids
        ]

        unmatched_detections = [

            detection_index

            for detection_index in detection_indices

            if detection_index not in matched_detection_indices
        ]

        return (

            matches,

            unmatched_tracks,

            unmatched_detections
        )

    def update(self, detections):

        self.frame_index += 1

        detections = list(
            detections
        )

        # -------------------------------------------------------------
        # STEP 1: Record which tracks were lost before this frame.
        # -------------------------------------------------------------

        previously_lost = {

            track_id

            for track_id, track in self.tracklets.items()

            if track.state == TrackState.LOST
        }

        # -------------------------------------------------------------
        # STEP 2: Select tracks for each association stage.
        # -------------------------------------------------------------

        confirmed_ids = [

            track_id

            for track_id, track in self.tracklets.items()

            if track.state == TrackState.CONFIRMED
        ]

        lost_ids = [

            track_id

            for track_id, track in self.tracklets.items()

            if track.state == TrackState.LOST
        ]

        tentative_ids = [

            track_id

            for track_id, track in self.tracklets.items()

            if track.state == TrackState.TENTATIVE
        ]

        available_detections = list(
            range(len(detections))
        )

        matched_track_ids = set()

        # -------------------------------------------------------------
        # STEP 3: Associate confirmed tracks first.
        # -------------------------------------------------------------

        matches, unmatched_confirmed, available_detections = (
            self._associate(

                confirmed_ids,

                available_detections,

                detections
            )
        )

        all_matches = list(
            matches
        )

        matched_track_ids.update(

            track_id

            for track_id, _, _ in matches
        )

        # -------------------------------------------------------------
        # STEP 4: Attempt recovery of lost tracks.
        # -------------------------------------------------------------

        matches, unmatched_lost, available_detections = (
            self._associate(

                lost_ids,

                available_detections,

                detections
            )
        )

        all_matches.extend(
            matches
        )

        matched_track_ids.update(

            track_id

            for track_id, _, _ in matches
        )

        # -------------------------------------------------------------
        # STEP 5: Associate tentative tracks.
        # -------------------------------------------------------------

        matches, unmatched_tentative, available_detections = (
            self._associate(

                tentative_ids,

                available_detections,

                detections
            )
        )

        all_matches.extend(
            matches
        )

        matched_track_ids.update(

            track_id

            for track_id, _, _ in matches
        )

        # -------------------------------------------------------------
        # STEP 6: Update matched tracks.
        # -------------------------------------------------------------

        for track_id, detection_index, cost in all_matches:

            track = self.tracklets[track_id]

            was_lost = track.state == TrackState.LOST

            was_tentative = (
                track.state == TrackState.TENTATIVE
            )

            track.update(

                detections[detection_index],

                self.frame_index,

                self.appearance_alpha
            )

            self.events["matches"] += 1

            if was_lost:

                self.events["recovered"] += 1

            if (

                was_tentative

                and track.state == TrackState.CONFIRMED
            ):

                self.events["confirmed"] += 1

        # -------------------------------------------------------------
        # STEP 7: Mark unmatched tracks as missed.
        # -------------------------------------------------------------

        unmatched_all = (

            unmatched_confirmed

            + unmatched_lost

            + unmatched_tentative
        )

        for track_id in unmatched_all:

            track = self.tracklets[track_id]

            old_state = track.state

            track.mark_missed()

            if (

                old_state == TrackState.CONFIRMED

                and track.state == TrackState.LOST
            ):

                self.events["lost"] += 1

        # -------------------------------------------------------------
        # STEP 8: Create new tracklets for remaining detections.
        # -------------------------------------------------------------

        for detection_index in available_detections:

            self._new_tracklet(
                detections[detection_index]
            )

        # -------------------------------------------------------------
        # STEP 9: Remove expired tracklets.
        # -------------------------------------------------------------

        for track_id, track in self.tracklets.items():

            if track.state == TrackState.LOST:

                if track.missed_frames > self.max_lost_frames:

                    track.state = TrackState.REMOVED

                    self.events["removed"] += 1

        # -------------------------------------------------------------
        # STEP 10: Return visible tracks.
        # -------------------------------------------------------------

        visible = [

            track

            for track in self.tracklets.values()

            if track.state in (

                TrackState.TENTATIVE,

                TrackState.CONFIRMED
            )
        ]

        return visible

    def active_tracks(self):

        return [

            track

            for track in self.tracklets.values()

            if track.state == TrackState.CONFIRMED
        ]

    def lost_tracks(self):

        return [

            track

            for track in self.tracklets.values()

            if track.state == TrackState.LOST
        ]

    def all_tracklets(self):

        return list(
            self.tracklets.values()
        )

    def summary(self):

        states = {

            state.value: 0

            for state in TrackState
        }

        for track in self.tracklets.values():

            states[track.state.value] += 1

        return {

            "frames_processed": self.frame_index,

            "total_tracklets": len(
                self.tracklets
            ),

            "next_track_id": self.next_track_id,

            "track_states": states,

            "events": self.events.copy()
        }


# =====================================================================
# DEMONSTRATION DETECTOR
# =====================================================================

class MotionDetector:

    """
    MOG2 foreground detector.

    This is only a demonstration detector.

    It does not identify bottles, cups, cans or other semantic
    categories.

    Replace it with YOLO or an instance segmentation model
    for real waste sorting.
    """

    def __init__(

        self,

        min_area=600,

        history=500,

        var_threshold=32
    ):

        self.min_area = min_area

        self.background = cv2.createBackgroundSubtractorMOG2(

            history=history,

            varThreshold=var_threshold,

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

            histogram = extract_hsv_histogram(
                frame,
                bbox
            )

            detections.append(

                Detection(

                    bbox=bbox,

                    appearance=histogram,

                    class_id=0,

                    confidence=1.0
                )
            )

        return detections, mask


# =====================================================================
# VISUALIZATION
# =====================================================================

STATE_COLORS = {

    TrackState.TENTATIVE: (0, 255, 255),

    TrackState.CONFIRMED: (0, 255, 0),

    TrackState.LOST: (0, 165, 255),

    TrackState.REMOVED: (100, 100, 100)
}


def draw_tracking_results(
    frame,
    tracks,
    manager,
    frame_index
):

    output = frame.copy()

    for track in tracks:

        if track.state == TrackState.TENTATIVE:

            # Show tentative tracks only after at least one hit.

            color = STATE_COLORS[
                TrackState.TENTATIVE
            ]

        else:

            color = STATE_COLORS[
                track.state
            ]

        x1, y1, x2, y2 = track.bbox.astype(
            int
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

            f"{track.state.value}"
        )

        cv2.putText(

            output,

            label,

            (x1, max(18, y1 - 7)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            color,

            2
        )

        # Draw trajectory.

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

    # Draw lost-track memory separately.

    lost_tracks = manager.lost_tracks()

    cv2.putText(

        output,

        f"Frame: {frame_index}",

        (15, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )

    cv2.putText(

        output,

        f"Active: {len(manager.active_tracks())}",

        (15, 60),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 255, 0),

        2
    )

    cv2.putText(

        output,

        f"Lost memory: {len(lost_tracks)}",

        (15, 85),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 165, 255),

        2
    )

    cv2.putText(

        output,

        f"Recovered: {manager.events['recovered']}",

        (15, 110),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 0),

        2
    )

    return output


# =====================================================================
# TRACKLET EXPORT
# =====================================================================

def export_tracklets(manager, output_path):

    records = [

        track.to_dict()

        for track in manager.all_tracklets()
    ]

    with open(
        output_path,
        "w"
    ) as file:

        json.dump(

            records,

            file,

            indent=4
        )


# =====================================================================
# VIDEO PROCESSING
# =====================================================================

def process_video(args):

    video_path = Path(
        args.video
    )

    if not video_path.exists():

        raise FileNotFoundError(
            video_path
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
            "Could not create output video."
        )

    detector = MotionDetector(
        min_area=args.min_area
    )

    manager = TrackletManager(

        min_hits=args.min_hits,

        max_lost_frames=args.max_lost_frames,

        max_center_distance=args.max_center_distance,

        appearance_threshold=args.appearance_threshold,

        max_association_cost=args.max_association_cost
    )

    frame_index = 0

    processing_times = []

    print("\nStarting long-term object tracking.")

    try:

        while True:

            start = time.perf_counter()

            ret, frame = capture.read()

            if not ret:
                break

            frame_index += 1

            detections, mask = detector.detect(
                frame
            )

            tracks = manager.update(
                detections
            )

            output = draw_tracking_results(

                frame,

                tracks,

                manager,

                frame_index
            )

            elapsed = (
                time.perf_counter() - start
            )

            processing_times.append(
                elapsed
            )

            writer.write(
                output
            )

            if not args.no_display:

                cv2.imshow(

                    "Lesson 48 - Long-Term Tracking",

                    output
                )

                if args.show_mask:

                    cv2.imshow(
                        "Foreground Mask",
                        mask
                    )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

            if frame_index % 100 == 0:

                print(

                    f"Frame {frame_index:6d} | "

                    f"Detections: {len(detections):3d} | "

                    f"Active: {len(manager.active_tracks()):3d} | "

                    f"Lost: {len(manager.lost_tracks()):3d} | "

                    f"Recovered: {manager.events['recovered']:3d}"
                )

    finally:

        capture.release()

        writer.release()

        cv2.destroyAllWindows()

    average_time = (

        float(np.mean(processing_times))

        if processing_times else 0.0
    )

    summary = manager.summary()

    summary.update(

        {

            "input_video": str(video_path),

            "output_video": str(output_path),

            "resolution": [
                width,
                height
            ],

            "average_processing_fps": (

                1.0 / average_time

                if average_time > 0 else 0.0
            )
        }
    )

    summary_path = (

        output_path.parent

        / "tracking_summary.json"
    )

    tracklets_path = (

        output_path.parent

        / "tracklets.json"
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

    export_tracklets(

        manager,

        tracklets_path
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

    print("Tracklets:", tracklets_path)


# =====================================================================
# UNIT TEST HELPERS
# =====================================================================

def make_histogram(color):

    """
    Create a normalized synthetic histogram for unit tests.

    This is numerical test data, not a synthetic photograph.
    """

    histogram = np.zeros(
        HISTOGRAM_H_BINS * HISTOGRAM_S_BINS,
        dtype=np.float32
    )

    histogram[color] = 1.0

    return histogram


def make_detection(
    bbox,
    histogram=None,
    class_id=0
):

    if histogram is None:

        histogram = make_histogram(0)

    return Detection(

        bbox=np.asarray(
            bbox,
            dtype=np.float64
        ),

        appearance=histogram,

        class_id=class_id,

        confidence=0.95
    )


# =====================================================================
# UNIT TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print("LESSON 48 — LONG-TERM TRACKING TESTS")

    print("=" * 70)

    # ---------------------------------------------------------------
    # Test 1: bounding-box center.
    # ---------------------------------------------------------------

    bbox = np.array(
        [10, 20, 50, 80]
    )

    assert np.allclose(
        bbox_center(bbox),
        [30, 50]
    )

    print("Test 1 passed: bounding-box center")

    # ---------------------------------------------------------------
    # Test 2: bounding-box area.
    # ---------------------------------------------------------------

    assert bbox_area(bbox) == 2400

    print("Test 2 passed: bounding-box area")

    # ---------------------------------------------------------------
    # Test 3: identical-box IoU.
    # ---------------------------------------------------------------

    assert abs(
        intersection_over_union(bbox, bbox) - 1.0
    ) < 1e-8

    print("Test 3 passed: identical-box IoU")

    # ---------------------------------------------------------------
    # Test 4: non-overlapping-box IoU.
    # ---------------------------------------------------------------

    other = np.array(
        [100, 100, 150, 150]
    )

    assert intersection_over_union(
        bbox,
        other
    ) == 0.0

    print("Test 4 passed: non-overlapping-box IoU")

    # ---------------------------------------------------------------
    # Test 5: center distance.
    # ---------------------------------------------------------------

    box_a = np.array(
        [0, 0, 20, 20]
    )

    box_b = np.array(
        [30, 40, 50, 60]
    )

    expected_distance = np.sqrt(
        50 ** 2 + 50 ** 2
    )

    assert abs(
        center_distance(box_a, box_b)
        - expected_distance
    ) < 1e-8

    print("Test 5 passed: center distance")

    # ---------------------------------------------------------------
    # Test 6: identical appearance histograms.
    # ---------------------------------------------------------------

    hist = make_histogram(10)

    distance = appearance_distance(
        hist,
        hist.copy()
    )

    assert abs(distance) < 1e-6

    print("Test 6 passed: identical appearance")

    # ---------------------------------------------------------------
    # Test 7: different appearance histograms.
    # ---------------------------------------------------------------

    hist_a = make_histogram(0)

    hist_b = make_histogram(100)

    distance = appearance_distance(
        hist_a,
        hist_b
    )

    assert distance > 0.5

    print("Test 7 passed: different appearance")

    # ---------------------------------------------------------------
    # Test 8: appearance update remains normalized.
    # ---------------------------------------------------------------

    hist_a = make_histogram(0)

    hist_b = make_histogram(1)

    updated = update_appearance(

        hist_a,

        hist_b,

        alpha=0.2
    )

    assert abs(
        float(updated.sum()) - 1.0
    ) < 1e-6

    assert updated[0] > updated[1]

    print("Test 8 passed: appearance descriptor update")

    # ---------------------------------------------------------------
    # Test 9: histogram extraction from real image data.
    # ---------------------------------------------------------------

    rng = np.random.default_rng(42)

    image = rng.integers(

        0,

        256,

        size=(120, 160, 3),

        dtype=np.uint8
    )

    histogram = extract_hsv_histogram(

        image,

        [20, 20, 100, 100]
    )

    assert histogram.shape == (

        HISTOGRAM_H_BINS * HISTOGRAM_S_BINS
    )

    assert abs(
        float(histogram.sum()) - 1.0
    ) < 1e-5

    print("Test 9 passed: HSV histogram extraction")

    # ---------------------------------------------------------------
    # Test 10: tracklet initialization.
    # ---------------------------------------------------------------

    track = Tracklet(

        track_id=1,

        bbox=bbox,

        appearance=make_histogram(0),

        class_id=0,

        frame_index=1,

        min_hits=3
    )

    assert track.state == TrackState.TENTATIVE

    assert track.hits == 1

    assert track.age == 1

    assert len(track.trajectory) == 1

    print("Test 10 passed: tracklet initialization")

    # ---------------------------------------------------------------
    # Test 11: tentative -> confirmed.
    # ---------------------------------------------------------------

    for frame_index in [2, 3]:

        track.update(

            make_detection(
                bbox
            ),

            frame_index
        )

    assert track.state == TrackState.CONFIRMED

    assert track.hits == 3

    print("Test 11 passed: tentative to confirmed")

    # ---------------------------------------------------------------
    # Test 12: confirmed -> lost.
    # ---------------------------------------------------------------

    track.mark_missed()

    assert track.state == TrackState.LOST

    assert track.missed_frames == 1

    print("Test 12 passed: confirmed to lost")

    # ---------------------------------------------------------------
    # Test 13: lost -> confirmed recovery.
    # ---------------------------------------------------------------

    track.update(

        make_detection(
            bbox
        ),

        frame_index=5
    )

    assert track.state == TrackState.CONFIRMED

    assert track.missed_frames == 0

    print("Test 13 passed: lost-track recovery")

    # ---------------------------------------------------------------
    # Test 14: tentative track removal.
    # ---------------------------------------------------------------

    tentative = Tracklet(

        track_id=2,

        bbox=bbox,

        appearance=make_histogram(0),

        class_id=0,

        frame_index=1,

        min_hits=3
    )

    tentative.mark_missed()

    assert tentative.state == TrackState.REMOVED

    print("Test 14 passed: tentative track removal")

    # ---------------------------------------------------------------
    # Test 15: tracklet manager creates a track.
    # ---------------------------------------------------------------

    manager = TrackletManager(
        min_hits=2
    )

    first = make_detection(
        [10, 10, 50, 50]
    )

    tracks = manager.update(
        [first]
    )

    assert len(tracks) == 1

    assert tracks[0].track_id == 1

    assert tracks[0].state == TrackState.TENTATIVE

    print("Test 15 passed: manager track creation")

    # ---------------------------------------------------------------
    # Test 16: manager confirms a track.
    # ---------------------------------------------------------------

    manager.update(

        [
            make_detection(
                [11, 10, 51, 50]
            )
        ]
    )

    track = manager.tracklets[1]

    assert track.state == TrackState.CONFIRMED

    print("Test 16 passed: manager track confirmation")

    # ---------------------------------------------------------------
    # Test 17: manager marks track as lost.
    # ---------------------------------------------------------------

    manager.update(
        []
    )

    assert manager.tracklets[1].state == TrackState.LOST

    print("Test 17 passed: manager lost-track transition")

    # ---------------------------------------------------------------
    # Test 18: manager recovers same track ID.
    # ---------------------------------------------------------------

    manager.update(

        [
            make_detection(
                [13, 10, 53, 50]
            )
        ]
    )

    assert manager.tracklets[1].state == TrackState.CONFIRMED

    assert manager.next_track_id == 2

    assert manager.events["recovered"] == 1

    print("Test 18 passed: recovery preserves original ID")

    # ---------------------------------------------------------------
    # Test 19: lost track expiration.
    # ---------------------------------------------------------------

    manager = TrackletManager(

        min_hits=1,

        max_lost_frames=2
    )

    manager.update(

        [
            make_detection(
                [10, 10, 50, 50]
            )
        ]
    )

    assert manager.tracklets[1].state == TrackState.CONFIRMED

    manager.update([])
    manager.update([])
    manager.update([])

    assert manager.tracklets[1].state == TrackState.REMOVED

    print("Test 19 passed: lost-track expiration")

    # ---------------------------------------------------------------
    # Test 20: appearance gating.
    # ---------------------------------------------------------------

    manager = TrackletManager(

        min_hits=1,

        appearance_threshold=0.2,

        max_center_distance=1000,

        max_association_cost=0.75
    )

    manager.update(

        [
            make_detection(

                [10, 10, 50, 50],

                make_histogram(0)
            )
        ]
    )

    manager.update([])

    manager.update(

        [
            make_detection(

                [10, 10, 50, 50],

                make_histogram(100)
            )
        ]
    )

    # A different appearance should not recover the old ID.
    # It creates a new track instead.

    assert manager.tracklets[1].state == TrackState.LOST

    assert manager.next_track_id == 3

    assert manager.events["recovered"] == 0

    print("Test 20 passed: appearance-based recovery rejection")

    # ---------------------------------------------------------------
    # Test 21: class-aware association.
    # ---------------------------------------------------------------

    manager = TrackletManager(
        min_hits=1
    )

    manager.update(

        [
            make_detection(
                [10, 10, 50, 50],
                class_id=0
            )
        ]
    )

    manager.update(

        [
            make_detection(
                [10, 10, 50, 50],
                class_id=1
            )
        ]
    )

    assert manager.tracklets[1].state == TrackState.LOST

    assert manager.next_track_id == 3

    print("Test 21 passed: class-aware association")

    # ---------------------------------------------------------------
    # Test 22: trajectory history.
    # ---------------------------------------------------------------

    manager = TrackletManager(
        min_hits=1
    )

    positions = [

        [10, 10, 50, 50],

        [20, 10, 60, 50],

        [30, 10, 70, 50]
    ]

    for position in positions:

        manager.update(

            [
                make_detection(position)
            ]
        )

    trajectory = manager.tracklets[1].trajectory

    assert len(trajectory) == 3

    assert np.allclose(

        trajectory[-1],

        [50, 30]
    )

    print("Test 22 passed: trajectory history")

    # ---------------------------------------------------------------
    # Test 23: summary generation.
    # ---------------------------------------------------------------

    summary = manager.summary()

    assert summary["frames_processed"] == 3

    assert summary["total_tracklets"] == 1

    assert summary["track_states"]["confirmed"] == 1

    print("Test 23 passed: tracker summary")

    print("\nAll 23 tests passed.")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Lesson 48 — Long-Term Object Tracking"
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "runs/lesson48/"
            "long_term_tracking.mp4"
        )
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=600
    )

    parser.add_argument(
        "--min-hits",
        type=int,
        default=3
    )

    parser.add_argument(
        "--max-lost-frames",
        type=int,
        default=45
    )

    parser.add_argument(
        "--max-center-distance",
        type=float,
        default=120.0
    )

    parser.add_argument(
        "--appearance-threshold",
        type=float,
        default=0.55
    )

    parser.add_argument(
        "--max-association-cost",
        type=float,
        default=0.75
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

        parser.error(
            "--min-area must be positive."
        )

    if args.min_hits < 1:

        parser.error(
            "--min-hits must be at least 1."
        )

    if args.max_lost_frames < 0:

        parser.error(
            "--max-lost-frames cannot be negative."
        )

    if args.max_center_distance <= 0:

        parser.error(
            "--max-center-distance must be positive."
        )

    if not 0 <= args.appearance_threshold <= 1:

        parser.error(
            "--appearance-threshold must be between 0 and 1."
        )

    if not 0 <= args.max_association_cost <= 1:

        parser.error(
            "--max-association-cost must be between 0 and 1."
        )

    print("=" * 70)

    print(
        "LESSON 48 — LONG-TERM OBJECT TRACKING"
    )

    print("=" * 70)

    run_tests()

    if args.check_only:

        print("\nCheck-only completed.")

        return

    if args.video is None:

        print("\nNo video supplied.")

        print(

            "Run with: python lesson48_long_term_object_tracking.py "

            "--video conveyor_video.mp4"
        )

        return

    process_video(args)


if __name__ == "__main__":

    main()