"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 41 — OBJECT TRACKING FROM SCRATCH
======================================================================

Filename:
    lesson41_object_tracking_from_scratch.py

PHASE:
    Phase 4 — Tracking and Visual Memory

======================================================================
LEARNING OBJECTIVES
======================================================================

Build a multi-object tracking system using:

    1. Kalman filtering
    2. Motion prediction
    3. Bounding-box representation
    4. Intersection over Union (IoU)
    5. Data association
    6. Track management
    7. Track visualization
    8. Real video processing

No pretrained object detector is used.

OpenCV is used for:
    - Video reading and writing
    - Background subtraction
    - Morphological filtering
    - Contour extraction
    - Visualization

The Kalman filter and data association are implemented manually.

======================================================================
1. DETECTION VS TRACKING
======================================================================

Object detection:

    Frame 1 -> Bottle
    Frame 2 -> Bottle
    Frame 3 -> Bottle

Detection alone does not tell us whether these are the
same physical bottle.

Object tracking:

    Frame 1 -> Bottle ID 1
    Frame 2 -> Bottle ID 1
    Frame 3 -> Bottle ID 1

The tracker maintains object identity over time.

======================================================================
2. KALMAN FILTER
======================================================================

A Kalman filter estimates the hidden state of a moving object.

We use the state:

    x = [
        center_x,
        center_y,
        velocity_x,
        velocity_y,
        width,
        height
    ]

State dimension:

    6

Measurement:

    z = [
        center_x,
        center_y,
        width,
        height
    ]

Measurement dimension:

    4

The constant-velocity motion model is:

    x(k) = F x(k-1) + w(k)

For time interval dt:

    F = [
        1  0  dt  0   0  0
        0  1  0   dt  0  0
        0  0  1   0   0  0
        0  0  0   1   0  0
        0  0  0   0   1  0
        0  0  0   0   0  1
    ]

Measurement model:

    z = H x + v

H selects:

    center_x
    center_y
    width
    height

======================================================================
3. KALMAN PREDICTION
======================================================================

Predicted state:

    x_pred = F x

Predicted covariance:

    P_pred = F P F.T + Q

Q is process-noise covariance.

It represents uncertainty in the motion model.

======================================================================
4. KALMAN UPDATE
======================================================================

Innovation:

    y = z - H x_pred

Innovation covariance:

    S = H P_pred H.T + R

Kalman gain:

    K = P_pred H.T inverse(S)

Updated state:

    x = x_pred + K y

Updated covariance:

    P = (I - K H) P_pred

We use the Joseph covariance update for better numerical
stability:

    P = (I-KH) P_pred (I-KH).T + K R K.T

======================================================================
5. DATA ASSOCIATION
======================================================================

A tracker may have several existing tracks and several new
detections.

We need to decide which detection belongs to which track.

We calculate:

    IoU(track_box, detection_box)

IoU:

    intersection_area / union_area

We use a greedy matching algorithm:

    1. Calculate all track-detection IoUs.
    2. Sort candidate pairs by descending IoU.
    3. Accept the highest remaining pair.
    4. Reject pairs below the IoU threshold.
    5. Do not reuse a track or detection.

This is a simple association strategy.

Production trackers may use Hungarian assignment, appearance
embeddings, motion gating, and more advanced association.

======================================================================
6. TRACK LIFECYCLE
======================================================================

Each track maintains:

    track_id
    Kalman filter
    age
    hits
    time_since_update
    confirmed status

A track is:

    TENTATIVE:
        Recently created.

    CONFIRMED:
        Observed consistently.

    LOST:
        Not observed in the current frame, but retained temporarily.

    DELETED:
        Missing for too many frames.

======================================================================
7. REAL VIDEO DETECTIONS
======================================================================

We use OpenCV MOG2 background subtraction.

Pipeline:

    Real video frame
        |
        v
    Background subtraction
        |
        v
    Foreground mask
        |
        v
    Morphological filtering
        |
        v
    Contour extraction
        |
        v
    Bounding boxes
        |
        v
    Kalman tracker
        |
        v
    Persistent track IDs

IMPORTANT:

Background subtraction works best with a stationary camera.

It detects moving foreground regions, not semantic object classes.

A moving shadow or a person may also become a detection.

======================================================================
8. ROBOTICS APPLICATION
======================================================================

Conveyor-belt waste sorting:

    RGB Camera
        |
        v
    Object Detection / Segmentation
        |
        v
    Object Tracking
        |
        v
    Persistent Object ID
        |
        v
    Position and Velocity Estimation
        |
        v
    Conveyor Motion Prediction
        |
        v
    Robot Interception Planning
        |
        v
    ACT / Manipulation Policy

Example:

    Bottle ID 12:

        Frame 100: x = 120
        Frame 101: x = 128
        Frame 102: x = 136
        Frame 103: x = 144

    Estimated velocity:

        vx = 8 pixels/frame

    Predicted position after 10 frames:

        x = 224 pixels

This prediction helps the robot estimate where the bottle
will be when the gripper reaches the conveyor.

======================================================================
RUN
======================================================================

Install:

    pip install numpy opencv-python matplotlib

Run numerical tests:

    python lesson41_object_tracking_from_scratch.py --check-only

Process a real video:

    python lesson41_object_tracking_from_scratch.py \
        --video conveyor_video.mp4

Save annotated output:

    python lesson41_object_tracking_from_scratch.py \
        --video conveyor_video.mp4 \
        --output runs/lesson41/tracked_video.mp4

Tune background subtraction:

    python lesson41_object_tracking_from_scratch.py \
        --video conveyor_video.mp4 \
        --min-area 500 \
        --max-age 15 \
        --min-hits 3

Display foreground mask:

    python lesson41_object_tracking_from_scratch.py \
        --video conveyor_video.mp4 \
        --show-mask

======================================================================
OUTPUT
======================================================================

runs/lesson41/
    tracked_video.mp4
    tracking_summary.json

======================================================================
"""

import argparse
import json
import math
import time

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# =====================================================================
# CONFIGURATION
# =====================================================================

OUTPUT_DIR = Path("runs/lesson41")

DEFAULT_MIN_AREA = 300

DEFAULT_MAX_AGE = 15

DEFAULT_MIN_HITS = 3

DEFAULT_IOU_THRESHOLD = 0.15

DEFAULT_MAX_DETECTIONS = 100


# =====================================================================
# BOUNDING BOX UTILITIES
# =====================================================================

def center_to_bbox(cx, cy, width, height):

    x1 = cx - width / 2.0

    y1 = cy - height / 2.0

    x2 = cx + width / 2.0

    y2 = cy + height / 2.0

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float64
    )


def bbox_to_measurement(box):

    x1, y1, x2, y2 = map(float, box)

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
        [cx, cy, width, height],
        dtype=np.float64
    )


def measurement_to_bbox(measurement):

    cx, cy, width, height = measurement

    width = max(
        1.0,
        float(width)
    )

    height = max(
        1.0,
        float(height)
    )

    return center_to_bbox(
        float(cx),
        float(cy),
        width,
        height
    )


def clip_bbox(box, image_width, image_height):

    x1, y1, x2, y2 = map(float, box)

    x1 = np.clip(
        x1,
        0,
        image_width - 1
    )

    y1 = np.clip(
        y1,
        0,
        image_height - 1
    )

    x2 = np.clip(
        x2,
        0,
        image_width - 1
    )

    y2 = np.clip(
        y2,
        0,
        image_height - 1
    )

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float64
    )


# =====================================================================
# INTERSECTION OVER UNION
# =====================================================================

def bbox_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = map(float, box_a)

    bx1, by1, bx2, by2 = map(float, box_b)

    ix1 = max(ax1, bx1)

    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)

    iy2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        ix2 - ix1
    )

    intersection_height = max(
        0.0,
        iy2 - iy1
    )

    intersection = (
        intersection_width
        * intersection_height
    )

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

    union = area_a + area_b - intersection

    return intersection / max(
        union,
        1e-12
    )


# =====================================================================
# KALMAN FILTER
# =====================================================================

class KalmanFilter:

    """
    Constant-velocity Kalman filter.

    State:

        [cx, cy, vx, vy, width, height]

    Measurement:

        [cx, cy, width, height]

    Position and velocity are estimated independently
    along the horizontal and vertical image axes.
    """

    STATE_DIM = 6

    MEASUREMENT_DIM = 4

    def __init__(
        self,
        initial_measurement,
        dt=1.0
    ):

        z = np.asarray(
            initial_measurement,
            dtype=np.float64
        )

        if z.shape != (4,):

            raise ValueError(
                "Measurement must have shape (4,)."
            )

        self.x = np.array(
            [
                z[0],
                z[1],
                0.0,
                0.0,
                z[2],
                z[3]
            ],
            dtype=np.float64
        )

        self.P = np.diag(
            [
                10.0,
                10.0,
                100.0,
                100.0,
                10.0,
                10.0
            ]
        )

        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ],
            dtype=np.float64
        )

        self.R = np.diag(
            [
                25.0,
                25.0,
                16.0,
                16.0
            ]
        )

        self.Q_base = np.diag(
            [
                1.0,
                1.0,
                4.0,
                4.0,
                1.0,
                1.0
            ]
        )

        self.I = np.eye(
            self.STATE_DIM,
            dtype=np.float64
        )

        self.F = np.eye(
            self.STATE_DIM,
            dtype=np.float64
        )

        self.set_dt(dt)

    def set_dt(self, dt):

        dt = float(dt)

        if dt <= 0:

            raise ValueError(
                "dt must be positive."
            )

        self.F = np.array(
            [
                [1, 0, dt, 0, 0, 0],
                [0, 1, 0, dt, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ],
            dtype=np.float64
        )

    def predict(self, dt=1.0):

        self.set_dt(dt)

        self.x = self.F @ self.x

        self.P = (
            self.F
            @ self.P
            @ self.F.T
            + self.Q_base
        )

        # Prevent invalid negative object dimensions.

        self.x[4] = max(
            1.0,
            self.x[4]
        )

        self.x[5] = max(
            1.0,
            self.x[5]
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

        # Innovation.

        y = z - self.H @ self.x

        # Innovation covariance.

        S = (
            self.H
            @ self.P
            @ self.H.T
            + self.R
        )

        # Kalman gain.
        #
        # Solve S * X = H * P instead of explicitly
        # calculating inverse(S).

        K = np.linalg.solve(
            S,
            self.H @ self.P
        ).T

        # State correction.

        self.x = self.x + K @ y

        # Joseph covariance update.

        A = self.I - K @ self.H

        self.P = (
            A @ self.P @ A.T
            + K @ self.R @ K.T
        )

        # Numerical symmetry.

        self.P = (
            self.P + self.P.T
        ) / 2.0

        self.x[4] = max(
            1.0,
            self.x[4]
        )

        self.x[5] = max(
            1.0,
            self.x[5]
        )

        return self.x.copy()

    def get_bbox(self):

        return measurement_to_bbox(
            [
                self.x[0],
                self.x[1],
                self.x[4],
                self.x[5]
            ]
        )

    def get_velocity(self):

        return np.array(
            [
                self.x[2],
                self.x[3]
            ],
            dtype=np.float64
        )


# =====================================================================
# TRACK DATA STRUCTURE
# =====================================================================

@dataclass
class Track:

    track_id: int

    kalman: KalmanFilter

    age: int = 1

    hits: int = 1

    time_since_update: int = 0

    confirmed: bool = False

    def predict(self, dt=1.0):

        self.kalman.predict(dt)

        self.age += 1

        self.time_since_update += 1

    def update(self, detection):

        measurement = bbox_to_measurement(
            detection
        )

        self.kalman.update(
            measurement
        )

        self.hits += 1

        self.time_since_update = 0

    def get_bbox(self):

        return self.kalman.get_bbox()

    def get_velocity(self):

        return self.kalman.get_velocity()


# =====================================================================
# DATA ASSOCIATION
# =====================================================================

def greedy_iou_assignment(
    tracks,
    detections,
    iou_threshold=DEFAULT_IOU_THRESHOLD
):

    """
    Greedy association.

    Returns:

        matches:
            List of (track_index, detection_index)

        unmatched_tracks:
            List of track indices

        unmatched_detections:
            List of detection indices
    """

    if len(tracks) == 0:

        return (
            [],
            [],
            list(range(len(detections)))
        )

    if len(detections) == 0:

        return (
            [],
            list(range(len(tracks))),
            []
        )

    iou_matrix = np.zeros(
        (
            len(tracks),
            len(detections)
        ),
        dtype=np.float64
    )

    for track_index, track in enumerate(tracks):

        predicted_box = track.get_bbox()

        for detection_index, detection in enumerate(detections):

            iou_matrix[
                track_index,
                detection_index
            ] = bbox_iou(
                predicted_box,
                detection
            )

    candidate_pairs = []

    for track_index in range(len(tracks)):

        for detection_index in range(len(detections)):

            candidate_pairs.append(
                (
                    iou_matrix[
                        track_index,
                        detection_index
                    ],
                    track_index,
                    detection_index
                )
            )

    candidate_pairs.sort(
        key=lambda item: item[0],
        reverse=True
    )

    matched_tracks = set()

    matched_detections = set()

    matches = []

    for iou, track_index, detection_index in candidate_pairs:

        if iou < iou_threshold:

            break

        if track_index in matched_tracks:

            continue

        if detection_index in matched_detections:

            continue

        matches.append(
            (
                track_index,
                detection_index
            )
        )

        matched_tracks.add(
            track_index
        )

        matched_detections.add(
            detection_index
        )

    unmatched_tracks = [
        index
        for index in range(len(tracks))
        if index not in matched_tracks
    ]

    unmatched_detections = [
        index
        for index in range(len(detections))
        if index not in matched_detections
    ]

    return (
        matches,
        unmatched_tracks,
        unmatched_detections
    )


# =====================================================================
# MULTI-OBJECT TRACKER
# =====================================================================

class ObjectTracker:

    def __init__(
        self,
        max_age=DEFAULT_MAX_AGE,
        min_hits=DEFAULT_MIN_HITS,
        iou_threshold=DEFAULT_IOU_THRESHOLD
    ):

        self.max_age = max_age

        self.min_hits = min_hits

        self.iou_threshold = iou_threshold

        self.tracks = []

        self.next_id = 1

        self.frame_count = 0

        self.total_created = 0

        self.total_deleted = 0

    def create_track(self, detection):

        measurement = bbox_to_measurement(
            detection
        )

        kalman = KalmanFilter(
            measurement
        )

        track = Track(

            track_id=self.next_id,

            kalman=kalman,

            age=1,

            hits=1,

            time_since_update=0,

            confirmed=(self.min_hits <= 1)
        )

        self.next_id += 1

        self.total_created += 1

        self.tracks.append(
            track
        )

    def update(self, detections, dt=1.0):

        self.frame_count += 1

        # Step 1: predict all existing tracks.

        for track in self.tracks:

            track.predict(dt)

        # Step 2: associate predictions with detections.

        (
            matches,
            unmatched_tracks,
            unmatched_detections
        ) = greedy_iou_assignment(

            self.tracks,

            detections,

            self.iou_threshold
        )

        # Step 3: update matched tracks.

        for track_index, detection_index in matches:

            track = self.tracks[track_index]

            track.update(
                detections[detection_index]
            )

            if track.hits >= self.min_hits:

                track.confirmed = True

        # Step 4: create new tracks.

        for detection_index in unmatched_detections:

            self.create_track(
                detections[detection_index]
            )

        # Step 5: delete tracks that have been missing too long.

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

        active = []

        for track in self.tracks:

            if not track.confirmed:

                continue

            active.append(
                {
                    "id": track.track_id,

                    "bbox": track.get_bbox(),

                    "velocity": track.get_velocity(),

                    "age": track.age,

                    "hits": track.hits,

                    "time_since_update": track.time_since_update,

                    "confirmed": track.confirmed,
                }
            )

        return active

    def get_summary(self):

        return {
            "frames_processed": self.frame_count,
            "tracks_created": self.total_created,
            "tracks_deleted": self.total_deleted,
            "tracks_remaining": len(self.tracks),
            "next_track_id": self.next_id,
        }


# =====================================================================
# REAL VIDEO DETECTOR
# =====================================================================

class BackgroundMotionDetector:

    """
    Motion detector for a stationary camera.

    MOG2 estimates the background model.

    Foreground regions are converted into bounding boxes.

    This is not a semantic object detector.
    """

    def __init__(
        self,
        min_area=DEFAULT_MIN_AREA,
        history=500,
        var_threshold=32,
        detect_shadows=False
    ):

        self.min_area = min_area

        self.background_subtractor = (
            cv2.createBackgroundSubtractorMOG2(

                history=history,

                varThreshold=var_threshold,

                detectShadows=detect_shadows
            )
        )

        self.kernel = cv2.getStructuringElement(

            cv2.MORPH_ELLIPSE,

            (5, 5)
        )

    def detect(self, frame):

        foreground = self.background_subtractor.apply(
            frame
        )

        # Remove isolated noise.

        foreground = cv2.morphologyEx(

            foreground,

            cv2.MORPH_OPEN,

            self.kernel,

            iterations=1
        )

        # Fill small holes.

        foreground = cv2.morphologyEx(

            foreground,

            cv2.MORPH_CLOSE,

            self.kernel,

            iterations=2
        )

        contours, _ = cv2.findContours(

            foreground,

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

            x, y, width, height = cv2.boundingRect(
                contour
            )

            if width < 5 or height < 5:

                continue

            detections.append(
                np.array(
                    [
                        x,
                        y,
                        x + width,
                        y + height
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

        detections = detections[
            :DEFAULT_MAX_DETECTIONS
        ]

        return detections, foreground


# =====================================================================
# VISUALIZATION
# =====================================================================

def draw_tracking_results(
    frame,
    detections,
    active_tracks,
    frame_number,
    fps
):

    output = frame.copy()

    # Draw raw detections.

    for box in detections:

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

    # Draw tracked objects.

    for track in active_tracks:

        x1, y1, x2, y2 = map(
            int,
            track["bbox"]
        )

        track_id = track["id"]

        vx, vy = track["velocity"]

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

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )

        cv2.circle(

            output,

            (center_x, center_y),

            4,

            color,

            -1
        )

        label = (
            f"ID {track_id} "
            f"vx={vx:.1f} vy={vy:.1f}"
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

        f"FPS: {fps:.1f}",

        (15, 60),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 255),

        2
    )

    cv2.putText(

        output,

        f"Active tracks: {len(active_tracks)}",

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

    tracker = ObjectTracker(

        max_age=args.max_age,

        min_hits=args.min_hits,

        iou_threshold=args.iou_threshold
    )

    frame_number = 0

    processing_times = []

    velocity_history = {}

    print("\nStarting video processing.")

    print("Input:", video_path)

    print("Resolution:", width, "x", height)

    print("Input FPS:", fps)

    print("Total frames:", total_frames)

    print("\nPress Q to stop the video window.")

    try:

        while True:

            start = time.perf_counter()

            ret, frame = capture.read()

            if not ret:

                break

            frame_number += 1

            # Motion detection.

            detections, foreground = detector.detect(
                frame
            )

            # Track update.

            active_tracks = tracker.update(
                detections,
                dt=1.0
            )

            # Store velocity estimates.

            for track in active_tracks:

                velocity_history.setdefault(
                    str(track["id"]),
                    []
                ).append(
                    {
                        "frame": frame_number,
                        "vx": float(track["velocity"][0]),
                        "vy": float(track["velocity"][1]),
                    }
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

            output_frame = draw_tracking_results(

                frame,

                detections,

                active_tracks,

                frame_number,

                processing_fps
            )

            writer.write(
                output_frame
            )

            if args.display:

                cv2.imshow(
                    "Lesson 41 - Object Tracking",
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

                    f"Processing FPS: {processing_fps:.1f}"
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

        "input_fps": fps,

        "frames_processed": frame_number,

        "average_processing_fps": (
            1.0 / average_time
            if average_time > 0 else 0.0
        ),

        "tracker": tracker.get_summary(),

        "velocity_history": velocity_history,
    }

    summary_path = (
        output_path.parent
        / "tracking_summary.json"
    )

    with open(summary_path, "w") as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print("\nProcessing completed.")

    print(
        json.dumps(
            tracker.get_summary(),
            indent=4
        )
    )

    print("\nAverage processing FPS:")

    print(
        summary["average_processing_fps"]
    )

    print("\nOutput video:", output_path)

    print("Summary:", summary_path)


# =====================================================================
# NUMERICAL TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print("LESSON 41 — NUMERICAL TESTS")

    print("=" * 70)

    # ---------------------------------------------------------------
    # Test 1: bounding box conversion
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
    # Test 2: identical boxes have IoU 1
    # ---------------------------------------------------------------

    assert abs(
        bbox_iou(box, box) - 1.0
    ) < 1e-8

    print("Test 2 passed: identical-box IoU")

    # ---------------------------------------------------------------
    # Test 3: non-overlapping boxes have IoU 0
    # ---------------------------------------------------------------

    other_box = np.array(
        [300, 300, 400, 400],
        dtype=np.float64
    )

    assert bbox_iou(
        box,
        other_box
    ) == 0.0

    print("Test 3 passed: non-overlapping IoU")

    # ---------------------------------------------------------------
    # Test 4: partial overlap
    # ---------------------------------------------------------------

    a = [0, 0, 100, 100]

    b = [50, 50, 150, 150]

    expected_iou = 2500 / 17500

    assert abs(
        bbox_iou(a, b) - expected_iou
    ) < 1e-8

    print("Test 4 passed: partial-overlap IoU")

    # ---------------------------------------------------------------
    # Test 5: Kalman filter prediction
    # ---------------------------------------------------------------

    initial = np.array(
        [100, 100, 40, 30],
        dtype=np.float64
    )

    kf = KalmanFilter(
        initial
    )

    kf.x[2] = 5.0

    kf.x[3] = 2.0

    predicted = kf.predict(
        dt=1.0
    )

    assert abs(
        predicted[0] - 105.0
    ) < 1e-8

    assert abs(
        predicted[1] - 102.0
    ) < 1e-8

    print("Test 5 passed: constant-velocity prediction")

    # ---------------------------------------------------------------
    # Test 6: Kalman update
    # ---------------------------------------------------------------

    measurement = np.array(
        [108, 104, 40, 30],
        dtype=np.float64
    )

    updated = kf.update(
        measurement
    )

    assert np.isfinite(
        updated
    ).all()

    assert abs(
        updated[0] - predicted[0]
    ) > 0

    print("Test 6 passed: Kalman measurement update")

    # ---------------------------------------------------------------
    # Test 7: covariance symmetry
    # ---------------------------------------------------------------

    assert np.allclose(
        kf.P,
        kf.P.T,
        atol=1e-8
    )

    print("Test 7 passed: covariance symmetry")

    # ---------------------------------------------------------------
    # Test 8: covariance positive semidefinite
    # ---------------------------------------------------------------

    eigenvalues = np.linalg.eigvalsh(
        kf.P
    )

    assert np.min(
        eigenvalues
    ) >= -1e-8

    print("Test 8 passed: covariance positive semidefinite")

    # ---------------------------------------------------------------
    # Test 9: data association
    # ---------------------------------------------------------------

    track_a = Track(
        track_id=1,
        kalman=KalmanFilter(
            bbox_to_measurement(
                [0, 0, 100, 100]
            )
        )
    )

    track_b = Track(
        track_id=2,
        kalman=KalmanFilter(
            bbox_to_measurement(
                [200, 200, 300, 300]
            )
        )
    )

    detections = [

        np.array(
            [5, 5, 105, 105],
            dtype=np.float64
        ),

        np.array(
            [205, 205, 305, 305],
            dtype=np.float64
        )
    ]

    matches, unmatched_tracks, unmatched_detections = (
        greedy_iou_assignment(
            [track_a, track_b],
            detections,
            iou_threshold=0.5
        )
    )

    assert len(matches) == 2

    assert len(unmatched_tracks) == 0

    assert len(unmatched_detections) == 0

    print("Test 9 passed: two-object data association")

    # ---------------------------------------------------------------
    # Test 10: unmatched detection
    # ---------------------------------------------------------------

    matches, unmatched_tracks, unmatched_detections = (
        greedy_iou_assignment(
            [track_a],
            [
                np.array(
                    [500, 500, 600, 600],
                    dtype=np.float64
                )
            ],
            iou_threshold=0.5
        )
    )

    assert len(matches) == 0

    assert len(unmatched_tracks) == 1

    assert len(unmatched_detections) == 1

    print("Test 10 passed: unmatched detection handling")

    # ---------------------------------------------------------------
    # Test 11: track creation
    # ---------------------------------------------------------------

    tracker = ObjectTracker(
        max_age=3,
        min_hits=1
    )

    tracker.update(
        [
            np.array(
                [10, 10, 50, 50],
                dtype=np.float64
            )
        ]
    )

    assert len(tracker.tracks) == 1

    assert tracker.tracks[0].track_id == 1

    print("Test 11 passed: track creation")

    # ---------------------------------------------------------------
    # Test 12: track ID persistence
    # ---------------------------------------------------------------

    first_id = tracker.tracks[0].track_id

    tracker.update(
        [
            np.array(
                [12, 10, 52, 50],
                dtype=np.float64
            )
        ]
    )

    assert len(tracker.tracks) == 1

    assert tracker.tracks[0].track_id == first_id

    assert tracker.tracks[0].hits == 2

    print("Test 12 passed: persistent track identity")

    # ---------------------------------------------------------------
    # Test 13: temporary missing detection
    # ---------------------------------------------------------------

    tracker.update([])

    assert len(tracker.tracks) == 1

    assert tracker.tracks[0].time_since_update == 1

    print("Test 13 passed: temporary missed detection")

    # ---------------------------------------------------------------
    # Test 14: track deletion
    # ---------------------------------------------------------------

    tracker.update([])

    tracker.update([])

    tracker.update([])

    assert len(tracker.tracks) == 0

    assert tracker.total_deleted == 1

    print("Test 14 passed: stale-track deletion")

    # ---------------------------------------------------------------
    # Test 15: velocity estimation
    # ---------------------------------------------------------------

    moving_tracker = ObjectTracker(
        max_age=5,
        min_hits=1,
        iou_threshold=0.1
    )

    positions = [
        [0, 0, 40, 40],
        [5, 0, 45, 40],
        [10, 0, 50, 40],
        [15, 0, 55, 40],
        [20, 0, 60, 40],
    ]

    for position in positions:

        moving_tracker.update(
            [
                np.array(
                    position,
                    dtype=np.float64
                )
            ]
        )

    velocity = moving_tracker.tracks[0].get_velocity()

    assert velocity[0] > 0

    print(
        "Test 15 passed: positive horizontal velocity",
        f"(vx={velocity[0]:.3f})"
    )

    print("\nAll 15 tests passed.\n")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description="Lesson 41: Object Tracking From Scratch"
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            OUTPUT_DIR / "tracked_video.mp4"
        )
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA
    )

    parser.add_argument(
        "--max-age",
        type=int,
        default=DEFAULT_MAX_AGE
    )

    parser.add_argument(
        "--min-hits",
        type=int,
        default=DEFAULT_MIN_HITS
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

    if args.min_hits < 1:

        parser.error("--min-hits must be at least 1")

    if not 0.0 <= args.iou_threshold <= 1.0:

        parser.error("--iou-threshold must be between 0 and 1")

    print("=" * 70)

    print("LESSON 41 — OBJECT TRACKING FROM SCRATCH")

    print("=" * 70)

    run_tests()

    if args.check_only:

        print("Check-only mode completed.")

        return

    if args.video is None:

        print(
            "\nNo video supplied."
        )

        print(
            "Provide a real video using --video."
        )

        print(
            "\nExample:"
        )

        print(
            "python lesson41_object_tracking_from_scratch.py "
            "--video conveyor_video.mp4"
        )

        return

    args.display = not args.no_display

    process_video(args)


if __name__ == "__main__":

    main()