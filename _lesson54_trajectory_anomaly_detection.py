"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 54 — TRAJECTORY ANOMALY DETECTION AND UNUSUAL MOTION RECOGNITION
======================================================================

Filename:
    lesson54_trajectory_anomaly_detection.py

ROADMAP
-------
This lesson extends trajectory prediction and collision analysis.

Lesson 53:
    Predict where objects will move and whether they may collide.

Lesson 54:
    Learn what normal movement looks like and detect unusual motion.

Learning objectives:
    1. Represent object movement as a time-ordered trajectory.
    2. Extract speed, acceleration, turning angle, and path deviation.
    3. Learn a robust normal-motion profile using median and MAD.
    4. Calculate anomaly scores using robust Z-scores.
    5. Detect sudden stops, acceleration changes, reversals, and
       irregular movement.
    6. Track moving regions in real video using OpenCV.
    7. Visualize anomalous trajectories and save event summaries.

THEORY
------

1. TRAJECTORY

An object's image-space trajectory is:

    P(t) = [x(t), y(t)]

For discrete video frames:

    P_0, P_1, ..., P_t

The displacement between consecutive frames is:

    ΔP_t = P_t - P_(t-1)

2. SPEED

For frame interval Δt:

    v_t = ||P_t - P_(t-1)|| / Δt

When Δt = 1 frame:

    v_t = ||P_t - P_(t-1)||

The units are pixels per frame.

3. ACCELERATION

Velocity:

    V_t = P_t - P_(t-1)

Acceleration:

    A_t = V_t - V_(t-1)

Acceleration magnitude:

    a_t = ||A_t||

The units are pixels per frame squared.

4. TURNING ANGLE

Given consecutive velocity vectors:

    V_(t-1), V_t

The angle is:

    θ = arccos(
        dot(V_(t-1), V_t) /
        (||V_(t-1)|| ||V_t||)
    )

Interpretation:

    0 degrees:
        Straight movement.

    90 degrees:
        A right-angle turn.

    180 degrees:
        Direction reversal.

5. LATERAL PATH DEVIATION

A normally moving conveyor object may approximately follow a
straight path.

Fit a line to recent trajectory points using PCA.

Calculate the perpendicular distance of each point from the line.

Normalize the RMS distance by trajectory length:

    deviation = RMS(perpendicular_distance) / path_length

This produces a dimensionless path-irregularity feature.

6. ROBUST NORMAL-MOTION MODEL

We assume that a normal-motion calibration segment is available.

For every feature, estimate:

    median = median(x)

    MAD = median(|x - median(x)|)

The robust standard deviation estimate is:

    robust_sigma = 1.4826 * MAD

Robust Z-score:

    z = |x - median| / robust_sigma

Why median and MAD?

    - Resistant to occasional extreme observations.
    - Less sensitive to outliers than mean and standard deviation.
    - Useful when motion measurements contain tracking noise.

We use a minimum scale for each feature to avoid division by zero.

7. ANOMALY SCORE

For feature vector:

    f = [speed, acceleration, turning_angle, path_deviation]

Calculate:

    z = [z_speed, z_acceleration, z_turn, z_deviation]

The anomaly score is:

    score = max(z)

An observation is anomalous when:

    score > threshold

This is an educational detector, not a calibrated probability.

8. IMPORTANT LIMITATION

The model assumes that the calibration segment represents normal
motion. If unusual motion appears during calibration, the learned
normal profile can become contaminated.

For production:
    - Calibrate with verified normal data.
    - Use separate profiles for different object classes.
    - Consider lighting and camera-motion changes.
    - Evaluate false positives and false negatives.
    - Use camera calibration when physical speed is required.

REAL-WORLD ROBOTICS EXAMPLE
---------------------------

OpenArm V2 waste-sorting conveyor:

    Camera
       |
       v
    Motion detection
       |
       v
    Object tracking
       |
       v
    Trajectory feature extraction
       |
       v
    Normal-motion model
       |
       +---- Normal movement
       |
       +---- Sudden stop
       |
       +---- Unexpected reversal
       |
       +---- Irregular trajectory
       |
       v
    Robot task manager

Possible applications:

    - Detect a bottle that stops unexpectedly.
    - Detect waste that falls off a conveyor.
    - Detect an object moving against conveyor direction.
    - Detect irregular movement caused by mechanical interference.
    - Trigger a robot safety response.

IMPORTANT:
    The video detector in this lesson uses MOG2 background subtraction.
    It detects moving image regions, not semantic waste categories.

    In a real waste-sorting system, replace the MOG2 detector with
    your YOLO or instance-segmentation detector. The trajectory
    anomaly detector can remain largely unchanged.

DEPENDENCIES
------------
    pip install numpy opencv-python scipy matplotlib

EXAMPLES
--------
    python lesson54_trajectory_anomaly_detection.py --check-only

    python lesson54_trajectory_anomaly_detection.py \
        --video conveyor_video.mp4

    python lesson54_trajectory_anomaly_detection.py \
        --video conveyor_video.mp4 \
        --warmup-frames 150 \
        --threshold 3.5

    python lesson54_trajectory_anomaly_detection.py \
        --video conveyor_video.mp4 \
        --no-display

OUTPUT
------
    runs/lesson54/
        anomaly_feature_demo.png
        anomaly_score_demo.png
        trajectory_anomaly.mp4
        anomaly_summary.json

NEXT LESSON
-----------
    Lesson 55 — Tracking Evaluation and Metrics
======================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import linear_sum_assignment


# ======================================================================
# CONFIGURATION
# ======================================================================

LESSON_NAME = "Lesson 54 - Trajectory Anomaly Detection"

FEATURE_NAMES = [
    "speed",
    "acceleration",
    "turning_angle",
    "path_deviation",
]

# Minimum robust scales prevent division by zero when a feature has
# almost no variation during calibration.
FEATURE_SCALE_FLOORS = np.array(
    [
        0.50,   # pixels/frame
        0.10,   # pixels/frame^2
        5.00,   # degrees
        0.01,   # normalized path deviation
    ],
    dtype=np.float64,
)

DEFAULT_OUTPUT_DIR = os.path.join("runs", "lesson54")


# ======================================================================
# TRAJECTORY
# ======================================================================

@dataclass
class Trajectory:
    """
    Store a time-ordered sequence of 2-D positions.

    Each point is represented as:

        [x, y]

    The deque limits memory usage for long videos.
    """

    max_length: int = 60

    points: deque = field(init=False)

    def __post_init__(self):
        self.points = deque(maxlen=self.max_length)

    def add(self, point):
        point = np.asarray(point, dtype=np.float64)

        if point.shape != (2,):
            raise ValueError("A trajectory point must have shape (2,).")

        self.points.append(point)

    def array(self):
        if not self.points:
            return np.empty((0, 2), dtype=np.float64)

        return np.asarray(self.points, dtype=np.float64)

    def __len__(self):
        return len(self.points)

    def latest(self):
        if not self.points:
            return None

        return self.points[-1]


# ======================================================================
# FEATURE EXTRACTION
# ======================================================================

class MotionFeatureExtractor:
    """
    Extract motion features from a trajectory.

    Feature order:

        0: speed
        1: acceleration magnitude
        2: turning angle in degrees
        3: normalized lateral path deviation
    """

    def __init__(self, minimum_points=5):
        self.minimum_points = int(minimum_points)

    @staticmethod
    def _safe_angle_degrees(v1, v2):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0

        cosine = np.dot(v1, v2) / (n1 * n2)
        cosine = np.clip(cosine, -1.0, 1.0)

        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def lateral_path_deviation(points):
        """
        Estimate trajectory deviation from its principal direction.

        PCA gives the dominant direction of the trajectory.

        The perpendicular distance of each point from the principal
        line is measured, then normalized by total path length.
        """

        points = np.asarray(points, dtype=np.float64)

        if len(points) < 3:
            return 0.0

        centered = points - np.mean(points, axis=0)

        covariance = np.cov(centered.T)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        direction = eigenvectors[:, np.argmax(eigenvalues)]

        # Perpendicular direction.
        perpendicular = np.array(
            [-direction[1], direction[0]],
            dtype=np.float64,
        )

        distances = centered @ perpendicular

        rms_distance = np.sqrt(np.mean(distances ** 2))

        segment_lengths = np.linalg.norm(
            np.diff(points, axis=0),
            axis=1,
        )

        path_length = float(np.sum(segment_lengths))

        if path_length < 1e-8:
            return 0.0

        return float(rms_distance / path_length)

    def extract(self, trajectory):
        """
        Return:

            feature_vector, or None if insufficient history.
        """

        points = (
            trajectory.array()
            if isinstance(trajectory, Trajectory)
            else np.asarray(trajectory, dtype=np.float64)
        )

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Trajectory must have shape (N, 2).")

        if len(points) < self.minimum_points:
            return None

        velocities = np.diff(points, axis=0)

        speed_values = np.linalg.norm(
            velocities,
            axis=1,
        )

        # Use the recent velocity rather than averaging the entire
        # trajectory. This makes sudden stops and reversals visible.
        speed = float(speed_values[-1])

        acceleration = 0.0

        if len(velocities) >= 2:
            acceleration_vector = (
                velocities[-1] - velocities[-2]
            )

            acceleration = float(
                np.linalg.norm(acceleration_vector)
            )

        turning_angle = 0.0

        if len(velocities) >= 2:
            turning_angle = self._safe_angle_degrees(
                velocities[-2],
                velocities[-1],
            )

        # Limit the PCA window to recent points.
        recent_points = points[-20:]

        path_deviation = self.lateral_path_deviation(
            recent_points
        )

        features = np.array(
            [
                speed,
                acceleration,
                turning_angle,
                path_deviation,
            ],
            dtype=np.float64,
        )

        return features


# ======================================================================
# ROBUST NORMAL-MOTION MODEL
# ======================================================================

class RobustNormalMotionModel:
    """
    Learn a normal-motion profile using median and MAD.

    No neural network is required.

    The model learns from feature vectors collected during a normal
    calibration period.
    """

    def __init__(
        self,
        threshold=3.5,
        minimum_samples=10,
    ):
        self.threshold = float(threshold)
        self.minimum_samples = int(minimum_samples)

        self.median = None
        self.mad = None
        self.robust_scale = None

        self.sample_count = 0
        self.fitted = False

    def fit(self, samples):
        samples = np.asarray(samples, dtype=np.float64)

        if samples.ndim != 2 or samples.shape[1] != 4:
            raise ValueError(
                "Expected samples with shape (N, 4)."
            )

        if len(samples) < self.minimum_samples:
            raise ValueError(
                f"Need at least {self.minimum_samples} samples; "
                f"received {len(samples)}."
            )

        if not np.all(np.isfinite(samples)):
            raise ValueError("Calibration data contains NaN or Inf.")

        self.median = np.median(samples, axis=0)

        self.mad = np.median(
            np.abs(samples - self.median),
            axis=0,
        )

        self.robust_scale = np.maximum(
            1.4826 * self.mad,
            FEATURE_SCALE_FLOORS,
        )

        self.sample_count = len(samples)
        self.fitted = True

        return self

    def score(self, features):
        if not self.fitted:
            raise RuntimeError(
                "The normal-motion model has not been fitted."
            )

        features = np.asarray(features, dtype=np.float64)

        if features.shape != (4,):
            raise ValueError("Expected a feature vector of shape (4,).")

        z_scores = (
            np.abs(features - self.median)
            / self.robust_scale
        )

        anomaly_score = float(np.max(z_scores))

        anomalous = anomaly_score > self.threshold

        dominant_feature_index = int(np.argmax(z_scores))

        return {
            "score": anomaly_score,
            "z_scores": z_scores,
            "anomalous": bool(anomalous),
            "dominant_feature": FEATURE_NAMES[
                dominant_feature_index
            ],
        }

    def to_dict(self):
        if not self.fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "sample_count": int(self.sample_count),
            "threshold": self.threshold,
            "median": self.median.tolist(),
            "mad": self.mad.tolist(),
            "robust_scale": self.robust_scale.tolist(),
            "feature_names": FEATURE_NAMES,
        }


# ======================================================================
# OBJECT TRACK
# ======================================================================

@dataclass
class ObjectTrack:
    track_id: int
    trajectory: Trajectory = field(
        default_factory=lambda: Trajectory(max_length=60)
    )

    missed_frames: int = 0
    age: int = 0

    anomaly_active: bool = False
    last_anomaly_score: float = 0.0
    last_dominant_feature: str = "none"

    def update(self, center):
        self.trajectory.add(center)
        self.missed_frames = 0
        self.age += 1

    def mark_missed(self):
        self.missed_frames += 1


# ======================================================================
# CENTROID TRACKER
# ======================================================================

class CentroidTracker:
    """
    A compact centroid-based tracker.

    This is intentionally independent of the anomaly detector.

    In production, replace this component with SORT, ByteTrack,
    Deep SORT, or detections from your YOLO pipeline.
    """

    def __init__(
        self,
        max_distance=80.0,
        max_missed=12,
        max_history=60,
    ):
        self.max_distance = float(max_distance)
        self.max_missed = int(max_missed)
        self.max_history = int(max_history)

        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def detection_center(box):
        x1, y1, x2, y2 = box

        return np.array(
            [
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
            ],
            dtype=np.float64,
        )

    def update(self, boxes):
        centers = [
            self.detection_center(box)
            for box in boxes
        ]

        track_ids = list(self.tracks.keys())

        if track_ids and centers:
            cost = np.zeros(
                (len(track_ids), len(centers)),
                dtype=np.float64,
            )

            for i, track_id in enumerate(track_ids):
                previous = self.tracks[
                    track_id
                ].trajectory.latest()

                for j, center in enumerate(centers):
                    cost[i, j] = np.linalg.norm(
                        previous - center
                    )

            rows, cols = linear_sum_assignment(cost)

            matched_tracks = set()
            matched_detections = set()

            for row, col in zip(rows, cols):
                if cost[row, col] > self.max_distance:
                    continue

                track_id = track_ids[row]

                self.tracks[track_id].update(
                    centers[col]
                )

                matched_tracks.add(track_id)
                matched_detections.add(col)

            for track_id in track_ids:
                if track_id not in matched_tracks:
                    self.tracks[track_id].mark_missed()

            for detection_index, center in enumerate(centers):
                if detection_index not in matched_detections:
                    self._create_track(center)

        elif track_ids:
            for track_id in track_ids:
                self.tracks[track_id].mark_missed()

        else:
            for center in centers:
                self._create_track(center)

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed_frames > self.max_missed
        ]

        for track_id in expired:
            del self.tracks[track_id]

        return self.tracks

    def _create_track(self, center):
        track = ObjectTrack(
            track_id=self.next_id,
            trajectory=Trajectory(
                max_length=self.max_history
            ),
        )

        track.update(center)

        self.tracks[self.next_id] = track

        self.next_id += 1


# ======================================================================
# VIDEO MOTION DETECTOR
# ======================================================================

class MotionDetector:
    """
    Detect moving regions using OpenCV MOG2.

    This is not semantic object detection.

    It is useful for demonstrating the tracking and anomaly pipeline
    on a mostly stationary-camera video.
    """

    def __init__(
        self,
        min_area=500,
        history=500,
        variance_threshold=25,
    ):
        self.min_area = float(min_area)

        self.background_subtractor = (
            cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=variance_threshold,
                detectShadows=False,
            )
        )

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

    def detect(self, frame):
        mask = self.background_subtractor.apply(frame)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.kernel,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            boxes.append(
                (x, y, x + w, y + h)
            )

        return boxes, mask


# ======================================================================
# SYNTHETIC MATHEMATICAL DEMONSTRATION
# ======================================================================

def generate_normal_trajectory_samples(
    number_of_trajectories=30,
    points_per_trajectory=30,
    seed=42,
):
    """
    Generate mathematical trajectories, not synthetic photographs.

    These are used only to demonstrate and test the feature model.
    """

    rng = np.random.default_rng(seed)

    samples = []

    for _ in range(number_of_trajectories):
        speed = rng.normal(2.5, 0.15)

        heading = rng.normal(0.0, 0.025)

        points = []

        position = np.array(
            [
                rng.uniform(0, 100),
                rng.uniform(0, 100),
            ],
            dtype=np.float64,
        )

        for _ in range(points_per_trajectory):
            heading += rng.normal(0.0, 0.002)

            direction = np.array(
                [
                    np.cos(heading),
                    np.sin(heading),
                ]
            )

            position = (
                position
                + speed * direction
                + rng.normal(0.0, 0.025, size=2)
            )

            points.append(position.copy())

            if len(points) >= 5:
                features = MotionFeatureExtractor().extract(
                    np.asarray(points)
                )

                if features is not None:
                    samples.append(features)

    return np.asarray(samples, dtype=np.float64)


def create_anomaly_demo(output_dir):
    """
    Train on normal mathematical motion and visualize an abrupt
    reversal and irregular movement.
    """

    os.makedirs(output_dir, exist_ok=True)

    normal_samples = generate_normal_trajectory_samples()

    model = RobustNormalMotionModel(
        threshold=3.5
    ).fit(normal_samples)

    extractor = MotionFeatureExtractor()

    # Normal straight trajectory.
    normal_points = np.column_stack(
        [
            np.linspace(0, 100, 30),
            np.linspace(20, 20, 30),
        ]
    )

    # Unusual trajectory: object suddenly reverses.
    reverse_points = np.column_stack(
        [
            np.concatenate(
                [
                    np.linspace(0, 60, 20),
                    np.linspace(60, 20, 10),
                ]
            ),
            np.full(30, 50.0),
        ]
    )

    # Unusual trajectory: irregular zig-zag.
    zigzag_x = np.linspace(0, 100, 30)

    zigzag_y = (
        80
        + 8 * np.sin(np.linspace(0, 8 * np.pi, 30))
    )

    zigzag_points = np.column_stack(
        [zigzag_x, zigzag_y]
    )

    trajectories = {
        "Normal": normal_points,
        "Direction reversal": reverse_points,
        "Irregular movement": zigzag_points,
    }

    results = {}

    for name, points in trajectories.items():
        features = extractor.extract(points)

        results[name] = model.score(features)

    # --------------------------------------------------------------
    # Trajectory feature visualization
    # --------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    for ax, (name, points) in zip(
        axes,
        trajectories.items(),
    ):
        ax.plot(
            points[:, 0],
            points[:, 1],
            marker="o",
            markersize=3,
        )

        ax.scatter(
            points[0, 0],
            points[0, 1],
            marker="s",
            s=70,
            label="Start",
        )

        ax.scatter(
            points[-1, 0],
            points[-1, 1],
            marker="*",
            s=120,
            label="End",
        )

        result = results[name]

        ax.set_title(
            f"{name}\n"
            f"Score: {result['score']:.2f} | "
            f"Anomaly: {result['anomalous']}"
        )

        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        ax.grid(True)
        ax.legend()

    fig.suptitle(
        "Lesson 54: Normal and Unusual Trajectories"
    )

    fig.tight_layout()

    feature_path = os.path.join(
        output_dir,
        "anomaly_feature_demo.png",
    )

    fig.savefig(
        feature_path,
        dpi=150,
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # Feature anomaly scores
    # --------------------------------------------------------------

    names = list(results.keys())

    scores = [
        results[name]["z_scores"]
        for name in names
    ]

    scores = np.asarray(scores)

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    x = np.arange(len(FEATURE_NAMES))

    width = 0.23

    for i, name in enumerate(names):
        ax.bar(
            x + (i - 1) * width,
            scores[i],
            width,
            label=name,
        )

    ax.axhline(
        model.threshold,
        linestyle="--",
        label="Anomaly threshold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Speed",
            "Acceleration",
            "Turning angle",
            "Path deviation",
        ]
    )

    ax.set_ylabel("Robust Z-score")
    ax.set_title("Motion Feature Anomaly Scores")
    ax.grid(axis="y")
    ax.legend()

    fig.tight_layout()

    score_path = os.path.join(
        output_dir,
        "anomaly_score_demo.png",
    )

    fig.savefig(
        score_path,
        dpi=150,
    )

    plt.close(fig)

    print("\nMathematical demonstration:")
    print("-" * 65)

    for name, result in results.items():
        print(
            f"{name:25s} "
            f"score={result['score']:8.3f} "
            f"anomalous={result['anomalous']} "
            f"feature={result['dominant_feature']}"
        )

    print(f"\nSaved: {feature_path}")
    print(f"Saved: {score_path}")

    return model


# ======================================================================
# VIDEO PROCESSING
# ======================================================================

def process_video(args):
    os.makedirs(args.output_dir, exist_ok=True)

    capture = cv2.VideoCapture(args.video)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.video}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    output_path = os.path.join(
        args.output_dir,
        "trajectory_anomaly.mp4",
    )

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()

        raise RuntimeError(
            "Could not create output video. "
            "Check your OpenCV video codec installation."
        )

    detector = MotionDetector(
        min_area=args.min_area
    )

    tracker = CentroidTracker(
        max_distance=args.max_distance,
        max_missed=args.max_missed,
    )

    extractor = MotionFeatureExtractor()

    model = RobustNormalMotionModel(
        threshold=args.threshold
    )

    calibration_samples = []

    frame_index = 0
    calibration_complete = False

    anomaly_events = []
    active_anomalies = {}

    while True:
        ok, frame = capture.read()

        if not ok:
            break

        frame_index += 1

        boxes, foreground_mask = detector.detect(
            frame
        )

        tracks = tracker.update(boxes)

        # ----------------------------------------------------------
        # Calibration phase
        # ----------------------------------------------------------

        if (
            not calibration_complete
            and frame_index <= args.warmup_frames
        ):
            for track in tracks.values():
                features = extractor.extract(
                    track.trajectory
                )

                if features is not None:
                    calibration_samples.append(
                        features
                    )

        if (
            not calibration_complete
            and frame_index >= args.warmup_frames
        ):
            if len(calibration_samples) >= 10:
                model.fit(
                    np.asarray(calibration_samples)
                )

                calibration_complete = True

                print(
                    "\nNormal-motion calibration complete."
                )

                print(
                    f"Calibration samples: "
                    f"{len(calibration_samples)}"
                )

            else:
                print(
                    "\nWarning: insufficient calibration "
                    "samples. Collecting more frames."
                )

                # If the video is short, keep collecting rather
                # than fitting an unreliable model.

        # If calibration is still incomplete, continue gathering
        # samples. This avoids silently fitting a poor baseline.
        if not calibration_complete:
            for track in tracks.values():
                features = extractor.extract(
                    track.trajectory
                )

                if features is not None:
                    calibration_samples.append(
                        features
                    )

        # ----------------------------------------------------------
        # Draw detections and analyze trajectories
        # ----------------------------------------------------------

        annotated = frame.copy()

        for box in boxes:
            x1, y1, x2, y2 = box

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (180, 180, 180),
                1,
            )

        for track_id, track in tracks.items():
            points = track.trajectory.array()

            if len(points) == 0:
                continue

            center = tuple(
                np.round(points[-1]).astype(int)
            )

            features = extractor.extract(
                track.trajectory
            )

            anomalous = False
            score = 0.0
            dominant_feature = "warming up"

            if (
                calibration_complete
                and features is not None
            ):
                result = model.score(features)

                anomalous = result["anomalous"]
                score = result["score"]

                dominant_feature = (
                    result["dominant_feature"]
                )

                track.last_anomaly_score = score
                track.last_dominant_feature = (
                    dominant_feature
                )

                # Record only anomaly onset, not every anomalous
                # frame. This makes event counts meaningful.
                if anomalous and not track.anomaly_active:
                    event = {
                        "track_id": int(track_id),
                        "frame": int(frame_index),
                        "time_seconds": float(
                            frame_index / fps
                        ),
                        "score": float(score),
                        "dominant_feature": (
                            dominant_feature
                        ),
                        "features": features.tolist(),
                    }

                    anomaly_events.append(event)

                    active_anomalies[track_id] = event

                elif (
                    not anomalous
                    and track.anomaly_active
                ):
                    active_anomalies.pop(
                        track_id,
                        None,
                    )

                track.anomaly_active = anomalous

            # Draw trajectory history.
            if len(points) >= 2:
                polyline = np.round(
                    points
                ).astype(np.int32)

                cv2.polylines(
                    annotated,
                    [polyline],
                    False,
                    (255, 180, 0),
                    2,
                )

            if anomalous:
                color = (0, 0, 255)

                label = (
                    f"ID {track_id} ANOMALY "
                    f"{score:.1f}"
                )
            else:
                color = (0, 220, 0)

                label = (
                    f"ID {track_id} "
                    f"{score:.1f}"
                )

            cv2.circle(
                annotated,
                center,
                5,
                color,
                -1,
            )

            cv2.putText(
                annotated,
                label,
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            if anomalous:
                cv2.putText(
                    annotated,
                    dominant_feature,
                    (center[0] + 8, center[1] + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

        # ----------------------------------------------------------
        # Status overlay
        # ----------------------------------------------------------

        if calibration_complete:
            status = "ANOMALY DETECTION ACTIVE"
            status_color = (0, 220, 0)
        else:
            status = (
                f"CALIBRATING NORMAL MOTION: "
                f"{frame_index}/{args.warmup_frames}"
            )

            status_color = (0, 200, 255)

        cv2.putText(
            annotated,
            status,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2,
        )

        cv2.putText(
            annotated,
            f"Tracks: {len(tracks)}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated,
            f"Anomaly events: {len(anomaly_events)}",
            (15, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        writer.write(annotated)

        if args.show_mask:
            cv2.imshow(
                "Foreground mask",
                foreground_mask,
            )

        if not args.no_display:
            cv2.imshow(
                "Trajectory Anomaly Detection",
                annotated,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        if frame_index % 100 == 0:
            print(
                f"Processed {frame_index} frames | "
                f"tracks={len(tracks)} | "
                f"events={len(anomaly_events)}"
            )

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    # --------------------------------------------------------------
    # Save summary
    # --------------------------------------------------------------

    summary = {
        "lesson": LESSON_NAME,
        "video": args.video,
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "frames_processed": int(frame_index),
        "calibration_complete": bool(
            calibration_complete
        ),
        "calibration_sample_count": int(
            len(calibration_samples)
        ),
        "anomaly_threshold": float(args.threshold),
        "normal_motion_model": model.to_dict(),
        "anomaly_event_count": int(
            len(anomaly_events)
        ),
        "anomaly_events": anomaly_events,
        "active_anomalies_at_end": [
            int(track_id)
            for track_id in active_anomalies
        ],
        "limitations": [
            "MOG2 detects moving regions, not semantic object classes.",
            "The camera should remain mostly stationary.",
            "Calibration assumes the warmup segment is normal.",
            "Measurements are in image coordinates, not physical units.",
            "The anomaly score is not a calibrated probability.",
        ],
    }

    summary_path = os.path.join(
        args.output_dir,
        "anomaly_summary.json",
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

    print("\nVideo processing complete.")
    print(f"Frames: {frame_index}")
    print(f"Calibration complete: {calibration_complete}")
    print(f"Anomaly events: {len(anomaly_events)}")
    print(f"Video: {output_path}")
    print(f"Summary: {summary_path}")


# ======================================================================
# NUMERICAL TESTS
# ======================================================================

def run_tests():
    print("\nRunning Lesson 54 tests...")
    print("=" * 65)

    extractor = MotionFeatureExtractor()

    # --------------------------------------------------------------
    # Test 1: trajectory storage
    # --------------------------------------------------------------

    trajectory = Trajectory(max_length=5)

    for i in range(8):
        trajectory.add([i, 0])

    assert len(trajectory) == 5
    assert np.allclose(
        trajectory.array()[0],
        [3, 0],
    )

    print("Test 01 passed: trajectory history limit.")

    # --------------------------------------------------------------
    # Test 2: trajectory point validation
    # --------------------------------------------------------------

    try:
        trajectory.add([1, 2, 3])
        raise AssertionError("Invalid point was accepted.")
    except ValueError:
        pass

    print("Test 02 passed: trajectory point validation.")

    # --------------------------------------------------------------
    # Test 3: insufficient history
    # --------------------------------------------------------------

    short_trajectory = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
        ],
        dtype=np.float64,
    )

    assert extractor.extract(short_trajectory) is None

    print("Test 03 passed: minimum trajectory length.")

    # --------------------------------------------------------------
    # Test 4: straight-line speed
    # --------------------------------------------------------------

    straight = np.column_stack(
        [
            np.arange(10, dtype=np.float64) * 2.0,
            np.zeros(10),
        ]
    )

    features = extractor.extract(straight)

    assert np.isclose(features[0], 2.0)

    print("Test 04 passed: straight-line speed.")

    # --------------------------------------------------------------
    # Test 5: straight-line acceleration
    # --------------------------------------------------------------

    assert np.isclose(features[1], 0.0)

    print("Test 05 passed: constant-velocity acceleration.")

    # --------------------------------------------------------------
    # Test 6: straight-line turning angle
    # --------------------------------------------------------------

    assert np.isclose(features[2], 0.0)

    print("Test 06 passed: straight-line turning angle.")

    # --------------------------------------------------------------
    # Test 7: straight-line path deviation
    # --------------------------------------------------------------

    assert features[3] < 1e-8

    print("Test 07 passed: straight-line path deviation.")

    # --------------------------------------------------------------
    # Test 8: direction reversal
    # --------------------------------------------------------------

    reversal = np.array(
        [
            [0, 0],
            [2, 0],
            [4, 0],
            [6, 0],
            [4, 0],
        ],
        dtype=np.float64,
    )

    reversal_features = extractor.extract(reversal)

    assert np.isclose(
        reversal_features[2],
        180.0,
        atol=1e-5,
    )

    print("Test 08 passed: 180-degree direction reversal.")

    # --------------------------------------------------------------
    # Test 9: acceleration magnitude
    # --------------------------------------------------------------

    accelerating = np.array(
        [
            [0, 0],
            [1, 0],
            [3, 0],
            [6, 0],
            [10, 0],
        ],
        dtype=np.float64,
    )

    acceleration_features = extractor.extract(
        accelerating
    )

    assert np.isclose(
        acceleration_features[1],
        1.0,
    )

    print("Test 09 passed: acceleration magnitude.")

    # --------------------------------------------------------------
    # Test 10: lateral path deviation
    # --------------------------------------------------------------

    curved = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 0],
            [3, -1],
            [4, 0],
            [5, 1],
            [6, 0],
        ],
        dtype=np.float64,
    )

    curved_deviation = (
        extractor.lateral_path_deviation(curved)
    )

    assert curved_deviation > 0.0

    print("Test 10 passed: lateral path deviation.")

    # --------------------------------------------------------------
    # Test 11: robust model fitting
    # --------------------------------------------------------------

    normal_samples = generate_normal_trajectory_samples(
        number_of_trajectories=20,
        points_per_trajectory=20,
    )

    model = RobustNormalMotionModel(
        threshold=3.5
    ).fit(normal_samples)

    assert model.fitted
    assert model.sample_count >= 10

    print("Test 11 passed: robust normal-motion model fitting.")

    # --------------------------------------------------------------
    # Test 12: normal observation
    # --------------------------------------------------------------

    normal_result = model.score(
        model.median.copy()
    )

    assert normal_result["score"] == 0.0
    assert not normal_result["anomalous"]

    print("Test 12 passed: normal observation classification.")

    # --------------------------------------------------------------
    # Test 13: anomalous observation
    # --------------------------------------------------------------

    unusual = model.median.copy()

    unusual[0] += 10.0 * model.robust_scale[0]

    unusual_result = model.score(unusual)

    assert unusual_result["anomalous"]
    assert unusual_result["score"] > model.threshold

    print("Test 13 passed: anomalous observation classification.")

    # --------------------------------------------------------------
    # Test 14: anomaly feature identification
    # --------------------------------------------------------------

    assert unusual_result["dominant_feature"] == "speed"

    print("Test 14 passed: dominant anomaly feature.")

    # --------------------------------------------------------------
    # Test 15: robust scale is never zero
    # --------------------------------------------------------------

    assert np.all(
        model.robust_scale >= FEATURE_SCALE_FLOORS
    )

    print("Test 15 passed: minimum robust feature scales.")

    # --------------------------------------------------------------
    # Test 16: invalid calibration shape
    # --------------------------------------------------------------

    try:
        RobustNormalMotionModel().fit(
            np.zeros((20, 3))
        )

        raise AssertionError(
            "Invalid calibration shape was accepted."
        )

    except ValueError:
        pass

    print("Test 16 passed: calibration shape validation.")

    # --------------------------------------------------------------
    # Test 17: tracker creates tracks
    # --------------------------------------------------------------

    tracker = CentroidTracker()

    boxes = [
        (0, 0, 20, 20),
        (100, 100, 120, 120),
    ]

    tracks = tracker.update(boxes)

    assert len(tracks) == 2

    print("Test 17 passed: tracker initialization.")

    # --------------------------------------------------------------
    # Test 18: tracker associates nearby detections
    # --------------------------------------------------------------

    tracks = tracker.update(
        [
            (2, 0, 22, 20),
            (102, 100, 122, 120),
        ]
    )

    assert len(tracks) == 2

    assert len(tracks[1].trajectory) == 2
    assert len(tracks[2].trajectory) == 2

    print("Test 18 passed: centroid association.")

    # --------------------------------------------------------------
    # Test 19: tracker creates a new track for distant detection
    # --------------------------------------------------------------

    tracks = tracker.update(
        [
            (4, 0, 24, 20),
            (104, 100, 124, 120),
            (300, 300, 320, 320),
        ]
    )

    assert len(tracks) == 3

    print("Test 19 passed: new track creation.")

    # --------------------------------------------------------------
    # Test 20: tracker removes expired tracks
    # --------------------------------------------------------------

    expiration_tracker = CentroidTracker(
        max_missed=2
    )

    expiration_tracker.update(
        [(0, 0, 20, 20)]
    )

    expiration_tracker.update([])
    expiration_tracker.update([])
    expiration_tracker.update([])

    assert len(expiration_tracker.tracks) == 0

    print("Test 20 passed: expired track removal.")

    # --------------------------------------------------------------
    # Test 21: model rejects non-finite values
    # --------------------------------------------------------------

    invalid_samples = normal_samples.copy()
    invalid_samples[0, 0] = np.nan

    try:
        RobustNormalMotionModel().fit(
            invalid_samples
        )

        raise AssertionError(
            "NaN calibration data was accepted."
        )

    except ValueError:
        pass

    print("Test 21 passed: non-finite calibration rejection.")

    # --------------------------------------------------------------
    # Test 22: model serialization
    # --------------------------------------------------------------

    model_dict = model.to_dict()

    assert model_dict["fitted"]
    assert model_dict["sample_count"] > 0
    assert len(model_dict["median"]) == 4

    print("Test 22 passed: model summary serialization.")

    print("=" * 65)
    print("All 22 tests passed.")


# ======================================================================
# COMMAND LINE
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=LESSON_NAME
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Input real video file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run numerical tests only.",
    )

    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=90,
        help="Number of initial frames used for normal calibration.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=3.5,
        help="Robust Z-score anomaly threshold.",
    )

    parser.add_argument(
        "--min-area",
        type=float,
        default=500,
        help="Minimum MOG2 foreground contour area.",
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=80.0,
        help="Maximum centroid association distance.",
    )

    parser.add_argument(
        "--max-missed",
        type=int,
        default=12,
        help="Maximum missed frames before track deletion.",
    )

    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Display the foreground mask.",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display window.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.threshold <= 0:
        raise ValueError("Threshold must be positive.")

    if args.warmup_frames < 1:
        raise ValueError("Warmup frames must be positive.")

    if args.check_only:
        run_tests()
        return

    run_tests()

    print("\nGenerating mathematical demonstration...")

    create_anomaly_demo(
        args.output_dir
    )

    if args.video is not None:
        print("\nProcessing real video...")

        process_video(args)

    else:
        print(
            "\nNo video supplied. "
            "Mathematical demonstration completed."
        )

        print(
            "To process real footage, run:\n"
            "python lesson54_trajectory_anomaly_detection.py "
            "--video conveyor_video.mp4"
        )


if __name__ == "__main__":
    main()