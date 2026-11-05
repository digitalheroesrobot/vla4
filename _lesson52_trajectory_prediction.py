"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 52 — TRAJECTORY PREDICTION AND FUTURE POSITION ESTIMATION
======================================================================

Filename:
    lesson52_trajectory_prediction.py

======================================================================
LEARNING OBJECTIVES
======================================================================

1. Understand the difference between tracking and prediction.
2. Estimate velocity from observed object positions.
3. Estimate acceleration from trajectory history.
4. Implement constant-velocity prediction.
5. Implement constant-acceleration prediction.
6. Implement exponential moving average velocity estimation.
7. Fit trajectories using least-squares regression.
8. Estimate prediction uncertainty from historical residuals.
9. Evaluate prediction error at different future horizons.
10. Visualize future object positions on real video.

======================================================================
1. TRACKING VS. TRAJECTORY PREDICTION
======================================================================

Tracking answers:

    Where is the object now?

Trajectory prediction answers:

    Where is the object likely to be later?

For a moving conveyor object:

    Current camera observation:
        x = 250 pixels

    Estimated velocity:
        vx = 4 pixels/frame

    Prediction horizon:
        10 frames

    Predicted position:
        x_future = 250 + 4*10
                 = 290 pixels

A robot that takes time to move its gripper must account for
this future position.

======================================================================
2. CONSTANT-VELOCITY MODEL
======================================================================

Assume velocity is constant:

    x(t) = x0 + vx*t

    y(t) = y0 + vy*t

In vector form:

    p(t) = p0 + v*t

Where:

    p0 = [x0, y0]
    v  = [vx, vy]

This model is appropriate when:

    - Objects move steadily on a conveyor.
    - Motion is approximately straight.
    - Acceleration is small.

It performs poorly when objects accelerate, decelerate, or turn.

======================================================================
3. VELOCITY ESTIMATION USING LEAST SQUARES
======================================================================

We have observed positions:

    (t1, x1), (t2, x2), ..., (tn, xn)

Fit:

    x(t) = a + b*t

In matrix form:

    A = [[1, t1],
         [1, t2],
         ...
         [1, tn]]

    theta = [a, b]

    y = A*theta

Least-squares solution:

    theta = argmin ||A*theta - y||^2

    theta = pinv(A)*y

The slope b estimates velocity.

We fit x and y independently.

IMPORTANT:
    Time is measured in frames in this lesson.
    If the video runs at FPS frames per second:

        velocity_pixels_per_second =
            velocity_pixels_per_frame * FPS

======================================================================
4. CONSTANT-ACCELERATION MODEL
======================================================================

Assume acceleration is constant:

    x(t) = x0 + vx*t + 0.5*ax*t^2

    y(t) = y0 + vy*t + 0.5*ay*t^2

Fit a quadratic polynomial:

    x(t) = a + b*t + c*t^2

Then:

    vx = b

    ax = 2*c

The same calculation applies to y.

This model is useful when an object speeds up or slows down.

However, quadratic fitting needs enough observations.
A small history window can make acceleration estimates noisy.

======================================================================
5. EXPONENTIAL MOVING AVERAGE VELOCITY
======================================================================

Raw velocity estimates may fluctuate because of noisy detections.

EMA:

    v_ema(k) = alpha*v(k)
               + (1-alpha)*v_ema(k-1)

Where:

    alpha in (0, 1]

Small alpha:
    More smoothing, slower response.

Large alpha:
    Faster response, less smoothing.

======================================================================
6. PREDICTION UNCERTAINTY
======================================================================

A prediction is not exact.

We estimate uncertainty using the residuals of the fitted model.

For observed point p_i and fitted point p_hat_i:

    residual_i = p_i - p_hat_i

Residual variance:

    sigma^2 = sum(residual_i^2) / degrees_of_freedom

This lesson uses a simplified uncertainty estimate:

    uncertainty(h) =
        residual_std + uncertainty_growth*h

This is a visualization aid, not a fully calibrated probabilistic
prediction interval.

A production system should propagate state covariance through its
motion model and validate uncertainty using held-out trajectories.

======================================================================
7. PREDICTION HORIZON
======================================================================

The prediction horizon determines how far into the future we predict.

Example at 30 FPS:

    5 frames  = 0.167 seconds
    15 frames = 0.500 seconds
    30 frames = 1.000 second

Longer horizons generally produce larger errors.

For robot grasping, the useful horizon should correspond to the
time between image capture and actual gripper contact.

======================================================================
8. ROBOTICS APPLICATION — OPENARM V2
======================================================================

Camera detects a cup on a conveyor.

    Camera detection
          |
          v
    Object tracker
          |
          v
    Position history
          |
          v
    Velocity estimation
          |
          v
    Future position prediction
          |
          v
    Pixel-to-world mapping
          |
          v
    Robot grasp target
          |
          v
    ACT / motion controller

IMPORTANT:

A pixel prediction is not yet a robot-coordinate prediction.

For a planar conveyor, use a calibrated homography:

    [X, Y, W].T = H * [u, v, 1].T

Then:

    X_world = X / W
    Y_world = Y / W

For a moving object, predict its future pixel position first,
then map that position into conveyor coordinates.

For full 3-D manipulation, camera calibration and camera-to-robot
extrinsics are also required.

======================================================================
9. DEMONSTRATION DETECTOR
======================================================================

This lesson uses MOG2 background subtraction to detect moving
regions in a mostly stationary-camera video.

It is NOT a semantic waste detector.

For a real OpenArm waste-sorting system, replace MotionDetector
with YOLO, FCOS, Mask R-CNN, or another detector.

The trajectory predictor does not depend on which detector is used.

======================================================================
DEPENDENCIES
======================================================================

    pip install numpy opencv-python scipy matplotlib

======================================================================
RUN
======================================================================

    python lesson52_trajectory_prediction.py --check-only

    python lesson52_trajectory_prediction.py \
        --video conveyor_video.mp4

    python lesson52_trajectory_prediction.py \
        --video conveyor_video.mp4 \
        --horizon 20

    python lesson52_trajectory_prediction.py \
        --video conveyor_video.mp4 \
        --horizon 30 \
        --no-display

======================================================================
OUTPUT
======================================================================

    runs/lesson52/
        trajectory_prediction.mp4
        trajectory_summary.json
        prediction_comparison.png
        prediction_error.png
        trajectory_examples.png
======================================================================
"""

import argparse
import json
import time

from dataclasses import dataclass, field

from pathlib import Path

import cv2

import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment


# =====================================================================
# CONFIGURATION
# =====================================================================

DEFAULT_HISTORY_LENGTH = 20

DEFAULT_MIN_OBSERVATIONS = 5

DEFAULT_HORIZON = 15

DEFAULT_MAX_ASSOCIATION_DISTANCE = 100.0

DEFAULT_MAX_MISSED_FRAMES = 20

DEFAULT_EMA_ALPHA = 0.35


# =====================================================================
# BASIC GEOMETRY
# =====================================================================

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

    x1 = max(ax1, bx1)

    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)

    y2 = min(ay2, by2)

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
# TRAJECTORY DATA
# =====================================================================

@dataclass
class Trajectory:

    track_id: int

    history_length: int = DEFAULT_HISTORY_LENGTH

    frames: list = field(
        default_factory=list
    )

    positions: list = field(
        default_factory=list
    )

    ema_velocity: np.ndarray = None

    previous_raw_velocity: np.ndarray = None

    def add_observation(
        self,
        frame_index,
        position
    ):

        position = np.asarray(

            position,

            dtype=np.float64
        )

        if self.frames:

            previous_frame = self.frames[-1]

            previous_position = np.asarray(
                self.positions[-1]
            )

            dt = frame_index - previous_frame

            if dt > 0:

                raw_velocity = (

                    position - previous_position

                ) / dt

                if self.ema_velocity is None:

                    self.ema_velocity = raw_velocity.copy()

                else:

                    self.ema_velocity = (

                        DEFAULT_EMA_ALPHA * raw_velocity

                        + (1.0 - DEFAULT_EMA_ALPHA)
                        * self.ema_velocity
                    )

                self.previous_raw_velocity = raw_velocity.copy()

        self.frames.append(
            int(frame_index)
        )

        self.positions.append(
            position.tolist()
        )

        if len(self.frames) > self.history_length:

            self.frames.pop(0)

            self.positions.pop(0)

    @property
    def observation_count(self):

        return len(
            self.frames
        )

    @property
    def latest_frame(self):

        if not self.frames:
            return None

        return self.frames[-1]

    @property
    def latest_position(self):

        if not self.positions:
            return None

        return np.asarray(

            self.positions[-1],

            dtype=np.float64
        )

    def arrays(self):

        return (

            np.asarray(
                self.frames,
                dtype=np.float64
            ),

            np.asarray(
                self.positions,
                dtype=np.float64
            )
        )


# =====================================================================
# TRAJECTORY MODEL
# =====================================================================

class TrajectoryPredictor:

    """
    Fits motion models to a trajectory.

    Models:

        1. Constant velocity.
        2. Constant acceleration.
        3. EMA velocity.

    Coordinates are fitted independently.

    Time is shifted so the latest observation occurs at t=0.
    This improves numerical conditioning.
    """

    def __init__(

        self,

        min_observations=DEFAULT_MIN_OBSERVATIONS,

        ema_alpha=DEFAULT_EMA_ALPHA
    ):

        self.min_observations = min_observations

        self.ema_alpha = ema_alpha

    def _prepare(self, trajectory):

        frames, positions = trajectory.arrays()

        if len(frames) < self.min_observations:

            raise ValueError(

                f"Need at least {self.min_observations} observations."
            )

        relative_time = (
            frames - frames[-1]
        )

        return relative_time, positions

    def _fit_polynomial(

        self,

        time_values,

        positions,

        degree
    ):

        """
        Fit each spatial coordinate independently.

        For degree 1:

            p(t) = a + b*t

        For degree 2:

            p(t) = a + b*t + c*t^2

        Returns coefficients ordered from highest degree to
        constant term, matching np.polyfit.
        """

        coefficients = []

        fitted_positions = np.zeros_like(
            positions
        )

        residuals = []

        for dimension in range(2):

            coeff = np.polyfit(

                time_values,

                positions[:, dimension],

                degree
            )

            coefficients.append(
                coeff
            )

            fitted = np.polyval(

                coeff,

                time_values
            )

            fitted_positions[:, dimension] = fitted

            residuals.append(

                positions[:, dimension] - fitted
            )

        residuals = np.column_stack(
            residuals
        )

        return (

            coefficients,

            fitted_positions,

            residuals
        )

    def fit_constant_velocity(self, trajectory):

        time_values, positions = self._prepare(
            trajectory
        )

        coefficients, fitted, residuals = (

            self._fit_polynomial(

                time_values,

                positions,

                degree=1
            )
        )

        current_position = positions[-1]

        velocity = np.array(

            [

                coefficients[0][0],

                coefficients[1][0]
            ],

            dtype=np.float64
        )

        residual_std = self._residual_std(
            residuals,
            parameter_count=2
        )

        return {

            "position": current_position.copy(),

            "velocity": velocity,

            "acceleration": np.zeros(2),

            "residual_std": residual_std,

            "fitted_positions": fitted,

            "residuals": residuals
        }

    def fit_constant_acceleration(self, trajectory):

        time_values, positions = self._prepare(
            trajectory
        )

        if len(time_values) < 3:

            raise ValueError(
                "Quadratic fitting needs at least 3 observations."
            )

        coefficients, fitted, residuals = (

            self._fit_polynomial(

                time_values,

                positions,

                degree=2
            )
        )

        current_position = positions[-1]

        velocity = np.array(

            [

                coefficients[0][1],

                coefficients[1][1]
            ],

            dtype=np.float64
        )

        acceleration = np.array(

            [

                2.0 * coefficients[0][0],

                2.0 * coefficients[1][0]
            ],

            dtype=np.float64
        )

        residual_std = self._residual_std(
            residuals,
            parameter_count=3
        )

        return {

            "position": current_position.copy(),

            "velocity": velocity,

            "acceleration": acceleration,

            "residual_std": residual_std,

            "fitted_positions": fitted,

            "residuals": residuals
        }

    @staticmethod
    def _residual_std(
        residuals,
        parameter_count
    ):

        n = len(
            residuals
        )

        degrees_of_freedom = max(

            1,

            n - parameter_count
        )

        squared_error = np.sum(
            residuals ** 2
        )

        variance = (

            squared_error

            / (2 * degrees_of_freedom)
        )

        return float(
            np.sqrt(max(variance, 0.0))
        )

    def predict_constant_velocity(

        self,

        trajectory,

        horizon
    ):

        model = self.fit_constant_velocity(
            trajectory
        )

        predicted_position = (

            model["position"]

            + model["velocity"] * horizon
        )

        return {

            **model,

            "predicted_position": predicted_position
        }

    def predict_constant_acceleration(

        self,

        trajectory,

        horizon
    ):

        model = self.fit_constant_acceleration(
            trajectory
        )

        predicted_position = (

            model["position"]

            + model["velocity"] * horizon

            + 0.5
            * model["acceleration"]
            * horizon ** 2
        )

        return {

            **model,

            "predicted_position": predicted_position
        }

    def predict_ema(

        self,

        trajectory,

        horizon
    ):

        if trajectory.observation_count < 2:

            raise ValueError(
                "EMA prediction needs at least two observations."
            )

        velocity = trajectory.ema_velocity

        if velocity is None:

            raise ValueError(
                "EMA velocity has not been initialized."
            )

        position = trajectory.latest_position

        predicted_position = (
            position + velocity * horizon
        )

        return {

            "position": position.copy(),

            "velocity": velocity.copy(),

            "acceleration": np.zeros(2),

            "residual_std": 0.0,

            "predicted_position": predicted_position
        }

    def predict_all(

        self,

        trajectory,

        horizon
    ):

        return {

            "constant_velocity": self.predict_constant_velocity(

                trajectory,

                horizon
            ),

            "constant_acceleration": self.predict_constant_acceleration(

                trajectory,

                horizon
            ),

            "ema": self.predict_ema(

                trajectory,

                horizon
            )
        }


# =====================================================================
# PREDICTION UNCERTAINTY
# =====================================================================

def prediction_uncertainty(

    residual_std,

    horizon,

    growth_per_frame=0.5
):

    """
    Simplified uncertainty visualization.

    This is NOT a statistically calibrated confidence interval.

    It provides a conservative visual indication that uncertainty
    generally increases as the prediction horizon grows.
    """

    return (

        float(residual_std)

        + growth_per_frame * float(horizon)
    )


# =====================================================================
# SIMPLE MULTI-OBJECT TRACKER
# =====================================================================

@dataclass
class ObjectTrack:

    track_id: int

    bbox: np.ndarray

    trajectory: Trajectory

    missed_frames: int = 0

    age: int = 1

    active: bool = True


class CentroidTracker:

    """
    Lightweight tracker for this lesson.

    Association is based on predicted centroid distance.

    It is deliberately kept separate from the trajectory predictor:
    prediction should be reusable with SORT, ByteTrack, Deep SORT,
    or another tracking system.
    """

    def __init__(

        self,

        max_distance=DEFAULT_MAX_ASSOCIATION_DISTANCE,

        max_missed_frames=DEFAULT_MAX_MISSED_FRAMES,

        history_length=DEFAULT_HISTORY_LENGTH
    ):

        self.max_distance = max_distance

        self.max_missed_frames = max_missed_frames

        self.history_length = history_length

        self.tracks = {}

        self.next_id = 1

        self.frame_index = 0

    def _predict_center(self, track):

        trajectory = track.trajectory

        if trajectory.observation_count < 2:

            return trajectory.latest_position

        frames, positions = trajectory.arrays()

        dt = frames[-1] - frames[-2]

        if dt <= 0:

            return trajectory.latest_position

        velocity = (

            positions[-1] - positions[-2]

        ) / dt

        return (
            positions[-1] + velocity
        )

    def _create_track(self, bbox):

        center = box_center(
            bbox
        )

        track_id = self.next_id

        self.next_id += 1

        trajectory = Trajectory(

            track_id=track_id,

            history_length=self.history_length
        )

        trajectory.add_observation(

            self.frame_index,

            center
        )

        track = ObjectTrack(

            track_id=track_id,

            bbox=np.asarray(
                bbox,
                dtype=np.float64
            ),

            trajectory=trajectory
        )

        self.tracks[track_id] = track

    def update(self, detections):

        self.frame_index += 1

        active_tracks = [

            track

            for track in self.tracks.values()

            if track.active
        ]

        if active_tracks and detections:

            cost = np.full(

                (

                    len(active_tracks),

                    len(detections)
                ),

                np.inf,

                dtype=np.float64
            )

            for i, track in enumerate(active_tracks):

                predicted_center = self._predict_center(
                    track
                )

                for j, bbox in enumerate(detections):

                    center = box_center(
                        bbox
                    )

                    distance = np.linalg.norm(

                        predicted_center - center
                    )

                    if distance <= self.max_distance:

                        cost[i, j] = distance

            rows, cols = linear_sum_assignment(

                np.where(

                    np.isfinite(cost),

                    cost,

                    1e9
                )
            )

            matched_track_indices = set()

            matched_detection_indices = set()

            for row, col in zip(rows, cols):

                if not np.isfinite(cost[row, col]):
                    continue

                track = active_tracks[row]

                bbox = np.asarray(

                    detections[col],

                    dtype=np.float64
                )

                track.bbox = bbox

                track.missed_frames = 0

                track.age += 1

                track.trajectory.add_observation(

                    self.frame_index,

                    box_center(bbox)
                )

                matched_track_indices.add(
                    row
                )

                matched_detection_indices.add(
                    col
                )

            for index, track in enumerate(active_tracks):

                if index not in matched_track_indices:

                    track.missed_frames += 1

                    if track.missed_frames > self.max_missed_frames:

                        track.active = False

            for index, bbox in enumerate(detections):

                if index not in matched_detection_indices:

                    self._create_track(
                        bbox
                    )

        else:

            for track in active_tracks:

                track.missed_frames += 1

                if track.missed_frames > self.max_missed_frames:

                    track.active = False

            for bbox in detections:

                self._create_track(
                    bbox
                )

        return [

            track

            for track in self.tracks.values()

            if track.active
        ]


# =====================================================================
# MOTION DETECTOR
# =====================================================================

class MotionDetector:

    """
    MOG2 foreground detector for a stationary camera.

    Replace this with a semantic detector for real robot deployment.
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

            if cv2.contourArea(contour) < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

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

        return detections, mask


# =====================================================================
# VISUALIZATION
# =====================================================================

def draw_cross(

    image,

    point,

    color,

    size=9,

    thickness=2
):

    x, y = np.asarray(
        point
    ).astype(
        int
    )

    cv2.line(

        image,

        (x - size, y),

        (x + size, y),

        color,

        thickness
    )

    cv2.line(

        image,

        (x, y - size),

        (x, y + size),

        color,

        thickness
    )


def draw_prediction(

    frame,

    track,

    predictor,

    horizon
):

    output = frame.copy()

    x1, y1, x2, y2 = track.bbox.astype(
        int
    )

    color = (

        int(track.track_id * 67 % 255),

        int(track.track_id * 131 % 255),

        int(track.track_id * 197 % 255)
    )

    cv2.rectangle(

        output,

        (x1, y1),

        (x2, y2),

        color,

        2
    )

    cv2.putText(

        output,

        f"ID {track.track_id}",

        (x1, max(20, y1 - 8)),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        color,

        2
    )

    trajectory = track.trajectory

    if trajectory.observation_count >= DEFAULT_MIN_OBSERVATIONS:

        try:

            predictions = predictor.predict_all(

                trajectory,

                horizon
            )

            current = trajectory.latest_position

            cv_position = predictions[

                "constant_velocity"

            ]["predicted_position"]

            ca_position = predictions[

                "constant_acceleration"

            ]["predicted_position"]

            ema_position = predictions[

                "ema"

            ]["predicted_position"]

            # Constant velocity: green.

            cv_point = np.round(
                cv_position
            ).astype(
                int
            )

            draw_cross(

                output,

                cv_point,

                (0, 255, 0)
            )

            cv2.line(

                output,

                tuple(np.round(current).astype(int)),

                tuple(cv_point),

                (0, 255, 0),

                2
            )

            # Constant acceleration: red.

            ca_point = np.round(
                ca_position
            ).astype(
                int
            )

            draw_cross(

                output,

                ca_point,

                (0, 0, 255)
            )

            cv2.line(

                output,

                tuple(np.round(current).astype(int)),

                tuple(ca_point),

                (0, 0, 255),

                2
            )

            # EMA: blue.

            ema_point = np.round(
                ema_position
            ).astype(
                int
            )

            draw_cross(

                output,

                ema_point,

                (255, 150, 0)
            )

            cv2.putText(

                output,

                f"CV: ({cv_point[0]}, {cv_point[1]})",

                (15, 60 + (track.track_id % 5) * 22),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.45,

                (0, 255, 0),

                1
            )

        except (ValueError, np.linalg.LinAlgError):

            pass

    return output


# =====================================================================
# SYNTHETIC TRAJECTORY GENERATION FOR NUMERICAL TESTS
# =====================================================================

def make_linear_trajectory(

    velocity=(3.0, -2.0),

    initial_position=(10.0, 20.0),

    number_of_frames=12,

    noise_std=0.0,

    seed=42
):

    rng = np.random.default_rng(
        seed
    )

    frames = np.arange(
        number_of_frames,
        dtype=np.float64
    )

    velocity = np.asarray(
        velocity,
        dtype=np.float64
    )

    initial_position = np.asarray(
        initial_position,
        dtype=np.float64
    )

    positions = (

        initial_position[None, :]

        + frames[:, None] * velocity[None, :]
    )

    if noise_std > 0:

        positions += rng.normal(

            0.0,

            noise_std,

            positions.shape
        )

    trajectory = Trajectory(
        track_id=1
    )

    for frame, position in zip(frames, positions):

        trajectory.add_observation(

            int(frame),

            position
        )

    return trajectory


def make_accelerating_trajectory(

    velocity=(1.0, 2.0),

    acceleration=(0.5, -0.25),

    initial_position=(10.0, 20.0),

    number_of_frames=12
):

    frames = np.arange(
        number_of_frames,
        dtype=np.float64
    )

    velocity = np.asarray(
        velocity,
        dtype=np.float64
    )

    acceleration = np.asarray(
        acceleration,
        dtype=np.float64
    )

    initial_position = np.asarray(
        initial_position,
        dtype=np.float64
    )

    positions = (

        initial_position[None, :]

        + frames[:, None] * velocity[None, :]

        + 0.5
        * frames[:, None] ** 2
        * acceleration[None, :]
    )

    trajectory = Trajectory(
        track_id=1
    )

    for frame, position in zip(frames, positions):

        trajectory.add_observation(

            int(frame),

            position
        )

    return trajectory


# =====================================================================
# PREDICTION EVALUATION
# =====================================================================

def evaluate_prediction_error(

    trajectory,

    predictor,

    horizons=(1, 3, 5, 10, 15)
):

    """
    Rolling-origin evaluation.

    For each eligible historical frame:

        1. Fit a model using observations up to that frame.
        2. Predict a future position.
        3. Compare with the actual future position.

    This is more informative than evaluating only the final
    fitted model against its own training data.
    """

    frames, positions = trajectory.arrays()

    results = {

        "constant_velocity": {},

        "constant_acceleration": {},

        "ema": {}
    }

    minimum_history = max(

        predictor.min_observations,

        3
    )

    for horizon in horizons:

        errors = {

            "constant_velocity": [],

            "constant_acceleration": [],

            "ema": []
        }

        for current_index in range(

            minimum_history - 1,

            len(frames) - horizon
        ):

            subset = Trajectory(

                track_id=trajectory.track_id,

                history_length=len(frames)
            )

            for frame, position in zip(

                frames[:current_index + 1],

                positions[:current_index + 1]
            ):

                subset.add_observation(

                    int(frame),

                    position
                )

            actual = positions[
                current_index + horizon
            ]

            try:

                predictions = predictor.predict_all(

                    subset,

                    horizon
                )

            except (ValueError, np.linalg.LinAlgError):

                continue

            for model_name, prediction in predictions.items():

                error = np.linalg.norm(

                    prediction["predicted_position"] - actual
                )

                errors[model_name].append(
                    float(error)
                )

        for model_name in results:

            values = errors[model_name]

            results[model_name][str(horizon)] = {

                "count": len(values),

                "mean_error": (

                    float(np.mean(values))

                    if values else None
                ),

                "median_error": (

                    float(np.median(values))

                    if values else None
                )
            }

    return results


# =====================================================================
# PLOTTING
# =====================================================================

def plot_prediction_comparison(

    output_path,

    trajectory=None,

    horizon=10
):

    if trajectory is None:

        trajectory = make_accelerating_trajectory(
            number_of_frames=20
        )

    predictor = TrajectoryPredictor(
        min_observations=5
    )

    predictions = predictor.predict_all(

        trajectory,

        horizon
    )

    frames, positions = trajectory.arrays()

    plt.figure(
        figsize=(10, 7)
    )

    plt.plot(

        positions[:, 0],

        positions[:, 1],

        "ko-",

        label="Observed trajectory"
    )

    current = positions[-1]

    plt.scatter(

        [current[0]],

        [current[1]],

        marker="s",

        s=90,

        label="Current position"
    )

    styles = {

        "constant_velocity": "g",

        "constant_acceleration": "r",

        "ema": "b"
    }

    for name, result in predictions.items():

        future = result["predicted_position"]

        color = styles[name]

        plt.plot(

            [current[0], future[0]],

            [current[1], future[1]],

            color + "--",

            label=name.replace("_", " ")
        )

        plt.scatter(

            [future[0]],

            [future[1]],

            color=color,

            s=100
        )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "Image X (pixels)"
    )

    plt.ylabel(
        "Image Y (pixels)"
    )

    plt.title(
        f"Trajectory Prediction: {horizon}-Frame Horizon"
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


def plot_prediction_errors(

    output_path,

    results
):

    model_names = list(
        results.keys()
    )

    horizons = sorted(

        {

            int(h)

            for model in results.values()

            for h in model.keys()
        }
    )

    plt.figure(
        figsize=(10, 6)
    )

    for model_name in model_names:

        x_values = []

        y_values = []

        for horizon in horizons:

            value = results[model_name][str(horizon)][

                "mean_error"
            ]

            if value is not None:

                x_values.append(
                    horizon
                )

                y_values.append(
                    value
                )

        if x_values:

            plt.plot(

                x_values,

                y_values,

                marker="o",

                label=model_name.replace("_", " ")
            )

    plt.xlabel(
        "Prediction horizon (frames)"
    )

    plt.ylabel(
        "Mean prediction error (pixels)"
    )

    plt.title(
        "Prediction Error vs. Future Horizon"
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


def plot_trajectory_examples(
    output_path
):

    linear = make_linear_trajectory(
        number_of_frames=20
    )

    accelerating = make_accelerating_trajectory(
        number_of_frames=20
    )

    plt.figure(
        figsize=(10, 7)
    )

    for trajectory, label in [

        (linear, "Constant velocity"),

        (accelerating, "Constant acceleration")
    ]:

        _, positions = trajectory.arrays()

        plt.plot(

            positions[:, 0],

            positions[:, 1],

            marker="o",

            label=label
        )

    plt.gca().invert_yaxis()

    plt.xlabel(
        "X position"
    )

    plt.ylabel(
        "Y position"
    )

    plt.title(
        "Example Trajectories"
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
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
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

        output_dir / "trajectory_prediction.mp4"
    )

    output_json = (

        output_dir / "trajectory_summary.json"
    )

    comparison_plot = (

        output_dir / "prediction_comparison.png"
    )

    error_plot = (

        output_dir / "prediction_error.png"
    )

    examples_plot = (

        output_dir / "trajectory_examples.png"
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

    tracker = CentroidTracker(

        max_distance=args.max_association_distance,

        max_missed_frames=args.max_missed_frames,

        history_length=args.history_length
    )

    predictor = TrajectoryPredictor(

        min_observations=args.min_observations
    )

    processing_times = []

    try:

        while True:

            start = time.perf_counter()

            ret, frame = capture.read()

            if not ret:
                break

            detections, mask = detector.detect(
                frame
            )

            tracks = tracker.update(
                detections
            )

            output = frame.copy()

            for track in tracks:

                x1, y1, x2, y2 = track.bbox.astype(
                    int
                )

                color = (

                    int(track.track_id * 67 % 255),

                    int(track.track_id * 131 % 255),

                    int(track.track_id * 197 % 255)
                )

                cv2.rectangle(

                    output,

                    (x1, y1),

                    (x2, y2),

                    color,

                    2
                )

                cv2.putText(

                    output,

                    f"ID {track.track_id}",

                    (x1, max(20, y1 - 8)),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    color,

                    2
                )

                center = track.trajectory.latest_position

                cv2.circle(

                    output,

                    tuple(np.round(center).astype(int)),

                    4,

                    color,

                    -1
                )

                if (

                    track.trajectory.observation_count

                    >= args.min_observations
                ):

                    try:

                        predictions = predictor.predict_all(

                            track.trajectory,

                            args.horizon
                        )

                        current = center

                        cv_position = predictions[

                            "constant_velocity"

                        ]["predicted_position"]

                        ca_position = predictions[

                            "constant_acceleration"

                        ]["predicted_position"]

                        ema_position = predictions[

                            "ema"

                        ]["predicted_position"]

                        # Constant velocity: green cross.

                        draw_cross(

                            output,

                            cv_position,

                            (0, 255, 0),

                            size=8
                        )

                        # Constant acceleration: red cross.

                        draw_cross(

                            output,

                            ca_position,

                            (0, 0, 255),

                            size=8
                        )

                        # EMA: blue cross.

                        draw_cross(

                            output,

                            ema_position,

                            (255, 150, 0),

                            size=8
                        )

                        current_point = tuple(

                            np.round(current).astype(int)
                        )

                        cv_point = tuple(

                            np.round(cv_position).astype(int)
                        )

                        ca_point = tuple(

                            np.round(ca_position).astype(int)
                        )

                        ema_point = tuple(

                            np.round(ema_position).astype(int)
                        )

                        cv2.line(

                            output,

                            current_point,

                            cv_point,

                            (0, 255, 0),

                            2
                        )

                        cv2.line(

                            output,

                            current_point,

                            ca_point,

                            (0, 0, 255),

                            2
                        )

                        cv2.line(

                            output,

                            current_point,

                            ema_point,

                            (255, 150, 0),

                            2
                        )

                    except (ValueError, np.linalg.LinAlgError):

                        pass

            cv2.putText(

                output,

                f"Frame {tracker.frame_index} | "
                f"Horizon: {args.horizon} frames",

                (15, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (255, 255, 255),

                2
            )

            cv2.putText(

                output,

                "CV: GREEN  CA: RED  EMA: BLUE",

                (15, 58),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                2
            )

            writer.write(
                output
            )

            processing_times.append(

                time.perf_counter() - start
            )

            if not args.no_display:

                cv2.imshow(

                    "Lesson 52 - Trajectory Prediction",

                    output
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

    finally:

        capture.release()

        writer.release()

        cv2.destroyAllWindows()

    # ---------------------------------------------------------------
    # Evaluate trajectories that have enough observations.
    # ---------------------------------------------------------------

    trajectory_results = {}

    for track_id, track in tracker.tracks.items():

        if (

            track.trajectory.observation_count

            < args.min_observations
        ):

            continue

        trajectory_results[str(track_id)] = (

            evaluate_prediction_error(

                track.trajectory,

                predictor,

                horizons=(1, 3, 5, 10, 15)
            )
        )

    # Aggregate the results over available tracks.

    aggregated = {

        "constant_velocity": {},

        "constant_acceleration": {},

        "ema": {}
    }

    for model_name in aggregated:

        for horizon in (1, 3, 5, 10, 15):

            values = []

            for track_result in trajectory_results.values():

                value = track_result[model_name][

                    str(horizon)
                ]["mean_error"]

                if value is not None:

                    values.append(
                        value
                    )

            aggregated[model_name][str(horizon)] = {

                "mean_error": (

                    float(np.mean(values))

                    if values else None
                ),

                "track_count": len(values)
            }

    plot_prediction_errors(

        error_plot,

        aggregated
    )

    plot_trajectory_examples(
        examples_plot
    )

    # Select the longest trajectory for the comparison plot.

    longest_track = None

    longest_length = 0

    for track in tracker.tracks.values():

        length = track.trajectory.observation_count

        if length > longest_length:

            longest_length = length

            longest_track = track

    if longest_track is not None:

        plot_prediction_comparison(

            comparison_plot,

            longest_track.trajectory,

            args.horizon
        )

    summary = {

        "input_video": str(video_path),

        "output_video": str(output_video),

        "fps": fps,

        "frame_count": tracker.frame_index,

        "track_count": len(tracker.tracks),

        "prediction_horizon_frames": args.horizon,

        "prediction_horizon_seconds": args.horizon / fps,

        "average_processing_fps": (

            1.0 / np.mean(processing_times)

            if processing_times and np.mean(processing_times) > 0

            else 0.0
        ),

        "prediction_results": aggregated,

        "individual_track_results": trajectory_results,

        "tracks": [

            {

                "track_id": track.track_id,

                "observation_count": track.trajectory.observation_count,

                "positions": track.trajectory.positions,

                "frames": track.trajectory.frames
            }

            for track in tracker.tracks.values()
        ]
    }

    with open(

        output_json,

        "w"
    ) as file:

        json.dump(

            summary,

            file,

            indent=4
        )

    print("\nVideo processing completed.")

    print(
        "Frames:",
        tracker.frame_index
    )

    print(
        "Tracks:",
        len(tracker.tracks)
    )

    print(
        "Average processing FPS:",
        summary["average_processing_fps"]
    )

    print("\nPrediction errors:")

    for model_name, model_results in aggregated.items():

        print("\n", model_name)

        for horizon, result in model_results.items():

            print(

                f"  {horizon:>2} frames: "

                f"{result['mean_error']} pixels"
            )

    print("\nOutput video:", output_video)

    print("Summary:", output_json)

    print("Prediction error plot:", error_plot)

    print("Comparison plot:", comparison_plot)

    print("Trajectory examples:", examples_plot)


# =====================================================================
# NUMERICAL TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print(
        "LESSON 52 — TRAJECTORY PREDICTION TESTS"
    )

    print("=" * 70)

    predictor = TrajectoryPredictor(
        min_observations=5
    )

    # ---------------------------------------------------------------
    # Test 1: trajectory observations.
    # ---------------------------------------------------------------

    trajectory = make_linear_trajectory()

    assert trajectory.observation_count == 12

    assert trajectory.latest_frame == 11

    print("Test 1 passed: trajectory observation storage")

    # ---------------------------------------------------------------
    # Test 2: latest position.
    # ---------------------------------------------------------------

    expected = np.array(
        [43.0, -2.0]
    )

    assert np.allclose(
        trajectory.latest_position,
        expected
    )

    print("Test 2 passed: latest position")

    # ---------------------------------------------------------------
    # Test 3: history length limit.
    # ---------------------------------------------------------------

    limited = Trajectory(

        track_id=2,

        history_length=3
    )

    for frame in range(10):

        limited.add_observation(

            frame,

            [frame, 2 * frame]
        )

    assert limited.observation_count == 3

    assert limited.frames == [7, 8, 9]

    print("Test 3 passed: bounded trajectory history")

    # ---------------------------------------------------------------
    # Test 4: constant-velocity estimation.
    # ---------------------------------------------------------------

    model = predictor.fit_constant_velocity(
        trajectory
    )

    assert np.allclose(

        model["velocity"],

        [3.0, -2.0],

        atol=1e-8
    )

    print("Test 4 passed: constant-velocity estimation")

    # ---------------------------------------------------------------
    # Test 5: constant-velocity prediction.
    # ---------------------------------------------------------------

    prediction = predictor.predict_constant_velocity(

        trajectory,

        horizon=5
    )

    expected = np.array(
        [58.0, -12.0]
    )

    assert np.allclose(

        prediction["predicted_position"],

        expected,

        atol=1e-8
    )

    print("Test 5 passed: constant-velocity prediction")

    # ---------------------------------------------------------------
    # Test 6: zero residual for perfect linear data.
    # ---------------------------------------------------------------

    assert model["residual_std"] < 1e-8

    print("Test 6 passed: zero linear residual")

    # ---------------------------------------------------------------
    # Test 7: constant acceleration estimation.
    # ---------------------------------------------------------------

    accelerated = make_accelerating_trajectory()

    model = predictor.fit_constant_acceleration(
        accelerated
    )

    assert np.allclose(

        model["velocity"],

        [1.0, 2.0],

        atol=1e-8
    )

    assert np.allclose(

        model["acceleration"],

        [0.5, -0.25],

        atol=1e-8
    )

    print("Test 7 passed: acceleration estimation")

    # ---------------------------------------------------------------
    # Test 8: constant-acceleration prediction.
    # ---------------------------------------------------------------

    prediction = predictor.predict_constant_acceleration(

        accelerated,

        horizon=4
    )

    current = np.array(
        [70.5, 21.25]
    )

    expected = (

        current

        + np.array([1.0, 2.0]) * 4

        + 0.5
        * np.array([0.5, -0.25])
        * 16
    )

    assert np.allclose(

        prediction["predicted_position"],

        expected,

        atol=1e-8
    )

    print("Test 8 passed: constant-acceleration prediction")

    # ---------------------------------------------------------------
    # Test 9: zero quadratic residual.
    # ---------------------------------------------------------------

    assert model["residual_std"] < 1e-8

    print("Test 9 passed: zero quadratic residual")

    # ---------------------------------------------------------------
    # Test 10: linear model on accelerating trajectory.
    # ---------------------------------------------------------------

    linear_prediction = predictor.predict_constant_velocity(

        accelerated,

        horizon=5
    )

    quadratic_prediction = predictor.predict_constant_acceleration(

        accelerated,

        horizon=5
    )

    true_position = (

        np.array([70.5, 21.25])

        + np.array([1.0, 2.0]) * 5

        + 0.5
        * np.array([0.5, -0.25])
        * 25
    )

    linear_error = np.linalg.norm(

        linear_prediction["predicted_position"]

        - true_position
    )

    quadratic_error = np.linalg.norm(

        quadratic_prediction["predicted_position"]

        - true_position
    )

    assert quadratic_error < linear_error

    print("Test 10 passed: quadratic model handles acceleration")

    # ---------------------------------------------------------------
    # Test 11: EMA velocity initialization.
    # ---------------------------------------------------------------

    linear = make_linear_trajectory(
        velocity=(4.0, 2.0)
    )

    assert np.allclose(

        linear.ema_velocity,

        [4.0, 2.0]
    )

    print("Test 11 passed: EMA velocity initialization")

    # ---------------------------------------------------------------
    # Test 12: EMA prediction.
    # ---------------------------------------------------------------

    prediction = predictor.predict_ema(

        linear,

        horizon=3
    )

    expected = (

        linear.latest_position

        + np.array([4.0, 2.0]) * 3
    )

    assert np.allclose(

        prediction["predicted_position"],

        expected
    )

    print("Test 12 passed: EMA prediction")

    # ---------------------------------------------------------------
    # Test 13: EMA smoothing.
    # ---------------------------------------------------------------

    noisy = Trajectory(
        track_id=3
    )

    positions = [

        [0, 0],

        [1, 0],

        [2, 0],

        [10, 0]
    ]

    for frame, position in enumerate(positions):

        noisy.add_observation(
            frame,
            position
        )

    raw_velocity = np.array(
        [8.0, 0.0]
    )

    assert noisy.ema_velocity[0] < raw_velocity[0]

    print("Test 13 passed: EMA smooths velocity changes")

    # ---------------------------------------------------------------
    # Test 14: insufficient observations.
    # ---------------------------------------------------------------

    short = Trajectory(
        track_id=4
    )

    short.add_observation(
        0,
        [0, 0]
    )

    try:

        predictor.fit_constant_velocity(
            short
        )

        raise AssertionError(
            "Expected ValueError."
        )

    except ValueError:

        pass

    print("Test 14 passed: insufficient observations rejected")

    # ---------------------------------------------------------------
    # Test 15: uncertainty increases with horizon.
    # ---------------------------------------------------------------

    uncertainty_short = prediction_uncertainty(

        residual_std=2.0,

        horizon=2
    )

    uncertainty_long = prediction_uncertainty(

        residual_std=2.0,

        horizon=10
    )

    assert uncertainty_long > uncertainty_short

    print("Test 15 passed: uncertainty grows with horizon")

    # ---------------------------------------------------------------
    # Test 16: zero residual uncertainty baseline.
    # ---------------------------------------------------------------

    uncertainty = prediction_uncertainty(

        residual_std=0.0,

        horizon=0
    )

    assert uncertainty == 0.0

    print("Test 16 passed: uncertainty baseline")

    # ---------------------------------------------------------------
    # Test 17: box center.
    # ---------------------------------------------------------------

    center = box_center(
        [10, 20, 50, 80]
    )

    assert np.allclose(
        center,
        [30, 50]
    )

    print("Test 17 passed: bounding-box center")

    # ---------------------------------------------------------------
    # Test 18: box IoU.
    # ---------------------------------------------------------------

    box = np.array(
        [10, 20, 50, 80],
        dtype=np.float64
    )

    assert abs(
        box_iou(box, box) - 1.0
    ) < 1e-8

    print("Test 18 passed: bounding-box IoU")

    # ---------------------------------------------------------------
    # Test 19: trajectory predictor returns all models.
    # ---------------------------------------------------------------

    results = predictor.predict_all(

        accelerated,

        horizon=5
    )

    assert set(results.keys()) == {

        "constant_velocity",

        "constant_acceleration",

        "ema"
    }

    print("Test 19 passed: multi-model prediction")

    # ---------------------------------------------------------------
    # Test 20: prediction dimensions.
    # ---------------------------------------------------------------

    for result in results.values():

        assert result["predicted_position"].shape == (
            2,
        )

    print("Test 20 passed: prediction dimensions")

    # ---------------------------------------------------------------
    # Test 21: rolling-origin evaluation.
    # ---------------------------------------------------------------

    evaluation = evaluate_prediction_error(

        accelerated,

        predictor,

        horizons=(1, 2, 3)
    )

    assert "constant_velocity" in evaluation

    assert "constant_acceleration" in evaluation

    assert "ema" in evaluation

    print("Test 21 passed: rolling-origin evaluation")

    # ---------------------------------------------------------------
    # Test 22: perfect quadratic rolling-origin prediction.
    # ---------------------------------------------------------------

    quadratic_error = evaluation[

        "constant_acceleration"

    ]["1"]["mean_error"]

    assert quadratic_error is not None

    assert quadratic_error < 1e-7

    print("Test 22 passed: quadratic rolling-origin prediction")

    # ---------------------------------------------------------------
    # Test 23: centroid tracker creates an object.
    # ---------------------------------------------------------------

    tracker = CentroidTracker()

    box = np.array(
        [10, 20, 50, 80],
        dtype=np.float64
    )

    tracks = tracker.update(
        [box]
    )

    assert len(tracks) == 1

    assert tracks[0].track_id == 1

    print("Test 23 passed: centroid tracker initialization")

    # ---------------------------------------------------------------
    # Test 24: centroid tracker updates an object.
    # ---------------------------------------------------------------

    moved_box = np.array(
        [15, 20, 55, 80],
        dtype=np.float64
    )

    tracks = tracker.update(
        [moved_box]
    )

    assert len(tracks) == 1

    assert tracks[0].track_id == 1

    assert tracks[0].trajectory.observation_count == 2

    print("Test 24 passed: centroid tracker association")

    # ---------------------------------------------------------------
    # Test 25: centroid tracker creates a second object.
    # ---------------------------------------------------------------

    second_box = np.array(
        [250, 200, 300, 260],
        dtype=np.float64
    )

    tracks = tracker.update(
        [moved_box, second_box]
    )

    assert len(tracks) == 2

    assert len(
        set(t.track_id for t in tracks)
    ) == 2

    print("Test 25 passed: multi-object tracking")

    # ---------------------------------------------------------------
    # Test 26: prediction units.
    # ---------------------------------------------------------------

    fps = 30.0

    horizon = 15

    seconds = horizon / fps

    assert abs(
        seconds - 0.5
    ) < 1e-8

    print("Test 26 passed: frame-to-time conversion")

    # ---------------------------------------------------------------
    # Test 27: noisy trajectory still produces finite prediction.
    # ---------------------------------------------------------------

    noisy_trajectory = make_linear_trajectory(

        velocity=(2.0, 1.0),

        number_of_frames=20,

        noise_std=0.5
    )

    prediction = predictor.predict_constant_velocity(

        noisy_trajectory,

        horizon=5
    )

    assert np.all(
        np.isfinite(
            prediction["predicted_position"]
        )
    )

    print("Test 27 passed: noisy trajectory prediction")

    # ---------------------------------------------------------------
    # Test 28: velocity has pixel-per-frame interpretation.
    # ---------------------------------------------------------------

    model = predictor.fit_constant_velocity(
        linear
    )

    velocity_pixels_per_second = (

        model["velocity"] * 30.0
    )

    assert np.allclose(

        velocity_pixels_per_second,

        [120.0, 60.0]
    )

    print("Test 28 passed: pixel-per-second conversion")

    print("\nAll 28 tests passed.")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(

            "Lesson 52 — Trajectory Prediction "
            "and Future Position Estimation"
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
        default="runs/lesson52"
    )

    parser.add_argument(
        "--history-length",
        type=int,
        default=DEFAULT_HISTORY_LENGTH
    )

    parser.add_argument(
        "--min-observations",
        type=int,
        default=DEFAULT_MIN_OBSERVATIONS
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON
    )

    parser.add_argument(
        "--max-association-distance",
        type=float,
        default=DEFAULT_MAX_ASSOCIATION_DISTANCE
    )

    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=DEFAULT_MAX_MISSED_FRAMES
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=500
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

    if args.history_length < 3:

        parser.error(
            "--history-length must be at least 3."
        )

    if args.min_observations < 3:

        parser.error(
            "--min-observations must be at least 3."
        )

    if args.horizon < 1:

        parser.error(
            "--horizon must be at least 1."
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