"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 53 — TRAJECTORY-BASED OBJECT INTERACTION AND COLLISION PREDICTION
======================================================================

Filename:
    lesson53_collision_prediction.py

======================================================================
LEARNING OBJECTIVES
======================================================================

1. Estimate motion for multiple tracked objects.
2. Understand relative position and relative velocity.
3. Calculate closest point of approach (CPA).
4. Calculate time to closest approach.
5. Estimate time to collision (TTC).
6. Detect potential collisions within a prediction horizon.
7. Account for object size using bounding-circle approximations.
8. Distinguish approaching objects from objects moving apart.
9. Visualize predicted collision points on real video.
10. Evaluate collision prediction using numerical tests.

======================================================================
1. WHY OBJECT INTERACTION MATTERS
======================================================================

Tracking tells us where individual objects are.

Interaction analysis tells us how objects relate to each other.

Examples:

    - Two objects moving toward each other.
    - Two objects moving in parallel.
    - One object overtaking another.
    - Two objects crossing paths.
    - Two objects moving apart.
    - Two objects likely to overlap in the future.

For robotic waste sorting:

    Camera
       |
       v
    Object detection
       |
       v
    Multi-object tracking
       |
       v
    Trajectory estimation
       |
       v
    Object interaction analysis
       |
       v
    Collision / overlap prediction
       |
       v
    Grasp scheduling and motion planning

======================================================================
2. RELATIVE POSITION
======================================================================

For two objects A and B:

    p_A = [x_A, y_A]

    p_B = [x_B, y_B]

Relative position:

    r = p_B - p_A

The distance between their centers is:

    d = ||r||

======================================================================
3. RELATIVE VELOCITY
======================================================================

Let:

    v_A = velocity of object A

    v_B = velocity of object B

Relative velocity:

    v = v_B - v_A

The relative position at future time t is:

    r(t) = r + v*t

This converts a two-object motion problem into a single
relative-motion problem.

======================================================================
4. CLOSEST POINT OF APPROACH (CPA)
======================================================================

The squared center distance is:

    D(t)^2 = ||r + v*t||^2

Expand:

    D(t)^2 =
        (r + v*t).T * (r + v*t)

Differentiate with respect to t:

    d(D^2)/dt = 2*v.T*(r + v*t)

Set derivative to zero:

    v.T*r + t*(v.T*v) = 0

Therefore:

    t_CPA = -(r.T*v) / (v.T*v)

For a finite prediction horizon:

    t_CPA = clip(
        -(r.T*v)/(v.T*v),
        0,
        horizon
    )

Closest predicted center distance:

    d_CPA = ||r + v*t_CPA||

If relative velocity is zero:

    t_CPA = 0

The objects maintain their current relative positions.

======================================================================
5. TIME TO COLLISION (TTC)
======================================================================

Objects have physical size.

We approximate each object by a circle:

    radius_A = object A radius

    radius_B = object B radius

Collision occurs when center distance is no greater than:

    R = radius_A + radius_B

The collision condition is:

    ||r + v*t||^2 <= R^2

Expand:

    (v.v)*t^2 + 2*(r.v)*t + (r.r - R^2) <= 0

Define:

    a = v.v

    b = 2*(r.v)

    c = r.r - R^2

The discriminant:

    Delta = b^2 - 4*a*c

If:

    Delta < 0

there is no collision under constant relative velocity.

Otherwise, roots are:

    t1 = (-b - sqrt(Delta)) / (2*a)

    t2 = (-b + sqrt(Delta)) / (2*a)

The first non-negative root is the predicted collision time,
provided it is within the prediction horizon.

If the objects already overlap:

    TTC = 0

======================================================================
6. OBJECT SIZE APPROXIMATION
======================================================================

For a bounding box:

    width  = x2 - x1

    height = y2 - y1

A conservative enclosing-circle radius is:

    radius = 0.5 * sqrt(width^2 + height^2)

This may overestimate the collision region for elongated objects.

Alternatives include:

    - Oriented bounding boxes.
    - Polygon intersection.
    - Separating Axis Theorem.
    - 3-D collision geometry.
    - Learned object shape representations.

The circle approximation is useful for a fast first implementation.

======================================================================
7. COLLISION VS. CLOSE PASSING
======================================================================

Two objects can approach each other without colliding.

Example:

    Object A center: (0, 0)

    Object B center: (100, 30)

    Combined radius: 20 pixels

Even if their future center distance becomes small, a collision
is predicted only when the center distance reaches the combined
radius.

This distinction matters for robotic planning.

======================================================================
8. IMPORTANT ASSUMPTIONS
======================================================================

This lesson assumes:

    - Both objects move at approximately constant velocity.
    - Motion is estimated in image coordinates.
    - Object shapes are approximated by circles.
    - Both trajectories use the same camera coordinate system.
    - Time is measured in video frames.

A camera moving through the scene can make stationary objects
appear to move. Camera-motion compensation is therefore important.

For a real robot, pixel-space collision predictions are not enough
when objects occupy different depths.

Use calibrated camera geometry or 3-D perception for physical
collision reasoning.

======================================================================
9. REAL ROBOTICS APPLICATION
======================================================================

Imagine two waste objects on a conveyor.

    Cup A: moving toward the robot.

    Bottle B: moving behind the cup.

The system predicts whether their image regions will overlap.

The robot can then:

    - Wait for the objects to separate.
    - Choose a different grasp order.
    - Avoid grasping between overlapping objects.
    - Recalculate a grasp target.
    - Schedule a pick before an object leaves the workspace.

Collision prediction is not a complete robot safety system.
Physical robot safety requires calibrated 3-D geometry, robot
kinematics, swept-volume checking, and controller-level safeguards.

======================================================================
DEPENDENCIES
======================================================================

    pip install numpy opencv-python scipy matplotlib

======================================================================
RUN
======================================================================

    python lesson53_collision_prediction.py --check-only

    python lesson53_collision_prediction.py \
        --video conveyor_video.mp4

    python lesson53_collision_prediction.py \
        --video conveyor_video.mp4 \
        --horizon 30

    python lesson53_collision_prediction.py \
        --video conveyor_video.mp4 \
        --horizon 30 \
        --no-display

======================================================================
OUTPUT
======================================================================

    runs/lesson53/
        collision_prediction.mp4
        collision_summary.json
        collision_geometry.png
        interaction_examples.png
======================================================================
"""

import argparse
import json
import math
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

DEFAULT_HORIZON = 30

DEFAULT_MAX_ASSOCIATION_DISTANCE = 100.0

DEFAULT_MAX_MISSED_FRAMES = 20

DEFAULT_MIN_DETECTION_AREA = 500

EPSILON = 1e-10


# =====================================================================
# GEOMETRY
# =====================================================================

def box_center(box):

    x1, y1, x2, y2 = np.asarray(
        box,
        dtype=np.float64
    )

    return np.array(

        [

            0.5 * (x1 + x2),

            0.5 * (y1 + y2)
        ],

        dtype=np.float64
    )


def box_width(box):

    return max(
        0.0,
        float(box[2] - box[0])
    )


def box_height(box):

    return max(
        0.0,
        float(box[3] - box[1])
    )


def box_radius(box):

    """
    Radius of the smallest circle centered at the box center
    that encloses the entire axis-aligned bounding box.
    """

    return 0.5 * math.sqrt(

        box_width(box) ** 2

        + box_height(box) ** 2
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

    area_a = (

        max(0.0, ax2 - ax1)

        * max(0.0, ay2 - ay1)
    )

    area_b = (

        max(0.0, bx2 - bx1)

        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - intersection

    return intersection / max(
        union,
        EPSILON
    )


# =====================================================================
# TRAJECTORY
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

    def add_observation(

        self,

        frame_index,

        position
    ):

        position = np.asarray(

            position,

            dtype=np.float64
        )

        if position.shape != (2,):

            raise ValueError(
                "Position must have shape (2,)."
            )

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
# VELOCITY ESTIMATION
# =====================================================================

def estimate_velocity(trajectory):

    """
    Estimate velocity using least-squares linear regression.

    x(t) = a_x + v_x*t

    y(t) = a_y + v_y*t

    Time is measured in frames.

    Returns:
        velocity: pixels/frame
    """

    frames, positions = trajectory.arrays()

    if len(frames) < 2:

        raise ValueError(
            "At least two observations are required."
        )

    relative_time = frames - frames[-1]

    if np.ptp(relative_time) <= 0:

        raise ValueError(
            "Trajectory timestamps must not all be identical."
        )

    velocities = []

    for dimension in range(2):

        coefficients = np.polyfit(

            relative_time,

            positions[:, dimension],

            deg=1
        )

        velocities.append(
            coefficients[0]
        )

    return np.asarray(

        velocities,

        dtype=np.float64
    )


# =====================================================================
# CONSTANT-VELOCITY PREDICTION
# =====================================================================

def predict_position(

    position,

    velocity,

    horizon
):

    position = np.asarray(
        position,
        dtype=np.float64
    )

    velocity = np.asarray(
        velocity,
        dtype=np.float64
    )

    return position + velocity * float(horizon)


# =====================================================================
# RELATIVE MOTION
# =====================================================================

@dataclass
class CollisionResult:

    track_a: int

    track_b: int

    initial_distance: float

    combined_radius: float

    relative_position: np.ndarray

    relative_velocity: np.ndarray

    time_to_closest_approach: float

    closest_distance: float

    closest_point_a: np.ndarray

    closest_point_b: np.ndarray

    time_to_collision: float | None

    collision_point: np.ndarray | None

    will_collide: bool

    is_approaching: bool

    prediction_horizon: float

    uncertainty_margin: float = 0.0

    def to_dict(self):

        return {

            "track_a": self.track_a,

            "track_b": self.track_b,

            "initial_distance": self.initial_distance,

            "combined_radius": self.combined_radius,

            "relative_position": self.relative_position.tolist(),

            "relative_velocity": self.relative_velocity.tolist(),

            "time_to_closest_approach": self.time_to_closest_approach,

            "closest_distance": self.closest_distance,

            "closest_point_a": self.closest_point_a.tolist(),

            "closest_point_b": self.closest_point_b.tolist(),

            "time_to_collision": self.time_to_collision,

            "collision_point": (

                self.collision_point.tolist()

                if self.collision_point is not None

                else None
            ),

            "will_collide": self.will_collide,

            "is_approaching": self.is_approaching,

            "prediction_horizon": self.prediction_horizon,

            "uncertainty_margin": self.uncertainty_margin
        }


# =====================================================================
# COLLISION PREDICTOR
# =====================================================================

class CollisionPredictor:

    """
    Predict interactions between two objects under constant velocity.

    The two-object problem is converted to relative motion:

        r(t) = r0 + v_relative*t

    Collision is modeled as overlap between two circles.

    The optional uncertainty margin expands the combined radius.
    It is a conservative heuristic, not a calibrated probability.
    """

    def __init__(

        self,

        horizon=DEFAULT_HORIZON,

        uncertainty_margin=0.0
    ):

        if horizon < 0:

            raise ValueError(
                "Prediction horizon must be non-negative."
            )

        if uncertainty_margin < 0:

            raise ValueError(
                "Uncertainty margin must be non-negative."
            )

        self.horizon = float(
            horizon
        )

        self.uncertainty_margin = float(
            uncertainty_margin
        )

    @staticmethod
    def closest_approach(

        relative_position,

        relative_velocity,

        horizon
    ):

        r = np.asarray(

            relative_position,

            dtype=np.float64
        )

        v = np.asarray(

            relative_velocity,

            dtype=np.float64
        )

        speed_squared = float(
            np.dot(v, v)
        )

        if speed_squared <= EPSILON:

            time_cpa = 0.0

        else:

            time_cpa = -float(
                np.dot(r, v)
            ) / speed_squared

            time_cpa = float(

                np.clip(

                    time_cpa,

                    0.0,

                    horizon
                )
            )

        closest_relative_position = (

            r + v * time_cpa
        )

        distance = float(
            np.linalg.norm(closest_relative_position)
        )

        return time_cpa, distance

    @staticmethod
    def time_to_collision(

        relative_position,

        relative_velocity,

        combined_radius,

        horizon
    ):

        """
        Solve:

            ||r + v*t||^2 = R^2

        Return the earliest non-negative collision time.

        Returns None if no collision occurs within the horizon.
        """

        r = np.asarray(

            relative_position,

            dtype=np.float64
        )

        v = np.asarray(

            relative_velocity,

            dtype=np.float64
        )

        radius = float(
            combined_radius
        )

        a = float(
            np.dot(v, v)
        )

        b = 2.0 * float(
            np.dot(r, v)
        )

        c = float(
            np.dot(r, r) - radius ** 2
        )

        # Already overlapping.

        if c <= 0:

            return 0.0

        # No relative motion.

        if a <= EPSILON:

            return None

        discriminant = b ** 2 - 4.0 * a * c

        if discriminant < 0:

            return None

        sqrt_discriminant = math.sqrt(
            max(discriminant, 0.0)
        )

        root_1 = (

            -b - sqrt_discriminant

        ) / (2.0 * a)

        root_2 = (

            -b + sqrt_discriminant

        ) / (2.0 * a)

        valid_roots = [

            root

            for root in (root_1, root_2)

            if root >= 0
        ]

        if not valid_roots:

            return None

        first_collision = min(
            valid_roots
        )

        if first_collision > horizon:

            return None

        return float(
            first_collision
        )

    def analyze(

        self,

        track_a,

        track_b,

        velocity_a,

        velocity_b,

        radius_a,

        radius_b
    ):

        position_a = np.asarray(

            track_a.latest_position,

            dtype=np.float64
        )

        position_b = np.asarray(

            track_b.latest_position,

            dtype=np.float64
        )

        velocity_a = np.asarray(

            velocity_a,

            dtype=np.float64
        )

        velocity_b = np.asarray(

            velocity_b,

            dtype=np.float64
        )

        relative_position = (
            position_b - position_a
        )

        relative_velocity = (
            velocity_b - velocity_a
        )

        initial_distance = float(
            np.linalg.norm(relative_position)
        )

        combined_radius = (

            float(radius_a)

            + float(radius_b)

            + self.uncertainty_margin
        )

        t_cpa, closest_distance = self.closest_approach(

            relative_position,

            relative_velocity,

            self.horizon
        )

        predicted_a = predict_position(

            position_a,

            velocity_a,

            t_cpa
        )

        predicted_b = predict_position(

            position_b,

            velocity_b,

            t_cpa
        )

        ttc = self.time_to_collision(

            relative_position,

            relative_velocity,

            combined_radius,

            self.horizon
        )

        will_collide = (

            ttc is not None
        )

        collision_point = None

        if will_collide:

            collision_a = predict_position(

                position_a,

                velocity_a,

                ttc
            )

            collision_b = predict_position(

                position_b,

                velocity_b,

                ttc
            )

            collision_point = (

                collision_a + collision_b
            ) / 2.0

        is_approaching = (

            float(
                np.dot(
                    relative_position,
                    relative_velocity
                )
            ) < 0
        )

        return CollisionResult(

            track_a=track_a.track_id,

            track_b=track_b.track_id,

            initial_distance=initial_distance,

            combined_radius=combined_radius,

            relative_position=relative_position,

            relative_velocity=relative_velocity,

            time_to_closest_approach=t_cpa,

            closest_distance=closest_distance,

            closest_point_a=predicted_a,

            closest_point_b=predicted_b,

            time_to_collision=ttc,

            collision_point=collision_point,

            will_collide=will_collide,

            is_approaching=is_approaching,

            prediction_horizon=self.horizon,

            uncertainty_margin=self.uncertainty_margin
        )


# =====================================================================
# OBJECT TRACKER
# =====================================================================

@dataclass
class ObjectTrack:

    track_id: int

    bbox: np.ndarray

    trajectory: Trajectory

    missed_frames: int = 0

    age: int = 1

    active: bool = True

    @property
    def latest_position(self):

        return self.trajectory.latest_position

    @property
    def radius(self):

        return box_radius(
            self.bbox
        )


class CentroidTracker:

    """
    Simple Hungarian-assignment centroid tracker.

    This is a demonstration tracker, not a replacement for SORT,
    ByteTrack, or Deep SORT.

    The collision predictor is independent of this class and can
    accept tracks from another tracking algorithm.
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

    def _create_track(self, bbox):

        bbox = np.asarray(

            bbox,

            dtype=np.float64
        )

        track_id = self.next_id

        self.next_id += 1

        trajectory = Trajectory(

            track_id=track_id,

            history_length=self.history_length
        )

        trajectory.add_observation(

            self.frame_index,

            box_center(bbox)
        )

        self.tracks[track_id] = ObjectTrack(

            track_id=track_id,

            bbox=bbox.copy(),

            trajectory=trajectory
        )

    @staticmethod
    def _predict_center(track):

        trajectory = track.trajectory

        if trajectory.observation_count < 2:

            return trajectory.latest_position

        velocity = estimate_velocity(
            trajectory
        )

        return predict_position(

            trajectory.latest_position,

            velocity,

            1.0
        )

    def update(self, detections):

        self.frame_index += 1

        active_tracks = [

            track

            for track in self.tracks.values()

            if track.active
        ]

        matched_tracks = set()

        matched_detections = set()

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

                predicted = self._predict_center(
                    track
                )

                for j, bbox in enumerate(detections):

                    center = box_center(
                        bbox
                    )

                    distance = np.linalg.norm(

                        predicted - center
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

            for row, col in zip(rows, cols):

                if not np.isfinite(cost[row, col]):
                    continue

                track = active_tracks[row]

                bbox = np.asarray(

                    detections[col],

                    dtype=np.float64
                )

                track.bbox = bbox.copy()

                track.missed_frames = 0

                track.age += 1

                track.trajectory.add_observation(

                    self.frame_index,

                    box_center(bbox)
                )

                matched_tracks.add(row)

                matched_detections.add(col)

        for index, track in enumerate(active_tracks):

            if index not in matched_tracks:

                track.missed_frames += 1

                if track.missed_frames > self.max_missed_frames:

                    track.active = False

        for index, bbox in enumerate(detections):

            if index not in matched_detections:

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
    MOG2 foreground detector.

    Best suited to a mostly stationary camera and moving objects
    against a relatively stable background.

    It does not classify waste categories.
    """

    def __init__(

        self,

        min_area=DEFAULT_MIN_DETECTION_AREA
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

            x, y, width, height = cv2.boundingRect(
                contour
            )

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

        return detections, mask


# =====================================================================
# DRAWING
# =====================================================================

def draw_cross(

    image,

    point,

    color,

    size=8,

    thickness=2
):

    x, y = np.round(
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


def draw_interaction(

    frame,

    track_a,

    track_b,

    result
):

    output = frame.copy()

    color = (

        (0, 0, 255)

        if result.will_collide

        else (0, 220, 255)

        if result.is_approaching

        else (150, 150, 150)
    )

    center_a = track_a.latest_position

    center_b = track_b.latest_position

    point_a = tuple(
        np.round(center_a).astype(int)
    )

    point_b = tuple(
        np.round(center_b).astype(int)
    )

    cv2.line(

        output,

        point_a,

        point_b,

        color,

        2
    )

    if result.will_collide:

        draw_cross(

            output,

            result.collision_point,

            (0, 0, 255),

            size=12,

            thickness=3
        )

        ttc_text = (
            f"TTC={result.time_to_collision:.1f} frames"
        )

    else:

        ttc_text = "No collision predicted"

    cpa_text = (

        f"CPA={result.closest_distance:.1f}px "
        f"at {result.time_to_closest_approach:.1f}f"
    )

    midpoint = (

        center_a + center_b
    ) / 2.0

    text_position = tuple(

        np.round(midpoint).astype(int)
    )

    cv2.putText(

        output,

        f"IDs {track_a.track_id}-{track_b.track_id}",

        (text_position[0], max(18, text_position[1] - 30)),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        color,

        1
    )

    cv2.putText(

        output,

        ttc_text,

        (text_position[0], max(36, text_position[1] - 12)),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.42,

        color,

        1
    )

    cv2.putText(

        output,

        cpa_text,

        (text_position[0], text_position[1] + 12),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.42,

        color,

        1
    )

    return output


# =====================================================================
# INTERACTION ANALYSIS
# =====================================================================

def analyze_all_pairs(

    tracks,

    predictor,

    min_observations=DEFAULT_MIN_OBSERVATIONS
):

    """
    Analyze every unordered pair of sufficiently observed tracks.

    Each pair appears exactly once.
    """

    results = []

    eligible = [

        track

        for track in tracks

        if (

            track.trajectory.observation_count

            >= min_observations
        )
    ]

    for i in range(len(eligible)):

        for j in range(i + 1, len(eligible)):

            track_a = eligible[i]

            track_b = eligible[j]

            try:

                velocity_a = estimate_velocity(

                    track_a.trajectory
                )

                velocity_b = estimate_velocity(

                    track_b.trajectory
                )

            except ValueError:

                continue

            result = predictor.analyze(

                track_a,

                track_b,

                velocity_a,

                velocity_b,

                track_a.radius,

                track_b.radius
            )

            results.append(
                result
            )

    return results


# =====================================================================
# NUMERICAL EXAMPLE GENERATION
# =====================================================================

def make_track(

    track_id,

    position,

    velocity,

    radius=5.0,

    history_length=20,

    number_of_observations=10
):

    """
    Build a test track with a constant-velocity history.

    The radius is represented by a square bounding box.
    """

    position = np.asarray(

        position,

        dtype=np.float64
    )

    velocity = np.asarray(

        velocity,

        dtype=np.float64
    )

    trajectory = Trajectory(

        track_id=track_id,

        history_length=history_length
    )

    for frame in range(number_of_observations):

        observed_position = (

            position

            + velocity * (frame - number_of_observations + 1)
        )

        trajectory.add_observation(

            frame,

            observed_position
        )

    half = float(radius) / math.sqrt(2.0)

    current = trajectory.latest_position

    bbox = np.array(

        [

            current[0] - half,

            current[1] - half,

            current[0] + half,

            current[1] + half
        ],

        dtype=np.float64
    )

    return ObjectTrack(

        track_id=track_id,

        bbox=bbox,

        trajectory=trajectory
    )


# =====================================================================
# PLOTS
# =====================================================================

def plot_collision_geometry(
    output_path
):

    predictor = CollisionPredictor(
        horizon=30
    )

    scenarios = [

        {

            "name": "Head-on collision",

            "a": make_track(
                1,
                [0, 0],
                [2, 0],
                radius=5
            ),

            "b": make_track(
                2,
                [100, 0],
                [-2, 0],
                radius=5
            )
        },

        {

            "name": "Parallel motion",

            "a": make_track(
                3,
                [0, 0],
                [2, 0],
                radius=5
            ),

            "b": make_track(
                4,
                [0, 30],
                [2, 0],
                radius=5
            )
        },

        {

            "name": "Crossing paths",

            "a": make_track(
                5,
                [0, 0],
                [2, 0],
                radius=5
            ),

            "b": make_track(
                6,
                [50, -50],
                [0, 2],
                radius=5
            )
        }
    ]

    fig, axes = plt.subplots(

        1,

        3,

        figsize=(16, 5)
    )

    for axis, scenario in zip(axes, scenarios):

        a = scenario["a"]

        b = scenario["b"]

        velocity_a = estimate_velocity(
            a.trajectory
        )

        velocity_b = estimate_velocity(
            b.trajectory
        )

        result = predictor.analyze(

            a,

            b,

            velocity_a,

            velocity_b,

            a.radius,

            b.radius
        )

        for track, velocity, label in [

            (a, velocity_a, "Object A"),

            (b, velocity_b, "Object B")
        ]:

            position = track.latest_position

            future = predict_position(

                position,

                velocity,

                predictor.horizon
            )

            axis.plot(

                [position[0], future[0]],

                [position[1], future[1]],

                "--",

                label=label
            )

            axis.scatter(

                [position[0]],

                [position[1]],

                s=70
            )

            circle = plt.Circle(

                position,

                track.radius,

                fill=False,

                alpha=0.7
            )

            axis.add_patch(
                circle
            )

        if result.will_collide:

            axis.scatter(

                [result.collision_point[0]],

                [result.collision_point[1]],

                marker="X",

                s=130,

                label="Predicted collision"
            )

        axis.set_title(
            scenario["name"]
        )

        axis.set_xlabel(
            "X (pixels)"
        )

        axis.set_ylabel(
            "Y (pixels)"
        )

        axis.grid(
            True,
            alpha=0.3
        )

        axis.legend()

        axis.set_aspect(
            "equal",
            adjustable="datalim"
        )

    fig.suptitle(
        "Object Interaction and Collision Prediction"
    )

    fig.tight_layout()

    fig.savefig(

        output_path,

        dpi=150
    )

    plt.close(fig)


def plot_interaction_examples(
    output_path
):

    predictor = CollisionPredictor(
        horizon=30
    )

    scenarios = [

        (

            "Approaching",

            [0, 0],

            [2, 0],

            [100, 0],

            [-2, 0]
        ),

        (

            "Moving apart",

            [0, 0],

            [-2, 0],

            [30, 0],

            [2, 0]
        ),

        (

            "Parallel",

            [0, 0],

            [2, 0],

            [0, 30],

            [2, 0]
        )
    ]

    fig, axes = plt.subplots(

        1,

        3,

        figsize=(15, 4)
    )

    for axis, scenario in zip(axes, scenarios):

        name, pa, va, pb, vb = scenario

        a = make_track(
            1,
            pa,
            va
        )

        b = make_track(
            2,
            pb,
            vb
        )

        result = predictor.analyze(

            a,

            b,

            estimate_velocity(a.trajectory),

            estimate_velocity(b.trajectory),

            a.radius,

            b.radius
        )

        axis.scatter(

            [pa[0], pb[0]],

            [pa[1], pb[1]],

            s=90,

            label="Current positions"
        )

        for position, velocity in [(pa, va), (pb, vb)]:

            future = predict_position(

                position,

                velocity,

                20
            )

            axis.plot(

                [position[0], future[0]],

                [position[1], future[1]],

                "--"
            )

        axis.set_title(

            f"{name}\n"
            f"Approaching: {result.is_approaching}\n"
            f"Collision: {result.will_collide}"
        )

        axis.set_xlabel(
            "X"
        )

        axis.set_ylabel(
            "Y"
        )

        axis.grid(
            True,
            alpha=0.3
        )

        axis.legend()

    fig.tight_layout()

    fig.savefig(

        output_path,

        dpi=150
    )

    plt.close(fig)


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

        output_dir / "collision_prediction.mp4"
    )

    output_json = (

        output_dir / "collision_summary.json"
    )

    geometry_plot = (

        output_dir / "collision_geometry.png"
    )

    examples_plot = (

        output_dir / "interaction_examples.png"
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

    predictor = CollisionPredictor(

        horizon=args.horizon,

        uncertainty_margin=args.uncertainty_margin
    )

    processing_times = []

    collision_events = []

    try:

        while True:

            start = time.perf_counter()

            ret, frame = capture.read()

            if not ret:
                break

            detections, _ = detector.detect(
                frame
            )

            tracks = tracker.update(
                detections
            )

            output = frame.copy()

            # Draw each active object.

            for track in tracks:

                x1, y1, x2, y2 = track.bbox.astype(
                    int
                )

                center = track.latest_position

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

                cv2.circle(

                    output,

                    tuple(np.round(center).astype(int)),

                    4,

                    color,

                    -1
                )

                cv2.putText(

                    output,

                    f"ID {track.track_id}",

                    (x1, max(20, y1 - 8)),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    color,

                    2
                )

            # Analyze all sufficiently observed pairs.

            pair_results = analyze_all_pairs(

                tracks,

                predictor,

                min_observations=args.min_observations
            )

            active_by_id = {

                track.track_id: track

                for track in tracks
            }

            for result in pair_results:

                track_a = active_by_id[
                    result.track_a
                ]

                track_b = active_by_id[
                    result.track_b
                ]

                output = draw_interaction(

                    output,

                    track_a,

                    track_b,

                    result
                )

                if result.will_collide:

                    collision_events.append({

                        "frame": tracker.frame_index,

                        **result.to_dict()
                    })

            cv2.putText(

                output,

                f"Frame {tracker.frame_index}",

                (15, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (255, 255, 255),

                2
            )

            cv2.putText(

                output,

                f"Active tracks: {len(tracks)}",

                (15, 58),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                2
            )

            cv2.putText(

                output,

                f"Prediction horizon: {args.horizon} frames",

                (15, 84),

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

                    "Lesson 53 - Collision Prediction",

                    output
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

    finally:

        capture.release()

        writer.release()

        cv2.destroyAllWindows()

    plot_collision_geometry(
        geometry_plot
    )

    plot_interaction_examples(
        examples_plot
    )

    # ---------------------------------------------------------------
    # Summarize unique predicted pairs.
    # ---------------------------------------------------------------

    unique_pairs = sorted(

        {

            (

                event["track_a"],

                event["track_b"]
            )

            for event in collision_events
        }
    )

    summary = {

        "input_video": str(video_path),

        "output_video": str(output_video),

        "fps": fps,

        "frame_count": tracker.frame_index,

        "track_count": len(tracker.tracks),

        "prediction_horizon_frames": args.horizon,

        "prediction_horizon_seconds": args.horizon / fps,

        "uncertainty_margin_pixels": args.uncertainty_margin,

        "average_processing_fps": (

            1.0 / np.mean(processing_times)

            if processing_times and np.mean(processing_times) > 0

            else 0.0
        ),

        "collision_event_count": len(collision_events),

        "unique_predicted_collision_pairs": [

            list(pair)

            for pair in unique_pairs
        ],

        "collision_events": collision_events,

        "tracks": [

            {

                "track_id": track.track_id,

                "observation_count": track.trajectory.observation_count,

                "positions": track.trajectory.positions,

                "frames": track.trajectory.frames,

                "last_bbox": track.bbox.tolist()
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
        "Collision predictions:",
        len(collision_events)
    )

    print(
        "Unique predicted pairs:",
        unique_pairs
    )

    print(
        "Average processing FPS:",
        summary["average_processing_fps"]
    )

    print("\nOutput video:", output_video)

    print("Summary:", output_json)

    print("Collision geometry:", geometry_plot)

    print("Interaction examples:", examples_plot)


# =====================================================================
# NUMERICAL TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print(
        "LESSON 53 — COLLISION PREDICTION TESTS"
    )

    print("=" * 70)

    predictor = CollisionPredictor(
        horizon=30
    )

    # ---------------------------------------------------------------
    # Test 1: bounding-box center.
    # ---------------------------------------------------------------

    center = box_center(
        [10, 20, 50, 80]
    )

    assert np.allclose(
        center,
        [30, 50]
    )

    print("Test 1 passed: bounding-box center")

    # ---------------------------------------------------------------
    # Test 2: bounding-circle radius.
    # ---------------------------------------------------------------

    radius = box_radius(
        [0, 0, 6, 8]
    )

    assert abs(radius - 5.0) < 1e-8

    print("Test 2 passed: bounding-circle radius")

    # ---------------------------------------------------------------
    # Test 3: box IoU.
    # ---------------------------------------------------------------

    box = np.array(
        [10, 10, 30, 30],
        dtype=np.float64
    )

    assert abs(box_iou(box, box) - 1.0) < 1e-8

    print("Test 3 passed: box IoU")

    # ---------------------------------------------------------------
    # Test 4: trajectory storage.
    # ---------------------------------------------------------------

    trajectory = Trajectory(
        track_id=1
    )

    for frame in range(5):

        trajectory.add_observation(

            frame,

            [2 * frame, 3 * frame]
        )

    assert trajectory.observation_count == 5

    assert np.allclose(

        trajectory.latest_position,

        [8, 12]
    )

    print("Test 4 passed: trajectory storage")

    # ---------------------------------------------------------------
    # Test 5: bounded history.
    # ---------------------------------------------------------------

    limited = Trajectory(

        track_id=2,

        history_length=3
    )

    for frame in range(10):

        limited.add_observation(

            frame,

            [frame, 0]
        )

    assert limited.frames == [7, 8, 9]

    print("Test 5 passed: bounded trajectory history")

    # ---------------------------------------------------------------
    # Test 6: least-squares velocity.
    # ---------------------------------------------------------------

    track = make_track(

        1,

        [10, 20],

        [3, -2]
    )

    velocity = estimate_velocity(
        track.trajectory
    )

    assert np.allclose(

        velocity,

        [3, -2],

        atol=1e-8
    )

    print("Test 6 passed: least-squares velocity")

    # ---------------------------------------------------------------
    # Test 7: constant-velocity prediction.
    # ---------------------------------------------------------------

    future = predict_position(

        [10, 20],

        [3, -2],

        5
    )

    assert np.allclose(
        future,
        [25, 10]
    )

    print("Test 7 passed: constant-velocity prediction")

    # ---------------------------------------------------------------
    # Test 8: relative position.
    # ---------------------------------------------------------------

    relative = np.array(
        [100, 0]
    ) - np.array(
        [0, 0]
    )

    assert np.allclose(
        relative,
        [100, 0]
    )

    print("Test 8 passed: relative position")

    # ---------------------------------------------------------------
    # Test 9: relative velocity.
    # ---------------------------------------------------------------

    relative_velocity = (

        np.array([-2, 0])

        - np.array([2, 0])
    )

    assert np.allclose(

        relative_velocity,

        [-4, 0]
    )

    print("Test 9 passed: relative velocity")

    # ---------------------------------------------------------------
    # Test 10: closest point of approach.
    # ---------------------------------------------------------------

    t_cpa, distance = predictor.closest_approach(

        relative_position=[100, 0],

        relative_velocity=[-4, 0],

        horizon=30
    )

    assert abs(t_cpa - 25.0) < 1e-8

    assert abs(distance) < 1e-8

    print("Test 10 passed: closest point of approach")

    # ---------------------------------------------------------------
    # Test 11: CPA with a non-colliding near pass.
    # ---------------------------------------------------------------

    t_cpa, distance = predictor.closest_approach(

        relative_position=[100, 30],

        relative_velocity=[-4, 0],

        horizon=30
    )

    assert abs(t_cpa - 25.0) < 1e-8

    assert abs(distance - 30.0) < 1e-8

    print("Test 11 passed: closest approach with lateral offset")

    # ---------------------------------------------------------------
    # Test 12: head-on collision.
    # ---------------------------------------------------------------

    ttc = predictor.time_to_collision(

        relative_position=[100, 0],

        relative_velocity=[-4, 0],

        combined_radius=10,

        horizon=30
    )

    assert ttc is not None

    assert abs(ttc - 22.5) < 1e-8

    print("Test 12 passed: head-on time to collision")

    # ---------------------------------------------------------------
    # Test 13: no collision due to lateral separation.
    # ---------------------------------------------------------------

    ttc = predictor.time_to_collision(

        relative_position=[100, 30],

        relative_velocity=[-4, 0],

        combined_radius=10,

        horizon=30
    )

    assert ttc is None

    print("Test 13 passed: no collision during near pass")

    # ---------------------------------------------------------------
    # Test 14: moving apart.
    # ---------------------------------------------------------------

    ttc = predictor.time_to_collision(

        relative_position=[30, 0],

        relative_velocity=[2, 0],

        combined_radius=10,

        horizon=30
    )

    assert ttc is None

    print("Test 14 passed: objects moving apart")

    # ---------------------------------------------------------------
    # Test 15: already overlapping.
    # ---------------------------------------------------------------

    ttc = predictor.time_to_collision(

        relative_position=[5, 0],

        relative_velocity=[0, 0],

        combined_radius=10,

        horizon=30
    )

    assert ttc == 0.0

    print("Test 15 passed: already-overlapping objects")

    # ---------------------------------------------------------------
    # Test 16: zero relative velocity without overlap.
    # ---------------------------------------------------------------

    ttc = predictor.time_to_collision(

        relative_position=[50, 0],

        relative_velocity=[0, 0],

        combined_radius=10,

        horizon=30
    )

    assert ttc is None

    print("Test 16 passed: zero relative velocity")

    # ---------------------------------------------------------------
    # Test 17: collision outside horizon.
    # ---------------------------------------------------------------

    ttc = predictor.time_to_collision(

        relative_position=[200, 0],

        relative_velocity=[-2, 0],

        combined_radius=10,

        horizon=30
    )

    assert ttc is None

    print("Test 17 passed: collision outside prediction horizon")

    # ---------------------------------------------------------------
    # Test 18: uncertainty margin.
    # ---------------------------------------------------------------

    ttc_without_margin = predictor.time_to_collision(

        relative_position=[100, 0],

        relative_velocity=[-4, 0],

        combined_radius=10,

        horizon=30
    )

    ttc_with_margin = predictor.time_to_collision(

        relative_position=[100, 0],

        relative_velocity=[-4, 0],

        combined_radius=20,

        horizon=30
    )

    assert ttc_with_margin < ttc_without_margin

    print("Test 18 passed: uncertainty margin expands collision region")

    # ---------------------------------------------------------------
    # Test 19: full collision analysis.
    # ---------------------------------------------------------------

    track_a = make_track(

        1,

        [0, 0],

        [2, 0],

        radius=5
    )

    track_b = make_track(

        2,

        [100, 0],

        [-2, 0],

        radius=5
    )

    result = predictor.analyze(

        track_a,

        track_b,

        estimate_velocity(track_a.trajectory),

        estimate_velocity(track_b.trajectory),

        track_a.radius,

        track_b.radius
    )

    assert result.will_collide

    assert result.is_approaching

    assert result.time_to_collision is not None

    print("Test 19 passed: complete collision analysis")

    # ---------------------------------------------------------------
    # Test 20: collision point.
    # ---------------------------------------------------------------

    assert result.collision_point is not None

    assert result.collision_point.shape == (2,)

    assert np.allclose(

        result.collision_point,

        [50, 0],

        atol=1e-8
    )

    print("Test 20 passed: collision point estimation")

    # ---------------------------------------------------------------
    # Test 21: parallel movement.
    # ---------------------------------------------------------------

    parallel_a = make_track(

        1,

        [0, 0],

        [2, 0]
    )

    parallel_b = make_track(

        2,

        [0, 30],

        [2, 0]
    )

    parallel_result = predictor.analyze(

        parallel_a,

        parallel_b,

        estimate_velocity(parallel_a.trajectory),

        estimate_velocity(parallel_b.trajectory),

        parallel_a.radius,

        parallel_b.radius
    )

    assert not parallel_result.will_collide

    assert not parallel_result.is_approaching

    print("Test 21 passed: parallel motion")

    # ---------------------------------------------------------------
    # Test 22: crossing paths.
    # ---------------------------------------------------------------

    crossing_a = make_track(

        1,

        [0, 0],

        [2, 0]
    )

    crossing_b = make_track(

        2,

        [50, -50],

        [0, 2]
    )

    crossing_result = predictor.analyze(

        crossing_a,

        crossing_b,

        estimate_velocity(crossing_a.trajectory),

        estimate_velocity(crossing_b.trajectory),

        crossing_a.radius,

        crossing_b.radius
    )

    assert crossing_result.will_collide

    print("Test 22 passed: crossing trajectories")

    # ---------------------------------------------------------------
    # Test 23: non-colliding close pass.
    # ---------------------------------------------------------------

    near_a = make_track(

        1,

        [0, 0],

        [2, 0],

        radius=5
    )

    near_b = make_track(

        2,

        [50, -30],

        [0, 2],

        radius=5
    )

    near_result = predictor.analyze(

        near_a,

        near_b,

        estimate_velocity(near_a.trajectory),

        estimate_velocity(near_b.trajectory),

        near_a.radius,

        near_b.radius
    )

    assert near_result.closest_distance > near_result.combined_radius

    assert not near_result.will_collide

    print("Test 23 passed: close pass without collision")

    # ---------------------------------------------------------------
    # Test 24: time-to-collision respects horizon.
    # ---------------------------------------------------------------

    short_predictor = CollisionPredictor(
        horizon=10
    )

    ttc = short_predictor.time_to_collision(

        relative_position=[100, 0],

        relative_velocity=[-4, 0],

        combined_radius=10,

        horizon=short_predictor.horizon
    )

    assert ttc is None

    print("Test 24 passed: horizon limit")

    # ---------------------------------------------------------------
    # Test 25: collision result serialization.
    # ---------------------------------------------------------------

    result_dict = result.to_dict()

    assert result_dict["will_collide"] is True

    assert "relative_velocity" in result_dict

    assert "collision_point" in result_dict

    print("Test 25 passed: result serialization")

    # ---------------------------------------------------------------
    # Test 26: tracker initialization.
    # ---------------------------------------------------------------

    tracker = CentroidTracker()

    tracks = tracker.update([

        np.array(
            [10, 10, 30, 30],
            dtype=np.float64
        )

    ])

    assert len(tracks) == 1

    assert tracks[0].track_id == 1

    print("Test 26 passed: tracker initialization")

    # ---------------------------------------------------------------
    # Test 27: tracker association.
    # ---------------------------------------------------------------

    tracks = tracker.update([

        np.array(
            [12, 10, 32, 30],
            dtype=np.float64
        )

    ])

    assert len(tracks) == 1

    assert tracks[0].track_id == 1

    assert tracks[0].trajectory.observation_count == 2

    print("Test 27 passed: tracker association")

    # ---------------------------------------------------------------
    # Test 28: tracker creates a second object.
    # ---------------------------------------------------------------

    tracks = tracker.update([

        np.array(
            [14, 10, 34, 30],
            dtype=np.float64
        ),

        np.array(
            [250, 200, 280, 230],
            dtype=np.float64
        )

    ])

    assert len(tracks) == 2

    assert len({

        track.track_id

        for track in tracks

    }) == 2

    print("Test 28 passed: multiple tracked objects")

    # ---------------------------------------------------------------
    # Test 29: multiple-pair analysis.
    # ---------------------------------------------------------------

    pair_results = analyze_all_pairs(

        [track_a, track_b, parallel_a],

        predictor,

        min_observations=5
    )

    assert len(pair_results) == 3

    print("Test 29 passed: all unordered object pairs")

    # ---------------------------------------------------------------
    # Test 30: frame-to-seconds conversion.
    # ---------------------------------------------------------------

    fps = 30.0

    horizon = 15

    seconds = horizon / fps

    assert abs(seconds - 0.5) < 1e-8

    print("Test 30 passed: frame-to-time conversion")

    print("\nAll 30 tests passed.")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(

            "Lesson 53 — Trajectory-Based Object Interaction "
            "and Collision Prediction"
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

        default="runs/lesson53"
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

        "--uncertainty-margin",

        type=float,

        default=0.0
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

        default=DEFAULT_MIN_DETECTION_AREA
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

    if args.uncertainty_margin < 0:

        parser.error(
            "--uncertainty-margin must be non-negative."
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