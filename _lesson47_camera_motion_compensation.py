"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 47 — CAMERA MOTION COMPENSATION FOR OBJECT TRACKING
======================================================================

Filename:
    lesson47_camera_motion_compensation.py

PHASE:
    Phase 4 — Tracking and Visual Memory

======================================================================
LEARNING OBJECTIVES
======================================================================

1. Understand global camera motion.
2. Extract feature points between consecutive frames.
3. Match image features using optical flow.
4. Estimate a global homography using RANSAC.
5. Compensate object bounding boxes for camera motion.
6. Separate camera motion from independent object motion.
7. Compare tracking with and without compensation.
8. Visualize camera movement and compensated trajectories.
9. Process real videos.

======================================================================
1. THE CAMERA MOTION PROBLEM
======================================================================

Suppose an object is stationary in the world.

Frame 1:
    Object center = (200, 150)

Frame 2:
    Object center = (220, 150)

The apparent movement is:

    dx = 20 pixels
    dy = 0 pixels

But perhaps the camera moved 20 pixels to the left.

The object did not move in world coordinates.

Therefore:

    Image motion = Object motion + Camera motion

A tracker that ignores camera motion may produce incorrect
object trajectories.

======================================================================
2. GLOBAL CAMERA MOTION
======================================================================

For a planar scene or approximately pure camera rotation,
the relationship between two frames can be modeled by:

    p2 ~ H * p1

Where:

    p1 = [x1, y1, 1]^T
    p2 = [x2, y2, 1]^T

    H = 3x3 homography matrix

Expanded:

    x2 = (h11*x1 + h12*y1 + h13)
         / (h31*x1 + h32*y1 + h33)

    y2 = (h21*x1 + h22*y1 + h23)
         / (h31*x1 + h32*y1 + h33)

A homography has eight independent parameters because its scale
is arbitrary.

It can model:

    - Translation
    - Rotation
    - Scale changes
    - Perspective transformation

A single homography cannot accurately model arbitrary 3D scene
motion with significant parallax.

======================================================================
3. FEATURE MATCHING
======================================================================

We extract good feature points in the previous frame:

    cv2.goodFeaturesToTrack()

Then track them in the next frame:

    cv2.calcOpticalFlowPyrLK()

The resulting correspondences are:

    previous_points <-> current_points

We use these correspondences to estimate global camera movement.

======================================================================
4. RANSAC HOMOGRAPHY
======================================================================

Some tracked points may belong to moving objects.

They should not influence the global camera-motion estimate.

RANSAC:

    1. Randomly select four point correspondences.
    2. Estimate a homography.
    3. Transform all previous points.
    4. Calculate reprojection errors.
    5. Count inliers.
    6. Repeat.
    7. Select the best model.

OpenCV:

    cv2.findHomography(
        previous_points,
        current_points,
        cv2.RANSAC
    )

We use this established algorithm rather than implementing a
production-grade RANSAC solver from scratch.

======================================================================
5. COMPENSATING OBJECT POSITIONS
======================================================================

Suppose a track has a previous-frame center:

    p_previous

The predicted camera-only location in the current frame is:

    p_camera = H * p_previous

This is where the object would appear if it were stationary
in the scene.

The measured current location is:

    p_current

Independent object displacement is approximately:

    d_object = p_current - p_camera

This is the key idea behind camera-motion compensation.

======================================================================
6. TRACKING COORDINATE SYSTEMS
======================================================================

Two useful coordinate systems:

IMAGE COORDINATES:
    Coordinates directly measured in each camera frame.

COMPENSATED COORDINATES:
    Coordinates after removing estimated global camera motion.

The compensated coordinates are useful for estimating independent
object movement.

For robotics, camera compensation does NOT replace camera-to-robot
extrinsic calibration.

======================================================================
7. IMPORTANT LIMITATIONS
======================================================================

Homography compensation works best when:

    - The scene is approximately planar.
    - Camera rotation dominates.
    - Camera translation is small relative to scene depth.
    - Enough background features are visible.

It may fail when:

    - There is strong parallax.
    - Most features belong to moving objects.
    - The scene has little texture.
    - Motion blur is severe.
    - The camera changes viewpoint significantly.

For a robot-mounted RGB-D camera, depth-aware 3D motion compensation
can be more appropriate.

======================================================================
8. ROBOTICS APPLICATION
======================================================================

OpenArm V2:

    Moving wrist camera
          |
          v
    Camera feature tracking
          |
          v
    Global camera-motion estimation
          |
          v
    Motion compensation
          |
          v
    Independent object movement
          |
          v
    Object tracking
          |
          v
    Grasp-point estimation

======================================================================
INSTALL
======================================================================

    pip install numpy opencv-python matplotlib

======================================================================
RUN
======================================================================

    python lesson47_camera_motion_compensation.py --check-only

    python lesson47_camera_motion_compensation.py \
        --video robot_camera.mp4

======================================================================
OUTPUT
======================================================================

    runs/lesson47/
        camera_motion_compensation.mp4
        camera_motion_summary.json
======================================================================
"""

import argparse
import json
import time

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# CONFIGURATION
# =====================================================================

MAX_FEATURES = 1500

QUALITY_LEVEL = 0.01

MIN_DISTANCE = 8

FLOW_WINDOW = 21

PYRAMID_LEVELS = 3

RANSAC_REPROJECTION_THRESHOLD = 3.0

MIN_HOMOGRAPHY_INLIERS = 8


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
        ]
    )


def transform_points(points, homography):

    """
    Apply a 3x3 homography to 2D points.

    Input:
        points: (N, 2)

    Output:
        transformed points: (N, 2)

    Uses homogeneous coordinates.
    """

    points = np.asarray(
        points,
        dtype=np.float64
    )

    if points.ndim != 2 or points.shape[1] != 2:

        raise ValueError(
            "points must have shape (N, 2)"
        )

    if homography.shape != (3, 3):

        raise ValueError(
            "homography must have shape (3, 3)"
        )

    homogeneous = np.concatenate(

        [
            points,

            np.ones(
                (len(points), 1),
                dtype=np.float64
            )
        ],

        axis=1
    )

    transformed = (
        homography @ homogeneous.T
    ).T

    denominator = transformed[:, 2]

    valid = np.abs(
        denominator
    ) > 1e-10

    result = np.full(
        (len(points), 2),
        np.nan,
        dtype=np.float64
    )

    result[valid] = (

        transformed[valid, :2]

        / denominator[valid, None]
    )

    return result


def transform_bbox(bbox, homography):

    """
    Transform all four bounding-box corners.

    Transforming only the center is insufficient when the
    homography contains rotation or perspective effects.
    """

    x1, y1, x2, y2 = np.asarray(
        bbox,
        dtype=np.float64
    )

    corners = np.array(

        [
            [x1, y1],

            [x2, y1],

            [x2, y2],

            [x1, y2]
        ],

        dtype=np.float64
    )

    transformed = transform_points(
        corners,
        homography
    )

    if not np.isfinite(transformed).all():

        return np.asarray(
            bbox,
            dtype=np.float64
        ).copy()

    minimum = transformed.min(
        axis=0
    )

    maximum = transformed.max(
        axis=0
    )

    return np.array(

        [
            minimum[0],

            minimum[1],

            maximum[0],

            maximum[1]
        ],

        dtype=np.float64
    )


# =====================================================================
# HOMOGRAPHY ESTIMATION
# =====================================================================

def estimate_homography(
    previous_points,
    current_points,
    reprojection_threshold=RANSAC_REPROJECTION_THRESHOLD,
    min_inliers=MIN_HOMOGRAPHY_INLIERS
):

    """
    Estimate global camera motion.

    H maps:

        previous frame coordinates
                |
                v
        current frame coordinates

    OpenCV uses RANSAC to reject incorrect correspondences.
    """

    previous_points = np.asarray(
        previous_points,
        dtype=np.float32
    ).reshape(
        -1,
        2
    )

    current_points = np.asarray(
        current_points,
        dtype=np.float32
    ).reshape(
        -1,
        2
    )

    if len(previous_points) < 4:

        return (
            np.eye(3, dtype=np.float64),

            np.zeros(
                len(previous_points),
                dtype=bool
            ),

            False
        )

    if len(previous_points) != len(current_points):

        raise ValueError(
            "Point correspondence counts must match."
        )

    H, mask = cv2.findHomography(

        previous_points,

        current_points,

        method=cv2.RANSAC,

        ransacReprojThreshold=reprojection_threshold,

        maxIters=3000,

        confidence=0.995
    )

    if H is None or mask is None:

        return (
            np.eye(3, dtype=np.float64),

            np.zeros(
                len(previous_points),
                dtype=bool
            ),

            False
        )

    H = np.asarray(
        H,
        dtype=np.float64
    )

    inliers = mask.reshape(-1).astype(bool)

    valid = (

        np.isfinite(H).all()

        and np.sum(inliers) >= min_inliers

        and abs(np.linalg.det(H)) > 1e-10
    )

    if not valid:

        return (
            np.eye(3, dtype=np.float64),

            inliers,

            False
        )

    # Normalize homography.

    if abs(H[2, 2]) > 1e-10:

        H /= H[2, 2]

    return (
        H,

        inliers,

        True
    )


# =====================================================================
# FEATURE EXTRACTION AND OPTICAL FLOW
# =====================================================================

def detect_background_features(gray):

    return cv2.goodFeaturesToTrack(

        gray,

        maxCorners=MAX_FEATURES,

        qualityLevel=QUALITY_LEVEL,

        minDistance=MIN_DISTANCE,

        blockSize=7,

        useHarrisDetector=False
    )


def track_features(
    previous_gray,
    current_gray,
    previous_points
):

    if previous_points is None or len(previous_points) < 4:

        return (

            np.empty((0, 2), dtype=np.float32),

            np.empty((0, 2), dtype=np.float32)
        )

    current_points, status, errors = (
        cv2.calcOpticalFlowPyrLK(

            previous_gray,

            current_gray,

            previous_points,

            None,

            winSize=(
                FLOW_WINDOW,
                FLOW_WINDOW
            ),

            maxLevel=PYRAMID_LEVELS,

            criteria=(

                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_COUNT,

                30,

                0.01
            )
        )
    )

    if current_points is None or status is None:

        return (

            np.empty((0, 2), dtype=np.float32),

            np.empty((0, 2), dtype=np.float32)
        )

    old_xy = previous_points.reshape(
        -1,
        2
    )

    new_xy = current_points.reshape(
        -1,
        2
    )

    valid = (
        status.reshape(-1).astype(bool)
        & np.isfinite(old_xy).all(axis=1)
        & np.isfinite(new_xy).all(axis=1)
    )

    if errors is not None:

        valid &= (
            errors.reshape(-1) < 30.0
        )

    return (
        old_xy[valid],

        new_xy[valid]
    )


# =====================================================================
# CAMERA MOTION MODEL
# =====================================================================

class CameraMotionEstimator:

    def __init__(self):

        self.previous_gray = None

        self.previous_points = None

        self.last_homography = np.eye(
            3,
            dtype=np.float64
        )

        self.total_frames = 0

        self.valid_estimates = 0

        self.failed_estimates = 0

    def initialize(self, frame):

        gray = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2GRAY
        )

        self.previous_gray = gray

        self.previous_points = detect_background_features(
            gray
        )

        self.total_frames = 1

    def update(self, frame):

        if self.previous_gray is None:

            self.initialize(frame)

            return {

                "homography": np.eye(3),

                "valid": False,

                "inliers": 0,

                "matches": 0,

                "translation": np.zeros(2),

                "rotation_degrees": 0.0,

                "scale": 1.0
            }

        current_gray = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2GRAY
        )

        old_points, new_points = track_features(

            self.previous_gray,

            current_gray,

            self.previous_points
        )

        H, inlier_mask, valid = estimate_homography(

            old_points,

            new_points
        )

        if valid:

            self.last_homography = H.copy()

            self.valid_estimates += 1

        else:

            # Do not use an invalid estimate.
            # Identity means no compensation this frame.

            H = np.eye(
                3,
                dtype=np.float64
            )

            self.failed_estimates += 1

        # Extract approximate translation, rotation and scale.
        #
        # These values are descriptive approximations only.
        # A general projective homography cannot always be
        # decomposed into one unique translation/rotation/scale.

        translation = np.array(
            [
                H[0, 2],

                H[1, 2]
            ]
        )

        rotation = np.degrees(
            np.arctan2(
                H[1, 0],
                H[0, 0]
            )
        )

        scale = np.sqrt(

            H[0, 0] ** 2

            + H[1, 0] ** 2
        )

        # Refresh features every frame.
        # This avoids repeatedly tracking stale points.

        self.previous_gray = current_gray

        self.previous_points = detect_background_features(
            current_gray
        )

        self.total_frames += 1

        return {

            "homography": H,

            "valid": valid,

            "inliers": int(
                np.sum(inlier_mask)
            ),

            "matches": len(old_points),

            "translation": translation,

            "rotation_degrees": float(rotation),

            "scale": float(scale),

            "previous_points": old_points,

            "current_points": new_points,

            "inlier_mask": inlier_mask
        }

    def summary(self):

        return {

            "frames": self.total_frames,

            "valid_estimates": self.valid_estimates,

            "failed_estimates": self.failed_estimates
        }


# =====================================================================
# COMPENSATED OBJECT TRACK
# =====================================================================

class CompensatedTrack:

    def __init__(
        self,
        track_id,
        bbox,
        class_id=0
    ):

        self.track_id = track_id

        self.bbox = np.asarray(
            bbox,
            dtype=np.float64
        )

        self.class_id = class_id

        self.age = 1

        self.hits = 1

        self.missed = 0

        self.image_trajectory = [

            bbox_center(self.bbox)
        ]

        self.compensated_trajectory = [

            bbox_center(self.bbox)
        ]

        self.independent_motion = np.zeros(
            2,
            dtype=np.float64
        )

        self.camera_motion = np.zeros(
            2,
            dtype=np.float64
        )

    def predict(self, homography):

        self.age += 1

        old_center = bbox_center(
            self.bbox
        )

        predicted_bbox = transform_bbox(

            self.bbox,

            homography
        )

        new_center = bbox_center(
            predicted_bbox
        )

        camera_displacement = (
            new_center - old_center
        )

        self.bbox = predicted_bbox

        self.camera_motion = camera_displacement

        # The compensated trajectory remains unchanged
        # when the object is stationary and only the camera moves.

        compensated_center = (
            self.compensated_trajectory[-1].copy()
        )

        self.image_trajectory.append(
            new_center.copy()
        )

        self.compensated_trajectory.append(
            compensated_center
        )

    def correct(self, detection):

        detection_center = bbox_center(
            detection["bbox"]
        )

        predicted_center = bbox_center(
            self.bbox
        )

        # Independent motion is the difference between measured
        # object position and camera-only predicted position.

        self.independent_motion = (

            detection_center - predicted_center
        )

        self.bbox = np.asarray(
            detection["bbox"],
            dtype=np.float64
        )

        self.compensated_trajectory[-1] += (
            self.independent_motion
        )

        self.image_trajectory[-1] = (
            detection_center.copy()
        )

        self.hits += 1

        self.missed = 0


# =====================================================================
# DETECTION-TRACK ASSOCIATION
# =====================================================================

def iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)

    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)

    y2 = min(ay2, by2)

    intersection = max(
        0.0,
        x2 - x1
    ) * max(
        0.0,
        y2 - y1
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

    union = (
        area_a + area_b - intersection
    )

    return intersection / max(
        union,
        1e-12
    )


class CameraCompensatedTracker:

    def __init__(
        self,
        iou_threshold=0.1,
        max_missed=10
    ):

        self.tracks = []

        self.next_id = 1

        self.iou_threshold = iou_threshold

        self.max_missed = max_missed

    def update(
        self,
        detections,
        homography
    ):

        # Step 1:
        # Predict all existing boxes using camera motion.

        for track in self.tracks:

            track.predict(
                homography
            )

        # Step 2:
        # Build association cost matrix.

        cost = np.ones(

            (
                len(self.tracks),

                len(detections)
            ),

            dtype=np.float64
        )

        for i, track in enumerate(self.tracks):

            for j, detection in enumerate(detections):

                if track.class_id != detection["class_id"]:
                    continue

                cost[i, j] = 1.0 - iou(

                    track.bbox,

                    detection["bbox"]
                )

        # Step 3:
        # Hungarian assignment.

        from scipy.optimize import linear_sum_assignment

        if cost.size:

            rows, cols = linear_sum_assignment(
                cost
            )

            assignments = list(
                zip(rows.tolist(), cols.tolist())
            )

        else:

            assignments = []

        matched_tracks = set()

        matched_detections = set()

        for i, j in assignments:

            if cost[i, j] > (
                1.0 - self.iou_threshold
            ):
                continue

            self.tracks[i].correct(
                detections[j]
            )

            matched_tracks.add(i)

            matched_detections.add(j)

        # Step 4:
        # Unmatched tracks.

        for i, track in enumerate(self.tracks):

            if i not in matched_tracks:

                track.missed += 1

        # Step 5:
        # New tracks.

        for j, detection in enumerate(detections):

            if j in matched_detections:
                continue

            track = CompensatedTrack(

                self.next_id,

                detection["bbox"],

                detection["class_id"]
            )

            self.tracks.append(
                track
            )

            self.next_id += 1

        # Step 6:
        # Remove expired tracks.

        self.tracks = [

            track

            for track in self.tracks

            if track.missed <= self.max_missed
        ]

        return self.tracks


# =====================================================================
# DEMONSTRATION DETECTOR
# =====================================================================

class MotionDetector:

    """
    Demonstration detector using MOG2.

    It detects foreground motion, not semantic object categories.

    Replace this detector with YOLO or another trained detector
    for real waste-sorting applications.
    """

    def __init__(self, min_area=500):

        self.min_area = min_area

        self.background = (
            cv2.createBackgroundSubtractorMOG2(

                history=500,

                varThreshold=32,

                detectShadows=False
            )
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

            detections.append(

                {

                    "bbox": np.array(

                        [
                            x,

                            y,

                            x + w,

                            y + h
                        ],

                        dtype=np.float64
                    ),

                    "class_id": 0,

                    "score": 1.0
                }
            )

        return detections, mask


# =====================================================================
# VISUALIZATION
# =====================================================================

def draw_tracks(
    frame,
    detections,
    tracks,
    motion_info,
    frame_number
):

    output = frame.copy()

    # Draw raw detections.

    for detection in detections:

        x1, y1, x2, y2 = map(
            int,
            detection["bbox"]
        )

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            (255, 0, 255),

            1
        )

    # Draw tracks.

    for track in tracks:

        x1, y1, x2, y2 = map(
            int,
            track.bbox
        )

        color = (

            int((track.track_id * 67) % 255),

            int((track.track_id * 131) % 255),

            int((track.track_id * 197) % 255)
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

            0.6,

            color,

            2
        )

        # Draw image-coordinate trajectory.

        trajectory = np.asarray(
            track.image_trajectory,
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

        # Draw compensated trajectory.

        compensated = np.asarray(
            track.compensated_trajectory,
            dtype=np.int32
        )

        if len(compensated) >= 2:

            cv2.polylines(

                output,

                [compensated.reshape(-1, 1, 2)],

                False,

                (0, 255, 255),

                2
            )

    # Camera motion information.

    dx, dy = motion_info["translation"]

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

        f"Camera translation: ({dx:.1f}, {dy:.1f})",

        (15, 60),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 255, 255),

        2
    )

    cv2.putText(

        output,

        f"Rotation: {motion_info['rotation_degrees']:.2f} deg",

        (15, 85),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 255, 255),

        2
    )

    cv2.putText(

        output,

        f"RANSAC inliers: {motion_info['inliers']}",

        (15, 110),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

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
            video_path
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            "Cannot open input video."
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
            "Cannot create output video."
        )

    camera = CameraMotionEstimator()

    tracker = CameraCompensatedTracker(

        iou_threshold=args.iou_threshold,

        max_missed=args.max_missed
    )

    detector = MotionDetector(
        min_area=args.min_area
    )

    frame_number = 0

    processing_times = []

    print("\nStarting camera-motion-compensated tracking.")

    try:

        while True:

            start = time.perf_counter()

            ret, frame = capture.read()

            if not ret:
                break

            frame_number += 1

            # ---------------------------------------------------------
            # First frame.
            # ---------------------------------------------------------

            if camera.previous_gray is None:

                camera.initialize(
                    frame
                )

                detections, mask = detector.detect(
                    frame
                )

                tracks = tracker.update(

                    detections,

                    np.eye(3)
                )

                motion_info = {

                    "translation": np.zeros(2),

                    "rotation_degrees": 0.0,

                    "inliers": 0
                }

            else:

                # -----------------------------------------------------
                # Estimate global camera motion.
                # -----------------------------------------------------

                motion_info = camera.update(
                    frame
                )

                # -----------------------------------------------------
                # Detect objects.
                # -----------------------------------------------------

                detections, mask = detector.detect(
                    frame
                )

                # -----------------------------------------------------
                # Compensate tracks and associate detections.
                # -----------------------------------------------------

                tracks = tracker.update(

                    detections,

                    motion_info["homography"]
                )

            elapsed = (
                time.perf_counter() - start
            )

            processing_times.append(
                elapsed
            )

            output = draw_tracks(

                frame,

                detections,

                tracks,

                motion_info,

                frame_number
            )

            writer.write(
                output
            )

            if not args.no_display:

                cv2.imshow(

                    "Lesson 47 - Camera Motion Compensation",

                    output
                )

                if args.show_mask:

                    cv2.imshow(
                        "Foreground Mask",
                        mask
                    )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_number % 100 == 0:

                current_fps = 1.0 / max(
                    elapsed,
                    1e-9
                )

                print(

                    f"Frame {frame_number} | "

                    f"Tracks: {len(tracks)} | "

                    f"Camera motion valid: {motion_info.get('valid', False)} | "

                    f"FPS: {current_fps:.1f}"
                )

    finally:

        capture.release()

        writer.release()

        cv2.destroyAllWindows()

    average_time = (

        np.mean(processing_times)

        if processing_times else 0.0
    )

    summary = {

        "input_video": str(video_path),

        "output_video": str(output_path),

        "frames_processed": frame_number,

        "resolution": [
            width,
            height
        ],

        "average_processing_fps": (

            1.0 / average_time

            if average_time > 0 else 0.0
        ),

        "camera_motion": camera.summary(),

        "remaining_tracks": len(
            tracker.tracks
        )
    }

    summary_path = (

        output_path.parent

        / "camera_motion_summary.json"
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

    print("\nOutput:", output_path)

    print("Summary:", summary_path)


# =====================================================================
# UNIT TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print("LESSON 47 — CAMERA MOTION COMPENSATION TESTS")

    print("=" * 70)

    # ---------------------------------------------------------------
    # Test 1: identity homography.
    # ---------------------------------------------------------------

    H = np.eye(3)

    points = np.array(

        [
            [10, 20],

            [100, 200]
        ],

        dtype=np.float64
    )

    transformed = transform_points(
        points,
        H
    )

    assert np.allclose(
        transformed,
        points
    )

    print("Test 1 passed: identity homography")

    # ---------------------------------------------------------------
    # Test 2: translation homography.
    # ---------------------------------------------------------------

    H = np.array(

        [
            [1, 0, 5],

            [0, 1, 3],

            [0, 0, 1]
        ],

        dtype=np.float64
    )

    transformed = transform_points(
        points,
        H
    )

    expected = points + np.array(
        [5, 3]
    )

    assert np.allclose(
        transformed,
        expected
    )

    print("Test 2 passed: translation homography")

    # ---------------------------------------------------------------
    # Test 3: scale homography.
    # ---------------------------------------------------------------

    H = np.array(

        [
            [2, 0, 0],

            [0, 2, 0],

            [0, 0, 1]
        ],

        dtype=np.float64
    )

    transformed = transform_points(
        points,
        H
    )

    assert np.allclose(
        transformed,
        points * 2
    )

    print("Test 3 passed: scale homography")

    # ---------------------------------------------------------------
    # Test 4: perspective transformation.
    # ---------------------------------------------------------------

    H = np.array(

        [
            [1, 0, 10],

            [0, 1, 20],

            [0.001, 0.002, 1]
        ],

        dtype=np.float64
    )

    transformed = transform_points(

        np.array(
            [[100, 100]]
        ),

        H
    )

    assert transformed.shape == (1, 2)

    assert np.isfinite(
        transformed
    ).all()

    print("Test 4 passed: perspective transformation")

    # ---------------------------------------------------------------
    # Test 5: bounding-box identity.
    # ---------------------------------------------------------------

    bbox = np.array(
        [10, 20, 100, 120]
    )

    transformed = transform_bbox(
        bbox,
        np.eye(3)
    )

    assert np.allclose(
        transformed,
        bbox
    )

    print("Test 5 passed: identity bounding-box transform")

    # ---------------------------------------------------------------
    # Test 6: bounding-box translation.
    # ---------------------------------------------------------------

    H = np.array(

        [
            [1, 0, 10],

            [0, 1, 5],

            [0, 0, 1]
        ]
    )

    transformed = transform_bbox(
        bbox,
        H
    )

    assert np.allclose(
        transformed,
        [20, 25, 110, 125]
    )

    print("Test 6 passed: translated bounding box")

    # ---------------------------------------------------------------
    # Test 7: bounding-box center.
    # ---------------------------------------------------------------

    center = bbox_center(
        bbox
    )

    assert np.allclose(
        center,
        [55, 70]
    )

    print("Test 7 passed: bounding-box center")

    # ---------------------------------------------------------------
    # Test 8: estimate homography from exact correspondences.
    # ---------------------------------------------------------------

    source = np.array(

        [
            [0, 0],

            [100, 0],

            [100, 100],

            [0, 100],

            [50, 50],

            [25, 75],

            [75, 25],

            [30, 40]
        ],

        dtype=np.float32
    )

    expected_H = np.array(

        [
            [1, 0, 12],

            [0, 1, -7],

            [0, 0, 1]
        ],

        dtype=np.float64
    )

    destination = transform_points(
        source,
        expected_H
    ).astype(
        np.float32
    )

    estimated_H, inliers, valid = estimate_homography(

        source,

        destination,

        min_inliers=4
    )

    assert valid

    assert np.allclose(
        estimated_H,
        expected_H,
        atol=1e-3
    )

    assert np.sum(inliers) >= 4

    print("Test 8 passed: exact homography estimation")

    # ---------------------------------------------------------------
    # Test 9: camera-only motion.
    # ---------------------------------------------------------------

    previous_center = np.array(
        [100, 100]
    )

    H = np.array(

        [
            [1, 0, 20],

            [0, 1, 10],

            [0, 0, 1]
        ]
    )

    current_center = transform_points(

        previous_center.reshape(1, 2),

        H
    )[0]

    camera_only_prediction = current_center

    independent_motion = (
        current_center - camera_only_prediction
    )

    assert np.allclose(
        independent_motion,
        [0, 0]
    )

    print("Test 9 passed: stationary object under camera motion")

    # ---------------------------------------------------------------
    # Test 10: independent object motion.
    # ---------------------------------------------------------------

    camera_prediction = np.array(
        [120, 110]
    )

    measured_position = np.array(
        [125, 108]
    )

    independent_motion = (
        measured_position - camera_prediction
    )

    assert np.allclose(
        independent_motion,
        [5, -2]
    )

    print("Test 10 passed: independent object motion")

    # ---------------------------------------------------------------
    # Test 11: track prediction.
    # ---------------------------------------------------------------

    track = CompensatedTrack(

        track_id=1,

        bbox=np.array(
            [10, 20, 60, 80]
        )
    )

    H = np.array(

        [
            [1, 0, 5],

            [0, 1, 3],

            [0, 0, 1]
        ]
    )

    track.predict(
        H
    )

    assert np.allclose(
        track.bbox,
        [15, 23, 65, 83]
    )

    print("Test 11 passed: camera-compensated track prediction")

    # ---------------------------------------------------------------
    # Test 12: track correction.
    # ---------------------------------------------------------------

    detection = {

        "bbox": np.array(
            [20, 23, 70, 83]
        ),

        "class_id": 0
    }

    track.correct(
        detection
    )

    assert np.allclose(
        track.independent_motion,
        [5, 0]
    )

    assert track.hits == 2

    print("Test 12 passed: independent motion correction")

    # ---------------------------------------------------------------
    # Test 13: invalid homography input.
    # ---------------------------------------------------------------

    source = np.array(
        [[0, 0], [1, 1], [2, 2]]
    )

    destination = source.copy()

    H, inliers, valid = estimate_homography(
        source,
        destination
    )

    assert not valid

    assert np.allclose(
        H,
        np.eye(3)
    )

    print("Test 13 passed: insufficient homography correspondences")

    # ---------------------------------------------------------------
    # Test 14: tracker initialization.
    # ---------------------------------------------------------------

    tracker = CameraCompensatedTracker()

    assert len(tracker.tracks) == 0

    assert tracker.next_id == 1

    print("Test 14 passed: tracker initialization")

    # ---------------------------------------------------------------
    # Test 15: IoU.
    # ---------------------------------------------------------------

    a = np.array(
        [0, 0, 100, 100]
    )

    b = np.array(
        [50, 0, 150, 100]
    )

    assert abs(
        iou(a, b) - 1.0 / 3.0
    ) < 1e-8

    print("Test 15 passed: IoU")

    # ---------------------------------------------------------------
    # Test 16: feature transform dimensions.
    # ---------------------------------------------------------------

    H = np.eye(3)

    points = np.random.default_rng(
        42
    ).uniform(
        0,
        100,
        size=(100, 2)
    )

    transformed = transform_points(
        points,
        H
    )

    assert transformed.shape == (100, 2)

    assert np.allclose(
        transformed,
        points
    )

    print("Test 16 passed: multiple-point transformation")

    print("\nAll tests passed.")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Lesson 47 — Camera Motion Compensation"
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
            "runs/lesson47/"
            "camera_motion_compensation.mp4"
        )
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--max-missed",
        type=int,
        default=10
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

    if not 0 <= args.iou_threshold <= 1:

        parser.error(
            "--iou-threshold must be between 0 and 1."
        )

    if args.max_missed < 0:

        parser.error(
            "--max-missed cannot be negative."
        )

    if args.min_area < 1:

        parser.error(
            "--min-area must be positive."
        )

    print("=" * 70)

    print(
        "LESSON 47 — CAMERA MOTION COMPENSATION"
    )

    print("=" * 70)

    run_tests()

    if args.check_only:

        print("\nCheck-only completed.")

        return

    if args.video is None:

        print("\nNo input video supplied.")

        print(
            "Use: python lesson47_camera_motion_compensation.py "
            "--video robot_camera.mp4"
        )

        return

    process_video(args)


if __name__ == "__main__":

    main()