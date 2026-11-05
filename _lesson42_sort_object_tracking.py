"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 42 — SORT OBJECT TRACKING
======================================================================

Filename:
    lesson42_sort_object_tracking.py

PHASE:
    Phase 4 — Tracking and Visual Memory

======================================================================
LEARNING OBJECTIVES
======================================================================

Implement SORT:

    Simple Online and Realtime Tracking

Core components:

    1. Kalman filter
    2. Bounding-box state estimation
    3. IoU association cost matrix
    4. Hungarian assignment algorithm
    5. Track initialization
    6. Track confirmation
    7. Track expiration
    8. Real-video tracking visualization

This lesson builds on Lesson 41.

No pretrained object detector is used.

OpenCV is used for:
    - Video input/output
    - Background subtraction
    - Morphological filtering
    - Contour extraction
    - Visualization

NumPy is used for:
    - Kalman filtering
    - IoU
    - Hungarian assignment
    - Track management

======================================================================
1. SORT ARCHITECTURE
======================================================================

Input video
    |
    v
Object detector
    |
    v
Bounding-box detections
    |
    v
Kalman prediction
    |
    v
IoU cost matrix
    |
    v
Hungarian assignment
    |
    v
Track update
    |
    v
Track management
    |
    v
Persistent object IDs

======================================================================
2. STATE REPRESENTATION
======================================================================

SORT commonly represents a bounding box as:

    z = [cx, cy, area, aspect_ratio]

Where:

    cx = bounding-box center x
    cy = bounding-box center y
    area = width * height
    aspect_ratio = width / height

The state is:

    x = [
        cx,
        cy,
        area,
        aspect_ratio,
        vx,
        vy,
        v_area
    ]

The standard SORT formulation commonly uses this
7-dimensional state.

The observation contains:

    [cx, cy, area, aspect_ratio]

======================================================================
3. KALMAN MOTION MODEL
======================================================================

For constant velocity:

    cx(k) = cx(k-1) + vx * dt

    cy(k) = cy(k-1) + vy * dt

    area(k) = area(k-1) + v_area * dt

Aspect ratio is assumed constant.

The transition matrix:

    F =

    [1 0 0 0 dt 0  0]
    [0 1 0 0 0  dt 0]
    [0 0 1 0 0  0  dt]
    [0 0 0 1 0  0  0]
    [0 0 0 0 1  0  0]
    [0 0 0 0 0  1  0]
    [0 0 0 0 0  0  1]

Measurement matrix:

    H =

    [1 0 0 0 0 0 0]
    [0 1 0 0 0 0 0]
    [0 0 1 0 0 0 0]
    [0 0 0 1 0 0 0]

======================================================================
4. IOU COST MATRIX
======================================================================

For N tracks and M detections:

    C is an N x M matrix.

Each element:

    C[i,j] = 1 - IoU(track_i, detection_j)

The assignment algorithm minimizes total cost.

A high IoU means a low cost.

======================================================================
5. HUNGARIAN ALGORITHM
======================================================================

The Hungarian algorithm solves the linear assignment problem.

Given a cost matrix C, find a one-to-one assignment:

    minimize:

        sum(C[i, assignment[i]])

Subject to:

    Each row is assigned at most once.

    Each column is assigned at most once.

This implementation uses the shortest augmenting path
formulation for rectangular cost matrices.

It supports N != M.

======================================================================
6. ASSOCIATION THRESHOLD
======================================================================

Hungarian assignment always attempts to assign the smaller
dimension.

However, some matches may be physically unreasonable.

Therefore:

    if IoU < threshold:

        reject the match

Rejected tracks and detections are handled separately.

======================================================================
7. TRACK MANAGEMENT
======================================================================

Each track maintains:

    ID
    Kalman state
    age
    hits
    time_since_update
    hit_streak

A new track is tentative.

It becomes confirmed after sufficient consecutive observations.

A track is deleted when it has been missing for too many frames.

======================================================================
8. ROBOTICS APPLICATION
======================================================================

Waste sorting conveyor:

    Camera detects:
        Bottle A
        Can B
        Cup C

    SORT assigns:
        Bottle A -> ID 1
        Can B    -> ID 2
        Cup C    -> ID 3

    Next frame:
        Bottle A -> ID 1
        Can B    -> ID 2
        Cup C    -> ID 3

The robot can maintain an object-specific target even as
the objects move through the camera field of view.

SORT tracks geometry and motion only.

It does not know whether ID 1 is a bottle or a can.
A separate detector/classifier provides that information.

======================================================================
RUN
======================================================================

Install:

    pip install numpy opencv-python

Numerical tests:

    python lesson42_sort_object_tracking.py --check-only

Real video:

    python lesson42_sort_object_tracking.py \
        --video conveyor_video.mp4

Save output:

    python lesson42_sort_object_tracking.py \
        --video conveyor_video.mp4 \
        --output runs/lesson42/sort_output.mp4

Disable display:

    python lesson42_sort_object_tracking.py \
        --video conveyor_video.mp4 \
        --no-display

======================================================================
"""

import argparse
import json
import time

from pathlib import Path

import cv2
import numpy as np


# =====================================================================
# BOUNDING BOX CONVERSIONS
# =====================================================================

def bbox_to_sort_measurement(box):

    """
    Convert:

        [x1, y1, x2, y2]

    Into:

        [cx, cy, area, aspect_ratio]
    """

    x1, y1, x2, y2 = map(
        float,
        box
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

    area = width * height

    aspect_ratio = width / height

    return np.array(
        [
            cx,
            cy,
            area,
            aspect_ratio
        ],
        dtype=np.float64
    )


def sort_state_to_bbox(state):

    """
    Convert SORT state into:

        [x1, y1, x2, y2]
    """

    cx = float(state[0])

    cy = float(state[1])

    area = max(
        1.0,
        float(state[2])
    )

    aspect_ratio = max(
        1e-6,
        float(state[3])
    )

    width = np.sqrt(
        area * aspect_ratio
    )

    height = area / width

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

def iou(box_a, box_b):

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

    union = area_a + area_b - intersection

    return intersection / max(
        union,
        1e-12
    )


def build_iou_cost_matrix(
    track_boxes,
    detection_boxes
):

    n = len(track_boxes)

    m = len(detection_boxes)

    cost = np.zeros(
        (n, m),
        dtype=np.float64
    )

    for i in range(n):

        for j in range(m):

            cost[i, j] = 1.0 - iou(
                track_boxes[i],
                detection_boxes[j]
            )

    return cost


# =====================================================================
# HUNGARIAN ASSIGNMENT
# =====================================================================

def hungarian_assignment(cost_matrix):

    """
    Solve rectangular linear assignment using the
    shortest augmenting path Hungarian algorithm.

    Input:
        cost_matrix: shape (N, M)

    Output:
        List of (row, column) assignments.

    This implementation supports:

        N < M
        N > M
        N = M
        Empty matrices

    Complexity:
        O(n^2 * m), where n <= m after transposition.
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

    # Potentials for rows and columns.

    u = np.zeros(
        n + 1,
        dtype=np.float64
    )

    v = np.zeros(
        m + 1,
        dtype=np.float64
    )

    # p[j] = row currently assigned to column j.

    p = np.zeros(
        m + 1,
        dtype=np.int64
    )

    # way[j] stores the previous column in the augmenting path.

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

        # Augment the matching.

        while True:

            j1 = way[j0]

            p[j0] = p[j1]

            j0 = j1

            if j0 == 0:

                break

    assignments = []

    for j in range(1, m + 1):

        if p[j] != 0:

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

class SORTKalmanFilter:

    """
    Seven-dimensional SORT Kalman filter.

    State:

        [cx, cy, area, ratio, vx, vy, v_area]

    Measurement:

        [cx, cy, area, ratio]
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
            7,
            dtype=np.float64
        )

        self.x[:4] = z

        self.P = np.eye(
            7,
            dtype=np.float64
        )

        self.P[4:, 4:] *= 1000.0

        self.P *= 10.0

        self.F = np.eye(
            7,
            dtype=np.float64
        )

        self.H = np.zeros(
            (4, 7),
            dtype=np.float64
        )

        self.H[:4, :4] = np.eye(4)

        self.R = np.eye(
            4,
            dtype=np.float64
        )

        self.R[2:, 2:] *= 10.0

        self.Q = np.eye(
            7,
            dtype=np.float64
        )

        self.Q[4:, 4:] *= 0.01

        self.I = np.eye(
            7,
            dtype=np.float64
        )

    def predict(self, dt=1.0):

        dt = float(dt)

        if dt <= 0:

            raise ValueError(
                "dt must be positive."
            )

        self.F = np.eye(
            7,
            dtype=np.float64
        )

        self.F[0, 4] = dt

        self.F[1, 5] = dt

        self.F[2, 6] = dt

        self.x = self.F @ self.x

        # Prevent non-physical negative area.

        self.x[2] = max(
            1.0,
            self.x[2]
        )

        self.P = (
            self.F
            @ self.P
            @ self.F.T
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
            self.H
            @ self.P
            @ self.H.T
            + self.R
        )

        K = np.linalg.solve(
            S,
            self.H @ self.P
        ).T

        self.x = (
            self.x
            + K @ innovation
        )

        A = self.I - K @ self.H

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
            1e-6,
            self.x[3]
        )

        return self.x.copy()

    def get_bbox(self):

        return sort_state_to_bbox(
            self.x
        )

    def get_velocity(self):

        return self.x[4:7].copy()


# =====================================================================
# SORT TRACK
# =====================================================================

class SORTTrack:

    def __init__(
        self,
        detection,
        track_id
    ):

        self.track_id = track_id

        measurement = bbox_to_sort_measurement(
            detection
        )

        self.kalman = SORTKalmanFilter(
            measurement
        )

        self.age = 1

        self.hits = 1

        self.hit_streak = 1

        self.time_since_update = 0

        self.history = []

    def predict(self, dt=1.0):

        self.kalman.predict(dt)

        self.age += 1

        self.time_since_update += 1

        if self.time_since_update > 0:

            self.hit_streak = 0

        return self.get_bbox()

    def update(self, detection):

        measurement = bbox_to_sort_measurement(
            detection
        )

        self.kalman.update(
            measurement
        )

        self.time_since_update = 0

        self.hits += 1

        self.hit_streak += 1

        self.history.append(
            self.get_bbox()
        )

    def get_bbox(self):

        return self.kalman.get_bbox()

    def get_velocity(self):

        return self.kalman.get_velocity()


# =====================================================================
# SORT TRACKER
# =====================================================================

class SORT:

    def __init__(
        self,
        max_age=10,
        min_hits=3,
        iou_threshold=0.15
    ):

        self.max_age = max_age

        self.min_hits = min_hits

        self.iou_threshold = iou_threshold

        self.tracks = []

        self.next_id = 1

        self.frame_count = 0

        self.total_created = 0

        self.total_deleted = 0

    def update(
        self,
        detections,
        dt=1.0
    ):

        detections = [
            np.asarray(
                detection,
                dtype=np.float64
            )
            for detection in detections
        ]

        self.frame_count += 1

        # -------------------------------------------------------------
        # Step 1: Kalman prediction
        # -------------------------------------------------------------

        predicted_boxes = []

        valid_tracks = []

        for track in self.tracks:

            predicted_box = track.predict(dt)

            if np.isfinite(
                predicted_box
            ).all():

                predicted_boxes.append(
                    predicted_box
                )

                valid_tracks.append(
                    track
                )

        self.tracks = valid_tracks

        predicted_boxes = np.asarray(
            predicted_boxes,
            dtype=np.float64
        ).reshape(-1, 4)

        detections_array = np.asarray(
            detections,
            dtype=np.float64
        ).reshape(-1, 4)

        # -------------------------------------------------------------
        # Step 2: IoU cost matrix
        # -------------------------------------------------------------

        cost_matrix = build_iou_cost_matrix(

            predicted_boxes,

            detections_array
        )

        # -------------------------------------------------------------
        # Step 3: Hungarian assignment
        # -------------------------------------------------------------

        assignments = hungarian_assignment(
            cost_matrix
        )

        matches = []

        unmatched_tracks = set(
            range(len(self.tracks))
        )

        unmatched_detections = set(
            range(len(detections))
        )

        for track_index, detection_index in assignments:

            similarity = (
                1.0 - cost_matrix[
                    track_index,
                    detection_index
                ]
            )

            if similarity < self.iou_threshold:

                continue

            matches.append(
                (
                    track_index,
                    detection_index
                )
            )

            unmatched_tracks.discard(
                track_index
            )

            unmatched_detections.discard(
                detection_index
            )

        # -------------------------------------------------------------
        # Step 4: update matched tracks
        # -------------------------------------------------------------

        for track_index, detection_index in matches:

            self.tracks[track_index].update(
                detections[detection_index]
            )

        # -------------------------------------------------------------
        # Step 5: create new tracks
        # -------------------------------------------------------------

        for detection_index in sorted(
            unmatched_detections
        ):

            track = SORTTrack(

                detections[detection_index],

                self.next_id
            )

            self.tracks.append(
                track
            )

            self.next_id += 1

            self.total_created += 1

        # -------------------------------------------------------------
        # Step 6: remove stale tracks
        # -------------------------------------------------------------

        retained = []

        for track in self.tracks:

            if track.time_since_update > self.max_age:

                self.total_deleted += 1

            else:

                retained.append(
                    track
                )

        self.tracks = retained

        return self.get_active_tracks()

    def get_active_tracks(self):

        results = []

        for track in self.tracks:

            if track.hits < self.min_hits:

                continue

            if track.time_since_update > 0:

                continue

            results.append(
                {
                    "id": track.track_id,

                    "bbox": track.get_bbox(),

                    "velocity": track.get_velocity(),

                    "age": track.age,

                    "hits": track.hits,

                    "hit_streak": track.hit_streak,

                    "time_since_update": track.time_since_update,
                }
            )

        return results

    def get_summary(self):

        return {

            "frames_processed": self.frame_count,

            "tracks_created": self.total_created,

            "tracks_deleted": self.total_deleted,

            "tracks_remaining": len(self.tracks),

            "next_track_id": self.next_id,
        }


# =====================================================================
# BACKGROUND MOTION DETECTOR
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
# VISUALIZATION
# =====================================================================

def draw_tracks(
    frame,
    detections,
    tracks,
    frame_number,
    processing_fps
):

    output = frame.copy()

    # Raw detector boxes: gray.

    for box in detections:

        x1, y1, x2, y2 = map(
            int,
            box
        )

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            (130, 130, 130),

            1
        )

    # SORT boxes.

    for track in tracks:

        x1, y1, x2, y2 = map(
            int,
            track["bbox"]
        )

        track_id = track["id"]

        vx, vy, v_area = track["velocity"]

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

        f"Processing FPS: {processing_fps:.1f}",

        (15, 60),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 255),

        2
    )

    cv2.putText(

        output,

        f"Active objects: {len(tracks)}",

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
            f"Video does not exist: {video_path}"
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

    tracker = SORT(

        max_age=args.max_age,

        min_hits=args.min_hits,

        iou_threshold=args.iou_threshold
    )

    processing_times = []

    frame_number = 0

    print("\nSORT video processing started.")

    print("Video:", video_path)

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

            # Detection.

            detections, foreground = detector.detect(
                frame
            )

            # Tracking.

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

            annotated = draw_tracks(

                frame,

                detections,

                active_tracks,

                frame_number,

                processing_fps
            )

            writer.write(
                annotated
            )

            if args.display:

                cv2.imshow(
                    "Lesson 42 - SORT",
                    annotated
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
        np.mean(processing_times)
        if processing_times else 0.0
    )

    summary = {

        "input_video": str(video_path),

        "output_video": str(output_path),

        "frames_processed": frame_number,

        "average_processing_fps": (
            1.0 / average_time
            if average_time > 0 else 0.0
        ),

        "sort": tracker.get_summary()
    }

    summary_path = (
        output_path.parent
        / "sort_summary.json"
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
# NUMERICAL TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print("LESSON 42 — SORT TESTS")

    print("=" * 70)

    # ---------------------------------------------------------------
    # Test 1: bounding-box conversion
    # ---------------------------------------------------------------

    box = np.array(
        [10, 20, 110, 220],
        dtype=np.float64
    )

    measurement = bbox_to_sort_measurement(
        box
    )

    reconstructed = sort_state_to_bbox(
        np.array(
            [
                measurement[0],
                measurement[1],
                measurement[2],
                measurement[3],
                0,
                0,
                0
            ],
            dtype=np.float64
        )
    )

    assert np.allclose(
        box,
        reconstructed
    )

    print("Test 1 passed: SORT bounding-box representation")

    # ---------------------------------------------------------------
    # Test 2: IoU
    # ---------------------------------------------------------------

    assert abs(
        iou(box, box) - 1.0
    ) < 1e-8

    print("Test 2 passed: identical-box IoU")

    # ---------------------------------------------------------------
    # Test 3: cost matrix
    # ---------------------------------------------------------------

    boxes = [
        [0, 0, 100, 100],
        [200, 200, 300, 300]
    ]

    cost = build_iou_cost_matrix(
        boxes,
        boxes
    )

    assert cost.shape == (2, 2)

    assert abs(
        cost[0, 0]
    ) < 1e-8

    assert abs(
        cost[1, 1]
    ) < 1e-8

    print("Test 3 passed: IoU cost matrix")

    # ---------------------------------------------------------------
    # Test 4: Hungarian square assignment
    # ---------------------------------------------------------------

    matrix = np.array(
        [
            [4, 1, 3],
            [2, 0, 5],
            [3, 2, 2]
        ],
        dtype=np.float64
    )

    assignment = hungarian_assignment(
        matrix
    )

    total_cost = sum(
        matrix[i, j]
        for i, j in assignment
    )

    assert len(assignment) == 3

    assert abs(
        total_cost - 5.0
    ) < 1e-8

    print("Test 4 passed: Hungarian square assignment")

    # ---------------------------------------------------------------
    # Test 5: rectangular assignment, rows < columns
    # ---------------------------------------------------------------

    matrix = np.array(
        [
            [10, 1, 5],
            [2, 8, 4]
        ],
        dtype=np.float64
    )

    assignment = hungarian_assignment(
        matrix
    )

    total_cost = sum(
        matrix[i, j]
        for i, j in assignment
    )

    assert len(assignment) == 2

    assert abs(
        total_cost - 3.0
    ) < 1e-8

    print("Test 5 passed: rectangular assignment (2 x 3)")

    # ---------------------------------------------------------------
    # Test 6: rectangular assignment, rows > columns
    # ---------------------------------------------------------------

    matrix = np.array(
        [
            [10, 1],
            [2, 8],
            [3, 4]
        ],
        dtype=np.float64
    )

    assignment = hungarian_assignment(
        matrix
    )

    total_cost = sum(
        matrix[i, j]
        for i, j in assignment
    )

    assert len(assignment) == 2

    assert abs(
        total_cost - 3.0
    ) < 1e-8

    print("Test 6 passed: rectangular assignment (3 x 2)")

    # ---------------------------------------------------------------
    # Test 7: empty assignment
    # ---------------------------------------------------------------

    assert hungarian_assignment(
        np.zeros((0, 3))
    ) == []

    assert hungarian_assignment(
        np.zeros((3, 0))
    ) == []

    print("Test 7 passed: empty assignment")

    # ---------------------------------------------------------------
    # Test 8: assignment uniqueness
    # ---------------------------------------------------------------

    matrix = np.array(
        [
            [1, 2, 3],
            [2, 1, 3],
            [3, 2, 1]
        ],
        dtype=np.float64
    )

    assignment = hungarian_assignment(
        matrix
    )

    rows = [
        pair[0]
        for pair in assignment
    ]

    columns = [
        pair[1]
        for pair in assignment
    ]

    assert len(rows) == len(set(rows))

    assert len(columns) == len(set(columns))

    print("Test 8 passed: assignment uniqueness")

    # ---------------------------------------------------------------
    # Test 9: Kalman prediction
    # ---------------------------------------------------------------

    measurement = np.array(
        [100, 100, 4000, 1.0],
        dtype=np.float64
    )

    kalman = SORTKalmanFilter(
        measurement
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

    print("Test 9 passed: SORT Kalman prediction")

    # ---------------------------------------------------------------
    # Test 10: Kalman update
    # ---------------------------------------------------------------

    kalman.update(
        np.array(
            [108, 104, 4000, 1.0]
        )
    )

    assert np.isfinite(
        kalman.x
    ).all()

    print("Test 10 passed: SORT Kalman update")

    # ---------------------------------------------------------------
    # Test 11: covariance symmetry
    # ---------------------------------------------------------------

    assert np.allclose(
        kalman.P,
        kalman.P.T,
        atol=1e-8
    )

    print("Test 11 passed: covariance symmetry")

    # ---------------------------------------------------------------
    # Test 12: tracker initialization
    # ---------------------------------------------------------------

    tracker = SORT(
        max_age=3,
        min_hits=1,
        iou_threshold=0.1
    )

    tracks = tracker.update(
        [
            np.array(
                [0, 0, 50, 50],
                dtype=np.float64
            )
        ]
    )

    assert len(tracks) == 1

    assert tracks[0]["id"] == 1

    print("Test 12 passed: tracker initialization")

    # ---------------------------------------------------------------
    # Test 13: ID persistence
    # ---------------------------------------------------------------

    tracker.update(
        [
            np.array(
                [5, 0, 55, 50],
                dtype=np.float64
            )
        ]
    )

    assert len(tracker.tracks) == 1

    assert tracker.tracks[0].track_id == 1

    assert tracker.tracks[0].hits == 2

    print("Test 13 passed: persistent ID")

    # ---------------------------------------------------------------
    # Test 14: two independent objects
    # ---------------------------------------------------------------

    tracker = SORT(
        max_age=5,
        min_hits=1,
        iou_threshold=0.1
    )

    tracker.update(
        [
            np.array([0, 0, 40, 40]),
            np.array([200, 0, 240, 40])
        ]
    )

    tracker.update(
        [
            np.array([5, 0, 45, 40]),
            np.array([205, 0, 245, 40])
        ]
    )

    assert len(tracker.tracks) == 2

    assert tracker.tracks[0].track_id != (
        tracker.tracks[1].track_id
    )

    print("Test 14 passed: two-object tracking")

    # ---------------------------------------------------------------
    # Test 15: missing detection and track aging
    # ---------------------------------------------------------------

    tracker = SORT(
        max_age=2,
        min_hits=1
    )

    tracker.update(
        [
            np.array([0, 0, 40, 40])
        ]
    )

    tracker.update([])

    assert len(tracker.tracks) == 1

    assert tracker.tracks[0].time_since_update == 1

    tracker.update([])

    assert len(tracker.tracks) == 1

    tracker.update([])

    assert len(tracker.tracks) == 0

    print("Test 15 passed: stale-track deletion")

    # ---------------------------------------------------------------
    # Test 16: velocity estimation
    # ---------------------------------------------------------------

    tracker = SORT(
        max_age=5,
        min_hits=1,
        iou_threshold=0.05
    )

    for x in [0, 5, 10, 15, 20]:

        tracker.update(
            [
                np.array(
                    [x, 0, x + 40, 40],
                    dtype=np.float64
                )
            ]
        )

    velocity = tracker.tracks[0].get_velocity()

    assert velocity[0] > 0

    print(
        "Test 16 passed: velocity estimation",
        f"(vx={velocity[0]:.3f})"
    )

    print("\nAll 16 tests passed.\n")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description="Lesson 42: SORT Object Tracking"
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/lesson42/sort_output.mp4"
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=300
    )

    parser.add_argument(
        "--max-age",
        type=int,
        default=10
    )

    parser.add_argument(
        "--min-hits",
        type=int,
        default=3
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.15
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

    if not 0 <= args.iou_threshold <= 1:

        parser.error(
            "--iou-threshold must be between 0 and 1"
        )

    print("=" * 70)

    print("LESSON 42 — SORT OBJECT TRACKING")

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
            "python lesson42_sort_object_tracking.py "
            "--video conveyor_video.mp4"
        )

        return

    args.display = not args.no_display

    process_video(args)


if __name__ == "__main__":

    main()