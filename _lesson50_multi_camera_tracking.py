"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 50 — MULTI-CAMERA TRACKING AND CROSS-CAMERA RE-ID
======================================================================

Filename:
    lesson50_multi_camera_tracking.py

PHASE:
    Phase 4 — Tracking and Visual Memory

======================================================================
LEARNING OBJECTIVES
======================================================================

1. Understand multi-camera object tracking.
2. Distinguish local track IDs from global object IDs.
3. Reuse a CNN appearance encoder trained in Lesson 49.
4. Track objects independently inside each camera.
5. Associate local tracks across different cameras.
6. Maintain a shared global identity gallery.
7. Use appearance similarity and elapsed time for association.
8. Apply camera-transition constraints.
9. Handle objects visible in multiple cameras.
10. Evaluate cross-camera association with numerical tests.

======================================================================
1. LOCAL TRACKING VS GLOBAL TRACKING
======================================================================

A local tracker operates inside one camera.

Camera A:
    Object -> Local ID A-1
    Object -> Local ID A-2

Camera B:
    Object -> Local ID B-1
    Object -> Local ID B-2

Local IDs are independent.

The same physical object may have different local IDs in different
cameras.

A global identity manager connects those local identities:

    Camera A: Local A-1
                    |
                    v
              Global ID 1
                    ^
                    |
    Camera B: Local B-1

The global ID represents the physical object, not the camera.

======================================================================
2. CROSS-CAMERA RE-IDENTIFICATION
======================================================================

When an object leaves Camera A and appears in Camera B:

    1. Camera B detects the object.
    2. The CNN extracts an appearance embedding.
    3. The global identity manager compares it with its gallery.
    4. The manager checks camera transition constraints.
    5. The manager checks elapsed time.
    6. The manager assigns an existing global ID or creates a new one.

======================================================================
3. APPEARANCE DISTANCE
======================================================================

For normalized appearance vectors a and b:

    cosine_similarity = a dot b

    cosine_distance = 1 - cosine_similarity

Lower distance indicates more similar appearance.

IMPORTANT:
    A learned embedding is not a perfect identity fingerprint.

Lighting, viewing angle, occlusion, and camera characteristics
can change the appearance embedding.

======================================================================
4. CAMERA TRANSITION CONSTRAINTS
======================================================================

Not every camera transition is physically possible.

Example:

    Camera 0 -> Camera 1: allowed
    Camera 1 -> Camera 0: allowed
    Camera 0 -> Camera 2: forbidden

We can configure allowed transitions.

The default configuration allows both directions between the
two supplied cameras.

======================================================================
5. TEMPORAL CONSTRAINTS
======================================================================

Suppose an object leaves Camera A at time t1 and appears in Camera B
at time t2.

    travel_time = t2 - t1

We only consider a match when:

    min_travel_time <= travel_time <= max_travel_time

For overlapping cameras, min_travel_time can be zero.

For physically separated cameras, use a realistic minimum travel
time based on the conveyor or robot-cell layout.

======================================================================
6. GLOBAL IDENTITY GALLERY
======================================================================

Each global identity stores:

    - Global ID
    - Appearance prototype
    - Last camera
    - Last seen time
    - Cameras in which it has appeared
    - Number of observations

The appearance prototype is updated gradually:

    prototype_new =
        alpha * observation + (1-alpha) * prototype_old

Then it is L2-normalized.

======================================================================
7. LOCAL TRACKING
======================================================================

Each camera has its own local tracker.

This educational implementation uses:

    - MOG2 foreground detection
    - Bounding-box association
    - Center displacement
    - IoU
    - CNN appearance embeddings

MOG2 is not semantic object detection.

For a real waste-sorting robot, replace MOG2 with YOLO or an
instance-segmentation detector. The identity manager can remain.

======================================================================
8. SYNCHRONIZATION
======================================================================

This implementation reads one frame from each camera per iteration.

It assumes both videos are synchronized and have compatible FPS.

For real cameras with different clocks or frame rates, use
timestamp-based synchronization rather than frame-index pairing.

======================================================================
9. ROBOTICS APPLICATION
======================================================================

Example: OpenArm V2 waste-sorting cell.

    Overhead Camera
          |
          v
    Local Tracking
          |
          |
          +----------+
                     |
                     v
             Global Identity Manager
                     ^
                     |
          +----------+
          |
          v
    Wrist / Side Camera
          |
          v
    Local Tracking

The global identity can help avoid duplicate handling and preserve
object history as it moves through different camera views.

For robot grasping, tracking identity is only one part of the task.
You still need calibrated camera-to-robot coordinates and a reliable
object localization system.

======================================================================
DEPENDENCIES
======================================================================

    pip install numpy opencv-python scipy torch matplotlib

======================================================================
RUN
======================================================================

Tests:

    python lesson50_multi_camera_tracking.py --check-only

Two synchronized videos:

    python lesson50_multi_camera_tracking.py \
        --camera-a camera_a.mp4 \
        --camera-b camera_b.mp4 \
        --checkpoint runs/lesson49/reid_encoder.pt

Set camera transition time constraints:

    python lesson50_multi_camera_tracking.py \
        --camera-a camera_a.mp4 \
        --camera-b camera_b.mp4 \
        --checkpoint runs/lesson49/reid_encoder.pt \
        --min-travel-time 0.5 \
        --max-travel-time 15.0

======================================================================
OUTPUT
======================================================================

    runs/lesson50/
        multi_camera_tracking.mp4
        global_identity_summary.json
======================================================================
"""

import argparse
import json
import time

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment


# =====================================================================
# CONFIGURATION
# =====================================================================

IMAGE_SIZE = 64

EMBEDDING_DIM = 128

DEFAULT_APPEARANCE_THRESHOLD = 0.45

DEFAULT_LOCAL_MAX_DISTANCE = 120.0

DEFAULT_LOCAL_MAX_AGE = 20

DEFAULT_GLOBAL_MAX_AGE = 30.0

DEFAULT_MIN_TRAVEL_TIME = 0.0

DEFAULT_MAX_TRAVEL_TIME = 30.0


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed=42):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =====================================================================
# GEOMETRY
# =====================================================================

def bbox_center(box):

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


def bbox_area(box):

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


# =====================================================================
# IMAGE PREPROCESSING
# =====================================================================

def crop_object(
    frame,
    bbox,
    image_size=IMAGE_SIZE
):

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = np.asarray(
        bbox,
        dtype=np.float64
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

            (
                image_size,
                image_size,
                3
            ),

            dtype=np.uint8
        )

    crop = frame[
        y1:y2,
        x1:x2
    ]

    crop = cv2.resize(

        crop,

        (image_size, image_size),

        interpolation=cv2.INTER_AREA
    )

    return cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )


def image_to_tensor(image):

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    return torch.from_numpy(
        image.copy()
    )


# =====================================================================
# LESSON 49 APPEARANCE ENCODER
# =====================================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(

                in_channels,

                out_channels,

                kernel_size=3,

                stride=stride,

                padding=1,

                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


class AppearanceEncoder(nn.Module):

    """
    Architecture-compatible with Lesson 49.

    This network is NOT pretrained.

    It must be loaded from a Lesson 49 checkpoint trained on
    real object crops.
    """

    def __init__(
        self,
        embedding_dim=EMBEDDING_DIM
    ):

        super().__init__()

        self.features = nn.Sequential(

            ConvBlock(
                3,
                32,
                stride=2
            ),

            ConvBlock(
                32,
                64,
                stride=2
            ),

            ConvBlock(
                64,
                128,
                stride=2
            ),

            ConvBlock(
                128,
                256,
                stride=2
            ),

            ConvBlock(
                256,
                256,
                stride=1
            )
        )

        self.pool = nn.AdaptiveAvgPool2d(
            1
        )

        self.projection = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                256,
                embedding_dim
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.projection(x)

        return F.normalize(
            x,
            p=2,
            dim=1
        )


# =====================================================================
# MODEL LOADING
# =====================================================================

def load_appearance_model(
    checkpoint_path,
    device="auto"
):

    if device == "auto":

        device = (

            "cuda"

            if torch.cuda.is_available()

            else "cpu"
        )

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(

            f"Checkpoint not found: {checkpoint_path}\n"

            "Train an appearance encoder with Lesson 49 first."
        )

    try:

        checkpoint = torch.load(

            checkpoint_path,

            map_location=device,

            weights_only=True
        )

    except TypeError:

        # Compatibility with older PyTorch versions.

        checkpoint = torch.load(

            checkpoint_path,

            map_location=device
        )

    embedding_dim = checkpoint.get(
        "embedding_dim",
        EMBEDDING_DIM
    )

    model = AppearanceEncoder(
        embedding_dim=embedding_dim
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, device


# =====================================================================
# EMBEDDING EXTRACTION
# =====================================================================

def extract_embedding(
    model,
    frame,
    bbox,
    device
):

    crop = crop_object(
        frame,
        bbox
    )

    tensor = image_to_tensor(
        crop
    ).unsqueeze(0).to(
        device
    )

    with torch.no_grad():

        embedding = model(
            tensor
        )

    return embedding.cpu().numpy()[0]


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

    similarity = np.dot(
        a,
        b
    ) / denominator

    return float(
        1.0 - np.clip(similarity, -1.0, 1.0)
    )


def normalize_embedding(embedding):

    embedding = np.asarray(
        embedding,
        dtype=np.float64
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm < 1e-12:

        return np.zeros_like(
            embedding
        )

    return embedding / norm


# =====================================================================
# MOTION DETECTOR
# =====================================================================

class MotionDetector:

    """
    Demonstration detector based on background subtraction.

    MOG2 detects moving foreground regions, not object categories.
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

                    "class_id": 0
                }
            )

        return detections, mask


# =====================================================================
# LOCAL TRACK
# =====================================================================

@dataclass
class LocalTrack:

    camera_id: int

    local_id: int

    global_id: int

    bbox: np.ndarray

    embedding: np.ndarray

    class_id: int

    last_seen: float

    missed: int = 0

    hits: int = 1

    trajectory: list = field(
        default_factory=list
    )

    def update(
        self,
        detection,
        embedding,
        timestamp,
        alpha=0.2
    ):

        self.bbox = np.asarray(
            detection["bbox"],
            dtype=np.float64
        ).copy()

        new_embedding = (

            (1.0 - alpha) * self.embedding

            + alpha * embedding
        )

        self.embedding = normalize_embedding(
            new_embedding
        )

        self.last_seen = timestamp

        self.missed = 0

        self.hits += 1

        self.trajectory.append(
            bbox_center(self.bbox).tolist()
        )


# =====================================================================
# LOCAL CAMERA TRACKER
# =====================================================================

class LocalCameraTracker:

    """
    Independent local tracker for one camera.

    It assigns local detections to local tracks using:

        - center distance
        - IoU
        - appearance distance

    New local tracks receive global IDs from the shared manager.
    """

    def __init__(

        self,

        camera_id,

        global_manager,

        max_distance=DEFAULT_LOCAL_MAX_DISTANCE,

        max_age=DEFAULT_LOCAL_MAX_AGE,

        appearance_weight=0.35,

        motion_weight=0.35,

        iou_weight=0.30,

        max_cost=0.90
    ):

        self.camera_id = camera_id

        self.global_manager = global_manager

        self.max_distance = max_distance

        self.max_age = max_age

        self.appearance_weight = appearance_weight

        self.motion_weight = motion_weight

        self.iou_weight = iou_weight

        self.max_cost = max_cost

        self.tracks = {}

        self.next_local_id = 1

    def _cost(
        self,
        track,
        detection,
        embedding
    ):

        if track.class_id != detection["class_id"]:

            return np.inf

        distance = center_distance(

            track.bbox,

            detection["bbox"]
        )

        if distance > self.max_distance:

            return np.inf

        appearance = cosine_distance(

            track.embedding,

            embedding
        )

        motion = (
            distance / self.max_distance
        )

        overlap = box_iou(

            track.bbox,

            detection["bbox"]
        )

        return (

            self.appearance_weight * appearance

            + self.motion_weight * motion

            + self.iou_weight * (1.0 - overlap)
        )

    def update(
        self,
        frame,
        detections,
        model,
        device,
        timestamp
    ):

        embeddings = [

            extract_embedding(

                model,

                frame,

                detection["bbox"],

                device
            )

            for detection in detections
        ]

        track_ids = list(
            self.tracks.keys()
        )

        cost = np.full(

            (
                len(track_ids),

                len(detections)
            ),

            np.inf,

            dtype=np.float64
        )

        for i, local_id in enumerate(track_ids):

            track = self.tracks[local_id]

            for j, detection in enumerate(detections):

                cost[i, j] = self._cost(

                    track,

                    detection,

                    embeddings[j]
                )

        if cost.size:

            rows, cols = linear_sum_assignment(

                np.where(
                    np.isfinite(cost),
                    cost,
                    1e6
                )
            )

        else:

            rows = []

            cols = []

        matched_tracks = set()

        matched_detections = set()

        for row, col in zip(rows, cols):

            if not np.isfinite(cost[row, col]):
                continue

            if cost[row, col] > self.max_cost:
                continue

            local_id = track_ids[row]

            self.tracks[local_id].update(

                detections[col],

                embeddings[col],

                timestamp
            )

            matched_tracks.add(
                local_id
            )

            matched_detections.add(
                col
            )

        # Age unmatched tracks.

        expired = []

        for local_id, track in self.tracks.items():

            if local_id not in matched_tracks:

                track.missed += 1

            if track.missed > self.max_age:

                expired.append(
                    local_id
                )

        for local_id in expired:

            del self.tracks[local_id]

        # Create local tracks for unmatched detections.

        for index, detection in enumerate(detections):

            if index in matched_detections:
                continue

            global_id = self.global_manager.assign(

                camera_id=self.camera_id,

                embedding=embeddings[index],

                timestamp=timestamp
            )

            local_id = self.next_local_id

            self.next_local_id += 1

            track = LocalTrack(

                camera_id=self.camera_id,

                local_id=local_id,

                global_id=global_id,

                bbox=np.asarray(
                    detection["bbox"],
                    dtype=np.float64
                ).copy(),

                embedding=embeddings[index].copy(),

                class_id=detection["class_id"],

                last_seen=timestamp
            )

            track.trajectory.append(
                bbox_center(track.bbox).tolist()
            )

            self.tracks[local_id] = track

        return [

            track

            for track in self.tracks.values()

            if track.missed == 0
        ]


# =====================================================================
# GLOBAL IDENTITY RECORD
# =====================================================================

@dataclass
class GlobalIdentity:

    global_id: int

    prototype: np.ndarray

    last_camera: int

    last_seen: float

    observations: int = 1

    cameras_seen: set = field(
        default_factory=set
    )

    def update(
        self,
        embedding,
        camera_id,
        timestamp,
        alpha=0.15
    ):

        updated = (

            (1.0 - alpha) * self.prototype

            + alpha * embedding
        )

        self.prototype = normalize_embedding(
            updated
        )

        self.last_camera = camera_id

        self.last_seen = timestamp

        self.observations += 1

        self.cameras_seen.add(
            camera_id
        )


# =====================================================================
# GLOBAL IDENTITY MANAGER
# =====================================================================

class GlobalIdentityManager:

    """
    Shared global identity gallery.

    The manager decides whether a new local track belongs to:

        - an existing global identity, or
        - a new physical object.

    The manager checks:

        1. Appearance similarity.
        2. Camera transition validity.
        3. Time since the last observation.
        4. Whether this global ID is already assigned in the
           destination camera at the same timestamp.

    Camera topology:
        allowed_transitions = {
            0: {1},
            1: {0}
        }

    This means transitions are allowed in both directions.
    """

    def __init__(

        self,

        appearance_threshold=DEFAULT_APPEARANCE_THRESHOLD,

        min_travel_time=DEFAULT_MIN_TRAVEL_TIME,

        max_travel_time=DEFAULT_MAX_TRAVEL_TIME,

        max_identity_age=DEFAULT_GLOBAL_MAX_AGE,

        allowed_transitions=None,

        prototype_alpha=0.15,

        simultaneous_tolerance=0.05
    ):

        self.appearance_threshold = appearance_threshold

        self.min_travel_time = min_travel_time

        self.max_travel_time = max_travel_time

        self.max_identity_age = max_identity_age

        self.prototype_alpha = prototype_alpha

        self.simultaneous_tolerance = simultaneous_tolerance

        self.identities = {}

        self.next_global_id = 1

        self.events = {

            "new_identities": 0,

            "cross_camera_matches": 0,

            "rejected_appearance": 0,

            "rejected_transition": 0,

            "rejected_time": 0
        }

        if allowed_transitions is None:

            allowed_transitions = {

                0: {1},

                1: {0}
            }

        self.allowed_transitions = {

            int(camera): set(

                int(destination)

                for destination in destinations
            )

            for camera, destinations in allowed_transitions.items()
        }

    def transition_allowed(
        self,
        source_camera,
        destination_camera
    ):

        return destination_camera in self.allowed_transitions.get(

            source_camera,

            set()
        )

    def _already_assigned_in_camera(
        self,
        global_id,
        camera_id,
        timestamp
    ):

        """
        Prevent two different local tracks in the same camera
        from receiving the same global ID at the same time.

        The same global ID may be visible in two different cameras
        simultaneously if the cameras overlap.
        """

        for identity in self.identities.values():

            if identity.global_id != global_id:
                continue

            if identity.last_camera != camera_id:
                continue

            if abs(identity.last_seen - timestamp) <= self.simultaneous_tolerance:

                return True

        return False

    def _candidate_cost(
        self,
        identity,
        camera_id,
        embedding,
        timestamp
    ):

        if identity.last_camera == camera_id:

            return np.inf

        if not self.transition_allowed(

            identity.last_camera,

            camera_id
        ):

            self.events["rejected_transition"] += 1

            return np.inf

        elapsed = timestamp - identity.last_seen

        if elapsed < self.min_travel_time:

            self.events["rejected_time"] += 1

            return np.inf

        if elapsed > self.max_travel_time:

            self.events["rejected_time"] += 1

            return np.inf

        if elapsed > self.max_identity_age:

            return np.inf

        if self._already_assigned_in_camera(

            identity.global_id,

            camera_id,

            timestamp
        ):

            return np.inf

        distance = cosine_distance(

            identity.prototype,

            embedding
        )

        if distance > self.appearance_threshold:

            self.events["rejected_appearance"] += 1

            return np.inf

        # Appearance is the principal association signal.
        # A small time penalty prefers more plausible transitions.

        time_penalty = (

            elapsed / max(
                self.max_travel_time,
                1e-6
            )
        )

        return distance + 0.02 * time_penalty

    def assign(
        self,
        camera_id,
        embedding,
        timestamp
    ):

        camera_id = int(
            camera_id
        )

        embedding = normalize_embedding(
            embedding
        )

        candidates = []

        for global_id, identity in self.identities.items():

            cost = self._candidate_cost(

                identity,

                camera_id,

                embedding,

                timestamp
            )

            if np.isfinite(cost):

                candidates.append(
                    (cost, global_id)
                )

        if candidates:

            candidates.sort(
                key=lambda item: item[0]
            )

            best_cost, best_id = candidates[0]

            if best_cost <= self.appearance_threshold:

                identity = self.identities[best_id]

                identity.update(

                    embedding,

                    camera_id,

                    timestamp,

                    alpha=self.prototype_alpha
                )

                self.events["cross_camera_matches"] += 1

                return best_id

        # No suitable existing identity: create a new one.

        global_id = self.next_global_id

        self.next_global_id += 1

        self.identities[global_id] = GlobalIdentity(

            global_id=global_id,

            prototype=embedding.copy(),

            last_camera=camera_id,

            last_seen=timestamp,

            observations=1,

            cameras_seen={camera_id}
        )

        self.events["new_identities"] += 1

        return global_id

    def summary(self):

        return {

            "global_identity_count": len(
                self.identities
            ),

            "next_global_id": self.next_global_id,

            "events": self.events.copy(),

            "identities": [

                {

                    "global_id": identity.global_id,

                    "last_camera": identity.last_camera,

                    "last_seen": identity.last_seen,

                    "observations": identity.observations,

                    "cameras_seen": sorted(
                        identity.cameras_seen
                    )
                }

                for identity in self.identities.values()
            ]
        }


# =====================================================================
# VISUALIZATION
# =====================================================================

def draw_camera_tracks(
    frame,
    tracks,
    camera_id,
    frame_index
):

    output = frame.copy()

    for track in tracks:

        x1, y1, x2, y2 = track.bbox.astype(
            int
        )

        global_id = track.global_id

        color = (

            int(global_id * 71 % 255),

            int(global_id * 137 % 255),

            int(global_id * 193 % 255)
        )

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            color,

            2
        )

        label = (

            f"Global {global_id} "

            f"| Cam {camera_id} "

            f"| Local {track.local_id}"
        )

        cv2.putText(

            output,

            label,

            (x1, max(20, y1 - 7)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.48,

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

    cv2.putText(

        output,

        f"Camera {camera_id} | Frame {frame_index}",

        (15, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )

    return output


# =====================================================================
# VIDEO HELPERS
# =====================================================================

def open_video(path):

    capture = cv2.VideoCapture(
        str(path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Cannot open video: {path}"
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

    return capture, width, height, fps


def resize_to_height(
    image,
    target_height
):

    height, width = image.shape[:2]

    if height == target_height:

        return image

    target_width = max(

        1,

        int(
            width * target_height / height
        )
    )

    return cv2.resize(

        image,

        (target_width, target_height),

        interpolation=cv2.INTER_AREA
    )


# =====================================================================
# MULTI-CAMERA VIDEO PROCESSING
# =====================================================================

def process_two_cameras(args):

    model, device = load_appearance_model(

        args.checkpoint,

        args.device
    )

    cap_a, width_a, height_a, fps_a = open_video(
        args.camera_a
    )

    cap_b, width_b, height_b, fps_b = open_video(
        args.camera_b
    )

    # This implementation pairs frames by index.
    # It assumes synchronized videos with compatible frame rates.

    if abs(fps_a - fps_b) > 0.5:

        print(
            "WARNING: Camera FPS values differ. "
            "Frame-index synchronization may be inaccurate."
        )

    output_fps = min(
        fps_a,
        fps_b
    )

    global_manager = GlobalIdentityManager(

        appearance_threshold=args.appearance_threshold,

        min_travel_time=args.min_travel_time,

        max_travel_time=args.max_travel_time,

        max_identity_age=args.max_identity_age,

        allowed_transitions={

            0: {1},

            1: {0}
        }
    )

    detector_a = MotionDetector(
        min_area=args.min_area
    )

    detector_b = MotionDetector(
        min_area=args.min_area
    )

    tracker_a = LocalCameraTracker(

        camera_id=0,

        global_manager=global_manager,

        max_distance=args.local_max_distance,

        max_age=args.local_max_age
    )

    tracker_b = LocalCameraTracker(

        camera_id=1,

        global_manager=global_manager,

        max_distance=args.local_max_distance,

        max_age=args.local_max_age
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True
    )

    target_height = min(
        height_a,
        height_b
    )

    output_width_a = int(
        width_a * target_height / height_a
    )

    output_width_b = int(
        width_b * target_height / height_b
    )

    combined_width = (
        output_width_a + output_width_b
    )

    writer = cv2.VideoWriter(

        str(output_path),

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        output_fps,

        (
            combined_width,
            target_height
        )
    )

    if not writer.isOpened():

        cap_a.release()

        cap_b.release()

        raise RuntimeError(
            "Could not open output video writer."
        )

    frame_index = 0

    processing_times = []

    try:

        while True:

            start = time.perf_counter()

            ret_a, frame_a = cap_a.read()

            ret_b, frame_b = cap_b.read()

            if not ret_a or not ret_b:
                break

            frame_index += 1

            # Frame-index timestamp.
            # Correct only when both videos are synchronized.

            timestamp = (
                (frame_index - 1) / output_fps
            )

            detections_a, _ = detector_a.detect(
                frame_a
            )

            detections_b, _ = detector_b.detect(
                frame_b
            )

            tracks_a = tracker_a.update(

                frame_a,

                detections_a,

                model,

                device,

                timestamp
            )

            tracks_b = tracker_b.update(

                frame_b,

                detections_b,

                model,

                device,

                timestamp
            )

            view_a = draw_camera_tracks(

                frame_a,

                tracks_a,

                0,

                frame_index
            )

            view_b = draw_camera_tracks(

                frame_b,

                tracks_b,

                1,

                frame_index
            )

            view_a = resize_to_height(

                view_a,

                target_height
            )

            view_b = resize_to_height(

                view_b,

                target_height
            )

            combined = np.concatenate(

                [
                    view_a,
                    view_b
                ],

                axis=1
            )

            elapsed = (
                time.perf_counter() - start
            )

            processing_times.append(
                elapsed
            )

            writer.write(
                combined
            )

            if not args.no_display:

                cv2.imshow(

                    "Camera A | Camera B | Global IDs",

                    combined
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_index % 100 == 0:

                summary = global_manager.summary()

                print(

                    f"Frame {frame_index} | "

                    f"Global identities: "

                    f"{summary['global_identity_count']} | "

                    f"Cross-camera matches: "

                    f"{summary['events']['cross_camera_matches']}"
                )

    finally:

        cap_a.release()

        cap_b.release()

        writer.release()

        cv2.destroyAllWindows()

    average_time = (

        float(np.mean(processing_times))

        if processing_times else 0.0
    )

    summary = global_manager.summary()

    summary.update(

        {

            "camera_a": args.camera_a,

            "camera_b": args.camera_b,

            "output_video": str(output_path),

            "frames_processed": frame_index,

            "assumed_synchronized": True,

            "camera_a_fps": fps_a,

            "camera_b_fps": fps_b,

            "output_fps": output_fps,

            "device": device,

            "average_processing_fps": (

                1.0 / average_time

                if average_time > 0 else 0.0
            )
        }
    )

    summary_path = (

        output_path.parent

        / "global_identity_summary.json"
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

    print("\nMulti-camera tracking completed.")

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

    print("LESSON 50 — MULTI-CAMERA TRACKING TESTS")

    print("=" * 70)

    # ---------------------------------------------------------------
    # Test 1: normalized embeddings.
    # ---------------------------------------------------------------

    embedding = np.array(
        [3.0, 4.0]
    )

    normalized = normalize_embedding(
        embedding
    )

    assert np.allclose(

        np.linalg.norm(normalized),

        1.0
    )

    print("Test 1 passed: embedding normalization")

    # ---------------------------------------------------------------
    # Test 2: cosine distance.
    # ---------------------------------------------------------------

    a = np.array(
        [1.0, 0.0]
    )

    b = np.array(
        [1.0, 0.0]
    )

    assert abs(
        cosine_distance(a, b)
    ) < 1e-8

    print("Test 2 passed: identical appearance distance")

    # ---------------------------------------------------------------
    # Test 3: orthogonal vectors.
    # ---------------------------------------------------------------

    a = np.array(
        [1.0, 0.0]
    )

    b = np.array(
        [0.0, 1.0]
    )

    assert abs(
        cosine_distance(a, b) - 1.0
    ) < 1e-8

    print("Test 3 passed: orthogonal appearance distance")

    # ---------------------------------------------------------------
    # Test 4: bounding-box IoU.
    # ---------------------------------------------------------------

    box = np.array(
        [0, 0, 100, 100]
    )

    assert abs(
        box_iou(box, box) - 1.0
    ) < 1e-8

    print("Test 4 passed: bounding-box IoU")

    # ---------------------------------------------------------------
    # Test 5: camera transition.
    # ---------------------------------------------------------------

    manager = GlobalIdentityManager(

        allowed_transitions={

            0: {1},

            1: {0}
        }
    )

    assert manager.transition_allowed(
        0,
        1
    )

    assert manager.transition_allowed(
        1,
        0
    )

    assert not manager.transition_allowed(
        0,
        2
    )

    print("Test 5 passed: camera topology")

    # ---------------------------------------------------------------
    # Test 6: first object creates a global identity.
    # ---------------------------------------------------------------

    manager = GlobalIdentityManager()

    embedding = np.zeros(
        8
    )

    embedding[0] = 1.0

    global_id = manager.assign(

        camera_id=0,

        embedding=embedding,

        timestamp=0.0
    )

    assert global_id == 1

    assert len(manager.identities) == 1

    print("Test 6 passed: global identity creation")

    # ---------------------------------------------------------------
    # Test 7: same appearance in Camera B reuses global ID.
    # ---------------------------------------------------------------

    global_id_b = manager.assign(

        camera_id=1,

        embedding=embedding,

        timestamp=2.0
    )

    assert global_id_b == global_id

    assert manager.events["cross_camera_matches"] == 1

    print("Test 7 passed: cross-camera identity association")

    # ---------------------------------------------------------------
    # Test 8: camera transition is rejected.
    # ---------------------------------------------------------------

    manager = GlobalIdentityManager(

        allowed_transitions={

            0: set(),

            1: set()
        }
    )

    global_id_a = manager.assign(

        0,

        embedding,

        0.0
    )

    global_id_b = manager.assign(

        1,

        embedding,

        1.0
    )

    assert global_id_a != global_id_b

    print("Test 8 passed: forbidden transition creates new ID")

    # ---------------------------------------------------------------
    # Test 9: appearance mismatch creates a new ID.
    # ---------------------------------------------------------------

    manager = GlobalIdentityManager(
        appearance_threshold=0.2
    )

    first = np.array(
        [1.0, 0.0, 0.0]
    )

    second = np.array(
        [0.0, 1.0, 0.0]
    )

    id_a = manager.assign(
        0,
        first,
        0.0
    )

    id_b = manager.assign(
        1,
        second,
        1.0
    )

    assert id_a != id_b

    print("Test 9 passed: appearance mismatch rejection")

    # ---------------------------------------------------------------
    # Test 10: minimum travel time.
    # ---------------------------------------------------------------

    manager = GlobalIdentityManager(

        min_travel_time=2.0,

        max_travel_time=10.0
    )

    id_a = manager.assign(
        0,
        first,
        0.0
    )

    id_b = manager.assign(
        1,
        first,
        1.0
    )

    assert id_a != id_b

    print("Test 10 passed: minimum travel-time constraint")

    # ---------------------------------------------------------------
    # Test 11: maximum travel time.
    # ---------------------------------------------------------------

    manager = GlobalIdentityManager(

        min_travel_time=0.0,

        max_travel_time=5.0
    )

    id_a = manager.assign(
        0,
        first,
        0.0
    )

    id_b = manager.assign(
        1,
        first,
        10.0
    )

    assert id_a != id_b

    print("Test 11 passed: maximum travel-time constraint")

    # ---------------------------------------------------------------
    # Test 12: simultaneous visibility across cameras.
    # ---------------------------------------------------------------

    manager = GlobalIdentityManager(

        min_travel_time=0.0,

        max_travel_time=10.0
    )

    id_a = manager.assign(
        0,
        first,
        1.0
    )

    id_b = manager.assign(
        1,
        first,
        1.0
    )

    assert id_a == id_b

    print("Test 12 passed: simultaneous cross-camera visibility")

    # ---------------------------------------------------------------
    # Test 13: duplicate assignment in the same camera is blocked.
    # ---------------------------------------------------------------

    manager = GlobalIdentityManager()

    id_a = manager.assign(
        0,
        first,
        1.0
    )

    # A second local track in Camera 0 at the same timestamp
    # must not be assigned to the same global identity.

    id_b = manager.assign(
        1,
        first,
        1.0
    )

    id_c = manager.assign(
        0,
        first,
        1.0
    )

    assert id_a == id_b

    assert id_c != id_a

    print("Test 13 passed: duplicate same-camera assignment prevention")

    # ---------------------------------------------------------------
    # Test 14: global prototype update.
    # ---------------------------------------------------------------

    manager = GlobalIdentityManager()

    id_a = manager.assign(
        0,
        first,
        0.0
    )

    identity = manager.identities[id_a]

    identity.update(

        np.array(
            [0.9, 0.1, 0.0]
        ),

        camera_id=0,

        timestamp=1.0
    )

    assert abs(
        np.linalg.norm(identity.prototype) - 1.0
    ) < 1e-8

    assert identity.observations == 2

    print("Test 14 passed: global appearance prototype update")

    # ---------------------------------------------------------------
    # Test 15: camera history.
    # ---------------------------------------------------------------

    manager = GlobalIdentityManager()

    identity_id = manager.assign(
        0,
        first,
        0.0
    )

    manager.assign(
        1,
        first,
        1.0
    )

    cameras_seen = manager.identities[
        identity_id
    ].cameras_seen

    assert cameras_seen == {
        0,
        1
    }

    print("Test 15 passed: global camera history")

    # ---------------------------------------------------------------
    # Test 16: local track update.
    # ---------------------------------------------------------------

    detection = {

        "bbox": np.array(
            [10, 10, 50, 50]
        ),

        "class_id": 0
    }

    track = LocalTrack(

        camera_id=0,

        local_id=1,

        global_id=1,

        bbox=detection["bbox"],

        embedding=first,

        class_id=0,

        last_seen=0.0
    )

    track.update(

        {

            "bbox": np.array(
                [20, 20, 60, 60]
            ),

            "class_id": 0
        },

        first,

        1.0
    )

    assert track.hits == 2

    assert track.last_seen == 1.0

    assert np.allclose(

        track.bbox,

        [20, 20, 60, 60]
    )

    print("Test 16 passed: local track update")

    # ---------------------------------------------------------------
    # Test 17: CNN embedding dimensions.
    # ---------------------------------------------------------------

    model = AppearanceEncoder()

    model.eval()

    x = torch.rand(
        2,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    with torch.no_grad():

        output = model(x)

    assert output.shape == (
        2,
        EMBEDDING_DIM
    )

    print("Test 17 passed: CNN embedding dimensions")

    # ---------------------------------------------------------------
    # Test 18: CNN output normalization.
    # ---------------------------------------------------------------

    norms = torch.norm(
        output,
        dim=1
    )

    assert torch.allclose(

        norms,

        torch.ones_like(norms),

        atol=1e-5
    )

    print("Test 18 passed: CNN embedding normalization")

    # ---------------------------------------------------------------
    # Test 19: image crop preprocessing.
    # ---------------------------------------------------------------

    image = np.random.default_rng(
        42
    ).integers(

        0,

        256,

        size=(100, 120, 3),

        dtype=np.uint8
    )

    crop = crop_object(

        image,

        [10, 10, 60, 60]
    )

    assert crop.shape == (
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )

    print("Test 19 passed: object crop preprocessing")

    # ---------------------------------------------------------------
    # Test 20: global identity summary.
    # ---------------------------------------------------------------

    manager = GlobalIdentityManager()

    manager.assign(
        0,
        first,
        0.0
    )

    summary = manager.summary()

    assert summary["global_identity_count"] == 1

    assert len(
        summary["identities"]
    ) == 1

    print("Test 20 passed: global identity summary")

    print("\nAll 20 tests passed.")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Lesson 50 — Multi-Camera Object Tracking"
        )
    )

    parser.add_argument(
        "--camera-a",
        type=str,
        default=None
    )

    parser.add_argument(
        "--camera-b",
        type=str,
        default=None
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/lesson49/reid_encoder.pt"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/lesson50/multi_camera_tracking.mp4"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"]
    )

    parser.add_argument(
        "--appearance-threshold",
        type=float,
        default=DEFAULT_APPEARANCE_THRESHOLD
    )

    parser.add_argument(
        "--min-travel-time",
        type=float,
        default=DEFAULT_MIN_TRAVEL_TIME
    )

    parser.add_argument(
        "--max-travel-time",
        type=float,
        default=DEFAULT_MAX_TRAVEL_TIME
    )

    parser.add_argument(
        "--max-identity-age",
        type=float,
        default=DEFAULT_GLOBAL_MAX_AGE
    )

    parser.add_argument(
        "--local-max-distance",
        type=float,
        default=DEFAULT_LOCAL_MAX_DISTANCE
    )

    parser.add_argument(
        "--local-max-age",
        type=int,
        default=DEFAULT_LOCAL_MAX_AGE
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

    if args.min_travel_time < 0:

        parser.error(
            "--min-travel-time cannot be negative."
        )

    if args.max_travel_time <= 0:

        parser.error(
            "--max-travel-time must be positive."
        )

    if args.min_travel_time > args.max_travel_time:

        parser.error(
            "Minimum travel time cannot exceed maximum travel time."
        )

    if args.check_only:

        run_tests()

        return

    if args.camera_a is None or args.camera_b is None:

        parser.error(
            "Both --camera-a and --camera-b are required."
        )

    set_seed(42)

    print("=" * 70)

    print(
        "LESSON 50 — MULTI-CAMERA TRACKING"
    )

    print("=" * 70)

    process_two_cameras(args)


if __name__ == "__main__":

    main()