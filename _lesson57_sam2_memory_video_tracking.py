"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 57 — SAM 2-INSPIRED MEMORY VIDEO OBJECT TRACKING
======================================================================

FILENAME
--------
lesson57_sam2_memory_video_tracking.py

IMPORTANT ARCHITECTURE NOTE
---------------------------
This is an educational SAM 2-inspired implementation.

It is NOT the official SAM 2 model.

It does not use:
    - SAM 2 source code
    - Pretrained SAM 2 weights
    - A pretrained image encoder
    - A trained transformer
    - A learned mask decoder

Instead, it implements simplified versions of important concepts:

    1. Prompt-based object initialization.
    2. Foreground/background visual feature extraction.
    3. A temporal memory bank.
    4. Memory-conditioned pixel classification.
    5. Optical-flow mask propagation.
    6. Confidence-based memory updates.
    7. Bounded memory management.

The goal is to understand why temporal memory is useful for
video object segmentation.

======================================================================
1. WHY VIDEO SEGMENTATION NEEDS MEMORY
======================================================================

Consider a bottle moving along a conveyor.

Frame 1:
    The bottle is clearly visible.

Frame 2:
    The bottle moves.

Frame 3:
    Part of the bottle is occluded.

Frame 4:
    The bottle reappears.

An image-only segmentation algorithm sees each frame independently.

It may confuse:
    - The bottle with a similarly colored object.
    - The bottle with the conveyor background.
    - The bottle before and after an occlusion.

A video segmentation system should remember what the object looked
like in earlier frames.

Conceptually:

    Frame 1
       |
       v
    Initial object mask
       |
       v
    Visual memory
       |
       +--------------------+
       |                    |
       v                    |
    Frame 2                 |
       |                    |
       v                    |
    Memory matching         |
       |                    |
       v                    |
    New object mask         |
       |                    |
       v                    |
    Update memory ----------+
       |
       v
    Frame 3
       |
       v
    Repeat

======================================================================
2. SAM 2 CONCEPTUAL ARCHITECTURE
======================================================================

The official SAM 2 architecture contains several learned components.

A simplified conceptual view:

    Current video frame
           |
           v
    Image encoder
           |
           v
    Current visual features
           |
           +--------------------------+
           |                          |
           v                          v
    Prompt encoder              Memory encoder
           |                          |
           v                          v
    Prompt information           Memory features
           |                          |
           +------------+-------------+
                        |
                        v
                 Memory attention
                        |
                        v
                  Mask decoder
                        |
                        v
                 Current object mask
                        |
                        v
                  Memory update

Our educational implementation replaces learned components with:

    Image encoder:
        Lab color + image gradients.

    Prompt encoder:
        A manually selected rectangle and GrabCut.

    Memory encoder:
        Samples foreground/background feature vectors.

    Memory attention:
        K-nearest visual-feature matching.

    Mask decoder:
        A foreground probability calculation.

    Temporal propagation:
        Optical flow.

This is deliberately much simpler than SAM 2.

======================================================================
3. VISUAL FEATURE EXTRACTION
======================================================================

For every pixel, construct a feature vector:

    f(x,y) = [L, a, b, gx, gy]

Where:

    L, a, b:
        CIELAB color channels.

    gx, gy:
        Image intensity gradients.

Color helps identify appearance.

Gradients provide local edge/orientation information.

All channels are normalized.

The resulting feature map has shape:

    H x W x 5

======================================================================
4. TEMPORAL MEMORY BANK
======================================================================

We maintain two memory banks:

    Foreground memory:
        Features belonging to the selected object.

    Background memory:
        Features belonging to the surrounding scene.

Each memory contains representative pixel feature vectors.

For a current pixel feature f, calculate its distance to stored
memory features:

    d_i^2 = sum_j w_j (f_j - m_ij)^2

where:

    f:
        Current pixel feature.

    m_i:
        Memory feature i.

    w_j:
        Feature-channel weight.

We select the K nearest memory features.

Convert distances into similarities:

    s_i = exp(-d_i^2 / temperature)

Then calculate a weighted foreground probability:

    P(FG | f) =
        sum(s_i for foreground memory)
        --------------------------------
        sum(s_i for foreground and background memory)

This is a simple non-parametric appearance classifier.

It is NOT the transformer attention mechanism used by SAM 2.

======================================================================
5. COMBINING MEMORY WITH OPTICAL FLOW
======================================================================

Memory matching can identify pixels that look like the object.

Optical flow can identify where the object moved.

They provide complementary information.

Let:

    P_memory:
        Foreground probability from visual memory.

    P_flow:
        Propagated mask probability.

Combine them:

    P_final =
        alpha * P_memory
        +
        (1 - alpha) * P_flow

The final mask is:

    M(x,y) = 1 if P_final(x,y) >= threshold

The memory and flow weights are configurable.

======================================================================
6. MEMORY UPDATE
======================================================================

Memory should not blindly learn every prediction.

A wrong mask can contaminate memory and cause identity drift.

We update using confident predictions only:

    Foreground:
        P_final >= foreground_update_threshold

    Background:
        P_final <= background_update_threshold

We also limit the number of new samples per frame.

The memory bank has a maximum capacity.

This prevents unbounded memory growth.

IMPORTANT:
    Confidence is a heuristic in this educational implementation.
    It is not a calibrated probability of correctness.

======================================================================
7. ROBOTICS APPLICATION — OPENARM V2
======================================================================

Example waste-sorting task:

    Camera
       |
       v
    Operator selects bottle
       |
       v
    Initial segmentation
       |
       v
    Visual memory creation
       |
       v
    Bottle moves on conveyor
       |
       v
    Memory-conditioned segmentation
       |
       v
    Object mask
       |
       v
    Object centroid / contour
       |
       v
    Depth estimation
       |
       v
    Camera-to-robot calibration
       |
       v
    OpenArm V2 grasp planning

Memory is especially useful when:
    - The object moves through the camera view.
    - The background is visually complex.
    - The object is temporarily occluded.
    - Several objects have similar shapes.

A single-object memory bank does not solve multi-object identity
management. Each tracked object needs its own identity and memory.

======================================================================
8. INPUT
======================================================================

Run with a real video:

    python lesson57_sam2_memory_video_tracking.py \
        --video conveyor_video.mp4

Select the object in the first frame.

Controls:

    q:
        Quit.

    p:
        Pause/resume.

    r:
        Reinitialize the selected object.

    + / -:
        Increase/decrease memory contribution.

======================================================================
9. OUTPUT
======================================================================

    runs/lesson57/
        sam2_inspired_memory_tracking.mp4
        memory_tracking_summary.json

======================================================================
10. TESTS
======================================================================

    python lesson57_sam2_memory_video_tracking.py --check-only

The tests cover:
    - Feature extraction.
    - Feature-map dimensions.
    - Memory initialization.
    - Memory capacity.
    - Foreground/background matching.
    - Memory probability bounds.
    - Optical-flow propagation.
    - Mask metrics.
    - Memory update thresholds.
    - Empty and invalid inputs.

======================================================================
NEXT LESSON
======================================================================

Lesson 58 — RGB-D Computer Vision

We will move from 2-D pixels to 3-D geometry using RGB images,
depth images, camera intrinsics, and point back-projection.

======================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import time

from dataclasses import dataclass

import cv2
import numpy as np


# ======================================================================
# CONFIGURATION
# ======================================================================

DEFAULT_OUTPUT_DIR = os.path.join(
    "runs",
    "lesson57",
)

FLOW_PARAMETERS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 21,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}

FEATURE_WEIGHTS = np.array(
    [1.0, 1.0, 1.0, 0.25, 0.25],
    dtype=np.float32,
)


# ======================================================================
# BASIC MASK OPERATIONS
# ======================================================================

def to_binary_mask(mask):
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(
            "Mask must be two-dimensional."
        )

    return (
        mask > 0
    ).astype(np.uint8) * 255


def mask_area(mask):
    return int(
        np.count_nonzero(
            to_binary_mask(mask)
        )
    )


def mask_centroid(mask):
    binary = to_binary_mask(mask)

    moments = cv2.moments(
        binary,
        binaryImage=True,
    )

    if moments["m00"] <= 0:
        return None

    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )


def mask_iou(mask_a, mask_b):
    a = to_binary_mask(mask_a) > 0
    b = to_binary_mask(mask_b) > 0

    if a.shape != b.shape:
        raise ValueError(
            "Masks must have identical dimensions."
        )

    intersection = np.count_nonzero(
        a & b
    )

    union = np.count_nonzero(
        a | b
    )

    if union == 0:
        return 1.0

    return float(
        intersection / union
    )


def mask_dice(mask_a, mask_b):
    a = to_binary_mask(mask_a) > 0
    b = to_binary_mask(mask_b) > 0

    if a.shape != b.shape:
        raise ValueError(
            "Masks must have identical dimensions."
        )

    denominator = (
        np.count_nonzero(a)
        + np.count_nonzero(b)
    )

    if denominator == 0:
        return 1.0

    intersection = np.count_nonzero(
        a & b
    )

    return float(
        2.0 * intersection / denominator
    )


# ======================================================================
# IMAGE FEATURE EXTRACTION
# ======================================================================

def extract_features(frame):
    """
    Build a five-dimensional feature vector per pixel:

        [L, a, b, gx, gy]

    The image must be a BGR uint8 image.
    """

    if frame is None:
        raise ValueError(
            "Frame cannot be None."
        )

    if (
        frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError(
            "Expected a BGR image with three channels."
        )

    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32) / 255.0

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    ) / 1020.0

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    ) / 1020.0

    features = np.dstack(
        [
            lab,
            gx,
            gy,
        ]
    ).astype(np.float32)

    return features


# ======================================================================
# MEMORY BANK
# ======================================================================

class VisualMemoryBank:
    """
    A bounded foreground/background feature memory.

    The bank stores feature vectors rather than entire images.

    Foreground and background memories are maintained separately.
    """

    def __init__(
        self,
        max_tokens=512,
        samples_per_update=32,
        seed=7,
    ):
        if max_tokens < 1:
            raise ValueError(
                "max_tokens must be positive."
            )

        if samples_per_update < 1:
            raise ValueError(
                "samples_per_update must be positive."
            )

        self.max_tokens = int(max_tokens)

        self.samples_per_update = int(
            samples_per_update
        )

        self.foreground = np.empty(
            (0, 5),
            dtype=np.float32,
        )

        self.background = np.empty(
            (0, 5),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(
            seed
        )

    @staticmethod
    def _validate_features(features):
        features = np.asarray(
            features,
            dtype=np.float32,
        )

        if (
            features.ndim != 3
            or features.shape[2] != 5
        ):
            raise ValueError(
                "Features must have shape (H, W, 5)."
            )

        return features

    def _sample_features(
        self,
        features,
        selection,
        count,
    ):
        candidates = features[
            selection
        ]

        if len(candidates) == 0:
            return np.empty(
                (0, 5),
                dtype=np.float32,
            )

        number = min(
            int(count),
            len(candidates),
        )

        indices = self.rng.choice(
            len(candidates),
            size=number,
            replace=False,
        )

        return candidates[
            indices
        ].astype(np.float32)

    def _append_bounded(
        self,
        existing,
        new_features,
    ):
        if len(new_features) == 0:
            return existing

        combined = np.concatenate(
            [
                existing,
                new_features,
            ],
            axis=0,
        )

        if len(combined) > self.max_tokens:
            # Retain a random subset to prevent memory growth.
            indices = self.rng.choice(
                len(combined),
                size=self.max_tokens,
                replace=False,
            )

            combined = combined[
                indices
            ]

        return combined.astype(
            np.float32
        )

    def initialize(
        self,
        features,
        mask,
    ):
        features = self._validate_features(
            features
        )

        mask = to_binary_mask(
            mask
        )

        if features.shape[:2] != mask.shape:
            raise ValueError(
                "Feature and mask dimensions do not match."
            )

        foreground_selection = (
            mask > 0
        )

        background_selection = (
            mask == 0
        )

        if not np.any(foreground_selection):
            raise ValueError(
                "Initial mask contains no foreground."
            )

        if not np.any(background_selection):
            raise ValueError(
                "Initial mask contains no background."
            )

        fg = self._sample_features(
            features,
            foreground_selection,
            self.max_tokens,
        )

        bg = self._sample_features(
            features,
            background_selection,
            self.max_tokens,
        )

        self.foreground = self._append_bounded(
            self.foreground,
            fg,
        )

        self.background = self._append_bounded(
            self.background,
            bg,
        )

    def update(
        self,
        features,
        foreground_selection,
        background_selection,
    ):
        features = self._validate_features(
            features
        )

        foreground_selection = np.asarray(
            foreground_selection,
            dtype=bool,
        )

        background_selection = np.asarray(
            background_selection,
            dtype=bool,
        )

        if (
            foreground_selection.shape
            != features.shape[:2]
        ):
            raise ValueError(
                "Foreground selection dimensions do not match."
            )

        if (
            background_selection.shape
            != features.shape[:2]
        ):
            raise ValueError(
                "Background selection dimensions do not match."
            )

        fg = self._sample_features(
            features,
            foreground_selection,
            self.samples_per_update,
        )

        bg = self._sample_features(
            features,
            background_selection,
            self.samples_per_update,
        )

        self.foreground = self._append_bounded(
            self.foreground,
            fg,
        )

        self.background = self._append_bounded(
            self.background,
            bg,
        )

    @property
    def foreground_count(self):
        return len(self.foreground)

    @property
    def background_count(self):
        return len(self.background)


# ======================================================================
# MEMORY MATCHING
# ======================================================================

def nearest_memory_similarity(
    query_features,
    memory_features,
    k=5,
    temperature=0.025,
    weights=None,
    chunk_size=4096,
):
    """
    Calculate a similarity score for each query feature.

    For each query, find the K nearest memory features.

    Similarity:

        s = exp(-distance_squared / temperature)

    Return the weighted average similarity.

    The function processes queries in chunks to limit memory use.
    """

    query_features = np.asarray(
        query_features,
        dtype=np.float32,
    )

    memory_features = np.asarray(
        memory_features,
        dtype=np.float32,
    )

    if query_features.ndim != 2:
        raise ValueError(
            "Query features must be 2-D."
        )

    if memory_features.ndim != 2:
        raise ValueError(
            "Memory features must be 2-D."
        )

    if (
        query_features.shape[1]
        != memory_features.shape[1]
    ):
        raise ValueError(
            "Feature dimensions must match."
        )

    if len(memory_features) == 0:
        raise ValueError(
            "Memory cannot be empty."
        )

    if k < 1:
        raise ValueError(
            "k must be positive."
        )

    if temperature <= 0:
        raise ValueError(
            "Temperature must be positive."
        )

    if chunk_size < 1:
        raise ValueError(
            "Chunk size must be positive."
        )

    if weights is None:
        weights = np.ones(
            query_features.shape[1],
            dtype=np.float32,
        )

    weights = np.asarray(
        weights,
        dtype=np.float32,
    )

    if weights.shape != (
        query_features.shape[1],
    ):
        raise ValueError(
            "Weight dimensions do not match features."
        )

    if np.any(weights < 0):
        raise ValueError(
            "Feature weights cannot be negative."
        )

    memory_count = len(
        memory_features
    )

    k = min(
        int(k),
        memory_count,
    )

    output = np.empty(
        len(query_features),
        dtype=np.float32,
    )

    for start in range(
        0,
        len(query_features),
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            len(query_features),
        )

        query = query_features[
            start:end
        ]

        difference = (
            query[:, None, :]
            - memory_features[None, :, :]
        )

        distances = np.sum(
            difference * difference * weights,
            axis=2,
        )

        nearest_indices = np.argpartition(
            distances,
            kth=k - 1,
            axis=1,
        )[:, :k]

        nearest_distances = np.take_along_axis(
            distances,
            nearest_indices,
            axis=1,
        )

        similarities = np.exp(
            -nearest_distances / temperature
        )

        output[start:end] = np.mean(
            similarities,
            axis=1,
        )

    return output


def predict_memory_probability(
    features,
    memory,
    k=5,
    temperature=0.025,
    weights=None,
    chunk_size=4096,
):
    """
    Estimate foreground probability from foreground/background
    memory similarities.

        P(FG) = S(FG) / (S(FG) + S(BG))
    """

    if memory.foreground_count == 0:
        raise ValueError(
            "Foreground memory is empty."
        )

    if memory.background_count == 0:
        raise ValueError(
            "Background memory is empty."
        )

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if (
        features.ndim != 3
        or features.shape[2] != 5
    ):
        raise ValueError(
            "Feature map must have shape (H, W, 5)."
        )

    height, width = features.shape[:2]

    flattened = features.reshape(
        -1,
        5,
    )

    fg_similarity = nearest_memory_similarity(
        flattened,
        memory.foreground,
        k=k,
        temperature=temperature,
        weights=weights,
        chunk_size=chunk_size,
    )

    bg_similarity = nearest_memory_similarity(
        flattened,
        memory.background,
        k=k,
        temperature=temperature,
        weights=weights,
        chunk_size=chunk_size,
    )

    denominator = (
        fg_similarity
        + bg_similarity
        + 1e-8
    )

    probability = (
        fg_similarity / denominator
    )

    return probability.reshape(
        height,
        width,
    )


# ======================================================================
# OPTICAL FLOW
# ======================================================================

def calculate_flow(
    previous_gray,
    current_gray,
):
    if previous_gray.shape != current_gray.shape:
        raise ValueError(
            "Grayscale frames must have identical dimensions."
        )

    return cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        **FLOW_PARAMETERS,
    )


def propagate_mask(
    previous_mask,
    flow,
):
    """
    Approximate mask propagation using inverse sampling.

    The optical flow is assumed to describe previous-to-current
    displacement.

    For each current destination pixel:

        source = destination - flow

    This approximation can fail around occlusions and motion
    boundaries.
    """

    previous_mask = to_binary_mask(
        previous_mask
    )

    height, width = previous_mask.shape

    flow = np.asarray(
        flow,
        dtype=np.float32,
    )

    if flow.shape != (
        height,
        width,
        2,
    ):
        raise ValueError(
            "Flow shape must match mask dimensions."
        )

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    map_x = (
        grid_x - flow[..., 0]
    )

    map_y = (
        grid_y - flow[..., 1]
    )

    result = cv2.remap(
        previous_mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return (
        result > 0
    ).astype(np.uint8) * 255


# ======================================================================
# INITIAL PROMPT
# ======================================================================

def initialize_mask(
    frame,
    rectangle,
    iterations=5,
):
    """
    Use a bounding box prompt and GrabCut to obtain the initial mask.
    """

    height, width = frame.shape[:2]

    x, y, box_width, box_height = map(
        int,
        rectangle,
    )

    if (
        x < 0
        or y < 0
        or box_width < 2
        or box_height < 2
        or x + box_width > width
        or y + box_height > height
    ):
        raise ValueError(
            "Invalid initial bounding box."
        )

    gc_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    bg_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    fg_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    cv2.grabCut(
        frame,
        gc_mask,
        (
            x,
            y,
            box_width,
            box_height,
        ),
        bg_model,
        fg_model,
        iterations,
        cv2.GC_INIT_WITH_RECT,
    )

    result = np.where(
        (
            gc_mask == cv2.GC_FGD
        ) | (
            gc_mask == cv2.GC_PR_FGD
        ),
        255,
        0,
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_OPEN,
        kernel,
    )

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return result


# ======================================================================
# MASK VISUALIZATION
# ======================================================================

def make_overlay(
    frame,
    mask,
    probability=None,
):
    binary = to_binary_mask(
        mask
    )

    overlay = frame.copy()

    foreground = binary > 0

    green = np.zeros_like(
        frame
    )

    green[:, :, 1] = 255

    overlay[foreground] = cv2.addWeighted(
        frame[foreground],
        0.55,
        green[foreground],
        0.45,
        0,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 255),
        2,
    )

    centroid = mask_centroid(
        binary
    )

    if centroid is not None:
        cv2.circle(
            overlay,
            (
                int(round(centroid[0])),
                int(round(centroid[1])),
            ),
            5,
            (0, 0, 255),
            -1,
        )

    if probability is not None:
        mean_probability = (
            float(np.mean(probability[foreground]))
            if np.any(foreground)
            else 0.0
        )

        cv2.putText(
            overlay,
            f"Mean FG probability: {mean_probability:.3f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return overlay


# ======================================================================
# SEGMENTATION ENGINE
# ======================================================================

@dataclass
class TrackingConfig:
    memory_weight: float = 0.65
    threshold: float = 0.50
    temperature: float = 0.025
    nearest_k: int = 5
    update_interval: int = 5
    foreground_update_threshold: float = 0.90
    background_update_threshold: float = 0.05
    max_tokens: int = 512
    samples_per_update: int = 32


class MemoryVideoSegmenter:
    """
    Single-object memory-conditioned video segmentation.

    This implementation combines:

        - Initial GrabCut mask.
        - Foreground/background visual memory.
        - Nearest-neighbor memory matching.
        - Optical-flow propagation.
        - Confidence-based memory updates.
    """

    def __init__(
        self,
        config=None,
    ):
        self.config = (
            config
            if config is not None
            else TrackingConfig()
        )

        if not 0.0 <= self.config.memory_weight <= 1.0:
            raise ValueError(
                "memory_weight must be in [0, 1]."
            )

        if not 0.0 < self.config.threshold < 1.0:
            raise ValueError(
                "threshold must be in (0, 1)."
            )

        self.memory = VisualMemoryBank(
            max_tokens=self.config.max_tokens,
            samples_per_update=self.config.samples_per_update,
        )

        self.previous_gray = None
        self.previous_mask = None

        self.frame_index = 0
        self.initialized = False

    def initialize(
        self,
        frame,
        rectangle,
        grabcut_iterations=5,
    ):
        mask = initialize_mask(
            frame,
            rectangle,
            iterations=grabcut_iterations,
        )

        if mask_area(mask) == 0:
            raise RuntimeError(
                "Initial segmentation is empty."
            )

        features = extract_features(
            frame
        )

        self.memory.initialize(
            features,
            mask,
        )

        self.previous_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        self.previous_mask = mask

        self.frame_index = 0
        self.initialized = True

        return mask

    def process(self, frame):
        if not self.initialized:
            raise RuntimeError(
                "Segmenter has not been initialized."
            )

        if frame.shape[:2] != self.previous_mask.shape:
            raise ValueError(
                "Frame dimensions changed during tracking."
            )

        current_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        # ----------------------------------------------------------
        # Step 1: Optical-flow prior
        # ----------------------------------------------------------

        flow = calculate_flow(
            self.previous_gray,
            current_gray,
        )

        propagated_mask = propagate_mask(
            self.previous_mask,
            flow,
        )

        flow_probability = (
            propagated_mask.astype(np.float32)
            / 255.0
        )

        # ----------------------------------------------------------
        # Step 2: Current visual features
        # ----------------------------------------------------------

        features = extract_features(
            frame
        )

        # ----------------------------------------------------------
        # Step 3: Memory-conditioned probability
        # ----------------------------------------------------------

        memory_probability = predict_memory_probability(
            features,
            self.memory,
            k=self.config.nearest_k,
            temperature=self.config.temperature,
            weights=FEATURE_WEIGHTS,
        )

        # ----------------------------------------------------------
        # Step 4: Combine memory and motion
        # ----------------------------------------------------------

        memory_weight = (
            self.config.memory_weight
        )

        final_probability = (
            memory_weight * memory_probability
            + (1.0 - memory_weight) * flow_probability
        )

        final_mask = (
            final_probability >= self.config.threshold
        ).astype(np.uint8) * 255

        # ----------------------------------------------------------
        # Step 5: Morphological cleanup
        # ----------------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        # ----------------------------------------------------------
        # Step 6: Confidence-based memory update
        # ----------------------------------------------------------

        should_update = (
            self.frame_index > 0
            and self.frame_index
            % self.config.update_interval == 0
        )

        if should_update:
            foreground_selection = (
                final_probability
                >= self.config.foreground_update_threshold
            )

            background_selection = (
                final_probability
                <= self.config.background_update_threshold
            )

            # Only update when both classes have enough samples.
            if (
                np.count_nonzero(foreground_selection) >= 5
                and np.count_nonzero(background_selection) >= 5
            ):
                self.memory.update(
                    features,
                    foreground_selection,
                    background_selection,
                )

        # ----------------------------------------------------------
        # Step 7: Update temporal state
        # ----------------------------------------------------------

        self.previous_gray = current_gray
        self.previous_mask = final_mask

        self.frame_index += 1

        return {
            "mask": final_mask,
            "probability": final_probability,
            "memory_probability": memory_probability,
            "flow_probability": flow_probability,
            "memory_foreground_count": (
                self.memory.foreground_count
            ),
            "memory_background_count": (
                self.memory.background_count
            ),
        }


# ======================================================================
# INTERACTIVE ROI SELECTION
# ======================================================================

def select_roi(frame):
    window = "Select object prompt"

    cv2.namedWindow(
        window,
        cv2.WINDOW_NORMAL,
    )

    roi = cv2.selectROI(
        window,
        frame,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(
        window
    )

    x, y, width, height = map(
        int,
        roi,
    )

    if width <= 0 or height <= 0:
        return None

    return (
        x,
        y,
        width,
        height,
    )


# ======================================================================
# VIDEO PROCESSING
# ======================================================================

def process_video(args):
    capture = cv2.VideoCapture(
        args.video
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {args.video}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    source_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    source_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if source_width <= 0 or source_height <= 0:
        capture.release()

        raise RuntimeError(
            "Invalid video dimensions."
        )

    scale = 1.0

    if (
        args.max_width > 0
        and source_width > args.max_width
    ):
        scale = (
            args.max_width / source_width
        )

    width = int(
        round(source_width * scale)
    )

    height = int(
        round(source_height * scale)
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        args.output_dir,
        "sam2_inspired_memory_tracking.mp4",
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
            "Could not create output video."
        )

    config = TrackingConfig(
        memory_weight=args.memory_weight,
        threshold=args.threshold,
        temperature=args.temperature,
        nearest_k=args.nearest_k,
        update_interval=args.update_interval,
        max_tokens=args.max_tokens,
        samples_per_update=args.samples_per_update,
    )

    segmenter = MemoryVideoSegmenter(
        config
    )

    frame_count = 0
    areas = []
    centroids = []
    memory_sizes = []

    start_time = time.perf_counter()

    try:
        ret, frame = capture.read()

        if not ret:
            raise RuntimeError(
                "Could not read first frame."
            )

        if scale != 1.0:
            frame = cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_AREA,
            )

        rectangle = select_roi(
            frame
        )

        if rectangle is None:
            raise RuntimeError(
                "No valid object prompt was selected."
            )

        initial_mask = segmenter.initialize(
            frame,
            rectangle,
            grabcut_iterations=args.initial_iterations,
        )

        overlay = make_overlay(
            frame,
            initial_mask,
        )

        cv2.putText(
            overlay,
            "INITIAL PROMPT MASK",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        writer.write(
            overlay
        )

        frame_count += 1

        areas.append(
            mask_area(initial_mask)
        )

        centroids.append(
            mask_centroid(initial_mask)
        )

        memory_sizes.append(
            segmenter.memory.foreground_count
            + segmenter.memory.background_count
        )

        cv2.imshow(
            "Memory Video Segmentation",
            overlay,
        )

        print("\nInitial object memory created.")
        print(
            f"Foreground memory tokens: "
            f"{segmenter.memory.foreground_count}"
        )
        print(
            f"Background memory tokens: "
            f"{segmenter.memory.background_count}"
        )

        print(
            "\nControls: q=quit, p=pause, r=reinitialize, "
            "+/-=memory contribution"
        )

        paused = False

        while True:
            if not paused:
                ret, frame = capture.read()

                if not ret:
                    break

                if scale != 1.0:
                    frame = cv2.resize(
                        frame,
                        (width, height),
                        interpolation=cv2.INTER_AREA,
                    )

                result = segmenter.process(
                    frame
                )

                mask = result["mask"]

                overlay = make_overlay(
                    frame,
                    mask,
                    result["probability"],
                )

                cv2.putText(
                    overlay,
                    f"Frame: {frame_count}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    overlay,
                    (
                        f"Memory FG/BG: "
                        f"{result['memory_foreground_count']}/"
                        f"{result['memory_background_count']}"
                    ),
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                writer.write(
                    overlay
                )

                frame_count += 1

                areas.append(
                    mask_area(mask)
                )

                centroids.append(
                    mask_centroid(mask)
                )

                memory_sizes.append(
                    result["memory_foreground_count"]
                    + result["memory_background_count"]
                )

                cv2.imshow(
                    "Memory Video Segmentation",
                    overlay,
                )

            key = cv2.waitKey(
                30 if paused else 1
            ) & 0xFF

            if key == ord("q") or key == 27:
                break

            elif key == ord("p"):
                paused = not paused

            elif key in (
                ord("+"),
                ord("="),
            ):
                segmenter.config.memory_weight = min(
                    1.0,
                    segmenter.config.memory_weight + 0.05,
                )

                print(
                    "Memory weight:",
                    segmenter.config.memory_weight,
                )

            elif key in (
                ord("-"),
                ord("_"),
            ):
                segmenter.config.memory_weight = max(
                    0.0,
                    segmenter.config.memory_weight - 0.05,
                )

                print(
                    "Memory weight:",
                    segmenter.config.memory_weight,
                )

            elif key == ord("r"):
                rectangle = select_roi(
                    frame
                )

                if rectangle is not None:
                    segmenter = MemoryVideoSegmenter(
                        config
                    )

                    initial_mask = segmenter.initialize(
                        frame,
                        rectangle,
                        grabcut_iterations=args.initial_iterations,
                    )

                    print(
                        "Object memory reinitialized."
                    )

    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed = (
        time.perf_counter() - start_time
    )

    processing_fps = (
        frame_count / elapsed
        if elapsed > 0
        else 0.0
    )

    valid_centroids = [
        list(point)
        for point in centroids
        if point is not None
    ]

    summary = {
        "lesson": (
            "Lesson 57 - SAM 2-Inspired Memory Video Tracking"
        ),
        "architecture": (
            "Optical flow + foreground/background visual memory"
        ),
        "official_sam2": False,
        "pretrained_weights": False,
        "input_video": args.video,
        "output_video": output_path,
        "frames_processed": int(frame_count),
        "input_fps": float(fps),
        "processing_fps": float(processing_fps),
        "processing_seconds": float(elapsed),
        "frame_width": int(width),
        "frame_height": int(height),
        "mean_mask_area": (
            float(np.mean(areas))
            if areas
            else 0.0
        ),
        "minimum_mask_area": (
            int(np.min(areas))
            if areas
            else 0
        ),
        "maximum_mask_area": (
            int(np.max(areas))
            if areas
            else 0
        ),
        "mean_memory_tokens": (
            float(np.mean(memory_sizes))
            if memory_sizes
            else 0.0
        ),
        "final_foreground_memory_tokens": (
            segmenter.memory.foreground_count
        ),
        "final_background_memory_tokens": (
            segmenter.memory.background_count
        ),
        "centroid_history": valid_centroids,
    }

    summary_path = os.path.join(
        args.output_dir,
        "memory_tracking_summary.json",
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

    print("\n" + "=" * 65)
    print("MEMORY VIDEO SEGMENTATION SUMMARY")
    print("=" * 65)

    print(
        f"Frames processed: {frame_count}"
    )

    print(
        f"Processing FPS: {processing_fps:.2f}"
    )

    print(
        f"Mean mask area: {summary['mean_mask_area']:.1f}"
    )

    print(
        "Final foreground memory:",
        segmenter.memory.foreground_count,
    )

    print(
        "Final background memory:",
        segmenter.memory.background_count,
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Summary: {summary_path}"
    )

    print("=" * 65)


# ======================================================================
# NUMERICAL TESTS
# ======================================================================

def run_tests():
    print("\nRunning Lesson 57 tests...")
    print("=" * 65)

    # --------------------------------------------------------------
    # Test 1: feature extraction shape
    # --------------------------------------------------------------

    frame = np.zeros(
        (32, 40, 3),
        dtype=np.uint8,
    )

    features = extract_features(
        frame
    )

    assert features.shape == (
        32,
        40,
        5,
    )

    print("Test 01 passed: feature extraction dimensions.")

    # --------------------------------------------------------------
    # Test 2: feature extraction dtype
    # --------------------------------------------------------------

    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))

    print("Test 02 passed: feature dtype and finite values.")

    # --------------------------------------------------------------
    # Test 3: invalid feature image
    # --------------------------------------------------------------

    try:
        extract_features(
            np.zeros((20, 20), dtype=np.uint8)
        )

        raise AssertionError(
            "Invalid image was accepted."
        )

    except ValueError:
        pass

    print("Test 03 passed: invalid feature input.")

    # --------------------------------------------------------------
    # Test 4: memory initialization
    # --------------------------------------------------------------

    feature_map = np.zeros(
        (20, 20, 5),
        dtype=np.float32,
    )

    mask = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    mask[5:15, 5:15] = 255

    memory = VisualMemoryBank(
        max_tokens=50,
        seed=1,
    )

    memory.initialize(
        feature_map,
        mask,
    )

    assert memory.foreground_count == 50
    assert memory.background_count == 50

    print("Test 04 passed: memory initialization.")

    # --------------------------------------------------------------
    # Test 5: memory capacity
    # --------------------------------------------------------------

    assert memory.foreground_count <= 50
    assert memory.background_count <= 50

    print("Test 05 passed: bounded memory capacity.")

    # --------------------------------------------------------------
    # Test 6: memory feature dimensions
    # --------------------------------------------------------------

    assert memory.foreground.shape[1] == 5
    assert memory.background.shape[1] == 5

    print("Test 06 passed: memory feature dimensions.")

    # --------------------------------------------------------------
    # Test 7: update memory
    # --------------------------------------------------------------

    memory.update(
        feature_map,
        mask > 0,
        mask == 0,
    )

    assert memory.foreground_count <= 50
    assert memory.background_count <= 50

    print("Test 07 passed: memory update.")

    # --------------------------------------------------------------
    # Test 8: empty foreground rejection
    # --------------------------------------------------------------

    try:
        VisualMemoryBank().initialize(
            feature_map,
            np.zeros((20, 20), dtype=np.uint8),
        )

        raise AssertionError(
            "Empty foreground was accepted."
        )

    except ValueError:
        pass

    print("Test 08 passed: empty foreground rejection.")

    # --------------------------------------------------------------
    # Test 9: empty background rejection
    # --------------------------------------------------------------

    try:
        VisualMemoryBank().initialize(
            feature_map,
            np.ones((20, 20), dtype=np.uint8) * 255,
        )

        raise AssertionError(
            "Empty background was accepted."
        )

    except ValueError:
        pass

    print("Test 09 passed: empty background rejection.")

    # --------------------------------------------------------------
    # Test 10: nearest-memory similarity
    # --------------------------------------------------------------

    queries = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    memory_features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
        ],
        dtype=np.float32,
    )

    similarity = nearest_memory_similarity(
        queries,
        memory_features,
        k=1,
        temperature=0.1,
        weights=np.ones(2, dtype=np.float32),
    )

    assert similarity.shape == (2,)
    assert similarity[0] > similarity[1]

    print("Test 10 passed: nearest-memory similarity.")

    # --------------------------------------------------------------
    # Test 11: similarity bounded
    # --------------------------------------------------------------

    assert np.all(similarity >= 0.0)
    assert np.all(similarity <= 1.0)

    print("Test 11 passed: similarity bounds.")

    # --------------------------------------------------------------
    # Test 12: memory probability dimensions
    # --------------------------------------------------------------

    probability = predict_memory_probability(
        feature_map,
        memory,
        k=3,
    )

    assert probability.shape == (
        20,
        20,
    )

    print("Test 12 passed: memory probability dimensions.")

    # --------------------------------------------------------------
    # Test 13: probability bounds
    # --------------------------------------------------------------

    assert np.all(probability >= 0.0)
    assert np.all(probability <= 1.0)

    print("Test 13 passed: foreground probability bounds.")

    # --------------------------------------------------------------
    # Test 14: zero-flow propagation
    # --------------------------------------------------------------

    mask = np.zeros(
        (30, 30),
        dtype=np.uint8,
    )

    mask[10:20, 10:20] = 255

    flow = np.zeros(
        (30, 30, 2),
        dtype=np.float32,
    )

    propagated = propagate_mask(
        mask,
        flow,
    )

    assert np.array_equal(
        propagated,
        mask,
    )

    print("Test 14 passed: zero-flow propagation.")

    # --------------------------------------------------------------
    # Test 15: horizontal mask propagation
    # --------------------------------------------------------------

    flow[..., 0] = 4.0

    propagated = propagate_mask(
        mask,
        flow,
    )

    points = cv2.findNonZero(
        propagated
    )

    bbox = cv2.boundingRect(
        points
    )

    assert bbox == (14, 10, 10, 10)

    print("Test 15 passed: horizontal mask propagation.")

    # --------------------------------------------------------------
    # Test 16: mask IoU
    # --------------------------------------------------------------

    a = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    b = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    a[5:15, 5:15] = 255
    b[5:15, 5:15] = 255

    assert mask_iou(a, b) == 1.0

    print("Test 16 passed: identical-mask IoU.")

    # --------------------------------------------------------------
    # Test 17: partial-overlap IoU
    # --------------------------------------------------------------

    a = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    b = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    a[0:10, 0:10] = 255
    b[0:10, 5:15] = 255

    expected_iou = 50.0 / 150.0

    assert np.isclose(
        mask_iou(a, b),
        expected_iou,
    )

    print("Test 17 passed: partial-overlap IoU.")

    # --------------------------------------------------------------
    # Test 18: Dice score
    # --------------------------------------------------------------

    assert mask_dice(a, b) == 0.5

    print("Test 18 passed: Dice score.")

    # --------------------------------------------------------------
    # Test 19: empty-mask IoU
    # --------------------------------------------------------------

    empty = np.zeros(
        (10, 10),
        dtype=np.uint8,
    )

    assert mask_iou(empty, empty) == 1.0

    print("Test 19 passed: empty-mask IoU.")

    # --------------------------------------------------------------
    # Test 20: centroid
    # --------------------------------------------------------------

    mask = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    mask[4:8, 6:10] = 255

    centroid = mask_centroid(mask)

    assert np.allclose(
        centroid,
        [7.5, 5.5],
    )

    print("Test 20 passed: mask centroid.")

    # --------------------------------------------------------------
    # Test 21: invalid memory feature shape
    # --------------------------------------------------------------

    try:
        VisualMemoryBank().initialize(
            np.zeros((20, 20, 3), dtype=np.float32),
            np.zeros((20, 20), dtype=np.uint8),
        )

        raise AssertionError(
            "Invalid feature map was accepted."
        )

    except ValueError:
        pass

    print("Test 21 passed: invalid memory feature shape.")

    # --------------------------------------------------------------
    # Test 22: invalid nearest-memory input
    # --------------------------------------------------------------

    try:
        nearest_memory_similarity(
            np.zeros((10, 5), dtype=np.float32),
            np.zeros((0, 5), dtype=np.float32),
        )

        raise AssertionError(
            "Empty memory was accepted."
        )

    except ValueError:
        pass

    print("Test 22 passed: empty nearest-memory rejection.")

    # --------------------------------------------------------------
    # Test 23: memory update dimensions
    # --------------------------------------------------------------

    try:
        memory.update(
            feature_map,
            np.zeros((10, 10), dtype=bool),
            np.zeros((20, 20), dtype=bool),
        )

        raise AssertionError(
            "Invalid update dimensions were accepted."
        )

    except ValueError:
        pass

    print("Test 23 passed: memory update dimension validation.")

    # --------------------------------------------------------------
    # Test 24: segmenter configuration
    # --------------------------------------------------------------

    try:
        MemoryVideoSegmenter(
            TrackingConfig(
                memory_weight=1.5
            )
        )

        raise AssertionError(
            "Invalid memory weight was accepted."
        )

    except ValueError:
        pass

    print("Test 24 passed: segmenter configuration validation.")

    print("=" * 65)
    print("All 24 tests passed.")


# ======================================================================
# COMMAND LINE
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Lesson 57 - SAM 2-Inspired Memory Video Tracking"
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a real video.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--initial-iterations",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--memory-weight",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.025,
    )

    parser.add_argument(
        "--nearest-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--update-interval",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--samples-per-update",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    return parser.parse_args()


# ======================================================================
# MAIN
# ======================================================================

def main():
    args = parse_args()

    if args.check_only:
        run_tests()
        return

    if args.video is None:
        raise ValueError(
            "Provide a real video using --video."
        )

    if args.max_width < 0:
        raise ValueError(
            "--max-width cannot be negative."
        )

    if args.nearest_k < 1:
        raise ValueError(
            "--nearest-k must be positive."
        )

    if args.update_interval < 1:
        raise ValueError(
            "--update-interval must be positive."
        )

    if args.max_tokens < 1:
        raise ValueError(
            "--max-tokens must be positive."
        )

    if args.samples_per_update < 1:
        raise ValueError(
            "--samples-per-update must be positive."
        )

    if args.temperature <= 0:
        raise ValueError(
            "--temperature must be positive."
        )

    if not 0.0 <= args.memory_weight <= 1.0:
        raise ValueError(
            "--memory-weight must be between 0 and 1."
        )

    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    run_tests()

    process_video(args)


if __name__ == "__main__":
    main()