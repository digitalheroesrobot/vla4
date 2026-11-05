"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 55 — TRACKING EVALUATION AND METRICS
======================================================================

Filename:
    lesson55_tracking_evaluation_metrics.py

COURSE POSITION
---------------
Lesson 41-54:
    Detection, tracking, identity association, trajectory analysis,
    collision prediction, and anomaly detection.

Lesson 55:
    Quantitatively evaluate a multi-object tracking system.

LEARNING OBJECTIVES
-------------------
1. Calculate Intersection over Union (IoU).
2. Match predicted detections to ground-truth objects.
3. Calculate false positives, false negatives, and true positives.
4. Implement MOTA and MOTP.
5. Count identity switches.
6. Implement IDF1 using global identity assignment.
7. Implement a HOTA-inspired association evaluation.
8. Evaluate multiple IoU thresholds.
9. Read a real annotated tracking sequence from JSON.
10. Visualize tracking evaluation results.

======================================================================
1. INTERSECTION OVER UNION (IoU)
======================================================================

A bounding box is represented as:

    [x1, y1, x2, y2]

Intersection area:

    A_intersection

Union area:

    A_union = A_box1 + A_box2 - A_intersection

IoU:

    IoU = A_intersection / A_union

IoU ranges from 0 to 1.

    0:
        No overlap.

    1:
        Perfect overlap.

======================================================================
2. FRAME-LEVEL DETECTION MATCHING
======================================================================

For every frame, construct an IoU matrix:

                 Prediction 1   Prediction 2
    GT object 1      0.90           0.10
    GT object 2      0.15           0.85

Use Hungarian assignment to maximize total IoU.

A match is valid only when:

    IoU >= IoU threshold

This prevents one prediction from matching multiple ground-truth
objects in the same frame.

======================================================================
3. DETECTION COUNTS
======================================================================

TP:
    A ground-truth object matched to a prediction.

FN:
    A ground-truth object without a valid prediction.

FP:
    A prediction without a valid ground-truth match.

======================================================================
4. MOTA — MULTIPLE OBJECT TRACKING ACCURACY
======================================================================

    MOTA = 1 - (FN + FP + IDSW) / GT

Where:

    FN:
        False negatives.

    FP:
        False positives.

    IDSW:
        Identity switches.

    GT:
        Total number of ground-truth detections.

MOTA can be negative when tracking errors are sufficiently large.

MOTA combines detection errors and identity switches.

======================================================================
5. MOTP — MULTIPLE OBJECT TRACKING PRECISION
======================================================================

This lesson reports MOTP as mean matched IoU:

    MOTP = sum(matched IoU) / TP

Higher is better.

IMPORTANT:
    Some tracking benchmarks define MOTP using localization distance
    rather than overlap. Always state the convention when comparing
    results from different evaluation tools.

======================================================================
6. ID SWITCHES
======================================================================

Suppose a physical object has ground-truth ID 7.

Frame 1:
    GT ID 7 -> Predicted ID 12

Frame 2:
    GT ID 7 -> Predicted ID 12

Frame 3:
    GT ID 7 -> Predicted ID 19

The tracker changed its identity assignment.

This is an identity switch.

======================================================================
7. IDF1 — IDENTITY F1 SCORE
======================================================================

IDF1 measures identity preservation over an entire sequence.

First count how often each ground-truth ID is matched with each
predicted ID.

Then perform global identity assignment using Hungarian matching.

IDTP:
    Correctly identified object detections.

IDFP:
    Incorrect predicted identity detections.

IDFN:
    Ground-truth detections not assigned to the correct predicted ID.

    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

A tracker can have excellent detection performance but poor IDF1
if it frequently changes identities.

======================================================================
8. HOTA-INSPIRED EVALUATION
======================================================================

HOTA balances detection and association.

Detection accuracy:

    DetA = TP / (TP + FN + FP)

Association accuracy:

    AssA = weighted mean association Jaccard similarity.

For a matched ground-truth / predicted identity pair:

    pair_TP = number of matched detections for this identity pair

    gt_count = number of detections for the ground-truth identity

    pred_count = number of detections for the predicted identity

    association_similarity =
        pair_TP / (gt_count + pred_count - pair_TP)

The association accuracy is weighted by pair_TP.

Then:

    HOTA = sqrt(DetA * AssA)

We calculate this at several IoU thresholds and average the results.

This is an educational HOTA-style implementation. For official
benchmark reporting, use the official TrackEval implementation and
its exact matching and aggregation conventions.

======================================================================
9. REAL-WORLD ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    Camera
       |
       v
    YOLO / instance segmentation
       |
       v
    SORT / ByteTrack / Deep SORT
       |
       v
    Predicted object IDs and boxes
       |
       v
    Compare against annotated ground truth
       |
       v
    MOTA / MOTP / IDF1 / HOTA
       |
       v
    Evaluate tracker quality

Example:

    If the robot sees a bottle moving on a conveyor, it should
    maintain the same identity while the bottle moves.

    If the tracker changes the bottle's ID every few frames,
    the robot may lose its grasp target.

======================================================================
10. INPUT JSON FORMAT
======================================================================

Example:

{
    "frames": [
        {
            "frame_index": 0,
            "gt": [
                {
                    "id": 1,
                    "bbox": [10, 20, 50, 80]
                }
            ],
            "pred": [
                {
                    "id": 101,
                    "bbox": [11, 20, 51, 80]
                }
            ]
        }
    ]
}

Each frame contains:

    gt:
        Ground-truth objects.

    pred:
        Tracker predictions.

Each object contains:

    id:
        Persistent identity.

    bbox:
        [x1, y1, x2, y2]

======================================================================
DEPENDENCIES
======================================================================

    pip install numpy scipy matplotlib opencv-python

======================================================================
RUN
======================================================================

Numerical tests:

    python lesson55_tracking_evaluation_metrics.py --check-only

Demonstration:

    python lesson55_tracking_evaluation_metrics.py

Evaluate a real annotated sequence:

    python lesson55_tracking_evaluation_metrics.py \
        --annotations tracking_annotations.json

======================================================================
OUTPUT
======================================================================

    runs/lesson55/
        metrics_summary.png
        tracking_sequence.png
        evaluation_results.json

======================================================================
NEXT LESSON
======================================================================

    Lesson 56 — Tracking System Integration and Real-Time Pipeline
======================================================================
"""

from __future__ import annotations

import argparse
import json
import os

from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import linear_sum_assignment


# ======================================================================
# CONFIGURATION
# ======================================================================

LESSON_NAME = "Lesson 55 - Tracking Evaluation and Metrics"

DEFAULT_OUTPUT_DIR = os.path.join("runs", "lesson55")

DEFAULT_IOU_THRESHOLD = 0.5

HOTA_THRESHOLDS = np.arange(
    0.05,
    0.96,
    0.05,
    dtype=np.float64,
)


# ======================================================================
# BOUNDING BOX OPERATIONS
# ======================================================================

def validate_box(box):
    box = np.asarray(box, dtype=np.float64)

    if box.shape != (4,):
        raise ValueError(
            "Bounding box must contain exactly four values."
        )

    if not np.all(np.isfinite(box)):
        raise ValueError(
            "Bounding box contains NaN or Inf."
        )

    x1, y1, x2, y2 = box

    if x2 < x1 or y2 < y1:
        raise ValueError(
            "Invalid bounding box coordinates."
        )

    return box


def box_area(box):
    x1, y1, x2, y2 = validate_box(box)

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return float(width * height)


def intersection_area(box_a, box_b):
    box_a = validate_box(box_a)
    box_b = validate_box(box_b)

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])

    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return float(width * height)


def box_iou(box_a, box_b):
    """
    Calculate IoU between two bounding boxes.
    """

    area_a = box_area(box_a)
    area_b = box_area(box_b)

    intersection = intersection_area(
        box_a,
        box_b,
    )

    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def calculate_iou_matrix(gt_boxes, pred_boxes):
    """
    Return an IoU matrix of shape:

        (number_of_gt_boxes, number_of_pred_boxes)
    """

    matrix = np.zeros(
        (len(gt_boxes), len(pred_boxes)),
        dtype=np.float64,
    )

    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            matrix[i, j] = box_iou(
                gt_box,
                pred_box,
            )

    return matrix


# ======================================================================
# ANNOTATION VALIDATION
# ======================================================================

def validate_object(obj):
    if "id" not in obj:
        raise ValueError(
            "Every object must contain an 'id'."
        )

    if "bbox" not in obj:
        raise ValueError(
            "Every object must contain a 'bbox'."
        )

    object_id = obj["id"]

    if isinstance(object_id, bool):
        raise ValueError("Boolean IDs are not supported.")

    if not isinstance(object_id, (int, np.integer)):
        raise ValueError(
            "Object IDs must be integers."
        )

    box = validate_box(obj["bbox"])

    if box_area(box) <= 0:
        raise ValueError(
            "Bounding boxes must have positive area."
        )

    return {
        "id": int(object_id),
        "bbox": box.tolist(),
    }


def validate_sequence(sequence):
    if not isinstance(sequence, dict):
        raise ValueError(
            "Sequence must be a dictionary."
        )

    if "frames" not in sequence:
        raise ValueError(
            "Sequence must contain a 'frames' list."
        )

    frames = sequence["frames"]

    if not isinstance(frames, list):
        raise ValueError(
            "'frames' must be a list."
        )

    validated_frames = []

    for frame_number, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(
                "Each frame must be a dictionary."
            )

        frame_index = int(
            frame.get("frame_index", frame_number)
        )

        gt = [
            validate_object(obj)
            for obj in frame.get("gt", [])
        ]

        pred = [
            validate_object(obj)
            for obj in frame.get("pred", [])
        ]

        gt_ids = [obj["id"] for obj in gt]
        pred_ids = [obj["id"] for obj in pred]

        if len(gt_ids) != len(set(gt_ids)):
            raise ValueError(
                f"Duplicate ground-truth ID in frame {frame_index}."
            )

        if len(pred_ids) != len(set(pred_ids)):
            raise ValueError(
                f"Duplicate predicted ID in frame {frame_index}."
            )

        validated_frames.append(
            {
                "frame_index": frame_index,
                "gt": gt,
                "pred": pred,
            }
        )

    return {
        "frames": validated_frames
    }


# ======================================================================
# FRAME-LEVEL MATCHING
# ======================================================================

def match_frame(
    gt_objects,
    pred_objects,
    iou_threshold=0.5,
):
    """
    Match ground-truth and predicted objects using Hungarian
    assignment with IoU as the matching objective.

    Returns:
        matches:
            List of (gt_index, pred_index, iou).

        unmatched_gt:
            Ground-truth indices without a match.

        unmatched_pred:
            Prediction indices without a match.
    """

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(
            "IoU threshold must be between 0 and 1."
        )

    gt_boxes = [
        obj["bbox"]
        for obj in gt_objects
    ]

    pred_boxes = [
        obj["bbox"]
        for obj in pred_objects
    ]

    iou_matrix = calculate_iou_matrix(
        gt_boxes,
        pred_boxes,
    )

    if len(gt_objects) == 0 or len(pred_objects) == 0:
        return (
            [],
            list(range(len(gt_objects))),
            list(range(len(pred_objects))),
        )

    # Hungarian assignment minimizes cost.
    # Invalid matches receive a very large cost.
    cost = 1.0 - iou_matrix

    invalid = iou_matrix < iou_threshold

    cost[invalid] = 1e6

    rows, cols = linear_sum_assignment(cost)

    matches = []

    matched_gt = set()
    matched_pred = set()

    for row, col in zip(rows, cols):
        iou = float(iou_matrix[row, col])

        if iou < iou_threshold:
            continue

        matches.append(
            (
                int(row),
                int(col),
                iou,
            )
        )

        matched_gt.add(int(row))
        matched_pred.add(int(col))

    unmatched_gt = [
        i
        for i in range(len(gt_objects))
        if i not in matched_gt
    ]

    unmatched_pred = [
        i
        for i in range(len(pred_objects))
        if i not in matched_pred
    ]

    return (
        matches,
        unmatched_gt,
        unmatched_pred,
    )


# ======================================================================
# TRACKING EVALUATOR
# ======================================================================

class TrackingEvaluator:
    """
    Evaluate one complete multi-object tracking sequence.

    Metrics:

        TP, FP, FN
        ID switches
        MOTA
        MOTP (mean matched IoU)
        IDTP, IDFP, IDFN
        ID precision, ID recall, IDF1
        DetA, AssA, HOTA-inspired score
    """

    def __init__(
        self,
        iou_threshold=DEFAULT_IOU_THRESHOLD,
    ):
        self.iou_threshold = float(iou_threshold)

    def evaluate(self, sequence):
        sequence = validate_sequence(sequence)

        total_gt = 0
        total_pred = 0

        total_tp = 0
        total_fp = 0
        total_fn = 0

        matched_ious = []

        identity_switches = 0

        # For IDF1 and association metrics.
        identity_pair_counts = defaultdict(int)

        gt_identity_counts = defaultdict(int)
        pred_identity_counts = defaultdict(int)

        # Previous matched predicted ID for each GT identity.
        previous_assignment = {}

        frame_results = []

        for frame in sequence["frames"]:
            frame_index = frame["frame_index"]

            gt_objects = frame["gt"]
            pred_objects = frame["pred"]

            total_gt += len(gt_objects)
            total_pred += len(pred_objects)

            for obj in gt_objects:
                gt_identity_counts[obj["id"]] += 1

            for obj in pred_objects:
                pred_identity_counts[obj["id"]] += 1

            matches, unmatched_gt, unmatched_pred = match_frame(
                gt_objects,
                pred_objects,
                self.iou_threshold,
            )

            tp = len(matches)
            fn = len(unmatched_gt)
            fp = len(unmatched_pred)

            total_tp += tp
            total_fn += fn
            total_fp += fp

            frame_match_records = []

            for gt_index, pred_index, iou in matches:
                gt_id = gt_objects[gt_index]["id"]
                pred_id = pred_objects[pred_index]["id"]

                matched_ious.append(iou)

                identity_pair_counts[
                    (gt_id, pred_id)
                ] += 1

                if gt_id in previous_assignment:
                    old_pred_id = previous_assignment[gt_id]

                    if old_pred_id != pred_id:
                        identity_switches += 1

                previous_assignment[gt_id] = pred_id

                frame_match_records.append(
                    {
                        "gt_id": int(gt_id),
                        "pred_id": int(pred_id),
                        "iou": float(iou),
                    }
                )

            frame_results.append(
                {
                    "frame_index": int(frame_index),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "matches": frame_match_records,
                }
            )

        # ----------------------------------------------------------
        # MOTA
        # ----------------------------------------------------------

        if total_gt > 0:
            mota = 1.0 - (
                total_fn
                + total_fp
                + identity_switches
            ) / total_gt
        else:
            # MOTA is undefined if there are no GT detections.
            mota = None

        # ----------------------------------------------------------
        # MOTP: mean matched IoU
        # ----------------------------------------------------------

        if total_tp > 0:
            motp = float(
                np.mean(matched_ious)
            )
        else:
            motp = None

        # ----------------------------------------------------------
        # IDF1: global identity assignment
        # ----------------------------------------------------------

        gt_ids = sorted(gt_identity_counts.keys())
        pred_ids = sorted(pred_identity_counts.keys())

        idtp = 0

        if gt_ids and pred_ids:
            pair_matrix = np.zeros(
                (len(gt_ids), len(pred_ids)),
                dtype=np.float64,
            )

            gt_index_map = {
                identity: index
                for index, identity in enumerate(gt_ids)
            }

            pred_index_map = {
                identity: index
                for index, identity in enumerate(pred_ids)
            }

            for (gt_id, pred_id), count in (
                identity_pair_counts.items()
            ):
                pair_matrix[
                    gt_index_map[gt_id],
                    pred_index_map[pred_id],
                ] = count

            # Maximize total matched identity detections.
            rows, cols = linear_sum_assignment(
                -pair_matrix
            )

            idtp = int(
                sum(
                    pair_matrix[row, col]
                    for row, col in zip(rows, cols)
                )
            )

        idfn = int(total_gt - idtp)
        idfp = int(total_pred - idtp)

        id_precision = (
            idtp / (idtp + idfp)
            if idtp + idfp > 0
            else 0.0
        )

        id_recall = (
            idtp / (idtp + idfn)
            if idtp + idfn > 0
            else 0.0
        )

        idf1_denominator = (
            2 * idtp + idfp + idfn
        )

        idf1 = (
            2.0 * idtp / idf1_denominator
            if idf1_denominator > 0
            else 0.0
        )

        # ----------------------------------------------------------
        # Detection accuracy
        # ----------------------------------------------------------

        deta_denominator = (
            total_tp + total_fn + total_fp
        )

        deta = (
            total_tp / deta_denominator
            if deta_denominator > 0
            else 0.0
        )

        # ----------------------------------------------------------
        # Association accuracy
        # ----------------------------------------------------------

        assa_numerator = 0.0

        for (gt_id, pred_id), pair_tp in (
            identity_pair_counts.items()
        ):
            gt_count = gt_identity_counts[gt_id]
            pred_count = pred_identity_counts[pred_id]

            union = (
                gt_count
                + pred_count
                - pair_tp
            )

            association_jaccard = (
                pair_tp / union
                if union > 0
                else 0.0
            )

            assa_numerator += (
                pair_tp * association_jaccard
            )

        assa = (
            assa_numerator / total_tp
            if total_tp > 0
            else 0.0
        )

        hota = float(
            np.sqrt(deta * assa)
        )

        return {
            "iou_threshold": self.iou_threshold,
            "frames": len(sequence["frames"]),
            "gt_detections": int(total_gt),
            "pred_detections": int(total_pred),
            "TP": int(total_tp),
            "FP": int(total_fp),
            "FN": int(total_fn),
            "IDSW": int(identity_switches),
            "MOTA": (
                float(mota)
                if mota is not None
                else None
            ),
            "MOTP_mean_IoU": (
                float(motp)
                if motp is not None
                else None
            ),
            "IDTP": int(idtp),
            "IDFP": int(idfp),
            "IDFN": int(idfn),
            "ID_precision": float(id_precision),
            "ID_recall": float(id_recall),
            "IDF1": float(idf1),
            "DetA": float(deta),
            "AssA": float(assa),
            "HOTA": float(hota),
            "frame_results": frame_results,
        }


# ======================================================================
# MULTI-THRESHOLD EVALUATION
# ======================================================================

def evaluate_hota_thresholds(
    sequence,
    thresholds=None,
):
    """
    Evaluate the HOTA-inspired score over multiple IoU thresholds.

    The final score is the arithmetic mean of the per-threshold
    square-root DetA * AssA values.
    """

    if thresholds is None:
        thresholds = HOTA_THRESHOLDS

    threshold_results = []

    for threshold in thresholds:
        evaluator = TrackingEvaluator(
            iou_threshold=float(threshold)
        )

        result = evaluator.evaluate(sequence)

        threshold_results.append(
            {
                "threshold": float(threshold),
                "DetA": result["DetA"],
                "AssA": result["AssA"],
                "HOTA": result["HOTA"],
            }
        )

    mean_hota = float(
        np.mean(
            [
                result["HOTA"]
                for result in threshold_results
            ]
        )
    )

    mean_deta = float(
        np.mean(
            [
                result["DetA"]
                for result in threshold_results
            ]
        )
    )

    mean_assa = float(
        np.mean(
            [
                result["AssA"]
                for result in threshold_results
            ]
        )
    )

    return {
        "mean_HOTA": mean_hota,
        "mean_DetA": mean_deta,
        "mean_AssA": mean_assa,
        "threshold_results": threshold_results,
    }


# ======================================================================
# DEMONSTRATION SEQUENCE
# ======================================================================

def make_demo_sequence():
    """
    Create a controlled multi-object tracking sequence.

    Ground truth:
        Object 1 moves from left to right.
        Object 2 moves from right to left.

    Predictions:
        Mostly accurate boxes.
        One missed detection.
        One false positive.
        A predicted identity swap in the second half.

    This is mathematical evaluation data, not synthetic imagery.
    """

    frames = []

    for frame_index in range(20):
        x1 = 20 + frame_index * 5
        x2 = 180 - frame_index * 5

        gt = [
            {
                "id": 1,
                "bbox": [
                    x1,
                    40,
                    x1 + 30,
                    70,
                ],
            },
            {
                "id": 2,
                "bbox": [
                    x2,
                    100,
                    x2 + 30,
                    130,
                ],
            },
        ]

        pred = []

        # One missing detection.
        if frame_index != 5:
            pred_id_1 = (
                202 if frame_index >= 10 else 101
            )

            pred.append(
                {
                    "id": pred_id_1,
                    "bbox": [
                        x1 + 1,
                        40,
                        x1 + 31,
                        70,
                    ],
                }
            )

        pred_id_2 = (
            101 if frame_index >= 10 else 202
        )

        pred.append(
            {
                "id": pred_id_2,
                "bbox": [
                    x2 + 1,
                    100,
                    x2 + 31,
                    130,
                ],
            }
        )

        # One false positive.
        if frame_index == 8:
            pred.append(
                {
                    "id": 999,
                    "bbox": [
                        300,
                        200,
                        330,
                        230,
                    ],
                }
            )

        frames.append(
            {
                "frame_index": frame_index,
                "gt": gt,
                "pred": pred,
            }
        )

    return {
        "frames": frames
    }


# ======================================================================
# VISUALIZATION
# ======================================================================

def create_metrics_plot(
    result,
    hota_result,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    names = [
        "MOTA",
        "MOTP\n(mean IoU)",
        "IDF1",
        "HOTA",
        "DetA",
        "AssA",
    ]

    values = [
        result["MOTA"],
        result["MOTP_mean_IoU"],
        result["IDF1"],
        hota_result["mean_HOTA"],
        hota_result["mean_DetA"],
        hota_result["mean_AssA"],
    ]

    # MOTA can be negative, so don't clamp the displayed value.
    plot_values = [
        value if value is not None else 0.0
        for value in values
    ]

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    bars = ax.bar(
        names,
        plot_values,
    )

    for bar, value in zip(bars, values):
        if value is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_ylabel("Metric value")
    ax.set_title(
        "Multi-Object Tracking Evaluation"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    path = os.path.join(
        output_dir,
        "metrics_summary.png",
    )

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # HOTA threshold curve
    # --------------------------------------------------------------

    thresholds = [
        item["threshold"]
        for item in hota_result["threshold_results"]
    ]

    hota_values = [
        item["HOTA"]
        for item in hota_result["threshold_results"]
    ]

    deta_values = [
        item["DetA"]
        for item in hota_result["threshold_results"]
    ]

    assa_values = [
        item["AssA"]
        for item in hota_result["threshold_results"]
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        thresholds,
        hota_values,
        marker="o",
        label="HOTA",
    )

    ax.plot(
        thresholds,
        deta_values,
        marker="s",
        label="DetA",
    )

    ax.plot(
        thresholds,
        assa_values,
        marker="^",
        label="AssA",
    )

    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Across IoU Thresholds")

    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    threshold_path = os.path.join(
        output_dir,
        "hota_threshold_curve.png",
    )

    fig.savefig(
        threshold_path,
        dpi=150,
    )

    plt.close(fig)

    print(f"Saved: {path}")
    print(f"Saved: {threshold_path}")


def create_sequence_plot(
    sequence,
    result,
    output_dir,
):
    """
    Plot the trajectories and identity assignments.

    GT trajectories use solid lines.
    Predicted trajectories use dashed lines.
    """

    gt_tracks = defaultdict(list)
    pred_tracks = defaultdict(list)

    for frame in sequence["frames"]:
        frame_index = frame["frame_index"]

        for obj in frame["gt"]:
            x1, y1, x2, y2 = obj["bbox"]

            center = (
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
            )

            gt_tracks[obj["id"]].append(
                (frame_index, center[0], center[1])
            )

        for obj in frame["pred"]:
            x1, y1, x2, y2 = obj["bbox"]

            center = (
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
            )

            pred_tracks[obj["id"]].append(
                (frame_index, center[0], center[1])
            )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for track_id, points in gt_tracks.items():
        points = np.asarray(points)

        ax.plot(
            points[:, 1],
            points[:, 2],
            marker="o",
            linestyle="-",
            label=f"GT ID {track_id}",
        )

    for track_id, points in pred_tracks.items():
        points = np.asarray(points)

        ax.plot(
            points[:, 1],
            points[:, 2],
            marker="x",
            linestyle="--",
            label=f"Pred ID {track_id}",
        )

    ax.set_xlabel("Image X")
    ax.set_ylabel("Image Y")
    ax.set_title("Ground Truth vs Predicted Trajectories")

    ax.invert_yaxis()
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    path = os.path.join(
        output_dir,
        "tracking_sequence.png",
    )

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(fig)

    print(f"Saved: {path}")


# ======================================================================
# REPORT
# ======================================================================

def print_report(result, hota_result):
    print("\n")
    print("=" * 72)
    print("MULTI-OBJECT TRACKING EVALUATION REPORT")
    print("=" * 72)

    print(f"Frames:              {result['frames']}")
    print(f"Ground-truth boxes:  {result['gt_detections']}")
    print(f"Predicted boxes:     {result['pred_detections']}")

    print("-" * 72)

    print(f"TP:                  {result['TP']}")
    print(f"FP:                  {result['FP']}")
    print(f"FN:                  {result['FN']}")
    print(f"Identity switches:   {result['IDSW']}")

    print("-" * 72)

    def display(value):
        return (
            "N/A"
            if value is None
            else f"{value:.4f}"
        )

    print(f"MOTA:                {display(result['MOTA'])}")
    print(
        "MOTP (mean IoU):     "
        f"{display(result['MOTP_mean_IoU'])}"
    )

    print(f"IDTP:                {result['IDTP']}")
    print(f"IDFP:                {result['IDFP']}")
    print(f"IDFN:                {result['IDFN']}")

    print(
        f"ID precision:        "
        f"{result['ID_precision']:.4f}"
    )

    print(
        f"ID recall:           "
        f"{result['ID_recall']:.4f}"
    )

    print(f"IDF1:                {result['IDF1']:.4f}")

    print("-" * 72)

    print(f"DetA:                {hota_result['mean_DetA']:.4f}")
    print(f"AssA:                {hota_result['mean_AssA']:.4f}")
    print(f"HOTA-inspired:       {hota_result['mean_HOTA']:.4f}")

    print("=" * 72)


# ======================================================================
# JSON INPUT / OUTPUT
# ======================================================================

def load_sequence(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        raw = json.load(file)

    return validate_sequence(raw)


def save_results(
    result,
    hota_result,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    # Frame details can be large. Keep the main result and
    # threshold-level metrics in the JSON report.
    result_without_frames = {
        key: value
        for key, value in result.items()
        if key != "frame_results"
    }

    output = {
        "lesson": LESSON_NAME,
        "metrics": result_without_frames,
        "hota_multi_threshold": hota_result,
    }

    path = os.path.join(
        output_dir,
        "evaluation_results.json",
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(f"Saved: {path}")


# ======================================================================
# NUMERICAL TESTS
# ======================================================================

def run_tests():
    print("\nRunning Lesson 55 tests...")
    print("=" * 72)

    # --------------------------------------------------------------
    # Test 1: identical box IoU
    # --------------------------------------------------------------

    box = [10, 10, 50, 50]

    assert np.isclose(
        box_iou(box, box),
        1.0,
    )

    print("Test 01 passed: identical-box IoU.")

    # --------------------------------------------------------------
    # Test 2: disjoint boxes
    # --------------------------------------------------------------

    assert np.isclose(
        box_iou(
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ),
        0.0,
    )

    print("Test 02 passed: disjoint-box IoU.")

    # --------------------------------------------------------------
    # Test 3: partial overlap
    # --------------------------------------------------------------

    iou = box_iou(
        [0, 0, 10, 10],
        [5, 0, 15, 10],
    )

    assert np.isclose(
        iou,
        1.0 / 3.0,
    )

    print("Test 03 passed: partial-overlap IoU.")

    # --------------------------------------------------------------
    # Test 4: zero-area boxes
    # --------------------------------------------------------------

    assert box_iou(
        [0, 0, 0, 0],
        [0, 0, 10, 10],
    ) == 0.0

    print("Test 04 passed: zero-area box handling.")

    # --------------------------------------------------------------
    # Test 5: IoU matrix
    # --------------------------------------------------------------

    matrix = calculate_iou_matrix(
        [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ],
        [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ],
    )

    assert matrix.shape == (2, 2)
    assert np.allclose(
        matrix,
        np.eye(2),
    )

    print("Test 05 passed: IoU matrix.")

    # --------------------------------------------------------------
    # Test 6: perfect one-object matching
    # --------------------------------------------------------------

    gt = [
        {"id": 1, "bbox": [0, 0, 10, 10]}
    ]

    pred = [
        {"id": 10, "bbox": [0, 0, 10, 10]}
    ]

    matches, unmatched_gt, unmatched_pred = match_frame(
        gt,
        pred,
        0.5,
    )

    assert len(matches) == 1
    assert len(unmatched_gt) == 0
    assert len(unmatched_pred) == 0

    print("Test 06 passed: perfect frame matching.")

    # --------------------------------------------------------------
    # Test 7: false negative
    # --------------------------------------------------------------

    matches, unmatched_gt, unmatched_pred = match_frame(
        gt,
        [],
        0.5,
    )

    assert len(matches) == 0
    assert len(unmatched_gt) == 1
    assert len(unmatched_pred) == 0

    print("Test 07 passed: false-negative identification.")

    # --------------------------------------------------------------
    # Test 8: false positive
    # --------------------------------------------------------------

    matches, unmatched_gt, unmatched_pred = match_frame(
        [],
        pred,
        0.5,
    )

    assert len(matches) == 0
    assert len(unmatched_gt) == 0
    assert len(unmatched_pred) == 1

    print("Test 08 passed: false-positive identification.")

    # --------------------------------------------------------------
    # Test 9: threshold rejection
    # --------------------------------------------------------------

    weak_pred = [
        {"id": 10, "bbox": [5, 0, 15, 10]}
    ]

    matches, unmatched_gt, unmatched_pred = match_frame(
        gt,
        weak_pred,
        0.5,
    )

    assert len(matches) == 0
    assert len(unmatched_gt) == 1
    assert len(unmatched_pred) == 1

    print("Test 09 passed: IoU threshold rejection.")

    # --------------------------------------------------------------
    # Test 10: invalid box
    # --------------------------------------------------------------

    try:
        validate_box([10, 10, 5, 5])
        raise AssertionError("Invalid box was accepted.")
    except ValueError:
        pass

    print("Test 10 passed: invalid box rejection.")

    # --------------------------------------------------------------
    # Test 11: perfect tracking metrics
    # --------------------------------------------------------------

    perfect_sequence = {
        "frames": [
            {
                "frame_index": 0,
                "gt": [
                    {"id": 1, "bbox": [0, 0, 10, 10]}
                ],
                "pred": [
                    {"id": 10, "bbox": [0, 0, 10, 10]}
                ],
            },
            {
                "frame_index": 1,
                "gt": [
                    {"id": 1, "bbox": [1, 0, 11, 10]}
                ],
                "pred": [
                    {"id": 10, "bbox": [1, 0, 11, 10]}
                ],
            },
        ]
    }

    result = TrackingEvaluator().evaluate(
        perfect_sequence
    )

    assert result["TP"] == 2
    assert result["FP"] == 0
    assert result["FN"] == 0
    assert result["IDSW"] == 0

    assert np.isclose(result["MOTA"], 1.0)
    assert np.isclose(result["MOTP_mean_IoU"], 1.0)
    assert np.isclose(result["IDF1"], 1.0)
    assert np.isclose(result["HOTA"], 1.0)

    print("Test 11 passed: perfect tracking metrics.")

    # --------------------------------------------------------------
    # Test 12: MOTA false-positive penalty
    # --------------------------------------------------------------

    fp_sequence = {
        "frames": [
            {
                "frame_index": 0,
                "gt": [
                    {"id": 1, "bbox": [0, 0, 10, 10]}
                ],
                "pred": [
                    {"id": 10, "bbox": [0, 0, 10, 10]},
                    {"id": 20, "bbox": [50, 50, 60, 60]},
                ],
            }
        ]
    }

    result = TrackingEvaluator().evaluate(
        fp_sequence
    )

    assert result["TP"] == 1
    assert result["FP"] == 1
    assert result["FN"] == 0
    assert np.isclose(result["MOTA"], 0.0)

    print("Test 12 passed: MOTA false-positive penalty.")

    # --------------------------------------------------------------
    # Test 13: MOTA false-negative penalty
    # --------------------------------------------------------------

    fn_sequence = {
        "frames": [
            {
                "frame_index": 0,
                "gt": [
                    {"id": 1, "bbox": [0, 0, 10, 10]},
                    {"id": 2, "bbox": [20, 20, 30, 30]},
                ],
                "pred": [
                    {"id": 10, "bbox": [0, 0, 10, 10]}
                ],
            }
        ]
    }

    result = TrackingEvaluator().evaluate(
        fn_sequence
    )

    assert result["TP"] == 1
    assert result["FN"] == 1
    assert np.isclose(result["MOTA"], 0.5)

    print("Test 13 passed: MOTA false-negative penalty.")

    # --------------------------------------------------------------
    # Test 14: identity switch
    # --------------------------------------------------------------

    switch_sequence = {
        "frames": [
            {
                "frame_index": 0,
                "gt": [
                    {"id": 1, "bbox": [0, 0, 10, 10]}
                ],
                "pred": [
                    {"id": 10, "bbox": [0, 0, 10, 10]}
                ],
            },
            {
                "frame_index": 1,
                "gt": [
                    {"id": 1, "bbox": [1, 0, 11, 10]}
                ],
                "pred": [
                    {"id": 20, "bbox": [1, 0, 11, 10]}
                ],
            },
        ]
    }

    result = TrackingEvaluator().evaluate(
        switch_sequence
    )

    assert result["IDSW"] == 1
    assert np.isclose(result["MOTA"], 0.5)

    print("Test 14 passed: identity-switch counting.")

    # --------------------------------------------------------------
    # Test 15: IDF1 global identity assignment
    # --------------------------------------------------------------

    # One ground-truth identity is consistently matched to a
    # different predicted ID. This is still a perfect global
    # identity assignment because IDs need not have equal numbers.
    id_sequence = {
        "frames": [
            {
                "frame_index": 0,
                "gt": [
                    {"id": 7, "bbox": [0, 0, 10, 10]}
                ],
                "pred": [
                    {"id": 99, "bbox": [0, 0, 10, 10]}
                ],
            },
            {
                "frame_index": 1,
                "gt": [
                    {"id": 7, "bbox": [1, 0, 11, 10]}
                ],
                "pred": [
                    {"id": 99, "bbox": [1, 0, 11, 10]}
                ],
            },
        ]
    }

    result = TrackingEvaluator().evaluate(
        id_sequence
    )

    assert result["IDTP"] == 2
    assert result["IDFP"] == 0
    assert result["IDFN"] == 0
    assert np.isclose(result["IDF1"], 1.0)

    print("Test 15 passed: global IDF1 assignment.")

    # --------------------------------------------------------------
    # Test 16: IDF1 detects fragmented identity
    # --------------------------------------------------------------

    fragmented_sequence = {
        "frames": [
            {
                "frame_index": 0,
                "gt": [
                    {"id": 1, "bbox": [0, 0, 10, 10]}
                ],
                "pred": [
                    {"id": 10, "bbox": [0, 0, 10, 10]}
                ],
            },
            {
                "frame_index": 1,
                "gt": [
                    {"id": 1, "bbox": [1, 0, 11, 10]}
                ],
                "pred": [
                    {"id": 20, "bbox": [1, 0, 11, 10]}
                ],
            },
        ]
    }

    result = TrackingEvaluator().evaluate(
        fragmented_sequence
    )

    assert result["IDTP"] == 1
    assert result["IDFP"] == 1
    assert result["IDFN"] == 1
    assert np.isclose(result["IDF1"], 0.5)

    print("Test 16 passed: IDF1 identity fragmentation.")

    # --------------------------------------------------------------
    # Test 17: DetA
    # --------------------------------------------------------------

    assert np.isclose(
        result["DetA"],
        1.0,
    )

    print("Test 17 passed: detection accuracy.")

    # --------------------------------------------------------------
    # Test 18: HOTA perfect sequence
    # --------------------------------------------------------------

    hota_result = evaluate_hota_thresholds(
        perfect_sequence,
        thresholds=[0.5, 0.75],
    )

    assert np.isclose(
        hota_result["mean_HOTA"],
        1.0,
    )

    print("Test 18 passed: multi-threshold HOTA evaluation.")

    # --------------------------------------------------------------
    # Test 19: duplicate IDs within one frame
    # --------------------------------------------------------------

    duplicate_sequence = {
        "frames": [
            {
                "frame_index": 0,
                "gt": [
                    {"id": 1, "bbox": [0, 0, 10, 10]},
                    {"id": 1, "bbox": [20, 20, 30, 30]},
                ],
                "pred": [],
            }
        ]
    }

    try:
        validate_sequence(duplicate_sequence)
        raise AssertionError(
            "Duplicate IDs were accepted."
        )
    except ValueError:
        pass

    print("Test 19 passed: duplicate identity rejection.")

    # --------------------------------------------------------------
    # Test 20: empty sequence
    # --------------------------------------------------------------

    empty_result = TrackingEvaluator().evaluate(
        {"frames": []}
    )

    assert empty_result["TP"] == 0
    assert empty_result["FP"] == 0
    assert empty_result["FN"] == 0
    assert empty_result["MOTA"] is None
    assert empty_result["IDF1"] == 0.0

    print("Test 20 passed: empty sequence handling.")

    # --------------------------------------------------------------
    # Test 21: invalid IoU threshold
    # --------------------------------------------------------------

    try:
        match_frame(
            gt,
            pred,
            1.5,
        )

        raise AssertionError(
            "Invalid IoU threshold was accepted."
        )
    except ValueError:
        pass

    print("Test 21 passed: IoU threshold validation.")

    # --------------------------------------------------------------
    # Test 22: demonstration sequence validation
    # --------------------------------------------------------------

    demo = make_demo_sequence()

    validated = validate_sequence(demo)

    assert len(validated["frames"]) == 20

    print("Test 22 passed: demonstration sequence validation.")

    # --------------------------------------------------------------
    # Test 23: no-match MOTP
    # --------------------------------------------------------------

    no_match_result = TrackingEvaluator().evaluate(
        {
            "frames": [
                {
                    "frame_index": 0,
                    "gt": [
                        {"id": 1, "bbox": [0, 0, 10, 10]}
                    ],
                    "pred": [
                        {"id": 10, "bbox": [50, 50, 60, 60]}
                    ],
                }
            ]
        }
    )

    assert no_match_result["MOTP_mean_IoU"] is None

    print("Test 23 passed: undefined MOTP with no matches.")

    # --------------------------------------------------------------
    # Test 24: MOTA can be negative
    # --------------------------------------------------------------

    negative_mota_sequence = {
        "frames": [
            {
                "frame_index": 0,
                "gt": [
                    {"id": 1, "bbox": [0, 0, 10, 10]}
                ],
                "pred": [
                    {"id": 10, "bbox": [50, 50, 60, 60]},
                    {"id": 20, "bbox": [70, 70, 80, 80]},
                    {"id": 30, "bbox": [90, 90, 100, 100]},
                ],
            }
        ]
    }

    negative_result = TrackingEvaluator().evaluate(
        negative_mota_sequence
    )

    assert negative_result["MOTA"] < 0.0

    print("Test 24 passed: negative MOTA behavior.")

    print("=" * 72)
    print("All 24 tests passed.")


# ======================================================================
# COMMAND LINE
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=LESSON_NAME
    )

    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        help="Path to a tracking annotation JSON file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run numerical tests only.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError(
            "IoU threshold must be between 0 and 1."
        )

    if args.check_only:
        run_tests()
        return

    run_tests()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    if args.annotations:
        print(
            f"\nLoading real annotated sequence: "
            f"{args.annotations}"
        )

        sequence = load_sequence(
            args.annotations
        )

    else:
        print(
            "\nNo annotation file supplied. "
            "Running the controlled mathematical demonstration."
        )

        sequence = validate_sequence(
            make_demo_sequence()
        )

    evaluator = TrackingEvaluator(
        iou_threshold=args.iou_threshold
    )

    result = evaluator.evaluate(
        sequence
    )

    hota_result = evaluate_hota_thresholds(
        sequence
    )

    print_report(
        result,
        hota_result,
    )

    create_metrics_plot(
        result,
        hota_result,
        args.output_dir,
    )

    create_sequence_plot(
        sequence,
        result,
        args.output_dir,
    )

    save_results(
        result,
        hota_result,
        args.output_dir,
    )

    print("\nLesson 55 completed successfully.")


if __name__ == "__main__":
    main()