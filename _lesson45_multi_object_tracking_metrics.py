"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 45 — MULTI-OBJECT TRACKING EVALUATION
======================================================================

Filename:
    lesson45_multi_object_tracking_metrics.py

PHASE:
    Phase 4 — Tracking and Visual Memory

======================================================================
LEARNING OBJECTIVES
======================================================================

Implement and understand:

    1. Frame-by-frame object matching
    2. IoU-based evaluation
    3. False positives and false negatives
    4. Identity switches
    5. MOTA
    6. MOTP
    7. ID Precision, ID Recall, and IDF1
    8. HOTA-style detection and association accuracy
    9. MOTChallenge CSV input
    10. Comparing SORT, Deep SORT, and ByteTrack

======================================================================
1. WHY TRACKING EVALUATION IS DIFFERENT
======================================================================

Object detection evaluation asks:

    Did the model detect the object?

Tracking evaluation additionally asks:

    Did the tracker preserve the object's identity?

Example:

    Ground truth:

        Frame 1: Bottle A = ID 1
        Frame 2: Bottle A = ID 1
        Frame 3: Bottle A = ID 1

    Tracker A:

        Frame 1: Bottle A = ID 1
        Frame 2: Bottle A = ID 1
        Frame 3: Bottle A = ID 1

    Tracker B:

        Frame 1: Bottle A = ID 1
        Frame 2: Bottle A = ID 1
        Frame 3: Bottle A = ID 7

Both trackers detected the object.

But Tracker B changed its identity.

A detection-only metric cannot fully describe this problem.

======================================================================
2. MOTA — MULTIPLE OBJECT TRACKING ACCURACY
======================================================================

MOTA measures tracking errors:

    MOTA = 1 - (FN + FP + IDSW) / GT

Where:

    FN   = False Negatives
    FP   = False Positives
    IDSW = Identity Switches
    GT   = Number of ground-truth detections

Higher is better.

MOTA can be negative if tracking errors exceed the number
of ground-truth detections.

======================================================================
3. MOTP — MULTIPLE OBJECT TRACKING PRECISION
======================================================================

MOTP measures localization accuracy.

For every matched ground-truth/prediction pair:

    IoU = Intersection / Union

Then:

    MOTP = Sum(matched IoU) / Number of matches

Higher is better.

This implementation reports MOTP as mean IoU.

======================================================================
4. IDF1 — IDENTITY F1 SCORE
======================================================================

IDF1 evaluates identity consistency over the entire sequence.

First, calculate global identity correspondence.

Then:

    IDTP = correctly identified detections

    IDFP = incorrectly identified detections

    IDFN = missed identity detections

    ID Precision = IDTP / (IDTP + IDFP)

    ID Recall = IDTP / (IDTP + IDFN)

    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

A tracker that frequently changes IDs will receive a lower IDF1.

======================================================================
5. HOTA — HIGHER ORDER TRACKING ACCURACY
======================================================================

HOTA combines detection and association quality.

    HOTA = sqrt(DetA * AssA)

Detection accuracy:

    DetA = TP / (TP + 0.5 * (FP + FN))

Association accuracy:

    AssA measures whether matched detections belong to
    consistently associated ground-truth and predicted IDs.

HOTA is evaluated over multiple IoU thresholds:

    0.05, 0.10, 0.15, ..., 0.95

This lesson implements a transparent educational HOTA-style
calculation using frame-level IoU matching and global identity
co-occurrence statistics.

IMPORTANT:
    This is not a bit-for-bit reproduction of the official HOTA
    implementation. Official HOTA uses global alignment-aware
    matching. Use TrackEval for benchmark-compliant HOTA results.

======================================================================
6. FRAME MATCHING
======================================================================

For every frame:

    1. Calculate the IoU matrix.
    2. Apply Hungarian assignment.
    3. Reject matches below the IoU threshold.
    4. Count unmatched GT objects as FN.
    5. Count unmatched predictions as FP.

This is the foundation of all metrics in this lesson.

======================================================================
7. MOTCHALLENGE CSV FORMAT
======================================================================

Ground truth and prediction files use:

    frame,id,x,y,width,height,confidence,class,visibility

Example:

    1,1,100,100,50,80,1,0,1
    2,1,105,100,50,80,1,0,1
    3,1,110,100,50,80,1,0,1

Coordinates are converted to:

    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height

======================================================================
INSTALL
======================================================================

    pip install numpy

======================================================================
RUN
======================================================================

Run numerical tests and demonstration:

    python lesson45_multi_object_tracking_metrics.py

Evaluate MOTChallenge files:

    python lesson45_multi_object_tracking_metrics.py \
        --gt gt.txt \
        --pred tracker_output.txt

Change IoU matching threshold:

    python lesson45_multi_object_tracking_metrics.py \
        --gt gt.txt \
        --pred tracker_output.txt \
        --iou-threshold 0.5

======================================================================
ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste-sorting conveyor:

    Camera
      |
      v
    Object detector
      |
      v
    SORT / Deep SORT / ByteTrack
      |
      v
    Tracking output
      |
      v
    Compare with ground truth
      |
      v
    MOTA / MOTP / IDF1 / HOTA
      |
      v
    Select the most reliable tracker

A tracker with good detection accuracy but poor IDF1 can
cause the robot to confuse two nearby waste objects.

======================================================================
"""

import argparse
import csv
import json

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


# =====================================================================
# DATA STRUCTURE
# =====================================================================

@dataclass
class TrackDetection:

    frame: int

    track_id: int

    bbox: np.ndarray

    score: float = 1.0

    class_id: int = 0

    def __post_init__(self):

        self.frame = int(self.frame)

        self.track_id = int(self.track_id)

        self.bbox = np.asarray(
            self.bbox,
            dtype=np.float64
        )

        self.score = float(self.score)

        self.class_id = int(self.class_id)

        if self.frame < 1:
            raise ValueError("Frame numbers must start at 1.")

        if self.bbox.shape != (4,):
            raise ValueError("bbox must have shape (4,).")

        if not np.isfinite(self.bbox).all():
            raise ValueError("bbox must contain finite values.")

        if self.bbox[2] <= self.bbox[0]:
            raise ValueError("Bounding-box width must be positive.")

        if self.bbox[3] <= self.bbox[1]:
            raise ValueError("Bounding-box height must be positive.")


# =====================================================================
# BOUNDING BOX UTILITIES
# =====================================================================

def bbox_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = map(float, box_a)

    bx1, by1, bx2, by2 = map(float, box_b)

    ix1 = max(ax1, bx1)

    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)

    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)

    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)

    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / max(union, 1e-12)


# =====================================================================
# HUNGARIAN ASSIGNMENT
# =====================================================================

def hungarian_assignment(cost_matrix):

    cost = np.asarray(
        cost_matrix,
        dtype=np.float64
    )

    if cost.ndim != 2:
        raise ValueError("Cost matrix must be 2D.")

    rows, cols = cost.shape

    if rows == 0 or cols == 0:
        return []

    if not np.isfinite(cost).all():
        raise ValueError("Cost matrix must be finite.")

    transposed = False

    if rows > cols:

        cost = cost.T

        transposed = True

    n, m = cost.shape

    u = np.zeros(n + 1)

    v = np.zeros(m + 1)

    p = np.zeros(m + 1, dtype=np.int64)

    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):

        p[0] = i

        j0 = 0

        minv = np.full(
            m + 1,
            np.inf
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
                raise RuntimeError("Assignment failed.")

            for j in range(m + 1):

                if used[j]:

                    u[p[j]] += delta

                    v[j] -= delta

                else:

                    minv[j] -= delta

            j0 = j1

            if p[j0] == 0:
                break

        while True:

            j1 = way[j0]

            p[j0] = p[j1]

            j0 = j1

            if j0 == 0:
                break

    result = []

    for j in range(1, m + 1):

        if p[j] == 0:
            continue

        row = int(p[j] - 1)

        col = int(j - 1)

        if transposed:

            result.append((col, row))

        else:

            result.append((row, col))

    return sorted(result)


# =====================================================================
# GROUP DETECTIONS BY FRAME
# =====================================================================

def group_by_frame(detections):

    grouped = defaultdict(list)

    for detection in detections:

        grouped[detection.frame].append(
            detection
        )

    return dict(grouped)


# =====================================================================
# FRAME-LEVEL MATCHING
# =====================================================================

def match_frame(
    ground_truth,
    predictions,
    iou_threshold=0.5,
    class_aware=True
):

    """
    Match predictions to ground truth for one frame.

    Returns:
        matches:
            List of (gt_index, prediction_index, IoU)

        unmatched_gt:
            Set of unmatched GT indices

        unmatched_predictions:
            Set of unmatched prediction indices
    """

    if not ground_truth or not predictions:

        return (

            [],

            set(range(len(ground_truth))),

            set(range(len(predictions)))
        )

    iou_matrix = np.zeros(
        (
            len(ground_truth),
            len(predictions)
        ),
        dtype=np.float64
    )

    for i, gt in enumerate(ground_truth):

        for j, pred in enumerate(predictions):

            if class_aware and gt.class_id != pred.class_id:
                continue

            iou_matrix[i, j] = bbox_iou(
                gt.bbox,
                pred.bbox
            )

    cost_matrix = 1.0 - iou_matrix

    assignments = hungarian_assignment(
        cost_matrix
    )

    matches = []

    unmatched_gt = set(
        range(len(ground_truth))
    )

    unmatched_predictions = set(
        range(len(predictions))
    )

    for gt_index, pred_index in assignments:

        iou = iou_matrix[
            gt_index,
            pred_index
        ]

        if iou < iou_threshold:
            continue

        matches.append(

            (
                gt_index,

                pred_index,

                float(iou)
            )
        )

        unmatched_gt.discard(
            gt_index
        )

        unmatched_predictions.discard(
            pred_index
        )

    return (
        matches,
        unmatched_gt,
        unmatched_predictions
    )


# =====================================================================
# PREPARE FRAME MATCHES
# =====================================================================

def evaluate_at_threshold(
    ground_truth,
    predictions,
    iou_threshold=0.5,
    class_aware=True
):

    gt_frames = group_by_frame(
        ground_truth
    )

    pred_frames = group_by_frame(
        predictions
    )

    all_frames = sorted(
        set(gt_frames) | set(pred_frames)
    )

    frame_results = {}

    total_tp = 0

    total_fp = 0

    total_fn = 0

    total_iou = 0.0

    # Ground-truth ID -> predicted ID from previous matched frame.

    previous_identity = {}

    identity_switches = 0

    # Number of matched detections for each GT/predicted ID pair.

    pair_counts = defaultdict(int)

    # Total number of detections for each identity.

    gt_identity_counts = defaultdict(int)

    pred_identity_counts = defaultdict(int)

    for detection in ground_truth:

        gt_identity_counts[
            (detection.class_id, detection.track_id)
        ] += 1

    for detection in predictions:

        pred_identity_counts[
            (detection.class_id, detection.track_id)
        ] += 1

    for frame in all_frames:

        gt = gt_frames.get(
            frame,
            []
        )

        pred = pred_frames.get(
            frame,
            []
        )

        matches, unmatched_gt, unmatched_pred = match_frame(

            gt,

            pred,

            iou_threshold=iou_threshold,

            class_aware=class_aware
        )

        total_tp += len(matches)

        total_fn += len(unmatched_gt)

        total_fp += len(unmatched_pred)

        frame_match_records = []

        for gt_index, pred_index, iou in matches:

            gt_detection = gt[gt_index]

            pred_detection = pred[pred_index]

            gt_key = (
                gt_detection.class_id,
                gt_detection.track_id
            )

            pred_key = (
                pred_detection.class_id,
                pred_detection.track_id
            )

            # ---------------------------------------------------------
            # Identity switch:
            #
            # Same GT identity was previously associated with a
            # different predicted identity.
            # ---------------------------------------------------------

            if gt_key in previous_identity:

                if previous_identity[gt_key] != pred_key:

                    identity_switches += 1

            previous_identity[gt_key] = pred_key

            pair_counts[
                (gt_key, pred_key)
            ] += 1

            total_iou += iou

            frame_match_records.append(

                {
                    "gt_id": gt_detection.track_id,

                    "pred_id": pred_detection.track_id,

                    "class_id": gt_detection.class_id,

                    "iou": iou
                }
            )

        frame_results[frame] = {

            "matches": frame_match_records,

            "tp": len(matches),

            "fp": len(unmatched_pred),

            "fn": len(unmatched_gt)
        }

    return {

        "tp": total_tp,

        "fp": total_fp,

        "fn": total_fn,

        "id_switches": identity_switches,

        "sum_iou": total_iou,

        "frame_results": frame_results,

        "pair_counts": pair_counts,

        "gt_identity_counts": gt_identity_counts,

        "pred_identity_counts": pred_identity_counts
    }


# =====================================================================
# MOTA
# =====================================================================

def calculate_mota(
    tp,
    fp,
    fn,
    id_switches,
    total_gt
):

    if total_gt == 0:

        return float("nan")

    return 1.0 - (

        fn + fp + id_switches

    ) / total_gt


# =====================================================================
# MOTP
# =====================================================================

def calculate_motp(
    sum_iou,
    total_matches
):

    if total_matches == 0:

        return float("nan")

    return sum_iou / total_matches


# =====================================================================
# IDF1
# =====================================================================

def calculate_idf1(result):

    """
    Global identity assignment.

    Build a matrix:

        Rows    = GT identities
        Columns = Predicted identities

        Entry   = Number of matched detections between them.

    Hungarian assignment finds the maximum global identity
    correspondence.
    """

    pair_counts = result["pair_counts"]

    gt_counts = result["gt_identity_counts"]

    pred_counts = result["pred_identity_counts"]

    gt_ids = sorted(
        gt_counts.keys()
    )

    pred_ids = sorted(
        pred_counts.keys()
    )

    total_gt = sum(
        gt_counts.values()
    )

    total_pred = sum(
        pred_counts.values()
    )

    if not gt_ids or not pred_ids:

        return {

            "idtp": 0,

            "idfp": total_pred,

            "idfn": total_gt,

            "id_precision": 0.0,

            "id_recall": 0.0,

            "idf1": 0.0
        }

    count_matrix = np.zeros(

        (
            len(gt_ids),

            len(pred_ids)
        ),

        dtype=np.float64
    )

    gt_index = {

        identity: index

        for index, identity in enumerate(gt_ids)
    }

    pred_index = {

        identity: index

        for index, identity in enumerate(pred_ids)
    }

    for (gt_id, pred_id), count in pair_counts.items():

        count_matrix[
            gt_index[gt_id],
            pred_index[pred_id]
        ] = count

    # Hungarian minimizes cost.
    # Negating counts maximizes global correspondence.

    assignments = hungarian_assignment(
        -count_matrix
    )

    idtp = int(
        sum(
            count_matrix[i, j]

            for i, j in assignments
        )
    )

    idfn = int(
        total_gt - idtp
    )

    idfp = int(
        total_pred - idtp
    )

    denominator_precision = idtp + idfp

    denominator_recall = idtp + idfn

    id_precision = (

        idtp / denominator_precision

        if denominator_precision > 0 else 0.0
    )

    id_recall = (

        idtp / denominator_recall

        if denominator_recall > 0 else 0.0
    )

    idf1_denominator = (
        2 * idtp + idfp + idfn
    )

    idf1 = (

        2 * idtp / idf1_denominator

        if idf1_denominator > 0 else 0.0
    )

    return {

        "idtp": idtp,

        "idfp": idfp,

        "idfn": idfn,

        "id_precision": id_precision,

        "id_recall": id_recall,

        "idf1": idf1
    }


# =====================================================================
# HOTA-STYLE METRICS
# =====================================================================

def calculate_hota_at_threshold(result):

    """
    Educational HOTA-style calculation.

    DetA:
        TP / (TP + 0.5 * (FP + FN))

    AssA:
        For every matched GT/predicted identity pair:

        pair_TP / (GT detections + predicted detections - pair_TP)

    Association accuracy is averaged over matched detections.

    NOTE:
        Official HOTA includes global alignment-aware matching.
        This implementation uses ordinary frame-level IoU matching
        and is intended for learning, not benchmark reporting.
    """

    tp = result["tp"]

    fp = result["fp"]

    fn = result["fn"]

    if tp == 0:

        return {

            "det_a": 0.0,

            "ass_a": 0.0,

            "hota": 0.0
        }

    det_denominator = (
        tp + 0.5 * (fp + fn)
    )

    det_a = (

        tp / det_denominator

        if det_denominator > 0 else 0.0
    )

    pair_counts = result["pair_counts"]

    gt_counts = result["gt_identity_counts"]

    pred_counts = result["pred_identity_counts"]

    association_sum = 0.0

    for (gt_id, pred_id), pair_tp in pair_counts.items():

        gt_total = gt_counts[gt_id]

        pred_total = pred_counts[pred_id]

        union = (
            gt_total + pred_total - pair_tp
        )

        association_accuracy = (

            pair_tp / union

            if union > 0 else 0.0
        )

        association_sum += (
            pair_tp * association_accuracy
        )

    ass_a = association_sum / tp

    hota = np.sqrt(
        det_a * ass_a
    )

    return {

        "det_a": float(det_a),

        "ass_a": float(ass_a),

        "hota": float(hota)
    }


def calculate_hota(
    ground_truth,
    predictions,
    thresholds=None,
    class_aware=True
):

    if thresholds is None:

        thresholds = np.arange(
            0.05,
            1.00,
            0.05
        )

    values = []

    for threshold in thresholds:

        result = evaluate_at_threshold(

            ground_truth,

            predictions,

            iou_threshold=float(threshold),

            class_aware=class_aware
        )

        metrics = calculate_hota_at_threshold(
            result
        )

        metrics["threshold"] = float(
            threshold
        )

        values.append(
            metrics
        )

    hota_values = [
        item["hota"]
        for item in values
    ]

    det_a_values = [
        item["det_a"]
        for item in values
    ]

    ass_a_values = [
        item["ass_a"]
        for item in values
    ]

    return {

        "hota": float(
            np.mean(hota_values)
        ),

        "det_a": float(
            np.mean(det_a_values)
        ),

        "ass_a": float(
            np.mean(ass_a_values)
        ),

        "threshold_results": values
    }


# =====================================================================
# COMPLETE EVALUATION
# =====================================================================

def evaluate_tracking(
    ground_truth,
    predictions,
    iou_threshold=0.5,
    class_aware=True
):

    result = evaluate_at_threshold(

        ground_truth,

        predictions,

        iou_threshold=iou_threshold,

        class_aware=class_aware
    )

    total_gt = sum(
        result["gt_identity_counts"].values()
    )

    total_pred = sum(
        result["pred_identity_counts"].values()
    )

    mota = calculate_mota(

        tp=result["tp"],

        fp=result["fp"],

        fn=result["fn"],

        id_switches=result["id_switches"],

        total_gt=total_gt
    )

    motp = calculate_motp(

        sum_iou=result["sum_iou"],

        total_matches=result["tp"]
    )

    id_metrics = calculate_idf1(
        result
    )

    hota_metrics = calculate_hota(

        ground_truth,

        predictions,

        class_aware=class_aware
    )

    return {

        "iou_threshold": iou_threshold,

        "ground_truth_detections": total_gt,

        "predicted_detections": total_pred,

        "tp": result["tp"],

        "fp": result["fp"],

        "fn": result["fn"],

        "identity_switches": result["id_switches"],

        "mota": float(mota),

        "motp_mean_iou": float(motp),

        **id_metrics,

        "hota_educational": hota_metrics["hota"],

        "det_a_educational": hota_metrics["det_a"],

        "ass_a_educational": hota_metrics["ass_a"],

        "hota_threshold_results": hota_metrics["threshold_results"]
    }


# =====================================================================
# MOTCHALLENGE CSV LOADER
# =====================================================================

def load_motchallenge_csv(
    path,
    is_ground_truth=False
):

    """
    Expected columns:

        frame,id,x,y,w,h,confidence,class,visibility

    A header row is optional.

    Ground-truth files sometimes use negative confidence values
    for ignored regions. This educational loader skips rows whose
    confidence is negative.

    Coordinates are assumed to be 1-based frame numbers and
    top-left x/y with width/height.
    """

    detections = []

    with open(
        path,
        "r",
        newline=""
    ) as file:

        reader = csv.reader(file)

        for row_number, row in enumerate(reader, start=1):

            if not row:
                continue

            if row[0].strip().startswith("#"):
                continue

            try:

                frame = int(float(row[0]))

                track_id = int(float(row[1]))

                x = float(row[2])

                y = float(row[3])

                width = float(row[4])

                height = float(row[5])

                confidence = (

                    float(row[6])

                    if len(row) > 6 and row[6].strip()

                    else 1.0
                )

                class_id = (

                    int(float(row[7]))

                    if len(row) > 7 and row[7].strip()

                    else 0
                )

            except (ValueError, IndexError):

                # Allow an optional CSV header.

                if row_number == 1:
                    continue

                raise ValueError(

                    f"Invalid MOT CSV row {row_number}: {row}"
                )

            if confidence < 0:
                continue

            if width <= 0 or height <= 0:
                continue

            bbox = np.array(

                [
                    x,

                    y,

                    x + width,

                    y + height
                ],

                dtype=np.float64
            )

            detections.append(

                TrackDetection(

                    frame=frame,

                    track_id=track_id,

                    bbox=bbox,

                    score=confidence,

                    class_id=class_id
                )
            )

    return detections


# =====================================================================
# RESULT DISPLAY
# =====================================================================

def print_metrics(metrics):

    print("\n" + "=" * 70)

    print("MULTI-OBJECT TRACKING EVALUATION")

    print("=" * 70)

    print(
        f"Ground-truth detections : "
        f"{metrics['ground_truth_detections']}"
    )

    print(
        f"Predicted detections    : "
        f"{metrics['predicted_detections']}"
    )

    print(
        f"True positives          : {metrics['tp']}"
    )

    print(
        f"False positives         : {metrics['fp']}"
    )

    print(
        f"False negatives         : {metrics['fn']}"
    )

    print(
        f"Identity switches       : "
        f"{metrics['identity_switches']}"
    )

    print("\n--- Main Metrics ---")

    print(
        f"MOTA                    : {metrics['mota']:.4f}"
    )

    print(
        f"MOTP (mean IoU)         : "
        f"{metrics['motp_mean_iou']:.4f}"
    )

    print(
        f"ID Precision            : "
        f"{metrics['id_precision']:.4f}"
    )

    print(
        f"ID Recall               : "
        f"{metrics['id_recall']:.4f}"
    )

    print(
        f"IDF1                    : {metrics['idf1']:.4f}"
    )

    print("\n--- HOTA-style Educational Metrics ---")

    print(
        f"HOTA                    : "
        f"{metrics['hota_educational']:.4f}"
    )

    print(
        f"DetA                    : "
        f"{metrics['det_a_educational']:.4f}"
    )

    print(
        f"AssA                    : "
        f"{metrics['ass_a_educational']:.4f}"
    )

    print(
        "\nNOTE: HOTA above is an educational approximation, "
        "not official benchmark HOTA."
    )

    print("=" * 70)


# =====================================================================
# DEMONSTRATION SEQUENCE
# =====================================================================

def make_demo_sequence():

    """
    Numerical demonstration with two persistent objects.

    No synthetic photographs are created.

    Ground truth:

        Object A = ID 1
        Object B = ID 2

    Predictions:

        Frames 1-2:
            Correct identities.

        Frames 3-4:
            Object A changes predicted identity from 1 to 7.

        Frame 4:
            Object B is missed.

        Frame 5:
            One false positive appears.
    """

    ground_truth = []

    predictions = []

    for frame in range(1, 6):

        # -------------------------------------------------------------
        # Ground-truth object A.
        # -------------------------------------------------------------

        x_a = 100 + frame * 5

        ground_truth.append(

            TrackDetection(

                frame=frame,

                track_id=1,

                bbox=np.array(
                    [
                        x_a,
                        100,
                        x_a + 50,
                        180
                    ],
                    dtype=np.float64
                ),

                class_id=0
            )
        )

        # -------------------------------------------------------------
        # Ground-truth object B.
        # -------------------------------------------------------------

        x_b = 300 + frame * 3

        ground_truth.append(

            TrackDetection(

                frame=frame,

                track_id=2,

                bbox=np.array(
                    [
                        x_b,
                        100,
                        x_b + 50,
                        180
                    ],
                    dtype=np.float64
                ),

                class_id=0
            )
        )

        # -------------------------------------------------------------
        # Prediction for object A.
        # -------------------------------------------------------------

        predicted_id_a = (

            1 if frame <= 2 else 7
        )

        predictions.append(

            TrackDetection(

                frame=frame,

                track_id=predicted_id_a,

                bbox=np.array(
                    [
                        x_a + 2,
                        100,
                        x_a + 52,
                        180
                    ],
                    dtype=np.float64
                ),

                class_id=0
            )
        )

        # -------------------------------------------------------------
        # Prediction for object B.
        # Miss object B in frame 4.
        # -------------------------------------------------------------

        if frame != 4:

            predictions.append(

                TrackDetection(

                    frame=frame,

                    track_id=2,

                    bbox=np.array(
                        [
                            x_b + 1,
                            100,
                            x_b + 51,
                            180
                        ],
                        dtype=np.float64
                    ),

                    class_id=0
                )
            )

        # -------------------------------------------------------------
        # Add a false positive in frame 5.
        # -------------------------------------------------------------

        if frame == 5:

            predictions.append(

                TrackDetection(

                    frame=frame,

                    track_id=99,

                    bbox=np.array(
                        [
                            500,
                            100,
                            550,
                            180
                        ],
                        dtype=np.float64
                    ),

                    class_id=0
                )
            )

    return ground_truth, predictions


# =====================================================================
# UNIT TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print("LESSON 45 — NUMERICAL TESTS")

    print("=" * 70)

    # ---------------------------------------------------------------
    # Test 1: identical boxes have IoU = 1
    # ---------------------------------------------------------------

    box = np.array(
        [0, 0, 100, 100],
        dtype=np.float64
    )

    assert abs(
        bbox_iou(box, box) - 1.0
    ) < 1e-10

    print("Test 1 passed: identical-box IoU")

    # ---------------------------------------------------------------
    # Test 2: no overlap
    # ---------------------------------------------------------------

    box_b = np.array(
        [200, 200, 300, 300],
        dtype=np.float64
    )

    assert bbox_iou(box, box_b) == 0.0

    print("Test 2 passed: zero-overlap IoU")

    # ---------------------------------------------------------------
    # Test 3: known IoU
    # ---------------------------------------------------------------

    box_a = np.array(
        [0, 0, 100, 100]
    )

    box_b = np.array(
        [50, 0, 150, 100]
    )

    assert abs(
        bbox_iou(box_a, box_b) - 1.0 / 3.0
    ) < 1e-8

    print("Test 3 passed: known IoU")

    # ---------------------------------------------------------------
    # Test 4: Hungarian assignment
    # ---------------------------------------------------------------

    cost = np.array(
        [
            [4, 1, 3],
            [2, 0, 5],
            [3, 2, 2]
        ],
        dtype=np.float64
    )

    assignments = hungarian_assignment(
        cost
    )

    total_cost = sum(
        cost[i, j]
        for i, j in assignments
    )

    assert len(assignments) == 3

    assert abs(
        total_cost - 5.0
    ) < 1e-8

    print("Test 4 passed: Hungarian assignment")

    # ---------------------------------------------------------------
    # Test 5: empty frame matching
    # ---------------------------------------------------------------

    matches, unmatched_gt, unmatched_pred = match_frame(
        [],
        []
    )

    assert len(matches) == 0

    assert len(unmatched_gt) == 0

    assert len(unmatched_pred) == 0

    print("Test 5 passed: empty frame matching")

    # ---------------------------------------------------------------
    # Test 6: perfect matching
    # ---------------------------------------------------------------

    gt = [

        TrackDetection(
            1,
            1,
            [0, 0, 100, 100]
        )
    ]

    pred = [

        TrackDetection(
            1,
            10,
            [0, 0, 100, 100]
        )
    ]

    result = evaluate_at_threshold(
        gt,
        pred,
        iou_threshold=0.5
    )

    assert result["tp"] == 1

    assert result["fp"] == 0

    assert result["fn"] == 0

    print("Test 6 passed: perfect detection matching")

    # ---------------------------------------------------------------
    # Test 7: false positive
    # ---------------------------------------------------------------

    gt = [

        TrackDetection(
            1,
            1,
            [0, 0, 100, 100]
        )
    ]

    pred = [

        TrackDetection(
            1,
            10,
            [0, 0, 100, 100]
        ),

        TrackDetection(
            1,
            20,
            [200, 200, 300, 300]
        )
    ]

    result = evaluate_at_threshold(
        gt,
        pred
    )

    assert result["tp"] == 1

    assert result["fp"] == 1

    assert result["fn"] == 0

    print("Test 7 passed: false positive counting")

    # ---------------------------------------------------------------
    # Test 8: false negative
    # ---------------------------------------------------------------

    gt = [

        TrackDetection(
            1,
            1,
            [0, 0, 100, 100]
        ),

        TrackDetection(
            1,
            2,
            [200, 200, 300, 300]
        )
    ]

    pred = [

        TrackDetection(
            1,
            10,
            [0, 0, 100, 100]
        )
    ]

    result = evaluate_at_threshold(
        gt,
        pred
    )

    assert result["tp"] == 1

    assert result["fn"] == 1

    print("Test 8 passed: false negative counting")

    # ---------------------------------------------------------------
    # Test 9: MOTA
    # ---------------------------------------------------------------

    mota = calculate_mota(

        tp=8,

        fp=1,

        fn=1,

        id_switches=2,

        total_gt=10
    )

    assert abs(
        mota - 0.6
    ) < 1e-10

    print("Test 9 passed: MOTA calculation")

    # ---------------------------------------------------------------
    # Test 10: MOTP
    # ---------------------------------------------------------------

    motp = calculate_motp(
        sum_iou=2.4,
        total_matches=3
    )

    assert abs(
        motp - 0.8
    ) < 1e-10

    print("Test 10 passed: MOTP calculation")

    # ---------------------------------------------------------------
    # Test 11: IDF1 perfect identity
    # ---------------------------------------------------------------

    gt = [

        TrackDetection(
            1,
            1,
            [0, 0, 100, 100]
        ),

        TrackDetection(
            2,
            1,
            [0, 0, 100, 100]
        )
    ]

    pred = [

        TrackDetection(
            1,
            10,
            [0, 0, 100, 100]
        ),

        TrackDetection(
            2,
            10,
            [0, 0, 100, 100]
        )
    ]

    result = evaluate_at_threshold(
        gt,
        pred
    )

    id_metrics = calculate_idf1(
        result
    )

    assert id_metrics["idtp"] == 2

    assert id_metrics["idf1"] == 1.0

    print("Test 11 passed: perfect IDF1")

    # ---------------------------------------------------------------
    # Test 12: IDF1 with identity fragmentation
    # ---------------------------------------------------------------

    gt = [

        TrackDetection(
            1,
            1,
            [0, 0, 100, 100]
        ),

        TrackDetection(
            2,
            1,
            [0, 0, 100, 100]
        )
    ]

    pred = [

        TrackDetection(
            1,
            10,
            [0, 0, 100, 100]
        ),

        TrackDetection(
            2,
            20,
            [0, 0, 100, 100]
        )
    ]

    result = evaluate_at_threshold(
        gt,
        pred
    )

    id_metrics = calculate_idf1(
        result
    )

    assert id_metrics["idtp"] == 1

    assert id_metrics["idfp"] == 1

    assert id_metrics["idfn"] == 1

    assert abs(
        id_metrics["idf1"] - 0.5
    ) < 1e-10

    print("Test 12 passed: IDF1 identity fragmentation")

    # ---------------------------------------------------------------
    # Test 13: identity switch
    # ---------------------------------------------------------------

    gt = [

        TrackDetection(
            1,
            1,
            [0, 0, 100, 100]
        ),

        TrackDetection(
            2,
            1,
            [0, 0, 100, 100]
        )
    ]

    pred = [

        TrackDetection(
            1,
            10,
            [0, 0, 100, 100]
        ),

        TrackDetection(
            2,
            20,
            [0, 0, 100, 100]
        )
    ]

    result = evaluate_at_threshold(
        gt,
        pred
    )

    assert result["id_switches"] == 1

    print("Test 13 passed: identity switch")

    # ---------------------------------------------------------------
    # Test 14: no identity switch when ID remains constant
    # ---------------------------------------------------------------

    gt = [

        TrackDetection(
            1,
            1,
            [0, 0, 100, 100]
        ),

        TrackDetection(
            2,
            1,
            [0, 0, 100, 100]
        )
    ]

    pred = [

        TrackDetection(
            1,
            10,
            [0, 0, 100, 100]
        ),

        TrackDetection(
            2,
            10,
            [0, 0, 100, 100]
        )
    ]

    result = evaluate_at_threshold(
        gt,
        pred
    )

    assert result["id_switches"] == 0

    print("Test 14 passed: stable identity")

    # ---------------------------------------------------------------
    # Test 15: DetA and AssA for perfect tracking
    # ---------------------------------------------------------------

    result = evaluate_at_threshold(
        gt,
        pred
    )

    hota = calculate_hota_at_threshold(
        result
    )

    assert abs(
        hota["det_a"] - 1.0
    ) < 1e-10

    assert abs(
        hota["ass_a"] - 1.0
    ) < 1e-10

    assert abs(
        hota["hota"] - 1.0
    ) < 1e-10

    print("Test 15 passed: perfect HOTA components")

    # ---------------------------------------------------------------
    # Test 16: different classes cannot match
    # ---------------------------------------------------------------

    gt = [

        TrackDetection(
            1,
            1,
            [0, 0, 100, 100],
            class_id=0
        )
    ]

    pred = [

        TrackDetection(
            1,
            10,
            [0, 0, 100, 100],
            class_id=1
        )
    ]

    result = evaluate_at_threshold(
        gt,
        pred,
        class_aware=True
    )

    assert result["tp"] == 0

    assert result["fp"] == 1

    assert result["fn"] == 1

    print("Test 16 passed: class-aware matching")

    # ---------------------------------------------------------------
    # Test 17: full demo sequence
    # ---------------------------------------------------------------

    demo_gt, demo_pred = make_demo_sequence()

    metrics = evaluate_tracking(
        demo_gt,
        demo_pred
    )

    assert metrics["identity_switches"] >= 1

    assert metrics["fp"] == 1

    assert metrics["fn"] == 1

    print("Test 17 passed: full evaluation pipeline")

    print("\nAll 17 tests passed.")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description="Lesson 45 — MOT Evaluation"
    )

    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Ground-truth MOTChallenge CSV"
    )

    parser.add_argument(
        "--pred",
        type=str,
        default=None,
        help="Tracker prediction MOTChallenge CSV"
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--class-agnostic",
        action="store_true"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="runs/lesson45/metrics.json"
    )

    parser.add_argument(
        "--check-only",
        action="store_true"
    )

    args = parser.parse_args()

    if not 0.0 <= args.iou_threshold <= 1.0:

        parser.error(
            "IoU threshold must be between 0 and 1."
        )

    print("=" * 70)

    print("LESSON 45 — MULTI-OBJECT TRACKING EVALUATION")

    print("=" * 70)

    run_tests()

    if args.check_only:

        print("\nCheck-only mode completed.")

        return

    # ---------------------------------------------------------------
    # Load real MOTChallenge files if supplied.
    # ---------------------------------------------------------------

    if args.gt is not None or args.pred is not None:

        if args.gt is None or args.pred is None:

            parser.error(
                "Both --gt and --pred must be provided."
            )

        ground_truth = load_motchallenge_csv(
            args.gt,
            is_ground_truth=True
        )

        predictions = load_motchallenge_csv(
            args.pred,
            is_ground_truth=False
        )

        print("\nLoaded ground truth:", len(ground_truth))

        print("Loaded predictions:", len(predictions))

    else:

        print("\nNo MOTChallenge files supplied.")

        print("Running the built-in numerical demonstration.")

        ground_truth, predictions = make_demo_sequence()

    # ---------------------------------------------------------------
    # Evaluate.
    # ---------------------------------------------------------------

    metrics = evaluate_tracking(

        ground_truth,

        predictions,

        iou_threshold=args.iou_threshold,

        class_aware=not args.class_agnostic
    )

    print_metrics(
        metrics
    )

    # ---------------------------------------------------------------
    # Save JSON.
    # ---------------------------------------------------------------

    from pathlib import Path

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Remove detailed threshold results from the summary JSON
    # only if you prefer a compact report. Here we retain them.

    with open(
        output_path,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
            allow_nan=True
        )

    print("\nSaved evaluation:", output_path)


if __name__ == "__main__":

    main()