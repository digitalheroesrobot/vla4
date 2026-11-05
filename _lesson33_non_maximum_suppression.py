"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 33 — NON-MAXIMUM SUPPRESSION (NMS)
======================================================================

LEARNING OBJECTIVES
-------------------
1. Understand why object detectors produce duplicate bounding boxes.
2. Understand Intersection over Union (IoU).
3. Implement greedy NMS from scratch using NumPy.
4. Implement class-aware and class-agnostic NMS.
5. Understand confidence thresholds and IoU thresholds.
6. Compare our implementation with OpenCV and PyTorch.
7. Visualize duplicate detections on REAL TACO waste photographs.

NO PRETRAINED MODELS
-------------------
This lesson does not train or load any pretrained model.

The overlapping candidate boxes used in the mathematical tests are
explicitly constructed to test NMS. They are not training images.

Real TACO photographs are used for image visualization.

----------------------------------------------------------------------
1. WHY DO WE NEED NMS?
----------------------------------------------------------------------

An object detector may predict multiple boxes around the same object.

Example:

    Detection A: confidence = 0.96
    Detection B: confidence = 0.91
    Detection C: confidence = 0.83

All three boxes may represent the same bottle.

Keeping all of them creates duplicate detections.

NMS selects the highest-confidence detection and removes sufficiently
overlapping lower-confidence detections.

----------------------------------------------------------------------
2. INTERSECTION OVER UNION
----------------------------------------------------------------------

For boxes A and B:

    IoU = Area(A intersection B) / Area(A union B)

IoU ranges from 0 to 1.

    IoU = 0: no overlap
    IoU = 1: identical boxes

----------------------------------------------------------------------
3. GREEDY NMS ALGORITHM
----------------------------------------------------------------------

Input:
    boxes: N x 4 array, format xyxy
    scores: N confidence scores
    threshold: IoU threshold

Algorithm:

    1. Sort detections by confidence, descending.
    2. Select the highest-confidence box.
    3. Calculate its IoU with every remaining box.
    4. Remove boxes whose IoU exceeds the threshold.
    5. Repeat until no boxes remain.

Output:
    Indices of retained boxes.

----------------------------------------------------------------------
4. CLASS-AWARE NMS
----------------------------------------------------------------------

In class-aware NMS, detections of different classes do not suppress
each other.

Example:

    Plastic bottle: box A
    Glass bottle:   box B

Even if A and B overlap, both may be valid because they represent
different predicted classes.

Class-agnostic NMS ignores class labels and suppresses overlapping
boxes regardless of category.

----------------------------------------------------------------------
5. SOFT-NMS
----------------------------------------------------------------------

Traditional NMS removes overlapping boxes.

Soft-NMS reduces their confidence instead.

For linear Soft-NMS:

    if IoU > threshold:

        score = score * (1 - IoU)

For Gaussian Soft-NMS:

        score = score * exp(-(IoU * IoU) / sigma)

Soft-NMS can preserve useful overlapping detections.

----------------------------------------------------------------------
6. REAL-WORLD ROBOTICS APPLICATION
----------------------------------------------------------------------

OpenArm V2 waste sorting:

    RGB camera
        |
        v
    Object detector
        |
        v
    Candidate bounding boxes
        |
        v
    NMS
        |
        v
    Final object detections
        |
        v
    Object localization
        |
        v
    ACT manipulation policy

NMS prevents the robot perception system from treating multiple
predictions of the same object as separate objects.

----------------------------------------------------------------------
7. INSTALLATION
----------------------------------------------------------------------

    pip install numpy matplotlib opencv-python torch torchvision pillow

----------------------------------------------------------------------
8. DATASET
----------------------------------------------------------------------

Expected:

    TACO/
        data/
            annotations.json
            batch_1/
            batch_2/
            ...

----------------------------------------------------------------------
9. RUN
----------------------------------------------------------------------

    python lesson33_non_maximum_suppression.py

Use a different TACO directory:

    python lesson33_non_maximum_suppression.py \
        --taco-root ./TACO/data

----------------------------------------------------------------------
10. EXPECTED OUTPUT
----------------------------------------------------------------------

    runs/lesson33/
        nms_visualization.png
        soft_nms_visualization.png
        real_taco_nms_visualization.png

All numerical tests should pass.

NEXT LESSON
-----------
Lesson 34 — Confidence Calibration for Object Detection.
======================================================================
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import nms as torch_nms


# ======================================================================
# 1. BASIC GEOMETRY
# ======================================================================

def box_area(boxes):
    """
    Calculate the area of xyxy boxes.

    Args:
        boxes: N x 4 array.

    Returns:
        N-element area array.
    """

    boxes = np.asarray(boxes, dtype=np.float32)

    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    return widths * heights


def pairwise_iou(box, boxes):
    """
    Calculate IoU between one box and multiple boxes.

    Args:
        box: shape (4,)
        boxes: shape (N, 4)

    Returns:
        IoU array of shape (N,)
    """

    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)

    intersection_x1 = np.maximum(box[0], boxes[:, 0])
    intersection_y1 = np.maximum(box[1], boxes[:, 1])
    intersection_x2 = np.minimum(box[2], boxes[:, 2])
    intersection_y2 = np.minimum(box[3], boxes[:, 3])

    intersection_width = np.maximum(
        0.0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = np.maximum(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection = intersection_width * intersection_height

    area_a = box_area(box.reshape(1, 4))[0]
    area_b = box_area(boxes)

    union = area_a + area_b - intersection

    iou = np.zeros_like(union)

    valid = union > 0

    iou[valid] = intersection[valid] / union[valid]

    return iou


def pairwise_iou_matrix(boxes):
    """
    Calculate the complete N x N IoU matrix.
    """

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    count = len(boxes)

    matrix = np.zeros((count, count), dtype=np.float32)

    for index in range(count):

        matrix[index] = pairwise_iou(
            boxes[index],
            boxes,
        )

    return matrix


# ======================================================================
# 2. GREEDY NMS FROM SCRATCH
# ======================================================================

def nms_numpy(
    boxes,
    scores,
    iou_threshold=0.5,
    score_threshold=0.0,
):
    """
    Greedy Non-Maximum Suppression implemented using NumPy.

    Args:
        boxes:
            N x 4 array, xyxy.

        scores:
            N confidence scores.

        iou_threshold:
            Suppress a box if IoU > threshold.

        score_threshold:
            Ignore boxes below this confidence.

    Returns:
        Retained original indices, ordered by descending confidence.
    """

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have equal length.")

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be between 0 and 1.")

    if not np.all(np.isfinite(boxes)):
        raise ValueError("Boxes contain non-finite coordinates.")

    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores contain non-finite values.")

    if np.any(boxes[:, 2] < boxes[:, 0]):
        raise ValueError("Invalid box: x2 must be >= x1.")

    if np.any(boxes[:, 3] < boxes[:, 1]):
        raise ValueError("Invalid box: y2 must be >= y1.")

    original_indices = np.arange(len(boxes))

    valid = scores >= score_threshold

    boxes = boxes[valid]
    scores = scores[valid]
    original_indices = original_indices[valid]

    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)

    # Stable sorting makes tie handling deterministic.
    order = np.argsort(-scores, kind="stable")

    keep = []

    while len(order) > 0:

        best_local_index = order[0]

        keep.append(original_indices[best_local_index])

        remaining = order[1:]

        if len(remaining) == 0:
            break

        overlaps = pairwise_iou(
            boxes[best_local_index],
            boxes[remaining],
        )

        # Suppress boxes with IoU greater than the threshold.
        order = remaining[overlaps <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


# ======================================================================
# 3. CLASS-AWARE NMS
# ======================================================================

def class_aware_nms(
    boxes,
    scores,
    labels,
    iou_threshold=0.5,
    score_threshold=0.0,
):
    """
    Run NMS independently for every class.

    Overlapping detections from different classes are not suppressed.
    """

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    if not (
        len(boxes) == len(scores) == len(labels)
    ):
        raise ValueError(
            "boxes, scores, and labels must have equal length."
        )

    retained = []

    for class_id in np.unique(labels):

        class_indices = np.where(labels == class_id)[0]

        class_keep = nms_numpy(
            boxes[class_indices],
            scores[class_indices],
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )

        retained.extend(class_indices[class_keep].tolist())

    if not retained:
        return np.zeros((0,), dtype=np.int64)

    retained = np.asarray(retained, dtype=np.int64)

    # Restore global confidence ordering.
    order = np.argsort(-scores[retained], kind="stable")

    return retained[order]


# ======================================================================
# 4. CLASS-AGNOSTIC NMS
# ======================================================================

def class_agnostic_nms(
    boxes,
    scores,
    iou_threshold=0.5,
    score_threshold=0.0,
):
    """
    Suppress overlapping detections regardless of their classes.
    """

    return nms_numpy(
        boxes,
        scores,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )


# ======================================================================
# 5. SOFT-NMS
# ======================================================================

def soft_nms(
    boxes,
    scores,
    iou_threshold=0.5,
    sigma=0.5,
    score_threshold=0.001,
    method="linear",
):
    """
    Soft-NMS.

    Unlike hard NMS, Soft-NMS decreases confidence rather than
    immediately deleting every overlapping candidate.

    Methods:
        linear
        gaussian

    Returns:
        keep_indices: original indices retained.
        updated_scores: final scores for retained indices.
    """

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1).copy()

    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have equal length.")

    if method not in ("linear", "gaussian"):
        raise ValueError("method must be 'linear' or 'gaussian'.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be between 0 and 1.")

    if not np.all(np.isfinite(boxes)):
        raise ValueError("Boxes contain non-finite coordinates.")

    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores contain non-finite values.")

    original_indices = np.arange(len(boxes))

    active = scores >= score_threshold

    boxes = boxes[active]
    scores = scores[active]
    original_indices = original_indices[active]

    keep = []
    kept_scores = []

    while len(scores) > 0:

        best_index = int(np.argmax(scores))

        best_box = boxes[best_index]
        best_score = float(scores[best_index])
        best_original_index = int(original_indices[best_index])

        keep.append(best_original_index)
        kept_scores.append(best_score)

        remaining_mask = np.ones(len(scores), dtype=bool)
        remaining_mask[best_index] = False

        boxes_remaining = boxes[remaining_mask]
        scores_remaining = scores[remaining_mask]
        indices_remaining = original_indices[remaining_mask]

        if len(boxes_remaining) == 0:
            break

        overlaps = pairwise_iou(
            best_box,
            boxes_remaining,
        )

        if method == "linear":

            decay = np.ones_like(overlaps)

            suppress = overlaps > iou_threshold

            decay[suppress] = 1.0 - overlaps[suppress]

        else:

            decay = np.exp(
                -(overlaps * overlaps) / sigma
            )

        scores_remaining = scores_remaining * decay

        valid = scores_remaining >= score_threshold

        boxes = boxes_remaining[valid]
        scores = scores_remaining[valid]
        original_indices = indices_remaining[valid]

    return (
        np.asarray(keep, dtype=np.int64),
        np.asarray(kept_scores, dtype=np.float32),
    )


# ======================================================================
# 6. PYTORCH NMS COMPARISON
# ======================================================================

def compare_torchvision_nms(
    boxes,
    scores,
    iou_threshold=0.5,
):
    """
    Compare our NumPy NMS against torchvision.ops.nms.
    """

    boxes_tensor = torch.as_tensor(
        boxes,
        dtype=torch.float32,
    )

    scores_tensor = torch.as_tensor(
        scores,
        dtype=torch.float32,
    )

    indices = torch_nms(
        boxes_tensor,
        scores_tensor,
        iou_threshold,
    )

    return indices.cpu().numpy()


# ======================================================================
# 7. VISUALIZATION
# ======================================================================

def draw_detections(
    image,
    boxes,
    scores,
    labels=None,
    keep=None,
    title="Detections",
):
    """
    Draw retained boxes in green and suppressed boxes in red.
    """

    output = image.copy()

    if keep is None:
        keep = np.arange(len(boxes))

    keep_set = set(np.asarray(keep).tolist())

    for index, box in enumerate(boxes):

        x1, y1, x2, y2 = np.round(box).astype(int)

        is_kept = index in keep_set

        color = (0, 255, 0) if is_kept else (255, 0, 0)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        text = f"{scores[index]:.2f}"

        if labels is not None:
            text = f"C{labels[index]} {text}"

        cv2.putText(
            output,
            text,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return output


def visualize_synthetic_candidates(output_dir):
    """
    Mathematical visualization of overlapping candidate boxes.

    These are deliberately constructed bounding boxes, not synthetic
    training photographs.
    """

    boxes = np.array(
        [
            [50, 50, 220, 220],
            [60, 60, 215, 215],
            [75, 65, 210, 210],
            [250, 70, 410, 230],
            [260, 80, 405, 225],
            [430, 80, 570, 220],
        ],
        dtype=np.float32,
    )

    scores = np.array(
        [0.96, 0.91, 0.83, 0.93, 0.78, 0.87],
        dtype=np.float32,
    )

    labels = np.array(
        [1, 1, 1, 2, 2, 1],
        dtype=np.int64,
    )

    image = np.ones(
        (300, 620, 3),
        dtype=np.uint8,
    ) * 245

    # Raw detections.
    raw_image = draw_detections(
        image,
        boxes,
        scores,
        labels,
        keep=[],
        title="Raw",
    )

    # Class-aware NMS.
    keep = class_aware_nms(
        boxes,
        scores,
        labels,
        iou_threshold=0.5,
    )

    nms_image = draw_detections(
        image,
        boxes,
        scores,
        labels,
        keep=keep,
    )

    # Soft-NMS.
    soft_keep, soft_scores = soft_nms(
        boxes,
        scores,
        iou_threshold=0.5,
        method="linear",
    )

    soft_image = image.copy()

    for index, box in enumerate(boxes):

        x1, y1, x2, y2 = np.round(box).astype(int)

        if index in soft_keep:
            score_index = np.where(soft_keep == index)[0][0]
            score = soft_scores[score_index]
            color = (0, 255, 0)
        else:
            score = scores[index]
            color = (255, 0, 0)

        cv2.rectangle(
            soft_image,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            soft_image,
            f"{score:.2f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    axes[0].imshow(raw_image)
    axes[0].set_title("Raw Candidate Boxes")

    axes[1].imshow(nms_image)
    axes[1].set_title("Hard NMS")

    axes[2].imshow(soft_image)
    axes[2].set_title("Soft-NMS")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()

    path = output_dir / "nms_visualization.png"

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Saved: {path}")


# ======================================================================
# 8. REAL TACO IMAGE VISUALIZATION
# ======================================================================

def load_taco_image(taco_root):
    """
    Load a real photograph from the TACO dataset.
    """

    taco_root = Path(taco_root)

    annotation_path = taco_root / "annotations.json"

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"TACO annotation file not found: {annotation_path}"
        )

    import json

    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    for record in data["images"]:

        file_name = Path(record["file_name"])

        candidates = [
            taco_root / file_name,
        ]

        for index, part in enumerate(file_name.parts):
            if part.startswith("batch_"):
                candidates.append(
                    taco_root.joinpath(*file_name.parts[index:])
                )
                break

        image_path = next(
            (
                path
                for path in candidates
                if path.is_file()
            ),
            None,
        )

        if image_path is None:
            continue

        image = cv2.imread(str(image_path))

        if image is not None:
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

    raise RuntimeError(
        "No readable TACO image found. Check the dataset paths."
    )


def visualize_real_taco_image(taco_root, output_dir):
    """
    Demonstrate NMS overlays on a real TACO photograph.

    The candidate boxes are illustrative proposals generated relative
    to the real image dimensions. They are NOT detector predictions.
    """

    image = load_taco_image(taco_root)

    height, width = image.shape[:2]

    # Illustrative candidate proposals positioned within the real
    # photograph. This is a visualization of NMS, not an evaluation.
    center_x = width * 0.5
    center_y = height * 0.5

    box_width = width * 0.35
    box_height = height * 0.35

    boxes = np.array(
        [
            [
                center_x - box_width / 2,
                center_y - box_height / 2,
                center_x + box_width / 2,
                center_y + box_height / 2,
            ],
            [
                center_x - box_width * 0.46,
                center_y - box_height * 0.46,
                center_x + box_width * 0.48,
                center_y + box_height * 0.48,
            ],
            [
                center_x - box_width * 0.35,
                center_y - box_height * 0.35,
                center_x + box_width * 0.60,
                center_y + box_height * 0.60,
            ],
        ],
        dtype=np.float32,
    )

    scores = np.array(
        [0.96, 0.88, 0.76],
        dtype=np.float32,
    )

    keep = nms_numpy(
        boxes,
        scores,
        iou_threshold=0.5,
    )

    raw = draw_detections(
        image,
        boxes,
        scores,
        keep=[],
    )

    filtered = draw_detections(
        image,
        boxes,
        scores,
        keep=keep,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7),
    )

    axes[0].imshow(raw)
    axes[0].set_title("Illustrative Candidate Boxes")

    axes[1].imshow(filtered)
    axes[1].set_title("After NMS")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()

    path = output_dir / "real_taco_nms_visualization.png"

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Saved: {path}")
    print(
        "NOTE: Real TACO photo used; boxes are illustrative proposals, "
        "not actual model predictions."
    )


# ======================================================================
# 9. NUMERICAL TESTS
# ======================================================================

def run_tests():
    print("\n" + "=" * 70)
    print("LESSON 33 — NUMERICAL TESTS")
    print("=" * 70)

    # Test 1: Area.
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [0, 0, 20, 10],
        ],
        dtype=np.float32,
    )

    areas = box_area(boxes)

    assert np.allclose(areas, [100, 200])

    print("Test 1 passed: box area.")

    # Test 2: Identical boxes have IoU 1.
    iou = pairwise_iou(
        [0, 0, 10, 10],
        [[0, 0, 10, 10]],
    )

    assert np.isclose(iou[0], 1.0)

    print("Test 2 passed: identical-box IoU.")

    # Test 3: Disjoint boxes have IoU 0.
    iou = pairwise_iou(
        [0, 0, 10, 10],
        [[20, 20, 30, 30]],
    )

    assert np.isclose(iou[0], 0.0)

    print("Test 3 passed: disjoint-box IoU.")

    # Test 4: Partial overlap.
    iou = pairwise_iou(
        [0, 0, 10, 10],
        [[5, 0, 15, 10]],
    )

    assert np.isclose(iou[0], 1.0 / 3.0)

    print("Test 4 passed: partial-overlap IoU.")

    # Test 5: IoU matrix.
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ],
        dtype=np.float32,
    )

    matrix = pairwise_iou_matrix(boxes)

    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.isclose(matrix[0, 1], 1.0)
    assert np.isclose(matrix[0, 2], 0.0)

    print("Test 5 passed: pairwise IoU matrix.")

    # Test 6: Duplicate removal.
    boxes = np.array(
        [
            [0, 0, 100, 100],
            [5, 5, 105, 105],
            [200, 200, 300, 300],
        ],
        dtype=np.float32,
    )

    scores = np.array([0.95, 0.80, 0.90])

    keep = nms_numpy(
        boxes,
        scores,
        iou_threshold=0.5,
    )

    assert set(keep.tolist()) == {0, 2}

    print("Test 6 passed: duplicate suppression.")

    # Test 7: Confidence threshold.
    keep = nms_numpy(
        boxes,
        scores,
        iou_threshold=0.5,
        score_threshold=0.85,
    )

    assert set(keep.tolist()) == {0, 2}

    print("Test 7 passed: confidence threshold.")

    # Test 8: Empty input.
    keep = nms_numpy(
        np.zeros((0, 4)),
        np.zeros((0,)),
    )

    assert len(keep) == 0

    print("Test 8 passed: empty input.")

    # Test 9: No overlap means all boxes survive.
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50],
        ],
        dtype=np.float32,
    )

    scores = np.array([0.9, 0.8, 0.7])

    keep = nms_numpy(boxes, scores, 0.5)

    assert len(keep) == 3

    print("Test 9 passed: non-overlapping boxes.")

    # Test 10: Class-aware NMS keeps overlapping different classes.
    boxes = np.array(
        [
            [0, 0, 100, 100],
            [5, 5, 105, 105],
        ],
        dtype=np.float32,
    )

    scores = np.array([0.95, 0.90])
    labels = np.array([1, 2])

    keep = class_aware_nms(
        boxes,
        scores,
        labels,
        iou_threshold=0.5,
    )

    assert len(keep) == 2

    print("Test 10 passed: class-aware NMS.")

    # Test 11: Class-agnostic NMS suppresses overlapping boxes.
    keep = class_agnostic_nms(
        boxes,
        scores,
        iou_threshold=0.5,
    )

    assert len(keep) == 1

    print("Test 11 passed: class-agnostic NMS.")

    # Test 12: Linear Soft-NMS reduces score.
    boxes = np.array(
        [
            [0, 0, 100, 100],
            [10, 10, 110, 110],
        ],
        dtype=np.float32,
    )

    scores = np.array([0.95, 0.90])

    keep, updated_scores = soft_nms(
        boxes,
        scores,
        iou_threshold=0.3,
        method="linear",
        score_threshold=0.001,
    )

    assert len(keep) == 2
    assert updated_scores[1] < 0.90

    print("Test 12 passed: linear Soft-NMS.")

    # Test 13: Gaussian Soft-NMS reduces overlapping score.
    keep, updated_scores = soft_nms(
        boxes,
        scores,
        method="gaussian",
        sigma=0.5,
        score_threshold=0.001,
    )

    assert len(keep) == 2
    assert updated_scores[1] < 0.90

    print("Test 13 passed: Gaussian Soft-NMS.")

    # Test 14: NumPy and torchvision NMS comparison.
    boxes = np.array(
        [
            [0, 0, 100, 100],
            [5, 5, 105, 105],
            [200, 200, 300, 300],
            [210, 210, 310, 310],
        ],
        dtype=np.float32,
    )

    scores = np.array([0.95, 0.80, 0.90, 0.70])

    numpy_keep = nms_numpy(
        boxes,
        scores,
        iou_threshold=0.5,
    )

    torch_keep = compare_torchvision_nms(
        boxes,
        scores,
        iou_threshold=0.5,
    )

    assert set(numpy_keep.tolist()) == set(torch_keep.tolist())

    print("Test 14 passed: NumPy vs torchvision NMS.")

    # Test 15: Invalid input.
    invalid_boxes = np.array(
        [
            [10, 10, 0, 0],
        ],
        dtype=np.float32,
    )

    try:
        nms_numpy(invalid_boxes, [0.9])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid boxes should raise ValueError.")

    print("Test 15 passed: invalid input rejection.")

    print("\nALL TESTS PASSED.\n")


# ======================================================================
# 10. MAIN
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Lesson 33 — Non-Maximum Suppression from scratch."
    )

    parser.add_argument(
        "--taco-root",
        type=str,
        default="./TACO/data",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./runs/lesson33",
    )

    parser.add_argument(
        "--skip-real-image",
        action="store_true",
        help="Skip TACO image visualization.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LESSON 33 — NON-MAXIMUM SUPPRESSION")
    print("=" * 70)

    run_tests()

    print("\nGenerating NMS visualizations...")

    visualize_synthetic_candidates(output_dir)

    if not args.skip_real_image:
        visualize_real_taco_image(
            args.taco_root,
            output_dir,
        )

    print("\n" + "=" * 70)
    print("LESSON 33 COMPLETED")
    print("=" * 70)

    print(f"Output directory: {output_dir.resolve()}")

    print("\nKey idea:")
    print(
        "NMS removes duplicate detections by retaining high-confidence "
        "boxes and suppressing boxes with excessive overlap."
    )


if __name__ == "__main__":
    main()