"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 26 — OBJECT DETECTION FUNDAMENTALS WITH REAL IMAGES
======================================================================

DATASET
-------
TACO — Trash Annotations in Context

Official repository:
https://github.com/pedropro/TACO

This lesson uses:
    - Real photographs of waste and litter.
    - Human-created bounding-box annotations.
    - Original COCO-format annotations.
    - Real category names and image dimensions.

NO SYNTHETIC TRAINING IMAGES.
NO GENERATED OBJECT IMAGES.
NO FAKE DETECTOR RESULTS.

IMPORTANT COURSE DIRECTION
--------------------------
From object detection onward, established detection architectures
such as YOLO are allowed.

We will learn the mathematics and detection pipeline, but we will
not unnecessarily reinvent production architectures.

LESSON 26 PURPOSE
-----------------
Understand the fundamental concepts required to use and evaluate
YOLO and other object detectors correctly.

STEP 1 — REAL DATASET LOADING
    Load real TACO photographs and COCO annotations.

STEP 2 — ANNOTATION VISUALIZATION
    Display original images and their real bounding boxes.

STEP 3 — BOUNDING BOX REPRESENTATIONS
    XYXY, XYWH, normalized coordinates.

STEP 4 — INTERSECTION OVER UNION
    Implement IoU from scratch.

STEP 5 — NON-MAXIMUM SUPPRESSION
    Implement class-aware NMS from scratch.

STEP 6 — DETECTION MATCHING
    Match predictions with ground-truth annotations.

STEP 7 — PRECISION AND RECALL
    Calculate TP, FP, FN, precision, recall and F1.

STEP 8 — PRECISION-RECALL CURVE
    Understand the relationship between confidence and recall.

STEP 9 — AVERAGE PRECISION
    Calculate interpolated AP.

STEP 10 — MEAN AVERAGE PRECISION
    Calculate mAP across classes and IoU thresholds.

STEP 11 — REAL DATASET ANALYSIS
    Analyze TACO category frequencies and annotation distributions.

STEP 12 — ROBOTICS APPLICATION
    Understand how object detection supports OpenArm V2 waste sorting.

IMPORTANT ABOUT METRICS
-----------------------
The numerical tests use controlled mathematical examples.

They are NOT presented as detector predictions.

Actual detector evaluation requires:
    1. Real images.
    2. Ground-truth annotations.
    3. Predictions from an actual trained or pretrained detector.

We will introduce actual YOLO predictions in subsequent lessons.

======================================================================
TACO DATASET DIRECTORY
======================================================================

Expected structure:

TACO/
    data/
        annotations.json
        batch_1/
            ...
        batch_2/
            ...
        batch_3/
            ...

Some images may fail to download because their original URLs
are unavailable. The script skips unavailable images and reports
the number successfully loaded.

======================================================================
RUN
======================================================================

python lesson26_object_detection_fundamentals.py

Or:

python lesson26_object_detection_fundamentals.py \
    --dataset-root ./TACO/data \
    --max-images 8

======================================================================
NEXT
======================================================================

LESSON 27 — YOLO OBJECT DETECTION FUNDAMENTALS
======================================================================
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

DEFAULT_DATASET_ROOT = "./TACO/data"

DEFAULT_MAX_IMAGES = 8

IOU_THRESHOLD = 0.5

CONFIDENCE_THRESHOLD = 0.25


# ======================================================================
# STEP 1 — COCO DATASET LOADER
# ======================================================================

class TACODataset:

    """
    Load real images and annotations from the TACO COCO dataset.

    COCO bounding boxes use:

        [x_min, y_min, width, height]

    We convert them into:

        [x_min, y_min, x_max, y_max]
    """

    def __init__(self, dataset_root):

        self.root = Path(dataset_root)

        self.annotation_file = (
            self.root / "annotations.json"
        )

        if not self.annotation_file.exists():

            raise FileNotFoundError(
                f"Cannot find {self.annotation_file}\n\n"
                "Download TACO first:\n"
                "git clone https://github.com/pedropro/TACO.git\n"
                "cd TACO\n"
                "python download.py\n"
            )

        with open(
            self.annotation_file,
            "r",
            encoding="utf-8",
        ) as file:

            self.data = json.load(file)

        self.images = {
            item["id"]: item
            for item in self.data["images"]
        }

        self.categories = {
            item["id"]: item["name"]
            for item in self.data["categories"]
        }

        self.annotations_by_image = {}

        for annotation in self.data["annotations"]:

            image_id = annotation["image_id"]

            self.annotations_by_image.setdefault(
                image_id,
                [],
            ).append(annotation)

        print("\nTACO DATASET LOADED")

        print(
            "Images in annotation file:",
            len(self.images),
        )

        print(
            "Categories:",
            len(self.categories),
        )

        print(
            "Annotations:",
            len(self.data["annotations"]),
        )

    def get_image_path(self, image_info):

        file_name = image_info["file_name"]

        return self.root / file_name

    def load_image(self, image_id):

        if image_id not in self.images:
            return None

        image_info = self.images[image_id]

        image_path = self.get_image_path(
            image_info
        )

        if not image_path.exists():
            return None

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            return None

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

    def get_annotations(self, image_id):

        annotations = self.annotations_by_image.get(
            image_id,
            [],
        )

        result = []

        for annotation in annotations:

            x, y, width, height = annotation["bbox"]

            box = np.array(
                [
                    x,
                    y,
                    x + width,
                    y + height,
                ],
                dtype=np.float32,
            )

            result.append(
                {
                    "box": box,
                    "category_id": annotation["category_id"],
                    "category_name": self.categories.get(
                        annotation["category_id"],
                        "Unknown",
                    ),
                    "area": annotation.get(
                        "area",
                        width * height,
                    ),
                    "iscrowd": annotation.get(
                        "iscrowd",
                        0,
                    ),
                }
            )

        return result

    def get_valid_image_ids(self):

        valid_ids = []

        for image_id in self.images:

            image = self.load_image(
                image_id
            )

            if image is not None:

                valid_ids.append(
                    image_id
                )

        return valid_ids


# ======================================================================
# STEP 2 — BOUNDING BOX CONVERSIONS
# ======================================================================

def xyxy_to_xywh(box):

    x1, y1, x2, y2 = np.asarray(
        box,
        dtype=np.float32,
    )

    width = x2 - x1

    height = y2 - y1

    center_x = (x1 + x2) / 2

    center_y = (y1 + y2) / 2

    return np.array(
        [
            center_x,
            center_y,
            width,
            height,
        ],
        dtype=np.float32,
    )


def xywh_to_xyxy(box):

    cx, cy, width, height = np.asarray(
        box,
        dtype=np.float32,
    )

    return np.array(
        [
            cx - width / 2,
            cy - height / 2,
            cx + width / 2,
            cy + height / 2,
        ],
        dtype=np.float32,
    )


def normalize_box(
    box,
    image_width,
    image_height,
):

    x1, y1, x2, y2 = box

    return np.array(
        [
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height,
        ],
        dtype=np.float32,
    )


def denormalize_box(
    box,
    image_width,
    image_height,
):

    x1, y1, x2, y2 = box

    return np.array(
        [
            x1 * image_width,
            y1 * image_height,
            x2 * image_width,
            y2 * image_height,
        ],
        dtype=np.float32,
    )


# ======================================================================
# STEP 3 — BOX AREA AND INTERSECTION
# ======================================================================

def box_area(box):

    x1, y1, x2, y2 = box

    width = max(
        0.0,
        float(x2 - x1),
    )

    height = max(
        0.0,
        float(y2 - y1),
    )

    return width * height


def intersection_area(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)

    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)

    y2 = min(ay2, by2)

    width = max(
        0.0,
        float(x2 - x1),
    )

    height = max(
        0.0,
        float(y2 - y1),
    )

    return width * height


# ======================================================================
# STEP 4 — INTERSECTION OVER UNION
# ======================================================================

def calculate_iou(box_a, box_b):

    intersection = intersection_area(
        box_a,
        box_b,
    )

    area_a = box_area(box_a)

    area_b = box_area(box_b)

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def pairwise_iou(boxes_a, boxes_b):

    result = np.zeros(
        (
            len(boxes_a),
            len(boxes_b),
        ),
        dtype=np.float32,
    )

    for i, box_a in enumerate(boxes_a):

        for j, box_b in enumerate(boxes_b):

            result[i, j] = calculate_iou(
                box_a,
                box_b,
            )

    return result


# ======================================================================
# STEP 5 — DETECTION STRUCTURE
# ======================================================================

class Detection:

    def __init__(
        self,
        box,
        category_id,
        confidence,
    ):

        self.box = np.asarray(
            box,
            dtype=np.float32,
        )

        if self.box.shape != (4,):

            raise ValueError(
                "Bounding box must have four coordinates."
            )

        if not np.all(np.isfinite(self.box)):

            raise ValueError(
                "Bounding box coordinates must be finite."
            )

        if self.box[2] < self.box[0]:

            raise ValueError(
                "Invalid horizontal box coordinates."
            )

        if self.box[3] < self.box[1]:

            raise ValueError(
                "Invalid vertical box coordinates."
            )

        if not 0 <= confidence <= 1:

            raise ValueError(
                "Confidence must be between 0 and 1."
            )

        self.category_id = int(
            category_id
        )

        self.confidence = float(
            confidence
        )


# ======================================================================
# STEP 6 — CONFIDENCE FILTERING
# ======================================================================

def confidence_filter(
    detections,
    threshold=0.25,
):

    return [
        detection
        for detection in detections
        if detection.confidence >= threshold
    ]


# ======================================================================
# STEP 7 — NON-MAXIMUM SUPPRESSION
# ======================================================================

def non_maximum_suppression(
    detections,
    iou_threshold=0.5,
    class_agnostic=False,
):

    """
    Greedy class-aware NMS.

    Sort by confidence.

    Keep the highest-confidence detection.

    Suppress overlapping detections of the same class.

    Different classes are preserved unless class_agnostic=True.
    """

    if not 0 <= iou_threshold <= 1:

        raise ValueError(
            "IoU threshold must be between 0 and 1."
        )

    remaining = sorted(
        detections,
        key=lambda d: d.confidence,
        reverse=True,
    )

    selected = []

    while remaining:

        best = remaining.pop(0)

        selected.append(
            best
        )

        survivors = []

        for candidate in remaining:

            if (
                not class_agnostic
                and candidate.category_id != best.category_id
            ):

                survivors.append(
                    candidate
                )

                continue

            iou = calculate_iou(
                best.box,
                candidate.box,
            )

            if iou <= iou_threshold:

                survivors.append(
                    candidate
                )

        remaining = survivors

    return selected


# ======================================================================
# STEP 8 — MATCH DETECTIONS WITH GROUND TRUTH
# ======================================================================

def match_detections(
    predictions,
    ground_truths,
    iou_threshold=0.5,
):

    """
    Ground truths must contain:
        box
        category_id

    Predictions must contain:
        box
        category_id
        confidence

    Greedy matching in descending confidence order.
    """

    predictions = sorted(
        predictions,
        key=lambda d: d.confidence,
        reverse=True,
    )

    matched = set()

    tp = []

    fp = []

    for prediction in predictions:

        best_iou = 0.0

        best_index = None

        for index, gt in enumerate(
            ground_truths
        ):

            if index in matched:
                continue

            if (
                prediction.category_id
                != gt["category_id"]
            ):
                continue

            iou = calculate_iou(
                prediction.box,
                gt["box"],
            )

            if iou > best_iou:

                best_iou = iou

                best_index = index

        if (
            best_index is not None
            and best_iou >= iou_threshold
        ):

            tp.append(1)

            fp.append(0)

            matched.add(
                best_index
            )

        else:

            tp.append(0)

            fp.append(1)

    fn = len(ground_truths) - len(matched)

    return (
        np.asarray(tp, dtype=np.int32),
        np.asarray(fp, dtype=np.int32),
        fn,
    )


# ======================================================================
# STEP 9 — PRECISION, RECALL AND F1
# ======================================================================

def calculate_precision_recall(
    tp,
    fp,
    fn,
):

    tp = int(tp)

    fp = int(fp)

    fn = int(fn)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    return precision, recall


def calculate_f1(
    precision,
    recall,
):

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ======================================================================
# STEP 10 — PRECISION-RECALL CURVE
# ======================================================================

def precision_recall_curve(
    predictions,
    ground_truths,
    iou_threshold=0.5,
):

    predictions = sorted(
        predictions,
        key=lambda d: d.confidence,
        reverse=True,
    )

    tp, fp, fn = match_detections(
        predictions,
        ground_truths,
        iou_threshold,
    )

    total_gt = len(ground_truths)

    if total_gt == 0:

        return (
            np.array([0.0]),
            np.array([0.0]),
        )

    cumulative_tp = np.cumsum(tp)

    cumulative_fp = np.cumsum(fp)

    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp,
        1e-12,
    )

    recall = cumulative_tp / total_gt

    precision = np.concatenate(
        [[1.0], precision]
    )

    recall = np.concatenate(
        [[0.0], recall]
    )

    return precision, recall


# ======================================================================
# STEP 11 — AVERAGE PRECISION
# ======================================================================

def calculate_average_precision(
    precision,
    recall,
):

    precision = np.asarray(
        precision,
        dtype=np.float64,
    )

    recall = np.asarray(
        recall,
        dtype=np.float64,
    )

    m_precision = np.concatenate(
        [
            [0.0],
            precision,
            [0.0],
        ]
    )

    m_recall = np.concatenate(
        [
            [0.0],
            recall,
            [1.0],
        ]
    )

    # Interpolated precision envelope.
    for index in range(
        len(m_precision) - 2,
        -1,
        -1,
    ):

        m_precision[index] = max(
            m_precision[index],
            m_precision[index + 1],
        )

    indices = np.where(
        m_recall[1:] != m_recall[:-1]
    )[0]

    ap = np.sum(
        (
            m_recall[indices + 1]
            - m_recall[indices]
        )
        * m_precision[indices + 1]
    )

    return float(ap)


# ======================================================================
# STEP 12 — DATASET-LEVEL AP AND MAP
# ======================================================================

def calculate_class_ap(
    predictions_by_image,
    ground_truths_by_image,
    category_id,
    iou_threshold=0.5,
):

    """
    Calculate AP for one category across multiple real images.

    Predictions must come from an actual detector.

    This function is implemented here for use in Lesson 27 onward.
    """

    if len(predictions_by_image) != len(
        ground_truths_by_image
    ):

        raise ValueError(
            "Prediction and ground-truth image counts differ."
        )

    all_predictions = []

    total_gt = 0

    for image_index in range(
        len(predictions_by_image)
    ):

        image_gt = [
            gt
            for gt in ground_truths_by_image[
                image_index
            ]
            if gt["category_id"] == category_id
        ]

        total_gt += len(image_gt)

        image_predictions = [
            prediction
            for prediction in predictions_by_image[
                image_index
            ]
            if prediction.category_id == category_id
        ]

        for prediction in image_predictions:

            all_predictions.append(
                (
                    prediction.confidence,
                    image_index,
                    prediction,
                )
            )

    if total_gt == 0:
        return None

    all_predictions.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    matched = [
        set()
        for _ in predictions_by_image
    ]

    tp = []

    fp = []

    for _, image_index, prediction in all_predictions:

        image_gt = [
            gt
            for gt in ground_truths_by_image[
                image_index
            ]
            if gt["category_id"] == category_id
        ]

        best_iou = 0.0

        best_index = None

        for index, gt in enumerate(image_gt):

            if index in matched[image_index]:
                continue

            iou = calculate_iou(
                prediction.box,
                gt["box"],
            )

            if iou > best_iou:

                best_iou = iou

                best_index = index

        if (
            best_index is not None
            and best_iou >= iou_threshold
        ):

            tp.append(1)

            fp.append(0)

            matched[image_index].add(
                best_index
            )

        else:

            tp.append(0)

            fp.append(1)

    cumulative_tp = np.cumsum(tp)

    cumulative_fp = np.cumsum(fp)

    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp,
        1e-12,
    )

    recall = cumulative_tp / total_gt

    precision = np.concatenate(
        [[1.0], precision]
    )

    recall = np.concatenate(
        [[0.0], recall]
    )

    return calculate_average_precision(
        precision,
        recall,
    )


def calculate_map(
    predictions_by_image,
    ground_truths_by_image,
    category_ids,
    iou_threshold=0.5,
):

    class_aps = {}

    for category_id in category_ids:

        ap = calculate_class_ap(
            predictions_by_image,
            ground_truths_by_image,
            category_id,
            iou_threshold,
        )

        if ap is not None:

            class_aps[category_id] = ap

    if not class_aps:

        return 0.0, {}

    return (
        float(
            np.mean(
                list(class_aps.values())
            )
        ),
        class_aps,
    )


def calculate_map_50_95(
    predictions_by_image,
    ground_truths_by_image,
    category_ids,
):

    thresholds = np.arange(
        0.50,
        0.951,
        0.05,
    )

    values = []

    for threshold in thresholds:

        mean_ap, _ = calculate_map(
            predictions_by_image,
            ground_truths_by_image,
            category_ids,
            iou_threshold=float(threshold),
        )

        values.append(mean_ap)

    return float(
        np.mean(values)
    )


# ======================================================================
# STEP 13 — REAL ANNOTATION VISUALIZATION
# ======================================================================

def visualize_real_image(
    image,
    annotations,
    categories,
    image_id,
):

    """
    Draw the actual human-created TACO annotations.
    """

    visual = image.copy()

    for annotation in annotations:

        x1, y1, x2, y2 = map(
            int,
            annotation["box"],
        )

        category_name = annotation[
            "category_name"
        ]

        color = (
            0,
            255,
            0,
        )

        cv2.rectangle(
            visual,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            visual,
            category_name,
            (
                x1,
                max(18, y1 - 5),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(visual)

    plt.title(
        f"TACO Real Image — Image ID {image_id}\n"
        f"Ground-truth annotations: {len(annotations)}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()


def visualize_real_dataset(
    dataset,
    max_images=8,
):

    print()
    print("=" * 70)
    print("REAL TACO IMAGE VISUALIZATION")
    print("=" * 70)

    valid_ids = dataset.get_valid_image_ids()

    print(
        "Images successfully loaded:",
        len(valid_ids),
    )

    if not valid_ids:

        raise RuntimeError(
            "No TACO images could be loaded. "
            "Check the dataset download and directory."
        )

    selected_ids = valid_ids[:max_images]

    for image_id in selected_ids:

        image = dataset.load_image(
            image_id
        )

        annotations = dataset.get_annotations(
            image_id
        )

        visualize_real_image(
            image,
            annotations,
            dataset.categories,
            image_id,
        )

    return valid_ids


# ======================================================================
# STEP 14 — REAL DATASET CATEGORY ANALYSIS
# ======================================================================

def analyze_categories(dataset):

    counter = Counter()

    for annotation in dataset.data["annotations"]:

        category_id = annotation["category_id"]

        category_name = dataset.categories.get(
            category_id,
            "Unknown",
        )

        counter[category_name] += 1

    print()
    print("=" * 70)
    print("TACO CATEGORY DISTRIBUTION")
    print("=" * 70)

    for name, count in counter.most_common(20):

        print(
            f"{name:<35} {count:>6}"
        )

    names = [
        name
        for name, _ in counter.most_common(20)
    ]

    counts = [
        count
        for _, count in counter.most_common(20)
    ]

    plt.figure(
        figsize=(12, 8)
    )

    plt.barh(
        names[::-1],
        counts[::-1],
    )

    plt.xlabel(
        "Number of annotations"
    )

    plt.title(
        "TACO — Top 20 Real Waste Categories"
    )

    plt.tight_layout()

    plt.show()

    return counter


# ======================================================================
# STEP 15 — REAL ANNOTATION BOX ANALYSIS
# ======================================================================

def analyze_bounding_boxes(dataset):

    widths = []

    heights = []

    areas = []

    for annotation in dataset.data["annotations"]:

        x, y, width, height = annotation["bbox"]

        if width <= 0 or height <= 0:
            continue

        widths.append(width)

        heights.append(height)

        areas.append(width * height)

    widths = np.asarray(widths)

    heights = np.asarray(heights)

    areas = np.asarray(areas)

    print()
    print("=" * 70)
    print("REAL BOUNDING BOX STATISTICS")
    print("=" * 70)

    print(
        "Valid boxes:",
        len(widths),
    )

    print(
        f"Mean width: {np.mean(widths):.2f} px"
    )

    print(
        f"Mean height: {np.mean(heights):.2f} px"
    )

    print(
        f"Median area: {np.median(areas):.2f} px^2"
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4),
    )

    axes[0].hist(
        widths,
        bins=40,
    )

    axes[0].set_title(
        "Real Bounding Box Width"
    )

    axes[0].set_xlabel(
        "Width (pixels)"
    )

    axes[1].hist(
        heights,
        bins=40,
    )

    axes[1].set_title(
        "Real Bounding Box Height"
    )

    axes[1].set_xlabel(
        "Height (pixels)"
    )

    axes[2].hist(
        np.log1p(areas),
        bins=40,
    )

    axes[2].set_title(
        "Log Real Bounding Box Area"
    )

    axes[2].set_xlabel(
        "log(1 + area)"
    )

    plt.tight_layout()

    plt.show()


# ======================================================================
# STEP 16 — NUMERICAL UNIT TESTS
# ======================================================================

def run_tests():

    print()
    print("=" * 70)
    print("RUNNING LESSON 26 NUMERICAL TESTS")
    print("=" * 70)

    # Test 1: XYXY to XYWH

    box = np.array(
        [10, 20, 50, 80],
        dtype=np.float32,
    )

    converted = xyxy_to_xywh(box)

    assert np.allclose(
        converted,
        [30, 50, 40, 60],
    )

    print("Test 1 passed: XYXY to XYWH.")

    # Test 2: XYWH to XYXY

    restored = xywh_to_xyxy(
        converted
    )

    assert np.allclose(
        restored,
        box,
    )

    print("Test 2 passed: XYWH to XYXY.")

    # Test 3: Normalization

    normalized = normalize_box(
        box,
        100,
        100,
    )

    assert np.allclose(
        normalized,
        [0.1, 0.2, 0.5, 0.8],
    )

    restored = denormalize_box(
        normalized,
        100,
        100,
    )

    assert np.allclose(
        restored,
        box,
    )

    print("Test 3 passed: Normalization.")

    # Test 4: Area

    assert box_area(
        [0, 0, 10, 20]
    ) == 200

    print("Test 4 passed: Bounding box area.")

    # Test 5: Intersection

    intersection = intersection_area(
        [0, 0, 10, 10],
        [5, 5, 15, 15],
    )

    assert intersection == 25

    print("Test 5 passed: Intersection.")

    # Test 6: IoU

    iou = calculate_iou(
        [0, 0, 10, 10],
        [5, 5, 15, 15],
    )

    assert np.isclose(
        iou,
        25 / 175,
    )

    print("Test 6 passed: IoU.")

    # Test 7: Identical boxes

    assert calculate_iou(
        [0, 0, 10, 10],
        [0, 0, 10, 10],
    ) == 1.0

    print("Test 7 passed: Identical-box IoU.")

    # Test 8: Non-overlapping boxes

    assert calculate_iou(
        [0, 0, 10, 10],
        [20, 20, 30, 30],
    ) == 0.0

    print("Test 8 passed: Non-overlapping IoU.")

    # Test 9: Pairwise IoU

    matrix = pairwise_iou(
        [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ],
        [
            [0, 0, 10, 10],
            [5, 5, 15, 15],
        ],
    )

    assert matrix.shape == (2, 2)

    assert np.isclose(
        matrix[0, 0],
        1.0,
    )

    print("Test 9 passed: Pairwise IoU.")

    # Test 10: Confidence filtering

    detections = [
        Detection(
            [0, 0, 100, 100],
            0,
            0.95,
        ),

        Detection(
            [0, 0, 100, 100],
            0,
            0.20,
        ),
    ]

    filtered = confidence_filter(
        detections,
        0.5,
    )

    assert len(filtered) == 1

    print("Test 10 passed: Confidence filtering.")

    # Test 11: NMS

    detections = [
        Detection(
            [0, 0, 100, 100],
            0,
            0.95,
        ),

        Detection(
            [5, 5, 105, 105],
            0,
            0.80,
        ),

        Detection(
            [200, 200, 300, 300],
            0,
            0.70,
        ),
    ]

    selected = non_maximum_suppression(
        detections,
        0.5,
    )

    assert len(selected) == 2

    print("Test 11 passed: NMS.")

    # Test 12: Class-aware NMS

    detections = [
        Detection(
            [0, 0, 100, 100],
            0,
            0.95,
        ),

        Detection(
            [0, 0, 100, 100],
            1,
            0.80,
        ),
    ]

    selected = non_maximum_suppression(
        detections,
        0.5,
        class_agnostic=False,
    )

    assert len(selected) == 2

    print("Test 12 passed: Class-aware NMS.")

    # Test 13: Precision and recall

    precision, recall = calculate_precision_recall(
        8,
        2,
        2,
    )

    assert np.isclose(
        precision,
        0.8,
    )

    assert np.isclose(
        recall,
        0.8,
    )

    print("Test 13 passed: Precision and recall.")

    # Test 14: F1

    f1 = calculate_f1(
        0.8,
        0.8,
    )

    assert np.isclose(
        f1,
        0.8,
    )

    print("Test 14 passed: F1 score.")

    # Test 15: Prediction matching

    gt = [
        {
            "box": np.array(
                [0, 0, 100, 100],
                dtype=np.float32,
            ),
            "category_id": 0,
        }
    ]

    predictions = [
        Detection(
            [5, 5, 105, 105],
            0,
            0.9,
        ),

        Detection(
            [200, 200, 300, 300],
            0,
            0.5,
        ),
    ]

    tp, fp, fn = match_detections(
        predictions,
        gt,
        0.5,
    )

    assert np.array_equal(
        tp,
        [1, 0],
    )

    assert np.array_equal(
        fp,
        [0, 1],
    )

    assert fn == 0

    print("Test 15 passed: Prediction matching.")

    # Test 16: Perfect AP

    ap = calculate_average_precision(
        [1.0, 1.0, 1.0],
        [0.0, 0.5, 1.0],
    )

    assert np.isclose(
        ap,
        1.0,
    )

    print("Test 16 passed: Perfect AP.")

    # Test 17: Dataset-level AP

    predictions_by_image = [
        [
            Detection(
                [0, 0, 100, 100],
                0,
                0.95,
            )
        ]
    ]

    ground_truths_by_image = [
        [
            {
                "box": np.array(
                    [0, 0, 100, 100],
                    dtype=np.float32,
                ),
                "category_id": 0,
            }
        ]
    ]

    map_value, class_aps = calculate_map(
        predictions_by_image,
        ground_truths_by_image,
        [0],
        0.5,
    )

    assert np.isclose(
        map_value,
        1.0,
    )

    assert np.isclose(
        class_aps[0],
        1.0,
    )

    print("Test 17 passed: Dataset-level mAP.")

    # Test 18: Empty NMS

    assert non_maximum_suppression(
        [],
        0.5,
    ) == []

    print("Test 18 passed: Empty NMS.")

    print()
    print("=" * 70)
    print("ALL LESSON 26 TESTS PASSED.")
    print("=" * 70)


# ======================================================================
# STEP 17 — MAIN EXPERIMENT
# ======================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("COMPUTER VISION — LESSON 26")
    print("OBJECT DETECTION FUNDAMENTALS")
    print("=" * 70)

    print(
        "NumPy:",
        np.__version__,
    )

    print(
        "OpenCV:",
        cv2.__version__,
    )

    # Mathematical verification.

    run_tests()

    # Load actual TACO dataset.

    dataset = TACODataset(
        args.dataset_root
    )

    # Inspect actual categories.

    analyze_categories(
        dataset
    )

    # Inspect actual box sizes.

    analyze_bounding_boxes(
        dataset
    )

    # Display real photographs and annotations.

    valid_ids = visualize_real_dataset(
        dataset,
        max_images=args.max_images,
    )

    print()
    print("=" * 70)
    print("LESSON 26 SUMMARY")
    print("=" * 70)

    print(
        "Real annotated images loaded:",
        len(valid_ids),
    )

    print(
        "TACO categories:",
        len(dataset.categories),
    )

    print(
        "Total real annotations:",
        len(dataset.data["annotations"]),
    )

    print()
    print("Implemented:")

    print("  - COCO annotation loading")

    print("  - Bounding box conversion")

    print("  - IoU")

    print("  - NMS")

    print("  - Precision and recall")

    print("  - F1 score")

    print("  - Precision-recall curve")

    print("  - AP and mAP")

    print()
    print(
        "No detector was trained or evaluated in this lesson."
    )

    print(
        "Actual YOLO predictions will be introduced in Lesson 27."
    )

    print()
    print("=" * 70)
    print("LESSON 26 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()