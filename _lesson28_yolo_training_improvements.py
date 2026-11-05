"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 28 — YOLO TRAINING IMPROVEMENTS AND REAL-WORLD EVALUATION
======================================================================

COURSE RULE
-----------
From Lesson 26 onward, object detection experiments use REAL IMAGES.

This lesson uses:
    - Real TACO waste photographs
    - Real YOLO-format bounding-box annotations
    - Ultralytics YOLO26
    - Randomly initialized model weights by default

IMPORTANT:
    We use an established YOLO implementation, as agreed for this
    object-detection section of the course.

    We do NOT use pretrained weights by default.

    YOLO26n is initialized from its architecture YAML file:
        YOLO("yolo26n.yaml")

    Training uses:
        pretrained=False

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1 — Understand training configuration
    Learn how optimizer, learning rate, weight decay, warm-up,
    batch size, and scheduler affect detector training.

STEP 2 — Improve training stability
    Use:
        - AdamW optimizer
        - Cosine learning-rate schedule
        - Warm-up
        - Early stopping
        - Weight decay

STEP 3 — Improve data augmentation
    Experiment with:
        - Mosaic augmentation
        - MixUp
        - HSV augmentation
        - Translation
        - Scaling
        - Horizontal flipping

STEP 4 — Understand detection metrics
    Evaluate:
        - Precision
        - Recall
        - F1 score
        - mAP@0.5
        - mAP@0.5:0.95

STEP 5 — Evaluate individual waste categories
    Compare performance for:
        - Plastic
        - Metal
        - Paper/cardboard
        - Glass
        - Other

    The actual class names are read from data.yaml.

STEP 6 — Analyze detection errors
    Compare predictions with ground-truth annotations.

    Identify:
        - True positives
        - False positives
        - False negatives

STEP 7 — Connect to OpenArm V2
    A waste-sorting robot needs more than a high mAP score.

    It must reliably detect objects before the robot attempts
    localization, grasp planning, and manipulation.

======================================================================
MATHEMATICAL FOUNDATION
======================================================================

1. PRECISION

                  TP
    Precision = --------
                 TP + FP

    TP = True Positives
    FP = False Positives

    High precision means fewer incorrect detections.

2. RECALL

                 TP
    Recall = -----------
             TP + FN

    FN = False Negatives

    High recall means fewer missed objects.

3. F1 SCORE

                     2 * Precision * Recall
    F1 = --------------------------------------------
              Precision + Recall

4. INTERSECTION OVER UNION

                 Area(B_prediction intersection B_GT)
    IoU = ------------------------------------------------
                 Area(B_prediction union B_GT)

5. AVERAGE PRECISION

    AP is the area under the precision-recall curve.

6. MEAN AVERAGE PRECISION

                  1
    mAP =         --- * sum(AP_class)
                  C

    C = number of object classes.

    mAP@0.5:
        A detection is considered correct when IoU >= 0.5.

    mAP@0.5:0.95:
        Average AP over IoU thresholds 0.50, 0.55, ..., 0.95.

======================================================================
DATASET REQUIREMENTS
======================================================================

This lesson expects the YOLO dataset created in Lesson 27:

    vla4/
    |
    +-- TACO/
    |   +-- data/
    |
    +-- TACO_yolo_dataset/
    |   +-- data.yaml
    |   +-- images/
    |   |   +-- train/
    |   |   +-- val/
    |   |
    |   +-- labels/
    |       +-- train/
    |       +-- val/
    |
    +-- lesson28_yolo_training_improvements.py

The data.yaml file must contain train, val, and names.

Example:

    path: /path/to/TACO_yolo_dataset
    train: images/train
    val: images/val

    names:
      0: plastic
      1: metal
      2: paper_cardboard
      3: glass
      4: other

======================================================================
INSTALLATION
======================================================================

    pip install -U ultralytics opencv-python matplotlib numpy pyyaml

======================================================================
EXPERIMENTS
======================================================================

EXPERIMENT A — Conservative augmentation

    python lesson28_yolo_training_improvements.py \
        --epochs 50 \
        --batch 8 \
        --augmentation conservative

EXPERIMENT B — Stronger augmentation

    python lesson28_yolo_training_improvements.py \
        --epochs 50 \
        --batch 8 \
        --augmentation strong

EXPERIMENT C — Short pipeline test

    python lesson28_yolo_training_improvements.py \
        --epochs 3 \
        --batch 4 \
        --max-audit-images 10

EXPERIMENT D — Evaluate an existing checkpoint

    python lesson28_yolo_training_improvements.py \
        --skip-training \
        --weights runs/lesson28/strong/weights/best.pt

======================================================================
EXPECTED OUTPUT
======================================================================

    runs/lesson28/
        conservative/
            weights/
            results.csv
            results.png

        strong/
            weights/
            results.csv
            results.png

        evaluation/
            per_class_metrics.csv
            detection_error_analysis.csv
            evaluation_summary.json
            audit_images/

======================================================================
IMPORTANT
======================================================================

Random initialization on a relatively small TACO dataset can produce
low detection accuracy, especially with a limited number of epochs.

A short run is a pipeline test, not proof of production readiness.

For a real OpenArm V2 deployment, evaluate:
    - Per-class recall
    - False detection rate
    - Missed-object rate
    - Detection latency
    - Performance under different lighting conditions
    - Performance with overlapping and partially occluded waste

======================================================================
NEXT LESSON
======================================================================

LESSON 29 — FASTER R-CNN OBJECT DETECTION FROM SCRATCH

We will study the two-stage detection approach and compare it with YOLO.
======================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time

from pathlib import Path

import cv2
import numpy as np
import yaml


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

DEFAULT_DATASET = Path("./TACO_yolo_dataset")
DEFAULT_PROJECT = Path("./runs/lesson28")

DEFAULT_MODEL = "yolo26n.yaml"

IOU_MATCH_THRESHOLD = 0.50


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    except ImportError:
        pass


# ======================================================================
# BASIC MATHEMATICS
# ======================================================================

def box_iou(box_a, box_b):
    """
    Calculate IoU between two xyxy bounding boxes.

    box = [x1, y1, x2, y2]
    """

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

    if union <= 0:
        return 0.0

    return intersection / union


def precision_recall_f1(tp, fp, fn):
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)

    f1 = (
        2.0 * precision * recall
        / max(precision + recall, 1e-12)
    )

    return precision, recall, f1


# ======================================================================
# YOLO LABEL PARSING
# ======================================================================

def load_yolo_labels(label_path, image_width, image_height):
    """
    Read YOLO-format labels.

    Each line:

        class_id center_x center_y width height

    Coordinates are normalized to [0, 1].

    Return:
        List of dictionaries containing:
            class_id
            box_xyxy
    """

    label_path = Path(label_path)

    if not label_path.exists():
        return []

    objects = []

    with open(label_path, "r", encoding="utf-8") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(float(values[0]))

            cx, cy, bw, bh = map(float, values[1:])

            x1 = (cx - bw / 2.0) * image_width
            y1 = (cy - bh / 2.0) * image_height

            x2 = (cx + bw / 2.0) * image_width
            y2 = (cy + bh / 2.0) * image_height

            x1 = np.clip(x1, 0, image_width - 1)
            y1 = np.clip(y1, 0, image_height - 1)

            x2 = np.clip(x2, 0, image_width - 1)
            y2 = np.clip(y2, 0, image_height - 1)

            if x2 <= x1 or y2 <= y1:
                continue

            objects.append({
                "class_id": class_id,
                "box_xyxy": np.array(
                    [x1, y1, x2, y2],
                    dtype=np.float32,
                ),
            })

    return objects


# ======================================================================
# DATASET CONFIGURATION
# ======================================================================

def load_dataset_config(dataset_yaml):
    dataset_yaml = Path(dataset_yaml).resolve()

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {dataset_yaml}\n"
            "Run Lesson 27 first to prepare the TACO dataset."
        )

    with open(dataset_yaml, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid data.yaml format.")

    if "train" not in config or "val" not in config:
        raise ValueError(
            "data.yaml must contain both train and val."
        )

    if "names" not in config:
        raise ValueError("data.yaml must contain class names.")

    dataset_root = Path(
        config.get("path", dataset_yaml.parent)
    )

    if not dataset_root.is_absolute():
        dataset_root = (dataset_yaml.parent / dataset_root).resolve()

    config["path"] = str(dataset_root)

    return config


def get_class_names(config):
    names = config["names"]

    if isinstance(names, list):
        return {
            index: str(name)
            for index, name in enumerate(names)
        }

    if isinstance(names, dict):
        return {
            int(key): str(value)
            for key, value in names.items()
        }

    raise ValueError("Unsupported class names format.")


def resolve_split_path(dataset_yaml, config, split):
    dataset_root = Path(config["path"])

    split_path = Path(config[split])

    if split_path.is_absolute():
        return split_path

    return dataset_root / split_path


def discover_images(dataset_yaml, config, split="val"):
    split_path = resolve_split_path(
        dataset_yaml,
        config,
        split,
    )

    if split_path.is_file():
        image_paths = [split_path]

    elif split_path.is_dir():

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        image_paths = [
            path
            for path in split_path.rglob("*")
            if path.suffix.lower() in extensions
        ]

    else:
        raise FileNotFoundError(
            f"Image split not found: {split_path}"
        )

    return sorted(image_paths)


def image_to_label_path(image_path):
    """
    Convert:

        images/val/example.jpg

    to:

        labels/val/example.txt
    """

    image_path = Path(image_path)

    parts = list(image_path.parts)

    try:
        index = parts.index("images")
        parts[index] = "labels"

    except ValueError:
        raise ValueError(
            f"Could not find 'images' directory in {image_path}"
        )

    label_path = Path(*parts).with_suffix(".txt")

    return label_path


# ======================================================================
# DATASET VALIDATION
# ======================================================================

def validate_dataset(dataset_yaml, config, class_names):
    print("\n" + "=" * 72)
    print("DATASET VALIDATION")
    print("=" * 72)

    for split in ("train", "val"):

        images = discover_images(
            dataset_yaml,
            config,
            split,
        )

        print(f"\n{split.upper()}")

        print("Images:", len(images))

        if not images:
            raise RuntimeError(
                f"No images found in {split} split."
            )

        total_objects = 0
        missing_labels = 0

        for image_path in images:

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            height, width = image.shape[:2]

            label_path = image_to_label_path(image_path)

            if not label_path.exists():
                missing_labels += 1
                continue

            labels = load_yolo_labels(
                label_path,
                width,
                height,
            )

            total_objects += len(labels)

            for obj in labels:

                class_id = obj["class_id"]

                if class_id not in class_names:
                    raise ValueError(
                        f"Unknown class ID {class_id} "
                        f"in {label_path}"
                    )

        print("Objects:", total_objects)
        print("Images without label files:", missing_labels)

    print("\nDataset validation completed.")


# ======================================================================
# TRAINING CONFIGURATION
# ======================================================================

def get_augmentation_config(mode):
    """
    Conservative:
        Less aggressive image augmentation.

    Strong:
        More aggressive augmentation for greater visual variation.

    These settings are experiments, not universally optimal values.
    """

    if mode == "conservative":

        return {
            "mosaic": 0.5,
            "mixup": 0.0,
            "hsv_h": 0.01,
            "hsv_s": 0.35,
            "hsv_v": 0.25,
            "degrees": 3.0,
            "translate": 0.05,
            "scale": 0.25,
            "fliplr": 0.5,
        }

    if mode == "strong":

        return {
            "mosaic": 1.0,
            "mixup": 0.10,
            "hsv_h": 0.02,
            "hsv_s": 0.60,
            "hsv_v": 0.40,
            "degrees": 8.0,
            "translate": 0.10,
            "scale": 0.50,
            "fliplr": 0.5,
        }

    raise ValueError(f"Unknown augmentation mode: {mode}")


def build_training_config(args):
    augmentation = get_augmentation_config(
        args.augmentation
    )

    config = {

        # Training duration
        "epochs": args.epochs,
        "patience": args.patience,

        # Input image resolution
        "imgsz": args.imgsz,

        # Batch size
        "batch": args.batch,

        # Optimizer
        "optimizer": "AdamW",
        "lr0": args.lr,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,

        # Learning-rate scheduling
        "cos_lr": True,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,

        # Loss-related parameters
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,

        # Augmentation
        **augmentation,

        # Training stability
        "close_mosaic": 10,
        "amp": True,
        "cache": False,

        # Data loading
        "workers": args.workers,

        # Reproducibility
        "seed": SEED,
        "deterministic": True,

        # Output
        "project": str(args.project),
        "name": args.experiment,
        "exist_ok": True,
        "plots": True,
        "save": True,
        "save_period": 10,
        "verbose": True,

    }

    if args.device is not None:
        config["device"] = args.device

    return config


# ======================================================================
# METRIC EXTRACTION
# ======================================================================

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def extract_class_metrics(metrics, class_names):
    """
    Extract class-level metrics from Ultralytics validation results.
    """

    box_metrics = metrics.box

    class_indices = getattr(
        box_metrics,
        "ap_class_index",
        [],
    )

    precision_values = getattr(
        box_metrics,
        "p",
        [],
    )

    recall_values = getattr(
        box_metrics,
        "r",
        [],
    )

    ap50_values = getattr(
        box_metrics,
        "ap50",
        [],
    )

    ap_values = getattr(
        box_metrics,
        "ap",
        [],
    )

    class_results = []

    for index, class_id in enumerate(class_indices):

        class_id = int(class_id)

        if class_id not in class_names:
            continue

        precision = safe_float(precision_values[index])
        recall = safe_float(recall_values[index])
        ap50 = safe_float(ap50_values[index])
        ap5095 = safe_float(
            np.mean(ap_values[index])
        )

        _, _, f1 = precision_recall_f1(
            precision,
            1.0 - precision,
            1.0 - recall,
        )

        class_results.append({

            "class_id": class_id,
            "class_name": class_names[class_id],

            "precision": precision,
            "recall": recall,
            "f1": f1,

            "ap50": ap50,
            "ap50_95": ap5095,

        })

    return class_results


def save_class_metrics(class_results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "class_id",
        "class_name",
        "precision",
        "recall",
        "f1",
        "ap50",
        "ap50_95",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(class_results)


def print_class_metrics(class_results):
    print("\n" + "=" * 90)
    print("PER-CLASS DETECTION METRICS")
    print("=" * 90)

    print(
        f"{'Class':24s}"
        f"{'Precision':>12s}"
        f"{'Recall':>12s}"
        f"{'F1':>12s}"
        f"{'AP50':>12s}"
        f"{'AP50-95':>12s}"
    )

    print("-" * 90)

    for row in class_results:

        print(
            f"{row['class_name']:24s}"
            f"{row['precision']:12.4f}"
            f"{row['recall']:12.4f}"
            f"{row['f1']:12.4f}"
            f"{row['ap50']:12.4f}"
            f"{row['ap50_95']:12.4f}"
        )

    print("-" * 90)


# ======================================================================
# DETECTION ERROR ANALYSIS
# ======================================================================

def match_predictions_to_ground_truth(
    predictions,
    ground_truth,
    iou_threshold=IOU_MATCH_THRESHOLD,
):
    """
    Greedy confidence-ordered matching.

    Predictions:
        class_id
        confidence
        box_xyxy

    Ground truth:
        class_id
        box_xyxy

    A prediction is a true positive when:
        1. Class ID matches.
        2. IoU >= threshold.
        3. Ground-truth object has not already been matched.
    """

    sorted_predictions = sorted(
        predictions,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    matched_gt = set()

    matches = []

    for pred_index, prediction in enumerate(
        sorted_predictions
    ):

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt in enumerate(ground_truth):

            if gt_index in matched_gt:
                continue

            if prediction["class_id"] != gt["class_id"]:
                continue

            iou = box_iou(
                prediction["box_xyxy"],
                gt["box_xyxy"],
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= iou_threshold
        ):

            matched_gt.add(best_gt_index)

            matches.append({
                "prediction": prediction,
                "ground_truth": ground_truth[best_gt_index],
                "iou": best_iou,
                "status": "TP",
            })

        else:

            matches.append({
                "prediction": prediction,
                "ground_truth": None,
                "iou": best_iou,
                "status": "FP",
            })

    unmatched_gt = [
        index
        for index in range(len(ground_truth))
        if index not in matched_gt
    ]

    return matches, unmatched_gt


# ======================================================================
# VISUALIZATION
# ======================================================================

def draw_box(
    image,
    box,
    color,
    label,
    thickness=2,
):
    x1, y1, x2, y2 = map(
        int,
        np.round(box),
    )

    x1 = max(0, x1)
    y1 = max(0, y1)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    text_y = max(20, y1 - 7)

    cv2.putText(
        image,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def create_error_audit(
    model,
    dataset_yaml,
    config,
    class_names,
    output_dir,
    confidence=0.25,
    max_images=30,
    iou_threshold=0.50,
):
    """
    Inspect actual validation photographs.

    Green:
        Correct detection / matched ground truth.

    Red:
        False-positive prediction.

    Orange:
        Ground-truth object missed by the model.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = discover_images(
        dataset_yaml,
        config,
        "val",
    )

    random.Random(SEED).shuffle(image_paths)

    image_paths = image_paths[:max_images]

    audit_rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    print("\n" + "=" * 72)
    print("REAL IMAGE DETECTION ERROR ANALYSIS")
    print("=" * 72)

    for image_index, image_path in enumerate(image_paths):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        label_path = image_to_label_path(image_path)

        ground_truth = load_yolo_labels(
            label_path,
            width,
            height,
        )

        results = model.predict(
            source=str(image_path),
            conf=confidence,
            iou=0.7,
            verbose=False,
        )

        predictions = []

        if results and results[0].boxes is not None:

            boxes = results[0].boxes

            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            scores = boxes.conf.cpu().numpy()

            for box, class_id, score in zip(
                xyxy,
                classes,
                scores,
            ):

                predictions.append({

                    "class_id": int(class_id),
                    "confidence": float(score),

                    "box_xyxy": box.astype(
                        np.float32
                    ),

                })

        matches, unmatched_gt = (
            match_predictions_to_ground_truth(
                predictions,
                ground_truth,
                iou_threshold,
            )
        )

        tp = sum(
            1
            for item in matches
            if item["status"] == "TP"
        )

        fp = sum(
            1
            for item in matches
            if item["status"] == "FP"
        )

        fn = len(unmatched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        # --------------------------------------------------------------
        # Draw predictions
        # --------------------------------------------------------------

        visualization = image.copy()

        for match in matches:

            prediction = match["prediction"]

            class_id = prediction["class_id"]

            class_name = class_names.get(
                class_id,
                str(class_id),
            )

            confidence_value = prediction["confidence"]

            if match["status"] == "TP":

                color = (0, 255, 0)

                label = (
                    f"TP {class_name} "
                    f"{confidence_value:.2f}"
                )

            else:

                color = (0, 0, 255)

                label = (
                    f"FP {class_name} "
                    f"{confidence_value:.2f}"
                )

            draw_box(
                visualization,
                prediction["box_xyxy"],
                color,
                label,
            )

        # --------------------------------------------------------------
        # Draw missed ground-truth objects
        # --------------------------------------------------------------

        for gt_index in unmatched_gt:

            gt = ground_truth[gt_index]

            class_name = class_names.get(
                gt["class_id"],
                str(gt["class_id"]),
            )

            draw_box(
                visualization,
                gt["box_xyxy"],
                (0, 165, 255),
                f"FN {class_name}",
            )

        # --------------------------------------------------------------
        # Add image summary
        # --------------------------------------------------------------

        summary = (
            f"TP: {tp}  FP: {fp}  FN: {fn}"
        )

        cv2.putText(
            visualization,
            summary,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        output_path = (
            output_dir
            / f"audit_{image_index:04d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            visualization,
        )

        audit_rows.append({

            "image": str(image_path),
            "ground_truth_objects": len(ground_truth),
            "predictions": len(predictions),

            "TP": tp,
            "FP": fp,
            "FN": fn,

        })

        print(
            f"[{image_index + 1}/{len(image_paths)}] "
            f"{image_path.name}: "
            f"TP={tp}, FP={fp}, FN={fn}"
        )

    precision, recall, f1 = precision_recall_f1(
        total_tp,
        total_fp,
        total_fn,
    )

    print("\nAGGREGATED AUDIT METRICS")
    print("------------------------")
    print("TP:", total_tp)
    print("FP:", total_fp)
    print("FN:", total_fn)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    # Save image-level analysis.

    csv_path = (
        output_dir.parent
        / "detection_error_analysis.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fields = [
            "image",
            "ground_truth_objects",
            "predictions",
            "TP",
            "FP",
            "FN",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(audit_rows)

    return {

        "images_audited": len(audit_rows),

        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,

        "precision": precision,
        "recall": recall,
        "f1": f1,

    }


# ======================================================================
# TRAINING CURVE VISUALIZATION
# ======================================================================

def visualize_training_results(results_csv):
    """
    Read Ultralytics results.csv and visualize important curves.
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    results_csv = Path(results_csv)

    if not results_csv.exists():
        print(
            "Training results.csv not found. "
            "Skipping custom training plots."
        )
        return

    dataframe = pd.read_csv(results_csv)

    dataframe.columns = [
        column.strip()
        for column in dataframe.columns
    ]

    metrics_to_plot = [
        column
        for column in dataframe.columns
        if (
            "loss" in column.lower()
            or "map50" in column.lower()
            or "map50-95" in column.lower()
            or "precision" in column.lower()
            or "recall" in column.lower()
        )
    ]

    if not metrics_to_plot:
        print("No recognized metric columns in results.csv.")
        return

    plt.figure(figsize=(13, 7))

    for column in metrics_to_plot:

        plt.plot(
            dataframe[column].values,
            label=column,
        )

    plt.title("YOLO Training and Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Metric / Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = (
        results_csv.parent
        / "lesson28_training_curves.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    print("Training curves saved:", output_path)

    plt.show()


# ======================================================================
# EVALUATION SUMMARY
# ======================================================================

def save_evaluation_summary(
    metrics,
    class_results,
    audit_results,
    output_path,
):
    box_metrics = metrics.box

    summary = {

        "map50": safe_float(box_metrics.map50),
        "map50_95": safe_float(box_metrics.map),

        "map75": safe_float(
            getattr(box_metrics, "map75", float("nan"))
        ),

        "per_class_metrics": class_results,
        "real_image_audit": audit_results,

    }

    output_path = Path(output_path)

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\nEvaluation summary saved:", output_path)


# ======================================================================
# NUMERICAL TESTS
# ======================================================================

def run_tests():
    print("\n" + "=" * 72)
    print("LESSON 28 — NUMERICAL TESTS")
    print("=" * 72)

    # Test 1: identical boxes have IoU 1.

    box_a = [10, 10, 50, 50]

    assert np.isclose(
        box_iou(box_a, box_a),
        1.0,
    )

    print("Test 1 passed: identical-box IoU.")

    # Test 2: non-overlapping boxes have IoU 0.

    box_b = [60, 60, 100, 100]

    assert np.isclose(
        box_iou(box_a, box_b),
        0.0,
    )

    print("Test 2 passed: non-overlapping IoU.")

    # Test 3: partial overlap.

    box_c = [30, 10, 70, 50]

    expected_iou = 20.0 * 40.0 / (
        40.0 * 40.0
        + 40.0 * 40.0
        - 20.0 * 40.0
    )

    assert np.isclose(
        box_iou(box_a, box_c),
        expected_iou,
    )

    print("Test 3 passed: partial-overlap IoU.")

    # Test 4: precision, recall, F1.

    precision, recall, f1 = precision_recall_f1(
        tp=8,
        fp=2,
        fn=2,
    )

    assert np.isclose(precision, 0.8)
    assert np.isclose(recall, 0.8)
    assert np.isclose(f1, 0.8)

    print("Test 4 passed: precision, recall, and F1.")

    # Test 5: all predictions correct.

    precision, recall, f1 = precision_recall_f1(
        tp=10,
        fp=0,
        fn=0,
    )

    assert np.isclose(precision, 1.0)
    assert np.isclose(recall, 1.0)
    assert np.isclose(f1, 1.0)

    print("Test 5 passed: perfect detection metrics.")

    # Test 6: no true positives.

    precision, recall, f1 = precision_recall_f1(
        tp=0,
        fp=5,
        fn=5,
    )

    assert np.isclose(precision, 0.0)
    assert np.isclose(recall, 0.0)
    assert np.isclose(f1, 0.0)

    print("Test 6 passed: zero-TP metrics.")

    # Test 7: prediction-to-ground-truth matching.

    ground_truth = [
        {
            "class_id": 0,
            "box_xyxy": np.array(
                [10, 10, 50, 50],
                dtype=np.float32,
            ),
        }
    ]

    predictions = [
        {
            "class_id": 0,
            "confidence": 0.95,
            "box_xyxy": np.array(
                [12, 12, 48, 48],
                dtype=np.float32,
            ),
        }
    ]

    matches, unmatched = match_predictions_to_ground_truth(
        predictions,
        ground_truth,
        0.5,
    )

    assert len(matches) == 1
    assert matches[0]["status"] == "TP"
    assert len(unmatched) == 0

    print("Test 7 passed: correct prediction matching.")

    # Test 8: wrong class is a false positive.

    wrong_class_predictions = [
        {
            "class_id": 1,
            "confidence": 0.95,
            "box_xyxy": np.array(
                [10, 10, 50, 50],
                dtype=np.float32,
            ),
        }
    ]

    matches, unmatched = match_predictions_to_ground_truth(
        wrong_class_predictions,
        ground_truth,
        0.5,
    )

    assert matches[0]["status"] == "FP"
    assert len(unmatched) == 1

    print("Test 8 passed: wrong-class prediction.")

    # Test 9: duplicate prediction.

    duplicate_predictions = [
        {
            "class_id": 0,
            "confidence": 0.95,
            "box_xyxy": np.array(
                [10, 10, 50, 50],
                dtype=np.float32,
            ),
        },
        {
            "class_id": 0,
            "confidence": 0.80,
            "box_xyxy": np.array(
                [11, 11, 49, 49],
                dtype=np.float32,
            ),
        },
    ]

    matches, unmatched = match_predictions_to_ground_truth(
        duplicate_predictions,
        ground_truth,
        0.5,
    )

    tp_count = sum(
        item["status"] == "TP"
        for item in matches
    )

    fp_count = sum(
        item["status"] == "FP"
        for item in matches
    )

    assert tp_count == 1
    assert fp_count == 1
    assert len(unmatched) == 0

    print("Test 9 passed: duplicate prediction handling.")

    # Test 10: augmentation configurations.

    conservative = get_augmentation_config(
        "conservative"
    )

    strong = get_augmentation_config(
        "strong"
    )

    assert strong["mosaic"] > conservative["mosaic"]
    assert strong["mixup"] > conservative["mixup"]

    print("Test 10 passed: augmentation configurations.")

    print("\nALL LESSON 28 TESTS PASSED.")


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Lesson 28: YOLO training improvements "
            "and real-world evaluation."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET / "data.yaml"),
    )

    parser.add_argument(
        "--project",
        type=str,
        default=str(DEFAULT_PROJECT),
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="strong",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional existing YOLO checkpoint.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Examples: 0, 0,1, or cpu.",
    )

    parser.add_argument(
        "--augmentation",
        choices=["conservative", "strong"],
        default="strong",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--max-audit-images",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the dataset without training.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
    )

    args = parser.parse_args()

    set_seed()

    print("=" * 72)
    print("LESSON 28 — YOLO TRAINING IMPROVEMENTS")
    print("=" * 72)

    print("Dataset:", args.dataset)
    print("Model:", args.model)
    print("Augmentation:", args.augmentation)
    print("Epochs:", args.epochs)
    print("Batch:", args.batch)
    print("Image size:", args.imgsz)

    # --------------------------------------------------------------
    # Run numerical tests
    # --------------------------------------------------------------

    run_tests()

    # --------------------------------------------------------------
    # Load dataset configuration
    # --------------------------------------------------------------

    dataset_yaml = Path(args.dataset).resolve()

    config = load_dataset_config(dataset_yaml)

    class_names = get_class_names(config)

    print("\nWaste categories:")

    for class_id, class_name in class_names.items():
        print(f"  {class_id}: {class_name}")

    # --------------------------------------------------------------
    # Validate real dataset
    # --------------------------------------------------------------

    validate_dataset(
        dataset_yaml,
        config,
        class_names,
    )

    if args.check_only:

        print("\nDataset check completed.")
        print("No model training was performed.")

        return

    # --------------------------------------------------------------
    # Load Ultralytics YOLO
    # --------------------------------------------------------------

    try:
        from ultralytics import YOLO

    except ImportError as error:

        raise RuntimeError(
            "Ultralytics is required.\n"
            "Install it with:\n"
            "pip install -U ultralytics"
        ) from error

    # --------------------------------------------------------------
    # Model initialization
    # --------------------------------------------------------------

    if args.skip_training:

        if args.weights is None:
            raise ValueError(
                "--skip-training requires --weights."
            )

        print("\nLoading existing checkpoint:")
        print(args.weights)

        model = YOLO(args.weights)

    elif args.weights is not None:

        print("\nLoading supplied checkpoint:")
        print(args.weights)

        model = YOLO(args.weights)

        print(
            "NOTE: This mode uses the supplied checkpoint. "
            "It may contain pretrained weights."
        )

    else:

        print("\nInitializing YOLO from architecture YAML.")

        print("Architecture:", args.model)
        print("Pretrained weights: DISABLED")

        model = YOLO(args.model)

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    if not args.skip_training:

        print("\n" + "=" * 72)
        print("YOLO TRAINING")
        print("=" * 72)

        training_config = build_training_config(args)

        print("\nTraining configuration:")

        for key, value in training_config.items():
            print(f"  {key}: {value}")

        start_time = time.time()

        if args.weights is None:

            model.train(
                data=str(dataset_yaml),
                pretrained=False,
                **training_config,
            )

        else:

            model.train(
                data=str(dataset_yaml),
                **training_config,
            )

        elapsed = time.time() - start_time

        print(
            f"\nTraining completed in "
            f"{elapsed / 60.0:.2f} minutes."
        )

        # Use the best checkpoint when available.

        experiment_dir = (
            Path(args.project)
            / args.experiment
        )

        best_checkpoint = (
            experiment_dir
            / "weights"
            / "best.pt"
        )

        if best_checkpoint.exists():

            print("\nLoading best checkpoint:")
            print(best_checkpoint)

            model = YOLO(str(best_checkpoint))

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    print("\n" + "=" * 72)
    print("FINAL VALIDATION")
    print("=" * 72)

    validation_start = time.time()

    metrics = model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        conf=0.001,
        iou=0.6,
        plots=not args.no_plots,
        verbose=True,
    )

    validation_time = time.time() - validation_start

    box_metrics = metrics.box

    print("\nOVERALL DETECTION PERFORMANCE")
    print("--------------------------------")

    print(f"mAP@0.5:       {box_metrics.map50:.4f}")
    print(f"mAP@0.5:0.95:  {box_metrics.map:.4f}")
    print(f"mAP@0.75:      {box_metrics.map75:.4f}")

    print(
        f"Validation time: "
        f"{validation_time:.2f} seconds"
    )

    # --------------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------------

    class_results = extract_class_metrics(
        metrics,
        class_names,
    )

    print_class_metrics(class_results)

    output_dir = (
        Path(args.project)
        / args.experiment
        / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_class_metrics(
        class_results,
        output_dir / "per_class_metrics.csv",
    )

    # --------------------------------------------------------------
    # Real-image error analysis
    # --------------------------------------------------------------

    audit_results = create_error_audit(
        model=model,
        dataset_yaml=dataset_yaml,
        config=config,
        class_names=class_names,
        output_dir=output_dir / "audit_images",
        confidence=args.confidence,
        max_images=args.max_audit_images,
    )

    # --------------------------------------------------------------
    # Save summary
    # --------------------------------------------------------------

    save_evaluation_summary(
        metrics=metrics,
        class_results=class_results,
        audit_results=audit_results,
        output_path=(
            output_dir
            / "evaluation_summary.json"
        ),
    )

    # --------------------------------------------------------------
    # Training curves
    # --------------------------------------------------------------

    if not args.skip_training and not args.no_plots:

        results_csv = (
            Path(args.project)
            / args.experiment
            / "results.csv"
        )

        visualize_training_results(
            results_csv
        )

    # --------------------------------------------------------------
    # Final report
    # --------------------------------------------------------------

    print("\n" + "=" * 72)
    print("LESSON 28 COMPLETED")
    print("=" * 72)

    print("Evaluation directory:")
    print(output_dir)

    print("\nImportant files:")

    print("  per_class_metrics.csv")
    print("  detection_error_analysis.csv")
    print("  evaluation_summary.json")
    print("  audit_images/")

    print("\nOpenArm V2 application:")
    print(
        "Use class-level recall and false-positive analysis "
        "to identify which waste categories need more "
        "demonstrations or better camera positioning."
    )

    print("\nNext lesson:")
    print("LESSON 29 — FASTER R-CNN OBJECT DETECTION FROM SCRATCH")


if __name__ == "__main__":
    main()