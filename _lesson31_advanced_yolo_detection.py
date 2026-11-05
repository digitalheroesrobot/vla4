"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 31 — ADVANCED YOLO OBJECT DETECTION
======================================================================

FILE
----
lesson31_advanced_yolo_detection.py

DATA
----
Real TACO waste photographs.

This lesson reuses the YOLO-format dataset prepared in Lesson 27.

Default dataset:
    ./TACO_yolo_dataset/data.yaml

MODEL
-----
YOLO26n architecture
Random initialization
No pretrained weights

EXPERIMENTS
-----------
1. Baseline YOLO
2. Advanced YOLO

Both experiments use the same training and validation datasets.

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1 — Understand the YOLO training pipeline.

STEP 2 — Understand multi-scale training.

    Small objects may appear at different pixel sizes.

    Multi-scale training randomly changes the input resolution
    during training.

    This helps the detector learn scale variation.

STEP 3 — Understand cosine learning-rate scheduling.

    Learning rate controls the size of parameter updates.

    Cosine scheduling gradually reduces the learning rate.

STEP 4 — Understand Mosaic augmentation.

    Multiple real training images are combined into one image.

    This increases scene diversity and object context.

STEP 5 — Understand MixUp augmentation.

    Two real images are blended.

    The model learns from objects appearing in mixed scenes.

STEP 6 — Understand closing Mosaic.

    Strong Mosaic augmentation is disabled near the end of training.

    This helps the model adapt to normal, unaugmented images.

STEP 7 — Understand class imbalance.

    Waste datasets often contain many examples of some classes
    and very few examples of others.

    Evaluate each class separately.

STEP 8 — Compare baseline and advanced YOLO.

======================================================================
MATHEMATICAL FOUNDATION
======================================================================

1. GRADIENT DESCENT

    theta_(t+1) = theta_t - learning_rate * gradient

2. COSINE LEARNING RATE

    lr(t) = lr_min
          + 0.5 * (lr_max - lr_min)
          * (1 + cos(pi * t / T))

    where:

        t = current training step or epoch
        T = total training duration

3. MIXUP

    image_mix = lambda * image_A
              + (1-lambda) * image_B

    label_mix = lambda * label_A
              + (1-lambda) * label_B

    Detection implementations handle the corresponding
    bounding-box annotations internally.

4. MULTI-SCALE TRAINING

    image_size = base_size * scale_factor

    Example:

        base size = 640
        scale range = 0.75 to 1.25

        possible sizes are approximately:
            480
            544
            640
            736
            800

    Actual sizes are rounded to model stride multiples.

5. PRECISION

                    TP
    Precision = -----------
                 TP + FP

6. RECALL

                 TP
    Recall = -----------
             TP + FN

7. F1 SCORE

                 2 * P * R
    F1 = -------------------------
              P + R

8. MEAN AVERAGE PRECISION

    mAP@0.5:
        Average precision at IoU threshold 0.5.

    mAP@0.5:0.95:
        Average precision averaged over IoU thresholds
        from 0.50 to 0.95.

======================================================================
EXPERIMENT CONFIGURATIONS
======================================================================

BASELINE
--------
    Multi-scale:       Disabled
    Cosine LR:         Disabled
    Mosaic:            Enabled
    MixUp:             Disabled
    Close Mosaic:      Default behavior
    Optimizer:         AdamW
    Pretrained:        False

ADVANCED
--------
    Multi-scale:       Enabled
    Cosine LR:         Enabled
    Mosaic:            Enabled
    MixUp:             Enabled
    Close Mosaic:      Enabled
    Optimizer:         AdamW
    Pretrained:        False

IMPORTANT:
    These are controlled experiments, not guaranteed improvements.

    The advanced experiment may perform worse on a small dataset.
    Compare validation metrics rather than assuming that more
    augmentation always produces better accuracy.

======================================================================
DATASET
======================================================================

Expected structure:

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
    |   +-- labels/
    |       +-- train/
    |       +-- val/
    |
    +-- lesson31_advanced_yolo_detection.py

======================================================================
INSTALLATION
======================================================================

    pip install -U ultralytics opencv-python matplotlib numpy pyyaml

======================================================================
RUN
======================================================================

Advanced experiment only:

    python lesson31_advanced_yolo_detection.py \
        --experiment advanced \
        --epochs 50

Compare baseline and advanced:

    python lesson31_advanced_yolo_detection.py \
        --experiment both \
        --epochs 50

Quick pipeline test:

    python lesson31_advanced_yolo_detection.py \
        --experiment advanced \
        --epochs 3 \
        --batch 4 \
        --imgsz 512

======================================================================
OUTPUT
======================================================================

    runs/lesson31/
        baseline/
            weights/
            results.csv
            results.png
            confusion_matrix.png

        advanced/
            weights/
            results.csv
            results.png
            confusion_matrix.png

        comparison.json
        comparison.png
        predictions/

======================================================================
OPENARM V2 APPLICATION
======================================================================

Real waste sorting requires detection under changing conditions:

    - Different object sizes
    - Different camera distances
    - Partial occlusion
    - Different lighting
    - Cluttered conveyor belts
    - Imbalanced waste categories

Advanced YOLO training techniques can improve robustness.

The final detector can provide:

    class
    confidence
    bounding box
    object center
    object size

These outputs can be passed to a robot localization and grasping
pipeline.

======================================================================
NEXT LESSON
======================================================================

LESSON 32 — ANCHOR-FREE OBJECT DETECTION
======================================================================
"""

from __future__ import annotations

import argparse
import json
import random
import time

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from ultralytics import YOLO


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

DEFAULT_DATA_YAML = Path(
    "./TACO_yolo_dataset/data.yaml"
)

DEFAULT_OUTPUT = Path(
    "./runs/lesson31"
)

DEFAULT_MODEL = "yolo26n.yaml"


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# DEVICE
# ======================================================================

def get_device(device_argument):

    if device_argument is not None:
        return device_argument

    if torch.cuda.is_available():
        return "0"

    return "cpu"


# ======================================================================
# DATASET VALIDATION
# ======================================================================

def validate_dataset(data_yaml):

    data_yaml = Path(data_yaml).resolve()

    if not data_yaml.exists():

        raise FileNotFoundError(
            f"Dataset YAML not found:\n{data_yaml}\n\n"
            "Prepare the YOLO dataset with Lesson 27 first."
        )

    with open(
        data_yaml,
        "r",
        encoding="utf-8",
    ) as file:

        data = yaml.safe_load(file)

    required = [
        "train",
        "val",
        "names",
    ]

    for key in required:

        if key not in data:

            raise ValueError(
                f"Missing '{key}' in dataset YAML."
            )

    if isinstance(data["names"], dict):

        names = [
            data["names"][key]
            for key in sorted(
                data["names"],
                key=lambda x: int(x),
            )
        ]

    else:

        names = list(data["names"])

    if len(names) == 0:

        raise ValueError(
            "No class names found in data.yaml."
        )

    # --------------------------------------------------------------
    # Resolve dataset paths.
    # --------------------------------------------------------------

    dataset_root = data.get(
        "path",
        str(data_yaml.parent),
    )

    dataset_root = Path(dataset_root)

    if not dataset_root.is_absolute():
        dataset_root = (
            data_yaml.parent / dataset_root
        ).resolve()

    def resolve_split(value):

        if isinstance(value, list):

            paths = []

            for item in value:

                path = Path(item)

                if not path.is_absolute():
                    path = dataset_root / path

                paths.append(path.resolve())

            return paths

        path = Path(value)

        if not path.is_absolute():
            path = dataset_root / path

        return [path.resolve()]

    train_paths = resolve_split(
        data["train"]
    )

    val_paths = resolve_split(
        data["val"]
    )

    for path in train_paths + val_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Dataset path does not exist:\n{path}"
            )

    print("\nDataset validation passed.")

    print("Dataset YAML:", data_yaml)
    print("Dataset root:", dataset_root)
    print("Classes:", names)

    print("\nTraining paths:")

    for path in train_paths:
        print(" ", path)

    print("\nValidation paths:")

    for path in val_paths:
        print(" ", path)

    return {
        "yaml": data_yaml,
        "root": dataset_root,
        "names": names,
        "train_paths": train_paths,
        "val_paths": val_paths,
    }


# ======================================================================
# DATASET STATISTICS
# ======================================================================

def count_images(path):

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    path = Path(path)

    if path.is_file():

        return int(
            path.suffix.lower() in extensions
        )

    if not path.exists():
        return 0

    return sum(
        1
        for file in path.rglob("*")
        if file.suffix.lower() in extensions
    )


def count_labels(path):

    path = Path(path)

    if not path.exists():
        return 0

    return len(
        list(path.rglob("*.txt"))
    )


def inspect_dataset(dataset_info):

    print("\n" + "=" * 72)
    print("DATASET STATISTICS")
    print("=" * 72)

    for split in ("train", "val"):

        paths = dataset_info[
            f"{split}_paths"
        ]

        total_images = sum(
            count_images(path)
            for path in paths
        )

        print(
            f"{split.upper()} images: {total_images}"
        )

    labels_root = (
        dataset_info["root"] / "labels"
    )

    if labels_root.exists():

        for split in ("train", "val"):

            label_dir = labels_root / split

            print(
                f"{split.upper()} label files: "
                f"{count_labels(label_dir)}"
            )


# ======================================================================
# CLASS DISTRIBUTION
# ======================================================================

def calculate_class_distribution(
    dataset_info,
):

    """
    Count YOLO labels by class ID.

    YOLO detection label format:

        class_id x_center y_center width height

    Coordinates are normalized to [0, 1].
    """

    labels_root = (
        dataset_info["root"] / "labels"
    )

    distribution = {}

    for split in ("train", "val"):

        split_dir = labels_root / split

        counts = {
            index: 0
            for index in range(
                len(dataset_info["names"])
            )
        }

        if split_dir.exists():

            for label_file in split_dir.rglob("*.txt"):

                with open(
                    label_file,
                    "r",
                    encoding="utf-8",
                ) as file:

                    for line in file:

                        parts = line.strip().split()

                        if len(parts) < 5:
                            continue

                        try:

                            class_id = int(parts[0])

                        except ValueError:
                            continue

                        if class_id in counts:
                            counts[class_id] += 1

        distribution[split] = counts

    return distribution


def plot_class_distribution(
    dataset_info,
    output_path,
):

    distribution = calculate_class_distribution(
        dataset_info
    )

    names = dataset_info["names"]

    x = np.arange(
        len(names)
    )

    train_counts = [
        distribution["train"][i]
        for i in range(len(names))
    ]

    val_counts = [
        distribution["val"][i]
        for i in range(len(names))
    ]

    width = 0.35

    plt.figure(
        figsize=(11, 5)
    )

    plt.bar(
        x - width / 2,
        train_counts,
        width,
        label="Train",
    )

    plt.bar(
        x + width / 2,
        val_counts,
        width,
        label="Validation",
    )

    plt.xticks(
        x,
        names,
        rotation=25,
        ha="right",
    )

    plt.ylabel("Number of annotated objects")
    plt.title("TACO Waste Class Distribution")

    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.show()

    return distribution


# ======================================================================
# EXPERIMENT CONFIGURATION
# ======================================================================

def baseline_configuration(args):

    return {

        "epochs": args.epochs,

        "imgsz": args.imgsz,

        "batch": args.batch,

        "optimizer": "AdamW",

        "lr0": 0.001,

        "lrf": 0.01,

        "weight_decay": 0.0005,

        "cos_lr": False,

        "multi_scale": 0.0,

        "mosaic": 1.0,

        "mixup": 0.0,

        "close_mosaic": 0,

        "warmup_epochs": 2.0,

        "degrees": 0.0,

        "translate": 0.1,

        "scale": 0.5,

        "fliplr": 0.5,

        "hsv_h": 0.015,

        "hsv_s": 0.7,

        "hsv_v": 0.4,

    }


def advanced_configuration(args):

    return {

        "epochs": args.epochs,

        "imgsz": args.imgsz,

        "batch": args.batch,

        "optimizer": "AdamW",

        "lr0": 0.001,

        "lrf": 0.01,

        "weight_decay": 0.0005,

        "cos_lr": True,

        "multi_scale": 0.25,

        "mosaic": 1.0,

        "mixup": 0.1,

        "close_mosaic": min(
            10,
            max(0, args.epochs - 1),
        ),

        "warmup_epochs": min(
            3.0,
            max(0.0, args.epochs / 3),
        ),

        "degrees": 5.0,

        "translate": 0.1,

        "scale": 0.5,

        "fliplr": 0.5,

        "hsv_h": 0.015,

        "hsv_s": 0.7,

        "hsv_v": 0.4,

    }


# ======================================================================
# TRAINING
# ======================================================================

def train_experiment(
    experiment_name,
    configuration,
    args,
    dataset_info,
    output_dir,
):

    print("\n" + "=" * 72)

    print(
        f"EXPERIMENT: {experiment_name.upper()}"
    )

    print("=" * 72)

    print("\nTraining configuration:")

    for key, value in configuration.items():

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------------
    # IMPORTANT:
    #
    # Every experiment constructs a fresh model from YAML.
    # No .pt checkpoint is loaded.
    # --------------------------------------------------------------

    model = YOLO(
        args.model
    )

    run_dir = (
        output_dir / experiment_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.time()

    # --------------------------------------------------------------
    # Train.
    # --------------------------------------------------------------

    results = model.train(

        data=str(dataset_info["yaml"]),

        project=str(output_dir),

        name=experiment_name,

        exist_ok=True,

        pretrained=False,

        seed=args.seed,

        deterministic=True,

        device=args.device,

        workers=args.workers,

        patience=args.patience,

        save=True,

        save_period=-1,

        plots=True,

        verbose=True,

        amp=args.amp,

        cache=args.cache,

        rect=False,

        # ----------------------------------------------------------
        # Training configuration.
        # ----------------------------------------------------------

        **configuration,

    )

    training_seconds = (
        time.time() - start_time
    )

    print(
        f"\nTraining time: "
        f"{training_seconds / 60:.2f} minutes"
    )

    # --------------------------------------------------------------
    # Locate best checkpoint.
    # --------------------------------------------------------------

    best_weights = (
        run_dir / "weights" / "best.pt"
    )

    if not best_weights.exists():

        raise FileNotFoundError(
            f"Training did not produce best.pt:\n"
            f"{best_weights}"
        )

    print(
        "Best checkpoint:",
        best_weights,
    )

    # --------------------------------------------------------------
    # Load trained checkpoint.
    # --------------------------------------------------------------

    trained_model = YOLO(
        str(best_weights)
    )

    # --------------------------------------------------------------
    # Validation.
    # --------------------------------------------------------------

    print(
        "\nEvaluating:",
        experiment_name,
    )

    validation_results = trained_model.val(

        data=str(dataset_info["yaml"]),

        split="val",

        imgsz=args.imgsz,

        batch=args.batch,

        device=args.device,

        workers=args.workers,

        plots=True,

        verbose=True,

        project=str(output_dir),

        name=f"{experiment_name}_validation",

        exist_ok=True,

    )

    box_metrics = validation_results.box

    metrics = {

        "experiment": experiment_name,

        "training_seconds": training_seconds,

        "precision": float(box_metrics.mp),

        "recall": float(box_metrics.mr),

        "map50": float(box_metrics.map50),

        "map50_95": float(box_metrics.map),

        "map75": float(
            box_metrics.map75
        ),

    }

    # --------------------------------------------------------------
    # Per-class AP.
    # --------------------------------------------------------------

    per_class_ap = {}

    ap_values = getattr(
        box_metrics,
        "ap",
        None,
    )

    if ap_values is not None:

        ap_values = np.asarray(
            ap_values
        ).reshape(-1)

        for index, value in enumerate(ap_values):

            if index < len(dataset_info["names"]):

                class_name = (
                    dataset_info["names"][index]
                )

                per_class_ap[class_name] = float(
                    value
                )

    metrics["per_class_ap50_95"] = per_class_ap

    print("\nValidation summary:")

    for key, value in metrics.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------------
    # Save metrics.
    # --------------------------------------------------------------

    with open(
        run_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    # --------------------------------------------------------------
    # Return trained model and metrics.
    # --------------------------------------------------------------

    return {
        "model": trained_model,
        "metrics": metrics,
        "best_weights": best_weights,
        "run_dir": run_dir,
    }


# ======================================================================
# PREDICTION VISUALIZATION
# ======================================================================

def generate_predictions(
    trained_model,
    dataset_info,
    output_dir,
    experiment_name,
    args,
):

    prediction_dir = (
        output_dir
        / "predictions"
        / experiment_name
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Use the validation image directory.
    source = [
        str(path)
        for path in dataset_info["val_paths"]
    ]

    print(
        "\nGenerating validation predictions:",
        experiment_name,
    )

    trained_model.predict(

        source=source,

        imgsz=args.imgsz,

        conf=args.confidence,

        iou=args.nms_iou,

        device=args.device,

        save=True,

        save_txt=True,

        save_conf=True,

        project=str(
            output_dir / "predictions"
        ),

        name=experiment_name,

        exist_ok=True,

        max_det=100,

        verbose=True,

    )

    print(
        "Prediction output:",
        prediction_dir,
    )


# ======================================================================
# COMPARISON
# ======================================================================

def compare_experiments(
    experiment_results,
    output_dir,
):

    comparison = {}

    for name, result in experiment_results.items():

        comparison[name] = result["metrics"]

    output_path = (
        output_dir / "comparison.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            comparison,
            file,
            indent=4,
        )

    print("\n" + "=" * 72)
    print("BASELINE VS ADVANCED YOLO")
    print("=" * 72)

    metric_names = [
        "precision",
        "recall",
        "map50",
        "map50_95",
        "map75",
    ]

    for metric_name in metric_names:

        print(
            f"\n{metric_name.upper()}"
        )

        for experiment_name, result in comparison.items():

            print(
                f"  {experiment_name:12s}: "
                f"{result[metric_name]:.4f}"
            )

    # --------------------------------------------------------------
    # Plot comparison.
    # --------------------------------------------------------------

    experiment_names = list(
        comparison.keys()
    )

    x = np.arange(
        len(metric_names)
    )

    width = 0.8 / max(
        len(experiment_names),
        1,
    )

    plt.figure(
        figsize=(11, 6)
    )

    for index, experiment_name in enumerate(
        experiment_names
    ):

        values = [
            comparison[experiment_name][metric]
            for metric in metric_names
        ]

        offset = (
            index - (len(experiment_names) - 1) / 2
        ) * width

        plt.bar(
            x + offset,
            values,
            width,
            label=experiment_name,
        )

    plt.xticks(
        x,
        [
            "Precision",
            "Recall",
            "mAP50",
            "mAP50-95",
            "mAP75",
        ],
    )

    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("YOLO Training Strategy Comparison")

    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    comparison_plot = (
        output_dir / "comparison.png"
    )

    plt.savefig(
        comparison_plot,
        dpi=150,
    )

    plt.show()

    print(
        "\nComparison saved:",
        comparison_plot,
    )

    return comparison


# ======================================================================
# TESTS
# ======================================================================

def run_tests():

    print("\n" + "=" * 72)
    print("LESSON 31 NUMERICAL TESTS")
    print("=" * 72)

    # --------------------------------------------------------------
    # Test 1: reproducibility.
    # --------------------------------------------------------------

    set_seed(42)

    a = torch.rand(5)

    set_seed(42)

    b = torch.rand(5)

    assert torch.allclose(
        a,
        b,
    )

    print("Test 1 passed: random seed reproducibility.")

    # --------------------------------------------------------------
    # Test 2: cosine learning rate.
    # --------------------------------------------------------------

    def cosine_lr(
        step,
        total_steps,
        lr_max,
        lr_min,
    ):

        progress = step / total_steps

        return (
            lr_min
            + 0.5
            * (lr_max - lr_min)
            * (
                1.0
                + np.cos(np.pi * progress)
            )
        )

    lr_start = cosine_lr(
        0,
        100,
        0.01,
        0.0001,
    )

    lr_end = cosine_lr(
        100,
        100,
        0.01,
        0.0001,
    )

    assert np.isclose(
        lr_start,
        0.01,
    )

    assert np.isclose(
        lr_end,
        0.0001,
    )

    print("Test 2 passed: cosine learning-rate endpoints.")

    # --------------------------------------------------------------
    # Test 3: learning rate decreases.
    # --------------------------------------------------------------

    rates = [
        cosine_lr(
            step,
            100,
            0.01,
            0.0001,
        )
        for step in range(101)
    ]

    assert rates[0] > rates[50] > rates[100]

    print("Test 3 passed: cosine learning-rate decay.")

    # --------------------------------------------------------------
    # Test 4: MixUp image equation.
    # --------------------------------------------------------------

    image_a = np.ones(
        (10, 10, 3),
        dtype=np.float32,
    )

    image_b = np.zeros(
        (10, 10, 3),
        dtype=np.float32,
    )

    lam = 0.7

    mixed = (
        lam * image_a
        + (1.0 - lam) * image_b
    )

    assert np.allclose(
        mixed,
        0.7,
    )

    print("Test 4 passed: MixUp image interpolation.")

    # --------------------------------------------------------------
    # Test 5: image-size scaling.
    # --------------------------------------------------------------

    base_size = 640

    low = int(
        base_size * 0.75
    )

    high = int(
        base_size * 1.25
    )

    assert low == 480
    assert high == 800

    print("Test 5 passed: multi-scale size range.")

    # --------------------------------------------------------------
    # Test 6: baseline configuration.
    # --------------------------------------------------------------

    class DummyArgs:

        epochs = 20
        imgsz = 640
        batch = 4

    baseline = baseline_configuration(
        DummyArgs()
    )

    assert baseline["multi_scale"] == 0.0
    assert baseline["cos_lr"] is False
    assert baseline["mixup"] == 0.0

    print("Test 6 passed: baseline configuration.")

    # --------------------------------------------------------------
    # Test 7: advanced configuration.
    # --------------------------------------------------------------

    advanced = advanced_configuration(
        DummyArgs()
    )

    assert advanced["multi_scale"] > 0
    assert advanced["cos_lr"] is True
    assert advanced["mixup"] > 0
    assert advanced["close_mosaic"] > 0

    print("Test 7 passed: advanced configuration.")

    # --------------------------------------------------------------
    # Test 8: precision and recall.
    # --------------------------------------------------------------

    tp = 8
    fp = 2
    fn = 2

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = (
        2 * precision * recall
        / (precision + recall)
    )

    assert np.isclose(precision, 0.8)
    assert np.isclose(recall, 0.8)
    assert np.isclose(f1, 0.8)

    print("Test 8 passed: detection metrics.")

    # --------------------------------------------------------------
    # Test 9: bounding box IoU.
    # --------------------------------------------------------------

    def box_iou_xyxy(a, b):

        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])

        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        intersection = (
            max(0, x2 - x1)
            * max(0, y2 - y1)
        )

        area_a = (
            (a[2] - a[0])
            * (a[3] - a[1])
        )

        area_b = (
            (b[2] - b[0])
            * (b[3] - b[1])
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        return intersection / union

    iou = box_iou_xyxy(
        [0, 0, 10, 10],
        [5, 0, 15, 10],
    )

    assert np.isclose(
        iou,
        1.0 / 3.0,
    )

    print("Test 9 passed: bounding-box IoU.")

    print("\nALL LESSON 31 TESTS PASSED.")


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Lesson 31: Advanced YOLO training "
            "on real TACO waste images."
        )
    )

    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_YAML),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=[
            "baseline",
            "advanced",
            "both",
        ],
        default="advanced",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--cache",
        action="store_true",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    args.device = get_device(
        args.device
    )

    args.amp = not args.no_amp

    set_seed(
        args.seed
    )

    output_dir = Path(
        args.output
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("LESSON 31 — ADVANCED YOLO OBJECT DETECTION")
    print("=" * 72)

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", args.device)
    print("Model:", args.model)
    print("Pretrained weights: DISABLED")

    # --------------------------------------------------------------
    # Tests.
    # --------------------------------------------------------------

    run_tests()

    # --------------------------------------------------------------
    # Validate model configuration.
    # --------------------------------------------------------------

    if not args.model.endswith(".yaml"):

        raise ValueError(
            "This lesson requires a YOLO model YAML file "
            "to ensure random initialization.\n"
            "Use --model yolo26n.yaml or another model YAML."
        )

    # --------------------------------------------------------------
    # Validate dataset.
    # --------------------------------------------------------------

    dataset_info = validate_dataset(
        args.data
    )

    inspect_dataset(
        dataset_info
    )

    plot_class_distribution(
        dataset_info,
        output_dir / "class_distribution.png",
    )

    # --------------------------------------------------------------
    # Select experiments.
    # --------------------------------------------------------------

    experiment_names = (
        ["baseline", "advanced"]
        if args.experiment == "both"
        else [args.experiment]
    )

    experiment_results = {}

    # --------------------------------------------------------------
    # Training.
    # --------------------------------------------------------------

    if not args.skip_training:

        for experiment_name in experiment_names:

            if experiment_name == "baseline":

                configuration = baseline_configuration(
                    args
                )

            else:

                configuration = advanced_configuration(
                    args
                )

            result = train_experiment(
                experiment_name=experiment_name,
                configuration=configuration,
                args=args,
                dataset_info=dataset_info,
                output_dir=output_dir,
            )

            experiment_results[
                experiment_name
            ] = result

            generate_predictions(
                trained_model=result["model"],
                dataset_info=dataset_info,
                output_dir=output_dir,
                experiment_name=experiment_name,
                args=args,
            )

    # --------------------------------------------------------------
    # Evaluate existing checkpoint.
    # --------------------------------------------------------------

    else:

        if args.weights is None:

            raise ValueError(
                "--skip-training requires --weights."
            )

        weights_path = Path(
            args.weights
        ).resolve()

        if not weights_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: {weights_path}"
            )

        model = YOLO(
            str(weights_path)
        )

        validation_results = model.val(

            data=str(dataset_info["yaml"]),

            split="val",

            imgsz=args.imgsz,

            batch=args.batch,

            device=args.device,

            plots=True,

        )

        box_metrics = validation_results.box

        metrics = {

            "precision": float(box_metrics.mp),

            "recall": float(box_metrics.mr),

            "map50": float(box_metrics.map50),

            "map50_95": float(box_metrics.map),

            "map75": float(box_metrics.map75),

        }

        print("\nEvaluation:")

        for key, value in metrics.items():

            print(
                f"{key}: {value:.4f}"
            )

        generate_predictions(
            trained_model=model,
            dataset_info=dataset_info,
            output_dir=output_dir,
            experiment_name="loaded_checkpoint",
            args=args,
        )

    # --------------------------------------------------------------
    # Compare experiments.
    # --------------------------------------------------------------

    if len(experiment_results) > 0:

        if len(experiment_results) == 2:

            compare_experiments(
                experiment_results,
                output_dir,
            )

        else:

            name = next(
                iter(experiment_results)
            )

            metrics = experiment_results[
                name
            ]["metrics"]

            with open(
                output_dir / "comparison.json",
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    {name: metrics},
                    file,
                    indent=4,
                )

    # --------------------------------------------------------------
    # Completion.
    # --------------------------------------------------------------

    print("\n" + "=" * 72)
    print("LESSON 31 COMPLETED")
    print("=" * 72)

    print("Output directory:", output_dir)

    print("\nImportant outputs:")

    print("  baseline/")
    print("  advanced/")
    print("  comparison.json")
    print("  comparison.png")
    print("  class_distribution.png")
    print("  predictions/")

    print("\nRemember:")

    print(
        "The advanced configuration is an experiment, "
        "not a guaranteed accuracy improvement."
    )

    print("\nNext lesson:")
    print("LESSON 32 — ANCHOR-FREE OBJECT DETECTION")


if __name__ == "__main__":
    main()