"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 27 — YOLO OBJECT DETECTION WITH REAL WASTE IMAGES
======================================================================

FILE:
    lesson27_yolo_object_detection.py

MODEL:
    Ultralytics YOLO26n

DATASET:
    TACO — Trash Annotations in Context

    Real photographs.
    Real waste objects.
    Human-created COCO annotations.

IMPORTANT:
    The YOLO architecture is implemented by the established
    Ultralytics framework.

    We are NOT implementing the complete YOLO architecture manually.

    We ARE learning the real training and inference workflow.

PRETRAINED WEIGHTS:
    DISABLED.

    The model is initialized from its YAML architecture configuration.
    Training starts with randomly initialized weights.

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1:
    Understand YOLO architecture configuration.

STEP 2:
    Convert COCO bounding boxes into YOLO labels.

STEP 3:
    Understand normalized bounding-box coordinates.

STEP 4:
    Prepare a real training and validation dataset.

STEP 5:
    Understand YOLO model initialization.

STEP 6:
    Train YOLO using real waste photographs.

STEP 7:
    Understand training losses.

STEP 8:
    Evaluate detection using:
        Precision
        Recall
        mAP@0.5
        mAP@0.5:0.95

STEP 9:
    Perform real image inference.

STEP 10:
    Visualize actual YOLO predictions.

======================================================================
WASTE CLASS CONFIGURATION
======================================================================

0: plastic
1: metal
2: paper_cardboard
3: glass
4: other

This is an initial broad-category mapping for robotics.

The mapping is heuristic, not a replacement for reviewing
the original TACO annotations.

Inspect the generated category_mapping.json before treating
these classes as final production labels.

======================================================================
TRAINING PIPELINE
======================================================================

TACO COCO DATASET
        |
        v
COCO JSON ANNOTATIONS
        |
        v
CATEGORY MAPPING
        |
        v
COCO TO YOLO FORMAT
        |
        v
TRAIN / VALIDATION SPLIT
        |
        v
YOLO26n RANDOM INITIALIZATION
        |
        v
TRAINING
        |
        v
VALIDATION
        |
        v
REAL IMAGE INFERENCE
        |
        v
WASTE DETECTION

======================================================================
INSTALLATION
======================================================================

pip install -U ultralytics opencv-python matplotlib numpy pyyaml

======================================================================
RUN
======================================================================

python lesson27_yolo_object_detection.py

Short experiment:

python lesson27_yolo_object_detection.py --max-images 300 --epochs 5

======================================================================
IMPORTANT
======================================================================

A short experiment is for verifying the complete pipeline.

It is NOT sufficient to establish production detection accuracy.

Training from random initialization generally requires substantially
more data and training time than fine-tuning pretrained weights.

======================================================================
"""

import argparse
import json
import random
import shutil

from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

from ultralytics import YOLO


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

DEFAULT_TACO_ROOT = "./TACO/data"

DEFAULT_OUTPUT_ROOT = "./TACO_yolo_dataset"

DEFAULT_MODEL = "yolo26n.yaml"

DEFAULT_EPOCHS = 30

DEFAULT_IMAGE_SIZE = 640

DEFAULT_BATCH_SIZE = 8

DEFAULT_WORKERS = 4

DEFAULT_CONFIDENCE = 0.25

DEFAULT_VALIDATION_RATIO = 0.20

CLASS_NAMES = [
    "plastic",
    "metal",
    "paper_cardboard",
    "glass",
    "other",
]

NUM_CLASSES = len(CLASS_NAMES)

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ======================================================================
# STEP 1 — RANDOM SEED
# ======================================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)


# ======================================================================
# STEP 2 — CATEGORY MAPPING
# ======================================================================

def map_taco_category(category_name):

    """
    Convert original TACO categories into broad waste classes.

    IMPORTANT:
        This is a heuristic mapping.

        Some ambiguous categories are assigned to "other".

        Inspect category_mapping.json and manually refine the
        mapping for the final robotics dataset.
    """

    name = category_name.lower()

    # Plastic.
    plastic_keywords = [
        "plastic",
        "polystyrene",
        "polyethylene",
        "polypropylene",
        "pet ",
        " pvc",
        "foam",
        "styrofoam",
    ]

    if any(
        keyword in name
        for keyword in plastic_keywords
    ):

        return "plastic"

    # Glass.
    glass_keywords = [
        "glass",
    ]

    if any(
        keyword in name
        for keyword in glass_keywords
    ):

        return "glass"

    # Metal.
    metal_keywords = [
        "aluminium",
        "aluminum",
        "metal",
        "steel",
        "tin",
        "drink can",
        "food can",
        "aerosol",
    ]

    if any(
        keyword in name
        for keyword in metal_keywords
    ):

        return "metal"

    # Paper and cardboard.
    paper_keywords = [
        "paper",
        "cardboard",
        "carton",
        "tissue",
        "newspaper",
        "magazine",
        "receipt",
        "paper bag",
    ]

    if any(
        keyword in name
        for keyword in paper_keywords
    ):

        return "paper_cardboard"

    return "other"


# ======================================================================
# STEP 3 — LOAD COCO DATASET
# ======================================================================

class TACODataset:

    def __init__(self, root):

        self.root = Path(root).resolve()

        self.annotation_file = (
            self.root / "annotations.json"
        )

        if not self.annotation_file.exists():

            raise FileNotFoundError(
                f"Missing annotation file:\n"
                f"{self.annotation_file}\n\n"
                "Download TACO before running this lesson."
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

        print("\nTACO DATASET")

        print(
            "Images:",
            len(self.images),
        )

        print(
            "Annotations:",
            len(self.data["annotations"]),
        )

        print(
            "Original categories:",
            len(self.categories),
        )

    def image_path(self, image_info):

        path = (
            self.root / image_info["file_name"]
        ).resolve()

        # Prevent accidentally reading paths outside the dataset.
        if not path.is_relative_to(self.root):

            raise ValueError(
                "Image path escapes the dataset directory."
            )

        return path

    def valid_image_ids(self):

        valid_ids = []

        for image_id, image_info in self.images.items():

            path = self.image_path(
                image_info
            )

            if not path.exists():
                continue

            image = cv2.imread(
                str(path)
            )

            if image is None:
                continue

            valid_ids.append(image_id)

        return valid_ids


# ======================================================================
# STEP 4 — COCO TO YOLO CONVERSION
# ======================================================================

def coco_bbox_to_yolo(
    bbox,
    image_width,
    image_height,
):
    """
    COCO:

        x_min, y_min, width, height

    YOLO:

        x_center, y_center, width, height

    All YOLO coordinates are normalized to [0, 1].
    """

    x, y, width, height = bbox

    x1 = max(
        0.0,
        float(x),
    )

    y1 = max(
        0.0,
        float(y),
    )

    x2 = min(
        float(image_width),
        float(x + width),
    )

    y2 = min(
        float(image_height),
        float(y + height),
    )

    box_width = x2 - x1

    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:

        return None

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2

    return np.array(
        [
            center_x / image_width,
            center_y / image_height,
            box_width / image_width,
            box_height / image_height,
        ],
        dtype=np.float32,
    )


# ======================================================================
# STEP 5 — DATASET PREPARATION
# ======================================================================

def prepare_output_directories(output_root):

    output_root = Path(output_root)

    marker = (
        output_root / ".lesson27_generated"
    )

    if output_root.exists():

        if not marker.exists():

            raise RuntimeError(
                f"Output directory already exists:\n"
                f"{output_root}\n\n"
                "Choose another --output-root directory "
                "to avoid overwriting unrelated files."
            )

        # Only remove directories created by this script.
        for name in [
            "images",
            "labels",
        ]:

            path = output_root / name

            if path.exists():

                shutil.rmtree(path)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    marker.touch()

    for split in [
        "train",
        "val",
    ]:

        (
            output_root
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            output_root
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    return output_root


def prepare_dataset(
    taco,
    output_root,
    max_images=0,
    validation_ratio=DEFAULT_VALIDATION_RATIO,
):
    """
    Build a YOLO-format dataset from real TACO photographs.

    The split is performed by image, not by annotation.

    Therefore, the same source image cannot appear in both
    training and validation sets.
    """

    output_root = prepare_output_directories(
        output_root
    )

    valid_ids = taco.valid_image_ids()

    print(
        "\nAvailable real images:",
        len(valid_ids),
    )

    if max_images > 0:

        valid_ids = valid_ids[:max_images]

    if len(valid_ids) < 2:

        raise RuntimeError(
            "At least two valid images are required."
        )

    rng = np.random.default_rng(
        SEED
    )

    valid_ids = np.asarray(
        valid_ids
    )

    rng.shuffle(
        valid_ids
    )

    split_index = int(
        len(valid_ids)
        * (1 - validation_ratio)
    )

    split_index = min(
        max(split_index, 1),
        len(valid_ids) - 1,
    )

    train_ids = valid_ids[:split_index]

    val_ids = valid_ids[split_index:]

    print(
        "Training images:",
        len(train_ids),
    )

    print(
        "Validation images:",
        len(val_ids),
    )

    category_mapping = {}

    for category_id, category_name in taco.categories.items():

        mapped_name = map_taco_category(
            category_name
        )

        category_mapping[str(category_id)] = {
            "original_name": category_name,
            "mapped_name": mapped_name,
            "mapped_id": CLASS_TO_ID[mapped_name],
        }

    annotation_counts = Counter()

    skipped_annotations = 0

    def process_split(
        image_ids,
        split,
    ):

        nonlocal skipped_annotations

        image_dir = (
            output_root
            / "images"
            / split
        )

        label_dir = (
            output_root
            / "labels"
            / split
        )

        copied_images = 0

        for image_id in image_ids:

            image_info = taco.images[
                int(image_id)
            ]

            source_path = taco.image_path(
                image_info
            )

            image = cv2.imread(
                str(source_path)
            )

            if image is None:
                continue

            image_height, image_width = image.shape[:2]

            image_filename = (
                f"image_{int(image_id):07d}_"
                f"{Path(image_info['file_name']).name}"
            )

            destination_image = (
                image_dir / image_filename
            )

            shutil.copy2(
                source_path,
                destination_image,
            )

            label_filename = (
                Path(image_filename).stem
                + ".txt"
            )

            label_path = (
                label_dir / label_filename
            )

            lines = []

            annotations = taco.annotations_by_image.get(
                int(image_id),
                [],
            )

            for annotation in annotations:

                # Crowd regions are not used as ordinary
                # individual object detection labels.
                if annotation.get("iscrowd", 0):
                    skipped_annotations += 1
                    continue

                category_id = annotation["category_id"]

                if str(category_id) not in category_mapping:

                    skipped_annotations += 1

                    continue

                mapped = category_mapping[
                    str(category_id)
                ]

                class_id = mapped["mapped_id"]

                yolo_box = coco_bbox_to_yolo(
                    annotation["bbox"],
                    image_width,
                    image_height,
                )

                if yolo_box is None:

                    skipped_annotations += 1

                    continue

                if not np.all(
                    np.isfinite(yolo_box)
                ):

                    skipped_annotations += 1

                    continue

                if np.any(yolo_box < 0) or np.any(yolo_box > 1):

                    skipped_annotations += 1

                    continue

                line = (
                    f"{class_id} "
                    f"{yolo_box[0]:.8f} "
                    f"{yolo_box[1]:.8f} "
                    f"{yolo_box[2]:.8f} "
                    f"{yolo_box[3]:.8f}"
                )

                lines.append(line)

                annotation_counts[
                    mapped["mapped_name"]
                ] += 1

            # Empty label files are valid for images
            # without usable object annotations.
            label_path.write_text(
                "\n".join(lines),
                encoding="utf-8",
            )

            copied_images += 1

        return copied_images

    train_count = process_split(
        train_ids,
        "train",
    )

    val_count = process_split(
        val_ids,
        "val",
    )

    # Dataset YAML.
    yaml_data = {
        "path": str(
            output_root.resolve()
        ),
        "train": "images/train",
        "val": "images/val",
        "names": {
            index: name
            for index, name in enumerate(CLASS_NAMES)
        },
    }

    yaml_path = (
        output_root / "data.yaml"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8",
    ) as file:

        yaml.safe_dump(
            yaml_data,
            file,
            sort_keys=False,
        )

    # Preserve original-to-target category mapping.
    mapping_path = (
        output_root / "category_mapping.json"
    )

    with open(
        mapping_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            category_mapping,
            file,
            indent=4,
        )

    print()
    print("=" * 70)
    print("YOLO DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print(
        "Training images copied:",
        train_count,
    )

    print(
        "Validation images copied:",
        val_count,
    )

    print(
        "Skipped annotations:",
        skipped_annotations,
    )

    print("\nMapped object counts:")

    for name in CLASS_NAMES:

        print(
            f"{name:<25} "
            f"{annotation_counts[name]}"
        )

    print(
        "\nDataset YAML:",
        yaml_path,
    )

    return yaml_path


# ======================================================================
# STEP 6 — DATASET VALIDATION
# ======================================================================

def validate_yolo_dataset(dataset_root):

    root = Path(dataset_root)

    print()
    print("=" * 70)
    print("VALIDATING YOLO LABELS")
    print("=" * 70)

    total_images = 0

    total_labels = 0

    for split in [
        "train",
        "val",
    ]:

        image_dir = (
            root / "images" / split
        )

        label_dir = (
            root / "labels" / split
        )

        image_files = []

        for extension in [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.bmp",
        ]:

            image_files.extend(
                image_dir.glob(extension)
            )

        total_images += len(
            image_files
        )

        for image_path in image_files:

            label_path = (
                label_dir
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():

                raise FileNotFoundError(
                    f"Missing label file: {label_path}"
                )

            content = label_path.read_text(
                encoding="utf-8"
            ).strip()

            if not content:
                continue

            for line in content.splitlines():

                values = line.split()

                if len(values) != 5:

                    raise ValueError(
                        f"Invalid YOLO label:\n{line}"
                    )

                class_id = int(values[0])

                box = np.asarray(
                    values[1:],
                    dtype=np.float32,
                )

                if not 0 <= class_id < NUM_CLASSES:

                    raise ValueError(
                        "Invalid class ID."
                    )

                if not np.all(
                    np.isfinite(box)
                ):

                    raise ValueError(
                        "Non-finite YOLO coordinate."
                    )

                if np.any(box < 0) or np.any(box > 1):

                    raise ValueError(
                        "YOLO coordinates must be normalized."
                    )

                if box[2] <= 0 or box[3] <= 0:

                    raise ValueError(
                        "Invalid bounding box size."
                    )

                total_labels += 1

        print(
            f"{split}: {len(image_files)} images"
        )

    print(
        "Total images:",
        total_images,
    )

    print(
        "Total object labels:",
        total_labels,
    )

    print("YOLO dataset validation passed.")

    return total_images, total_labels


# ======================================================================
# STEP 7 — VISUALIZE YOLO DATASET LABELS
# ======================================================================

def visualize_dataset_samples(
    dataset_root,
    num_samples=6,
):

    root = Path(dataset_root)

    image_dir = (
        root / "images" / "train"
    )

    label_dir = (
        root / "labels" / "train"
    )

    images = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
    ]:

        images.extend(
            image_dir.glob(extension)
        )

    if not images:

        print(
            "No training images available for visualization."
        )

        return

    rng = np.random.default_rng(
        SEED
    )

    selected = rng.choice(
        images,
        size=min(num_samples, len(images)),
        replace=False,
    )

    columns = 2

    rows = int(
        np.ceil(len(selected) / columns)
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 6 * rows),
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for axis, image_path in zip(
        axes,
        selected,
    ):

        image = cv2.imread(
            str(image_path)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        height, width = image.shape[:2]

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        if label_path.exists():

            content = label_path.read_text(
                encoding="utf-8"
            ).strip()

            if content:

                for line in content.splitlines():

                    class_id, cx, cy, bw, bh = map(
                        float,
                        line.split(),
                    )

                    x1 = int(
                        (cx - bw / 2) * width
                    )

                    y1 = int(
                        (cy - bh / 2) * height
                    )

                    x2 = int(
                        (cx + bw / 2) * width
                    )

                    y2 = int(
                        (cy + bh / 2) * height
                    )

                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    label = CLASS_NAMES[
                        int(class_id)
                    ]

                    cv2.putText(
                        image,
                        label,
                        (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

        axis.imshow(image)

        axis.set_title(
            image_path.name
        )

        axis.axis("off")

    for axis in axes[len(selected):]:

        axis.axis("off")

    plt.suptitle(
        "Real TACO Images — YOLO Ground Truth Labels",
        fontsize=16,
    )

    plt.tight_layout()

    plt.show()


# ======================================================================
# STEP 8 — YOLO MODEL INITIALIZATION
# ======================================================================

def initialize_yolo_model(model_config):

    print()
    print("=" * 70)
    print("INITIALIZING YOLO")
    print("=" * 70)

    print(
        "Model configuration:",
        model_config,
    )

    print(
        "Pretrained weights: DISABLED"
    )

    # Construct the architecture from YAML.
    # No pretrained .pt file is loaded.
    model = YOLO(
        model_config
    )

    print(
        "\nYOLO model initialized from configuration."
    )

    return model


# ======================================================================
# STEP 9 — TRAIN YOLO
# ======================================================================

def train_yolo(
    model,
    data_yaml,
    epochs,
    image_size,
    batch_size,
    workers,
    device,
):

    print()
    print("=" * 70)
    print("YOLO TRAINING")
    print("=" * 70)

    print(
        "Epochs:",
        epochs,
    )

    print(
        "Image size:",
        image_size,
    )

    print(
        "Batch size:",
        batch_size,
    )

    print(
        "Device:",
        device if device else "Auto",
    )

    print(
        "Initialization: Random weights"
    )

    results = model.train(

        data=str(data_yaml),

        epochs=epochs,

        imgsz=image_size,

        batch=batch_size,

        workers=workers,

        device=device if device else None,

        pretrained=False,

        seed=SEED,

        deterministic=True,

        patience=max(
            10,
            epochs // 3,
        ),

        optimizer="auto",

        project="runs/lesson27",

        name="taco_yolo26n",

        exist_ok=True,

        plots=True,

        save=True,

        verbose=True,
    )

    print()
    print(
        "YOLO training completed."
    )

    return results


# ======================================================================
# STEP 10 — VALIDATION
# ======================================================================

def evaluate_yolo(
    model,
    data_yaml,
    image_size,
    device,
):

    print()
    print("=" * 70)
    print("YOLO VALIDATION")
    print("=" * 70)

    metrics = model.val(

        data=str(data_yaml),

        imgsz=image_size,

        device=device if device else None,

        plots=True,

        verbose=True,
    )

    print()
    print("VALIDATION METRICS")

    print(
        "mAP@0.5:",
        float(metrics.box.map50),
    )

    print(
        "mAP@0.5:0.95:",
        float(metrics.box.map),
    )

    print(
        "mAP@0.75:",
        float(metrics.box.map75),
    )

    print(
        "Mean precision:",
        float(metrics.box.mp),
    )

    print(
        "Mean recall:",
        float(metrics.box.mr),
    )

    return metrics


# ======================================================================
# STEP 11 — REAL IMAGE INFERENCE
# ======================================================================

def run_inference(
    model,
    dataset_root,
    confidence,
    image_size,
    device,
    max_images=8,
):

    print()
    print("=" * 70)
    print("REAL IMAGE INFERENCE")
    print("=" * 70)

    image_dir = (
        Path(dataset_root)
        / "images"
        / "val"
    )

    image_paths = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
    ]:

        image_paths.extend(
            image_dir.glob(extension)
        )

    if not image_paths:

        print(
            "No validation images found."
        )

        return

    image_paths = image_paths[:max_images]

    results = model.predict(

        source=[
            str(path)
            for path in image_paths
        ],

        conf=confidence,

        imgsz=image_size,

        device=device if device else None,

        save=True,

        project="runs/lesson27",

        name="real_predictions",

        exist_ok=True,

        verbose=True,
    )

    print()
    print(
        "Images processed:",
        len(results),
    )

    for index, result in enumerate(results):

        boxes = result.boxes

        print()
        print(
            f"Image {index + 1}:"
        )

        print(
            "Detections:",
            len(boxes),
        )

        for box in boxes:

            class_id = int(
                box.cls.item()
            )

            confidence_value = float(
                box.conf.item()
            )

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            print(
                f"  {CLASS_NAMES[class_id]} "
                f"confidence={confidence_value:.3f} "
                f"box={coordinates.round(1)}"
            )

    # Display actual model prediction images.
    num_results = len(results)

    columns = 2

    rows = int(
        np.ceil(num_results / columns)
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 6 * rows),
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for axis, result in zip(
        axes,
        results,
    ):

        # result.plot() returns a BGR image.
        visual = result.plot()

        visual = cv2.cvtColor(
            visual,
            cv2.COLOR_BGR2RGB,
        )

        axis.imshow(visual)

        axis.set_title(
            Path(result.path).name
        )

        axis.axis("off")

    for axis in axes[num_results:]:

        axis.axis("off")

    plt.suptitle(
        "YOLO26n — Actual Predictions on Real TACO Images",
        fontsize=16,
    )

    plt.tight_layout()

    plt.show()

    print()
    print(
        "Prediction images saved under:"
    )

    print(
        "runs/lesson27/real_predictions/"
    )


# ======================================================================
# STEP 12 — TRAINING CURVE VISUALIZATION
# ======================================================================

def visualize_training_results():

    results_file = (
        Path("runs/lesson27/taco_yolo26n/results.png")
    )

    if not results_file.exists():

        print(
            "Training results image not found."
        )

        return

    image = cv2.imread(
        str(results_file)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    plt.figure(
        figsize=(16, 10)
    )

    plt.imshow(image)

    plt.title(
        "YOLO Training Results"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()


# ======================================================================
# STEP 13 — MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--taco-root",
        type=str,
        default=DEFAULT_TACO_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help=(
            "Optional trained YOLO .pt checkpoint "
            "for validation/inference when using --skip-training."
        ),
    )

    args = parser.parse_args()

    set_seed()

    print("=" * 70)

    print(
        "COMPUTER VISION — LESSON 27"
    )

    print(
        "YOLO OBJECT DETECTION"
    )

    print("=" * 70)

    print(
        "Model:",
        args.model,
    )

    print(
        "Pretrained weights:",
        "Disabled by default",
    )

    # --------------------------------------------------------------
    # Load TACO.
    # --------------------------------------------------------------

    taco = TACODataset(
        args.taco_root
    )

    # --------------------------------------------------------------
    # Prepare YOLO dataset.
    # --------------------------------------------------------------

    data_yaml = prepare_dataset(
        taco,
        args.output_root,
        max_images=args.max_images,
    )

    # --------------------------------------------------------------
    # Verify converted annotations.
    # --------------------------------------------------------------

    validate_yolo_dataset(
        args.output_root
    )

    # --------------------------------------------------------------
    # Visualize real training images and annotations.
    # --------------------------------------------------------------

    visualize_dataset_samples(
        args.output_root
    )

    if args.prepare_only:

        print(
            "\nDataset preparation finished. Training skipped."
        )

        return

    # --------------------------------------------------------------
    # Initialize model.
    # --------------------------------------------------------------

    if args.skip_training:

        if not args.weights:

            raise ValueError(
                "--skip-training requires --weights."
            )

        model = YOLO(
            args.weights
        )

    else:

        model = initialize_yolo_model(
            args.model
        )

        # ----------------------------------------------------------
        # Train from random initialization.
        # ----------------------------------------------------------

        train_yolo(
            model,
            data_yaml,
            args.epochs,
            args.imgsz,
            args.batch,
            args.workers,
            args.device,
        )

    # --------------------------------------------------------------
    # Validation.
    # --------------------------------------------------------------

    evaluate_yolo(
        model,
        data_yaml,
        args.imgsz,
        args.device,
    )

    # --------------------------------------------------------------
    # Actual image inference.
    # --------------------------------------------------------------

    run_inference(
        model,
        args.output_root,
        args.confidence,
        args.imgsz,
        args.device,
    )

    # --------------------------------------------------------------
    # Training curves.
    # --------------------------------------------------------------

    if not args.skip_training:

        visualize_training_results()

    print()
    print("=" * 70)

    print(
        "LESSON 27 COMPLETED"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()