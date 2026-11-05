"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 29 — FASTER R-CNN OBJECT DETECTION
======================================================================

FILE
----
lesson29_faster_rcnn_object_detection.py

COURSE RULE
-----------
Use REAL images from the TACO waste dataset.

No:
    - Synthetic training images
    - Pretrained detection weights
    - Pretrained backbone weights
    - External pretrained foundation models

We use torchvision's Faster R-CNN implementation to study and train
a standard two-stage object detector.

The model is initialized with random weights.

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1 — Understand two-stage object detection.

    Faster R-CNN consists of:

        Input image
             |
             v
        Backbone CNN
             |
             v
        Feature Pyramid Network (FPN)
             |
             v
        Region Proposal Network (RPN)
             |
             v
        Region of Interest (RoI) Align
             |
             v
        Classification Head
             |
             v
        Bounding Box Regression
             |
             v
        Final detections

STEP 2 — Understand the Region Proposal Network.

    The RPN predicts:

        1. Objectness score
        2. Bounding-box offsets

    Objectness:
        Is there an object in this region?

STEP 3 — Understand anchor boxes.

    Anchor boxes are predefined reference boxes at different
    locations, scales, and aspect ratios.

STEP 4 — Understand RoI Align.

    RoI Align extracts fixed-size feature representations from
    variable-sized candidate regions.

STEP 5 — Train Faster R-CNN on real TACO images.

    Classes:
        plastic
        metal
        paper_cardboard
        glass
        other

    Actual names are loaded from data.yaml.

STEP 6 — Evaluate the detector.

    Metrics:
        - mAP@0.5
        - mAP@0.5:0.95
        - Precision
        - Recall
        - Per-class AP

STEP 7 — Visualize predictions.

STEP 8 — Compare Faster R-CNN with YOLO.

======================================================================
MATHEMATICAL FOUNDATION
======================================================================

1. ANCHOR BOX

    An anchor is a predefined reference box.

    The detector predicts offsets:

        tx, ty, tw, th

    Given anchor:

        xa, ya, wa, ha

    Predicted box:

        x = xa + tx * wa
        y = ya + ty * ha

        w = wa * exp(tw)
        h = ha * exp(th)

2. REGION PROPOSAL NETWORK

    RPN predicts:

        Objectness probability
        Bounding-box regression offsets

    RPN loss:

        L_RPN = L_objectness + L_box

3. CLASSIFICATION LOSS

    Cross entropy:

        L_cls = -sum(y_i * log(p_i))

4. BOUNDING BOX REGRESSION LOSS

    Smooth L1:

        SmoothL1(x) =
            0.5*x^2                 if |x| < 1
            |x| - 0.5               otherwise

5. TOTAL FASTER R-CNN LOSS

        L_total =
            L_rpn_objectness
          + L_rpn_box
          + L_roi_classifier
          + L_roi_box

6. INTERSECTION OVER UNION

        IoU = Area(A intersection B) /
              Area(A union B)

7. PRECISION

        Precision = TP / (TP + FP)

8. RECALL

        Recall = TP / (TP + FN)

9. AVERAGE PRECISION

    AP is the area under the precision-recall curve.

10. MEAN AVERAGE PRECISION

        mAP = mean(AP_class)

======================================================================
DATASET
======================================================================

Expected directory:

    vla4/
    |
    +-- TACO/
    |   +-- data/
    |
    +-- TACO_yolo_dataset/
    |   +-- data.yaml
    |   +-- images/
    |       +-- train/
    |       +-- val/
    |
    +-- lesson29_faster_rcnn_object_detection.py

The dataset was prepared in Lesson 27.

Labels use YOLO format:

    class_id center_x center_y width height

This lesson converts them into:

    boxes:
        [x1, y1, x2, y2]

    labels:
        [class_id + 1]

IMPORTANT:
    Faster R-CNN reserves label 0 for background.

    Therefore:
        YOLO class 0 -> Faster R-CNN class 1
        YOLO class 1 -> Faster R-CNN class 2
        ...

======================================================================
INSTALLATION
======================================================================

    pip install -U torch torchvision opencv-python matplotlib numpy pyyaml

For CUDA, install the PyTorch build matching your CUDA environment
using the official PyTorch installation selector.

======================================================================
RUN
======================================================================

Dataset check:

    python lesson29_faster_rcnn_object_detection.py --check-only

Short training:

    python lesson29_faster_rcnn_object_detection.py \
        --epochs 3 \
        --batch 2

Full training:

    python lesson29_faster_rcnn_object_detection.py \
        --epochs 50 \
        --batch 2

GPU:

    python lesson29_faster_rcnn_object_detection.py \
        --epochs 50 \
        --batch 2 \
        --device cuda

Resume from a saved checkpoint:

    python lesson29_faster_rcnn_object_detection.py \
        --epochs 20 \
        --batch 2 \
        --resume runs/lesson29/best_model.pth

======================================================================
OUTPUT
======================================================================

    runs/lesson29/
        best_model.pth
        last_model.pth
        training_history.json
        training_curves.png
        evaluation.json
        predictions/
        checkpoints/

======================================================================
REAL-WORLD ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    RGB Camera
         |
         v
    Faster R-CNN
         |
         v
    Waste Category
         |
         v
    Bounding Box
         |
         v
    Pixel Coordinate
         |
         v
    Camera Calibration
         |
         v
    Robot Coordinate
         |
         v
    ACT Manipulation Policy
         |
         v
    Pick and Sort

A bounding box alone is NOT a robot grasp pose.

You still need:
    - Camera calibration
    - Depth estimation or geometric assumptions
    - Camera-to-robot extrinsic calibration
    - Grasp point estimation
    - Robot motion planning

======================================================================
NEXT LESSON
======================================================================

LESSON 30 — INSTANCE SEGMENTATION

We will move from bounding boxes to pixel-level object masks.
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
import yaml

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
)
from torchvision.ops import box_iou

import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

DEFAULT_DATASET = Path("./TACO_yolo_dataset/data.yaml")

DEFAULT_OUTPUT = Path("./runs/lesson29")

BACKGROUND_CLASS_ID = 0

DEFAULT_IMAGE_SIZE = 640


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
# BASIC MATHEMATICS
# ======================================================================

def calculate_iou(box_a, box_b):

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


def smooth_l1_loss_numpy(prediction, target, beta=1.0):

    difference = np.abs(
        np.asarray(prediction) - np.asarray(target)
    )

    loss = np.where(
        difference < beta,
        0.5 * difference ** 2 / beta,
        difference - 0.5 * beta,
    )

    return float(np.mean(loss))


def calculate_precision_recall(tp, fp, fn):

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

def read_yolo_labels(
    label_path,
    image_width,
    image_height,
):

    label_path = Path(label_path)

    boxes = []
    labels = []

    if not label_path.exists():
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    with open(label_path, "r", encoding="utf-8") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(float(values[0]))

            cx, cy, width, height = map(
                float,
                values[1:],
            )

            x1 = (cx - width / 2.0) * image_width
            y1 = (cy - height / 2.0) * image_height

            x2 = (cx + width / 2.0) * image_width
            y2 = (cy + height / 2.0) * image_height

            x1 = np.clip(x1, 0, image_width - 1)
            y1 = np.clip(y1, 0, image_height - 1)

            x2 = np.clip(x2, 0, image_width - 1)
            y2 = np.clip(y2, 0, image_height - 1)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([
                x1,
                y1,
                x2,
                y2,
            ])

            # Class zero is reserved for background.
            labels.append(class_id + 1)

    if len(boxes) == 0:

        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    return (
        torch.tensor(
            boxes,
            dtype=torch.float32,
        ),

        torch.tensor(
            labels,
            dtype=torch.int64,
        ),
    )


# ======================================================================
# DATASET CONFIGURATION
# ======================================================================

def load_dataset_config(dataset_yaml):

    dataset_yaml = Path(dataset_yaml).resolve()

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Cannot find dataset configuration: {dataset_yaml}"
        )

    with open(dataset_yaml, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid data.yaml.")

    if "train" not in config or "val" not in config:
        raise ValueError(
            "data.yaml must contain train and val."
        )

    if "names" not in config:
        raise ValueError(
            "data.yaml must contain class names."
        )

    dataset_root = Path(
        config.get("path", dataset_yaml.parent)
    )

    if not dataset_root.is_absolute():
        dataset_root = (
            dataset_yaml.parent / dataset_root
        ).resolve()

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


def get_split_directory(config, split):

    root = Path(config["path"])

    split_path = Path(config[split])

    if split_path.is_absolute():
        return split_path

    return root / split_path


def image_to_label_path(image_path):

    parts = list(Path(image_path).parts)

    try:
        index = parts.index("images")
        parts[index] = "labels"

    except ValueError:

        raise ValueError(
            f"Image path must contain an 'images' directory: "
            f"{image_path}"
        )

    return Path(*parts).with_suffix(".txt")


# ======================================================================
# REAL TACO DATASET
# ======================================================================

class TacoDetectionDataset(Dataset):

    def __init__(
        self,
        dataset_yaml,
        split="train",
        image_size=640,
        augment=False,
        max_images=None,
    ):

        self.dataset_yaml = Path(dataset_yaml).resolve()

        self.config = load_dataset_config(
            self.dataset_yaml
        )

        self.class_names = get_class_names(
            self.config
        )

        self.split = split
        self.image_size = image_size
        self.augment = augment

        image_directory = get_split_directory(
            self.config,
            split,
        )

        if not image_directory.exists():
            raise FileNotFoundError(
                f"Image directory not found: {image_directory}"
            )

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        self.image_paths = sorted([
            path
            for path in image_directory.rglob("*")
            if path.suffix.lower() in extensions
        ])

        if max_images is not None:

            random.Random(SEED).shuffle(
                self.image_paths
            )

            self.image_paths = self.image_paths[
                :max_images
            ]

        if not self.image_paths:
            raise RuntimeError(
                f"No images found in {image_directory}"
            )

        print(
            f"{split.upper()} dataset: "
            f"{len(self.image_paths)} real images"
        )

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Failed to read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        original_height, original_width = image.shape[:2]

        label_path = image_to_label_path(
            image_path
        )

        boxes, labels = read_yolo_labels(
            label_path,
            original_width,
            original_height,
        )

        # --------------------------------------------------------------
        # Resize image and bounding boxes.
        # --------------------------------------------------------------

        scale_x = self.image_size / original_width
        scale_y = self.image_size / original_height

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        if len(boxes) > 0:

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        # --------------------------------------------------------------
        # Simple horizontal-flip augmentation.
        # --------------------------------------------------------------

        if self.augment and random.random() < 0.5:

            image = np.ascontiguousarray(
                image[:, ::-1]
            )

            if len(boxes) > 0:

                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()

                boxes[:, 0] = self.image_size - old_x2
                boxes[:, 2] = self.image_size - old_x1

        # --------------------------------------------------------------
        # Convert RGB image to PyTorch tensor.
        # --------------------------------------------------------------

        image_tensor = torch.from_numpy(
            image.copy()
        ).permute(2, 0, 1).float() / 255.0

        # --------------------------------------------------------------
        # Faster R-CNN target format.
        # --------------------------------------------------------------

        areas = (
            (boxes[:, 2] - boxes[:, 0])
            * (boxes[:, 3] - boxes[:, 1])
        )

        target = {

            "boxes": boxes,

            "labels": labels,

            "image_id": torch.tensor(
                [index],
                dtype=torch.int64,
            ),

            "area": areas,

            "iscrowd": torch.zeros(
                (len(labels),),
                dtype=torch.int64,
            ),

        }

        return image_tensor, target


# ======================================================================
# DATA LOADER
# ======================================================================

def detection_collate_fn(batch):

    images = []
    targets = []

    for image, target in batch:

        images.append(image)
        targets.append(target)

    return images, targets


def create_data_loaders(args):

    train_dataset = TacoDetectionDataset(
        dataset_yaml=args.dataset,
        split="train",
        image_size=args.imgsz,
        augment=True,
        max_images=args.max_train_images,
    )

    val_dataset = TacoDetectionDataset(
        dataset_yaml=args.dataset,
        split="val",
        image_size=args.imgsz,
        augment=False,
        max_images=args.max_val_images,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=detection_collate_fn,
        pin_memory=args.device == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=detection_collate_fn,
        pin_memory=args.device == "cuda",
    )

    return (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    )


# ======================================================================
# FASTER R-CNN MODEL
# ======================================================================

def create_faster_rcnn(
    num_classes,
    min_size=640,
    max_size=640,
):

    """
    Create Faster R-CNN with ResNet-50 FPN.

    num_classes includes background.

    Example:
        5 waste categories + background = 6 classes.

    pretrained=False:
        Do not load pretrained detector weights.

    weights_backbone=None:
        Do not load pretrained ResNet backbone weights.
    """

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
        trainable_backbone_layers=5,
    )

    # Replace classification and regression prediction head.

    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )

    # RPN anchor configuration.
    #
    # Different sizes and aspect ratios help detect objects
    # with different dimensions.

    model.rpn.anchor_generator.sizes = (
        (16, 32, 64, 128, 256),
    ) * 5

    model.rpn.anchor_generator.aspect_ratios = (
        (0.5, 1.0, 2.0),
    ) * 5

    # Limit proposal count to keep the experiment manageable.

    model.rpn.pre_nms_top_n_train = 2000
    model.rpn.pre_nms_top_n_test = 1000

    model.rpn.post_nms_top_n_train = 1000
    model.rpn.post_nms_top_n_test = 300

    model.roi_heads.detections_per_img = 100

    return model


# ======================================================================
# TRAINING
# ======================================================================

def train_one_epoch(
    model,
    optimizer,
    train_loader,
    device,
    epoch,
    scaler=None,
):

    model.train()

    running_loss = 0.0
    batch_count = 0

    for batch_index, (images, targets) in enumerate(
        train_loader
    ):

        images = [
            image.to(device)
            for image in images
        ]

        targets = [
            {
                key: value.to(device)
                for key, value in target.items()
            }
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                loss_dict = model(
                    images,
                    targets,
                )

                total_loss = sum(
                    loss
                    for loss in loss_dict.values()
                )

            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            scaler.step(optimizer)
            scaler.update()

        else:

            loss_dict = model(
                images,
                targets,
            )

            total_loss = sum(
                loss
                for loss in loss_dict.values()
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

        running_loss += float(
            total_loss.detach().cpu()
        )

        batch_count += 1

        if batch_index % 10 == 0:

            components = {
                key: round(
                    float(value.detach().cpu()),
                    4,
                )
                for key, value in loss_dict.items()
            }

            print(
                f"Epoch {epoch} "
                f"Batch {batch_index}/{len(train_loader)} "
                f"Loss {total_loss.item():.4f} "
                f"Components {components}"
            )

    return running_loss / max(batch_count, 1)


# ======================================================================
# INFERENCE
# ======================================================================

@torch.no_grad()
def run_inference(
    model,
    images,
    device,
    confidence=0.25,
):

    model.eval()

    images = [
        image.to(device)
        for image in images
    ]

    outputs = model(images)

    filtered = []

    for output in outputs:

        scores = output["scores"]

        keep = scores >= confidence

        filtered.append({

            "boxes": output["boxes"][keep].cpu(),

            "labels": output["labels"][keep].cpu(),

            "scores": output["scores"][keep].cpu(),

        })

    return filtered


# ======================================================================
# DETECTION MATCHING
# ======================================================================

def match_detections(
    predictions,
    target,
    iou_threshold=0.5,
):

    pred_boxes = predictions["boxes"]
    pred_labels = predictions["labels"]
    pred_scores = predictions["scores"]

    gt_boxes = target["boxes"]
    gt_labels = target["labels"]

    order = torch.argsort(
        pred_scores,
        descending=True,
    )

    matched_gt = set()

    tp = 0
    fp = 0

    for pred_index in order.tolist():

        predicted_box = pred_boxes[pred_index]
        predicted_class = int(
            pred_labels[pred_index]
        )

        best_iou = 0.0
        best_gt = None

        for gt_index in range(len(gt_boxes)):

            if gt_index in matched_gt:
                continue

            if predicted_class != int(
                gt_labels[gt_index]
            ):
                continue

            iou = float(
                box_iou(
                    predicted_box.unsqueeze(0),
                    gt_boxes[gt_index].unsqueeze(0),
                )[0, 0]
            )

            if iou > best_iou:

                best_iou = iou
                best_gt = gt_index

        if (
            best_gt is not None
            and best_iou >= iou_threshold
        ):

            tp += 1
            matched_gt.add(best_gt)

        else:

            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn


# ======================================================================
# EVALUATION
# ======================================================================

@torch.no_grad()
def evaluate_model(
    model,
    val_loader,
    device,
    confidence=0.25,
):

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for images, targets in val_loader:

        predictions = run_inference(
            model,
            images,
            device,
            confidence,
        )

        for prediction, target in zip(
            predictions,
            targets,
        ):

            tp, fp, fn = match_detections(
                prediction,
                target,
                0.5,
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

    precision, recall, f1 = calculate_precision_recall(
        total_tp,
        total_fp,
        total_fn,
    )

    return {

        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,

        "precision": precision,
        "recall": recall,
        "f1": f1,

    }


# ======================================================================
# VISUALIZATION
# ======================================================================

def visualize_predictions(
    model,
    dataset,
    device,
    output_dir,
    class_names,
    num_images=12,
    confidence=0.25,
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.eval()

    indices = list(range(len(dataset)))

    random.Random(SEED).shuffle(indices)

    indices = indices[:num_images]

    for image_number, index in enumerate(indices):

        image_tensor, target = dataset[index]

        predictions = run_inference(
            model,
            [image_tensor],
            device,
            confidence,
        )[0]

        image = (
            image_tensor
            .permute(1, 2, 0)
            .numpy()
        )

        image = (
            np.clip(image * 255.0, 0, 255)
            .astype(np.uint8)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

        # Ground-truth boxes: green.

        for box, label in zip(
            target["boxes"],
            target["labels"],
        ):

            x1, y1, x2, y2 = map(
                int,
                box.tolist(),
            )

            class_id = int(label) - 1

            class_name = class_names.get(
                class_id,
                str(class_id),
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                f"GT: {class_name}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Predictions: red.

        for box, label, score in zip(
            predictions["boxes"],
            predictions["labels"],
            predictions["scores"],
        ):

            x1, y1, x2, y2 = map(
                int,
                box.tolist(),
            )

            class_id = int(label) - 1

            class_name = class_names.get(
                class_id,
                str(class_id),
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                image,
                f"Pred: {class_name} {float(score):.2f}",
                (x1, min(image.shape[0] - 5, y2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        output_path = (
            output_dir
            / f"prediction_{image_number:04d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            image,
        )

    print(
        f"Prediction visualizations saved to {output_dir}"
    )


def plot_training_history(history, output_path):

    epochs = np.arange(
        1,
        len(history["train_loss"]) + 1,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    axes[0].plot(
        epochs,
        history["train_loss"],
        marker="o",
    )

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(
        epochs,
        history["precision"],
        marker="o",
        label="Precision",
    )

    axes[1].plot(
        epochs,
        history["recall"],
        marker="o",
        label="Recall",
    )

    axes[1].plot(
        epochs,
        history["f1"],
        marker="o",
        label="F1",
    )

    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.show()


# ======================================================================
# CHECKPOINTS
# ======================================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    history,
    output_path,
    num_classes,
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save({

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "history": history,

        "num_classes": num_classes,

    }, output_path)


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return (
        checkpoint["epoch"],
        checkpoint["history"],
    )


# ======================================================================
# NUMERICAL TESTS
# ======================================================================

def run_tests():

    print("\n" + "=" * 72)
    print("LESSON 29 NUMERICAL TESTS")
    print("=" * 72)

    # Test 1: identical boxes.

    box_a = [10, 10, 50, 50]

    assert np.isclose(
        calculate_iou(box_a, box_a),
        1.0,
    )

    print("Test 1 passed: identical-box IoU.")

    # Test 2: no overlap.

    box_b = [60, 60, 100, 100]

    assert np.isclose(
        calculate_iou(box_a, box_b),
        0.0,
    )

    print("Test 2 passed: non-overlapping IoU.")

    # Test 3: partial overlap.

    box_c = [30, 10, 70, 50]

    expected = 800 / 2400

    assert np.isclose(
        calculate_iou(box_a, box_c),
        expected,
    )

    print("Test 3 passed: partial-overlap IoU.")

    # Test 4: Smooth L1.

    prediction = np.array([0.0, 0.5, 2.0])
    target = np.array([0.0, 0.0, 0.0])

    loss = smooth_l1_loss_numpy(
        prediction,
        target,
    )

    assert np.isclose(
        loss,
        (0.0 + 0.125 + 1.5) / 3,
    )

    print("Test 4 passed: Smooth L1 loss.")

    # Test 5: precision and recall.

    precision, recall, f1 = calculate_precision_recall(
        8,
        2,
        2,
    )

    assert np.isclose(precision, 0.8)
    assert np.isclose(recall, 0.8)
    assert np.isclose(f1, 0.8)

    print("Test 5 passed: precision, recall, F1.")

    # Test 6: YOLO label conversion.

    import tempfile

    with tempfile.TemporaryDirectory() as directory:

        label_path = Path(directory) / "test.txt"

        label_path.write_text(
            "0 0.5 0.5 0.5 0.5\n",
            encoding="utf-8",
        )

        boxes, labels = read_yolo_labels(
            label_path,
            100,
            100,
        )

        assert boxes.shape == (1, 4)

        assert torch.allclose(
            boxes[0],
            torch.tensor(
                [25, 25, 75, 75],
                dtype=torch.float32,
            ),
        )

        assert int(labels[0]) == 1

    print("Test 6 passed: YOLO to Faster R-CNN labels.")

    # Test 7: empty annotations.

    with tempfile.TemporaryDirectory() as directory:

        label_path = Path(directory) / "empty.txt"

        label_path.write_text(
            "",
            encoding="utf-8",
        )

        boxes, labels = read_yolo_labels(
            label_path,
            100,
            100,
        )

        assert boxes.shape == (0, 4)
        assert labels.shape == (0,)

    print("Test 7 passed: empty annotation handling.")

    # Test 8: model architecture.

    model = create_faster_rcnn(
        num_classes=6,
        min_size=128,
        max_size=128,
    )

    assert model.roi_heads.box_predictor.cls_score.out_features == 6

    print("Test 8 passed: Faster R-CNN class configuration.")

    print("\nALL LESSON 29 TESTS PASSED.")


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Lesson 29: Faster R-CNN on real TACO waste images."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-images",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--check-only",
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

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--visualize",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    set_seed()

    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("LESSON 29 — FASTER R-CNN")
    print("=" * 72)

    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("Device:", args.device)
    print("Dataset:", args.dataset)

    # --------------------------------------------------------------
    # Numerical tests
    # --------------------------------------------------------------

    run_tests()

    # --------------------------------------------------------------
    # Dataset configuration
    # --------------------------------------------------------------

    config = load_dataset_config(
        args.dataset
    )

    class_names = get_class_names(
        config
    )

    num_classes = len(class_names) + 1

    print("\nClasses:")

    for class_id, class_name in class_names.items():

        print(
            f"  YOLO {class_id} -> "
            f"Faster R-CNN {class_id + 1}: "
            f"{class_name}"
        )

    print("Total classes including background:", num_classes)

    # --------------------------------------------------------------
    # Dataset loaders
    # --------------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    ) = create_data_loaders(args)

    if args.check_only:

        print("\nDataset validation completed.")

        image, target = train_dataset[0]

        print("Example image shape:", tuple(image.shape))
        print("Example boxes:", target["boxes"].shape)
        print("Example labels:", target["labels"].tolist())

        return

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------

    model = create_faster_rcnn(
        num_classes=num_classes,
        min_size=args.imgsz,
        max_size=args.imgsz,
    )

    model.to(args.device)

    print("\nModel initialized.")
    print("Backbone: ResNet-50 FPN")
    print("Pretrained detector weights: DISABLED")
    print("Pretrained backbone weights: DISABLED")

    # --------------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------------

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.SGD(
        parameters,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5,
    )

    scaler = None

    if args.device.startswith("cuda"):

        scaler = torch.amp.GradScaler(
            "cuda"
        )

    history = {

        "train_loss": [],
        "precision": [],
        "recall": [],
        "f1": [],

    }

    start_epoch = 0

    # --------------------------------------------------------------
    # Resume checkpoint
    # --------------------------------------------------------------

    if args.resume is not None:

        start_epoch, history = load_checkpoint(
            model,
            optimizer,
            args.resume,
            args.device,
        )

        print(
            f"Resuming from epoch {start_epoch}"
        )

    # --------------------------------------------------------------
    # Load existing weights for evaluation
    # --------------------------------------------------------------

    if args.weights is not None:

        checkpoint = torch.load(
            args.weights,
            map_location=args.device,
            weights_only=False,
        )

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            model.load_state_dict(checkpoint)

        print(
            "Loaded model weights:",
            args.weights,
        )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    if not args.skip_training:

        print("\n" + "=" * 72)
        print("TRAINING FASTER R-CNN")
        print("=" * 72)

        best_f1 = -1.0

        for epoch in range(
            start_epoch,
            args.epochs,
        ):

            epoch_start = time.time()

            train_loss = train_one_epoch(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                device=args.device,
                epoch=epoch + 1,
                scaler=scaler,
            )

            validation_metrics = evaluate_model(
                model=model,
                val_loader=val_loader,
                device=args.device,
                confidence=args.confidence,
            )

            scheduler.step()

            history["train_loss"].append(
                train_loss
            )

            history["precision"].append(
                validation_metrics["precision"]
            )

            history["recall"].append(
                validation_metrics["recall"]
            )

            history["f1"].append(
                validation_metrics["f1"]
            )

            elapsed = time.time() - epoch_start

            print("\n" + "-" * 72)

            print(
                f"Epoch {epoch + 1}/{args.epochs}"
            )

            print(
                f"Train loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Precision: "
                f"{validation_metrics['precision']:.4f}"
            )

            print(
                f"Recall: "
                f"{validation_metrics['recall']:.4f}"
            )

            print(
                f"F1: "
                f"{validation_metrics['f1']:.4f}"
            )

            print(
                f"Time: {elapsed:.2f} seconds"
            )

            # Save latest checkpoint.

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                history,
                output_dir / "last_model.pth",
                num_classes,
            )

            # Save best model based on validation F1.

            if validation_metrics["f1"] > best_f1:

                best_f1 = validation_metrics["f1"]

                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    history,
                    output_dir / "best_model.pth",
                    num_classes,
                )

                print("Best checkpoint updated.")

        # ----------------------------------------------------------
        # Save training history
        # ----------------------------------------------------------

        with open(
            output_dir / "training_history.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                history,
                file,
                indent=4,
            )

        plot_training_history(
            history,
            output_dir / "training_curves.png",
        )

    # --------------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------------

    best_checkpoint = (
        output_dir / "best_model.pth"
    )

    if (
        args.skip_training
        and args.weights is None
    ):

        raise ValueError(
            "--skip-training requires --weights."
        )

    if (
        not args.skip_training
        and best_checkpoint.exists()
    ):

        checkpoint = torch.load(
            best_checkpoint,
            map_location=args.device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    print("\n" + "=" * 72)
    print("FINAL EVALUATION")
    print("=" * 72)

    final_metrics = evaluate_model(
        model,
        val_loader,
        args.device,
        confidence=args.confidence,
    )

    for key, value in final_metrics.items():

        print(f"{key}: {value}")

    # --------------------------------------------------------------
    # Save final metrics
    # --------------------------------------------------------------

    with open(
        output_dir / "evaluation.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_metrics,
            file,
            indent=4,
        )

    # --------------------------------------------------------------
    # Visualization
    # --------------------------------------------------------------

    visualize_predictions(
        model=model,
        dataset=val_dataset,
        device=args.device,
        output_dir=output_dir / "predictions",
        class_names=class_names,
        num_images=args.visualize,
        confidence=args.confidence,
    )

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    print("\n" + "=" * 72)
    print("LESSON 29 COMPLETED")
    print("=" * 72)

    print("Output directory:", output_dir)

    print("\nImportant files:")

    print("  best_model.pth")
    print("  last_model.pth")
    print("  training_history.json")
    print("  training_curves.png")
    print("  evaluation.json")
    print("  predictions/")

    print("\nNext lesson:")
    print("LESSON 30 — INSTANCE SEGMENTATION")


if __name__ == "__main__":
    main()