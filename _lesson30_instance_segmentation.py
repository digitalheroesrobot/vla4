"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 30 — INSTANCE SEGMENTATION WITH MASK R-CNN
======================================================================

FILE
----
lesson30_instance_segmentation.py

COURSE RULE
-----------
Use REAL TACO waste photographs and their original polygon annotations.

No:
    - Synthetic training images
    - Pretrained model weights
    - Pretrained backbone weights
    - Rectangular boxes pretending to be segmentation masks

MODEL
-----
Mask R-CNN
    Backbone: ResNet-50 FPN
    Initialization: Random
    Number of foreground classes: 5
    Background class: 0

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1 — Semantic vs Instance Segmentation

    Semantic segmentation:
        Assign a class to every pixel.

    Instance segmentation:
        Identify individual objects and assign a separate mask
        to every object.

    Example:

        Image contains:
            2 plastic bottles
            1 metal can
            1 paper cup

        Instance segmentation produces:
            Bottle 1 mask
            Bottle 2 mask
            Can 1 mask
            Cup 1 mask

STEP 2 — Understand Mask R-CNN

    Input image
        |
        v
    ResNet-50 Backbone
        |
        v
    Feature Pyramid Network
        |
        v
    Region Proposal Network
        |
        v
    RoI Align
        |
        +----------------------+
        |                      |
        v                      v
    Classification        Box Regression
        |
        v
    Mask Prediction
        |
        v
    Individual Object Masks

STEP 3 — Understand binary masks.

    A mask is a two-dimensional array:

        mask[y, x] = 1  -> object pixel
        mask[y, x] = 0  -> background pixel

STEP 4 — Convert TACO polygons to masks.

    Original TACO annotations contain polygon segmentation data.

    We rasterize those polygons into binary masks using OpenCV.

    Each annotated object gets its own mask.

STEP 5 — Train Mask R-CNN.

    Train using:
        - Real images
        - Real instance masks
        - Real category labels

STEP 6 — Evaluate instance segmentation.

    Calculate:
        - Mask IoU
        - Mask precision
        - Mask recall
        - Mask F1

STEP 7 — Visualize predicted masks.

STEP 8 — Connect segmentation to robotic grasping.

======================================================================
MATHEMATICAL FOUNDATION
======================================================================

1. BINARY MASK

    M(x,y) belongs to {0,1}

2. MASK INTERSECTION OVER UNION

                     |M_prediction intersection M_GT|
    Mask IoU =       ---------------------------------
                     |M_prediction union M_GT|

3. MASK PRECISION

                     TP
    Precision = -------------
                 TP + FP

4. MASK RECALL

                    TP
    Recall = ------------
             TP + FN

5. MASK F1

                  2 * Precision * Recall
    F1 =           -----------------------
                  Precision + Recall

6. MASK R-CNN LOSS

    L_total =
        L_rpn_objectness
      + L_rpn_box
      + L_roi_classifier
      + L_roi_box
      + L_mask

    The mask branch predicts a binary mask for each RoI.

======================================================================
TACO DATASET
======================================================================

Expected original dataset:

    vla4/
    |
    +-- TACO/
    |   +-- data/
    |       +-- annotations.json
    |       +-- batch_1/
    |       +-- batch_2/
    |       +-- ...
    |
    +-- lesson30_instance_segmentation.py

The script uses the ORIGINAL TACO annotations.json.

Expected COCO-style fields:

    images:
        id
        file_name
        width
        height

    annotations:
        image_id
        category_id
        segmentation
        bbox

    categories:
        id
        name

The script reads polygon segmentation annotations.

RLE annotations are not silently converted into rectangular masks.
They are skipped with a warning unless pycocotools decoding is
implemented for the particular annotation file.

======================================================================
CLASS MAPPING
======================================================================

TACO categories are mapped into five broad waste groups:

    1. plastic
    2. metal
    3. paper_cardboard
    4. glass
    5. other

The mapping is based on category-name keywords.

IMPORTANT:
    Review category_mapping.json after preparation.

    Ambiguous categories should be manually checked before
    using the dataset for a real waste-sorting robot.

======================================================================
INSTALLATION
======================================================================

    pip install -U torch torchvision opencv-python matplotlib numpy

======================================================================
RUN
======================================================================

Check dataset:

    python lesson30_instance_segmentation.py --check-only

Short training:

    python lesson30_instance_segmentation.py \
        --epochs 3 \
        --batch 2 \
        --max-train-images 200 \
        --max-val-images 50

Full training:

    python lesson30_instance_segmentation.py \
        --epochs 50 \
        --batch 2 \
        --device cuda

======================================================================
OUTPUT
======================================================================

    runs/lesson30/
        best_model.pth
        last_model.pth
        category_mapping.json
        training_history.json
        training_curves.png
        evaluation.json
        predictions/

======================================================================
OPENARM V2 APPLICATION
======================================================================

Object detection:
    "There is a plastic bottle in this region."

Instance segmentation:
    "These exact pixels belong to this plastic bottle."

A segmentation mask can help estimate:

    - Object silhouette
    - Object orientation
    - Visible surface area
    - Approximate object center
    - Candidate grasp regions
    - Object overlap and occlusion

However, a 2-D mask alone does not provide a complete 3-D grasp pose.

For robotic grasping, combine it with:
    - Camera calibration
    - Depth information
    - Camera-to-robot calibration
    - 3-D geometry
    - Grasp planning

======================================================================
NEXT LESSON
======================================================================

LESSON 31 — ADVANCED YOLO OBJECT DETECTION
======================================================================
"""

from __future__ import annotations

import argparse
import json
import random
import time

from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

DEFAULT_TACO_ROOT = Path("./TACO/data")
DEFAULT_OUTPUT = Path("./runs/lesson30")

CLASS_NAMES = {
    1: "plastic",
    2: "metal",
    3: "paper_cardboard",
    4: "glass",
    5: "other",
}

BACKGROUND_CLASS_ID = 0
NUM_CLASSES = len(CLASS_NAMES) + 1


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
# CATEGORY MAPPING
# ======================================================================

def map_category(category_name):
    """
    Map original TACO category names into five broad waste groups.

    Review the generated category_mapping.json manually.
    """

    name = category_name.lower()

    # --------------------------------------------------------------
    # Glass
    # --------------------------------------------------------------

    glass_keywords = [
        "glass",
        "wine bottle",
        "beer bottle",
        "broken glass",
    ]

    if any(word in name for word in glass_keywords):
        return 4

    # --------------------------------------------------------------
    # Metal
    # --------------------------------------------------------------

    metal_keywords = [
        "aluminium",
        "aluminum",
        "metal",
        "steel",
        "tin",
        "can",
        "foil",
        "pop tab",
        "battery",
        "wire",
    ]

    if any(word in name for word in metal_keywords):
        return 2

    # --------------------------------------------------------------
    # Paper and cardboard
    # --------------------------------------------------------------

    paper_keywords = [
        "paper",
        "cardboard",
        "carton",
        "paperboard",
        "tissue",
        "napkin",
        "receipt",
        "magazine",
        "newspaper",
    ]

    if any(word in name for word in paper_keywords):
        return 3

    # --------------------------------------------------------------
    # Plastic
    # --------------------------------------------------------------

    plastic_keywords = [
        "plastic",
        "polystyrene",
        "styrofoam",
        "pet",
        "wrapper",
        "plastic bag",
        "plastic bottle",
        "straw",
        "lid",
        "cup",
        "cigarette butt",
    ]

    if any(word in name for word in plastic_keywords):
        return 1

    return 5


def build_category_mapping(categories):

    mapping = {}

    for category in categories:

        category_id = int(category["id"])
        category_name = str(category["name"])

        class_id = map_category(
            category_name
        )

        mapping[category_id] = {

            "original_category_id": category_id,
            "original_category_name": category_name,

            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],

        }

    return mapping


# ======================================================================
# POLYGON TO MASK
# ======================================================================

def polygons_to_mask(
    segmentation,
    height,
    width,
):
    """
    Convert COCO polygon segmentation into a binary mask.

    COCO polygon format:

        [
            [x1, y1, x2, y2, ...],
            [x1, y1, x2, y2, ...],
        ]

    Multiple polygons are combined into one instance mask.

    Returns:
        uint8 mask with shape (H, W).
    """

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    if not isinstance(segmentation, list):
        return mask

    polygons = []

    # Some annotation files may store a single polygon as a flat list.
    if segmentation and isinstance(segmentation[0], (int, float)):
        segmentation = [segmentation]

    for polygon in segmentation:

        if not isinstance(polygon, (list, tuple)):
            continue

        if len(polygon) < 6 or len(polygon) % 2 != 0:
            continue

        points = np.asarray(
            polygon,
            dtype=np.float32,
        ).reshape(-1, 2)

        points[:, 0] = np.clip(
            points[:, 0],
            0,
            width - 1,
        )

        points[:, 1] = np.clip(
            points[:, 1],
            0,
            height - 1,
        )

        points = np.round(
            points
        ).astype(np.int32)

        polygons.append(points)

    if polygons:

        cv2.fillPoly(
            mask,
            polygons,
            color=1,
        )

    return mask


# ======================================================================
# MASK TO BOUNDING BOX
# ======================================================================

def mask_to_box(mask):
    """
    Calculate xyxy bounding box from a binary instance mask.
    """

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())

    # Use exclusive x2/y2 coordinates.
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    return [
        x1,
        y1,
        x2,
        y2,
    ]


# ======================================================================
# MASK METRICS
# ======================================================================

def mask_iou(mask_a, mask_b):

    mask_a = np.asarray(mask_a) > 0
    mask_b = np.asarray(mask_b) > 0

    intersection = np.logical_and(
        mask_a,
        mask_b,
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b,
    ).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def calculate_precision_recall(tp, fp, fn):

    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)

    f1 = (
        2.0 * precision * recall
        / max(precision + recall, 1e-12)
    )

    return precision, recall, f1


# ======================================================================
# DATASET
# ======================================================================

class TacoInstanceDataset(Dataset):

    def __init__(
        self,
        taco_root,
        split="train",
        val_ratio=0.2,
        seed=SEED,
        augment=False,
        max_images=None,
    ):

        self.taco_root = Path(taco_root).resolve()

        self.annotation_path = (
            self.taco_root / "annotations.json"
        )

        if not self.annotation_path.exists():
            raise FileNotFoundError(
                f"Missing TACO annotation file: "
                f"{self.annotation_path}"
            )

        with open(
            self.annotation_path,
            "r",
            encoding="utf-8",
        ) as file:

            self.coco = json.load(file)

        self.images = self.coco["images"]
        self.annotations = self.coco["annotations"]
        self.categories = self.coco["categories"]

        self.category_mapping = build_category_mapping(
            self.categories
        )

        self.augment = augment
        self.split = split

        # ----------------------------------------------------------
        # Group annotations by image.
        # ----------------------------------------------------------

        self.annotations_by_image = defaultdict(list)

        for annotation in self.annotations:

            image_id = int(annotation["image_id"])

            self.annotations_by_image[image_id].append(
                annotation
            )

        # ----------------------------------------------------------
        # Deterministic train/validation split.
        # ----------------------------------------------------------

        image_ids = sorted(
            int(image["id"])
            for image in self.images
        )

        random_generator = random.Random(seed)

        random_generator.shuffle(
            image_ids
        )

        val_count = max(
            1,
            int(len(image_ids) * val_ratio),
        )

        val_ids = set(
            image_ids[:val_count]
        )

        train_ids = set(
            image_ids[val_count:]
        )

        selected_ids = (
            train_ids
            if split == "train"
            else val_ids
        )

        self.image_records = [
            image
            for image in self.images
            if int(image["id"]) in selected_ids
        ]

        # ----------------------------------------------------------
        # Optional image limit.
        # ----------------------------------------------------------

        if max_images is not None:

            random_generator.shuffle(
                self.image_records
            )

            self.image_records = (
                self.image_records[:max_images]
            )

        print(
            f"{split.upper()} images: "
            f"{len(self.image_records)}"
        )

        if not self.image_records:
            raise RuntimeError(
                f"No images selected for {split} split."
            )

    def __len__(self):

        return len(self.image_records)

    def get_image_path(self, image_record):

        filename = Path(
            image_record["file_name"]
        )

        return self.taco_root / filename

    def build_instances(
        self,
        image_record,
        height,
        width,
    ):

        image_id = int(image_record["id"])

        instance_masks = []
        instance_labels = []
        instance_boxes = []

        for annotation in self.annotations_by_image[image_id]:

            category_id = int(
                annotation["category_id"]
            )

            if category_id not in self.category_mapping:
                continue

            segmentation = annotation.get(
                "segmentation",
                [],
            )

            # This lesson handles polygon annotations.
            # RLE requires an explicit COCO RLE decoder.
            if not isinstance(segmentation, list):
                continue

            mask = polygons_to_mask(
                segmentation,
                height,
                width,
            )

            if mask.sum() == 0:
                continue

            box = mask_to_box(mask)

            if box is None:
                continue

            x1, y1, x2, y2 = box

            if x2 <= x1 or y2 <= y1:
                continue

            class_id = self.category_mapping[
                category_id
            ]["class_id"]

            instance_masks.append(mask)
            instance_labels.append(class_id)
            instance_boxes.append(box)

        if len(instance_masks) == 0:

            masks = np.zeros(
                (0, height, width),
                dtype=np.uint8,
            )

            boxes = np.zeros(
                (0, 4),
                dtype=np.float32,
            )

            labels = np.zeros(
                (0,),
                dtype=np.int64,
            )

        else:

            masks = np.stack(
                instance_masks,
                axis=0,
            )

            boxes = np.asarray(
                instance_boxes,
                dtype=np.float32,
            )

            labels = np.asarray(
                instance_labels,
                dtype=np.int64,
            )

        return masks, boxes, labels

    def __getitem__(self, index):

        image_record = self.image_records[index]

        image_path = self.get_image_path(
            image_record
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        height, width = image.shape[:2]

        masks, boxes, labels = self.build_instances(
            image_record,
            height,
            width,
        )

        # ----------------------------------------------------------
        # Horizontal flip augmentation.
        # ----------------------------------------------------------

        if self.augment and random.random() < 0.5:

            image = np.ascontiguousarray(
                image[:, ::-1]
            )

            if len(masks) > 0:

                masks = np.ascontiguousarray(
                    masks[:, :, ::-1]
                )

                old_x1 = boxes[:, 0].copy()
                old_x2 = boxes[:, 2].copy()

                boxes[:, 0] = width - old_x2
                boxes[:, 2] = width - old_x1

        # ----------------------------------------------------------
        # Image tensor.
        # ----------------------------------------------------------

        image_tensor = torch.from_numpy(
            image.copy()
        ).permute(2, 0, 1).float() / 255.0

        # ----------------------------------------------------------
        # Target tensors.
        # ----------------------------------------------------------

        mask_tensor = torch.from_numpy(
            masks.copy()
        ).to(torch.uint8)

        box_tensor = torch.as_tensor(
            boxes,
            dtype=torch.float32,
        ).reshape(-1, 4)

        label_tensor = torch.as_tensor(
            labels,
            dtype=torch.int64,
        )

        area_tensor = (
            (box_tensor[:, 2] - box_tensor[:, 0])
            * (box_tensor[:, 3] - box_tensor[:, 1])
        )

        target = {

            "boxes": box_tensor,

            "labels": label_tensor,

            "masks": mask_tensor,

            "image_id": torch.tensor(
                [int(image_record["id"])],
                dtype=torch.int64,
            ),

            "area": area_tensor,

            "iscrowd": torch.zeros(
                len(label_tensor),
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


# ======================================================================
# MODEL
# ======================================================================

def create_mask_rcnn(num_classes=NUM_CLASSES):

    """
    Create Mask R-CNN from random initialization.

    No pretrained detector or backbone weights.
    """

    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=None,
        min_size=512,
        max_size=1024,
        trainable_backbone_layers=5,
    )

    # --------------------------------------------------------------
    # Replace box classification head.
    # --------------------------------------------------------------

    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes,
    )

    # --------------------------------------------------------------
    # Replace mask prediction head.
    # --------------------------------------------------------------

    in_features_mask = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )

    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )

    model.roi_heads.detections_per_img = 100

    return model


# ======================================================================
# TRAINING
# ======================================================================

def train_one_epoch(
    model,
    optimizer,
    loader,
    device,
    epoch,
    scaler=None,
):

    model.train()

    total_loss = 0.0
    batches = 0

    for batch_index, (images, targets) in enumerate(loader):

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

        optimizer.zero_grad(
            set_to_none=True
        )

        if scaler is not None:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                loss_dict = model(
                    images,
                    targets,
                )

                loss = sum(
                    value
                    for value in loss_dict.values()
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            scaler.step(optimizer)
            scaler.update()

        else:

            loss_dict = model(
                images,
                targets,
            )

            loss = sum(
                value
                for value in loss_dict.values()
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

        total_loss += float(
            loss.detach().cpu()
        )

        batches += 1

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
                f"Batch {batch_index}/{len(loader)} "
                f"Loss {loss.item():.4f} "
                f"Components {components}"
            )

    return total_loss / max(batches, 1)


# ======================================================================
# INFERENCE
# ======================================================================

@torch.no_grad()
def predict(
    model,
    images,
    device,
    confidence=0.5,
):

    model.eval()

    images = [
        image.to(device)
        for image in images
    ]

    outputs = model(images)

    filtered = []

    for output in outputs:

        keep = (
            output["scores"] >= confidence
        )

        filtered.append({

            "boxes": output["boxes"][keep].cpu(),

            "labels": output["labels"][keep].cpu(),

            "scores": output["scores"][keep].cpu(),

            "masks": (
                output["masks"][keep]
                .cpu()
                .squeeze(1)
                >= 0.5
            ),

        })

    return filtered


# ======================================================================
# INSTANCE MASK MATCHING
# ======================================================================

def match_instances(
    prediction,
    target,
    iou_threshold=0.5,
):

    pred_masks = prediction["masks"].numpy()
    pred_labels = prediction["labels"].numpy()
    pred_scores = prediction["scores"].numpy()

    gt_masks = target["masks"].numpy()
    gt_labels = target["labels"].numpy()

    order = np.argsort(
        -pred_scores
    )

    matched_gt = set()

    tp = 0
    fp = 0
    matched_ious = []

    for pred_index in order:

        predicted_mask = pred_masks[pred_index]
        predicted_class = int(
            pred_labels[pred_index]
        )

        best_iou = 0.0
        best_gt_index = None

        for gt_index in range(len(gt_masks)):

            if gt_index in matched_gt:
                continue

            if predicted_class != int(
                gt_labels[gt_index]
            ):
                continue

            iou = mask_iou(
                predicted_mask,
                gt_masks[gt_index],
            )

            if iou > best_iou:

                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= iou_threshold
        ):

            tp += 1
            matched_gt.add(best_gt_index)
            matched_ious.append(best_iou)

        else:

            fp += 1

    fn = len(gt_masks) - len(matched_gt)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "matched_ious": matched_ious,
    }


# ======================================================================
# EVALUATION
# ======================================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    confidence=0.5,
):

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_ious = []

    per_class = {
        class_id: {
            "TP": 0,
            "FP": 0,
            "FN": 0,
        }
        for class_id in CLASS_NAMES
    }

    for images, targets in loader:

        predictions = predict(
            model,
            images,
            device,
            confidence,
        )

        for prediction, target in zip(
            predictions,
            targets,
        ):

            result = match_instances(
                prediction,
                target,
                iou_threshold=0.5,
            )

            total_tp += result["TP"]
            total_fp += result["FP"]
            total_fn += result["FN"]

            all_ious.extend(
                result["matched_ious"]
            )

            # Class-specific matching.
            # Each class is evaluated independently.

            for class_id in CLASS_NAMES:

                class_pred = {

                    "masks": prediction["masks"][
                        prediction["labels"] == class_id
                    ],

                    "labels": prediction["labels"][
                        prediction["labels"] == class_id
                    ],

                    "scores": prediction["scores"][
                        prediction["labels"] == class_id
                    ],

                }

                class_gt = {

                    "masks": target["masks"][
                        target["labels"] == class_id
                    ],

                    "labels": target["labels"][
                        target["labels"] == class_id
                    ],

                }

                class_result = match_instances(
                    class_pred,
                    class_gt,
                    iou_threshold=0.5,
                )

                for key in ("TP", "FP", "FN"):
                    per_class[class_id][key] += (
                        class_result[key]
                    )

    precision, recall, f1 = calculate_precision_recall(
        total_tp,
        total_fp,
        total_fn,
    )

    class_results = {}

    for class_id, counts in per_class.items():

        p, r, class_f1 = calculate_precision_recall(
            counts["TP"],
            counts["FP"],
            counts["FN"],
        )

        class_results[CLASS_NAMES[class_id]] = {

            **counts,

            "precision": p,
            "recall": r,
            "f1": class_f1,

        }

    return {

        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,

        "precision": precision,
        "recall": recall,
        "f1": f1,

        "mean_matched_mask_iou": (
            float(np.mean(all_ious))
            if all_ious
            else 0.0
        ),

        "per_class": class_results,

    }


# ======================================================================
# VISUALIZATION
# ======================================================================

def visualize_predictions(
    model,
    dataset,
    device,
    output_dir,
    num_images=12,
    confidence=0.5,
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    indices = list(
        range(len(dataset))
    )

    random.Random(SEED).shuffle(
        indices
    )

    indices = indices[:num_images]

    model.eval()

    for image_number, index in enumerate(indices):

        image_tensor, target = dataset[index]

        prediction = predict(
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

        image = np.clip(
            image * 255,
            0,
            255,
        ).astype(np.uint8)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

        overlay = image.copy()

        # ----------------------------------------------------------
        # Draw ground-truth contours.
        # ----------------------------------------------------------

        for mask, label in zip(
            target["masks"].numpy(),
            target["labels"].numpy(),
        ):

            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            cv2.drawContours(
                overlay,
                contours,
                -1,
                (0, 255, 0),
                2,
            )

        # ----------------------------------------------------------
        # Draw predicted instance masks.
        # ----------------------------------------------------------

        for mask, label, score in zip(
            prediction["masks"].numpy(),
            prediction["labels"].numpy(),
            prediction["scores"].numpy(),
        ):

            class_name = CLASS_NAMES.get(
                int(label),
                "unknown",
            )

            mask_uint8 = (
                mask.astype(np.uint8) * 255
            )

            colored = np.zeros_like(
                overlay
            )

            colored[:, :, 1] = mask_uint8

            overlay = cv2.addWeighted(
                overlay,
                1.0,
                colored,
                0.35,
                0,
            )

            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            cv2.drawContours(
                overlay,
                contours,
                -1,
                (0, 0, 255),
                2,
            )

            ys, xs = np.where(mask)

            if len(xs) > 0:

                x = int(xs.min())
                y = int(ys.min())

                cv2.putText(
                    overlay,
                    f"{class_name} {score:.2f}",
                    (x, max(20, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        output_path = (
            output_dir
            / f"instance_{image_number:04d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            overlay,
        )

    print(
        "Prediction images saved:",
        output_dir,
    )


# ======================================================================
# TRAINING CURVES
# ======================================================================

def plot_history(history, output_path):

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

    axes[0].set_title("Mask R-CNN Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    for key in (
        "precision",
        "recall",
        "f1",
        "mean_matched_mask_iou",
    ):

        axes[1].plot(
            epochs,
            history[key],
            marker="o",
            label=key,
        )

    axes[1].set_title("Validation Mask Metrics")
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
# CHECKPOINT
# ======================================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    history,
    output_path,
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

        "num_classes": NUM_CLASSES,

    }, output_path)


# ======================================================================
# TESTS
# ======================================================================

def run_tests():

    print("\n" + "=" * 72)
    print("LESSON 30 NUMERICAL TESTS")
    print("=" * 72)

    # --------------------------------------------------------------
    # Test 1: polygon to mask.
    # --------------------------------------------------------------

    polygon = [
        [
            10, 10,
            40, 10,
            40, 40,
            10, 40,
        ]
    ]

    mask = polygons_to_mask(
        polygon,
        64,
        64,
    )

    assert mask.shape == (64, 64)
    assert mask.sum() > 0

    print("Test 1 passed: polygon-to-mask conversion.")

    # --------------------------------------------------------------
    # Test 2: mask to bounding box.
    # --------------------------------------------------------------

    box = mask_to_box(mask)

    assert box == [10, 10, 41, 41]

    print("Test 2 passed: mask-to-box conversion.")

    # --------------------------------------------------------------
    # Test 3: identical mask IoU.
    # --------------------------------------------------------------

    assert np.isclose(
        mask_iou(mask, mask),
        1.0,
    )

    print("Test 3 passed: identical-mask IoU.")

    # --------------------------------------------------------------
    # Test 4: disjoint mask IoU.
    # --------------------------------------------------------------

    other = np.zeros(
        (64, 64),
        dtype=np.uint8,
    )

    other[45:55, 45:55] = 1

    assert np.isclose(
        mask_iou(mask, other),
        0.0,
    )

    print("Test 4 passed: disjoint-mask IoU.")

    # --------------------------------------------------------------
    # Test 5: partial overlap.
    # --------------------------------------------------------------

    mask_a = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    mask_b = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    mask_a[0:10, 0:10] = 1
    mask_b[5:15, 0:10] = 1

    assert np.isclose(
        mask_iou(mask_a, mask_b),
        1.0 / 3.0,
    )

    print("Test 5 passed: partial mask overlap.")

    # --------------------------------------------------------------
    # Test 6: precision and recall.
    # --------------------------------------------------------------

    precision, recall, f1 = calculate_precision_recall(
        8,
        2,
        2,
    )

    assert np.isclose(precision, 0.8)
    assert np.isclose(recall, 0.8)
    assert np.isclose(f1, 0.8)

    print("Test 6 passed: mask precision and recall.")

    # --------------------------------------------------------------
    # Test 7: category mapping.
    # --------------------------------------------------------------

    assert map_category("Plastic bottle") == 1
    assert map_category("Aluminium can") == 2
    assert map_category("Paper cup") == 3
    assert map_category("Broken glass") == 4

    print("Test 7 passed: category mapping.")

    # --------------------------------------------------------------
    # Test 8: invalid polygon.
    # --------------------------------------------------------------

    invalid_mask = polygons_to_mask(
        [[1, 2, 3]],
        32,
        32,
    )

    assert invalid_mask.sum() == 0

    print("Test 8 passed: invalid polygon handling.")

    # --------------------------------------------------------------
    # Test 9: empty mask.
    # --------------------------------------------------------------

    empty_mask = np.zeros(
        (32, 32),
        dtype=np.uint8,
    )

    assert mask_to_box(empty_mask) is None

    print("Test 9 passed: empty-mask handling.")

    # --------------------------------------------------------------
    # Test 10: Mask R-CNN architecture.
    # --------------------------------------------------------------

    model = create_mask_rcnn(
        num_classes=NUM_CLASSES
    )

    assert (
        model.roi_heads.box_predictor.cls_score.out_features
        == NUM_CLASSES
    )

    assert (
        model.roi_heads.mask_predictor.mask_fcn_logits.out_channels
        == NUM_CLASSES
    )

    print("Test 10 passed: Mask R-CNN architecture.")

    print("\nALL LESSON 30 TESTS PASSED.")


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Lesson 30: Mask R-CNN instance segmentation "
            "on real TACO images."
        )
    )

    parser.add_argument(
        "--taco-root",
        type=str,
        default=str(DEFAULT_TACO_ROOT),
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
        "--val-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--visualize",
        type=int,
        default=12,
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
        "--resume",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    set_seed()

    taco_root = Path(
        args.taco_root
    ).resolve()

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("LESSON 30 — MASK R-CNN INSTANCE SEGMENTATION")
    print("=" * 72)

    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("Device:", args.device)
    print("TACO root:", taco_root)

    # --------------------------------------------------------------
    # Numerical tests.
    # --------------------------------------------------------------

    run_tests()

    # --------------------------------------------------------------
    # Create datasets.
    # --------------------------------------------------------------

    train_dataset = TacoInstanceDataset(
        taco_root=taco_root,
        split="train",
        val_ratio=args.val_ratio,
        augment=True,
        max_images=args.max_train_images,
    )

    val_dataset = TacoInstanceDataset(
        taco_root=taco_root,
        split="val",
        val_ratio=args.val_ratio,
        augment=False,
        max_images=args.max_val_images,
    )

    # --------------------------------------------------------------
    # Save category mapping.
    # --------------------------------------------------------------

    mapping_path = (
        output_dir / "category_mapping.json"
    )

    with open(
        mapping_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            train_dataset.category_mapping,
            file,
            indent=4,
        )

    print(
        "\nCategory mapping saved:",
        mapping_path,
    )

    # --------------------------------------------------------------
    # Dataset inspection.
    # --------------------------------------------------------------

    image, target = train_dataset[0]

    print("\nExample training sample:")
    print("Image shape:", tuple(image.shape))
    print("Instance masks:", tuple(target["masks"].shape))
    print("Boxes:", tuple(target["boxes"].shape))
    print("Labels:", target["labels"].tolist())

    if args.check_only:

        print("\nDataset inspection completed.")
        print("No training performed.")

        return

    # --------------------------------------------------------------
    # Data loaders.
    # --------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=detection_collate_fn,
        pin_memory=args.device.startswith("cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=detection_collate_fn,
        pin_memory=args.device.startswith("cuda"),
    )

    # --------------------------------------------------------------
    # Model.
    # --------------------------------------------------------------

    model = create_mask_rcnn(
        num_classes=NUM_CLASSES
    )

    model.to(args.device)

    print("\nMask R-CNN initialized.")
    print("Backbone: ResNet-50 FPN")
    print("Detector weights: random initialization")
    print("Backbone weights: random initialization")

    # --------------------------------------------------------------
    # Optimizer.
    # --------------------------------------------------------------

    optimizer = torch.optim.SGD(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
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
        "mean_matched_mask_iou": [],

    }

    start_epoch = 0

    # --------------------------------------------------------------
    # Resume.
    # --------------------------------------------------------------

    if args.resume is not None:

        checkpoint = torch.load(
            args.resume,
            map_location=args.device,
            weights_only=False,
        )

        if checkpoint["num_classes"] != NUM_CLASSES:
            raise ValueError(
                "Checkpoint class count does not match this dataset."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = int(
            checkpoint["epoch"]
        )

        history = checkpoint["history"]

        print(
            f"Resuming from epoch {start_epoch}"
        )

    # --------------------------------------------------------------
    # Load weights for evaluation.
    # --------------------------------------------------------------

    if args.weights is not None:

        checkpoint = torch.load(
            args.weights,
            map_location=args.device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            "Loaded checkpoint:",
            args.weights,
        )

    # --------------------------------------------------------------
    # Training.
    # --------------------------------------------------------------

    if not args.skip_training:

        best_f1 = -1.0

        print("\n" + "=" * 72)
        print("MASK R-CNN TRAINING")
        print("=" * 72)

        for epoch in range(
            start_epoch,
            args.epochs,
        ):

            epoch_start = time.time()

            train_loss = train_one_epoch(
                model=model,
                optimizer=optimizer,
                loader=train_loader,
                device=args.device,
                epoch=epoch + 1,
                scaler=scaler,
            )

            validation = evaluate_model(
                model=model,
                loader=val_loader,
                device=args.device,
                confidence=args.confidence,
            )

            scheduler.step()

            history["train_loss"].append(
                train_loss
            )

            for key in (
                "precision",
                "recall",
                "f1",
                "mean_matched_mask_iou",
            ):

                history[key].append(
                    validation[key]
                )

            elapsed = time.time() - epoch_start

            print("\n" + "-" * 72)
            print(
                f"Epoch {epoch + 1}/{args.epochs}"
            )

            print(
                f"Training loss: {train_loss:.4f}"
            )

            print(
                f"Mask precision: "
                f"{validation['precision']:.4f}"
            )

            print(
                f"Mask recall: "
                f"{validation['recall']:.4f}"
            )

            print(
                f"Mask F1: "
                f"{validation['f1']:.4f}"
            )

            print(
                f"Mean matched mask IoU: "
                f"{validation['mean_matched_mask_iou']:.4f}"
            )

            print(
                f"Epoch time: {elapsed:.2f} seconds"
            )

            # ------------------------------------------------------
            # Save latest checkpoint.
            # ------------------------------------------------------

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                history,
                output_dir / "last_model.pth",
            )

            # ------------------------------------------------------
            # Save best checkpoint.
            # ------------------------------------------------------

            if validation["f1"] > best_f1:

                best_f1 = validation["f1"]

                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    history,
                    output_dir / "best_model.pth",
                )

                print("Best model checkpoint updated.")

        # ----------------------------------------------------------
        # Save history.
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

        plot_history(
            history,
            output_dir / "training_curves.png",
        )

    # --------------------------------------------------------------
    # Load best checkpoint for final evaluation.
    # --------------------------------------------------------------

    if args.skip_training and args.weights is None:

        raise ValueError(
            "--skip-training requires --weights."
        )

    best_checkpoint = (
        output_dir / "best_model.pth"
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

    # --------------------------------------------------------------
    # Final evaluation.
    # --------------------------------------------------------------

    print("\n" + "=" * 72)
    print("FINAL INSTANCE SEGMENTATION EVALUATION")
    print("=" * 72)

    evaluation = evaluate_model(
        model=model,
        loader=val_loader,
        device=args.device,
        confidence=args.confidence,
    )

    print("\nOverall metrics:")

    for key in (
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "f1",
        "mean_matched_mask_iou",
    ):

        print(
            f"{key}: {evaluation[key]}"
        )

    print("\nPer-class metrics:")

    for class_name, values in evaluation["per_class"].items():

        print(f"\n{class_name}")

        for key, value in values.items():

            print(
                f"  {key}: {value}"
            )

    with open(
        output_dir / "evaluation.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation,
            file,
            indent=4,
        )

    # --------------------------------------------------------------
    # Visualize real images.
    # --------------------------------------------------------------

    visualize_predictions(
        model=model,
        dataset=val_dataset,
        device=args.device,
        output_dir=output_dir / "predictions",
        num_images=args.visualize,
        confidence=args.confidence,
    )

    # --------------------------------------------------------------
    # Final report.
    # --------------------------------------------------------------

    print("\n" + "=" * 72)
    print("LESSON 30 COMPLETED")
    print("=" * 72)

    print("Output:", output_dir)

    print("\nFiles:")
    print("  best_model.pth")
    print("  last_model.pth")
    print("  category_mapping.json")
    print("  training_history.json")
    print("  training_curves.png")
    print("  evaluation.json")
    print("  predictions/")

    print("\nOpenArm V2 application:")
    print(
        "Use instance masks to estimate object silhouettes "
        "and candidate grasp regions."
    )

    print("\nNext lesson:")
    print("LESSON 31 — ADVANCED YOLO OBJECT DETECTION")


if __name__ == "__main__":
    main()