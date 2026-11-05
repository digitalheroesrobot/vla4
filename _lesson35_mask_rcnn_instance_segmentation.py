"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 35 — ADVANCED MASK R-CNN INSTANCE SEGMENTATION
======================================================================

FILENAME:
    lesson35_mask_rcnn_instance_segmentation.py

COURSE RULES
------------
1. No pretrained weights.
2. Use real TACO waste photographs.
3. Implement mathematical components where educationally useful.
4. Use torchvision's Mask R-CNN architecture for the full detector.
5. Include tests, training, evaluation, and visualizations.
6. Keep the complete lesson runnable as one Python file.

======================================================================
1. SEMANTIC SEGMENTATION VS INSTANCE SEGMENTATION
======================================================================

Semantic segmentation:

    Every pixel receives a semantic class.

    Example:
        all plastic pixels -> plastic
        all metal pixels   -> metal

Instance segmentation:

    Every object receives its own mask.

    Example:
        plastic bottle A -> mask A
        plastic bottle B -> mask B
        metal can A      -> mask C

Two objects belonging to the same class remain separate instances.

This distinction matters in robotic waste sorting.

A robot needs to know not only that plastic exists, but also where
each individual plastic object is located.

======================================================================
2. MASK R-CNN ARCHITECTURE
======================================================================

Input image
    |
    v
Backbone CNN
    |
    v
Feature Pyramid Network (FPN)
    |
    +---------------------------+
    |                           |
    v                           v
Region Proposal Network       Feature maps
(RPN)                          |
    |                           |
    v                           |
Proposals                       |
    |                           |
    +------------+--------------+
                 |
                 v
              RoI Align
                 |
          +------+------+
          |             |
          v             v
    Box predictor    Mask predictor
          |             |
          v             v
    Class + box      Instance masks
    coordinates

======================================================================
3. REGION PROPOSAL NETWORK
======================================================================

RPN predicts:

    1. Objectness score
    2. Bounding-box coordinate adjustments

For anchor box A:

    A = (x, y, w, h)

Predicted transformation:

    tx, ty, tw, th

Decoded box:

    x' = x + tx * w
    y' = y + ty * h

    w' = w * exp(tw)
    h' = h * exp(th)

RPN produces candidate regions, not final semantic classes.

======================================================================
4. RoI ALIGN
======================================================================

RoI Align extracts a fixed-size feature representation for every
proposal.

Unlike traditional RoI Pooling, RoI Align avoids coordinate
quantization.

This improves spatial alignment between features and object masks.

======================================================================
5. MASK PREDICTION
======================================================================

For every detected object, Mask R-CNN predicts a small spatial mask.

For K classes:

    mask output shape:

        [N, K, H, W]

where:

    N = number of detected instances
    K = number of classes
    H, W = mask resolution

For each detection, the mask corresponding to its predicted class
is selected.

======================================================================
6. TRAINING LOSSES
======================================================================

Mask R-CNN uses multiple objectives:

    L_total =
        L_classifier
        + L_box
        + L_mask
        + L_objectness
        + L_rpn_box

The torchvision training implementation handles these losses.

======================================================================
7. INSTANCE MASK EVALUATION
======================================================================

For binary masks A and B:

    Mask IoU = |A intersection B| / |A union B|

For each predicted instance:

    Match with an unused ground-truth instance of the same class.

    If mask IoU >= threshold:
        true positive

    Otherwise:
        false positive

Unmatched ground-truth instances:
    false negatives

This lesson implements instance precision, recall, F1, and mean
matched mask IoU.

These are educational metrics, not official COCO mask AP.

======================================================================
8. REAL-WORLD ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    RGB cameras
       |
       v
    Mask R-CNN
       |
       v
    Individual object masks
       |
       v
    Object classification
       |
       v
    Object center / grasp region
       |
       v
    Camera-to-robot coordinate transformation
       |
       v
    ACT grasping policy

Instance masks help separate overlapping waste objects.

However, a 2D mask alone does not provide full 3D grasp geometry.
Depth sensing or calibrated geometry is still required.

======================================================================
9. INSTALLATION
======================================================================

    pip install -U torch torchvision numpy matplotlib opencv-python pillow

======================================================================
10. DATASET
======================================================================

Expected layout:

    TACO/
        data/
            annotations.json
            batch_1/
            batch_2/
            ...

TACO polygon annotations are converted into binary instance masks.

======================================================================
11. RUN
======================================================================

Run tests:

    python lesson35_mask_rcnn_instance_segmentation.py --check-only

Train:

    python lesson35_mask_rcnn_instance_segmentation.py \
        --epochs 5 \
        --max-images 500

Train on CUDA:

    python lesson35_mask_rcnn_instance_segmentation.py \
        --epochs 10 \
        --max-images 1000 \
        --device cuda

Resume is not implemented in this educational version.

======================================================================
OUTPUT
======================================================================

    runs/lesson35/
        best_model.pth
        training_history.json
        training_curve.png
        evaluation.json
        predictions/

======================================================================
NEXT LESSON
======================================================================

Lesson 36 — YOLO Instance Segmentation.
======================================================================
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# ======================================================================
# CONFIGURATION
# ======================================================================

CLASS_NAMES = [
    "plastic",
    "metal",
    "paper_cardboard",
    "glass",
    "other",
]

NUM_CLASSES = len(CLASS_NAMES)

BACKGROUND_CLASS = 0


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# TACO CATEGORY MAPPING
# ======================================================================

CATEGORY_KEYWORDS = {

    "glass": [
        "glass",
        "glass bottle",
        "glass jar",
    ],

    "metal": [
        "metal",
        "aluminium",
        "aluminum",
        "tin",
        "can",
        "foil",
        "steel",
    ],

    "paper_cardboard": [
        "paper",
        "cardboard",
        "carton",
        "newspaper",
        "magazine",
        "receipt",
        "tissue",
    ],

    "plastic": [
        "plastic",
        "bottle",
        "wrapper",
        "bag",
        "packaging",
        "polystyrene",
        "styrofoam",
        "pet",
    ],
}


def map_category(name):

    name = (
        str(name)
        .lower()
        .replace("_", " ")
        .replace("/", " ")
    )

    for class_name in [
        "glass",
        "metal",
        "paper_cardboard",
        "plastic",
    ]:

        if any(
            keyword in name
            for keyword in CATEGORY_KEYWORDS[class_name]
        ):

            return CLASS_NAMES.index(class_name) + 1

    return CLASS_NAMES.index("other") + 1


# ======================================================================
# POLYGON CONVERSION
# ======================================================================

def segmentation_to_mask(segmentation, height, width):

    """
    Convert COCO polygon segmentation into a binary mask.

    COCO polygon format:

        [
            [x1, y1, x2, y2, ..., xn, yn],
            ...
        ]

    Multiple polygons are combined into one binary instance mask.

    This implementation handles polygon segmentation.
    COCO compressed RLE annotations are not decoded here.
    """

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    if not isinstance(segmentation, list):
        return mask

    for polygon in segmentation:

        if not isinstance(polygon, list):
            continue

        points = np.asarray(
            polygon,
            dtype=np.float32,
        )

        if points.size < 6:
            continue

        points = points.reshape(-1, 2)

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

        cv2.fillPoly(
            mask,
            [points],
            1,
        )

    return mask


# ======================================================================
# LOAD TACO DATASET
# ======================================================================

def resolve_image_path(root, filename):

    filename = Path(str(filename))

    candidates = [
        root / filename,
    ]

    for index, part in enumerate(filename.parts):

        if part.startswith("batch_"):

            candidates.append(
                root.joinpath(*filename.parts[index:])
            )

            break

    for candidate in candidates:

        if candidate.is_file():
            return candidate.resolve()

    return None


def load_taco_records(root, max_images=None):

    root = Path(root)

    annotation_path = root / "annotations.json"

    if not annotation_path.exists():

        raise FileNotFoundError(
            f"Cannot find {annotation_path}"
        )

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    category_mapping = {

        int(category["id"]): map_category(
            category["name"]
        )

        for category in data["categories"]

    }

    annotations_by_image = defaultdict(list)

    for annotation in data["annotations"]:

        annotations_by_image[
            int(annotation["image_id"])
        ].append(annotation)

    records = []

    for image_info in data["images"]:

        image_id = int(image_info["id"])

        image_path = resolve_image_path(
            root,
            image_info["file_name"],
        )

        if image_path is None:
            continue

        try:

            with Image.open(image_path) as image:

                width, height = image.size

        except Exception:

            continue

        instances = []

        for annotation in annotations_by_image[image_id]:

            category_id = int(
                annotation["category_id"]
            )

            if category_id not in category_mapping:
                continue

            segmentation = annotation.get(
                "segmentation",
                [],
            )

            mask = segmentation_to_mask(
                segmentation,
                height,
                width,
            )

            if mask.sum() == 0:
                continue

            ys, xs = np.where(mask > 0)

            x1 = int(xs.min())
            y1 = int(ys.min())

            x2 = int(xs.max()) + 1
            y2 = int(ys.max()) + 1

            if x2 <= x1 or y2 <= y1:
                continue

            instances.append({

                "mask": mask,

                "box": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],

                "label": category_mapping[category_id],

            })

        if len(instances) == 0:
            continue

        records.append({

            "image_id": image_id,

            "path": str(image_path),

            "width": width,

            "height": height,

            "instances": instances,

        })

    random.Random(42).shuffle(records)

    if max_images is not None:
        records = records[:max_images]

    if len(records) == 0:

        raise RuntimeError(
            "No valid polygon-annotated TACO images found. "
            "Check your dataset and annotation format."
        )

    return records


# ======================================================================
# DATASET
# ======================================================================

class TACOMaskDataset(Dataset):

    def __init__(
        self,
        records,
        augment=False,
    ):

        self.records = records

        self.augment = augment

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image = cv2.imread(
            record["path"]
        )

        if image is None:

            raise RuntimeError(
                f"Cannot read image: {record['path']}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        masks = []
        boxes = []
        labels = []

        for instance in record["instances"]:

            mask = instance["mask"]

            masks.append(mask)

            boxes.append(instance["box"])

            labels.append(instance["label"])

        masks = np.asarray(
            masks,
            dtype=np.uint8,
        )

        boxes = np.asarray(
            boxes,
            dtype=np.float32,
        ).reshape(-1, 4)

        labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        # Horizontal flip: image, boxes, and masks must all transform.
        if self.augment and random.random() < 0.5:

            image = np.ascontiguousarray(
                image[:, ::-1]
            )

            masks = np.ascontiguousarray(
                masks[:, :, ::-1]
            )

            width = image.shape[1]

            old_x1 = boxes[:, 0].copy()
            old_x2 = boxes[:, 2].copy()

            boxes[:, 0] = width - old_x2
            boxes[:, 2] = width - old_x1

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(
                image.transpose(2, 0, 1)
            )
        ).float() / 255.0

        target_masks = torch.from_numpy(
            np.ascontiguousarray(masks)
        ).to(torch.uint8)

        target_boxes = torch.from_numpy(
            np.ascontiguousarray(boxes)
        ).float()

        target_labels = torch.from_numpy(
            np.ascontiguousarray(labels)
        ).long()

        area = (
            (target_boxes[:, 2] - target_boxes[:, 0])
            *
            (target_boxes[:, 3] - target_boxes[:, 1])
        )

        target = {

            "boxes": target_boxes,

            "labels": target_labels,

            "masks": target_masks,

            "image_id": torch.tensor(
                [record["image_id"]],
                dtype=torch.int64,
            ),

            "area": area,

            "iscrowd": torch.zeros(
                len(target_labels),
                dtype=torch.int64,
            ),

        }

        return image_tensor, target, record


def collate_fn(batch):

    return tuple(zip(*batch))


# ======================================================================
# MASK R-CNN MODEL
# ======================================================================

def build_mask_rcnn_model():

    """
    Build Mask R-CNN with random initialization.

    weights=None:
        No pretrained COCO detector weights.

    weights_backbone=None:
        No pretrained ResNet backbone weights.
    """

    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES + 1,
    )

    # Explicitly configure box predictor.
    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES + 1,
    )

    # Explicitly configure mask predictor.
    in_features_mask = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )

    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        NUM_CLASSES + 1,
    )

    return model


# ======================================================================
# MASK IoU
# ======================================================================

def binary_mask_iou(mask_a, mask_b):

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

    return float(
        intersection / union
    )


def resize_prediction_mask(
    mask,
    output_height,
    output_width,
):

    mask = np.asarray(mask, dtype=np.uint8)

    return cv2.resize(
        mask,
        (output_width, output_height),
        interpolation=cv2.INTER_NEAREST,
    )


# ======================================================================
# INSTANCE MATCHING AND EVALUATION
# ======================================================================

def evaluate_instance_predictions(
    predictions,
    targets,
    score_threshold=0.5,
    mask_threshold=0.5,
    iou_threshold=0.5,
):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    matched_ious = []

    for prediction, target in zip(
        predictions,
        targets,
    ):

        pred_scores = (
            prediction["scores"].detach().cpu().numpy()
        )

        pred_labels = (
            prediction["labels"].detach().cpu().numpy()
        )

        pred_masks = (
            prediction["masks"].detach().cpu().numpy()
        )

        gt_labels = (
            target["labels"].detach().cpu().numpy()
        )

        gt_masks = (
            target["masks"].detach().cpu().numpy()
        )

        height = gt_masks.shape[-2]
        width = gt_masks.shape[-1]

        order = np.argsort(
            -pred_scores,
            kind="stable",
        )

        used_gt = set()

        for pred_index in order:

            if pred_scores[pred_index] < score_threshold:
                continue

            pred_label = int(
                pred_labels[pred_index]
            )

            pred_mask = (
                pred_masks[pred_index, 0]
                >= mask_threshold
            )

            pred_mask = resize_prediction_mask(
                pred_mask,
                height,
                width,
            )

            best_iou = 0.0
            best_gt = None

            for gt_index in range(len(gt_labels)):

                if gt_index in used_gt:
                    continue

                if pred_label != int(gt_labels[gt_index]):
                    continue

                overlap = binary_mask_iou(
                    pred_mask,
                    gt_masks[gt_index],
                )

                if overlap > best_iou:

                    best_iou = overlap
                    best_gt = gt_index

            if (
                best_gt is not None
                and best_iou >= iou_threshold
            ):

                total_tp += 1

                used_gt.add(best_gt)

                matched_ious.append(best_iou)

            else:

                total_fp += 1

        total_fn += (
            len(gt_labels) - len(used_gt)
        )

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    mean_iou = (
        float(np.mean(matched_ious))
        if matched_ious
        else 0.0
    )

    return {

        "TP": total_tp,

        "FP": total_fp,

        "FN": total_fn,

        "precision": precision,

        "recall": recall,

        "F1": f1,

        "mean_matched_mask_IoU": mean_iou,

    }


# ======================================================================
# TRAINING
# ======================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
):

    model.train()

    total_loss = 0.0

    for step, (images, targets, _) in enumerate(loader):

        images = [
            image.to(device)
            for image in images
        ]

        targets = [

            {
                key: value.to(device)
                if isinstance(value, torch.Tensor)
                else value

                for key, value in target.items()
            }

            for target in targets

        ]

        loss_dict = model(
            images,
            targets,
        )

        loss = sum(
            loss_value
            for loss_value in loss_dict.values()
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite training loss: {loss.item()}"
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += loss.item()

        if step % 10 == 0:

            print(
                f"Epoch {epoch} | "
                f"Step {step}/{len(loader)} | "
                f"Loss {loss.item():.4f}"
            )

    return total_loss / max(
        len(loader),
        1,
    )


# ======================================================================
# INFERENCE
# ======================================================================

@torch.no_grad()
def run_inference(
    model,
    loader,
    device,
):

    model.eval()

    predictions = []
    targets = []
    records = []

    for images, batch_targets, batch_records in loader:

        images = [
            image.to(device)
            for image in images
        ]

        batch_predictions = model(images)

        predictions.extend(
            batch_predictions
        )

        targets.extend(
            batch_targets
        )

        records.extend(
            batch_records
        )

    return predictions, targets, records


# ======================================================================
# VISUALIZATION
# ======================================================================

def visualize_predictions(
    records,
    predictions,
    targets,
    output_dir,
    max_examples=8,
    score_threshold=0.4,
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, (
        record,
        prediction,
        target,
    ) in enumerate(
        zip(records, predictions, targets)
    ):

        if index >= max_examples:
            break

        image = cv2.imread(
            record["path"]
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        height, width = image.shape[:2]

        pred_boxes = (
            prediction["boxes"].cpu().numpy()
        )

        pred_labels = (
            prediction["labels"].cpu().numpy()
        )

        pred_scores = (
            prediction["scores"].cpu().numpy()
        )

        pred_masks = (
            prediction["masks"].cpu().numpy()
        )

        gt_masks = (
            target["masks"].numpy()
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6),
        )

        axes[0].imshow(image)
        axes[0].set_title("Real TACO Image")

        axes[1].imshow(image)
        axes[1].set_title("Ground Truth Instances")

        for mask_index, mask in enumerate(gt_masks):

            color_mask = np.zeros_like(
                image
            )

            color_mask[:, :, 1] = (
                mask > 0
            ) * 180

            axes[1].imshow(
                color_mask,
                alpha=0.35,
            )

            axes[1].text(
                5,
                15 + mask_index * 14,
                f"GT: {CLASS_NAMES[int(target['labels'][mask_index]) - 1]}",
                color="yellow",
                fontsize=8,
            )

        axes[2].imshow(image)
        axes[2].set_title("Mask R-CNN Predictions")

        for box, label, score, mask in zip(
            pred_boxes,
            pred_labels,
            pred_scores,
            pred_masks,
        ):

            if score < score_threshold:
                continue

            x1, y1, x2, y2 = box

            mask = cv2.resize(
                mask[0],
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

            binary_mask = mask >= 0.5

            overlay = np.zeros_like(
                image
            )

            overlay[:, :, 0] = (
                binary_mask * 255
            ).astype(np.uint8)

            axes[2].imshow(
                overlay,
                alpha=0.3,
            )

            axes[2].add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="yellow",
                    linewidth=2,
                )
            )

            axes[2].text(
                x1,
                y1,
                f"{CLASS_NAMES[int(label) - 1]} {score:.2f}",
                color="white",
                fontsize=8,
                bbox={
                    "facecolor": "red",
                    "alpha": 0.7,
                },
            )

        for axis in axes:
            axis.axis("off")

        fig.tight_layout()

        fig.savefig(
            output_dir / f"instance_{index:03d}.png",
            dpi=140,
        )

        plt.close(fig)


# ======================================================================
# TRAINING CURVE
# ======================================================================

def plot_training_history(
    history,
    output_path,
):

    plt.figure(figsize=(9, 5))

    plt.plot(
        history["train_loss"],
        marker="o",
        label="Training loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Mask R-CNN Training")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# ======================================================================
# NUMERICAL TESTS
# ======================================================================

def run_tests():

    print("\n" + "=" * 70)
    print("LESSON 35 — NUMERICAL TESTS")
    print("=" * 70)

    # Test 1: Polygon mask.
    mask = segmentation_to_mask(
        [[2, 2, 12, 2, 12, 12, 2, 12]],
        20,
        20,
    )

    assert mask.shape == (20, 20)
    assert mask.sum() > 0

    print("Test 1 passed: polygon-to-mask conversion.")

    # Test 2: Empty polygon.
    empty = segmentation_to_mask(
        [],
        20,
        20,
    )

    assert empty.sum() == 0

    print("Test 2 passed: empty segmentation.")

    # Test 3: Mask IoU identical.
    a = np.zeros((10, 10), dtype=np.uint8)
    a[2:8, 2:8] = 1

    assert np.isclose(
        binary_mask_iou(a, a),
        1.0,
    )

    print("Test 3 passed: identical mask IoU.")

    # Test 4: Mask IoU disjoint.
    b = np.zeros((10, 10), dtype=np.uint8)
    b[0:2, 0:2] = 1

    assert np.isclose(
        binary_mask_iou(a, b),
        0.0,
    )

    print("Test 4 passed: disjoint mask IoU.")

    # Test 5: Half-overlapping masks.
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)

    a[:, :6] = 1
    b[:, 4:] = 1

    expected = 20 / 100

    assert np.isclose(
        binary_mask_iou(a, b),
        expected,
    )

    print("Test 5 passed: partial mask IoU.")

    # Test 6: Category mapping.
    assert map_category("Plastic bottle") == 1
    assert map_category("Aluminium can") == 2
    assert map_category("Cardboard") == 3
    assert map_category("Glass jar") == 4

    print("Test 6 passed: TACO class mapping.")

    # Test 7: Bounding-box conversion from mask.
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[3:10, 5:15] = 1

    ys, xs = np.where(mask > 0)

    box = [
        xs.min(),
        ys.min(),
        xs.max() + 1,
        ys.max() + 1,
    ]

    assert box == [5, 3, 15, 10]

    print("Test 7 passed: mask bounding box.")

    # Test 8: Model architecture.
    model = build_mask_rcnn_model()

    assert (
        model.roi_heads.box_predictor.cls_score.out_features
        == NUM_CLASSES + 1
    )

    assert (
        model.roi_heads.mask_predictor.mask_fcn_logits.out_channels
        == NUM_CLASSES + 1
    )

    print("Test 8 passed: Mask R-CNN architecture.")

    # Test 9: Instance evaluation.
    gt_mask = np.zeros((20, 20), dtype=np.uint8)
    gt_mask[2:10, 2:10] = 1

    prediction = {

        "scores": torch.tensor([0.95]),

        "labels": torch.tensor([1]),

        "masks": torch.tensor(
            gt_mask[None, None].astype(np.float32)
        ),

    }

    target = {

        "labels": torch.tensor([1]),

        "masks": torch.tensor(
            gt_mask[None].astype(np.uint8)
        ),

    }

    result = evaluate_instance_predictions(
        [prediction],
        [target],
    )

    assert result["TP"] == 1
    assert result["FP"] == 0
    assert result["FN"] == 0
    assert np.isclose(result["F1"], 1.0)
    assert np.isclose(
        result["mean_matched_mask_IoU"],
        1.0,
    )

    print("Test 9 passed: perfect instance evaluation.")

    print("\nALL TESTS PASSED.\n")


# ======================================================================
# ARGUMENTS
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--taco-root",
        default="./TACO/data",
    )

    parser.add_argument(
        "--output",
        default="./runs/lesson35",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
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
        default=0.005,
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--device",
        default=None,
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

    set_seed(42)

    print("=" * 70)
    print("LESSON 35 — MASK R-CNN INSTANCE SEGMENTATION")
    print("=" * 70)

    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")

    run_tests()

    if args.check_only:

        print("CHECK-ONLY: all tests passed.")

        return

    device = torch.device(
        args.device
        if args.device
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    print(f"Device: {device}")

    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_taco_records(
        args.taco_root,
        args.max_images,
    )

    if len(records) < 2:

        raise RuntimeError(
            "At least two valid images are required."
        )

    # Separate training and validation images.
    split = max(
        1,
        int(0.8 * len(records)),
    )

    split = min(
        split,
        len(records) - 1,
    )

    train_records = records[:split]
    val_records = records[split:]

    print(f"Training images: {len(train_records)}")
    print(f"Validation images: {len(val_records)}")

    train_dataset = TACOMaskDataset(
        train_records,
        augment=True,
    )

    val_dataset = TACOMaskDataset(
        val_records,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    model = build_mask_rcnn_model().to(device)

    optimizer = torch.optim.SGD(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1,
    )

    history = {
        "train_loss": [],
    }

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
        )

        scheduler.step()

        history["train_loss"].append(
            train_loss
        )

        print(
            f"\nEpoch {epoch}/{args.epochs} "
            f"| Mean loss: {train_loss:.5f}"
        )

        if train_loss < best_loss:

            best_loss = train_loss

            torch.save(
                model.state_dict(),
                output_dir / "best_model.pth",
            )

            print("Saved best training-loss checkpoint.")

    with (
        output_dir / "training_history.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            history,
            file,
            indent=2,
        )

    plot_training_history(
        history,
        output_dir / "training_curve.png",
    )

    # Evaluate the best training checkpoint.
    checkpoint = torch.load(
        output_dir / "best_model.pth",
        map_location=device,
    )

    model.load_state_dict(checkpoint)

    predictions, targets, val_records = run_inference(
        model,
        val_loader,
        device,
    )

    metrics = evaluate_instance_predictions(
        predictions,
        targets,
        score_threshold=args.score_threshold,
    )

    print("\nVALIDATION RESULTS")

    for key, value in metrics.items():

        print(f"{key}: {value}")

    with (
        output_dir / "evaluation.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    visualize_predictions(
        val_records,
        predictions,
        targets,
        output_dir / "predictions",
    )

    print("\nLESSON 35 COMPLETED")

    print(
        f"Results saved to: {output_dir.resolve()}"
    )

    print(
        "\nReminder: this lesson's instance precision, recall, F1, "
        "and matched mask IoU are educational metrics. They are "
        "not COCO mask AP or AP@[0.50:0.95]."
    )


if __name__ == "__main__":
    main()