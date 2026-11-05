"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 32 — ANCHOR-FREE OBJECT DETECTION WITH FCOS
======================================================================

Filename:
    lesson32_anchor_free_object_detection.py

This lesson trains a randomly initialized FCOS detector on REAL TACO
waste photographs. It does not use pretrained detector or backbone
weights and does not create synthetic training images.

FCOS predicts, at each feature-map location:
    l = distance to left box edge
    t = distance to top box edge
    r = distance to right box edge
    b = distance to bottom box edge

For a feature-map location (x, y):
    x1 = x - l
    y1 = y - t
    x2 = x + r
    y2 = y + b

FCOS also predicts:
    - class scores
    - box distances
    - centerness

Centerness reduces the influence of predictions from locations far
from an object's center.

Dataset:
    TACO/data/annotations.json
    TACO/data/batch_1/...
    TACO/data/batch_2/...

Five broad classes:
    1 plastic
    2 metal
    3 paper_cardboard
    4 glass
    5 other

The category mapping is heuristic. Inspect:
    runs/lesson32/category_mapping.json

Install:
    pip install -U torch torchvision opencv-python matplotlib numpy pillow

Quick checks:
    python lesson32_anchor_free_object_detection.py --check-only

Small pipeline test:
    python lesson32_anchor_free_object_detection.py \
        --epochs 2 --max-train-images 100 --max-val-images 30 \
        --batch 2 --device cuda

Larger experiment:
    python lesson32_anchor_free_object_detection.py \
        --epochs 50 --batch 2 --device cuda

Low-memory experiment:
    python lesson32_anchor_free_object_detection.py \
        --epochs 20 --batch 1 --min-size 384 --max-size 640 \
        --device cuda

Outputs:
    runs/lesson32/
        best_model.pth
        last_model.pth
        category_mapping.json
        training_history.json
        evaluation.json
        training_curves.png
        predictions/

Metrics in this lesson are custom class-aware IoU matching metrics
(precision, recall, F1, mean matched IoU), NOT COCO mAP.

Robotics connection:
    RGB camera -> FCOS detections -> object localization ->
    camera/robot coordinate transformation -> ACT manipulation policy.

FCOS does not directly estimate depth, grasp pose, or robot commands.
======================================================================
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fcos_resnet50_fpn


CLASS_NAMES = [
    "plastic",
    "metal",
    "paper_cardboard",
    "glass",
    "other",
]
NUM_CLASSES = len(CLASS_NAMES)

# Specific material terms are checked before generic terms such as
# "bottle", because a glass bottle must not be mapped to plastic.
CLASS_KEYWORDS = {
    "glass": [
        "glass", "glass bottle", "glass jar",
    ],
    "metal": [
        "metal", "aluminium", "aluminum", "tin", "can",
        "foil", "steel",
    ],
    "paper_cardboard": [
        "paper", "cardboard", "carton", "newspaper",
        "magazine", "receipt", "tissue", "paperboard",
    ],
    "plastic": [
        "plastic", "bottle", "wrapper", "bag", "packaging",
        "polystyrene", "styrofoam", "film", "pet",
    ],
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def map_category(category_name):
    name = str(category_name).lower().replace("_", " ").replace("/", " ")
    for class_name in ("glass", "metal", "paper_cardboard", "plastic"):
        if any(keyword in name for keyword in CLASS_KEYWORDS[class_name]):
            return CLASS_NAMES.index(class_name) + 1
    return CLASS_NAMES.index("other") + 1


def resolve_image_path(root, file_name):
    root = Path(root)
    candidate_name = Path(str(file_name))
    candidates = []

    if candidate_name.is_absolute():
        candidates.append(candidate_name)

    candidates.append(root / candidate_name)

    for index, part in enumerate(candidate_name.parts):
        if part.startswith("batch_"):
            candidates.append(root.joinpath(*candidate_name.parts[index:]))
            break

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None


def load_taco_annotations(taco_root):
    taco_root = Path(taco_root)
    annotation_path = taco_root / "annotations.json"

    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Cannot find {annotation_path}. "
            "Expected the TACO data directory containing annotations.json."
        )

    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    categories = {
        int(item["id"]): str(item["name"])
        for item in data["categories"]
    }
    category_mapping = {
        category_id: map_category(name)
        for category_id, name in categories.items()
    }

    images = {
        int(item["id"]): item
        for item in data["images"]
    }

    annotations_by_image = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    mapping_json = {
        str(category_id): {
            "original_category": categories[category_id],
            "mapped_class_id": category_mapping[category_id],
            "mapped_class_name": CLASS_NAMES[category_mapping[category_id] - 1],
        }
        for category_id in categories
    }

    return images, annotations_by_image, category_mapping, mapping_json


def valid_coco_box(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None

    try:
        x, y, box_width, box_height = map(float, box)
    except (TypeError, ValueError):
        return None

    if not np.all(np.isfinite([x, y, box_width, box_height])):
        return None

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + box_width)
    y2 = min(float(height), y + box_height)

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def build_records(taco_root, max_images=None):
    images, annotations_by_image, category_mapping, mapping_json = (
        load_taco_annotations(taco_root)
    )

    records = []

    for image_id, image_info in images.items():
        image_path = resolve_image_path(
            taco_root, image_info.get("file_name", "")
        )
        if image_path is None:
            continue

        try:
            with Image.open(image_path) as pil_image:
                actual_width, actual_height = pil_image.size
        except Exception:
            continue

        width = int(image_info.get("width") or actual_width)
        height = int(image_info.get("height") or actual_height)

        # Annotation dimensions should normally agree with the actual
        # image. Use actual dimensions if metadata is missing/invalid.
        if width <= 0 or height <= 0:
            width, height = actual_width, actual_height

        valid_annotations = []
        for annotation in annotations_by_image.get(image_id, []):
            category_id = int(annotation.get("category_id", -1))
            if category_id not in category_mapping:
                continue

            box = valid_coco_box(
                annotation.get("bbox"), width, height
            )
            if box is None:
                continue

            valid_annotations.append({
                "bbox": box,
                "label": category_mapping[category_id],
                "iscrowd": int(annotation.get("iscrowd", 0)),
            })

        records.append({
            "image_id": image_id,
            "path": str(image_path),
            "width": width,
            "height": height,
            "annotations": valid_annotations,
        })

    # Shuffle before truncation so a small debug subset is not simply
    # the first block of image records in the COCO JSON.
    random.Random(42).shuffle(records)

    if max_images is not None:
        records = records[:max_images]

    if len(records) < 2:
        raise RuntimeError(
            f"Only {len(records)} usable images were found. "
            "Check --taco-root and the file_name paths in annotations.json."
        )

    return records, mapping_json


def split_records(records, val_ratio=0.2):
    shuffled = list(records)
    random.Random(42).shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)

    return shuffled[val_count:], shuffled[:val_count]


class TACODetectionDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_bgr = cv2.imread(record["path"])

        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {record['path']}")

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        boxes = []
        labels = []
        iscrowd = []

        for annotation in record["annotations"]:
            box = valid_coco_box(annotation["bbox"], width, height)
            if box is None:
                continue
            boxes.append(box)
            labels.append(annotation["label"])
            iscrowd.append(annotation["iscrowd"])

        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64)
        iscrowd = np.asarray(iscrowd, dtype=np.int64)

        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            if len(boxes):
                old_x1 = boxes[:, 0].copy()
                old_x2 = boxes[:, 2].copy()
                boxes[:, 0] = width - old_x2
                boxes[:, 2] = width - old_x1

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        ).float() / 255.0

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        if len(boxes):
            area = (
                (boxes[:, 2] - boxes[:, 0])
                * (boxes[:, 3] - boxes[:, 1])
            )
        else:
            area = np.zeros((0,), dtype=np.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([record["image_id"]], dtype=torch.int64),
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_fcos_model(min_size=512, max_size=800):
    # weights=None and weights_backbone=None mean random initialization.
    return fcos_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES + 1,
        min_size=min_size,
        max_size=max_size,
    )


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def match_detections(
    prediction, target, confidence=0.3, iou_threshold=0.5
):
    pred_boxes = prediction["boxes"].detach().cpu().numpy()
    pred_labels = prediction["labels"].detach().cpu().numpy()
    pred_scores = prediction["scores"].detach().cpu().numpy()

    gt_boxes = target["boxes"].detach().cpu().numpy()
    gt_labels = target["labels"].detach().cpu().numpy()

    keep = pred_scores >= confidence
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_labels = pred_labels[order]

    used_gt = set()
    tp = 0
    fp = 0
    matched_ious = []

    for pred_box, pred_label in zip(pred_boxes, pred_labels):
        best_iou = 0.0
        best_index = None

        for gt_index, (gt_box, gt_label) in enumerate(
            zip(gt_boxes, gt_labels)
        ):
            if gt_index in used_gt or pred_label != gt_label:
                continue

            current_iou = box_iou(pred_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = gt_index

        if best_index is not None and best_iou >= iou_threshold:
            tp += 1
            used_gt.add(best_index)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(used_gt)
    return {"tp": tp, "fp": fp, "fn": fn, "ious": matched_ious}


def calculate_metrics(tp, fp, fn, matched_ious):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": mean_iou,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def make_grad_scaler(enabled):
    # Support both newer and older PyTorch AMP APIs.
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model, loader, optimizer, device, scaler, use_amp, epoch
):
    model.train()
    total_loss = 0.0
    batches = 0
    start = time.time()

    for step, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            print(f"WARNING: non-finite loss at epoch {epoch}, step {step}; skipped.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        total_loss += float(loss.detach().item())
        batches += 1

        if step % 10 == 0 or step == len(loader):
            print(
                f"Epoch {epoch:03d} | Step {step:04d}/{len(loader):04d} "
                f"| Loss {loss.item():.4f}"
            )

    return total_loss / max(batches, 1), time.time() - start


@torch.no_grad()
def evaluate_model(
    model, loader, device, confidence=0.3, iou_threshold=0.5
):
    model.eval()
    total_tp = total_fp = total_fn = 0
    all_ious = []

    for images, targets in loader:
        images = [image.to(device) for image in images]
        predictions = model(images)

        for prediction, target in zip(predictions, targets):
            result = match_detections(
                prediction,
                target,
                confidence=confidence,
                iou_threshold=iou_threshold,
            )
            total_tp += result["tp"]
            total_fp += result["fp"]
            total_fn += result["fn"]
            all_ious.extend(result["ious"])

    return calculate_metrics(total_tp, total_fp, total_fn, all_ious)


def draw_boxes(image, boxes, labels, scores=None, color=(0, 255, 0)):
    output = image.copy()

    for index, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = np.round(box).astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(output.shape[1] - 1, x2)
        y2 = min(output.shape[0] - 1, y2)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        class_id = int(label)
        if 1 <= class_id <= NUM_CLASSES:
            class_name = CLASS_NAMES[class_id - 1]
        else:
            class_name = f"class_{class_id}"

        text = class_name
        if scores is not None:
            text += f": {float(scores[index]):.2f}"

        cv2.putText(
            output,
            text,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


@torch.no_grad()
def save_predictions(
    model, records, device, output_dir,
    confidence=0.3, max_visualizations=12
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    for index, record in enumerate(records[:max_visualizations]):
        image_bgr = cv2.imread(record["path"])
        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(
            np.ascontiguousarray(image_rgb.transpose(2, 0, 1))
        ).float() / 255.0

        prediction = model([tensor.to(device)])[0]

        pred_boxes = prediction["boxes"].cpu().numpy()
        pred_labels = prediction["labels"].cpu().numpy()
        pred_scores = prediction["scores"].cpu().numpy()
        keep = pred_scores >= confidence

        predicted_image = draw_boxes(
            image_rgb,
            pred_boxes[keep],
            pred_labels[keep],
            pred_scores[keep],
            color=(255, 0, 0),
        )

        gt_boxes = np.asarray(
            [item["bbox"] for item in record["annotations"]],
            dtype=np.float32,
        ).reshape(-1, 4)
        gt_labels = np.asarray(
            [item["label"] for item in record["annotations"]],
            dtype=np.int64,
        )

        ground_truth_image = draw_boxes(
            image_rgb, gt_boxes, gt_labels, color=(0, 255, 0)
        )

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        axes[0].imshow(ground_truth_image)
        axes[0].set_title("Ground Truth — Green")
        axes[0].axis("off")
        axes[1].imshow(predicted_image)
        axes[1].set_title("FCOS Predictions — Red")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"prediction_{index:03d}.png", dpi=130)
        plt.close(fig)

    print(f"Saved prediction visualizations to: {output_dir}")


def save_training_curves(history, output_path):
    if not history:
        return

    epochs = [item["epoch"] for item in history]
    losses = [item["loss"] for item in history]
    precisions = [item["precision"] for item in history]
    recalls = [item["recall"] for item in history]
    f1_scores = [item["f1"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, losses, marker="o")
    axes[0].set_title("FCOS Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(epochs, precisions, marker="o", label="Precision")
    axes[1].plot(epochs, recalls, marker="o", label="Recall")
    axes[1].plot(epochs, f1_scores, marker="o", label="F1")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_tests():
    print("\n" + "=" * 68)
    print("LESSON 32 — NUMERICAL TESTS")
    print("=" * 68)

    assert np.isclose(box_iou([10, 10, 50, 50], [10, 10, 50, 50]), 1.0)
    print("Test 1 passed: identical-box IoU.")

    assert np.isclose(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
    print("Test 2 passed: disjoint-box IoU.")

    assert np.isclose(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1.0 / 3.0)
    print("Test 3 passed: partial-overlap IoU.")

    assert np.allclose(
        valid_coco_box([10, 20, 30, 40], 100, 100),
        [10, 20, 40, 60],
    )
    print("Test 4 passed: COCO xywh to xyxy.")

    assert np.allclose(
        valid_coco_box([-5, -5, 20, 20], 100, 100),
        [0, 0, 15, 15],
    )
    print("Test 5 passed: bounding-box clipping.")

    assert valid_coco_box([10, 10, 0, 20], 100, 100) is None
    assert valid_coco_box([10, 10, -5, 20], 100, 100) is None
    print("Test 6 passed: invalid bounding-box rejection.")

    assert map_category("Plastic bottle") == 1
    assert map_category("Aluminium can") == 2
    assert map_category("Cardboard") == 3
    assert map_category("Glass bottle") == 4
    assert map_category("Unknown object") == 5
    print("Test 7 passed: waste category mapping.")

    metrics = calculate_metrics(10, 0, 0, [1.0] * 10)
    assert np.isclose(metrics["precision"], 1.0)
    assert np.isclose(metrics["recall"], 1.0)
    assert np.isclose(metrics["f1"], 1.0)
    assert np.isclose(metrics["mean_matched_iou"], 1.0)
    print("Test 8 passed: perfect detection metrics.")

    metrics = calculate_metrics(5, 5, 5, [0.6] * 5)
    assert np.isclose(metrics["precision"], 0.5)
    assert np.isclose(metrics["recall"], 0.5)
    assert np.isclose(metrics["f1"], 0.5)
    print("Test 9 passed: precision/recall/F1.")

    metrics = calculate_metrics(0, 0, 0, [])
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    print("Test 10 passed: empty metric handling.")

    prediction = {
        "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
        "labels": torch.tensor([2]),
        "scores": torch.tensor([0.99]),
    }
    target = {
        "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
        "labels": torch.tensor([1]),
    }

    result = match_detections(prediction, target, 0.1, 0.5)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)
    print("Test 11 passed: class-aware matching.")

    prediction["labels"] = torch.tensor([1])
    result = match_detections(prediction, target, 0.1, 0.5)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)
    print("Test 12 passed: correct detection matching.")

    model = build_fcos_model(min_size=256, max_size=384)
    assert model.head.classification_head.num_classes == NUM_CLASSES + 1
    print("Test 13 passed: FCOS architecture and class count.")

    print("\nALL TESTS PASSED.\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lesson 32 — Anchor-Free FCOS on real TACO images."
    )
    parser.add_argument("--taco-root", default="./TACO/data")
    parser.add_argument("--output", default="./runs/lesson32")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--min-size", type=int, default=512)
    parser.add_argument("--max-size", type=int, default=800)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--weights", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    print("=" * 68)
    print("LESSON 32 — ANCHOR-FREE OBJECT DETECTION WITH FCOS")
    print("=" * 68)

    if args.min_size <= 0 or args.max_size <= 0:
        raise ValueError("Image sizes must be positive.")
    if args.min_size > args.max_size:
        raise ValueError("--min-size cannot exceed --max-size.")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch < 1:
        raise ValueError("--batch must be at least 1.")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    print(f"Device: {device}")
    print(f"Classes: {CLASS_NAMES}")
    print("Pretrained weights: DISABLED")

    run_tests()

    print("\nLoading real TACO dataset...")
    records, mapping_json = build_records(
        args.taco_root, max_images=args.max_images
    )
    train_records, val_records = split_records(records, args.val_ratio)

    if args.max_train_images is not None:
        train_records = train_records[:args.max_train_images]
    if args.max_val_images is not None:
        val_records = val_records[:args.max_val_images]

    if not train_records or not val_records:
        raise RuntimeError(
            "Training or validation set is empty. Increase the image "
            "limits or adjust --max-images and --val-ratio."
        )

    train_ids = {record["image_id"] for record in train_records}
    val_ids = {record["image_id"] for record in val_records}
    if not train_ids.isdisjoint(val_ids):
        raise RuntimeError("Train/validation image leakage detected.")

    print(f"Usable images: {len(records)}")
    print(f"Training images: {len(train_records)}")
    print(f"Validation images: {len(val_records)}")
    print(
        "Training objects:",
        sum(len(record["annotations"]) for record in train_records),
    )
    print(
        "Validation objects:",
        sum(len(record["annotations"]) for record in val_records),
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "category_mapping.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(mapping_json, file, indent=2)

    if args.check_only:
        print("\nCHECK-ONLY MODE: dataset and tests passed; no training started.")
        return

    train_dataset = TACODetectionDataset(train_records, augment=True)
    val_dataset = TACODetectionDataset(val_records, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    print("\nBuilding randomly initialized FCOS model...")
    model = build_fcos_model(args.min_size, args.max_size).to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count:,}")

    if args.weights:
        checkpoint = torch.load(args.weights, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)
        print(f"Loaded weights: {args.weights}")

    if args.skip_training and not args.weights:
        raise ValueError("--skip-training requires --weights.")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, args.epochs // 3),
        gamma=0.1,
    )

    use_amp = device.type == "cuda"
    scaler = make_grad_scaler(use_amp)

    history = []
    best_f1 = -1.0

    if not args.skip_training:
        print("\nStarting FCOS training...")
        for epoch in range(1, args.epochs + 1):
            loss, elapsed = train_one_epoch(
                model, train_loader, optimizer,
                device, scaler, use_amp, epoch
            )

            metrics = evaluate_model(
                model,
                val_loader,
                device,
                confidence=args.confidence,
                iou_threshold=args.iou_threshold,
            )
            scheduler.step()

            row = {
                "epoch": epoch,
                "loss": loss,
                **metrics,
                "seconds": elapsed,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)

            print(
                f"\nEpoch {epoch}/{args.epochs}\n"
                f"  Loss:      {loss:.4f}\n"
                f"  Precision: {metrics['precision']:.4f}\n"
                f"  Recall:    {metrics['recall']:.4f}\n"
                f"  F1:        {metrics['f1']:.4f}\n"
                f"  Mean IoU:  {metrics['mean_matched_iou']:.4f}\n"
                f"  Time:      {elapsed:.1f} sec"
            )

            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch,
                "class_names": CLASS_NAMES,
                "history": history,
                "model_name": "FCOS ResNet50 FPN",
                "pretrained": False,
            }
            torch.save(checkpoint, output_dir / "last_model.pth")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(checkpoint, output_dir / "best_model.pth")
                print("Saved best_model.pth")

            with (output_dir / "training_history.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(history, file, indent=2)

        save_training_curves(
            history, output_dir / "training_curves.png"
        )
    else:
        print("Training skipped; evaluating the supplied checkpoint.")

    print("\nFinal validation evaluation...")
    final_metrics = evaluate_model(
        model,
        val_loader,
        device,
        confidence=args.confidence,
        iou_threshold=args.iou_threshold,
    )

    print("\nFINAL RESULTS")
    for key, value in final_metrics.items():
        print(f"{key:22s}: {value}")

    with (output_dir / "evaluation.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(final_metrics, file, indent=2)

    save_predictions(
        model,
        val_records,
        device,
        output_dir / "predictions",
        confidence=args.confidence,
    )

    print("\nLESSON 32 COMPLETED")
    print(f"Output directory: {output_dir.resolve()}")
    print("Note: custom IoU metrics are not COCO mAP.")


if __name__ == "__main__":
    main()