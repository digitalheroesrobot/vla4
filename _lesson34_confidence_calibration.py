"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 34 — CONFIDENCE CALIBRATION FOR OBJECT DETECTION
======================================================================

FILENAME:
    lesson34_confidence_calibration.py

LEARNING OBJECTIVES
-------------------
1. Understand classification confidence versus detection confidence.
2. Understand why neural networks can be overconfident.
3. Implement Expected Calibration Error (ECE).
4. Implement Maximum Calibration Error (MCE).
5. Implement the Brier score.
6. Create reliability diagrams.
7. Implement temperature scaling.
8. Evaluate calibration before and after scaling.
9. Understand confidence calibration in robotic perception.

IMPORTANT:
    This lesson uses real TACO waste photographs and annotations.

    No pretrained detector is used.

    A randomly initialized FCOS detector can optionally be loaded
    from Lesson 32 if you have trained one.

    For meaningful calibration, use a trained detector and a
    sufficiently large validation dataset.

----------------------------------------------------------------------
1. WHAT IS CONFIDENCE CALIBRATION?
----------------------------------------------------------------------

Suppose a detector produces these predictions:

    Prediction A: confidence = 0.95, correct
    Prediction B: confidence = 0.90, correct
    Prediction C: confidence = 0.85, incorrect
    Prediction D: confidence = 0.80, correct

A detector is calibrated when predictions assigned confidence p
are correct approximately p fraction of the time.

For example:

    Among predictions with confidence around 0.80,
    approximately 80% should be correct.

High confidence does not automatically mean high reliability.

----------------------------------------------------------------------
2. RELIABILITY DIAGRAM
----------------------------------------------------------------------

Divide confidence scores into bins.

Example:

    Bin 1: [0.0, 0.1)
    Bin 2: [0.1, 0.2)
    ...
    Bin 10: [0.9, 1.0]

For each bin:

    mean_confidence = mean(prediction confidence)

    accuracy = correct predictions / total predictions

A perfectly calibrated model lies on:

    accuracy = confidence

----------------------------------------------------------------------
3. EXPECTED CALIBRATION ERROR
----------------------------------------------------------------------

ECE measures the weighted difference between accuracy and confidence.

    ECE = sum over bins:

          (number of predictions in bin / total predictions)
          * abs(accuracy - mean_confidence)

Lower ECE is better.

ECE = 0 means perfect calibration for the selected bins.

----------------------------------------------------------------------
4. MAXIMUM CALIBRATION ERROR
----------------------------------------------------------------------

    MCE = max over bins:

          abs(accuracy - mean_confidence)

MCE measures the worst calibration gap.

----------------------------------------------------------------------
5. BRIER SCORE
----------------------------------------------------------------------

For binary correctness labels:

    Brier = mean((confidence - correctness)^2)

where:

    correctness = 1 for correct detection
    correctness = 0 for incorrect detection

Lower is better.

----------------------------------------------------------------------
6. TEMPERATURE SCALING
----------------------------------------------------------------------

Temperature scaling transforms logits:

    calibrated_probability = softmax(logits / T)

where:

    T > 0

Temperature is optimized using validation data.

    T > 1: generally reduces overconfidence.
    T < 1: generally increases confidence.

Temperature scaling preserves the ordering of class logits for a
single prediction, but changes their probability values.

For object detection, this lesson uses a simplified binary
correctness-calibration experiment on matched detections.

It does not modify bounding-box coordinates.

----------------------------------------------------------------------
7. OBJECT DETECTION MATCHING
----------------------------------------------------------------------

A prediction is considered correct when:

    predicted class == ground-truth class
    IoU(predicted box, ground-truth box) >= threshold

Predictions are matched greedily in descending confidence order.

Unmatched predictions are false positives.

Ground-truth objects without matched predictions are false negatives.

For calibration, unmatched predictions are assigned correctness 0.

----------------------------------------------------------------------
8. ROBOTICS APPLICATION
----------------------------------------------------------------------

OpenArm V2 waste sorting:

    Camera
      |
      v
    Object detector
      |
      v
    Confidence calibration
      |
      v
    Decision policy

Example:

    confidence >= 0.90:
        proceed with normal grasp planning

    confidence 0.60 - 0.90:
        request another camera observation

    confidence < 0.60:
        defer sorting or request human review

These are illustrative thresholds, not universal safety limits.

A robot should also consider object localization uncertainty,
grasp feasibility, collision risk, and actuator constraints.

----------------------------------------------------------------------
9. INSTALLATION
----------------------------------------------------------------------

    pip install -U torch torchvision opencv-python matplotlib numpy pillow

----------------------------------------------------------------------
10. RUN
----------------------------------------------------------------------

Run numerical tests and real TACO image/annotation preparation:

    python lesson34_confidence_calibration.py

Evaluate a trained Lesson 32 checkpoint:

    python lesson34_confidence_calibration.py \
        --weights runs/lesson32/best_model.pth \
        --max-images 500

For a larger validation experiment:

    python lesson34_confidence_calibration.py \
        --weights runs/lesson32/best_model.pth \
        --max-images 2000 \
        --confidence 0.05

----------------------------------------------------------------------
OUTPUT
----------------------------------------------------------------------

    runs/lesson34/
        reliability_before.png
        reliability_after.png
        calibration_comparison.png
        calibration_metrics.json
        temperature_scaling.json
        real_detections/

NEXT LESSON:
    Lesson 35 — Mask R-CNN and Instance Segmentation Improvements.
======================================================================
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fcos_resnet50_fpn


CLASS_NAMES = [
    "plastic",
    "metal",
    "paper_cardboard",
    "glass",
    "other",
]

NUM_CLASSES = len(CLASS_NAMES)


# ======================================================================
# 1. REPRODUCIBILITY
# ======================================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# 2. TACO CATEGORY MAPPING
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

    name = str(name).lower().replace("_", " ").replace("/", " ")

    for class_name in [
        "glass",
        "metal",
        "paper_cardboard",
        "plastic",
    ]:

        keywords = CATEGORY_KEYWORDS[class_name]

        if any(keyword in name for keyword in keywords):

            return CLASS_NAMES.index(class_name) + 1

    return CLASS_NAMES.index("other") + 1


# ======================================================================
# 3. LOAD TACO DATASET
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
            f"Missing TACO annotations: {annotation_path}"
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

        boxes = []
        labels = []

        for annotation in annotations_by_image[image_id]:

            category_id = int(annotation["category_id"])

            if category_id not in category_mapping:
                continue

            x, y, w, h = map(
                float,
                annotation["bbox"],
            )

            x1 = max(0.0, x)
            y1 = max(0.0, y)

            x2 = min(float(width), x + w)
            y2 = min(float(height), y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])

            labels.append(
                category_mapping[category_id]
            )

        records.append({

            "image_id": image_id,

            "path": str(image_path),

            "boxes": boxes,

            "labels": labels,

        })

    random.Random(42).shuffle(records)

    if max_images is not None:

        records = records[:max_images]

    if len(records) == 0:

        raise RuntimeError(
            "No usable TACO images found."
        )

    return records


class TACODataset(Dataset):

    def __init__(self, records):

        self.records = records

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image = cv2.imread(record["path"])

        if image is None:

            raise RuntimeError(
                f"Could not read {record['path']}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        tensor = torch.from_numpy(
            np.ascontiguousarray(
                image.transpose(2, 0, 1)
            )
        ).float() / 255.0

        boxes = torch.tensor(
            record["boxes"],
            dtype=torch.float32,
        ).reshape(-1, 4)

        labels = torch.tensor(
            record["labels"],
            dtype=torch.int64,
        )

        target = {

            "boxes": boxes,

            "labels": labels,

        }

        return tensor, target, record


def collate_fn(batch):

    return tuple(zip(*batch))


# ======================================================================
# 4. BOX IoU
# ======================================================================

def box_iou(box_a, box_b):

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])

    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(
        0.0,
        x2 - x1,
    ) * max(
        0.0,
        y2 - y1,
    )

    area_a = max(
        0.0,
        box_a[2] - box_a[0],
    ) * max(
        0.0,
        box_a[3] - box_a[1],
    )

    area_b = max(
        0.0,
        box_b[2] - box_b[0],
    ) * max(
        0.0,
        box_b[3] - box_b[1],
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ======================================================================
# 5. DETECTION MATCHING
# ======================================================================

def match_predictions(
    prediction,
    target,
    confidence_threshold=0.05,
    iou_threshold=0.5,
):

    boxes = prediction["boxes"].detach().cpu().numpy()

    scores = prediction["scores"].detach().cpu().numpy()

    labels = prediction["labels"].detach().cpu().numpy()

    gt_boxes = target["boxes"].detach().cpu().numpy()

    gt_labels = target["labels"].detach().cpu().numpy()

    keep = scores >= confidence_threshold

    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    order = np.argsort(-scores, kind="stable")

    boxes = boxes[order]
    scores = scores[order]
    labels = labels[order]

    used_gt = set()

    correctness = []

    for box, label in zip(boxes, labels):

        best_iou = 0.0
        best_index = None

        for gt_index, (gt_box, gt_label) in enumerate(
            zip(gt_boxes, gt_labels)
        ):

            if gt_index in used_gt:
                continue

            if label != gt_label:
                continue

            overlap = box_iou(box, gt_box)

            if overlap > best_iou:

                best_iou = overlap

                best_index = gt_index

        if (
            best_index is not None
            and best_iou >= iou_threshold
        ):

            correctness.append(1.0)

            used_gt.add(best_index)

        else:

            correctness.append(0.0)

    return scores, np.asarray(correctness, dtype=np.float32)


# ======================================================================
# 6. CALIBRATION METRICS
# ======================================================================

def expected_calibration_error(
    confidences,
    correctness,
    num_bins=15,
):

    confidences = np.asarray(
        confidences,
        dtype=np.float64,
    )

    correctness = np.asarray(
        correctness,
        dtype=np.float64,
    )

    if len(confidences) != len(correctness):

        raise ValueError(
            "Confidence and correctness arrays must have equal length."
        )

    if len(confidences) == 0:

        return 0.0, []

    if not np.all(np.isfinite(confidences)):

        raise ValueError("Confidence contains non-finite values.")

    if not np.all(np.isfinite(correctness)):

        raise ValueError("Correctness contains non-finite values.")

    if np.any((confidences < 0) | (confidences > 1)):

        raise ValueError("Confidence must lie in [0, 1].")

    if np.any((correctness < 0) | (correctness > 1)):

        raise ValueError("Correctness must lie in [0, 1].")

    bin_edges = np.linspace(
        0.0,
        1.0,
        num_bins + 1,
    )

    ece = 0.0

    bins = []

    for index in range(num_bins):

        lower = bin_edges[index]
        upper = bin_edges[index + 1]

        if index == num_bins - 1:

            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )

        else:

            mask = (
                (confidences >= lower)
                & (confidences < upper)
            )

        count = int(np.sum(mask))

        if count == 0:

            continue

        mean_confidence = float(
            np.mean(confidences[mask])
        )

        accuracy = float(
            np.mean(correctness[mask])
        )

        gap = abs(
            accuracy - mean_confidence
        )

        weight = count / len(confidences)

        ece += weight * gap

        bins.append({

            "lower": float(lower),

            "upper": float(upper),

            "count": count,

            "mean_confidence": mean_confidence,

            "accuracy": accuracy,

            "gap": gap,

        })

    return float(ece), bins


def maximum_calibration_error(bins):

    if len(bins) == 0:
        return 0.0

    return float(
        max(item["gap"] for item in bins)
    )


def brier_score(confidences, correctness):

    confidences = np.asarray(
        confidences,
        dtype=np.float64,
    )

    correctness = np.asarray(
        correctness,
        dtype=np.float64,
    )

    if len(confidences) != len(correctness):

        raise ValueError("Array lengths must match.")

    if len(confidences) == 0:
        return 0.0

    return float(
        np.mean(
            (confidences - correctness) ** 2
        )
    )


def calculate_calibration_metrics(
    confidences,
    correctness,
    num_bins=15,
):

    ece, bins = expected_calibration_error(
        confidences,
        correctness,
        num_bins,
    )

    mce = maximum_calibration_error(bins)

    brier = brier_score(
        confidences,
        correctness,
    )

    return {

        "ECE": ece,

        "MCE": mce,

        "Brier": brier,

        "num_predictions": len(confidences),

        "bins": bins,

    }


# ======================================================================
# 7. RELIABILITY DIAGRAM
# ======================================================================

def plot_reliability_diagram(
    confidences,
    correctness,
    output_path,
    title,
    num_bins=15,
):

    _, bins = expected_calibration_error(
        confidences,
        correctness,
        num_bins,
    )

    bin_centers = [
        (item["lower"] + item["upper"]) / 2
        for item in bins
    ]

    accuracies = [
        item["accuracy"]
        for item in bins
    ]

    mean_confidences = [
        item["mean_confidence"]
        for item in bins
    ]

    counts = [
        item["count"]
        for item in bins
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    axes[0].plot(
        [0, 1],
        [0, 1],
        "--",
        label="Perfect calibration",
    )

    axes[0].plot(
        mean_confidences,
        accuracies,
        marker="o",
        linewidth=2,
        label="Model",
    )

    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    axes[0].set_xlabel("Mean confidence")
    axes[0].set_ylabel("Observed accuracy")

    axes[0].set_title(title)

    axes[0].legend()
    axes[0].grid(True)

    axes[1].bar(
        bin_centers,
        counts,
        width=1.0 / num_bins * 0.8,
    )

    axes[1].set_xlabel("Confidence bin")
    axes[1].set_ylabel("Number of predictions")

    axes[1].set_title("Prediction Distribution")

    axes[1].grid(True)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


# ======================================================================
# 8. TEMPERATURE SCALING
# ======================================================================

def binary_logits_from_probabilities(confidences):

    probabilities = torch.as_tensor(
        confidences,
        dtype=torch.float32,
    )

    probabilities = probabilities.clamp(
        1e-6,
        1.0 - 1e-6,
    )

    positive_logits = torch.log(
        probabilities / (1.0 - probabilities)
    )

    logits = torch.stack(
        [
            torch.zeros_like(positive_logits),
            positive_logits,
        ],
        dim=1,
    )

    return logits


def fit_temperature(
    confidences,
    correctness,
    max_iterations=500,
):

    """
    Fit one positive temperature using binary correctness labels.

    This is a calibration experiment over matched detector outputs.
    It does not train the FCOS detector itself.
    """

    if len(confidences) == 0:

        raise ValueError(
            "Cannot fit temperature without predictions."
        )

    logits = binary_logits_from_probabilities(
        confidences
    )

    labels = torch.as_tensor(
        correctness,
        dtype=torch.long,
    )

    # Optimize log(T) so that T remains positive.
    log_temperature = torch.nn.Parameter(
        torch.zeros(1)
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iterations,
        line_search_fn="strong_wolfe",
    )

    def closure():

        optimizer.zero_grad()

        temperature = torch.exp(
            log_temperature
        ).clamp(0.05, 20.0)

        loss = F.cross_entropy(
            logits / temperature,
            labels,
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    temperature = float(
        torch.exp(log_temperature.detach()).clamp(
            0.05,
            20.0,
        ).item()
    )

    return temperature


def apply_temperature(confidences, temperature):

    logits = binary_logits_from_probabilities(
        confidences
    )

    temperature_tensor = torch.tensor(
        temperature,
        dtype=torch.float32,
    )

    probabilities = torch.softmax(
        logits / temperature_tensor,
        dim=1,
    )[:, 1]

    return probabilities.numpy()


# ======================================================================
# 9. FCOS CHECKPOINT LOADING
# ======================================================================

def build_fcos_model():

    return fcos_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES + 1,
    )


def load_fcos_checkpoint(weights_path, device):

    model = build_fcos_model()

    checkpoint = torch.load(
        weights_path,
        map_location=device,
    )

    state_dict = checkpoint.get(
        "model",
        checkpoint,
    )

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model


# ======================================================================
# 10. RUN DETECTOR ON VALIDATION IMAGES
# ======================================================================

@torch.no_grad()
def collect_detector_predictions(
    model,
    loader,
    device,
    confidence_threshold,
    iou_threshold,
):

    all_confidences = []
    all_correctness = []

    visualization_data = []

    model.eval()

    for images, targets, records in loader:

        images = [
            image.to(device)
            for image in images
        ]

        predictions = model(images)

        for image, prediction, target, record in zip(
            images,
            predictions,
            targets,
            records,
        ):

            scores, correctness = match_predictions(
                prediction,
                target,
                confidence_threshold,
                iou_threshold,
            )

            all_confidences.extend(
                scores.tolist()
            )

            all_correctness.extend(
                correctness.tolist()
            )

            visualization_data.append({

                "record": record,

                "prediction": prediction,

                "target": target,

            })

    return (

        np.asarray(
            all_confidences,
            dtype=np.float32,
        ),

        np.asarray(
            all_correctness,
            dtype=np.float32,
        ),

        visualization_data,

    )


# ======================================================================
# 11. SAVE REAL DETECTION EXAMPLES
# ======================================================================

def save_detection_examples(
    visualization_data,
    output_dir,
    max_examples=8,
    confidence_threshold=0.3,
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, item in enumerate(
        visualization_data[:max_examples]
    ):

        record = item["record"]

        prediction = item["prediction"]

        target = item["target"]

        image = cv2.imread(
            record["path"]
        )

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        pred_boxes = prediction["boxes"].cpu().numpy()

        pred_labels = prediction["labels"].cpu().numpy()

        pred_scores = prediction["scores"].cpu().numpy()

        gt_boxes = target["boxes"].numpy()

        gt_labels = target["labels"].numpy()

        pred_keep = pred_scores >= confidence_threshold

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(15, 7),
        )

        axes[0].imshow(image)
        axes[0].set_title("Real TACO Image — Ground Truth")

        for box, label in zip(gt_boxes, gt_labels):

            x1, y1, x2, y2 = box

            axes[0].add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
            )

            axes[0].text(
                x1,
                y1,
                CLASS_NAMES[int(label) - 1],
                color="white",
                fontsize=8,
                bbox={"facecolor": "green", "alpha": 0.7},
            )

        axes[1].imshow(image)
        axes[1].set_title("FCOS Predictions")

        for box, label, score in zip(
            pred_boxes[pred_keep],
            pred_labels[pred_keep],
            pred_scores[pred_keep],
        ):

            x1, y1, x2, y2 = box

            axes[1].add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
            )

            axes[1].text(
                x1,
                y1,
                f"{CLASS_NAMES[int(label) - 1]} {score:.2f}",
                color="white",
                fontsize=8,
                bbox={"facecolor": "red", "alpha": 0.7},
            )

        for axis in axes:
            axis.axis("off")

        fig.tight_layout()

        fig.savefig(
            output_dir / f"detection_{index:03d}.png",
            dpi=130,
        )

        plt.close(fig)


# ======================================================================
# 12. NUMERICAL TESTS
# ======================================================================

def run_tests():

    print("\n" + "=" * 70)
    print("LESSON 34 — NUMERICAL TESTS")
    print("=" * 70)

    # Test 1: Perfect calibration.
    confidence = np.array([
        0.0,
        1.0,
        0.0,
        1.0,
    ])

    correctness = np.array([
        0.0,
        1.0,
        0.0,
        1.0,
    ])

    ece, _ = expected_calibration_error(
        confidence,
        correctness,
        num_bins=10,
    )

    assert np.isclose(ece, 0.0)

    print("Test 1 passed: perfect calibration.")

    # Test 2: ECE detects overconfidence.
    confidence = np.array([
        0.9,
        0.9,
        0.9,
        0.9,
    ])

    correctness = np.array([
        1.0,
        0.0,
        0.0,
        0.0,
    ])

    ece, _ = expected_calibration_error(
        confidence,
        correctness,
        num_bins=10,
    )

    assert ece > 0.0

    print("Test 2 passed: overconfidence detection.")

    # Test 3: MCE.
    _, bins = expected_calibration_error(
        confidence,
        correctness,
        num_bins=10,
    )

    mce = maximum_calibration_error(bins)

    assert np.isclose(mce, 0.65)

    print("Test 3 passed: maximum calibration error.")

    # Test 4: Brier score.
    score = brier_score(
        [0.9, 0.1],
        [1.0, 0.0],
    )

    assert np.isclose(score, 0.01)

    print("Test 4 passed: Brier score.")

    # Test 5: Empty calibration data.
    ece, bins = expected_calibration_error(
        [],
        [],
    )

    assert ece == 0.0
    assert bins == []

    print("Test 5 passed: empty calibration input.")

    # Test 6: Invalid confidence.
    try:

        expected_calibration_error(
            [1.2],
            [1.0],
        )

    except ValueError:

        pass

    else:

        raise AssertionError(
            "Invalid confidence should raise ValueError."
        )

    print("Test 6 passed: confidence range validation.")

    # Test 7: IoU identical boxes.
    assert np.isclose(
        box_iou(
            [0, 0, 10, 10],
            [0, 0, 10, 10],
        ),
        1.0,
    )

    print("Test 7 passed: identical-box IoU.")

    # Test 8: IoU disjoint boxes.
    assert np.isclose(
        box_iou(
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ),
        0.0,
    )

    print("Test 8 passed: disjoint-box IoU.")

    # Test 9: Detection matching.
    prediction = {

        "boxes": torch.tensor([
            [0, 0, 10, 10],
            [30, 30, 40, 40],
        ], dtype=torch.float32),

        "scores": torch.tensor([
            0.95,
            0.80,
        ]),

        "labels": torch.tensor([
            1,
            1,
        ]),

    }

    target = {

        "boxes": torch.tensor([
            [0, 0, 10, 10],
        ], dtype=torch.float32),

        "labels": torch.tensor([
            1,
        ]),

    }

    scores, correctness = match_predictions(
        prediction,
        target,
        confidence_threshold=0.1,
        iou_threshold=0.5,
    )

    assert len(scores) == 2

    assert np.allclose(
        correctness,
        [1.0, 0.0],
    )

    print("Test 9 passed: detection correctness matching.")

    # Test 10: Binary logits.
    logits = binary_logits_from_probabilities(
        [0.5, 0.9]
    )

    assert logits.shape == (2, 2)

    print("Test 10 passed: binary confidence logits.")

    # Test 11: Temperature scaling.
    original = np.array([
        0.9,
        0.8,
        0.7,
    ])

    calibrated = apply_temperature(
        original,
        temperature=2.0,
    )

    assert np.all(calibrated < original)

    print("Test 11 passed: temperature scaling reduces high probabilities.")

    # Test 12: Temperature fitting.
    confidence = np.array([
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
    ])

    correctness = np.array([
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
    ], dtype=np.float32)

    temperature = fit_temperature(
        confidence,
        correctness,
    )

    assert np.isfinite(temperature)
    assert temperature > 0

    print("Test 12 passed: positive temperature fitting.")

    print("\nALL TESTS PASSED.\n")


# ======================================================================
# 13. MAIN
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Lesson 34 — Object detection confidence calibration."
    )

    parser.add_argument(
        "--taco-root",
        default="./TACO/data",
    )

    parser.add_argument(
        "--output",
        default="./runs/lesson34",
    )

    parser.add_argument(
        "--weights",
        default=None,
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
        "--confidence",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--num-bins",
        type=int,
        default=15,
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


def main():

    args = parse_args()

    set_seed(42)

    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("LESSON 34 — CONFIDENCE CALIBRATION")
    print("=" * 70)

    run_tests()

    if args.check_only:

        print("CHECK-ONLY: numerical tests passed.")

        return

    if args.weights is None:

        print(
            "\nNo trained FCOS checkpoint was supplied."
        )

        print(
            "A real TACO photograph will be loaded and visualized, "
            "but calibration metrics require trained detector weights."
        )

        records = load_taco_records(
            args.taco_root,
            max_images=1,
        )

        record = records[0]

        image = cv2.imread(record["path"])

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        plt.figure(figsize=(9, 7))
        plt.imshow(image)
        plt.title("Real TACO Photograph")
        plt.axis("off")
        plt.tight_layout()

        plt.savefig(
            output_dir / "real_taco_example.png",
            dpi=150,
        )

        plt.close()

        print(
            f"Saved: {output_dir / 'real_taco_example.png'}"
        )

        print(
            "\nTo run actual calibration, provide a trained "
            "Lesson 32 FCOS checkpoint with --weights."
        )

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

    records = load_taco_records(
        args.taco_root,
        max_images=args.max_images,
    )

    random.Random(123).shuffle(records)

    val_count = max(
        1,
        int(0.2 * len(records)),
    )

    validation_records = records[:val_count]

    dataset = TACODataset(
        validation_records
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    model = load_fcos_checkpoint(
        args.weights,
        device,
    )

    print(
        f"Validation images: {len(validation_records)}"
    )

    confidences, correctness, visualization_data = (
        collect_detector_predictions(
            model,
            loader,
            device,
            args.confidence,
            args.iou_threshold,
        )
    )

    if len(confidences) == 0:

        raise RuntimeError(
            "No detections passed the confidence threshold. "
            "Try reducing --confidence."
        )

    print(
        f"Matched evaluation candidates: {len(confidences)}"
    )

    # Calibration before temperature scaling.
    before = calculate_calibration_metrics(
        confidences,
        correctness,
        args.num_bins,
    )

    temperature = fit_temperature(
        confidences,
        correctness,
    )

    calibrated_confidences = apply_temperature(
        confidences,
        temperature,
    )

    # Calibration after temperature scaling.
    after = calculate_calibration_metrics(
        calibrated_confidences,
        correctness,
        args.num_bins,
    )

    print("\nCALIBRATION BEFORE")
    print(f"ECE:    {before['ECE']:.5f}")
    print(f"MCE:    {before['MCE']:.5f}")
    print(f"Brier:  {before['Brier']:.5f}")

    print("\nTEMPERATURE")
    print(f"T = {temperature:.5f}")

    print("\nCALIBRATION AFTER")
    print(f"ECE:    {after['ECE']:.5f}")
    print(f"MCE:    {after['MCE']:.5f}")
    print(f"Brier:  {after['Brier']:.5f}")

    plot_reliability_diagram(
        confidences,
        correctness,
        output_dir / "reliability_before.png",
        "Before Temperature Scaling",
        args.num_bins,
    )

    plot_reliability_diagram(
        calibrated_confidences,
        correctness,
        output_dir / "reliability_after.png",
        "After Temperature Scaling",
        args.num_bins,
    )

    comparison = {

        "before": {
            key: value
            for key, value in before.items()
            if key != "bins"
        },

        "after": {
            key: value
            for key, value in after.items()
            if key != "bins"
        },

        "temperature": temperature,

        "iou_threshold": args.iou_threshold,

        "confidence_threshold": args.confidence,

    }

    with (
        output_dir / "calibration_metrics.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            comparison,
            file,
            indent=2,
        )

    with (
        output_dir / "temperature_scaling.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            {
                "temperature": temperature,
                "interpretation": (
                    "Apply this temperature to the binary "
                    "correctness logits used in this experiment."
                ),
            },
            file,
            indent=2,
        )

    save_detection_examples(
        visualization_data,
        output_dir / "real_detections",
    )

    print("\nLESSON 34 COMPLETED")
    print(f"Output: {output_dir.resolve()}")

    print(
        "\nImportant: temperature scaling here calibrates the "
        "probability that a detection is correct under the selected "
        "IoU and class-matching definition. It is not a complete "
        "multiclass or COCO detection-calibration method."
    )


if __name__ == "__main__":
    main()