"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 36 — YOLO INSTANCE SEGMENTATION
======================================================================

FILENAME:
    lesson36_yolo_instance_segmentation.py

IMPORTANT COURSE RULES
----------------------
1. Real TACO photographs only.
2. No pretrained weights.
3. Randomly initialize the YOLO segmentation architecture.
4. Convert real COCO polygon annotations into YOLO segmentation format.
5. Train, validate, and visualize predictions.
6. Include numerical tests.

NOTE:
    YOLO's internal architecture and loss implementation are provided
    by Ultralytics. This lesson focuses on understanding the design,
    dataset preparation, training workflow, and evaluation.

======================================================================
1. YOLO INSTANCE SEGMENTATION
======================================================================

YOLO segmentation performs detection and segmentation together.

Input image
    |
    v
Backbone
    |
    v
Neck / Feature Pyramid
    |
    v
Segmentation Head
    |
    +-------------------------+
    |                         |
    v                         v
Detection predictions    Mask predictions
    |                         |
    v                         v
Bounding boxes           Prototype masks
Class scores              Mask coefficients
    |                         |
    +------------+------------+
                 |
                 v
          Instance masks

======================================================================
2. PROTOTYPE MASK CONCEPT
======================================================================

Instead of predicting a completely independent large mask for every
object, prototype-based segmentation creates shared mask prototypes.

Let:

    P = prototype mask tensor

    P shape = [K, H, W]

For a particular object:

    c = mask coefficient vector

    c shape = [K]

The predicted mask logits are:

    M = sum(c[k] * P[k])

Or, in matrix form:

    M = c @ P

The mask is then resized to the object region and thresholded.

This makes mask generation more efficient.

======================================================================
3. YOLO VS MASK R-CNN
======================================================================

Mask R-CNN:

    Backbone
       |
       v
    Region Proposal Network
       |
       v
    RoI Align
       |
       +------> Box head
       |
       +------> Mask head

YOLO segmentation:

    Backbone
       |
       v
    Feature fusion
       |
       v
    Unified detection + segmentation head

YOLO avoids the separate region-proposal and RoI processing stages.

======================================================================
4. YOLO SEGMENTATION LABEL FORMAT
======================================================================

One text file per image.

One line per object.

Format:

    class_id x1 y1 x2 y2 x3 y3 ...

Coordinates are normalized:

    x_normalized = x / image_width

    y_normalized = y / image_height

Example:

    0 0.12 0.15 0.30 0.15 0.32 0.40 0.10 0.42

This represents one plastic object.

Class IDs start at zero.

======================================================================
5. TACO DATA CONVERSION
======================================================================

Source:

    TACO/data/annotations.json

Destination:

    runs/lesson36/dataset/

        images/
            train/
            val/

        labels/
            train/
            val/

        taco_yolo.yaml

COCO annotations contain polygons.

We rasterize the polygons and extract the largest external contour
to generate one YOLO polygon per annotated object.

This is an approximation:
    - Small disconnected polygon components may be discarded.
    - Interior holes are not represented.
    - Polygon simplification can remove fine details.

For production datasets, preserve full polygon geometry whenever
possible.

======================================================================
6. TRAINING FROM RANDOM INITIALIZATION
======================================================================

    model = YOLO("yolo26n-seg.yaml")

    model.train(
        data="taco_yolo.yaml",
        pretrained=False,
    )

Important:

    pretrained=False

This prevents loading pretrained weights.

The model architecture is still a known YOLO segmentation design.

======================================================================
7. EVALUATION
======================================================================

Ultralytics segmentation validation provides:

    Box mAP50
    Box mAP50-95
    Mask mAP50
    Mask mAP50-95

Mask mAP evaluates predicted instance masks against ground truth.

======================================================================
8. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    RGB camera
       |
       v
    YOLO segmentation
       |
       v
    Individual object masks
       |
       v
    Waste classification
       |
       v
    Object center and grasp region
       |
       v
    Camera-to-robot transformation
       |
       v
    ACT manipulation policy

For overlapping waste objects, instance masks help distinguish
individual objects.

A 2D mask does not directly provide a reliable 3D grasp point.
Depth sensing and camera-to-robot calibration remain necessary.

======================================================================
9. INSTALLATION
======================================================================

    pip install -U ultralytics numpy opencv-python matplotlib pillow pyyaml

======================================================================
10. RUN
======================================================================

Prepare dataset and inspect annotations:

    python lesson36_yolo_instance_segmentation.py --prepare-only

Run tests:

    python lesson36_yolo_instance_segmentation.py --check-only

Train:

    python lesson36_yolo_instance_segmentation.py \
        --epochs 50 \
        --imgsz 640 \
        --batch 8 \
        --device 0

For a smaller experiment:

    python lesson36_yolo_instance_segmentation.py \
        --epochs 5 \
        --max-images 200 \
        --batch 4 \
        --device 0

======================================================================
OUTPUT
======================================================================

    runs/lesson36/
        dataset/
        weights/
        results/
        predictions/

NEXT:
    Lesson 37 — Vision Transformer From Scratch.
======================================================================
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


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

CATEGORY_KEYWORDS = {

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

    "glass": [
        "glass",
        "glass bottle",
        "glass jar",
    ],
}


# ======================================================================
# 1. REPRODUCIBILITY
# ======================================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)


# ======================================================================
# 2. CLASS MAPPING
# ======================================================================

def map_category(name):

    name = (
        str(name)
        .lower()
        .replace("_", " ")
        .replace("/", " ")
    )

    # More specific material terms are checked first.
    for class_name in [
        "glass",
        "metal",
        "paper_cardboard",
        "plastic",
    ]:

        keywords = CATEGORY_KEYWORDS[class_name]

        if any(
            keyword in name
            for keyword in keywords
        ):

            return CLASS_NAMES.index(class_name)

    return CLASS_NAMES.index("other")


# ======================================================================
# 3. COCO POLYGON TO BINARY MASK
# ======================================================================

def polygon_segmentation_to_mask(
    segmentation,
    height,
    width,
):

    """
    Convert COCO polygon segmentation to a binary mask.

    COCO polygon format:

        [
            [x1, y1, x2, y2, ...],
            [x1, y1, x2, y2, ...],
        ]

    RLE annotations are intentionally skipped in this lesson.
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
# 4. MASK TO YOLO POLYGON
# ======================================================================

def mask_to_yolo_polygon(
    mask,
    epsilon_ratio=0.002,
):

    """
    Convert a binary instance mask to one normalized polygon.

    Steps:

        1. Extract external contours.
        2. Select the largest contour.
        3. Simplify the contour.
        4. Normalize coordinates to [0, 1].

    Returns:
        Nx2 float32 polygon, or None.
    """

    height, width = mask.shape

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return None

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    if cv2.contourArea(contour) < 2:
        return None

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    epsilon = max(
        0.5,
        epsilon_ratio * perimeter,
    )

    simplified = cv2.approxPolyDP(
        contour,
        epsilon,
        True,
    )

    points = simplified.reshape(
        -1,
        2,
    ).astype(np.float32)

    if len(points) < 3:

        points = contour.reshape(
            -1,
            2,
        ).astype(np.float32)

    if len(points) < 3:
        return None

    points[:, 0] /= max(width, 1)
    points[:, 1] /= max(height, 1)

    points = np.clip(
        points,
        0.0,
        1.0,
    )

    return points


# ======================================================================
# 5. YOLO LABEL LINE
# ======================================================================

def create_yolo_segmentation_line(
    class_id,
    polygon,
):

    if polygon is None:
        raise ValueError("Polygon cannot be None.")

    if len(polygon) < 3:
        raise ValueError("Polygon needs at least 3 points.")

    if not 0 <= class_id < NUM_CLASSES:

        raise ValueError(
            f"Invalid class ID: {class_id}"
        )

    values = [
        str(int(class_id))
    ]

    for x, y in polygon:

        values.append(
            f"{float(x):.6f}"
        )

        values.append(
            f"{float(y):.6f}"
        )

    return " ".join(values)


# ======================================================================
# 6. PATH RESOLUTION
# ======================================================================

def resolve_image_path(root, filename):

    filename = Path(str(filename))

    candidates = [
        root / filename,
    ]

    for index, part in enumerate(filename.parts):

        if part.startswith("batch_"):

            candidates.append(
                root.joinpath(
                    *filename.parts[index:]
                )
            )

            break

    for candidate in candidates:

        if candidate.is_file():

            return candidate.resolve()

    return None


# ======================================================================
# 7. LOAD COCO JSON
# ======================================================================

def load_coco_json(root):

    annotation_path = (
        Path(root) / "annotations.json"
    )

    if not annotation_path.exists():

        raise FileNotFoundError(
            f"Missing annotation file: {annotation_path}"
        )

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ======================================================================
# 8. PREPARE YOLO DATASET
# ======================================================================

def prepare_dataset(
    taco_root,
    output_dir,
    max_images=None,
    val_ratio=0.2,
):

    """
    Convert TACO annotations into YOLO segmentation format.

    Images are copied into the generated dataset directory.

    This avoids broken relative paths when training from a different
    working directory.
    """

    taco_root = Path(taco_root)

    output_dir = Path(output_dir)

    data = load_coco_json(
        taco_root
    )

    category_mapping = {

        int(category["id"]): map_category(
            category["name"]
        )

        for category in data["categories"]

    }

    annotations_by_image = {}

    for annotation in data["annotations"]:

        image_id = int(
            annotation["image_id"]
        )

        annotations_by_image.setdefault(
            image_id,
            [],
        ).append(annotation)

    candidates = []

    for image_info in data["images"]:

        image_id = int(
            image_info["id"]
        )

        image_path = resolve_image_path(
            taco_root,
            image_info["file_name"],
        )

        if image_path is None:
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        height, width = image.shape[:2]

        labels = []

        for annotation in annotations_by_image.get(
            image_id,
            [],
        ):

            category_id = int(
                annotation["category_id"]
            )

            if category_id not in category_mapping:
                continue

            segmentation = annotation.get(
                "segmentation",
                [],
            )

            mask = polygon_segmentation_to_mask(
                segmentation,
                height,
                width,
            )

            if mask.sum() == 0:
                continue

            polygon = mask_to_yolo_polygon(
                mask
            )

            if polygon is None:
                continue

            class_id = category_mapping[
                category_id
            ]

            labels.append(
                create_yolo_segmentation_line(
                    class_id,
                    polygon,
                )
            )

        if not labels:
            continue

        candidates.append({

            "image_id": image_id,

            "image_path": image_path,

            "labels": labels,

        })

    random.Random(42).shuffle(
        candidates
    )

    if max_images is not None:

        candidates = candidates[:max_images]

    if len(candidates) < 2:

        raise RuntimeError(
            "Need at least two images with polygon annotations."
        )

    val_count = max(
        1,
        int(len(candidates) * val_ratio),
    )

    val_count = min(
        val_count,
        len(candidates) - 1,
    )

    val_records = candidates[:val_count]

    train_records = candidates[val_count:]

    # Create YOLO dataset directory structure.
    for split in [
        "train",
        "val",
    ]:

        (output_dir / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

        (output_dir / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

    def write_split(records, split):

        for record in records:

            image_id = record["image_id"]

            source = record["image_path"]

            suffix = source.suffix.lower()

            if suffix not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
            ]:

                suffix = ".jpg"

            filename = (
                f"taco_{image_id:06d}{suffix}"
            )

            destination_image = (
                output_dir
                / "images"
                / split
                / filename
            )

            destination_label = (
                output_dir
                / "labels"
                / split
                / f"taco_{image_id:06d}.txt"
            )

            shutil.copy2(
                source,
                destination_image,
            )

            destination_label.write_text(
                "\n".join(record["labels"]) + "\n",
                encoding="utf-8",
            )

    write_split(
        train_records,
        "train",
    )

    write_split(
        val_records,
        "val",
    )

    # Dataset YAML.
    yaml_lines = [

        f"path: {output_dir.resolve()}",

        "train: images/train",

        "val: images/val",

        "names:",

    ]

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        yaml_lines.append(
            f"  {class_id}: {class_name}"
        )

    yaml_path = (
        output_dir / "taco_yolo.yaml"
    )

    yaml_path.write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )

    summary = {

        "total_images": len(candidates),

        "train_images": len(train_records),

        "val_images": len(val_records),

        "num_classes": NUM_CLASSES,

        "class_names": CLASS_NAMES,

        "dataset_yaml": str(yaml_path.resolve()),

    }

    with (
        output_dir / "dataset_summary.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\nDATASET PREPARATION COMPLETED")

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    return yaml_path, train_records, val_records


# ======================================================================
# 9. VISUALIZE REAL ANNOTATIONS
# ======================================================================

def visualize_dataset_annotations(
    taco_root,
    records,
    output_dir,
    max_examples=6,
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, record in enumerate(
        records[:max_examples]
    ):

        image = cv2.imread(
            str(record["image_path"])
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        height, width = image.shape[:2]

        fig, axis = plt.subplots(
            figsize=(10, 8)
        )

        axis.imshow(image)

        for line in record["labels"]:

            values = line.split()

            class_id = int(
                values[0]
            )

            points = np.asarray(
                values[1:],
                dtype=np.float32,
            ).reshape(-1, 2)

            points[:, 0] *= width
            points[:, 1] *= height

            axis.fill(
                points[:, 0],
                points[:, 1],
                alpha=0.25,
                label=CLASS_NAMES[class_id],
            )

            axis.plot(
                np.append(
                    points[:, 0],
                    points[0, 0],
                ),
                np.append(
                    points[:, 1],
                    points[0, 1],
                ),
                linewidth=1.5,
            )

        axis.set_title(
            f"TACO image {record['image_id']} — Real annotations"
        )

        axis.axis("off")

        handles, labels = axis.get_legend_handles_labels()

        if handles:

            unique = dict(
                zip(labels, handles)
            )

            axis.legend(
                unique.values(),
                unique.keys(),
                loc="upper right",
            )

        fig.tight_layout()

        fig.savefig(
            output_dir / f"annotation_{index:03d}.png",
            dpi=140,
        )

        plt.close(fig)


# ======================================================================
# 10. TRAIN YOLO SEGMENTATION
# ======================================================================

def train_yolo(
    dataset_yaml,
    output_dir,
    model_config,
    epochs,
    imgsz,
    batch,
    workers,
    device,
):

    try:

        from ultralytics import YOLO

    except ImportError as error:

        raise ImportError(
            "Install Ultralytics first:\n"
            "pip install -U ultralytics"
        ) from error

    print("\nBUILDING YOLO SEGMENTATION MODEL")

    print(f"Architecture: {model_config}")

    print("Initialization: RANDOM")
    print("Pretrained weights: DISABLED")

    # YAML model definition builds a fresh model.
    model = YOLO(
        model_config
    )

    results = model.train(

        data=str(dataset_yaml),

        epochs=epochs,

        imgsz=imgsz,

        batch=batch,

        workers=workers,

        device=device,

        pretrained=False,

        project=str(output_dir),

        name="yolo_segmentation",

        exist_ok=True,

        seed=42,

        deterministic=True,

        plots=True,

        save=True,

        patience=20,

        optimizer="auto",

    )

    best_weights = (
        Path(results.save_dir)
        / "weights"
        / "best.pt"
    )

    if not best_weights.exists():

        raise FileNotFoundError(
            f"Training checkpoint not found: {best_weights}"
        )

    print(
        f"\nBest checkpoint: {best_weights}"
    )

    return best_weights


# ======================================================================
# 11. VALIDATION
# ======================================================================

def validate_yolo(
    weights,
    dataset_yaml,
    imgsz,
    batch,
    device,
):

    from ultralytics import YOLO

    model = YOLO(
        str(weights)
    )

    metrics = model.val(

        data=str(dataset_yaml),

        imgsz=imgsz,

        batch=batch,

        device=device,

        plots=True,

        verbose=True,

    )

    output = {

        "box_map50": float(
            metrics.box.map50
        ),

        "box_map50_95": float(
            metrics.box.map
        ),

        "mask_map50": float(
            metrics.seg.map50
        ),

        "mask_map50_95": float(
            metrics.seg.map
        ),

    }

    print("\nVALIDATION METRICS")

    print(
        json.dumps(
            output,
            indent=2,
        )
    )

    return output


# ======================================================================
# 12. REAL IMAGE PREDICTIONS
# ======================================================================

def predict_validation_images(
    weights,
    dataset_dir,
    output_dir,
    imgsz,
    device,
    confidence,
):

    from ultralytics import YOLO

    model = YOLO(
        str(weights)
    )

    validation_images = (
        Path(dataset_dir)
        / "images"
        / "val"
    )

    results = model.predict(

        source=str(validation_images),

        imgsz=imgsz,

        device=device,

        conf=confidence,

        save=True,

        project=str(output_dir),

        name="predictions",

        exist_ok=True,

        retina_masks=True,

    )

    print(
        f"\nSaved prediction images: {len(results)}"
    )


# ======================================================================
# 13. NUMERICAL TESTS
# ======================================================================

def run_tests():

    print("\n" + "=" * 70)
    print("LESSON 36 — NUMERICAL TESTS")
    print("=" * 70)

    # Test 1: Class mapping.
    assert map_category("Plastic bottle") == 0
    assert map_category("Aluminium can") == 1
    assert map_category("Cardboard") == 2
    assert map_category("Glass jar") == 3

    print("Test 1 passed: category mapping.")

    # Test 2: Polygon mask.
    mask = polygon_segmentation_to_mask(

        [[
            2, 2,
            12, 2,
            12, 12,
            2, 12,
        ]],

        height=20,

        width=20,

    )

    assert mask.shape == (20, 20)
    assert mask.sum() > 0

    print("Test 2 passed: polygon-to-mask conversion.")

    # Test 3: Empty polygon.
    empty = polygon_segmentation_to_mask(
        [],
        20,
        20,
    )

    assert empty.sum() == 0

    print("Test 3 passed: empty polygon.")

    # Test 4: Mask to polygon.
    mask = np.zeros(
        (100, 120),
        dtype=np.uint8,
    )

    mask[20:70, 30:90] = 1

    polygon = mask_to_yolo_polygon(
        mask
    )

    assert polygon is not None
    assert polygon.shape[1] == 2
    assert len(polygon) >= 3

    print("Test 4 passed: mask-to-polygon conversion.")

    # Test 5: Normalized coordinates.
    assert np.all(polygon >= 0.0)
    assert np.all(polygon <= 1.0)

    print("Test 5 passed: normalized polygon coordinates.")

    # Test 6: YOLO label line.
    line = create_yolo_segmentation_line(
        0,
        polygon,
    )

    values = line.split()

    assert int(values[0]) == 0

    assert len(values) >= 7

    assert (len(values) - 1) % 2 == 0

    print("Test 6 passed: YOLO segmentation label format.")

    # Test 7: Invalid class ID.
    try:

        create_yolo_segmentation_line(
            99,
            polygon,
        )

    except ValueError:

        pass

    else:

        raise AssertionError(
            "Invalid class ID must raise ValueError."
        )

    print("Test 7 passed: class ID validation.")

    # Test 8: Small polygon rejection.
    tiny = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    tiny[1, 1] = 1

    result = mask_to_yolo_polygon(
        tiny
    )

    assert result is None

    print("Test 8 passed: tiny contour rejection.")

    print("\nALL TESTS PASSED.\n")


# ======================================================================
# 14. COMMAND LINE
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Lesson 36 — YOLO instance segmentation."
    )

    parser.add_argument(
        "--taco-root",
        default="./TACO/data",
    )

    parser.add_argument(
        "--output",
        default="./runs/lesson36",
    )

    parser.add_argument(
        "--model",
        default="yolo26n-seg.yaml",
        help=(
            "YOLO segmentation architecture YAML. "
            "For older Ultralytics installations, try yolo11n-seg.yaml."
        ),
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
        "--max-images",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        default="0",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    return parser.parse_args()


# ======================================================================
# 15. MAIN
# ======================================================================

def main():

    args = parse_args()

    set_seed(42)

    print("=" * 70)
    print("LESSON 36 — YOLO INSTANCE SEGMENTATION")
    print("=" * 70)

    run_tests()

    if args.check_only:

        print("CHECK-ONLY: all tests passed.")

        return

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_dir = (
        output_dir / "dataset"
    )

    dataset_yaml, train_records, val_records = prepare_dataset(

        taco_root=args.taco_root,

        output_dir=dataset_dir,

        max_images=args.max_images,

    )

    visualize_dataset_annotations(

        args.taco_root,

        val_records,

        output_dir / "annotation_preview",

    )

    print(
        f"\nTraining images: {len(train_records)}"
    )

    print(
        f"Validation images: {len(val_records)}"
    )

    if args.prepare_only:

        print("\nDATASET PREPARATION ONLY.")
        print(f"Dataset YAML: {dataset_yaml}")

        return

    best_weights = train_yolo(

        dataset_yaml=dataset_yaml,

        output_dir=output_dir / "weights",

        model_config=args.model,

        epochs=args.epochs,

        imgsz=args.imgsz,

        batch=args.batch,

        workers=args.workers,

        device=args.device,

    )

    metrics = validate_yolo(

        weights=best_weights,

        dataset_yaml=dataset_yaml,

        imgsz=args.imgsz,

        batch=args.batch,

        device=args.device,

    )

    with (
        output_dir / "evaluation.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    predict_validation_images(

        weights=best_weights,

        dataset_dir=dataset_dir,

        output_dir=output_dir,

        imgsz=args.imgsz,

        device=args.device,

        confidence=args.confidence,

    )

    print("\nLESSON 36 COMPLETED")

    print(
        f"Results directory: {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()