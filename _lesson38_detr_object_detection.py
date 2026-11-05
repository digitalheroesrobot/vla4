"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 38 — DETR: DETECTION TRANSFORMER FROM SCRATCH
======================================================================

FILENAME:
    lesson38_detr_object_detection.py

COURSE RULES
------------
- No pretrained models.
- No pretrained weights.
- Real TACO waste photographs and COCO annotations.
- Implement the DETR architecture and loss ourselves.
- Use SciPy's linear_sum_assignment for Hungarian matching.
- Include numerical tests, training, evaluation and visualization.

======================================================================
1. WHAT IS DETR?
======================================================================

DETR = DEtection TRansformer.

Traditional object detectors often use:
    - Anchor boxes
    - Region proposals
    - Non-maximum suppression
    - Hand-designed matching rules

DETR takes a different approach.

    Image
      |
      v
    CNN Backbone
      |
      v
    Feature Map
      |
      v
    Transformer Encoder
      |
      v
    Transformer Decoder
      |
      v
    Object Queries
      |
      v
    Class Predictions + Bounding Boxes

DETR predicts a fixed-size set of objects.

For example, with 50 object queries:

    Query 0  -> plastic bottle
    Query 1  -> metal can
    Query 2  -> no object
    Query 3  -> cardboard
    ...
    Query 49 -> no object

The model learns which queries should represent actual objects.

======================================================================
2. CNN BACKBONE
======================================================================

We first extract spatial features from the input image.

Input:

    [B, 3, 128, 128]

Our randomly initialized CNN produces:

    [B, 128, 16, 16]

The spatial feature map contains 256 feature locations.

Unlike a pretrained ResNet, this backbone is built specifically
for this educational implementation.

======================================================================
3. POSITIONAL ENCODING
======================================================================

Transformers do not inherently understand image coordinates.

We add 2D sine/cosine positional encodings.

For a coordinate p and dimension i:

    PE(p, 2i)   = sin(p / 10000^(2i/d))
    PE(p, 2i+1) = cos(p / 10000^(2i/d))

We encode both x and y positions.

This gives the Transformer spatial information.

======================================================================
4. TRANSFORMER ENCODER
======================================================================

The encoder processes all image feature locations.

Self-attention:

    Q = XWq
    K = XWk
    V = XWv

    Attention(Q,K,V) =
        softmax(QK^T / sqrt(d_k)) V

The encoder allows one image location to interact with other
locations throughout the image.

======================================================================
5. OBJECT QUERIES
======================================================================

This is one of DETR's most important ideas.

We create Q learnable object-query vectors:

    ObjectQueries = [q1, q2, ..., qQ]

Each query is a learned representation that attempts to identify
one object.

For this lesson:

    Number of queries = 50

The Transformer decoder processes these queries using:

    1. Query self-attention
    2. Cross-attention with image features
    3. Feed-forward network

The queries compete and cooperate to represent objects.

======================================================================
6. PREDICTION HEADS
======================================================================

Each query produces:

A. Class logits:

    [num_classes + 1]

The extra class represents:

    no-object

B. Bounding box:

    [cx, cy, width, height]

All four coordinates are normalized to [0, 1].

======================================================================
7. HUNGARIAN MATCHING
======================================================================

DETR predicts a fixed number of boxes, but an image may contain
a different number of actual objects.

We must assign predictions to ground-truth objects.

We construct a matching cost:

    Cost =
        lambda_class * classification_cost
      + lambda_bbox  * L1_bbox_cost
      + lambda_giou  * (-GIoU)

Then solve the minimum-cost assignment problem:

    HungarianMatching(Cost)

We use SciPy's linear_sum_assignment.

This produces a one-to-one assignment between predictions and
ground-truth objects.

Unmatched queries are trained as no-object.

======================================================================
8. GENERALIZED IoU
======================================================================

IoU:

    IoU = Intersection / Union

IoU alone provides weak optimization signals when boxes do not
overlap.

Generalized IoU:

    GIoU = IoU - (EnclosingArea - Union) / EnclosingArea

GIoU gives a useful geometric penalty even when boxes do not overlap.

======================================================================
9. DETR LOSS
======================================================================

Total loss:

    L =
        lambda_cls  * L_class
      + lambda_bbox * L_bbox
      + lambda_giou * L_giou

Classification:

    CrossEntropyLoss

Bounding box:

    L1 distance

Geometry:

    1 - GIoU

======================================================================
10. IMPORTANT DIFFERENCE FROM YOLO
======================================================================

YOLO:
    Dense predictions.
    Usually uses confidence filtering and NMS.

DETR:
    Fixed number of object queries.
    Hungarian matching during training.
    No NMS required during inference.

======================================================================
11. REAL TACO DATASET
======================================================================

Expected layout:

    TACO/
      data/
        annotations.json
        batch_1/
        batch_2/
        ...

We use real images and COCO bounding boxes.

The original TACO categories are mapped into five broad classes:

    plastic
    metal
    paper_cardboard
    glass
    other

The category mapping is heuristic. Inspect the mapping and dataset
summary before treating it as a production taxonomy.

Images are resized to 128 x 128 for this educational model.
Bounding boxes are scaled accordingly.

Train/validation splitting is performed by source image ID to
avoid image-level leakage.

======================================================================
12. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    RGB Camera
       |
       v
    DETR
       |
       v
    Object class + bounding box
       |
       v
    Object center / localization
       |
       v
    Camera-to-robot coordinate conversion
       |
       v
    Grasp planning
       |
       v
    ACT manipulation policy

DETR can provide object locations and categories.

For real robot grasping, 2D bounding boxes alone are insufficient:
depth or calibrated geometric information is also needed.

======================================================================
13. INSTALLATION
======================================================================

    pip install -U torch numpy matplotlib opencv-python pillow scipy

======================================================================
14. RUN
======================================================================

Numerical tests:

    python lesson38_detr_object_detection.py --check-only

Small experiment:

    python lesson38_detr_object_detection.py \
        --epochs 5 \
        --max-images 300 \
        --batch 4

Larger experiment:

    python lesson38_detr_object_detection.py \
        --epochs 50 \
        --max-images 2000 \
        --batch 4 \
        --device cuda

======================================================================
15. OUTPUT
======================================================================

    runs/lesson38/
        best_model.pth
        dataset_summary.json
        training_history.json
        evaluation.json
        training_curves.png
        predictions.png

======================================================================
NEXT LESSON
======================================================================

Lesson 39 — Segment Anything Model (SAM) Concepts From Scratch.
======================================================================
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError(
        "SciPy is required. Install it with: pip install scipy"
    ) from exc


# ======================================================================
# CONFIGURATION
# ======================================================================

IMAGE_SIZE = 128

NUM_CLASSES = 5

CLASS_NAMES = [
    "plastic",
    "metal",
    "paper_cardboard",
    "glass",
    "other",
]

NO_OBJECT_CLASS = NUM_CLASSES

EMBED_DIM = 128

NUM_HEADS = 8

NUM_ENCODER_LAYERS = 3

NUM_DECODER_LAYERS = 3

NUM_OBJECT_QUERIES = 50

FEEDFORWARD_DIM = 512

DROPOUT = 0.1

# Training loss weights.
CLASS_LOSS_WEIGHT = 1.0
BBOX_LOSS_WEIGHT = 5.0
GIOU_LOSS_WEIGHT = 2.0

# Matching cost weights.
MATCH_CLASS_WEIGHT = 1.0
MATCH_BBOX_WEIGHT = 5.0
MATCH_GIOU_WEIGHT = 2.0

# No-object class receives a smaller classification weight.
NO_OBJECT_WEIGHT = 0.1


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
# 2. CATEGORY MAPPING
# ======================================================================

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
        "jar",
    ],
}


def map_category(category_name):

    name = str(category_name).lower()

    normalized = (
        name.replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace(",", " ")
    )

    tokens = set(normalized.split())

    for class_name in [
        "glass",
        "metal",
        "paper_cardboard",
        "plastic",
    ]:

        for keyword in CATEGORY_KEYWORDS[class_name]:

            keyword_tokens = set(
                keyword.split()
            )

            if (
                keyword in normalized
                or keyword_tokens.issubset(tokens)
            ):

                return CLASS_NAMES.index(
                    class_name
                )

    return CLASS_NAMES.index("other")


# ======================================================================
# 3. TACO IMAGE PATH RESOLUTION
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
# 4. LOAD TACO DETECTION RECORDS
# ======================================================================

def load_taco_records(
    taco_root,
    max_images=None,
):

    taco_root = Path(taco_root)

    annotation_path = (
        taco_root / "annotations.json"
    )

    if not annotation_path.exists():

        raise FileNotFoundError(
            f"Missing annotations: {annotation_path}"
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

        if annotation.get("iscrowd", 0):
            continue

        image_id = int(
            annotation["image_id"]
        )

        annotations_by_image[image_id].append(
            annotation
        )

    records = []

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

        boxes = []
        labels = []

        for annotation in annotations_by_image[image_id]:

            category_id = int(
                annotation["category_id"]
            )

            if category_id not in category_mapping:
                continue

            x, y, width, height = map(
                float,
                annotation["bbox"],
            )

            if width <= 1 or height <= 1:
                continue

            boxes.append([
                x,
                y,
                x + width,
                y + height,
            ])

            labels.append(
                category_mapping[category_id]
            )

        if len(boxes) == 0:
            continue

        records.append({

            "image_id": image_id,

            "path": str(image_path),

            "boxes": boxes,

            "labels": labels,

        })

    random.Random(42).shuffle(records)

    if max_images is not None:
        records = records[:max_images]

    if len(records) < 2:

        raise RuntimeError(
            "Need at least two valid annotated images."
        )

    return records


# ======================================================================
# 5. TRAIN / VALIDATION SPLIT
# ======================================================================

def split_records(
    records,
    val_ratio=0.2,
):

    records = list(records)

    random.Random(123).shuffle(
        records
    )

    val_count = max(
        1,
        int(len(records) * val_ratio),
    )

    val_count = min(
        val_count,
        len(records) - 1,
    )

    val_records = records[:val_count]

    train_records = records[val_count:]

    return train_records, val_records


# ======================================================================
# 6. BOX FORMAT CONVERSIONS
# ======================================================================

def box_xyxy_to_cxcywh(boxes):

    x1, y1, x2, y2 = boxes.unbind(
        dim=-1
    )

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    width = x2 - x1
    height = y2 - y1

    return torch.stack(
        [
            cx,
            cy,
            width,
            height,
        ],
        dim=-1,
    )


def box_cxcywh_to_xyxy(boxes):

    cx, cy, width, height = boxes.unbind(
        dim=-1
    )

    x1 = cx - width / 2.0
    y1 = cy - height / 2.0

    x2 = cx + width / 2.0
    y2 = cy + height / 2.0

    return torch.stack(
        [
            x1,
            y1,
            x2,
            y2,
        ],
        dim=-1,
    )


# ======================================================================
# 7. IoU AND GENERALIZED IoU
# ======================================================================

def pairwise_iou(boxes1, boxes2):

    """
    boxes1: [N, 4], xyxy
    boxes2: [M, 4], xyxy

    Output:
        [N, M]
    """

    if boxes1.numel() == 0 or boxes2.numel() == 0:

        return boxes1.new_zeros(
            (boxes1.shape[0], boxes2.shape[0])
        )

    left_top = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )

    right_bottom = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    intersection_size = (
        right_bottom - left_top
    ).clamp(min=0)

    intersection = (
        intersection_size[..., 0]
        *
        intersection_size[..., 1]
    )

    area1 = (

        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)

        *

        (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)

    )

    area2 = (

        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)

        *

        (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    )

    union = (
        area1[:, None]
        + area2[None, :]
        - intersection
    )

    return intersection / union.clamp(
        min=1e-7
    )


def generalized_box_iou(boxes1, boxes2):

    """
    Pairwise Generalized IoU.

    Both inputs use xyxy coordinates.
    """

    iou = pairwise_iou(
        boxes1,
        boxes2,
    )

    left_top = torch.minimum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )

    right_bottom = torch.maximum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    enclosing_size = (
        right_bottom - left_top
    ).clamp(min=0)

    enclosing_area = (
        enclosing_size[..., 0]
        *
        enclosing_size[..., 1]
    )

    area1 = (

        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)

        *

        (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)

    )

    area2 = (

        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)

        *

        (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    )

    left_top_intersection = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )

    right_bottom_intersection = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    intersection_size = (
        right_bottom_intersection
        - left_top_intersection
    ).clamp(min=0)

    intersection = (
        intersection_size[..., 0]
        *
        intersection_size[..., 1]
    )

    union = (
        area1[:, None]
        + area2[None, :]
        - intersection
    )

    giou = iou - (
        enclosing_area - union
    ) / enclosing_area.clamp(min=1e-7)

    return giou


# ======================================================================
# 8. REAL TACO DATASET
# ======================================================================

class TACODetectionDataset(Dataset):

    def __init__(
        self,
        records,
        image_size=IMAGE_SIZE,
        augment=False,
    ):

        self.records = records

        self.image_size = image_size

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

        original_height, original_width = image.shape[:2]

        boxes = np.asarray(
            record["boxes"],
            dtype=np.float32,
        ).copy()

        labels = np.asarray(
            record["labels"],
            dtype=np.int64,
        ).copy()

        # Resize image to a fixed square resolution.
        image = cv2.resize(
            image,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=cv2.INTER_AREA,
        )

        # Scale box coordinates independently in x and y.
        boxes[:, [0, 2]] *= (
            self.image_size / original_width
        )

        boxes[:, [1, 3]] *= (
            self.image_size / original_height
        )

        if self.augment:

            if random.random() < 0.5:

                image = cv2.flip(
                    image,
                    1,
                )

                old_x1 = boxes[:, 0].copy()
                old_x2 = boxes[:, 2].copy()

                boxes[:, 0] = (
                    self.image_size - old_x2
                )

                boxes[:, 2] = (
                    self.image_size - old_x1
                )

            if random.random() < 0.5:

                alpha = random.uniform(
                    0.85,
                    1.15,
                )

                beta = random.uniform(
                    -12,
                    12,
                )

                image = cv2.convertScaleAbs(
                    image,
                    alpha=alpha,
                    beta=beta,
                )

        # Clip boxes to image boundaries.
        boxes[:, [0, 2]] = np.clip(
            boxes[:, [0, 2]],
            0,
            self.image_size,
        )

        boxes[:, [1, 3]] = np.clip(
            boxes[:, [1, 3]],
            0,
            self.image_size,
        )

        valid = (
            (boxes[:, 2] > boxes[:, 0])
            &
            (boxes[:, 3] > boxes[:, 1])
        )

        boxes = boxes[valid]
        labels = labels[valid]

        if len(boxes) == 0:

            # The original record contained boxes, but clipping
            # removed all of them. This should be rare.
            boxes = np.array(
                [[0, 0, 1, 1]],
                dtype=np.float32,
            )

            labels = np.array(
                [CLASS_NAMES.index("other")],
                dtype=np.int64,
            )

        boxes = torch.from_numpy(
            boxes
        )

        labels = torch.from_numpy(
            labels
        )

        # Normalize coordinates to [0, 1].
        boxes = boxes / float(
            self.image_size
        )

        # DETR predicts normalized cx, cy, w, h.
        boxes = box_xyxy_to_cxcywh(
            boxes
        )

        image = image.astype(
            np.float32
        ) / 255.0

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(
                image.transpose(2, 0, 1)
            )
        )

        target = {

            "boxes": boxes.float(),

            "labels": labels.long(),

            "image_id": int(record["image_id"]),

        }

        return image_tensor, target


def detection_collate(batch):

    images = torch.stack(
        [
            item[0]
            for item in batch
        ],
        dim=0,
    )

    targets = [
        item[1]
        for item in batch
    ]

    return images, targets


# ======================================================================
# 9. CNN BACKBONE
# ======================================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                8,
                out_channels,
            ),

            nn.GELU(),

        )

    def forward(self, x):

        return self.block(x)


class CNNBackbone(nn.Module):

    """
    Randomly initialized educational CNN backbone.

    Input:
        [B, 3, 128, 128]

    Output:
        [B, 128, 16, 16]
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
    ):

        super().__init__()

        self.layers = nn.Sequential(

            ConvBlock(
                3,
                32,
                stride=2,
            ),

            ConvBlock(
                32,
                64,
                stride=2,
            ),

            ConvBlock(
                64,
                embed_dim,
                stride=2,
            ),

        )

    def forward(self, x):

        return self.layers(x)


# ======================================================================
# 10. 2D SINE POSITIONAL ENCODING
# ======================================================================

def sine_position_encoding(
    height,
    width,
    embed_dim,
    device,
    temperature=10000,
):

    """
    Return:
        [1, embed_dim, height, width]

    The embedding dimension must be divisible by four.
    """

    if embed_dim % 4 != 0:

        raise ValueError(
            "embed_dim must be divisible by 4."
        )

    y = torch.arange(
        height,
        device=device,
        dtype=torch.float32,
    )

    x = torch.arange(
        width,
        device=device,
        dtype=torch.float32,
    )

    y_grid, x_grid = torch.meshgrid(
        y,
        x,
        indexing="ij",
    )

    # Normalize coordinates.
    y_grid = (
        y_grid + 0.5
    ) / max(height, 1)

    x_grid = (
        x_grid + 0.5
    ) / max(width, 1)

    num_pos_feats = embed_dim // 2

    dim_t = torch.arange(
        num_pos_feats,
        device=device,
        dtype=torch.float32,
    )

    dim_t = temperature ** (
        2 * torch.div(
            dim_t,
            2,
            rounding_mode="floor",
        ) / num_pos_feats
    )

    pos_x = (
        x_grid[..., None] * 2 * np.pi
    ) / dim_t

    pos_y = (
        y_grid[..., None] * 2 * np.pi
    ) / dim_t

    pos_x = torch.stack(
        (
            pos_x[..., 0::2].sin(),
            pos_x[..., 1::2].cos(),
        ),
        dim=-1,
    ).flatten(-2)

    pos_y = torch.stack(
        (
            pos_y[..., 0::2].sin(),
            pos_y[..., 1::2].cos(),
        ),
        dim=-1,
    ).flatten(-2)

    position = torch.cat(
        [
            pos_y,
            pos_x,
        ],
        dim=-1,
    )

    position = position.permute(
        2,
        0,
        1,
    ).unsqueeze(0)

    return position


# ======================================================================
# 11. MULTI-HEAD ATTENTION
# ======================================================================

class MultiHeadAttention(nn.Module):

    """
    Cross-attention or self-attention.

    query:
        [B, Nq, D]

    key_value:
        [B, Nk, D]

    output:
        [B, Nq, D]
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
    ):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.head_dim = (
            embed_dim // num_heads
        )

        self.scale = (
            self.head_dim ** -0.5
        )

        self.query_projection = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.key_projection = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.value_projection = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.output_projection = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def split_heads(self, x):

        batch_size, sequence_length, _ = x.shape

        x = x.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(
            1,
            2,
        )

    def forward(
        self,
        query,
        key_value=None,
    ):

        if key_value is None:
            key_value = query

        q = self.query_projection(
            query
        )

        k = self.key_projection(
            key_value
        )

        v = self.value_projection(
            key_value
        )

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # [B, heads, Nq, Nk]
        attention_scores = (
            q @ k.transpose(-2, -1)
        ) * self.scale

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights
        )

        output = (
            attention_weights @ v
        )

        output = output.transpose(
            1,
            2,
        ).contiguous()

        batch_size = output.shape[0]

        query_length = output.shape[1]

        output = output.reshape(
            batch_size,
            query_length,
            self.embed_dim,
        )

        return self.output_projection(
            output
        )


# ======================================================================
# 12. FEED-FORWARD NETWORK
# ======================================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        dropout=0.1,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embed_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                embed_dim,
            ),

            nn.Dropout(dropout),

        )

    def forward(self, x):

        return self.network(x)


# ======================================================================
# 13. TRANSFORMER ENCODER LAYER
# ======================================================================

class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        feedforward_dim,
        dropout=0.1,
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.self_attention = MultiHeadAttention(
            embed_dim,
            num_heads,
            dropout,
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        self.feedforward = FeedForward(
            embed_dim,
            feedforward_dim,
            dropout,
        )

    def forward(
        self,
        source,
        position,
    ):

        normalized = self.norm1(
            source
        )

        query = normalized + position

        source = source + self.self_attention(
            query,
            query,
        )

        source = source + self.feedforward(
            self.norm2(source)
        )

        return source


# ======================================================================
# 14. TRANSFORMER DECODER LAYER
# ======================================================================

class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        feedforward_dim,
        dropout=0.1,
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.self_attention = MultiHeadAttention(
            embed_dim,
            num_heads,
            dropout,
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        self.cross_attention = MultiHeadAttention(
            embed_dim,
            num_heads,
            dropout,
        )

        self.norm3 = nn.LayerNorm(
            embed_dim
        )

        self.feedforward = FeedForward(
            embed_dim,
            feedforward_dim,
            dropout,
        )

    def forward(
        self,
        target,
        memory,
        query_position,
        memory_position,
    ):

        # 1. Query self-attention.
        normalized = self.norm1(
            target
        )

        query = (
            normalized + query_position
        )

        target = target + self.self_attention(
            query,
            query,
        )

        # 2. Cross-attention with image features.
        normalized = self.norm2(
            target
        )

        query = (
            normalized + query_position
        )

        key = (
            memory + memory_position
        )

        target = target + self.cross_attention(
            query,
            key,
        )

        # 3. Feed-forward network.
        target = target + self.feedforward(
            self.norm3(target)
        )

        return target


# ======================================================================
# 15. DETR MODEL
# ======================================================================

class DETR(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        num_queries=NUM_OBJECT_QUERIES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        encoder_layers=NUM_ENCODER_LAYERS,
        decoder_layers=NUM_DECODER_LAYERS,
        feedforward_dim=FEEDFORWARD_DIM,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.num_classes = num_classes

        self.num_queries = num_queries

        self.embed_dim = embed_dim

        self.backbone = CNNBackbone(
            embed_dim
        )

        self.input_projection = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=1,
        )

        self.encoder = nn.ModuleList([

            TransformerEncoderLayer(
                embed_dim,
                num_heads,
                feedforward_dim,
                dropout,
            )

            for _ in range(encoder_layers)

        ])

        self.decoder = nn.ModuleList([

            TransformerDecoderLayer(
                embed_dim,
                num_heads,
                feedforward_dim,
                dropout,
            )

            for _ in range(decoder_layers)

        ])

        # Learnable object queries.
        self.query_embedding = nn.Embedding(
            num_queries,
            embed_dim,
        )

        # Learnable initial query content.
        self.query_content = nn.Embedding(
            num_queries,
            embed_dim,
        )

        # Classifier includes one extra no-object class.
        self.class_head = nn.Linear(
            embed_dim,
            num_classes + 1,
        )

        # Bounding-box regression head.
        self.box_head = nn.Sequential(

            nn.Linear(
                embed_dim,
                embed_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                embed_dim,
                embed_dim,
            ),

            nn.ReLU(),

            nn.Linear(
                embed_dim,
                4,
            ),

        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.initialize_weights()

    def initialize_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

        nn.init.normal_(
            self.query_embedding.weight,
            std=0.02,
        )

        nn.init.normal_(
            self.query_content.weight,
            std=0.02,
        )

    def forward(self, images):

        batch_size = images.shape[0]

        # ----------------------------------------------------------
        # CNN backbone.
        # ----------------------------------------------------------

        features = self.backbone(
            images
        )

        features = self.input_projection(
            features
        )

        _, channels, height, width = features.shape

        # ----------------------------------------------------------
        # Spatial positional encoding.
        # ----------------------------------------------------------

        position = sine_position_encoding(

            height,

            width,

            channels,

            features.device,

        )

        # [B, D, H, W] -> [B, H*W, D]
        memory = features.flatten(2).transpose(
            1,
            2,
        )

        position = position.flatten(2).transpose(
            1,
            2,
        )

        position = position.expand(
            batch_size,
            -1,
            -1,
        )

        # ----------------------------------------------------------
        # Transformer encoder.
        # ----------------------------------------------------------

        for encoder_layer in self.encoder:

            memory = encoder_layer(
                memory,
                position,
            )

        # ----------------------------------------------------------
        # Object query initialization.
        # ----------------------------------------------------------

        query_position = self.query_embedding.weight.unsqueeze(
            0
        ).expand(
            batch_size,
            -1,
            -1,
        )

        target = self.query_content.weight.unsqueeze(
            0
        ).expand(
            batch_size,
            -1,
            -1,
        )

        target = self.dropout(
            target
        )

        # ----------------------------------------------------------
        # Transformer decoder.
        # ----------------------------------------------------------

        for decoder_layer in self.decoder:

            target = decoder_layer(

                target,

                memory,

                query_position,

                position,

            )

        # ----------------------------------------------------------
        # Prediction heads.
        # ----------------------------------------------------------

        class_logits = self.class_head(
            target
        )

        # Sigmoid produces normalized cx, cy, w, h.
        predicted_boxes = self.box_head(
            target
        ).sigmoid()

        return {

            "pred_logits": class_logits,

            "pred_boxes": predicted_boxes,

        }


# ======================================================================
# 16. HUNGARIAN MATCHER
# ======================================================================

class HungarianMatcher:

    """
    One-to-one assignment between predicted boxes and ground truth.

    The assignment itself is computed without gradients.
    The matched prediction tensors are used later to calculate
    differentiable training losses.
    """

    def __init__(
        self,
        class_weight=MATCH_CLASS_WEIGHT,
        bbox_weight=MATCH_BBOX_WEIGHT,
        giou_weight=MATCH_GIOU_WEIGHT,
    ):

        self.class_weight = class_weight

        self.bbox_weight = bbox_weight

        self.giou_weight = giou_weight

    @torch.no_grad()
    def __call__(
        self,
        outputs,
        targets,
    ):

        batch_size = outputs["pred_logits"].shape[0]

        assignments = []

        for batch_index in range(batch_size):

            logits = outputs["pred_logits"][
                batch_index
            ]

            predicted_boxes = outputs["pred_boxes"][
                batch_index
            ]

            target_labels = targets[batch_index][
                "labels"
            ]

            target_boxes = targets[batch_index][
                "boxes"
            ]

            if len(target_labels) == 0:

                assignments.append((
                    torch.empty(
                        0,
                        dtype=torch.long,
                    ),

                    torch.empty(
                        0,
                        dtype=torch.long,
                    ),

                ))

                continue

            probabilities = logits.softmax(
                dim=-1
            )

            # Classification cost.
            class_cost = -probabilities[
                :,
                target_labels,
            ]

            # L1 bounding-box cost.
            bbox_cost = torch.cdist(
                predicted_boxes,
                target_boxes,
                p=1,
            )

            # Generalized IoU cost.
            predicted_xyxy = box_cxcywh_to_xyxy(
                predicted_boxes
            )

            target_xyxy = box_cxcywh_to_xyxy(
                target_boxes
            )

            giou_cost = -generalized_box_iou(
                predicted_xyxy,
                target_xyxy,
            )

            total_cost = (

                self.class_weight * class_cost

                +

                self.bbox_weight * bbox_cost

                +

                self.giou_weight * giou_cost

            )

            # Solve the minimum-cost assignment.
            row_indices, column_indices = linear_sum_assignment(
                total_cost.cpu().numpy()
            )

            assignments.append((

                torch.as_tensor(
                    row_indices,
                    dtype=torch.long,
                ),

                torch.as_tensor(
                    column_indices,
                    dtype=torch.long,
                ),

            ))

        return assignments


# ======================================================================
# 17. DETR LOSS
# ======================================================================

class DETRLoss(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        class_weight=CLASS_LOSS_WEIGHT,
        bbox_weight=BBOX_LOSS_WEIGHT,
        giou_weight=GIOU_LOSS_WEIGHT,
        no_object_weight=NO_OBJECT_WEIGHT,
    ):

        super().__init__()

        self.num_classes = num_classes

        self.class_weight = class_weight

        self.bbox_weight = bbox_weight

        self.giou_weight = giou_weight

        self.matcher = HungarianMatcher()

        class_weights = torch.ones(
            num_classes + 1,
            dtype=torch.float32,
        )

        class_weights[-1] = no_object_weight

        self.register_buffer(
            "class_weights",
            class_weights,
        )

    def forward(
        self,
        outputs,
        targets,
    ):

        logits = outputs["pred_logits"]

        predicted_boxes = outputs["pred_boxes"]

        batch_size, num_queries, _ = logits.shape

        assignments = self.matcher(
            outputs,
            targets,
        )

        # Every query initially represents no-object.
        target_classes = torch.full(

            (batch_size, num_queries),

            self.num_classes,

            dtype=torch.long,

            device=logits.device,

        )

        matched_predicted_boxes = []
        matched_target_boxes = []

        for batch_index, (
            source_indices,
            target_indices,
        ) in enumerate(assignments):

            source_indices = source_indices.to(
                logits.device
            )

            target_indices = target_indices.to(
                logits.device
            )

            if len(source_indices) == 0:
                continue

            target_classes[
                batch_index,
                source_indices,
            ] = targets[batch_index]["labels"][
                target_indices
            ].to(logits.device)

            matched_predicted_boxes.append(
                predicted_boxes[
                    batch_index,
                    source_indices,
                ]
            )

            matched_target_boxes.append(
                targets[batch_index]["boxes"][
                    target_indices
                ].to(logits.device)
            )

        # Classification loss for every query.
        loss_class = F.cross_entropy(

            logits.transpose(1, 2),

            target_classes,

            weight=self.class_weights.to(
                logits.device
            ),

        )

        if matched_predicted_boxes:

            matched_predicted_boxes = torch.cat(
                matched_predicted_boxes,
                dim=0,
            )

            matched_target_boxes = torch.cat(
                matched_target_boxes,
                dim=0,
            )

            number_of_matches = max(
                len(matched_predicted_boxes),
                1,
            )

            loss_bbox = F.l1_loss(

                matched_predicted_boxes,

                matched_target_boxes,

                reduction="sum",

            ) / number_of_matches

            predicted_xyxy = box_cxcywh_to_xyxy(
                matched_predicted_boxes
            )

            target_xyxy = box_cxcywh_to_xyxy(
                matched_target_boxes
            )

            giou = generalized_box_iou(
                predicted_xyxy,
                target_xyxy,
            )

            matched_giou = torch.diag(
                giou
            )

            loss_giou = (
                1.0 - matched_giou
            ).sum() / number_of_matches

        else:

            # Keep losses connected to the model graph.
            loss_bbox = predicted_boxes.sum() * 0.0

            loss_giou = predicted_boxes.sum() * 0.0

        total_loss = (

            self.class_weight * loss_class

            +

            self.bbox_weight * loss_bbox

            +

            self.giou_weight * loss_giou

        )

        return {

            "loss": total_loss,

            "loss_class": loss_class,

            "loss_bbox": loss_bbox,

            "loss_giou": loss_giou,

            "assignments": assignments,

        }


# ======================================================================
# 18. POST-PROCESSING
# ======================================================================

@torch.no_grad()
def postprocess_predictions(
    outputs,
    confidence_threshold=0.35,
    max_detections=50,
):

    probabilities = outputs["pred_logits"].softmax(
        dim=-1
    )

    scores, labels = probabilities[
        ...,
        :-1,
    ].max(
        dim=-1
    )

    boxes = outputs["pred_boxes"]

    results = []

    for batch_index in range(len(scores)):

        keep = (
            scores[batch_index] >= confidence_threshold
        )

        selected_scores = scores[
            batch_index,
            keep,
        ]

        selected_labels = labels[
            batch_index,
            keep,
        ]

        selected_boxes = boxes[
            batch_index,
            keep,
        ]

        if len(selected_scores) > max_detections:

            top_indices = torch.topk(

                selected_scores,

                max_detections,

            ).indices

            selected_scores = selected_scores[
                top_indices
            ]

            selected_labels = selected_labels[
                top_indices
            ]

            selected_boxes = selected_boxes[
                top_indices
            ]

        selected_boxes = box_cxcywh_to_xyxy(
            selected_boxes
        ).clamp(
            0,
            1,
        )

        results.append({

            "scores": selected_scores,

            "labels": selected_labels,

            "boxes": selected_boxes,

        })

    return results


# ======================================================================
# 19. DETECTION METRICS
# ======================================================================

def evaluate_detection_metrics(
    predictions,
    targets,
    iou_threshold=0.5,
):

    """
    Custom class-aware precision / recall / F1 evaluation.

    This is not COCO mAP.

    Each prediction is matched to the highest-IoU unmatched
    ground-truth object of the same class.
    """

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    matched_ious = []

    for prediction, target in zip(
        predictions,
        targets,
    ):

        pred_boxes = prediction["boxes"]

        pred_labels = prediction["labels"]

        pred_scores = prediction["scores"]

        gt_boxes = box_cxcywh_to_xyxy(
            target["boxes"]
        ).to(pred_boxes.device)

        gt_labels = target["labels"].to(
            pred_boxes.device
        )

        matched_gt = set()

        order = torch.argsort(
            pred_scores,
            descending=True,
        )

        for pred_index in order.tolist():

            pred_box = pred_boxes[
                pred_index:pred_index + 1
            ]

            pred_label = int(
                pred_labels[pred_index]
            )

            candidates = [

                index

                for index in range(len(gt_labels))

                if (
                    index not in matched_gt
                    and int(gt_labels[index]) == pred_label
                )

            ]

            if not candidates:

                false_positives += 1

                continue

            candidate_boxes = gt_boxes[
                candidates
            ]

            ious = pairwise_iou(
                pred_box,
                candidate_boxes,
            )[0]

            best_position = int(
                ious.argmax()
            )

            best_iou = float(
                ious[best_position]
            )

            best_gt_index = candidates[
                best_position
            ]

            if best_iou >= iou_threshold:

                true_positives += 1

                matched_gt.add(
                    best_gt_index
                )

                matched_ious.append(
                    best_iou
                )

            else:

                false_positives += 1

        false_negatives += (
            len(gt_labels) - len(matched_gt)
        )

    precision = (
        true_positives
        /
        max(
            true_positives + false_positives,
            1,
        )
    )

    recall = (
        true_positives
        /
        max(
            true_positives + false_negatives,
            1,
        )
    )

    f1 = (
        2 * precision * recall
        /
        max(
            precision + recall,
            1e-12,
        )
    )

    mean_iou = (
        float(np.mean(matched_ious))
        if matched_ious
        else 0.0
    )

    return {

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "mean_matched_iou": mean_iou,

        "true_positives": true_positives,

        "false_positives": false_positives,

        "false_negatives": false_negatives,

    }


# ======================================================================
# 20. TRAINING
# ======================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp,
):

    model.train()

    total_loss = 0.0

    total_class_loss = 0.0
    total_bbox_loss = 0.0
    total_giou_loss = 0.0

    total_images = 0

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = [

            {

                "boxes": target["boxes"].to(
                    device,
                    non_blocking=True,
                ),

                "labels": target["labels"].to(
                    device,
                    non_blocking=True,
                ),

                "image_id": target["image_id"],

            }

            for target in targets

        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):

            outputs = model(images)

            losses = criterion(
                outputs,
                targets,
            )

            loss = losses["loss"]

        if use_amp:

            scaler.scale(loss).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=0.1,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=0.1,
            )

            optimizer.step()

        batch_size = len(images)

        total_images += batch_size

        total_loss += (
            loss.item() * batch_size
        )

        total_class_loss += (
            losses["loss_class"].item() * batch_size
        )

        total_bbox_loss += (
            losses["loss_bbox"].item() * batch_size
        )

        total_giou_loss += (
            losses["loss_giou"].item() * batch_size
        )

    denominator = max(
        total_images,
        1,
    )

    return {

        "loss": total_loss / denominator,

        "class_loss": total_class_loss / denominator,

        "bbox_loss": total_bbox_loss / denominator,

        "giou_loss": total_giou_loss / denominator,

    }


# ======================================================================
# 21. VALIDATION
# ======================================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    confidence_threshold=0.35,
):

    model.eval()

    total_loss = 0.0

    total_images = 0

    all_predictions = []
    all_targets = []

    for images, targets in loader:

        images = images.to(
            device
        )

        targets_device = [

            {

                "boxes": target["boxes"].to(
                    device
                ),

                "labels": target["labels"].to(
                    device
                ),

                "image_id": target["image_id"],

            }

            for target in targets

        ]

        outputs = model(images)

        losses = criterion(
            outputs,
            targets_device,
        )

        total_loss += (
            losses["loss"].item() * len(images)
        )

        total_images += len(images)

        predictions = postprocess_predictions(

            outputs,

            confidence_threshold=confidence_threshold,

        )

        all_predictions.extend(
            predictions
        )

        all_targets.extend(
            targets_device
        )

    metrics = evaluate_detection_metrics(

        all_predictions,

        all_targets,

        iou_threshold=0.5,

    )

    metrics["loss"] = (
        total_loss / max(total_images, 1)
    )

    metrics["predictions"] = all_predictions

    metrics["targets"] = all_targets

    return metrics


# ======================================================================
# 22. VISUALIZATION
# ======================================================================

def plot_training_history(
    history,
    output_path,
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    axes[0].plot(
        history["train_loss"],
        label="Train",
    )

    axes[0].plot(
        history["val_loss"],
        label="Validation",
    )

    axes[0].set_title(
        "DETR Training Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        history["val_precision"],
        label="Precision",
    )

    axes[1].plot(
        history["val_recall"],
        label="Recall",
    )

    axes[1].plot(
        history["val_f1"],
        label="F1",
    )

    axes[1].set_title(
        "Validation Detection Metrics"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Metric"
    )

    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


def draw_boxes(
    image,
    boxes,
    labels,
    scores=None,
    color=(0, 255, 0),
):

    height, width = image.shape[:2]

    for index, box in enumerate(boxes):

        x1, y1, x2, y2 = box

        x1 = int(x1 * width)
        x2 = int(x2 * width)

        y1 = int(y1 * height)
        y2 = int(y2 * height)

        class_id = int(
            labels[index]
        )

        class_name = CLASS_NAMES[
            class_id
        ]

        if scores is not None:

            text = (
                f"{class_name} "
                f"{float(scores[index]):.2f}"
            )

        else:

            text = class_name

        cv2.rectangle(

            image,

            (x1, y1),

            (x2, y2),

            color,

            2,

        )

        cv2.putText(

            image,

            text,

            (x1, max(15, y1 - 5)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            color,

            1,

            cv2.LINE_AA,

        )

    return image


def save_prediction_visualization(
    model,
    dataset,
    device,
    output_path,
    count=8,
    confidence_threshold=0.25,
):

    model.eval()

    count = min(
        count,
        len(dataset),
    )

    if count == 0:
        return

    columns = 2

    rows = int(
        np.ceil(count / columns)
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(13, 5 * rows),
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for index in range(count):

        image_tensor, target = dataset[index]

        with torch.no_grad():

            outputs = model(
                image_tensor.unsqueeze(0).to(device)
            )

            predictions = postprocess_predictions(

                outputs,

                confidence_threshold=confidence_threshold,

            )[0]

        image = image_tensor.numpy().transpose(
            1,
            2,
            0,
        )

        image = np.clip(
            image * 255,
            0,
            255,
        ).astype(
            np.uint8
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

        # Draw ground truth in blue.
        gt_boxes = box_cxcywh_to_xyxy(
            target["boxes"]
        ).numpy()

        gt_labels = target["labels"].numpy()

        image = draw_boxes(

            image,

            gt_boxes,

            gt_labels,

            color=(255, 0, 0),

        )

        # Draw predictions in green.
        pred_boxes = predictions["boxes"].cpu().numpy()

        pred_labels = predictions["labels"].cpu().numpy()

        pred_scores = predictions["scores"].cpu().numpy()

        image = draw_boxes(

            image,

            pred_boxes,

            pred_labels,

            scores=pred_scores,

            color=(0, 255, 0),

        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        axes[index].imshow(image)

        axes[index].set_title(
            f"Image ID: {target['image_id']}\n"
            "Ground truth: blue | Predictions: green"
        )

        axes[index].axis("off")

    for index in range(count, len(axes)):
        axes[index].axis("off")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


# ======================================================================
# 23. NUMERICAL TESTS
# ======================================================================

def run_tests():

    print("\n" + "=" * 70)
    print("LESSON 38 — NUMERICAL TESTS")
    print("=" * 70)

    device = torch.device("cpu")

    # --------------------------------------------------------------
    # Test 1: Box conversion.
    # --------------------------------------------------------------

    boxes_xyxy = torch.tensor(
        [
            [0.1, 0.2, 0.5, 0.8],
            [0.2, 0.1, 0.6, 0.7],
        ],
        dtype=torch.float32,
    )

    boxes_cxcywh = box_xyxy_to_cxcywh(
        boxes_xyxy
    )

    reconstructed = box_cxcywh_to_xyxy(
        boxes_cxcywh
    )

    assert torch.allclose(
        boxes_xyxy,
        reconstructed,
        atol=1e-6,
    )

    print("Test 1 passed: bounding-box format conversion.")

    # --------------------------------------------------------------
    # Test 2: IoU of identical boxes.
    # --------------------------------------------------------------

    iou = pairwise_iou(
        boxes_xyxy[:1],
        boxes_xyxy[:1],
    )

    assert torch.allclose(
        iou,
        torch.ones_like(iou),
        atol=1e-6,
    )

    print("Test 2 passed: IoU of identical boxes.")

    # --------------------------------------------------------------
    # Test 3: IoU of non-overlapping boxes.
    # --------------------------------------------------------------

    box_a = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0]]
    )

    box_b = torch.tensor(
        [[2.0, 2.0, 3.0, 3.0]]
    )

    iou = pairwise_iou(
        box_a,
        box_b,
    )

    assert torch.allclose(
        iou,
        torch.zeros_like(iou),
    )

    print("Test 3 passed: IoU of non-overlapping boxes.")

    # --------------------------------------------------------------
    # Test 4: GIoU of identical boxes.
    # --------------------------------------------------------------

    giou = generalized_box_iou(
        box_a,
        box_a,
    )

    assert torch.allclose(
        giou,
        torch.ones_like(giou),
        atol=1e-6,
    )

    print("Test 4 passed: GIoU of identical boxes.")

    # --------------------------------------------------------------
    # Test 5: GIoU for non-overlapping boxes.
    # --------------------------------------------------------------

    giou = generalized_box_iou(
        box_a,
        box_b,
    )

    assert giou.item() < 0.0

    print("Test 5 passed: GIoU penalizes non-overlapping boxes.")

    # --------------------------------------------------------------
    # Test 6: Positional encoding.
    # --------------------------------------------------------------

    position = sine_position_encoding(
        height=16,
        width=16,
        embed_dim=128,
        device=device,
    )

    assert position.shape == (
        1,
        128,
        16,
        16,
    )

    print("Test 6 passed: 2D positional encoding.")

    # --------------------------------------------------------------
    # Test 7: CNN backbone.
    # --------------------------------------------------------------

    backbone = CNNBackbone(
        embed_dim=128
    )

    image = torch.randn(
        2,
        3,
        128,
        128,
    )

    features = backbone(image)

    assert features.shape == (
        2,
        128,
        16,
        16,
    )

    print("Test 7 passed: CNN backbone feature dimensions.")

    # --------------------------------------------------------------
    # Test 8: Multi-head cross-attention.
    # --------------------------------------------------------------

    attention = MultiHeadAttention(
        embed_dim=128,
        num_heads=8,
    )

    query = torch.randn(
        2,
        10,
        128,
    )

    memory = torch.randn(
        2,
        256,
        128,
    )

    output = attention(
        query,
        memory,
    )

    assert output.shape == (
        2,
        10,
        128,
    )

    print("Test 8 passed: multi-head cross-attention.")

    # --------------------------------------------------------------
    # Test 9: DETR output dimensions.
    # --------------------------------------------------------------

    model = DETR(
        num_classes=5,
        num_queries=20,
        embed_dim=128,
        num_heads=8,
        encoder_layers=2,
        decoder_layers=2,
        feedforward_dim=256,
    )

    image = torch.randn(
        2,
        3,
        128,
        128,
    )

    outputs = model(image)

    assert outputs["pred_logits"].shape == (
        2,
        20,
        6,
    )

    assert outputs["pred_boxes"].shape == (
        2,
        20,
        4,
    )

    print("Test 9 passed: DETR prediction dimensions.")

    # --------------------------------------------------------------
    # Test 10: Bounding boxes are normalized.
    # --------------------------------------------------------------

    assert torch.all(
        outputs["pred_boxes"] >= 0
    )

    assert torch.all(
        outputs["pred_boxes"] <= 1
    )

    print("Test 10 passed: normalized box predictions.")

    # --------------------------------------------------------------
    # Test 11: Hungarian assignment.
    # --------------------------------------------------------------

    matcher = HungarianMatcher()

    sample_outputs = {

        "pred_logits": torch.tensor(
            [
                [
                    [5.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0],
                    [0.0, 0.0, 5.0],
                ]
            ]
        ),

        "pred_boxes": torch.tensor(
            [
                [
                    [0.2, 0.2, 0.1, 0.1],
                    [0.7, 0.7, 0.1, 0.1],
                    [0.5, 0.5, 0.2, 0.2],
                ]
            ]
        ),

    }

    sample_targets = [

        {

            "labels": torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),

            "boxes": torch.tensor(
                [
                    [0.2, 0.2, 0.1, 0.1],
                    [0.7, 0.7, 0.1, 0.1],
                ]
            ),

        }

    ]

    assignments = matcher(
        sample_outputs,
        sample_targets,
    )

    source_indices, target_indices = assignments[0]

    assert len(source_indices) == 2

    assert len(set(source_indices.tolist())) == 2

    assert len(set(target_indices.tolist())) == 2

    assert source_indices.tolist() == [0, 1]

    print("Test 11 passed: Hungarian one-to-one assignment.")

    # --------------------------------------------------------------
    # Test 12: DETR loss.
    # --------------------------------------------------------------

    criterion = DETRLoss(
        num_classes=5
    )

    loss_output = criterion(
        outputs,
        [
            {
                "labels": torch.tensor(
                    [0, 2],
                    dtype=torch.long,
                ),

                "boxes": torch.tensor(
                    [
                        [0.2, 0.2, 0.1, 0.1],
                        [0.7, 0.7, 0.2, 0.2],
                    ]
                ),
            },

            {
                "labels": torch.tensor(
                    [1],
                    dtype=torch.long,
                ),

                "boxes": torch.tensor(
                    [
                        [0.5, 0.5, 0.2, 0.2],
                    ]
                ),
            },
        ],
    )

    assert torch.isfinite(
        loss_output["loss"]
    )

    assert loss_output["loss"].item() >= 0

    print("Test 12 passed: DETR loss calculation.")

    # --------------------------------------------------------------
    # Test 13: Gradient propagation.
    # --------------------------------------------------------------

    model.train()

    model.zero_grad()

    image = torch.randn(
        2,
        3,
        128,
        128,
    )

    outputs = model(image)

    targets = [

        {

            "labels": torch.tensor(
                [0, 2],
                dtype=torch.long,
            ),

            "boxes": torch.tensor(
                [
                    [0.2, 0.2, 0.1, 0.1],
                    [0.7, 0.7, 0.2, 0.2],
                ]
            ),

        },

        {

            "labels": torch.tensor(
                [1],
                dtype=torch.long,
            ),

            "boxes": torch.tensor(
                [
                    [0.5, 0.5, 0.2, 0.2],
                ]
            ),

        },

    ]

    losses = criterion(
        outputs,
        targets,
    )

    losses["loss"].backward()

    assert model.backbone.layers[0].block[0].weight.grad is not None

    assert model.query_embedding.weight.grad is not None

    assert model.class_head.weight.grad is not None

    assert model.box_head[-1].weight.grad is not None

    print("Test 13 passed: gradients reach backbone and query embeddings.")

    # --------------------------------------------------------------
    # Test 14: Empty target matching.
    # --------------------------------------------------------------

    empty_targets = [

        {

            "labels": torch.empty(
                0,
                dtype=torch.long,
            ),

            "boxes": torch.empty(
                0,
                4,
            ),

        }

    ]

    empty_outputs = {

        "pred_logits": torch.randn(
            1,
            5,
            6,
        ),

        "pred_boxes": torch.rand(
            1,
            5,
            4,
        ),

    }

    empty_assignment = matcher(
        empty_outputs,
        empty_targets,
    )

    assert len(empty_assignment[0][0]) == 0

    print("Test 14 passed: empty target matching.")

    # --------------------------------------------------------------
    # Test 15: Post-processing.
    # --------------------------------------------------------------

    test_outputs = {

        "pred_logits": torch.tensor(
            [
                [
                    [8.0, 0.0, 0.0],
                    [0.0, 7.0, 0.0],
                    [0.0, 0.0, 8.0],
                ]
            ]
        ),

        "pred_boxes": torch.tensor(
            [
                [
                    [0.2, 0.2, 0.1, 0.1],
                    [0.7, 0.7, 0.1, 0.1],
                    [0.5, 0.5, 0.2, 0.2],
                ]
            ]
        ),

    }

    processed = postprocess_predictions(
        test_outputs,
        confidence_threshold=0.5,
    )

    assert len(processed) == 1

    assert len(processed[0]["boxes"]) == 2

    print("Test 15 passed: no-object filtering.")

    # --------------------------------------------------------------
    # Test 16: Detection metrics.
    # --------------------------------------------------------------

    metric_prediction = [

        {

            "boxes": torch.tensor(
                [
                    [0.1, 0.1, 0.5, 0.5],
                    [0.6, 0.6, 0.9, 0.9],
                ]
            ),

            "labels": torch.tensor(
                [0, 1]
            ),

            "scores": torch.tensor(
                [0.95, 0.90]
            ),

        }

    ]

    metric_target = [

        {

            "boxes": torch.tensor(
                [
                    [0.3, 0.3, 0.4, 0.4],
                    [0.75, 0.75, 0.3, 0.3],
                ]
            ),

            "labels": torch.tensor(
                [0, 1]
            ),

        }

    ]

    metrics = evaluate_detection_metrics(
        metric_prediction,
        metric_target,
    )

    assert metrics["true_positives"] == 2

    assert metrics["false_positives"] == 0

    assert metrics["false_negatives"] == 0

    assert np.isclose(
        metrics["f1"],
        1.0,
    )

    print("Test 16 passed: detection precision, recall and F1.")

    print("\nALL 16 TESTS PASSED.\n")


# ======================================================================
# 24. ARGUMENTS
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Lesson 38 — DETR from scratch."
    )

    parser.add_argument(
        "--taco-root",
        default="./TACO/data",
    )

    parser.add_argument(
        "--output",
        default="./runs/lesson38",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    return parser.parse_args()


# ======================================================================
# 25. MAIN
# ======================================================================

def main():

    args = parse_args()

    set_seed(42)

    print("=" * 70)
    print("LESSON 38 — DETR OBJECT DETECTION FROM SCRATCH")
    print("=" * 70)

    print(
        f"PyTorch version: {torch.__version__}"
    )

    print(
        f"Image size: {IMAGE_SIZE}"
    )

    print(
        f"Embedding dimension: {EMBED_DIM}"
    )

    print(
        f"Object queries: {NUM_OBJECT_QUERIES}"
    )

    print(
        f"Encoder layers: {NUM_ENCODER_LAYERS}"
    )

    print(
        f"Decoder layers: {NUM_DECODER_LAYERS}"
    )

    run_tests()

    if args.check_only:

        print(
            "CHECK-ONLY: all numerical tests passed."
        )

        return

    # --------------------------------------------------------------
    # Device.
    # --------------------------------------------------------------

    device = torch.device(

        args.device

        if args.device

        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    )

    print(
        f"\nDevice: {device}"
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Load real TACO data.
    # --------------------------------------------------------------

    records = load_taco_records(

        taco_root=args.taco_root,

        max_images=args.max_images,

    )

    train_records, val_records = split_records(
        records
    )

    print(
        f"\nTotal annotated images: {len(records)}"
    )

    print(
        f"Training images: {len(train_records)}"
    )

    print(
        f"Validation images: {len(val_records)}"
    )

    # --------------------------------------------------------------
    # Dataset summary.
    # --------------------------------------------------------------

    train_class_counts = Counter()
    val_class_counts = Counter()

    for record in train_records:

        train_class_counts.update(
            record["labels"]
        )

    for record in val_records:

        val_class_counts.update(
            record["labels"]
        )

    print("\nTRAIN CLASS COUNTS")

    for class_id, class_name in enumerate(CLASS_NAMES):

        print(
            f"{class_name:20s}: "
            f"{train_class_counts[class_id]}"
        )

    print("\nVALIDATION CLASS COUNTS")

    for class_id, class_name in enumerate(CLASS_NAMES):

        print(
            f"{class_name:20s}: "
            f"{val_class_counts[class_id]}"
        )

    summary = {

        "total_images": len(records),

        "training_images": len(train_records),

        "validation_images": len(val_records),

        "training_class_counts": {

            CLASS_NAMES[class_id]: train_class_counts[class_id]

            for class_id in range(NUM_CLASSES)

        },

        "validation_class_counts": {

            CLASS_NAMES[class_id]: val_class_counts[class_id]

            for class_id in range(NUM_CLASSES)

        },

    }

    with (
        output_dir / "dataset_summary.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # --------------------------------------------------------------
    # DataLoaders.
    # --------------------------------------------------------------

    train_dataset = TACODetectionDataset(
        train_records,
        augment=True,
    )

    val_dataset = TACODetectionDataset(
        val_records,
        augment=False,
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch,

        shuffle=True,

        num_workers=args.workers,

        pin_memory=(device.type == "cuda"),

        collate_fn=detection_collate,

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=args.batch,

        shuffle=False,

        num_workers=args.workers,

        pin_memory=(device.type == "cuda"),

        collate_fn=detection_collate,

    )

    # --------------------------------------------------------------
    # Model and loss.
    # --------------------------------------------------------------

    model = DETR().to(device)

    criterion = DETRLoss().to(device)

    parameter_count = sum(

        parameter.numel()

        for parameter in model.parameters()

        if parameter.requires_grad

    )

    print(
        f"\nTrainable parameters: {parameter_count:,}"
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=args.lr,

        weight_decay=args.weight_decay,

    )

    scheduler = torch.optim.lr_scheduler.StepLR(

        optimizer,

        step_size=max(
            args.epochs // 2,
            1,
        ),

        gamma=0.1,

    )

    use_amp = (
        device.type == "cuda"
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    # --------------------------------------------------------------
    # Training history.
    # --------------------------------------------------------------

    history = {

        "train_loss": [],

        "val_loss": [],

        "val_precision": [],

        "val_recall": [],

        "val_f1": [],

    }

    best_f1 = -1.0

    best_path = (
        output_dir / "best_model.pth"
    )

    # --------------------------------------------------------------
    # Training loop.
    # --------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):

        train_metrics = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            device,

            scaler,

            use_amp,

        )

        val_metrics = validate(

            model,

            val_loader,

            criterion,

            device,

            confidence_threshold=args.confidence,

        )

        scheduler.step()

        history["train_loss"].append(
            train_metrics["loss"]
        )

        history["val_loss"].append(
            val_metrics["loss"]
        )

        history["val_precision"].append(
            val_metrics["precision"]
        )

        history["val_recall"].append(
            val_metrics["recall"]
        )

        history["val_f1"].append(
            val_metrics["f1"]
        )

        print(
            f"\nEpoch {epoch}/{args.epochs}"
        )

        print(
            f"Train loss: {train_metrics['loss']:.4f}"
        )

        print(
            f"  Classification: {train_metrics['class_loss']:.4f}"
        )

        print(
            f"  Bounding box:   {train_metrics['bbox_loss']:.4f}"
        )

        print(
            f"  GIoU:           {train_metrics['giou_loss']:.4f}"
        )

        print(
            f"Validation loss: {val_metrics['loss']:.4f}"
        )

        print(
            f"Precision: {val_metrics['precision']:.4f}"
        )

        print(
            f"Recall: {val_metrics['recall']:.4f}"
        )

        print(
            f"F1: {val_metrics['f1']:.4f}"
        )

        print(
            f"Mean matched IoU: {val_metrics['mean_matched_iou']:.4f}"
        )

        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]

            torch.save(

                {

                    "model_state_dict": model.state_dict(),

                    "epoch": epoch,

                    "best_f1": best_f1,

                    "class_names": CLASS_NAMES,

                    "image_size": IMAGE_SIZE,

                    "num_queries": NUM_OBJECT_QUERIES,

                    "embed_dim": EMBED_DIM,

                },

                best_path,

            )

            print(
                f"Saved best checkpoint: {best_path}"
            )

    # --------------------------------------------------------------
    # Save history.
    # --------------------------------------------------------------

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

        output_dir / "training_curves.png",

    )

    # --------------------------------------------------------------
    # Load best checkpoint.
    # --------------------------------------------------------------

    checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------------
    # Final validation.
    # --------------------------------------------------------------

    final_metrics = validate(

        model,

        val_loader,

        criterion,

        device,

        confidence_threshold=args.confidence,

    )

    print("\nFINAL VALIDATION RESULTS")

    print(
        f"Precision: {final_metrics['precision']:.4f}"
    )

    print(
        f"Recall: {final_metrics['recall']:.4f}"
    )

    print(
        f"F1: {final_metrics['f1']:.4f}"
    )

    print(
        f"Mean matched IoU: {final_metrics['mean_matched_iou']:.4f}"
    )

    print(
        f"True positives: {final_metrics['true_positives']}"
    )

    print(
        f"False positives: {final_metrics['false_positives']}"
    )

    print(
        f"False negatives: {final_metrics['false_negatives']}"
    )

    # --------------------------------------------------------------
    # Prediction visualization.
    # --------------------------------------------------------------

    save_prediction_visualization(

        model,

        val_dataset,

        device,

        output_dir / "predictions.png",

        count=8,

        confidence_threshold=args.confidence,

    )

    # --------------------------------------------------------------
    # Save final evaluation.
    # --------------------------------------------------------------

    evaluation = {

        "precision": final_metrics["precision"],

        "recall": final_metrics["recall"],

        "f1": final_metrics["f1"],

        "mean_matched_iou": final_metrics["mean_matched_iou"],

        "true_positives": final_metrics["true_positives"],

        "false_positives": final_metrics["false_positives"],

        "false_negatives": final_metrics["false_negatives"],

        "best_epoch": checkpoint["epoch"],

        "trainable_parameters": parameter_count,

        "evaluation_note": (
            "Custom class-aware precision/recall/F1 at IoU 0.5; "
            "this is not COCO mAP."
        ),

    }

    with (
        output_dir / "evaluation.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            evaluation,
            file,
            indent=2,
        )

    print("\nLESSON 38 COMPLETED")

    print(
        f"Best checkpoint: {best_path.resolve()}"
    )

    print(
        f"Output directory: {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()