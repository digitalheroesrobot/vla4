"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 37 — VISION TRANSFORMER (ViT) FROM SCRATCH
======================================================================

FILENAME:
    lesson37_vision_transformer_from_scratch.py

COURSE RULES
------------
1. No pretrained models.
2. No pretrained weights.
3. Use real TACO waste photographs.
4. Implement the essential Transformer architecture ourselves.
5. Include mathematical explanations and numerical tests.
6. Include training, evaluation, and visualizations.

======================================================================
1. WHY VISION TRANSFORMERS?
======================================================================

CNN:
    Uses convolutional filters to extract local visual features.

ViT:
    Converts image patches into tokens.
    Uses self-attention to learn relationships between tokens.

The fundamental difference:

    CNN:
        Local spatial processing through convolution.

    ViT:
        Global relationships through self-attention.

======================================================================
2. IMAGE PATCHIFICATION
======================================================================

Suppose the input image is:

    Image = 128 x 128 x 3

Choose patch size:

    P = 16

Number of patches:

    N = (128 / 16) * (128 / 16)

      = 8 * 8

      = 64 patches

Each patch has:

    16 * 16 * 3 = 768 values

Therefore, the image becomes:

    [64, 768]

Each patch is a token.

======================================================================
3. PATCH EMBEDDING
======================================================================

Each patch is projected into a feature vector:

    z = xW + b

where:

    x = flattened image patch
    W = learnable projection matrix
    b = bias
    z = embedding vector

For embedding dimension D = 192:

    [64, 768] -> [64, 192]

We implement this with a linear layer.

======================================================================
4. CLASS TOKEN
======================================================================

Add a learnable classification token:

    [CLS]

The token sequence becomes:

    [CLS, Patch1, Patch2, ..., Patch64]

Sequence length:

    65 tokens

The final CLS representation is used for image classification.

======================================================================
5. POSITIONAL EMBEDDING
======================================================================

Self-attention does not inherently know the original spatial
position of each patch.

We therefore add learnable positional embeddings:

    Z = PatchEmbedding + PositionalEmbedding

The CLS token also receives a positional embedding.

======================================================================
6. SELF-ATTENTION
======================================================================

For input tokens X:

    Q = XWq
    K = XWk
    V = XWv

Attention:

    Attention(Q,K,V) =
        softmax(QK^T / sqrt(d_k)) V

Interpretation:

    Q: What information am I looking for?

    K: What information does each token contain?

    V: What information should be transferred?

Each token can attend to other tokens.

======================================================================
7. MULTI-HEAD SELF-ATTENTION
======================================================================

Instead of one attention operation, use multiple heads.

    MultiHead(Q,K,V) =
        Concat(head1, head2, ..., headH) Wo

For this lesson:

    embedding dimension = 192
    attention heads = 6
    dimension per head = 32

    192 / 6 = 32

Each head learns different relationships.

======================================================================
8. TRANSFORMER ENCODER
======================================================================

Each encoder block contains:

    X = X + MultiHeadAttention(LayerNorm(X))

    X = X + MLP(LayerNorm(X))

The residual connections improve optimization.

The MLP:

    Linear -> GELU -> Linear

======================================================================
9. CLASSIFICATION
======================================================================

Take the final CLS token:

    representation = X[:, 0]

Then:

    logits = Linear(representation)

For this lesson:

    5 waste categories:

        0: plastic
        1: metal
        2: paper_cardboard
        3: glass
        4: other

======================================================================
10. REAL DATASET
======================================================================

We use TACO photographs.

Instead of assigning one arbitrary label to an entire image,
we extract individual object crops using the real COCO bounding
box annotations.

Each crop becomes one classification example.

Important:
    The crop is taken from a real photograph.
    No synthetic object images are generated.

We split by original image ID, not by crop, to prevent crops from
the same source image leaking into both training and validation.

======================================================================
11. TRAINING
======================================================================

Loss:

    CrossEntropyLoss(logits, class_label)

Optimizer:

    AdamW

Evaluation:

    Accuracy
    Precision
    Recall
    F1 score
    Confusion matrix

======================================================================
12. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    Camera image
       |
       v
    Object detection / localization
       |
       v
    Crop individual object
       |
       v
    Vision Transformer
       |
       v
    Material classification
       |
       v
    Grasp planning
       |
       v
    ACT manipulation policy

A ViT classifier can complement a detector or instance segmentation
model.

For real-time robotics, inference latency and camera resolution must
also be considered.

======================================================================
13. INSTALLATION
======================================================================

    pip install -U torch numpy matplotlib opencv-python pillow

======================================================================
14. RUN
======================================================================

Numerical tests:

    python lesson37_vision_transformer_from_scratch.py --check-only

Train:

    python lesson37_vision_transformer_from_scratch.py \
        --epochs 30 \
        --max-images 2000 \
        --batch 32 \
        --device cuda

Smaller experiment:

    python lesson37_vision_transformer_from_scratch.py \
        --epochs 5 \
        --max-images 300 \
        --batch 16

======================================================================
OUTPUT
======================================================================

    runs/lesson37/
        best_model.pth
        training_history.json
        training_curves.png
        confusion_matrix.png
        predictions.png
        dataset_summary.json

======================================================================
NEXT LESSON
======================================================================

Lesson 38 — DETR: Detection Transformer From Scratch.
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

IMAGE_SIZE = 128
PATCH_SIZE = 16

EMBED_DIM = 192
NUM_HEADS = 6
DEPTH = 6
MLP_RATIO = 4

DROPOUT = 0.1

NUM_PATCHES = (
    IMAGE_SIZE // PATCH_SIZE
) ** 2

NUM_TOKENS = NUM_PATCHES + 1


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
        "glass bottle",
        "glass jar",
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

            return CLASS_NAMES.index(class_name)

    return CLASS_NAMES.index("other")


# ======================================================================
# 3. LOAD REAL TACO OBJECT CROPS
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


def load_taco_object_records(
    taco_root,
    max_images=None,
    padding_ratio=0.15,
):

    """
    Create object-level classification records.

    Every example references:
        - A real TACO image
        - A bounding box from COCO annotations
        - A mapped waste class

    We split by source image later to avoid data leakage.
    """

    taco_root = Path(taco_root)

    annotation_path = (
        taco_root / "annotations.json"
    )

    if not annotation_path.exists():

        raise FileNotFoundError(
            f"Missing TACO annotations: {annotation_path}"
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

    image_records = []

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

        objects = []

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

            if width <= 2 or height <= 2:
                continue

            objects.append({

                "bbox": [
                    x,
                    y,
                    x + width,
                    y + height,
                ],

                "label": category_mapping[
                    category_id
                ],

            })

        if not objects:
            continue

        image_records.append({

            "image_id": image_id,

            "path": str(image_path),

            "objects": objects,

        })

    random.Random(42).shuffle(
        image_records
    )

    if max_images is not None:

        image_records = image_records[:max_images]

    # Expand image-level records into object-level examples.
    object_records = []

    for image_record in image_records:

        for object_info in image_record["objects"]:

            object_records.append({

                "image_id": image_record["image_id"],

                "path": image_record["path"],

                "bbox": object_info["bbox"],

                "label": object_info["label"],

                "padding_ratio": padding_ratio,

            })

    if len(object_records) == 0:

        raise RuntimeError(
            "No valid TACO object annotations found."
        )

    return image_records, object_records


# ======================================================================
# 4. SPLIT BY SOURCE IMAGE
# ======================================================================

def split_by_image(
    image_records,
    object_records,
    val_ratio=0.2,
):

    """
    Split by original image ID.

    This is important.

    If two object crops come from the same photograph, they must
    not appear in both training and validation datasets.
    """

    image_ids = [
        record["image_id"]
        for record in image_records
    ]

    random.Random(123).shuffle(
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

    train_records = [

        record
        for record in object_records
        if record["image_id"] in train_ids

    ]

    val_records = [

        record
        for record in object_records
        if record["image_id"] in val_ids

    ]

    if not train_records or not val_records:

        raise RuntimeError(
            "The train/validation split is empty."
        )

    return train_records, val_records


# ======================================================================
# 5. DATASET
# ======================================================================

class TACOObjectDataset(Dataset):

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
                f"Cannot load image: {record['path']}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image_height, image_width = image.shape[:2]

        x1, y1, x2, y2 = record["bbox"]

        box_width = x2 - x1
        box_height = y2 - y1

        pad_x = box_width * record["padding_ratio"]
        pad_y = box_height * record["padding_ratio"]

        x1 = max(
            0,
            int(x1 - pad_x),
        )

        y1 = max(
            0,
            int(y1 - pad_y),
        )

        x2 = min(
            image_width,
            int(x2 + pad_x),
        )

        y2 = min(
            image_height,
            int(y2 + pad_y),
        )

        crop = image[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:

            raise RuntimeError(
                f"Invalid crop: {record}"
            )

        crop = cv2.resize(
            crop,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        if self.augment:

            # Horizontal flip.
            if random.random() < 0.5:

                crop = cv2.flip(
                    crop,
                    1,
                )

            # Mild brightness/contrast augmentation.
            if random.random() < 0.5:

                alpha = random.uniform(
                    0.85,
                    1.15,
                )

                beta = random.uniform(
                    -15,
                    15,
                )

                crop = cv2.convertScaleAbs(
                    crop,
                    alpha=alpha,
                    beta=beta,
                )

        crop = crop.astype(
            np.float32
        ) / 255.0

        # Convert HWC -> CHW.
        tensor = torch.from_numpy(
            np.ascontiguousarray(
                crop.transpose(2, 0, 1)
            )
        )

        label = torch.tensor(
            record["label"],
            dtype=torch.long,
        )

        return tensor, label


# ======================================================================
# 6. PATCH EMBEDDING
# ======================================================================

class PatchEmbedding(nn.Module):

    """
    Divide the image into non-overlapping patches.

    Input:
        [B, C, H, W]

    Output:
        [B, N, D]

    A convolution with kernel_size=P and stride=P performs the
    patch projection efficiently.

    This is mathematically equivalent to applying a linear projection
    to each flattened patch using shared weights.
    """

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,
        embed_dim=EMBED_DIM,
    ):

        super().__init__()

        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = (
            image_size // patch_size
        ) ** 2

        self.projection = nn.Conv2d(

            in_channels,

            embed_dim,

            kernel_size=patch_size,

            stride=patch_size,

        )

    def forward(self, x):

        # [B, C, H, W]
        x = self.projection(x)

        # [B, D, H/P, W/P]
        x = x.flatten(2)

        # [B, D, N] -> [B, N, D]
        x = x.transpose(1, 2)

        return x


# ======================================================================
# 7. MULTI-HEAD SELF-ATTENTION
# ======================================================================

class MultiHeadSelfAttention(nn.Module):

    """
    Self-attention implemented directly.

    Q = XWq
    K = XWk
    V = XWv

    Attention = softmax(QK^T / sqrt(head_dim)) V

    Multiple attention heads operate in parallel.
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

        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(
            embed_dim,
            3 * embed_dim,
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        self.projection = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.projection_dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        return_attention=False,
    ):

        batch_size, num_tokens, channels = x.shape

        # Combined QKV projection.
        qkv = self.qkv(x)

        # [B, N, 3D] -> [B, N, 3, H, Dh]
        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )

        # [3, B, H, N, Dh]
        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4,
        )

        query, key, value = qkv.unbind(
            dim=0
        )

        # [B, H, N, N]
        attention_logits = (
            query @ key.transpose(-2, -1)
        ) * self.scale

        attention = torch.softmax(
            attention_logits,
            dim=-1,
        )

        attention = self.attention_dropout(
            attention
        )

        # [B, H, N, Dh]
        output = attention @ value

        # [B, N, H, Dh]
        output = output.transpose(
            1,
            2,
        )

        # [B, N, D]
        output = output.reshape(
            batch_size,
            num_tokens,
            channels,
        )

        output = self.projection(
            output
        )

        output = self.projection_dropout(
            output
        )

        if return_attention:

            return output, attention

        return output


# ======================================================================
# 8. MLP BLOCK
# ======================================================================

class MLPBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        dropout=0.1,
    ):

        super().__init__()

        self.layers = nn.Sequential(

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

        return self.layers(x)


# ======================================================================
# 9. TRANSFORMER ENCODER BLOCK
# ======================================================================

class TransformerEncoderBlock(nn.Module):

    """
    Pre-Norm Transformer block.

    First sub-layer:

        X = X + Attention(LayerNorm(X))

    Second sub-layer:

        X = X + MLP(LayerNorm(X))
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4,
        dropout=0.1,
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.attention = MultiHeadSelfAttention(
            embed_dim,
            num_heads,
            dropout,
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        self.mlp = MLPBlock(
            embed_dim,
            embed_dim * mlp_ratio,
            dropout,
        )

    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# ======================================================================
# 10. VISION TRANSFORMER
# ======================================================================

class VisionTransformer(nn.Module):

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
        )

        num_patches = (
            image_size // patch_size
        ) ** 2

        # Learnable CLS token.
        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim,
            )
        )

        # Learnable positional embeddings.
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                embed_dim,
            )
        )

        self.position_dropout = nn.Dropout(
            dropout
        )

        self.encoder = nn.ModuleList([

            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )

            for _ in range(depth)

        ])

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.head = nn.Linear(
            embed_dim,
            num_classes,
        )

        self.initialize_weights()

    def initialize_weights(self):

        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
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

    def forward_features(
        self,
        x,
        return_attention=False,
    ):

        batch_size = x.shape[0]

        # Convert image into patch tokens.
        x = self.patch_embedding(x)

        # [B, N, D]

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        # Add CLS token.
        x = torch.cat(
            [
                cls_tokens,
                x,
            ],
            dim=1,
        )

        # Add positional information.
        x = x + self.position_embedding

        x = self.position_dropout(x)

        attention_maps = []

        for block in self.encoder:

            if return_attention:

                x = x + 0.0  # Explicit residual-path identity.

                normalized = block.norm1(x)

                _, attention = block.attention(
                    normalized,
                    return_attention=True,
                )

                attention_maps.append(
                    attention
                )

            x = block(x)

        x = self.norm(x)

        cls_representation = x[:, 0]

        if return_attention:

            return cls_representation, attention_maps

        return cls_representation

    def forward(self, x):

        features = self.forward_features(x)

        logits = self.head(
            features
        )

        return logits


# ======================================================================
# 11. METRICS
# ======================================================================

def calculate_metrics(
    predictions,
    targets,
    num_classes=NUM_CLASSES,
):

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    targets = np.asarray(
        targets,
        dtype=np.int64,
    )

    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    for target, prediction in zip(
        targets,
        predictions,
    ):

        confusion[target, prediction] += 1

    accuracy = (
        np.trace(confusion) / confusion.sum()
        if confusion.sum() > 0
        else 0.0
    )

    precision_values = []
    recall_values = []
    f1_values = []

    for class_id in range(num_classes):

        tp = confusion[class_id, class_id]

        fp = (
            confusion[:, class_id].sum() - tp
        )

        fn = (
            confusion[class_id, :].sum() - tp
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    return {

        "accuracy": float(accuracy),

        "macro_precision": float(
            np.mean(precision_values)
        ),

        "macro_recall": float(
            np.mean(recall_values)
        ),

        "macro_f1": float(
            np.mean(f1_values)
        ),

        "confusion_matrix": confusion,

    }


# ======================================================================
# 12. TRAINING
# ======================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    use_amp=False,
):

    model.train()

    total_loss = 0.0

    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if use_amp:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    labels,
                )

            scaler.scale(loss).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        total_loss += loss.item() * len(labels)

        predictions = logits.argmax(
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += len(labels)

    return {

        "loss": total_loss / max(total_samples, 1),

        "accuracy": total_correct / max(total_samples, 1),

    }


# ======================================================================
# 13. VALIDATION
# ======================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_targets = []

    all_images = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        predictions = logits.argmax(
            dim=1
        )

        total_loss += (
            loss.item() * len(labels)
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_targets.extend(
            labels.cpu().numpy().tolist()
        )

        all_images.extend(
            images.cpu()
        )

    metrics = calculate_metrics(
        all_predictions,
        all_targets,
    )

    metrics["loss"] = (
        total_loss / max(len(all_targets), 1)
    )

    metrics["predictions"] = all_predictions

    metrics["targets"] = all_targets

    metrics["images"] = all_images

    return metrics


# ======================================================================
# 14. VISUALIZATION
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

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy")

    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        history["train_accuracy"],
        label="Train",
    )

    axes[1].plot(
        history["val_accuracy"],
        label="Validation",
    )

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


def plot_confusion_matrix(
    confusion,
    output_path,
):

    fig, axis = plt.subplots(
        figsize=(8, 7)
    )

    image = axis.imshow(
        confusion,
        interpolation="nearest",
    )

    axis.set_title(
        "ViT Confusion Matrix"
    )

    axis.set_xlabel(
        "Predicted class"
    )

    axis.set_ylabel(
        "True class"
    )

    axis.set_xticks(
        range(NUM_CLASSES)
    )

    axis.set_yticks(
        range(NUM_CLASSES)
    )

    axis.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        CLASS_NAMES
    )

    for row in range(NUM_CLASSES):

        for column in range(NUM_CLASSES):

            axis.text(
                column,
                row,
                str(confusion[row, column]),
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=axis,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


def save_prediction_examples(
    images,
    predictions,
    targets,
    output_path,
    max_examples=16,
):

    count = min(
        max_examples,
        len(images),
    )

    if count == 0:
        return

    columns = 4

    rows = int(
        np.ceil(count / columns)
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 3.5 * rows),
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for index in range(count):

        image = images[index].numpy()

        image = image.transpose(
            1,
            2,
            0,
        )

        image = np.clip(
            image,
            0,
            1,
        )

        prediction = predictions[index]
        target = targets[index]

        axes[index].imshow(image)

        axes[index].set_title(

            f"True: {CLASS_NAMES[target]}\n"
            f"Pred: {CLASS_NAMES[prediction]}"

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
# 15. ATTENTION VISUALIZATION
# ======================================================================

@torch.no_grad()
def visualize_attention(
    model,
    image_tensor,
    device,
    output_path,
):

    """
    Visualize CLS-token attention to image patches.

    This is an attention visualization, not a guaranteed explanation
    of the model's decision.
    """

    model.eval()

    image_tensor = image_tensor.unsqueeze(
        0
    ).to(device)

    _, attention_maps = model.forward_features(
        image_tensor,
        return_attention=True,
    )

    # Use the final Transformer block.
    attention = attention_maps[-1]

    # [1, heads, tokens, tokens]
    attention = attention[0]

    # Average attention across heads.
    attention = attention.mean(
        dim=0
    )

    # CLS token attends to all patch tokens.
    cls_attention = attention[0, 1:]

    grid_size = (
        IMAGE_SIZE // PATCH_SIZE
    )

    attention_map = cls_attention.reshape(
        grid_size,
        grid_size,
    )

    attention_map = F.interpolate(

        attention_map[None, None],

        size=(IMAGE_SIZE, IMAGE_SIZE),

        mode="bilinear",

        align_corners=False,

    )[0, 0].cpu().numpy()

    image = image_tensor[0].cpu().numpy()

    image = image.transpose(
        1,
        2,
        0,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4),
    )

    axes[0].imshow(image)
    axes[0].set_title("Input image")

    axes[1].imshow(
        attention_map,
        cmap="viridis",
    )

    axes[1].set_title("CLS attention")

    axes[2].imshow(image)

    axes[2].imshow(
        attention_map,
        cmap="jet",
        alpha=0.45,
    )

    axes[2].set_title("Attention overlay")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


# ======================================================================
# 16. NUMERICAL TESTS
# ======================================================================

def run_tests():

    print("\n" + "=" * 70)
    print("LESSON 37 — NUMERICAL TESTS")
    print("=" * 70)

    device = torch.device("cpu")

    # --------------------------------------------------------------
    # Test 1: Patch embedding dimensions.
    # --------------------------------------------------------------

    patch_embedding = PatchEmbedding(
        image_size=128,
        patch_size=16,
        embed_dim=192,
    )

    x = torch.randn(
        2,
        3,
        128,
        128,
    )

    tokens = patch_embedding(x)

    assert tokens.shape == (
        2,
        64,
        192,
    )

    print("Test 1 passed: patch embedding dimensions.")

    # --------------------------------------------------------------
    # Test 2: Correct patch count.
    # --------------------------------------------------------------

    assert patch_embedding.num_patches == 64

    print("Test 2 passed: number of image patches.")

    # --------------------------------------------------------------
    # Test 3: Attention output dimensions.
    # --------------------------------------------------------------

    attention = MultiHeadSelfAttention(
        embed_dim=192,
        num_heads=6,
    )

    tokens = torch.randn(
        2,
        65,
        192,
    )

    output, attention_map = attention(
        tokens,
        return_attention=True,
    )

    assert output.shape == tokens.shape

    assert attention_map.shape == (
        2,
        6,
        65,
        65,
    )

    print("Test 3 passed: multi-head attention dimensions.")

    # --------------------------------------------------------------
    # Test 4: Attention rows sum to one.
    # --------------------------------------------------------------

    attention_sum = attention_map.sum(
        dim=-1
    )

    assert torch.allclose(
        attention_sum,
        torch.ones_like(attention_sum),
        atol=1e-5,
    )

    print("Test 4 passed: attention probability normalization.")

    # --------------------------------------------------------------
    # Test 5: Transformer block.
    # --------------------------------------------------------------

    block = TransformerEncoderBlock(
        embed_dim=192,
        num_heads=6,
    )

    x = torch.randn(
        2,
        65,
        192,
    )

    y = block(x)

    assert y.shape == x.shape

    print("Test 5 passed: Transformer encoder block.")

    # --------------------------------------------------------------
    # Test 6: ViT output shape.
    # --------------------------------------------------------------

    model = VisionTransformer(
        image_size=128,
        patch_size=16,
        num_classes=5,
        embed_dim=192,
        depth=2,
        num_heads=6,
    )

    x = torch.randn(
        2,
        3,
        128,
        128,
    )

    logits = model(x)

    assert logits.shape == (
        2,
        5,
    )

    print("Test 6 passed: ViT classification output.")

    # --------------------------------------------------------------
    # Test 7: CLS token.
    # --------------------------------------------------------------

    assert model.cls_token.shape == (
        1,
        1,
        192,
    )

    print("Test 7 passed: learnable CLS token.")

    # --------------------------------------------------------------
    # Test 8: Positional embedding.
    # --------------------------------------------------------------

    assert model.position_embedding.shape == (
        1,
        65,
        192,
    )

    print("Test 8 passed: positional embedding dimensions.")

    # --------------------------------------------------------------
    # Test 9: Attention visualization output.
    # --------------------------------------------------------------

    features, attention_maps = model.forward_features(
        x,
        return_attention=True,
    )

    assert features.shape == (
        2,
        192,
    )

    assert len(attention_maps) == 2

    print("Test 9 passed: attention extraction.")

    # --------------------------------------------------------------
    # Test 10: Gradient propagation.
    # --------------------------------------------------------------

    model.train()

    x = torch.randn(
        2,
        3,
        128,
        128,
    )

    labels = torch.tensor([
        0,
        2,
    ])

    logits = model(x)

    loss = F.cross_entropy(
        logits,
        labels,
    )

    loss.backward()

    assert model.patch_embedding.projection.weight.grad is not None

    assert model.cls_token.grad is not None

    assert model.position_embedding.grad is not None

    print("Test 10 passed: gradient propagation.")

    # --------------------------------------------------------------
    # Test 11: Optimizer update.
    # --------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    before = (
        model.head.weight.detach().clone()
    )

    optimizer.step()

    after = model.head.weight.detach()

    assert not torch.allclose(
        before,
        after,
    )

    print("Test 11 passed: optimizer parameter update.")

    # --------------------------------------------------------------
    # Test 12: Classification metrics.
    # --------------------------------------------------------------

    metrics = calculate_metrics(
        predictions=[0, 1, 2, 2],
        targets=[0, 1, 1, 2],
        num_classes=3,
    )

    assert np.isclose(
        metrics["accuracy"],
        0.75,
    )

    assert metrics["confusion_matrix"].shape == (
        3,
        3,
    )

    print("Test 12 passed: classification metrics.")

    print("\nALL TESTS PASSED.\n")


# ======================================================================
# 17. ARGUMENTS
# ======================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Lesson 37 — Vision Transformer from scratch."
    )

    parser.add_argument(
        "--taco-root",
        default="./TACO/data",
    )

    parser.add_argument(
        "--output",
        default="./runs/lesson37",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
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
        "--check-only",
        action="store_true",
    )

    return parser.parse_args()


# ======================================================================
# 18. MAIN
# ======================================================================

def main():

    args = parse_args()

    set_seed(42)

    print("=" * 70)
    print("LESSON 37 — VISION TRANSFORMER FROM SCRATCH")
    print("=" * 70)

    print(f"PyTorch: {torch.__version__}")

    print(f"Image size: {IMAGE_SIZE}")

    print(f"Patch size: {PATCH_SIZE}")

    print(f"Number of patches: {NUM_PATCHES}")

    print(f"Embedding dimension: {EMBED_DIM}")

    print(f"Attention heads: {NUM_HEADS}")

    print(f"Transformer depth: {DEPTH}")

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

    print(f"\nDevice: {device}")

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Load real TACO object crops.
    # --------------------------------------------------------------

    image_records, object_records = load_taco_object_records(

        taco_root=args.taco_root,

        max_images=args.max_images,

    )

    train_records, val_records = split_by_image(

        image_records,

        object_records,

    )

    print(
        f"Source images: {len(image_records)}"
    )

    print(
        f"Training object crops: {len(train_records)}"
    )

    print(
        f"Validation object crops: {len(val_records)}"
    )

    train_counts = Counter(
        record["label"]
        for record in train_records
    )

    val_counts = Counter(
        record["label"]
        for record in val_records
    )

    print("\nTRAINING CLASS COUNTS")

    for class_id, name in enumerate(CLASS_NAMES):

        print(
            f"{name:20s}: {train_counts[class_id]}"
        )

    print("\nVALIDATION CLASS COUNTS")

    for class_id, name in enumerate(CLASS_NAMES):

        print(
            f"{name:20s}: {val_counts[class_id]}"
        )

    summary = {

        "source_images": len(image_records),

        "training_crops": len(train_records),

        "validation_crops": len(val_records),

        "training_class_counts": {
            CLASS_NAMES[key]: value
            for key, value in train_counts.items()
        },

        "validation_class_counts": {
            CLASS_NAMES[key]: value
            for key, value in val_counts.items()
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
    # Dataset and DataLoaders.
    # --------------------------------------------------------------

    train_dataset = TACOObjectDataset(
        train_records,
        augment=True,
    )

    val_dataset = TACOObjectDataset(
        val_records,
        augment=False,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch,

        shuffle=True,

        num_workers=args.workers,

        pin_memory=pin_memory,

        drop_last=False,

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=args.batch,

        shuffle=False,

        num_workers=args.workers,

        pin_memory=pin_memory,

    )

    # --------------------------------------------------------------
    # Model.
    # --------------------------------------------------------------

    model = VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTrainable parameters: {parameter_count:,}"
    )

    # --------------------------------------------------------------
    # Class weights for imbalance.
    # --------------------------------------------------------------

    counts = np.asarray(
        [
            train_counts.get(class_id, 0)
            for class_id in range(NUM_CLASSES)
        ],
        dtype=np.float32,
    )

    class_weights = np.zeros(
        NUM_CLASSES,
        dtype=np.float32,
    )

    present = counts > 0

    if np.any(present):

        total = counts[present].sum()

        class_weights[present] = (
            total
            /
            (
                np.sum(present)
                * counts[present]
            )
        )

    # Classes absent from the training split receive zero weight.
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=args.lr,

        weight_decay=args.weight_decay,

    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=args.epochs,

    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    history = {

        "train_loss": [],

        "val_loss": [],

        "train_accuracy": [],

        "val_accuracy": [],

    }

    best_accuracy = -1.0

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

            optimizer,

            criterion,

            device,

            scaler=scaler,

            use_amp=use_amp,

        )

        val_metrics = evaluate(

            model,

            val_loader,

            criterion,

            device,

        )

        scheduler.step()

        history["train_loss"].append(
            train_metrics["loss"]
        )

        history["val_loss"].append(
            val_metrics["loss"]
        )

        history["train_accuracy"].append(
            train_metrics["accuracy"]
        )

        history["val_accuracy"].append(
            val_metrics["accuracy"]
        )

        print(
            f"\nEpoch {epoch}/{args.epochs}"
        )

        print(
            f"Train loss: {train_metrics['loss']:.4f}"
        )

        print(
            f"Train accuracy: {train_metrics['accuracy']:.4f}"
        )

        print(
            f"Val loss: {val_metrics['loss']:.4f}"
        )

        print(
            f"Val accuracy: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_accuracy:

            best_accuracy = val_metrics["accuracy"]

            torch.save(

                {

                    "model_state_dict": model.state_dict(),

                    "epoch": epoch,

                    "validation_accuracy": best_accuracy,

                    "class_names": CLASS_NAMES,

                    "image_size": IMAGE_SIZE,

                    "patch_size": PATCH_SIZE,

                    "embed_dim": EMBED_DIM,

                    "depth": DEPTH,

                    "num_heads": NUM_HEADS,

                },

                best_path,

            )

            print(
                f"Saved best model: {best_path}"
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
    # Load best model.
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

    final_metrics = evaluate(

        model,

        val_loader,

        criterion,

        device,

    )

    print("\nFINAL VALIDATION METRICS")

    print(
        f"Accuracy: {final_metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision: {final_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall: {final_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1: {final_metrics['macro_f1']:.4f}"
    )

    plot_confusion_matrix(

        final_metrics["confusion_matrix"],

        output_dir / "confusion_matrix.png",

    )

    save_prediction_examples(

        final_metrics["images"],

        final_metrics["predictions"],

        final_metrics["targets"],

        output_dir / "predictions.png",

    )

    # --------------------------------------------------------------
    # Attention visualization.
    # --------------------------------------------------------------

    sample_image, _ = val_dataset[0]

    visualize_attention(

        model,

        sample_image,

        device,

        output_dir / "attention_visualization.png",

    )

    # --------------------------------------------------------------
    # Save evaluation metrics.
    # --------------------------------------------------------------

    evaluation = {

        "accuracy": final_metrics["accuracy"],

        "macro_precision": final_metrics["macro_precision"],

        "macro_recall": final_metrics["macro_recall"],

        "macro_f1": final_metrics["macro_f1"],

        "best_epoch": checkpoint["epoch"],

        "trainable_parameters": parameter_count,

    }

    with (
        output_dir / "evaluation.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            evaluation,
            file,
            indent=2,
        )

    print("\nLESSON 37 COMPLETED")

    print(
        f"Best checkpoint: {best_path.resolve()}"
    )

    print(
        f"Output directory: {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()