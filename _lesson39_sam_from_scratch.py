"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 39 — SAM-INSPIRED PROMPTABLE SEGMENTATION
======================================================================

Filename:
    lesson39_sam_from_scratch.py

DATA:
    Real TACO images and polygon annotations.

MODEL:
    Lightweight SAM-inspired promptable segmentation network.

INITIALIZATION:
    Random weights. No pretrained model or pretrained weights.

======================================================================
1. LEARNING OBJECTIVES
======================================================================

Build a promptable image segmentation system.

Input:
    RGB image
    Bounding-box prompt
    Foreground point prompt

Output:
    Binary object segmentation mask

Architecture:

    RGB IMAGE
        |
        v
    IMAGE ENCODER
        |
        v
    IMAGE FEATURES
        |
        +----------------------+
                               |
    BOX PROMPT                 |
        |                      |
        v                      |
    PROMPT ENCODER              |
        |                      |
    POINT PROMPT                |
        |                      |
        v                      |
    PROMPT TOKENS               |
        |                      |
        +----------+-----------+
                   |
                   v
          CROSS ATTENTION
                   |
                   v
             MASK DECODER
                   |
                   v
             OBJECT MASK

======================================================================
2. WHAT IS SAM?
======================================================================

SAM means Segment Anything Model.

The original SAM architecture contains:

    1. Image Encoder
    2. Prompt Encoder
    3. Mask Decoder

The image encoder extracts visual features.

The prompt encoder represents user-provided prompts, such as:

    - Foreground points
    - Background points
    - Bounding boxes
    - Mask prompts

The mask decoder combines image and prompt information
to predict segmentation masks.

======================================================================
3. IMAGE ENCODER
======================================================================

We use a lightweight CNN.

It converts:

    B x 3 x H x W

into:

    B x D x H' x W'

This is the visual feature representation.

======================================================================
4. PROMPT ENCODER
======================================================================

Bounding-box prompt:

    [x1, y1, x2, y2]

Coordinates are normalized to [0,1].

Point prompt:

    [x, y]

The box and point are transformed into learnable embeddings.

======================================================================
5. PROMPT-IMAGE CROSS ATTENTION
======================================================================

The prompt embeddings are used as queries.

The image features are keys and values.

    Q = PromptEmbedding
    K = ImageFeatures
    V = ImageFeatures

    Attention(Q,K,V) =
        softmax(QK^T / sqrt(d))V

This allows prompts to retrieve relevant image information.

======================================================================
6. MASK DECODER
======================================================================

The decoder combines:

    - Image features
    - Prompt-conditioned representation

It produces one mask logit per image pixel.

The final probability is:

    P(mask) = sigmoid(mask_logits)

======================================================================
7. TRAINING
======================================================================

TACO provides real polygon annotations.

We convert each polygon into a binary mask.

Training prompts are derived from the ground-truth bounding box
and object center.

This is supervised prompt-conditioned segmentation.

IMPORTANT:
    Training uses ground-truth-derived prompts.
    During inference, prompts can come from a human or detector.

======================================================================
8. LOSS FUNCTION
======================================================================

Binary Cross Entropy:

    L_BCE = BCEWithLogits(prediction, target)

Dice loss:

    Dice = (2 * intersection + epsilon)
           / (prediction_sum + target_sum + epsilon)

    L_Dice = 1 - Dice

Total:

    L = L_BCE + L_Dice

======================================================================
9. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    Camera
       |
       v
    Object Detector
       |
       v
    Bounding Box Prompt
       |
       v
    Promptable Segmentation
       |
       v
    Object Mask
       |
       v
    Object Center / Grasp Region
       |
       v
    Camera-to-Robot Calibration
       |
       v
    ACT Manipulation Policy

======================================================================
10. LIMITATIONS
======================================================================

This is a compact educational SAM-inspired model.

It is NOT the original SAM architecture.

It does not implement:
    - Original SAM image encoder
    - Original Fourier positional encoding
    - Original two-way transformer
    - Multiple mask hypotheses
    - Predicted mask quality scores
    - Iterative mask prompting

======================================================================
"""

import argparse
import json
import random
import time

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# =====================================================================
# CONFIGURATION
# =====================================================================

SEED = 42

IMAGE_SIZE = 256

EMBED_DIM = 128

BATCH_SIZE = 4

EPOCHS = 30

LEARNING_RATE = 1e-4

OUTPUT_DIR = Path("runs/lesson39")

TACO_ROOT = Path("./TACO/data")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# BOX UTILITIES
# =====================================================================

def normalize_box(box, width, height):

    x, y, w, h = box

    return np.array(
        [
            x / width,
            y / height,
            (x + w) / width,
            (y + h) / height,
        ],
        dtype=np.float32
    ).clip(0.0, 1.0)


def box_center(box):

    x1, y1, x2, y2 = box

    return np.array(
        [
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        ],
        dtype=np.float32
    )


# =====================================================================
# POLYGON TO MASK
# =====================================================================

def polygons_to_mask(polygons, width, height):

    mask = Image.new(
        "L",
        (width, height),
        0
    )

    draw = ImageDraw.Draw(mask)

    for polygon in polygons:

        if not isinstance(polygon, list):
            continue

        if len(polygon) < 6:
            continue

        points = []

        for i in range(0, len(polygon) - 1, 2):

            points.append(
                (
                    float(polygon[i]),
                    float(polygon[i + 1])
                )
            )

        if len(points) >= 3:

            draw.polygon(
                points,
                fill=1
            )

    return np.asarray(
        mask,
        dtype=np.uint8
    )


# =====================================================================
# DATASET
# =====================================================================

class TACOPromptDataset(Dataset):

    def __init__(
        self,
        root,
        max_images=0,
        max_instances=0,
        augment=False
    ):

        self.root = Path(root)

        annotation_file = self.root / "annotations.json"

        if not annotation_file.exists():

            raise FileNotFoundError(
                f"Cannot find {annotation_file}"
            )

        with open(annotation_file, "r") as f:

            data = json.load(f)

        images = {
            item["id"]: item
            for item in data["images"]
        }

        categories = {
            item["id"]: item["name"]
            for item in data["categories"]
        }

        image_annotations = {}

        for ann in data["annotations"]:

            if ann.get("iscrowd", 0):
                continue

            segmentation = ann.get("segmentation")

            if not isinstance(segmentation, list):
                continue

            valid_polygons = [
                p for p in segmentation
                if isinstance(p, list) and len(p) >= 6
            ]

            if not valid_polygons:
                continue

            image_id = ann["image_id"]

            if image_id not in images:
                continue

            image_annotations.setdefault(
                image_id,
                []
            ).append(ann)

        image_ids = list(image_annotations.keys())

        random.shuffle(image_ids)

        if max_images > 0:
            image_ids = image_ids[:max_images]

        self.samples = []

        for image_id in image_ids:

            image_info = images[image_id]

            for ann in image_annotations[image_id]:

                bbox = ann.get("bbox")

                if bbox is None or len(bbox) != 4:
                    continue

                x, y, w, h = map(float, bbox)

                if w < 3 or h < 3:
                    continue

                if w * h < 20:
                    continue

                self.samples.append(
                    {
                        "image": image_info,
                        "annotation": ann,
                        "category": categories.get(
                            ann["category_id"],
                            "unknown"
                        )
                    }
                )

                if (
                    max_instances > 0
                    and len(self.samples) >= max_instances
                ):
                    break

            if (
                max_instances > 0
                and len(self.samples) >= max_instances
            ):
                break

        if not self.samples:

            raise RuntimeError(
                "No valid polygon instances found in TACO."
            )

        self.augment = augment

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        info = sample["image"]

        ann = sample["annotation"]

        filename = self.root / info["file_name"]

        if not filename.exists():

            raise FileNotFoundError(filename)

        image = Image.open(filename).convert("RGB")

        width, height = image.size

        polygons = ann["segmentation"]

        mask = polygons_to_mask(
            polygons,
            width,
            height
        )

        x, y, w, h = map(float, ann["bbox"])

        x1 = max(0, int(np.floor(x)))
        y1 = max(0, int(np.floor(y)))

        x2 = min(width, int(np.ceil(x + w)))
        y2 = min(height, int(np.ceil(y + h)))

        if x2 <= x1 or y2 <= y1:

            raise RuntimeError(
                "Invalid bounding box encountered."
            )

        # Preserve the full image.
        # The box prompt tells the model which object to segment.

        box = normalize_box(
            [x, y, w, h],
            width,
            height
        )

        point = box_center(box)

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        mask_image = Image.fromarray(
            mask * 255
        )

        mask_image = mask_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        mask_array = (
            np.asarray(mask_image, dtype=np.float32)
            / 255.0
        )

        if self.augment and random.random() < 0.5:

            image_array = np.fliplr(
                image_array
            ).copy()

            mask_array = np.fliplr(
                mask_array
            ).copy()

            box[0], box[2] = (
                1.0 - box[2],
                1.0 - box[0]
            )

            point[0] = 1.0 - point[0]

        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1)
        ).float()

        mask_tensor = torch.from_numpy(
            mask_array
        ).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "box": torch.from_numpy(box).float(),
            "point": torch.from_numpy(point).float(),
            "category": sample["category"],
        }


# =====================================================================
# IMAGE ENCODER
# =====================================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.GELU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.GELU()
        )

    def forward(self, x):

        return self.block(x)


class ImageEncoder(nn.Module):

    def __init__(self, embed_dim=EMBED_DIM):

        super().__init__()

        self.network = nn.Sequential(

            ConvBlock(3, 32),

            ConvBlock(32, 64),

            ConvBlock(64, embed_dim),

        )

    def forward(self, image):

        return self.network(image)


# =====================================================================
# PROMPT ENCODER
# =====================================================================

class PromptEncoder(nn.Module):

    def __init__(self, embed_dim=EMBED_DIM):

        super().__init__()

        self.box_encoder = nn.Sequential(

            nn.Linear(4, embed_dim),

            nn.GELU(),

            nn.Linear(embed_dim, embed_dim)
        )

        self.point_encoder = nn.Sequential(

            nn.Linear(2, embed_dim),

            nn.GELU(),

            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, box, point):

        box_token = self.box_encoder(box)

        point_token = self.point_encoder(point)

        tokens = torch.stack(
            [
                box_token,
                point_token
            ],
            dim=1
        )

        return tokens


# =====================================================================
# PROMPT-IMAGE CROSS ATTENTION
# =====================================================================

class PromptCrossAttention(nn.Module):

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_heads=4
    ):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(

            nn.Linear(embed_dim, embed_dim * 4),

            nn.GELU(),

            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, prompt_tokens, image_tokens):

        attended, weights = self.attention(

            query=prompt_tokens,

            key=image_tokens,

            value=image_tokens,

            need_weights=True
        )

        x = self.norm1(
            prompt_tokens + attended
        )

        x = self.norm2(
            x + self.mlp(x)
        )

        return x, weights


# =====================================================================
# MASK DECODER
# =====================================================================

class MaskDecoder(nn.Module):

    def __init__(self, embed_dim=EMBED_DIM):

        super().__init__()

        self.film = nn.Sequential(

            nn.Linear(embed_dim, embed_dim * 2),

            nn.GELU(),

            nn.Linear(embed_dim * 2, embed_dim * 2)
        )

        self.mask_head = nn.Sequential(

            nn.Conv2d(
                embed_dim,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                32,
                1,
                kernel_size=1
            )
        )

    def forward(self, image_features, prompt_features, output_size):

        # Combine box and point prompt representations.

        prompt = prompt_features.mean(dim=1)

        gamma, beta = self.film(prompt).chunk(
            2,
            dim=-1
        )

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)

        beta = beta.unsqueeze(-1).unsqueeze(-1)

        conditioned = (
            image_features * (1.0 + gamma)
            + beta
        )

        logits = self.mask_head(conditioned)

        logits = F.interpolate(

            logits,

            size=output_size,

            mode="bilinear",

            align_corners=False
        )

        return logits


# =====================================================================
# COMPLETE PROMPTABLE SEGMENTATION MODEL
# =====================================================================

class PromptableSegmenter(nn.Module):

    def __init__(self):

        super().__init__()

        self.image_encoder = ImageEncoder()

        self.prompt_encoder = PromptEncoder()

        self.cross_attention = PromptCrossAttention()

        self.mask_decoder = MaskDecoder()

    def forward(self, image, box, point):

        image_features = self.image_encoder(image)

        batch, channels, height, width = (
            image_features.shape
        )

        image_tokens = image_features.flatten(
            2
        ).transpose(1, 2)

        prompt_tokens = self.prompt_encoder(
            box,
            point
        )

        prompt_features, attention = (
            self.cross_attention(
                prompt_tokens,
                image_tokens
            )
        )

        mask_logits = self.mask_decoder(

            image_features,

            prompt_features,

            output_size=(
                image.shape[-2],
                image.shape[-1]
            )
        )

        return {
            "mask_logits": mask_logits,
            "attention": attention,
        }


# =====================================================================
# DICE LOSS
# =====================================================================

def dice_loss(logits, targets, epsilon=1e-6):

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.flatten(1)

    targets = targets.flatten(1)

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    denominator = (
        probabilities.sum(dim=1)
        +
        targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + epsilon
    ) / (
        denominator + epsilon
    )

    return (1.0 - dice).mean()


def segmentation_loss(logits, targets):

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets
    )

    dice = dice_loss(
        logits,
        targets
    )

    total = bce + dice

    return total, bce.detach(), dice.detach()


# =====================================================================
# METRICS
# =====================================================================

@torch.no_grad()
def calculate_metrics(logits, targets):

    predictions = (
        torch.sigmoid(logits) >= 0.5
    )

    targets = targets >= 0.5

    predictions = predictions.flatten(1)

    targets = targets.flatten(1)

    intersection = (
        predictions & targets
    ).sum(dim=1).float()

    union = (
        predictions | targets
    ).sum(dim=1).float()

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        predictions.sum(dim=1).float()
        +
        targets.sum(dim=1).float()
        + 1e-6
    )

    iou = (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )

    return dice.mean().item(), iou.mean().item()


# =====================================================================
# TRAINING
# =====================================================================

def train_one_epoch(model, loader, optimizer):

    model.train()

    total_loss = 0.0

    total_bce = 0.0

    total_dice = 0.0

    count = 0

    for batch in loader:

        image = batch["image"].to(DEVICE)

        mask = batch["mask"].to(DEVICE)

        box = batch["box"].to(DEVICE)

        point = batch["point"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        output = model(
            image,
            box,
            point
        )

        loss, bce, dice = segmentation_loss(

            output["mask_logits"],

            mask
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        total_bce += bce.item()

        total_dice += dice.item()

        count += 1

    return {
        "loss": total_loss / max(count, 1),
        "bce": total_bce / max(count, 1),
        "dice_loss": total_dice / max(count, 1),
    }


# =====================================================================
# VALIDATION
# =====================================================================

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    total_loss = 0.0

    total_dice = 0.0

    total_iou = 0.0

    count = 0

    for batch in loader:

        image = batch["image"].to(DEVICE)

        mask = batch["mask"].to(DEVICE)

        box = batch["box"].to(DEVICE)

        point = batch["point"].to(DEVICE)

        output = model(
            image,
            box,
            point
        )

        logits = output["mask_logits"]

        loss, _, _ = segmentation_loss(
            logits,
            mask
        )

        dice, iou = calculate_metrics(
            logits,
            mask
        )

        total_loss += loss.item()

        total_dice += dice

        total_iou += iou

        count += 1

    return {
        "loss": total_loss / max(count, 1),
        "dice": total_dice / max(count, 1),
        "iou": total_iou / max(count, 1),
    }


# =====================================================================
# VISUALIZATION
# =====================================================================

@torch.no_grad()
def save_predictions(model, loader, output_path, device):

    model.eval()

    batch = next(iter(loader))

    image = batch["image"].to(device)

    mask = batch["mask"].to(device)

    box = batch["box"].to(device)

    point = batch["point"].to(device)

    output = model(
        image,
        box,
        point
    )

    predictions = (
        torch.sigmoid(output["mask_logits"]) >= 0.5
    ).float()

    count = min(4, image.shape[0])

    fig, axes = plt.subplots(
        count,
        4,
        figsize=(14, 4 * count),
        squeeze=False
    )

    for i in range(count):

        img = (
            image[i]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        gt = mask[i, 0].cpu().numpy()

        pred = predictions[i, 0].cpu().numpy()

        h, w = gt.shape

        x1, y1, x2, y2 = (
            box[i].cpu().numpy()
        )

        p_x, p_y = (
            point[i].cpu().numpy()
        )

        axes[i, 0].imshow(img)

        axes[i, 0].add_patch(
            plt.Rectangle(
                (x1 * w, y1 * h),
                (x2 - x1) * w,
                (y2 - y1) * h,
                fill=False,
                linewidth=2
            )
        )

        axes[i, 0].scatter(
            [p_x * w],
            [p_y * h],
            marker="+",
            s=100
        )

        axes[i, 0].set_title(
            "Image + Box + Point Prompt"
        )

        axes[i, 1].imshow(
            gt,
            cmap="gray"
        )

        axes[i, 1].set_title(
            "Ground Truth"
        )

        axes[i, 2].imshow(
            pred,
            cmap="gray"
        )

        axes[i, 2].set_title(
            "Predicted Mask"
        )

        overlay = img.copy()

        overlay[pred > 0.5] = (
            0.5 * overlay[pred > 0.5]
            + 0.5 * np.array([0, 1, 0])
        )

        axes[i, 3].imshow(overlay)

        axes[i, 3].set_title(
            "Segmentation Overlay"
        )

        for ax in axes[i]:

            ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()


# =====================================================================
# TRAINING CURVES
# =====================================================================

def save_training_curves(history, output_path):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Train"
    )

    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Validation"
    )

    axes[0].set_title("Segmentation Loss")

    axes[0].set_xlabel("Epoch")

    axes[0].legend()

    axes[1].plot(
        epochs,
        history["val_dice"],
        label="Dice"
    )

    axes[1].plot(
        epochs,
        history["val_iou"],
        label="IoU"
    )

    axes[1].set_title("Validation Metrics")

    axes[1].set_xlabel("Epoch")

    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()


# =====================================================================
# DATASET SUMMARY
# =====================================================================

def save_dataset_summary(dataset, output_path):

    categories = {}

    for sample in dataset.samples:

        category = sample["category"]

        categories[category] = (
            categories.get(category, 0) + 1
        )

    summary = {
        "num_instances": len(dataset),
        "categories": categories,
        "image_size": IMAGE_SIZE,
        "training_mode": "real TACO polygon masks",
    }

    with open(output_path, "w") as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print("\nDataset summary:")

    print(json.dumps(summary, indent=4))


# =====================================================================
# NUMERICAL TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print("LESSON 39 — NUMERICAL TESTS")

    print("=" * 70)

    # Test 1: box normalization.

    box = normalize_box(
        [10, 20, 30, 40],
        100,
        100
    )

    assert np.allclose(
        box,
        [0.1, 0.2, 0.4, 0.6]
    )

    print("Test 1 passed: box normalization")

    # Test 2: box center.

    center = box_center(box)

    assert np.allclose(
        center,
        [0.25, 0.4]
    )

    print("Test 2 passed: box center")

    # Test 3: polygon mask.

    polygon = [
        [
            10, 10,
            40, 10,
            40, 40,
            10, 40
        ]
    ]

    mask = polygons_to_mask(
        polygon,
        64,
        64
    )

    assert mask.shape == (64, 64)

    assert mask.sum() > 0

    print("Test 3 passed: polygon rasterization")

    # Test 4: IoU.

    a = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0]]
    )

    b = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0]]
    )

    iou = box_iou(a, b)

    assert torch.allclose(
        iou,
        torch.ones_like(iou)
    )

    print("Test 4 passed: identical-box IoU")

    # Test 5: GIoU.

    giou = generalized_box_iou(a, b)

    assert torch.allclose(
        giou,
        torch.ones_like(giou),
        atol=1e-5
    )

    print("Test 5 passed: identical-box GIoU")

    # Test 6: model output.

    model = PromptableSegmenter()

    images = torch.rand(
        2, 3, IMAGE_SIZE, IMAGE_SIZE
    )

    boxes = torch.rand(2, 4)

    points = torch.rand(2, 2)

    output = model(
        images,
        boxes,
        points
    )

    assert output["mask_logits"].shape == (
        2, 1, IMAGE_SIZE, IMAGE_SIZE
    )

    print("Test 6 passed: model output dimensions")

    # Test 7: attention dimensions.

    assert output["attention"].shape == (
        2, 2, IMAGE_SIZE // 4 * (IMAGE_SIZE // 4)
    )

    print("Test 7 passed: prompt-image attention dimensions")

    # Test 8: loss and gradients.

    targets = torch.randint(
        0,
        2,
        (2, 1, IMAGE_SIZE, IMAGE_SIZE)
    ).float()

    loss, _, _ = segmentation_loss(
        output["mask_logits"],
        targets
    )

    assert torch.isfinite(loss)

    loss.backward()

    assert model.mask_decoder.mask_head[-1].weight.grad is not None

    print("Test 8 passed: loss and gradients")

    # Test 9: Dice metric.

    dice, metric_iou = calculate_metrics(
        output["mask_logits"].detach(),
        targets
    )

    assert 0.0 <= dice <= 1.0

    assert 0.0 <= metric_iou <= 1.0

    print("Test 9 passed: Dice and IoU metrics")

    print("\nAll 9 tests passed.\n")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--taco-root",
        type=str,
        default=str(TACO_ROOT)
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH_SIZE
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=0
    )

    parser.add_argument(
        "--max-instances",
        type=int,
        default=0
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--check-only",
        action="store_true"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto"
    )

    args = parser.parse_args()

    set_seed()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    global DEVICE

    if args.device == "cpu":

        DEVICE = torch.device("cpu")

    elif args.device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested but not available."
            )

        DEVICE = torch.device("cuda")

    else:

        DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print("=" * 70)

    print("LESSON 39 — SAM-INSPIRED PROMPTABLE SEGMENTATION")

    print("=" * 70)

    print("Device:", DEVICE)

    print("Image size:", IMAGE_SIZE)

    print("Embedding dimension:", EMBED_DIM)

    print("TACO root:", args.taco_root)

    run_tests()

    if args.check_only:

        print("Check-only mode completed.")

        return

    # ---------------------------------------------------------------
    # DATASET
    # ---------------------------------------------------------------

    dataset = TACOPromptDataset(

        root=args.taco_root,

        max_images=args.max_images,

        max_instances=args.max_instances,

        augment=False
    )

    save_dataset_summary(
        dataset,
        OUTPUT_DIR / "dataset_summary.json"
    )

    indices = list(range(len(dataset)))

    random.shuffle(indices)

    if len(indices) < 2:

        raise RuntimeError(
            "At least two valid object instances are required."
        )

    split = max(
        1,
        int(len(indices) * 0.8)
    )

    split = min(
        split,
        len(indices) - 1
    )

    train_indices = indices[:split]

    val_indices = indices[split:]

    train_dataset = torch.utils.data.Subset(
        dataset,
        train_indices
    )

    val_dataset = torch.utils.data.Subset(
        dataset,
        val_indices
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch,

        shuffle=True,

        num_workers=args.workers,

        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=args.batch,

        shuffle=False,

        num_workers=args.workers,

        pin_memory=(DEVICE.type == "cuda")
    )

    print("\nTraining instances:", len(train_dataset))

    print("Validation instances:", len(val_dataset))

    # ---------------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------------

    model = PromptableSegmenter().to(DEVICE)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTrainable parameters:", f"{parameters:,}")

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=max(args.epochs, 1)
    )

    # ---------------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
    }

    best_dice = -1.0

    for epoch in range(args.epochs):

        start = time.time()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        val_metrics = evaluate(
            model,
            val_loader
        )

        scheduler.step()

        history["train_loss"].append(
            train_metrics["loss"]
        )

        history["val_loss"].append(
            val_metrics["loss"]
        )

        history["val_dice"].append(
            val_metrics["dice"]
        )

        history["val_iou"].append(
            val_metrics["iou"]
        )

        elapsed = time.time() - start

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"Train Loss {train_metrics['loss']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Dice {val_metrics['dice']:.4f} | "
            f"IoU {val_metrics['iou']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["dice"] > best_dice:

            best_dice = val_metrics["dice"]

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_dice": best_dice,
                    "image_size": IMAGE_SIZE,
                },
                OUTPUT_DIR / "best_model.pt"
            )

    # ---------------------------------------------------------------
    # FINAL EVALUATION
    # ---------------------------------------------------------------

    checkpoint = torch.load(
        OUTPUT_DIR / "best_model.pt",
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    final_metrics = evaluate(
        model,
        val_loader
    )

    print("\nBest checkpoint epoch:", checkpoint["epoch"])

    print("Final validation Dice:", final_metrics["dice"])

    print("Final validation IoU:", final_metrics["iou"])

    # ---------------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------------

    save_training_curves(
        history,
        OUTPUT_DIR / "training_curves.png"
    )

    save_predictions(
        model,
        val_loader,
        OUTPUT_DIR / "predictions.png",
        DEVICE
    )

    print("\nOutputs saved to:", OUTPUT_DIR)

    print("  best_model.pt")

    print("  training_curves.png")

    print("  predictions.png")

    print("  dataset_summary.json")

    print("\nLesson 39 completed.")


if __name__ == "__main__":

    main()