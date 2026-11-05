"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 40 — OPEN-VOCABULARY OBJECT DETECTION
======================================================================

Filename:
    lesson40_open_vocabulary_object_detection.py

PHASE:
    Phase 3 — Advanced Detection and Segmentation

DATA:
    Real TACO waste images and COCO-format annotations.

MODEL:
    Lightweight CLIP-inspired image-text model.

INITIALIZATION:
    Random weights. No pretrained models or pretrained weights.

======================================================================
1. LEARNING OBJECTIVES
======================================================================

Build a simplified vision-language object detection pipeline.

We implement:

    1. CNN image encoder
    2. Character-level text encoder
    3. Image-text embedding alignment
    4. Contrastive learning
    5. Sliding-window region proposals
    6. Text-image similarity scoring
    7. Non-Maximum Suppression
    8. Text-prompted object detection

======================================================================
2. WHAT IS OPEN-VOCABULARY DETECTION?
======================================================================

Traditional object detector:

    Image -> Detector -> Fixed Class IDs

For example:

    0 = Plastic
    1 = Metal
    2 = Paper
    3 = Glass

The model is restricted to its trained categories.

Open-vocabulary detection:

    Image + Text Query -> Object Locations

Example:

    Query: "plastic bottle"

    Output:
        Bounding box 1
        Similarity score 1

        Bounding box 2
        Similarity score 2

The model uses a shared image-text embedding space.

======================================================================
3. IMAGE-TEXT CONTRASTIVE LEARNING
======================================================================

The image encoder produces:

    image_embedding = f(image)

The text encoder produces:

    text_embedding = g(text)

Both embeddings are normalized:

    z_image = f(image) / ||f(image)||

    z_text = g(text) / ||g(text)||

Similarity:

    similarity = z_image * z_text

For a batch of N image-text pairs:

    S = Z_image @ Z_text.T

The diagonal represents matching pairs.

We use a multi-positive contrastive objective because
multiple images can share the same category.

======================================================================
4. WHY CHARACTER-LEVEL TEXT ENCODING?
======================================================================

We use a small character-level GRU.

Advantages:

    - No external tokenizer.
    - No pretrained language model.
    - Can process category names not seen as exact strings.
    - Simple and easy to inspect.

Limitations:

    Character-level encoding does NOT automatically understand
    semantic relationships.

    For example, knowing "bottle" does not guarantee that the
    model understands "container" or "recyclable packaging".

======================================================================
5. REGION PROPOSALS
======================================================================

This educational implementation uses multi-scale sliding windows.

For each image:

    1. Generate square candidate regions.
    2. Crop each region.
    3. Encode each crop.
    4. Compare with text query embeddings.
    5. Keep high-scoring regions.
    6. Apply NMS.

This is intentionally simple.

Production open-vocabulary detectors typically use learned
region proposals, feature pyramids, or transformer-based
region-text alignment.

======================================================================
6. IMPORTANT LIMITATION
======================================================================

This is a small educational vision-language detector.

It is NOT:

    - CLIP
    - OWL-ViT
    - Grounding DINO
    - GLIP

It does not provide the semantic generalization of those
large-scale pretrained models.

Its text queries are flexible at inference, but detection
quality depends on the visual concepts learned from TACO.

The model is trained on real TACO object crops and category names.

======================================================================
7. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    RGB Camera
        |
        v
    Text Query
        |
        v
    Vision-Language Detector
        |
        v
    Bounding Box + Similarity
        |
        v
    Object Segmentation
        |
        v
    3D Localization
        |
        v
    Grasp Planning
        |
        v
    ACT Policy
        |
        v
    OpenArm V2

Example:

    Query: "plastic bottle"

    The detector searches for regions that visually match
    the text description.

======================================================================
8. OUTPUTS
======================================================================

runs/lesson40/
    best_model.pt
    training_curves.png
    detection_preview.png
    dataset_summary.json

======================================================================
9. RUN
======================================================================

Tests:

    python lesson40_open_vocabulary_object_detection.py --check-only

Small experiment:

    python lesson40_open_vocabulary_object_detection.py \
        --max-images 200 \
        --max-instances 1000 \
        --epochs 5

Full experiment:

    python lesson40_open_vocabulary_object_detection.py \
        --epochs 30 \
        --batch 32

Detection queries:

    python lesson40_open_vocabulary_object_detection.py \
        --inference-only \
        --query "plastic bottle;aluminum can;glass bottle"

======================================================================
"""

import argparse
import json
import math
import random
import time

from collections import Counter
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

IMAGE_SIZE = 128

EMBED_DIM = 128

TEXT_LENGTH = 64

TEXT_VOCAB_SIZE = 129

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 1e-3

TEMPERATURE = 0.07

OUTPUT_DIR = Path("runs/lesson40")

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
# CHARACTER-LEVEL TOKENIZER
# =====================================================================

def encode_text(text, max_length=TEXT_LENGTH):

    """
    Convert text into character IDs.

    Character encoding:

        Padding:       0
        ASCII 0-126:   1-127
        Unknown:       128

    No external tokenizer is used.
    """

    text = str(text).lower().strip()

    tokens = []

    for character in text[:max_length]:

        code = ord(character)

        if 0 <= code <= 126:

            tokens.append(code + 1)

        else:

            tokens.append(128)

    length = len(tokens)

    tokens += [0] * (max_length - length)

    return torch.tensor(
        tokens,
        dtype=torch.long
    ), max(length, 1)


# =====================================================================
# IMAGE PREPROCESSING
# =====================================================================

def preprocess_image(image):

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        array.transpose(2, 0, 1)
    ).float()

    return tensor


def crop_object(image, bbox, padding=1.3):

    width, height = image.size

    x, y, w, h = map(float, bbox)

    cx = x + w / 2.0

    cy = y + h / 2.0

    side = max(w, h) * padding

    side = max(side, 8.0)

    x1 = int(round(cx - side / 2.0))

    y1 = int(round(cy - side / 2.0))

    x2 = int(round(cx + side / 2.0))

    y2 = int(round(cy + side / 2.0))

    # Pad regions outside the original image with black pixels.

    crop = Image.new(
        "RGB",
        (x2 - x1, y2 - y1),
        (0, 0, 0)
    )

    intersection = (
        max(0, x1),
        max(0, y1),
        min(width, x2),
        min(height, y2)
    )

    if (
        intersection[2] <= intersection[0]
        or intersection[3] <= intersection[1]
    ):

        return None

    source = image.crop(intersection)

    crop.paste(
        source,
        (
            intersection[0] - x1,
            intersection[1] - y1
        )
    )

    return crop


# =====================================================================
# TACO DATASET
# =====================================================================

class TACOVisionLanguageDataset(Dataset):

    def __init__(
        self,
        root,
        image_ids=None,
        max_images=0,
        max_instances=0,
        augment=False
    ):

        self.root = Path(root)

        annotation_file = (
            self.root / "annotations.json"
        )

        if not annotation_file.exists():

            raise FileNotFoundError(
                f"TACO annotations not found: {annotation_file}"
            )

        with open(annotation_file, "r") as f:

            data = json.load(f)

        self.images = {
            item["id"]: item
            for item in data["images"]
        }

        self.categories = {
            item["id"]: item["name"]
            for item in data["categories"]
        }

        # Keep only image IDs belonging to this split.

        if image_ids is None:

            selected_ids = set(self.images.keys())

        else:

            selected_ids = set(image_ids)

        self.samples = []

        annotations_by_image = {}

        for ann in data["annotations"]:

            if ann.get("iscrowd", 0):

                continue

            image_id = ann["image_id"]

            if image_id not in selected_ids:

                continue

            if image_id not in self.images:

                continue

            category_id = ann["category_id"]

            if category_id not in self.categories:

                continue

            bbox = ann.get("bbox")

            if bbox is None or len(bbox) != 4:

                continue

            x, y, w, h = map(float, bbox)

            if w < 3 or h < 3:

                continue

            if w * h < 20:

                continue

            annotations_by_image.setdefault(
                image_id,
                []
            ).append(ann)

        available_ids = list(
            annotations_by_image.keys()
        )

        if max_images > 0:

            available_ids = available_ids[:max_images]

        for image_id in available_ids:

            for ann in annotations_by_image[image_id]:

                self.samples.append(
                    {
                        "image": self.images[image_id],
                        "annotation": ann,
                        "category_name": self.categories[
                            ann["category_id"]
                        ],
                        "category_id": ann["category_id"],
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
                "No valid TACO bounding-box annotations found."
            )

        # Convert original category IDs into compact integer IDs.

        self.category_names = sorted(
            set(
                sample["category_name"]
                for sample in self.samples
            )
        )

        self.category_to_index = {
            name: index
            for index, name in enumerate(
                self.category_names
            )
        }

        self.augment = augment

        self.category_counts = Counter(
            sample["category_name"]
            for sample in self.samples
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image_path = (
            self.root / sample["image"]["file_name"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        bbox = sample["annotation"]["bbox"]

        crop = crop_object(
            image,
            bbox
        )

        if crop is None:

            raise RuntimeError(
                f"Invalid object crop: {image_path}"
            )

        # Random horizontal flip preserves the category label.

        if self.augment and random.random() < 0.5:

            crop = crop.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

        image_tensor = preprocess_image(crop)

        category_name = sample["category_name"]

        text_tokens, text_length = encode_text(
            category_name
        )

        category_index = self.category_to_index[
            category_name
        ]

        return {
            "image": image_tensor,
            "text": text_tokens,
            "text_length": torch.tensor(text_length),
            "category": torch.tensor(category_index),
        }


# =====================================================================
# CNN IMAGE ENCODER
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

        self.features = nn.Sequential(

            ConvBlock(3, 32),

            ConvBlock(32, 64),

            ConvBlock(64, 128),

            ConvBlock(128, 192),

            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            nn.Linear(192, embed_dim)
        )

    def forward(self, image):

        features = self.features(image)

        return F.normalize(
            features,
            dim=-1
        )


# =====================================================================
# CHARACTER-LEVEL TEXT ENCODER
# =====================================================================

class TextEncoder(nn.Module):

    def __init__(
        self,
        vocab_size=TEXT_VOCAB_SIZE,
        embed_dim=EMBED_DIM
    ):

        super().__init__()

        hidden_dim = 96

        self.embedding = nn.Embedding(

            num_embeddings=vocab_size,

            embedding_dim=48,

            padding_idx=0
        )

        self.gru = nn.GRU(

            input_size=48,

            hidden_size=hidden_dim,

            batch_first=True,

            bidirectional=True
        )

        self.projection = nn.Sequential(

            nn.Linear(hidden_dim * 2, embed_dim),

            nn.GELU(),

            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, tokens, lengths):

        embedded = self.embedding(tokens)

        output, _ = self.gru(embedded)

        # Select the output at the last valid character.

        positions = (
            lengths - 1
        ).clamp(
            min=0,
            max=output.shape[1] - 1
        )

        batch_indices = torch.arange(
            output.shape[0],
            device=output.device
        )

        features = output[
            batch_indices,
            positions
        ]

        projected = self.projection(features)

        return F.normalize(
            projected,
            dim=-1
        )


# =====================================================================
# COMPLETE VISION-LANGUAGE MODEL
# =====================================================================

class VisionLanguageModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.image_encoder = ImageEncoder()

        self.text_encoder = TextEncoder()

        # Learnable log temperature, following the CLIP idea.

        self.logit_scale = nn.Parameter(
            torch.tensor(
                math.log(1.0 / TEMPERATURE)
            )
        )

    def encode_image(self, image):

        return self.image_encoder(image)

    def encode_text(self, tokens, lengths):

        return self.text_encoder(
            tokens,
            lengths
        )

    def forward(self, image, tokens, lengths):

        image_embedding = self.encode_image(
            image
        )

        text_embedding = self.encode_text(
            tokens,
            lengths
        )

        scale = self.logit_scale.exp().clamp(
            max=100
        )

        similarity = (
            image_embedding
            @ text_embedding.T
        ) * scale

        return {
            "image_embedding": image_embedding,
            "text_embedding": text_embedding,
            "similarity": similarity,
        }


# =====================================================================
# MULTI-POSITIVE CONTRASTIVE LOSS
# =====================================================================

def multi_positive_loss(logits, labels):

    """
    Several images may share the same category.

    Therefore, all same-category pairs are positives.

    This avoids treating another image of the same waste
    category as a negative example.
    """

    positive_mask = (
        labels[:, None] == labels[None, :]
    )

    log_probabilities = F.log_softmax(
        logits,
        dim=1
    )

    positive_log_probabilities = (
        log_probabilities.masked_fill(
            ~positive_mask,
            0.0
        )
    )

    positive_counts = positive_mask.sum(
        dim=1
    ).clamp(min=1)

    loss = -(
        positive_log_probabilities.sum(dim=1)
        / positive_counts
    )

    return loss.mean()


def contrastive_loss(
    image_embedding,
    text_embedding,
    labels,
    temperature=TEMPERATURE
):

    similarity = (
        image_embedding
        @ text_embedding.T
    )

    similarity = similarity / temperature

    image_to_text = multi_positive_loss(
        similarity,
        labels
    )

    text_to_image = multi_positive_loss(
        similarity.T,
        labels
    )

    return (
        image_to_text + text_to_image
    ) / 2.0


# =====================================================================
# TRAINING
# =====================================================================

def train_one_epoch(
    model,
    loader,
    optimizer
):

    model.train()

    total_loss = 0.0

    count = 0

    for batch in loader:

        image = batch["image"].to(DEVICE)

        tokens = batch["text"].to(DEVICE)

        lengths = batch["text_length"].to(DEVICE)

        labels = batch["category"].to(DEVICE)

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            image,
            tokens,
            lengths
        )

        loss = contrastive_loss(

            output["image_embedding"],

            output["text_embedding"],

            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        count += 1

    return total_loss / max(count, 1)


# =====================================================================
# VALIDATION
# =====================================================================

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    count = 0

    for batch in loader:

        image = batch["image"].to(DEVICE)

        tokens = batch["text"].to(DEVICE)

        lengths = batch["text_length"].to(DEVICE)

        labels = batch["category"].to(DEVICE)

        output = model(
            image,
            tokens,
            lengths
        )

        loss = contrastive_loss(

            output["image_embedding"],

            output["text_embedding"],

            labels
        )

        predictions = (
            output["similarity"].argmax(dim=1)
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.numel()

        total_loss += loss.item()

        count += 1

    return {
        "loss": total_loss / max(count, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


# =====================================================================
# TEXT QUERY ENCODING
# =====================================================================

@torch.no_grad()
def encode_queries(model, queries, device):

    tokens = []

    lengths = []

    for query in queries:

        encoded, length = encode_text(
            query
        )

        tokens.append(encoded)

        lengths.append(length)

    tokens = torch.stack(tokens).to(device)

    lengths = torch.tensor(
        lengths,
        dtype=torch.long,
        device=device
    )

    return model.encode_text(
        tokens,
        lengths
    )


# =====================================================================
# REGION PROPOSALS
# =====================================================================

def generate_sliding_windows(
    width,
    height,
    scales=(0.25, 0.40, 0.60),
    overlap=0.5,
    max_proposals=400
):

    """
    Generate square windows at multiple scales.

    This is a simple educational proposal generator.

    It is not a learned region proposal network.
    """

    proposals = []

    for scale in scales:

        window = int(
            min(width, height) * scale
        )

        window = max(
            32,
            min(window, width, height)
        )

        stride = max(
            16,
            int(window * (1.0 - overlap))
        )

        xs = list(
            range(0, max(width - window + 1, 1), stride)
        )

        ys = list(
            range(0, max(height - window + 1, 1), stride)
        )

        if not xs or xs[-1] != width - window:

            xs.append(width - window)

        if not ys or ys[-1] != height - window:

            ys.append(height - window)

        for y in ys:

            for x in xs:

                proposals.append(
                    [
                        x,
                        y,
                        x + window,
                        y + window
                    ]
                )

                if len(proposals) >= max_proposals:

                    return proposals

    return proposals


# =====================================================================
# BOX IOU
# =====================================================================

def numpy_box_iou(box_a, box_b):

    x1 = max(
        box_a[0],
        box_b[0]
    )

    y1 = max(
        box_a[1],
        box_b[1]
    )

    x2 = min(
        box_a[2],
        box_b[2]
    )

    y2 = min(
        box_a[3],
        box_b[3]
    )

    intersection = max(
        0,
        x2 - x1
    ) * max(
        0,
        y2 - y1
    )

    area_a = max(
        0,
        box_a[2] - box_a[0]
    ) * max(
        0,
        box_a[3] - box_a[1]
    )

    area_b = max(
        0,
        box_b[2] - box_b[0]
    ) * max(
        0,
        box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    return intersection / max(union, 1e-8)


# =====================================================================
# NON-MAXIMUM SUPPRESSION
# =====================================================================

def nms_numpy(
    boxes,
    scores,
    iou_threshold=0.4
):

    if len(boxes) == 0:

        return []

    order = np.argsort(scores)[::-1]

    keep = []

    while len(order) > 0:

        current = order[0]

        keep.append(current)

        remaining = []

        for index in order[1:]:

            iou = numpy_box_iou(
                boxes[current],
                boxes[index]
            )

            if iou < iou_threshold:

                remaining.append(index)

        order = np.asarray(
            remaining,
            dtype=np.int64
        )

    return keep


# =====================================================================
# TEXT-PROMPTED DETECTION
# =====================================================================

@torch.no_grad()
def detect_text_queries(
    model,
    image,
    queries,
    device,
    score_threshold=0.15,
    nms_threshold=0.4,
    max_proposals=400,
    proposal_batch_size=32
):

    """
    Detect candidate regions matching one or more text queries.

    Input:
        image: PIL RGB image
        queries: list of strings

    Output:
        List of dictionaries:

            {
                "box": [x1, y1, x2, y2],
                "query": "plastic bottle",
                "score": 0.42
            }

    Similarity scores are cosine similarities.

    They are NOT calibrated probabilities.
    """

    model.eval()

    width, height = image.size

    query_embeddings = encode_queries(
        model,
        queries,
        device
    )

    proposals = generate_sliding_windows(
        width,
        height,
        max_proposals=max_proposals
    )

    crops = []

    for x1, y1, x2, y2 in proposals:

        crop = image.crop(
            (x1, y1, x2, y2)
        )

        crops.append(
            preprocess_image(crop)
        )

    if not crops:

        return []

    embeddings = []

    for start in range(
        0,
        len(crops),
        proposal_batch_size
    ):

        batch = torch.stack(
            crops[
                start:start + proposal_batch_size
            ]
        ).to(device)

        features = model.encode_image(
            batch
        )

        embeddings.append(features)

    image_embeddings = torch.cat(
        embeddings,
        dim=0
    )

    similarities = (
        image_embeddings
        @ query_embeddings.T
    )

    similarities = similarities.cpu().numpy()

    all_detections = []

    for query_index, query in enumerate(queries):

        scores = similarities[:, query_index]

        selected = np.where(
            scores >= score_threshold
        )[0]

        if len(selected) == 0:

            continue

        selected_boxes = [
            proposals[index]
            for index in selected
        ]

        selected_scores = [
            float(scores[index])
            for index in selected
        ]

        keep = nms_numpy(
            selected_boxes,
            selected_scores,
            iou_threshold=nms_threshold
        )

        for index in keep:

            all_detections.append(
                {
                    "box": selected_boxes[index],
                    "query": query,
                    "score": selected_scores[index],
                }
            )

    all_detections.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return all_detections


# =====================================================================
# VISUALIZATION
# =====================================================================

def visualize_detections(
    image,
    detections,
    output_path
):

    result = image.copy()

    draw = ImageDraw.Draw(result)

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        query = detection["query"]

        score = detection["score"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(0, 255, 0),
            width=3
        )

        label = (
            f"{query}: {score:.3f}"
        )

        text_box = draw.textbbox(
            (x1, max(0, y1 - 18)),
            label
        )

        draw.rectangle(
            text_box,
            fill=(0, 100, 0)
        )

        draw.text(
            (x1, max(0, y1 - 18)),
            label,
            fill=(255, 255, 255)
        )

    result.save(
        output_path
    )


def save_training_curves(
    history,
    output_path
):

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

    axes[0].set_title(
        "Contrastive Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].legend()

    axes[1].plot(
        epochs,
        history["val_accuracy"],
        label="Validation Retrieval Accuracy"
    )

    axes[1].set_title(
        "Image-Text Matching Accuracy"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

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

def save_dataset_summary(
    dataset,
    output_path
):

    summary = {
        "num_instances": len(dataset),
        "num_categories": len(dataset.category_names),
        "categories": dataset.category_counts,
        "category_names": dataset.category_names,
        "image_size": IMAGE_SIZE,
        "data_source": "Real TACO images and annotations",
    }

    summary["categories"] = dict(
        summary["categories"]
    )

    with open(output_path, "w") as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print("\nDataset summary:")

    print(
        json.dumps(
            summary,
            indent=4
        )
    )


# =====================================================================
# NUMERICAL TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print("LESSON 40 — NUMERICAL TESTS")

    print("=" * 70)

    # Test 1: character encoding.

    tokens, length = encode_text(
        "Plastic Bottle"
    )

    assert tokens.shape == (
        TEXT_LENGTH,
    )

    assert length == len(
        "plastic bottle"
    )

    assert tokens[0].item() > 0

    print("Test 1 passed: character-level encoding")

    # Test 2: padding.

    assert torch.all(
        tokens[length:] == 0
    )

    print("Test 2 passed: text padding")

    # Test 3: empty text.

    empty_tokens, empty_length = encode_text("")

    assert empty_length == 1

    assert empty_tokens.sum().item() == 0

    print("Test 3 passed: empty text handling")

    # Test 4: image preprocessing.

    test_image = Image.new(
        "RGB",
        (100, 80),
        color=(120, 150, 200)
    )

    image_tensor = preprocess_image(
        test_image
    )

    assert image_tensor.shape == (
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    assert torch.isfinite(
        image_tensor
    ).all()

    print("Test 4 passed: image preprocessing")

    # Test 5: object crop.

    crop = crop_object(
        test_image,
        [20, 20, 30, 30]
    )

    assert crop is not None

    assert crop.size[0] > 0

    assert crop.size[1] > 0

    print("Test 5 passed: object crop")

    # Test 6: image encoder.

    image_encoder = ImageEncoder()

    images = torch.rand(
        2,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    image_features = image_encoder(
        images
    )

    assert image_features.shape == (
        2,
        EMBED_DIM
    )

    norms = image_features.norm(
        dim=-1
    )

    assert torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=1e-5
    )

    print("Test 6 passed: normalized image embeddings")

    # Test 7: text encoder.

    text_encoder = TextEncoder()

    text_batch = torch.stack(
        [
            encode_text("plastic bottle")[0],
            encode_text("metal can")[0]
        ]
    )

    lengths = torch.tensor(
        [
            len("plastic bottle"),
            len("metal can")
        ]
    )

    text_features = text_encoder(
        text_batch,
        lengths
    )

    assert text_features.shape == (
        2,
        EMBED_DIM
    )

    print("Test 7 passed: text encoder")

    # Test 8: normalized text embeddings.

    assert torch.allclose(
        text_features.norm(dim=-1),
        torch.ones(2),
        atol=1e-5
    )

    print("Test 8 passed: normalized text embeddings")

    # Test 9: contrastive loss.

    labels = torch.tensor(
        [0, 1, 0, 1]
    )

    features_a = F.normalize(
        torch.randn(4, EMBED_DIM),
        dim=-1
    )

    features_b = F.normalize(
        torch.randn(4, EMBED_DIM),
        dim=-1
    )

    loss = contrastive_loss(
        features_a,
        features_b,
        labels
    )

    assert torch.isfinite(loss)

    assert loss.item() >= 0

    print("Test 9 passed: multi-positive contrastive loss")

    # Test 10: sliding-window proposals.

    proposals = generate_sliding_windows(
        640,
        480,
        max_proposals=100
    )

    assert len(proposals) > 0

    assert len(proposals) <= 100

    for box in proposals:

        x1, y1, x2, y2 = box

        assert 0 <= x1 < x2 <= 640

        assert 0 <= y1 < y2 <= 480

    print("Test 10 passed: sliding-window proposals")

    # Test 11: IoU.

    box_a = [0, 0, 100, 100]

    box_b = [0, 0, 100, 100]

    assert abs(
        numpy_box_iou(box_a, box_b) - 1.0
    ) < 1e-6

    print("Test 11 passed: bounding-box IoU")

    # Test 12: NMS.

    boxes = [
        [0, 0, 100, 100],
        [5, 5, 105, 105],
        [200, 200, 300, 300]
    ]

    scores = [
        0.9,
        0.8,
        0.7
    ]

    keep = nms_numpy(
        boxes,
        scores,
        iou_threshold=0.5
    )

    assert len(keep) == 2

    assert keep[0] == 0

    print("Test 12 passed: Non-Maximum Suppression")

    # Test 13: full model.

    model = VisionLanguageModel()

    test_tokens = torch.stack(
        [
            encode_text("plastic bottle")[0],
            encode_text("metal can")[0]
        ]
    )

    test_lengths = torch.tensor(
        [
            len("plastic bottle"),
            len("metal can")
        ]
    )

    output = model(
        images,
        test_tokens,
        test_lengths
    )

    assert output["similarity"].shape == (
        2,
        2
    )

    print("Test 13 passed: complete vision-language model")

    # Test 14: gradient flow.

    model.zero_grad()

    labels = torch.tensor(
        [0, 1]
    )

    loss = contrastive_loss(
        output["image_embedding"],
        output["text_embedding"],
        labels
    )

    loss.backward()

    assert (
        model.image_encoder.features[-1].weight.grad
        is not None
    )

    assert (
        model.text_encoder.projection[-1].weight.grad
        is not None
    )

    print("Test 14 passed: gradient propagation")

    print("\nAll 14 tests passed.\n")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Lesson 40: Open-Vocabulary Detection"
    )

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
        "--workers",
        type=int,
        default=0
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
        "--query",
        type=str,
        default="plastic bottle;aluminum can;cardboard;glass bottle"
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.15
    )

    parser.add_argument(
        "--max-proposals",
        type=int,
        default=400
    )

    parser.add_argument(
        "--check-only",
        action="store_true"
    )

    parser.add_argument(
        "--inference-only",
        action="store_true"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto"
    )

    args = parser.parse_args()

    set_seed()

    global DEVICE

    if args.device == "cpu":

        DEVICE = torch.device("cpu")

    elif args.device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        DEVICE = torch.device("cuda")

    else:

        DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)

    print("LESSON 40 — OPEN-VOCABULARY OBJECT DETECTION")

    print("=" * 70)

    print("Device:", DEVICE)

    print("Image size:", IMAGE_SIZE)

    print("Embedding dimension:", EMBED_DIM)

    run_tests()

    if args.check_only:

        print("Check-only mode completed.")

        return

    # ---------------------------------------------------------------
    # LOAD CHECKPOINT FOR INFERENCE
    # ---------------------------------------------------------------

    model = VisionLanguageModel().to(DEVICE)

    checkpoint_path = (
        OUTPUT_DIR / "best_model.pt"
    )

    if args.inference_only:

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                "No trained checkpoint found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        print(
            "\nLoaded checkpoint:",
            checkpoint_path
        )

        dataset = TACOVisionLanguageDataset(
            args.taco_root,
            max_images=1
        )

        sample = dataset.samples[0]

        image_path = (
            Path(args.taco_root)
            / sample["image"]["file_name"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        queries = [
            query.strip()
            for query in args.query.split(";")
            if query.strip()
        ]

        detections = detect_text_queries(

            model,

            image,

            queries,

            DEVICE,

            score_threshold=args.score_threshold,

            max_proposals=args.max_proposals
        )

        visualize_detections(

            image,

            detections,

            OUTPUT_DIR / "detection_preview.png"
        )

        print("\nText queries:", queries)

        print("Detections:", len(detections))

        for detection in detections:

            print(
                f"  {detection['query']}: "
                f"score={detection['score']:.4f}, "
                f"box={detection['box']}"
            )

        print(
            "\nSaved:",
            OUTPUT_DIR / "detection_preview.png"
        )

        return

    # ---------------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------------

    # Split by image ID, not by annotation instance.
    # This prevents crops from the same photograph leaking
    # into both training and validation.

    with open(
        Path(args.taco_root) / "annotations.json",
        "r"
    ) as f:

        annotation_data = json.load(f)

    all_image_ids = [
        item["id"]
        for item in annotation_data["images"]
    ]

    random.shuffle(
        all_image_ids
    )

    if args.max_images > 0:

        all_image_ids = all_image_ids[
            :args.max_images
        ]

    if len(all_image_ids) < 2:

        raise RuntimeError(
            "At least two images are required."
        )

    split = max(
        1,
        int(len(all_image_ids) * 0.8)
    )

    split = min(
        split,
        len(all_image_ids) - 1
    )

    train_ids = set(
        all_image_ids[:split]
    )

    val_ids = set(
        all_image_ids[split:]
    )

    train_dataset = TACOVisionLanguageDataset(

        root=args.taco_root,

        image_ids=train_ids,

        max_instances=args.max_instances,

        augment=True
    )

    val_dataset = TACOVisionLanguageDataset(

        root=args.taco_root,

        image_ids=val_ids,

        max_instances=args.max_instances,

        augment=False
    )

    # Both datasets must use the same category-index mapping.

    val_dataset.category_to_index = (
        train_dataset.category_to_index
    )

    val_dataset.category_names = (
        train_dataset.category_names
    )

    # Remove validation categories not represented in training.
    # Such categories cannot be evaluated by this trained model.

    train_categories = set(
        train_dataset.category_names
    )

    val_dataset.samples = [
        sample
        for sample in val_dataset.samples
        if sample["category_name"] in train_categories
    ]

    if not val_dataset.samples:

        raise RuntimeError(
            "No validation instances share categories with training. "
            "Increase --max-images."
        )

    print("\nTraining instances:", len(train_dataset))

    print("Validation instances:", len(val_dataset))

    print("Training categories:", len(train_dataset.category_names))

    save_dataset_summary(
        train_dataset,
        OUTPUT_DIR / "dataset_summary.json"
    )

    # ---------------------------------------------------------------
    # DATA LOADERS
    # ---------------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch,

        shuffle=True,

        num_workers=args.workers,

        pin_memory=(DEVICE.type == "cuda"),

        drop_last=False
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=args.batch,

        shuffle=False,

        num_workers=args.workers,

        pin_memory=(DEVICE.type == "cuda")
    )

    # ---------------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------------

    model = VisionLanguageModel().to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "\nTrainable parameters:",
        f"{parameter_count:,}"
    )

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
        "val_accuracy": [],
    }

    best_accuracy = -1.0

    for epoch in range(args.epochs):

        start = time.time()

        train_loss = train_one_epoch(

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
            train_loss
        )

        history["val_loss"].append(
            val_metrics["loss"]
        )

        history["val_accuracy"].append(
            val_metrics["accuracy"]
        )

        elapsed = time.time() - start

        print(

            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "

            f"Train Loss: {train_loss:.4f} | "

            f"Val Loss: {val_metrics['loss']:.4f} | "

            f"Retrieval Accuracy: "
            f"{val_metrics['accuracy']:.4f} | "

            f"Time: {elapsed:.1f}s"
        )

        if val_metrics["accuracy"] > best_accuracy:

            best_accuracy = val_metrics["accuracy"]

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_accuracy": best_accuracy,
                    "category_names": train_dataset.category_names,
                    "image_size": IMAGE_SIZE,
                },
                checkpoint_path
            )

    # ---------------------------------------------------------------
    # LOAD BEST CHECKPOINT
    # ---------------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    print(
        "\nBest checkpoint epoch:",
        checkpoint["epoch"]
    )

    print(
        "Best retrieval accuracy:",
        checkpoint["best_accuracy"]
    )

    # ---------------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------------

    save_training_curves(
        history,
        OUTPUT_DIR / "training_curves.png"
    )

    # Use a real validation photograph for the detection demo.

    first_sample = val_dataset.samples[0]

    image_path = (
        Path(args.taco_root)
        / first_sample["image"]["file_name"]
    )

    original_image = Image.open(
        image_path
    ).convert("RGB")

    queries = [
        query.strip()
        for query in args.query.split(";")
        if query.strip()
    ]

    detections = detect_text_queries(

        model,

        original_image,

        queries,

        DEVICE,

        score_threshold=args.score_threshold,

        max_proposals=args.max_proposals
    )

    visualize_detections(

        original_image,

        detections,

        OUTPUT_DIR / "detection_preview.png"
    )

    print("\nDetection queries:", queries)

    print("Detection count:", len(detections))

    for detection in detections:

        print(
            f"  {detection['query']}: "
            f"similarity={detection['score']:.4f}, "
            f"box={detection['box']}"
        )

    print("\nOutputs saved to:", OUTPUT_DIR)

    print("  best_model.pt")

    print("  training_curves.png")

    print("  detection_preview.png")

    print("  dataset_summary.json")

    print("\nLesson 40 completed.")


if __name__ == "__main__":

    main()