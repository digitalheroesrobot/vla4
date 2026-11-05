"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 49 — MULTI-OBJECT TRACKING WITH APPEARANCE RE-IDENTIFICATION
======================================================================

Filename:
    lesson49_multi_object_tracking_reid.py

PHASE:
    Phase 4 — Tracking and Visual Memory

======================================================================
LEARNING OBJECTIVES
======================================================================

1. Understand person/object re-identification (Re-ID).
2. Understand why color histograms are insufficient.
3. Build a CNN appearance encoder from scratch.
4. Train an embedding model using supervised contrastive learning.
5. Normalize appearance embeddings.
6. Calculate cosine distance between object identities.
7. Combine appearance, motion and IoU for association.
8. Recover lost tracks using appearance similarity.
9. Evaluate embedding quality using retrieval accuracy.
10. Integrate learned appearance features into a tracker.

======================================================================
1. WHAT IS RE-IDENTIFICATION?
======================================================================

Object detection answers:

    What objects are visible in this frame?

Object tracking answers:

    Which detection belongs to which existing track?

Re-identification answers:

    Have I seen this object before?

Example:

    Frame 1:
        Bottle A -> ID 1
        Bottle B -> ID 2

    Frame 2:
        Bottle A becomes occluded.
        Bottle B remains visible.

    Frame 20:
        Bottle A appears again.

The tracker should recover:

        Bottle A -> ID 1

Even if its position changed significantly.

======================================================================
2. APPEARANCE EMBEDDINGS
======================================================================

A CNN converts an image crop into a feature vector:

    image crop
        |
        v
    CNN
        |
        v
    embedding vector

For example:

    embedding = [0.12, -0.34, 0.81, ...]

The vector represents visual appearance.

We normalize embeddings:

    e_normalized = e / ||e||

Cosine similarity:

    similarity = e1 dot e2

Cosine distance:

    distance = 1 - similarity

For normalized embeddings:

    similarity is between -1 and 1.

    distance is between 0 and 2.

Smaller cosine distance means more similar appearance.

======================================================================
3. CNN APPEARANCE ENCODER
======================================================================

We implement:

    Conv -> BatchNorm -> ReLU
    Conv -> BatchNorm -> ReLU
    Conv -> BatchNorm -> ReLU
    Global Average Pooling
    Fully Connected Projection
    L2 Normalization

The network receives RGB object crops and produces normalized
appearance embeddings.

No pretrained model is used.

======================================================================
4. TRAINING THE EMBEDDING MODEL
======================================================================

Training labels represent object identities.

For this educational implementation, we generate training
examples from real video frames:

    Frame t:
        Crop A -> identity 0

    Frame t+1:
        Crop A -> identity 0

    Frame t+2:
        Crop A -> identity 0

This assumes the demonstration detector can associate an object
across nearby frames using geometry.

The resulting training data is weakly supervised.

It is not equivalent to a manually identity-labeled Re-ID dataset.

We use supervised contrastive learning.

For anchor embedding i:

    P(i) = samples with the same identity.

    N(i) = samples with different identities.

The supervised contrastive objective encourages:

    same identity -> similar embeddings
    different identities -> separated embeddings

======================================================================
5. SUPERVISED CONTRASTIVE LOSS
======================================================================

For normalized embeddings:

    similarity(i,j) = zi dot zj

Temperature-scaled similarity:

    logits(i,j) = similarity(i,j) / temperature

The loss encourages positive pairs to have larger similarity
than negative pairs.

A batch must contain at least two samples per identity.

======================================================================
6. TRACK ASSOCIATION
======================================================================

We combine:

    Appearance distance:
        1 - cosine_similarity

    Motion distance:
        normalized center displacement

    IoU distance:
        1 - IoU

Combined cost:

    C = wa * appearance_distance
      + wm * motion_distance
      + wi * IoU_distance

The Hungarian algorithm finds a globally consistent assignment.

======================================================================
7. LOST-TRACK RECOVERY
======================================================================

A lost track retains its last appearance embedding.

When a detection returns:

    1. Compare its embedding with lost-track embeddings.
    2. Apply a motion gate.
    3. Apply an appearance threshold.
    4. Associate using Hungarian assignment.
    5. Restore the original ID.

======================================================================
8. IMPORTANT LIMITATIONS
======================================================================

This lesson uses MOG2 foreground detection for demonstration.

MOG2:
    - Works best with a stationary camera.
    - Does not recognize semantic waste categories.
    - Can merge nearby objects.
    - Can produce false detections during camera movement.

The training identities are generated by short-term geometric
association, not manually annotated ground-truth identities.

Therefore, the learned embedding is educational and may not
generalize to unseen objects.

For production Re-ID, use manually identity-labeled sequences,
hard-negative mining, stronger augmentation and larger datasets.

======================================================================
9. ROBOTICS APPLICATION
======================================================================

OpenArm V2 waste sorting:

    Camera frames
        |
        v
    YOLO / Instance Segmentation
        |
        v
    Object crops
        |
        v
    CNN appearance encoder
        |
        v
    Appearance embeddings
        |
        v
    Multi-object tracker
        |
        v
    Stable object identities
        |
        v
    Camera-to-robot localization
        |
        v
    Grasp planning

A stable object identity can help the robot avoid duplicate
handling and preserve the history of a partially occluded item.

======================================================================
INSTALL
======================================================================

    pip install numpy opencv-python scipy torch matplotlib

======================================================================
RUN
======================================================================

    python lesson49_multi_object_tracking_reid.py --check-only

    python lesson49_multi_object_tracking_reid.py \
        --video conveyor_video.mp4 \
        --train

    python lesson49_multi_object_tracking_reid.py \
        --video conveyor_video.mp4 \
        --checkpoint runs/lesson49/reid_encoder.pt

======================================================================
OUTPUT
======================================================================

    runs/lesson49/
        reid_encoder.pt
        training_curves.png
        retrieval_metrics.json
        multi_object_reid_tracking.mp4
        tracking_summary.json
======================================================================
"""

import argparse
import json
import random
import time

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt


# =====================================================================
# CONFIGURATION
# =====================================================================

IMAGE_SIZE = 64

EMBEDDING_DIM = 128

DEFAULT_BATCH_SIZE = 32

DEFAULT_EPOCHS = 10

DEFAULT_LEARNING_RATE = 1e-3

DEFAULT_TEMPERATURE = 0.07

HISTOGRAM_UNUSED = False


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =====================================================================
# GEOMETRY
# =====================================================================

def bbox_center(bbox):

    x1, y1, x2, y2 = np.asarray(
        bbox,
        dtype=np.float64
    )

    return np.array(
        [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0
        ],
        dtype=np.float64
    )


def bbox_area(bbox):

    x1, y1, x2, y2 = np.asarray(
        bbox,
        dtype=np.float64
    )

    return max(
        0.0,
        x2 - x1
    ) * max(
        0.0,
        y2 - y1
    )


def iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)

    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)

    y2 = min(ay2, by2)

    intersection = (

        max(0.0, x2 - x1)

        * max(0.0, y2 - y1)
    )

    union = (

        bbox_area(box_a)

        + bbox_area(box_b)

        - intersection
    )

    return intersection / max(
        union,
        1e-12
    )


def center_distance(box_a, box_b):

    return float(
        np.linalg.norm(
            bbox_center(box_a) - bbox_center(box_b)
        )
    )


# =====================================================================
# IMAGE CROP PREPROCESSING
# =====================================================================

def crop_object(
    frame,
    bbox,
    image_size=IMAGE_SIZE
):

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = np.asarray(
        bbox,
        dtype=np.float64
    )

    x1 = int(
        np.clip(x1, 0, width - 1)
    )

    y1 = int(
        np.clip(y1, 0, height - 1)
    )

    x2 = int(
        np.clip(x2, 0, width)
    )

    y2 = int(
        np.clip(y2, 0, height)
    )

    if x2 <= x1 or y2 <= y1:

        return np.zeros(

            (
                image_size,
                image_size,
                3
            ),

            dtype=np.uint8
        )

    crop = frame[
        y1:y2,
        x1:x2
    ]

    crop = cv2.resize(

        crop,

        (image_size, image_size),

        interpolation=cv2.INTER_AREA
    )

    return cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )


def image_to_tensor(image):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image /= 255.0

    image = np.transpose(

        image,

        (2, 0, 1)
    )

    return torch.from_numpy(
        image.copy()
    )


# =====================================================================
# DATASET
# =====================================================================

class ReIDCropDataset(Dataset):

    """
    Real image crops with automatically generated identity labels.

    Identity labels are assigned by short-term geometric association.

    These are weak labels, not ground-truth identities.

    We deliberately keep this limitation visible because a
    tracker-generated label can contain identity mistakes.
    """

    def __init__(
        self,
        images,
        labels,
        augment=True
    ):

        self.images = images

        self.labels = np.asarray(
            labels,
            dtype=np.int64
        )

        self.augment = augment

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index].copy()

        label = int(
            self.labels[index]
        )

        if self.augment:

            # Horizontal flip.

            if random.random() < 0.5:

                image = np.fliplr(
                    image
                ).copy()

            # Brightness augmentation.

            if random.random() < 0.5:

                factor = random.uniform(
                    0.8,
                    1.2
                )

                image = np.clip(

                    image.astype(np.float32) * factor,

                    0,

                    255
                ).astype(
                    np.uint8
                )

            # Small contrast augmentation.

            if random.random() < 0.3:

                mean = image.mean(
                    axis=(0, 1),
                    keepdims=True
                )

                image = np.clip(

                    (image.astype(np.float32) - mean) * 1.1 + mean,

                    0,

                    255
                ).astype(
                    np.uint8
                )

        return (
            image_to_tensor(image),
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# =====================================================================
# CNN APPEARANCE ENCODER
# =====================================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(

                in_channels,

                out_channels,

                kernel_size=3,

                stride=stride,

                padding=1,

                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


class AppearanceEncoder(nn.Module):

    """
    CNN Re-ID encoder trained from scratch.

    Input:
        B x 3 x 64 x 64

    Output:
        B x embedding_dim

    Embeddings are L2 normalized.
    """

    def __init__(
        self,
        embedding_dim=EMBEDDING_DIM
    ):

        super().__init__()

        self.features = nn.Sequential(

            ConvBlock(
                3,
                32,
                stride=2
            ),

            ConvBlock(
                32,
                64,
                stride=2
            ),

            ConvBlock(
                64,
                128,
                stride=2
            ),

            ConvBlock(
                128,
                256,
                stride=2
            ),

            ConvBlock(
                256,
                256,
                stride=1
            )
        )

        self.pool = nn.AdaptiveAvgPool2d(
            1
        )

        self.projection = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                256,
                embedding_dim
            )
        )

    def forward(self, x):

        features = self.features(
            x
        )

        pooled = self.pool(
            features
        )

        embedding = self.projection(
            pooled
        )

        embedding = F.normalize(

            embedding,

            p=2,

            dim=1
        )

        return embedding


# =====================================================================
# SUPERVISED CONTRASTIVE LOSS
# =====================================================================

class SupervisedContrastiveLoss(nn.Module):

    """
    Supervised contrastive loss.

    For each anchor, positive examples share the same identity label.

    The loss encourages same-identity embeddings to be close
    and different-identity embeddings to be separated.
    """

    def __init__(
        self,
        temperature=DEFAULT_TEMPERATURE
    ):

        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        embeddings,
        labels
    ):

        batch_size = embeddings.shape[0]

        if batch_size < 2:

            return embeddings.sum() * 0.0

        embeddings = F.normalize(

            embeddings,

            dim=1
        )

        similarity = (

            embeddings @ embeddings.T
        )

        logits = (
            similarity / self.temperature
        )

        labels = labels.reshape(-1, 1)

        positive_mask = (
            labels == labels.T
        ).float()

        # Exclude self-comparisons.

        identity_mask = torch.eye(

            batch_size,

            device=embeddings.device
        )

        positive_mask = (
            positive_mask - identity_mask
        )

        negative_infinity = torch.finfo(
            logits.dtype
        ).min

        logits = logits.masked_fill(

            identity_mask.bool(),

            negative_infinity
        )

        log_prob = (

            logits

            - torch.logsumexp(
                logits,
                dim=1,
                keepdim=True
            )
        )

        positive_count = positive_mask.sum(
            dim=1
        )

        valid = positive_count > 0

        if not valid.any():

            return embeddings.sum() * 0.0

        mean_positive_log_prob = (

            positive_mask * log_prob
        ).sum(
            dim=1
        ) / positive_count.clamp_min(1.0)

        loss = -mean_positive_log_prob[
            valid
        ].mean()

        return loss


# =====================================================================
# TRAINING BATCH SAMPLER
# =====================================================================

class IdentityBatchSampler:

    """
    Build batches containing multiple examples per identity.

    The contrastive objective needs positive pairs.

    Without at least two samples from one identity, there are
    no positive pairs for that identity.
    """

    def __init__(
        self,
        labels,
        identities_per_batch=8,
        samples_per_identity=4
    ):

        self.labels = np.asarray(
            labels
        )

        self.identities_per_batch = identities_per_batch

        self.samples_per_identity = samples_per_identity

        self.identity_indices = {}

        for index, label in enumerate(self.labels):

            self.identity_indices.setdefault(
                int(label),
                []
            ).append(index)

        self.valid_identities = [

            identity

            for identity, indices in self.identity_indices.items()

            if len(indices) >= 2
        ]

    def batches(self, batch_size):

        if not self.valid_identities:

            raise RuntimeError(

                "No identity has at least two training crops. "

                "More video frames or more stable detections are needed."
            )

        samples_per_identity = self.samples_per_identity

        identities_per_batch = max(

            1,

            batch_size // samples_per_identity
        )

        identities_per_batch = min(

            identities_per_batch,

            self.identities_per_batch,

            len(self.valid_identities)
        )

        indices = []

        random.shuffle(
            self.valid_identities
        )

        for start in range(

            0,

            len(self.valid_identities),

            identities_per_batch
        ):

            selected = self.valid_identities[

                start:start + identities_per_batch
            ]

            batch = []

            for identity in selected:

                candidates = self.identity_indices[
                    identity
                ]

                selected_samples = random.choices(

                    candidates,

                    k=samples_per_identity
                )

                batch.extend(
                    selected_samples
                )

            if len(batch) >= 2:

                indices.append(
                    batch
                )

        return indices


# =====================================================================
# WEAK-LABEL DATA COLLECTION
# =====================================================================

def build_training_crops(
    video_path,
    max_frames=1500,
    min_area=500
):

    """
    Collect real object crops from a video.

    Identity labels are generated by geometric association between
    detections in consecutive frames.

    This is weak supervision, not manual identity annotation.

    For a proper production Re-ID dataset, use manually labeled
    object identities across multiple frames and occlusions.
    """

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    background = cv2.createBackgroundSubtractorMOG2(

        history=500,

        varThreshold=32,

        detectShadows=False
    )

    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (5, 5)
    )

    images = []

    labels = []

    next_identity = 0

    previous = []

    frame_index = 0

    while frame_index < max_frames:

        ret, frame = capture.read()

        if not ret:
            break

        frame_index += 1

        mask = background.apply(
            frame
        )

        mask = cv2.morphologyEx(

            mask,

            cv2.MORPH_OPEN,

            kernel
        )

        mask = cv2.morphologyEx(

            mask,

            cv2.MORPH_CLOSE,

            kernel,

            iterations=2
        )

        contours, _ = cv2.findContours(

            mask,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE
        )

        current = []

        for contour in contours:

            if cv2.contourArea(contour) < min_area:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            bbox = np.array(

                [
                    x,
                    y,
                    x + w,
                    y + h
                ],

                dtype=np.float64
            )

            current.append(
                bbox
            )

        # Associate consecutive-frame detections.

        if previous and current:

            cost = np.full(

                (
                    len(previous),

                    len(current)
                ),

                1e6,

                dtype=np.float64
            )

            for i, old in enumerate(previous):

                for j, new in enumerate(current):

                    distance = center_distance(
                        old["bbox"],
                        new
                    )

                    overlap = iou(
                        old["bbox"],
                        new
                    )

                    if distance < 80 or overlap > 0.1:

                        cost[i, j] = (

                            distance / 80.0

                            + 1.0 - overlap
                        )

            rows, cols = linear_sum_assignment(
                cost
            )

            identity_assignments = {}

            for row, col in zip(rows, cols):

                if cost[row, col] < 2.0:

                    identity_assignments[col] = (
                        previous[row]["identity"]
                    )

        else:

            identity_assignments = {}

        new_previous = []

        for index, bbox in enumerate(current):

            identity = identity_assignments.get(
                index
            )

            if identity is None:

                identity = next_identity

                next_identity += 1

            crop = crop_object(
                frame,
                bbox
            )

            images.append(
                crop
            )

            labels.append(
                identity
            )

            new_previous.append(

                {

                    "bbox": bbox,

                    "identity": identity
                }
            )

        previous = new_previous

    capture.release()

    print("\nWeak-label dataset collection:")

    print("Video frames processed:", frame_index)

    print("Training crops:", len(images))

    print("Generated identities:", len(set(labels)))

    if images:

        counts = np.bincount(
            labels
        )

        valid = counts[counts >= 2]

        print(
            "Identities with at least two crops:",
            len(valid)
        )

    return images, labels


# =====================================================================
# MODEL TRAINING
# =====================================================================

def train_reid_model(
    images,
    labels,
    output_dir,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    device="auto"
):

    if len(images) < 4:

        raise RuntimeError(
            "Not enough training crops. Provide a longer video."
        )

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    unique, counts = np.unique(

        labels,

        return_counts=True
    )

    valid = unique[
        counts >= 2
    ]

    keep = np.isin(
        labels,
        valid
    )

    images = [

        image

        for image, selected in zip(images, keep)

        if selected
    ]

    labels = labels[keep]

    if len(images) < 4 or len(np.unique(labels)) < 2:

        raise RuntimeError(

            "Need at least two identities with multiple crops. "

            "The input video may not contain enough distinct objects."
        )

    dataset = ReIDCropDataset(

        images,

        labels,

        augment=True
    )

    batch_sampler = IdentityBatchSampler(
        labels
    )

    if device == "auto":

        device = (

            "cuda"

            if torch.cuda.is_available()

            else "cpu"
        )

    model = AppearanceEncoder().to(
        device
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=learning_rate,

        weight_decay=1e-4
    )

    criterion = SupervisedContrastiveLoss()

    epoch_losses = []

    print("\nTraining appearance encoder.")

    print("Device:", device)

    print("Training crops:", len(dataset))

    print("Identities:", len(np.unique(labels)))

    for epoch in range(epochs):

        model.train()

        losses = []

        batches = batch_sampler.batches(
            batch_size
        )

        for batch_indices in batches:

            batch = [

                dataset[index]

                for index in batch_indices
            ]

            images_batch = torch.stack(

                [

                    item[0]

                    for item in batch
                ]
            ).to(
                device
            )

            labels_batch = torch.stack(

                [

                    item[1]

                    for item in batch
                ]
            ).to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            embeddings = model(
                images_batch
            )

            loss = criterion(

                embeddings,

                labels_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=5.0
            )

            optimizer.step()

            losses.append(
                float(loss.item())
            )

        epoch_loss = (

            float(np.mean(losses))

            if losses else 0.0
        )

        epoch_losses.append(
            epoch_loss
        )

        print(

            f"Epoch {epoch + 1:03d}/{epochs} | "

            f"Loss: {epoch_loss:.5f}"
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True
    )

    checkpoint_path = (
        output_dir / "reid_encoder.pt"
    )

    torch.save(

        {

            "model_state_dict": model.state_dict(),

            "embedding_dim": EMBEDDING_DIM,

            "image_size": IMAGE_SIZE,

            "identity_count": int(
                len(np.unique(labels))
            ),

            "training_samples": len(images),

            "epoch_losses": epoch_losses
        },

        checkpoint_path
    )

    # Training curve.

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epoch_losses,
        marker="o"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Supervised contrastive loss")

    plt.title("Re-ID Training Loss")

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(

        output_dir / "training_curves.png",

        dpi=150
    )

    plt.close()

    metrics = evaluate_retrieval(

        model,

        images,

        labels,

        device
    )

    with open(

        output_dir / "retrieval_metrics.json",

        "w"

    ) as file:

        json.dump(

            metrics,

            file,

            indent=4
        )

    print("\nModel saved:", checkpoint_path)

    print("Retrieval metrics:", metrics)

    return model


# =====================================================================
# RETRIEVAL EVALUATION
# =====================================================================

def evaluate_retrieval(
    model,
    images,
    labels,
    device
):

    """
    Evaluate nearest-neighbor retrieval on the collected crops.

    This is a diagnostic metric, not a held-out generalization score.

    The same crop collection is used to build the gallery and
    evaluate queries. A production evaluation must use separate
    identity-disjoint or sequence-disjoint train/test data.
    """

    model.eval()

    embeddings = []

    with torch.no_grad():

        for image in images:

            tensor = image_to_tensor(
                image
            ).unsqueeze(0).to(
                device
            )

            embedding = model(
                tensor
            )

            embeddings.append(
                embedding.cpu().numpy()[0]
            )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    labels = np.asarray(
        labels
    )

    similarities = (
        embeddings @ embeddings.T
    )

    np.fill_diagonal(
        similarities,
        -np.inf
    )

    nearest = np.argmax(

        similarities,

        axis=1
    )

    accuracy = float(
        np.mean(
            labels[nearest] == labels
        )
    )

    return {

        "nearest_neighbor_accuracy": accuracy,

        "samples": len(labels),

        "identities": len(np.unique(labels)),

        "evaluation_note": (
            "In-sample diagnostic only; not a held-out test metric."
        )
    }


# =====================================================================
# MODEL LOADING
# =====================================================================

def load_reid_model(
    checkpoint_path,
    device="auto"
):

    if device == "auto":

        device = (

            "cuda"

            if torch.cuda.is_available()

            else "cpu"
        )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device,

        weights_only=True
    )

    model = AppearanceEncoder(

        embedding_dim=checkpoint.get(
            "embedding_dim",
            EMBEDDING_DIM
        )
    ).to(
        device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, device


# =====================================================================
# EMBEDDING EXTRACTION
# =====================================================================

def extract_embedding(
    model,
    frame,
    bbox,
    device
):

    crop = crop_object(
        frame,
        bbox
    )

    tensor = image_to_tensor(
        crop
    ).unsqueeze(0).to(
        device
    )

    with torch.no_grad():

        embedding = model(
            tensor
        )

    return embedding.cpu().numpy()[0]


def cosine_distance(embedding_a, embedding_b):

    a = np.asarray(
        embedding_a,
        dtype=np.float32
    )

    b = np.asarray(
        embedding_b,
        dtype=np.float32
    )

    denominator = (

        np.linalg.norm(a)

        * np.linalg.norm(b)
    )

    if denominator < 1e-12:

        return 1.0

    similarity = float(
        np.dot(a, b) / denominator
    )

    return float(
        1.0 - np.clip(similarity, -1.0, 1.0)
    )


# =====================================================================
# DETECTION
# =====================================================================

class MotionDetector:

    """
    Demonstration detector.

    MOG2 detects foreground regions, not semantic objects.
    """

    def __init__(
        self,
        min_area=500
    ):

        self.min_area = min_area

        self.background = cv2.createBackgroundSubtractorMOG2(

            history=500,

            varThreshold=32,

            detectShadows=False
        )

        self.kernel = cv2.getStructuringElement(

            cv2.MORPH_ELLIPSE,

            (5, 5)
        )

    def detect(self, frame):

        mask = self.background.apply(
            frame
        )

        mask = cv2.morphologyEx(

            mask,

            cv2.MORPH_OPEN,

            self.kernel
        )

        mask = cv2.morphologyEx(

            mask,

            cv2.MORPH_CLOSE,

            self.kernel,

            iterations=2
        )

        contours, _ = cv2.findContours(

            mask,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        for contour in contours:

            if cv2.contourArea(contour) < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            detections.append(

                {

                    "bbox": np.array(

                        [
                            x,
                            y,
                            x + w,
                            y + h
                        ],

                        dtype=np.float64
                    ),

                    "class_id": 0,

                    "confidence": 1.0
                }
            )

        return detections, mask


# =====================================================================
# TRACK DATA
# =====================================================================

class ReIDTrack:

    def __init__(
        self,
        track_id,
        detection,
        embedding,
        frame_index
    ):

        self.track_id = track_id

        self.bbox = detection["bbox"].copy()

        self.embedding = embedding.copy()

        self.class_id = detection["class_id"]

        self.state = "confirmed"

        self.age = 1

        self.hits = 1

        self.missed = 0

        self.last_seen = frame_index

        self.trajectory = [

            bbox_center(
                self.bbox
            ).tolist()
        ]

    def update(
        self,
        detection,
        embedding,
        frame_index,
        appearance_alpha=0.15
    ):

        self.bbox = detection["bbox"].copy()

        # Update long-term appearance memory.

        updated = (

            (1.0 - appearance_alpha) * self.embedding

            + appearance_alpha * embedding
        )

        norm = np.linalg.norm(
            updated
        )

        if norm > 1e-12:

            updated /= norm

        self.embedding = updated

        self.hits += 1

        self.age += 1

        self.missed = 0

        self.state = "confirmed"

        self.last_seen = frame_index

        self.trajectory.append(

            bbox_center(
                self.bbox
            ).tolist()
        )


# =====================================================================
# RE-ID TRACKER
# =====================================================================

class ReIDTracker:

    def __init__(

        self,

        appearance_threshold=0.45,

        max_center_distance=150.0,

        max_lost_frames=60,

        max_cost=0.75,

        appearance_weight=0.60,

        motion_weight=0.20,

        iou_weight=0.20
    ):

        self.tracks = {}

        self.next_id = 1

        self.frame_index = 0

        self.appearance_threshold = appearance_threshold

        self.max_center_distance = max_center_distance

        self.max_lost_frames = max_lost_frames

        self.max_cost = max_cost

        self.appearance_weight = appearance_weight

        self.motion_weight = motion_weight

        self.iou_weight = iou_weight

        self.events = {

            "created": 0,

            "recovered": 0,

            "removed": 0,

            "matches": 0
        }

    def association_cost(
        self,
        track,
        detection,
        embedding
    ):

        if track.class_id != detection["class_id"]:

            return np.inf

        distance = center_distance(

            track.bbox,

            detection["bbox"]
        )

        if distance > self.max_center_distance:

            return np.inf

        app_distance = cosine_distance(

            track.embedding,

            embedding
        )

        if app_distance > self.appearance_threshold:

            return np.inf

        overlap = iou(

            track.bbox,

            detection["bbox"]
        )

        normalized_motion = (

            distance / self.max_center_distance
        )

        return (

            self.appearance_weight * app_distance

            + self.motion_weight * normalized_motion

            + self.iou_weight * (1.0 - overlap)
        )

    def update(
        self,
        frame,
        detections,
        model,
        device
    ):

        self.frame_index += 1

        # Extract embeddings for current detections.

        embeddings = [

            extract_embedding(

                model,

                frame,

                detection["bbox"],

                device
            )

            for detection in detections
        ]

        track_ids = list(
            self.tracks.keys()
        )

        detection_indices = list(
            range(len(detections))
        )

        cost = np.full(

            (
                len(track_ids),

                len(detections)
            ),

            np.inf,

            dtype=np.float64
        )

        for i, track_id in enumerate(track_ids):

            track = self.tracks[track_id]

            for j, detection in enumerate(detections):

                cost[i, j] = self.association_cost(

                    track,

                    detection,

                    embeddings[j]
                )

        if cost.size:

            solver_cost = np.where(

                np.isfinite(cost),

                cost,

                1e6
            )

            rows, cols = linear_sum_assignment(
                solver_cost
            )

        else:

            rows = []

            cols = []

        matched_tracks = set()

        matched_detections = set()

        for row, col in zip(rows, cols):

            if not np.isfinite(cost[row, col]):
                continue

            if cost[row, col] > self.max_cost:
                continue

            track_id = track_ids[row]

            track = self.tracks[track_id]

            was_lost = (
                track.state == "lost"
            )

            track.update(

                detections[col],

                embeddings[col],

                self.frame_index
            )

            if was_lost:

                self.events["recovered"] += 1

            self.events["matches"] += 1

            matched_tracks.add(
                track_id
            )

            matched_detections.add(
                col
            )

        # Unmatched existing tracks become lost.

        for track_id in track_ids:

            if track_id not in matched_tracks:

                track = self.tracks[track_id]

                track.missed += 1

                track.age += 1

                track.state = "lost"

        # Unmatched detections become new tracks.

        for index, detection in enumerate(detections):

            if index in matched_detections:
                continue

            track = ReIDTrack(

                self.next_id,

                detection,

                embeddings[index],

                self.frame_index
            )

            self.tracks[self.next_id] = track

            self.next_id += 1

            self.events["created"] += 1

        # Remove expired tracks.

        expired = [

            track_id

            for track_id, track in self.tracks.items()

            if (

                track.state == "lost"

                and track.missed > self.max_lost_frames
            )
        ]

        for track_id in expired:

            del self.tracks[track_id]

            self.events["removed"] += 1

        return [

            track

            for track in self.tracks.values()

            if track.state == "confirmed"
        ]

    def summary(self):

        return {

            "frames_processed": self.frame_index,

            "active_tracks": sum(

                track.state == "confirmed"

                for track in self.tracks.values()
            ),

            "lost_tracks": sum(

                track.state == "lost"

                for track in self.tracks.values()
            ),

            "total_retained_tracks": len(
                self.tracks
            ),

            "events": self.events.copy()
        }


# =====================================================================
# VISUALIZATION
# =====================================================================

def draw_results(
    frame,
    tracks,
    frame_index,
    tracker
):

    output = frame.copy()

    for track in tracks:

        x1, y1, x2, y2 = track.bbox.astype(
            int
        )

        color = (

            int(track.track_id * 71 % 255),

            int(track.track_id * 137 % 255),

            int(track.track_id * 193 % 255)
        )

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            color,

            2
        )

        cv2.putText(

            output,

            f"ID {track.track_id}",

            (x1, max(20, y1 - 7)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            color,

            2
        )

        trajectory = np.asarray(
            track.trajectory,
            dtype=np.int32
        )

        if len(trajectory) >= 2:

            cv2.polylines(

                output,

                [trajectory.reshape(-1, 1, 2)],

                False,

                color,

                2
            )

    cv2.putText(

        output,

        f"Frame: {frame_index}",

        (15, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )

    cv2.putText(

        output,

        f"Active: {tracker.summary()['active_tracks']}",

        (15, 60),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 255, 0),

        2
    )

    cv2.putText(

        output,

        f"Lost: {tracker.summary()['lost_tracks']}",

        (15, 85),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (0, 165, 255),

        2
    )

    cv2.putText(

        output,

        f"Recovered: {tracker.events['recovered']}",

        (15, 110),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 0),

        2
    )

    return output


# =====================================================================
# VIDEO PROCESSING
# =====================================================================

def process_video(args):

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(

            f"Checkpoint not found: {checkpoint_path}\n"

            "Train the model first using --train."
        )

    model, device = load_reid_model(

        checkpoint_path,

        args.device
    )

    capture = cv2.VideoCapture(
        args.video
    )

    if not capture.isOpened():

        raise RuntimeError(
            "Cannot open video."
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if not np.isfinite(fps) or fps <= 0:

        fps = 30.0

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(

        parents=True,

        exist_ok=True
    )

    writer = cv2.VideoWriter(

        str(output_path),

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (width, height)
    )

    if not writer.isOpened():

        capture.release()

        raise RuntimeError(
            "Could not open output video writer."
        )

    detector = MotionDetector(
        min_area=args.min_area
    )

    tracker = ReIDTracker(

        appearance_threshold=args.appearance_threshold,

        max_center_distance=args.max_center_distance,

        max_lost_frames=args.max_lost_frames,

        max_cost=args.max_cost
    )

    frame_index = 0

    processing_times = []

    try:

        while True:

            start = time.perf_counter()

            ret, frame = capture.read()

            if not ret:
                break

            frame_index += 1

            detections, mask = detector.detect(
                frame
            )

            tracks = tracker.update(

                frame,

                detections,

                model,

                device
            )

            output = draw_results(

                frame,

                tracks,

                frame_index,

                tracker
            )

            elapsed = (
                time.perf_counter() - start
            )

            processing_times.append(
                elapsed
            )

            writer.write(
                output
            )

            if not args.no_display:

                cv2.imshow(
                    "Lesson 49 - Re-ID Tracking",
                    output
                )

                if args.show_mask:

                    cv2.imshow(
                        "Foreground Mask",
                        mask
                    )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_index % 100 == 0:

                print(

                    f"Frame {frame_index} | "

                    f"Detections: {len(detections)} | "

                    f"Active: {tracker.summary()['active_tracks']} | "

                    f"Lost: {tracker.summary()['lost_tracks']} | "

                    f"Recovered: {tracker.events['recovered']}"
                )

    finally:

        capture.release()

        writer.release()

        cv2.destroyAllWindows()

    average_time = (

        float(np.mean(processing_times))

        if processing_times else 0.0
    )

    summary = tracker.summary()

    summary.update(

        {

            "input_video": args.video,

            "output_video": str(output_path),

            "resolution": [
                width,
                height
            ],

            "device": device,

            "average_processing_fps": (

                1.0 / average_time

                if average_time > 0 else 0.0
            )
        }
    )

    summary_path = (

        output_path.parent

        / "tracking_summary.json"
    )

    with open(
        summary_path,
        "w"
    ) as file:

        json.dump(

            summary,

            file,

            indent=4
        )

    print("\nTracking completed.")

    print(
        json.dumps(
            summary,
            indent=4
        )
    )

    print("\nOutput video:", output_path)

    print("Summary:", summary_path)


# =====================================================================
# UNIT TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print("LESSON 49 — RE-ID TRACKING TESTS")

    print("=" * 70)

    # ---------------------------------------------------------------
    # Test 1: CNN output dimensions.
    # ---------------------------------------------------------------

    model = AppearanceEncoder(
        embedding_dim=128
    )

    x = torch.rand(
        4,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    model.eval()

    with torch.no_grad():

        embeddings = model(x)

    assert embeddings.shape == (
        4,
        128
    )

    print("Test 1 passed: CNN embedding dimensions")

    # ---------------------------------------------------------------
    # Test 2: embeddings are normalized.
    # ---------------------------------------------------------------

    norms = torch.norm(
        embeddings,
        dim=1
    )

    assert torch.allclose(

        norms,

        torch.ones_like(norms),

        atol=1e-5
    )

    print("Test 2 passed: L2-normalized embeddings")

    # ---------------------------------------------------------------
    # Test 3: cosine distance.
    # ---------------------------------------------------------------

    a = np.array(
        [1.0, 0.0, 0.0]
    )

    b = np.array(
        [1.0, 0.0, 0.0]
    )

    assert abs(
        cosine_distance(a, b)
    ) < 1e-6

    print("Test 3 passed: identical embedding distance")

    # ---------------------------------------------------------------
    # Test 4: orthogonal embeddings.
    # ---------------------------------------------------------------

    a = np.array(
        [1.0, 0.0]
    )

    b = np.array(
        [0.0, 1.0]
    )

    assert abs(
        cosine_distance(a, b) - 1.0
    ) < 1e-6

    print("Test 4 passed: orthogonal embedding distance")

    # ---------------------------------------------------------------
    # Test 5: opposite embeddings.
    # ---------------------------------------------------------------

    a = np.array(
        [1.0, 0.0]
    )

    b = np.array(
        [-1.0, 0.0]
    )

    assert abs(
        cosine_distance(a, b) - 2.0
    ) < 1e-6

    print("Test 5 passed: opposite embedding distance")

    # ---------------------------------------------------------------
    # Test 6: supervised contrastive loss.
    # ---------------------------------------------------------------

    criterion = SupervisedContrastiveLoss()

    embeddings = F.normalize(

        torch.randn(
            8,
            32
        ),

        dim=1
    )

    labels = torch.tensor(

        [
            0, 0,
            1, 1,
            2, 2,
            3, 3
        ]
    )

    loss = criterion(
        embeddings,
        labels
    )

    assert torch.isfinite(loss)

    assert loss.item() >= 0

    print("Test 6 passed: supervised contrastive loss")

    # ---------------------------------------------------------------
    # Test 7: contrastive loss backward.
    # ---------------------------------------------------------------

    embeddings = torch.randn(

        8,

        32,

        requires_grad=True
    )

    labels = torch.tensor(
        [0, 0, 1, 1, 2, 2, 3, 3]
    )

    loss = criterion(
        embeddings,
        labels
    )

    loss.backward()

    assert embeddings.grad is not None

    assert torch.isfinite(
        embeddings.grad
    ).all()

    print("Test 7 passed: contrastive gradients")

    # ---------------------------------------------------------------
    # Test 8: no positive pairs.
    # ---------------------------------------------------------------

    embeddings = torch.randn(
        4,
        16,
        requires_grad=True
    )

    labels = torch.tensor(
        [0, 1, 2, 3]
    )

    loss = criterion(
        embeddings,
        labels
    )

    assert loss.item() == 0.0

    print("Test 8 passed: no-positive-pair handling")

    # ---------------------------------------------------------------
    # Test 9: image preprocessing.
    # ---------------------------------------------------------------

    image = np.random.default_rng(
        42
    ).integers(

        0,

        256,

        size=(80, 100, 3),

        dtype=np.uint8
    )

    tensor = image_to_tensor(
        image
    )

    assert tensor.shape == (
        3,
        80,
        100
    )

    assert tensor.min() >= 0

    assert tensor.max() <= 1

    print("Test 9 passed: image-to-tensor conversion")

    # ---------------------------------------------------------------
    # Test 10: object crop.
    # ---------------------------------------------------------------

    crop = crop_object(

        image,

        [10, 10, 60, 60]
    )

    assert crop.shape == (
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )

    print("Test 10 passed: object crop extraction")

    # ---------------------------------------------------------------
    # Test 11: box IoU.
    # ---------------------------------------------------------------

    box = np.array(
        [0, 0, 100, 100]
    )

    assert abs(
        iou(box, box) - 1.0
    ) < 1e-8

    print("Test 11 passed: bounding-box IoU")

    # ---------------------------------------------------------------
    # Test 12: tracker initialization.
    # ---------------------------------------------------------------

    tracker = ReIDTracker()

    assert tracker.next_id == 1

    assert len(tracker.tracks) == 0

    print("Test 12 passed: tracker initialization")

    # ---------------------------------------------------------------
    # Test 13: track creation.
    # ---------------------------------------------------------------

    detection = {

        "bbox": np.array(
            [10, 10, 50, 50]
        ),

        "class_id": 0
    }

    embedding = np.zeros(
        128,
        dtype=np.float32
    )

    embedding[0] = 1.0

    track = ReIDTrack(

        1,

        detection,

        embedding,

        1
    )

    assert track.track_id == 1

    assert track.state == "confirmed"

    print("Test 13 passed: Re-ID track creation")

    # ---------------------------------------------------------------
    # Test 14: track appearance update.
    # ---------------------------------------------------------------

    new_embedding = np.zeros(
        128,
        dtype=np.float32
    )

    new_embedding[1] = 1.0

    track.update(

        detection,

        new_embedding,

        2
    )

    assert abs(
        np.linalg.norm(track.embedding) - 1.0
    ) < 1e-5

    print("Test 14 passed: normalized appearance update")

    # ---------------------------------------------------------------
    # Test 15: appearance-based association.
    # ---------------------------------------------------------------

    tracker = ReIDTracker(

        appearance_threshold=0.2,

        max_center_distance=1000
    )

    track = ReIDTrack(

        1,

        detection,

        embedding,

        1
    )

    similar = embedding.copy()

    cost = tracker.association_cost(

        track,

        detection,

        similar
    )

    assert np.isfinite(cost)

    print("Test 15 passed: similar appearance association")

    # ---------------------------------------------------------------
    # Test 16: dissimilar appearance rejection.
    # ---------------------------------------------------------------

    different = np.zeros(
        128,
        dtype=np.float32
    )

    different[1] = 1.0

    cost = tracker.association_cost(

        track,

        detection,

        different
    )

    assert not np.isfinite(cost)

    print("Test 16 passed: dissimilar appearance rejection")

    # ---------------------------------------------------------------
    # Test 17: end-to-end track creation.
    # ---------------------------------------------------------------

    model = AppearanceEncoder()

    model.eval()

    tracker = ReIDTracker()

    frame = np.random.default_rng(
        123
    ).integers(

        0,

        256,

        size=(128, 128, 3),

        dtype=np.uint8
    )

    detections = [

        {

            "bbox": np.array(
                [20, 20, 70, 70]
            ),

            "class_id": 0
        }
    ]

    tracks = tracker.update(

        frame,

        detections,

        model,

        "cpu"
    )

    assert len(tracks) == 1

    assert tracks[0].track_id == 1

    print("Test 17 passed: end-to-end track creation")

    # ---------------------------------------------------------------
    # Test 18: track recovery after one missing frame.
    # ---------------------------------------------------------------

    tracker.update(

        frame,

        [],

        model,

        "cpu"
    )

    assert tracker.tracks[1].state == "lost"

    tracker.update(

        frame,

        detections,

        model,

        "cpu"
    )

    assert tracker.tracks[1].state == "confirmed"

    assert tracker.next_id == 2

    assert tracker.events["recovered"] == 1

    print("Test 18 passed: lost-track recovery")

    # ---------------------------------------------------------------
    # Test 19: track expiration.
    # ---------------------------------------------------------------

    tracker = ReIDTracker(
        max_lost_frames=2
    )

    tracker.update(

        frame,

        detections,

        model,

        "cpu"
    )

    tracker.update(frame, [], model, "cpu")

    tracker.update(frame, [], model, "cpu")

    tracker.update(frame, [], model, "cpu")

    assert len(tracker.tracks) == 0

    assert tracker.events["removed"] == 1

    print("Test 19 passed: track expiration")

    # ---------------------------------------------------------------
    # Test 20: tracker summary.
    # ---------------------------------------------------------------

    tracker = ReIDTracker()

    tracker.update(

        frame,

        detections,

        model,

        "cpu"
    )

    summary = tracker.summary()

    assert summary["frames_processed"] == 1

    assert summary["active_tracks"] == 1

    print("Test 20 passed: tracker summary")

    print("\nAll 20 tests passed.")


# =====================================================================
# MAIN
# =====================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Lesson 49 — Re-ID Multi-Object Tracking"
        )
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "runs/lesson49/"
            "multi_object_reid_tracking.mp4"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "runs/lesson49/"
            "reid_encoder.pt"
        )
    )

    parser.add_argument(
        "--train",
        action="store_true"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=1500
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=500
    )

    parser.add_argument(
        "--appearance-threshold",
        type=float,
        default=0.45
    )

    parser.add_argument(
        "--max-center-distance",
        type=float,
        default=150.0
    )

    parser.add_argument(
        "--max-lost-frames",
        type=int,
        default=60
    )

    parser.add_argument(
        "--max-cost",
        type=float,
        default=0.75
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto"
    )

    parser.add_argument(
        "--show-mask",
        action="store_true"
    )

    parser.add_argument(
        "--no-display",
        action="store_true"
    )

    parser.add_argument(
        "--check-only",
        action="store_true"
    )

    args = parser.parse_args()

    if args.epochs < 1:

        parser.error(
            "--epochs must be positive."
        )

    if args.batch_size < 4:

        parser.error(
            "--batch-size must be at least 4."
        )

    if args.max_frames < 1:

        parser.error(
            "--max-frames must be positive."
        )

    if args.device not in (
        "auto",
        "cpu",
        "cuda"
    ):

        parser.error(
            "--device must be auto, cpu or cuda."
        )

    set_seed(42)

    print("=" * 70)

    print(
        "LESSON 49 — RE-ID MULTI-OBJECT TRACKING"
    )

    print("=" * 70)

    if args.check_only:

        run_tests()

        print("\nCheck-only completed.")

        return

    if args.video is None:

        parser.error(
            "--video is required unless --check-only is used."
        )

    if args.train:

        images, labels = build_training_crops(

            args.video,

            max_frames=args.max_frames,

            min_area=args.min_area
        )

        if not images:

            raise RuntimeError(
                "No training crops were collected."
            )

        train_reid_model(

            images,

            labels,

            Path(args.checkpoint).parent,

            epochs=args.epochs,

            batch_size=args.batch_size,

            learning_rate=args.learning_rate,

            device=args.device
        )

    process_video(args)


if __name__ == "__main__":

    main()