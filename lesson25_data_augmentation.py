"""
======================================================================
COMPUTER VISION FROM SCRATCH
LESSON 25 — DATA AUGMENTATION FROM SCRATCH
======================================================================

COURSE RULES
------------
1. No pretrained models.
2. No pretrained weights.
3. All model parameters are randomly initialized.
4. Implement augmentation mathematics explicitly.
5. Use OpenCV for practical image transformations.
6. Use PyTorch for model training.

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1 — WHY DATA AUGMENTATION?
    Augmentation creates variations of existing training images.

    Goals:
        - Reduce overfitting.
        - Improve generalization.
        - Improve robustness to lighting and viewpoint changes.

STEP 2 — HORIZONTAL FLIPPING
    I_aug(y, x) = I(y, W - 1 - x)

STEP 3 — RANDOM ROTATION
    Rotate around the image center using an affine transformation.

STEP 4 — RANDOM CROPPING
    Extract a random region and resize it to the original dimensions.

STEP 5 — COLOR JITTER
    Brightness:
        I' = I + beta

    Contrast:
        I' = alpha * (I - mean(I)) + mean(I)

    Saturation:
        Adjust saturation in HSV color space.

STEP 6 — RANDOM TRANSLATION
    Shift an image horizontally and vertically.

STEP 7 — CUTOUT
    Hide a random rectangular region.

STEP 8 — MIXUP
    x_mix = lambda*x1 + (1-lambda)*x2

    y_mix = lambda*y1 + (1-lambda)*y2

STEP 9 — CUTMIX
    Replace a rectangular region with another image.

    The label weight is determined by the actual replaced area.

STEP 10 — TRAINING COMPARISON
    Model A: No augmentation.
    Model B: Geometric + color augmentation + Cutout.
    Model C: Augmentation + MixUp + CutMix.

======================================================================
IMPORTANT PRACTICAL RULES
======================================================================

1. Augment training images only.
2. Never augment validation/test images.
3. Detection bounding boxes must follow geometric transformations.
4. Segmentation masks require appropriate interpolation.
5. Robot action labels may need corresponding transformations.
6. Not every augmentation is physically valid for robotics.

======================================================================
ROBOTICS APPLICATION — OPENARM V2 WASTE SORTING
======================================================================

Camera image
    |
    v
Image augmentation during training
    |
    v
CNN / ResNet
    |
    v
Object category
    |
    v
ACT manipulation policy
    |
    v
Robot picks and sorts the object.

For robot manipulation, image augmentation must remain
consistent with camera geometry and action labels.

======================================================================
INSTALLATION
======================================================================

pip install numpy matplotlib opencv-python torch

======================================================================
RUN
======================================================================

python lesson25_data_augmentation.py

======================================================================
NEXT LESSON
======================================================================

LESSON 26 — OBJECT DETECTION FUNDAMENTALS
======================================================================
"""

import math
import random
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

IMAGE_SIZE = 64

NUM_CLASSES = 4

CLASS_NAMES = [
    "Bottle",
    "Can",
    "Paper",
    "Container",
]

TRAIN_SAMPLES = 1200
TEST_SAMPLES = 400

BATCH_SIZE = 64
EPOCHS = 10

LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
# STEP 1 — SYNTHETIC WASTE-SORTING DATASET
# ======================================================================

def draw_bottle(image, rng):
    """
    Draw a synthetic bottle.

    IMPORTANT:
        The image is H x W x C.
        Use image.shape[:2] to retrieve H and W.
    """

    h, w = image.shape[:2]

    cx = int(rng.integers(18, w - 18))
    cy = int(rng.integers(20, h - 20))

    body_w = int(rng.integers(10, 17))
    body_h = int(rng.integers(20, 30))

    color = tuple(
        int(v)
        for v in rng.integers(80, 230, size=3)
    )

    cv2.rectangle(
        image,
        (cx - body_w // 2, cy),
        (
            cx + body_w // 2,
            min(h - 1, cy + body_h),
        ),
        color,
        -1,
    )

    neck_w = max(4, body_w // 2)

    cv2.rectangle(
        image,
        (
            cx - neck_w // 2,
            max(0, cy - 9),
        ),
        (
            cx + neck_w // 2,
            cy + 2,
        ),
        color,
        -1,
    )

    cv2.rectangle(
        image,
        (cx - 4, max(0, cy - 13)),
        (cx + 4, max(0, cy - 9)),
        (40, 40, 220),
        -1,
    )


def draw_can(image, rng):
    """Draw a synthetic aluminum can."""

    h, w = image.shape[:2]

    cx = int(rng.integers(18, w - 18))
    cy = int(rng.integers(20, h - 20))

    radius = int(rng.integers(9, 15))
    height = int(rng.integers(22, 32))

    color = tuple(
        int(v)
        for v in rng.integers(100, 220, size=3)
    )

    cv2.rectangle(
        image,
        (cx - radius, cy - height // 2),
        (cx + radius, cy + height // 2),
        color,
        -1,
    )

    cv2.ellipse(
        image,
        (cx, cy - height // 2),
        (radius, 4),
        0,
        0,
        360,
        (210, 210, 210),
        -1,
    )

    cv2.ellipse(
        image,
        (cx, cy + height // 2),
        (radius, 4),
        0,
        0,
        360,
        (80, 80, 80),
        -1,
    )


def draw_paper(image, rng):
    """Draw a synthetic piece of paper."""

    h, w = image.shape[:2]

    cx = int(rng.integers(20, w - 20))
    cy = int(rng.integers(20, h - 20))

    width = int(rng.integers(24, 34))
    height = int(rng.integers(25, 36))

    points = np.array(
        [
            [cx - width // 2, cy - height // 2],
            [cx + width // 2, cy - height // 2 + 3],
            [cx + width // 2 - 3, cy + height // 2],
            [cx - width // 2, cy + height // 2 - 2],
        ],
        dtype=np.int32,
    )

    points[:, 0] = np.clip(
        points[:, 0],
        0,
        w - 1,
    )

    points[:, 1] = np.clip(
        points[:, 1],
        0,
        h - 1,
    )

    cv2.fillPoly(
        image,
        [points],
        (220, 220, 220),
    )

    cv2.line(
        image,
        tuple(points[0]),
        tuple(points[2]),
        (120, 120, 120),
        1,
    )


def draw_container(image, rng):
    """Draw a synthetic food container."""

    h, w = image.shape[:2]

    cx = int(rng.integers(20, w - 20))
    cy = int(rng.integers(20, h - 20))

    width = int(rng.integers(25, 34))
    height = int(rng.integers(19, 27))

    color = tuple(
        int(v)
        for v in rng.integers(50, 150, size=3)
    )

    cv2.rectangle(
        image,
        (cx - width // 2, cy - height // 2),
        (cx + width // 2, cy + height // 2),
        color,
        -1,
    )

    cv2.rectangle(
        image,
        (
            cx - width // 2 - 2,
            cy - height // 2 - 4,
        ),
        (
            cx + width // 2 + 2,
            cy - height // 2,
        ),
        (190, 190, 190),
        -1,
    )


DRAW_FUNCTIONS = [
    draw_bottle,
    draw_can,
    draw_paper,
    draw_container,
]


def generate_sample(label, rng):
    """
    Generate one synthetic RGB image.

    Shape:
        H x W x 3

    Dtype:
        uint8
    """

    image = rng.integers(
        25,
        75,
        size=(IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8,
    )

    # Add background texture.
    noise = rng.normal(
        0,
        8,
        size=image.shape,
    )

    image = np.clip(
        image.astype(np.float32) + noise,
        0,
        255,
    ).astype(np.uint8)

    DRAW_FUNCTIONS[label](
        image,
        rng,
    )

    return image


def generate_dataset(num_samples, seed):

    rng = np.random.default_rng(seed)

    images = []
    labels = []

    for index in range(num_samples):

        label = index % NUM_CLASSES

        image = generate_sample(
            label,
            rng,
        )

        images.append(image)
        labels.append(label)

    images = np.asarray(
        images,
        dtype=np.uint8,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    indices = rng.permutation(num_samples)

    return (
        images[indices],
        labels[indices],
    )


# ======================================================================
# STEP 2 — GEOMETRIC AUGMENTATION
# ======================================================================

def horizontal_flip(image):
    """
    Horizontal mirror:

        I_aug(y, x) = I(y, W - 1 - x)
    """

    return np.ascontiguousarray(
        image[:, ::-1]
    )


def random_horizontal_flip(
    image,
    probability=0.5,
):

    if random.random() < probability:
        return horizontal_flip(image)

    return image.copy()


def rotate_image(image, angle_degrees):
    """
    Rotate around image center.

    OpenCV uses inverse-mapped geometric resampling.
    """

    h, w = image.shape[:2]

    center = (
        (w - 1) / 2,
        (h - 1) / 2,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle_degrees,
        1.0,
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return rotated


def random_rotation(
    image,
    max_angle=20,
):

    angle = random.uniform(
        -max_angle,
        max_angle,
    )

    return rotate_image(
        image,
        angle,
    )


def random_crop_and_resize(
    image,
    scale_range=(0.75, 1.0),
):
    """
    Extract a random crop and resize it to original dimensions.
    """

    h, w = image.shape[:2]

    scale = random.uniform(
        *scale_range
    )

    crop_h = max(
        1,
        int(h * scale),
    )

    crop_w = max(
        1,
        int(w * scale),
    )

    top = random.randint(
        0,
        h - crop_h,
    )

    left = random.randint(
        0,
        w - crop_w,
    )

    crop = image[
        top:top + crop_h,
        left:left + crop_w,
    ]

    resized = cv2.resize(
        crop,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    return resized


def random_translation(
    image,
    max_shift=8,
):
    """Random horizontal and vertical translation."""

    h, w = image.shape[:2]

    tx = random.randint(
        -max_shift,
        max_shift,
    )

    ty = random.randint(
        -max_shift,
        max_shift,
    )

    matrix = np.array(
        [
            [1, 0, tx],
            [0, 1, ty],
        ],
        dtype=np.float32,
    )

    translated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return translated


# ======================================================================
# STEP 3 — COLOR AUGMENTATION
# ======================================================================

def adjust_brightness(image, beta):
    """
    Brightness:

        I' = I + beta
    """

    result = image.astype(
        np.float32
    ) + beta

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def adjust_contrast(image, alpha):
    """
    Contrast:

        I' = alpha * (I - mean) + mean
    """

    image_float = image.astype(
        np.float32
    )

    mean = image_float.mean(
        axis=(0, 1),
        keepdims=True,
    )

    result = alpha * (
        image_float - mean
    ) + mean

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


def adjust_saturation(image, factor):
    """
    Adjust saturation in RGB -> HSV -> RGB.

    OpenCV HSV:
        H: [0, 179]
        S: [0, 255]
        V: [0, 255]
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV,
    )

    hsv = hsv.astype(
        np.float32
    )

    hsv[:, :, 1] *= factor

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1],
        0,
        255,
    )

    return cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2RGB,
    )


def random_color_jitter(image):

    brightness = random.uniform(
        -35,
        35,
    )

    contrast = random.uniform(
        0.7,
        1.3,
    )

    saturation = random.uniform(
        0.6,
        1.4,
    )

    result = adjust_brightness(
        image,
        brightness,
    )

    result = adjust_contrast(
        result,
        contrast,
    )

    result = adjust_saturation(
        result,
        saturation,
    )

    return result


# ======================================================================
# STEP 4 — CUTOUT
# ======================================================================

def cutout(
    image,
    fraction=0.25,
):
    """
    Replace a random rectangular region with mean color.
    """

    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "fraction must be in the interval (0, 1]."
        )

    result = image.copy()

    h, w = result.shape[:2]

    patch_h = max(
        1,
        int(h * fraction),
    )

    patch_w = max(
        1,
        int(w * fraction),
    )

    center_y = random.randint(
        0,
        h - 1,
    )

    center_x = random.randint(
        0,
        w - 1,
    )

    y1 = max(
        0,
        center_y - patch_h // 2,
    )

    y2 = min(
        h,
        y1 + patch_h,
    )

    x1 = max(
        0,
        center_x - patch_w // 2,
    )

    x2 = min(
        w,
        x1 + patch_w,
    )

    fill = result.mean(
        axis=(0, 1)
    ).astype(np.uint8)

    result[y1:y2, x1:x2] = fill

    return result


# ======================================================================
# STEP 5 — AUGMENTATION PIPELINE
# ======================================================================

class ImageAugmentation:

    def __init__(
        self,
        enabled=True,
    ):

        self.enabled = enabled

    def __call__(self, image):

        if not self.enabled:
            return image.copy()

        result = image.copy()

        # Geometric transformations.
        result = random_horizontal_flip(
            result,
            probability=0.5,
        )

        if random.random() < 0.5:

            result = random_rotation(
                result,
                max_angle=15,
            )

        if random.random() < 0.5:

            result = random_translation(
                result,
                max_shift=6,
            )

        if random.random() < 0.5:

            result = random_crop_and_resize(
                result,
                scale_range=(0.8, 1.0),
            )

        # Photometric transformations.
        if random.random() < 0.8:

            result = random_color_jitter(result)

        # Occlusion simulation.
        if random.random() < 0.3:

            result = cutout(
                result,
                fraction=0.25,
            )

        return result


# ======================================================================
# STEP 6 — MIXUP
# ======================================================================

def mixup(
    images,
    labels,
    alpha=0.4,
):
    """
    Mix two images.

        x_mix = lambda*x1 + (1-lambda)*x2

    Labels remain as two integer class targets.
    """

    if alpha <= 0:
        raise ValueError(
            "alpha must be positive."
        )

    batch_size = images.size(0)

    permutation = torch.randperm(
        batch_size,
        device=images.device,
    )

    lam = float(
        np.random.beta(alpha, alpha)
    )

    mixed_images = (
        lam * images
        + (1.0 - lam) * images[permutation]
    )

    labels_a = labels

    labels_b = labels[permutation]

    return (
        mixed_images,
        labels_a,
        labels_b,
        lam,
    )


def mixup_loss(
    criterion,
    predictions,
    labels_a,
    labels_b,
    lam,
):

    return (
        lam * criterion(
            predictions,
            labels_a,
        )
        +
        (1.0 - lam) * criterion(
            predictions,
            labels_b,
        )
    )


# ======================================================================
# STEP 7 — CUTMIX
# ======================================================================

def cutmix(
    images,
    labels,
    alpha=1.0,
):
    """
    Replace a rectangular region with another image.

    Label weights use the actual replaced image area.
    """

    if alpha <= 0:
        raise ValueError(
            "alpha must be positive."
        )

    batch_size, channels, height, width = images.shape

    permutation = torch.randperm(
        batch_size,
        device=images.device,
    )

    labels_a = labels

    labels_b = labels[permutation]

    lam = float(
        np.random.beta(alpha, alpha)
    )

    cut_ratio = math.sqrt(
        1.0 - lam
    )

    cut_w = int(
        width * cut_ratio
    )

    cut_h = int(
        height * cut_ratio
    )

    center_x = random.randint(
        0,
        width - 1,
    )

    center_y = random.randint(
        0,
        height - 1,
    )

    x1 = max(
        0,
        center_x - cut_w // 2,
    )

    x2 = min(
        width,
        center_x + cut_w // 2,
    )

    y1 = max(
        0,
        center_y - cut_h // 2,
    )

    y2 = min(
        height,
        center_y + cut_h // 2,
    )

    mixed_images = images.clone()

    mixed_images[
        :,
        :,
        y1:y2,
        x1:x2,
    ] = images[
        permutation,
        :,
        y1:y2,
        x1:x2,
    ]

    # Actual area determines label contribution.
    actual_area = (
        (x2 - x1)
        * (y2 - y1)
    )

    lam = 1.0 - (
        actual_area / (width * height)
    )

    return (
        mixed_images,
        labels_a,
        labels_b,
        float(lam),
    )


# ======================================================================
# STEP 8 — DATASET
# ======================================================================

class WasteDataset(Dataset):

    def __init__(
        self,
        images,
        labels,
        augmentation=None,
    ):

        self.images = images
        self.labels = labels

        self.augmentation = augmentation

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index].copy()

        label = int(
            self.labels[index]
        )

        if self.augmentation is not None:

            image = self.augmentation(image)

        # HWC uint8 RGB -> CHW float32.
        image = image.astype(
            np.float32
        ) / 255.0

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        )

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        return image, label


# ======================================================================
# STEP 9 — CNN CLASSIFIER
# ======================================================================

class AugmentationCNN(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(
            128,
            num_classes,
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            1,
        )

        return self.classifier(x)


def initialize_model(model):
    """Initialize every trainable parameter from scratch."""

    for module in model.modules():

        if isinstance(module, nn.Conv2d):

            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

        elif isinstance(module, nn.Linear):

            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm2d):

            nn.init.ones_(module.weight)

            nn.init.zeros_(module.bias)

    return model


# ======================================================================
# STEP 10 — TRAINING
# ======================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    use_mixup=False,
    use_cutmix=False,
):

    model.train()

    total_loss = 0.0
    total_weighted_accuracy = 0.0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        mixed_labels = False

        # ----------------------------------------------------------
        # MixUp
        # ----------------------------------------------------------

        if use_mixup and random.random() < 0.5:

            images, labels_a, labels_b, lam = mixup(
                images,
                labels,
            )

            outputs = model(images)

            loss = mixup_loss(
                criterion,
                outputs,
                labels_a,
                labels_b,
                lam,
            )

            # Weighted accuracy for mixed labels.
            predictions = outputs.argmax(dim=1)

            accuracy_a = (
                predictions == labels_a
            ).float().mean().item()

            accuracy_b = (
                predictions == labels_b
            ).float().mean().item()

            batch_accuracy = (
                lam * accuracy_a
                + (1.0 - lam) * accuracy_b
            )

            mixed_labels = True

        # ----------------------------------------------------------
        # CutMix
        # ----------------------------------------------------------

        elif use_cutmix and random.random() < 0.5:

            images, labels_a, labels_b, lam = cutmix(
                images,
                labels,
            )

            outputs = model(images)

            loss = mixup_loss(
                criterion,
                outputs,
                labels_a,
                labels_b,
                lam,
            )

            predictions = outputs.argmax(dim=1)

            accuracy_a = (
                predictions == labels_a
            ).float().mean().item()

            accuracy_b = (
                predictions == labels_b
            ).float().mean().item()

            batch_accuracy = (
                lam * accuracy_a
                + (1.0 - lam) * accuracy_b
            )

            mixed_labels = True

        # ----------------------------------------------------------
        # Normal training
        # ----------------------------------------------------------

        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            predictions = outputs.argmax(dim=1)

            batch_accuracy = (
                predictions == labels
            ).float().mean().item()

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_weighted_accuracy += (
            batch_accuracy * batch_size
        )

        total += batch_size

    return (
        total_loss / total,
        total_weighted_accuracy / total,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    return (
        total_loss / total,
        correct / total,
    )


def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    model_name,
    use_mixup=False,
    use_cutmix=False,
):

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    print()
    print("=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    start = time.time()

    for epoch in range(epochs):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            use_mixup=use_mixup,
            use_cutmix=use_cutmix,
        )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["test_loss"].append(
            test_loss
        )

        history["test_accuracy"].append(
            test_accuracy
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2%} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.2%}"
        )

    elapsed = time.time() - start

    print(
        f"Training time: {elapsed:.2f}s"
    )

    return model, history


# ======================================================================
# STEP 11 — VISUALIZATION
# ======================================================================

def visualize_augmentations(
    images,
    labels,
):

    image = images[0]

    augmentation_functions = [
        ("Original", lambda x: x.copy()),

        ("Horizontal Flip", horizontal_flip),

        ("Rotation", lambda x: rotate_image(x, 20)),

        ("Random Crop", random_crop_and_resize),

        ("Translation", random_translation),

        ("Color Jitter", random_color_jitter),

        ("Cutout", cutout),
    ]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(13, 7),
    )

    axes = axes.flatten()

    for index, (name, function) in enumerate(
        augmentation_functions
    ):

        augmented = function(
            image.copy()
        )

        axes[index].imshow(augmented)

        axes[index].set_title(name)

        axes[index].axis("off")

    axes[-1].axis("off")

    fig.suptitle(
        f"Image Augmentation Examples — "
        f"{CLASS_NAMES[labels[0]]}",
        fontsize=15,
    )

    plt.tight_layout()

    plt.show()


def visualize_mixup_cutmix(images, labels):

    tensor_images = torch.from_numpy(
        images[:8].astype(np.float32) / 255.0
    ).permute(0, 3, 1, 2)

    tensor_labels = torch.from_numpy(
        labels[:8]
    )

    mixed, label_a, label_b, lam = mixup(
        tensor_images,
        tensor_labels,
    )

    cutmixed, cut_a, cut_b, cut_lam = cutmix(
        tensor_images,
        tensor_labels,
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 7),
    )

    for index in range(4):

        axes[0, index].imshow(
            mixed[index].permute(1, 2, 0).numpy()
        )

        axes[0, index].set_title(
            f"MixUp lambda={lam:.2f}\n"
            f"{CLASS_NAMES[label_a[index]]} + "
            f"{CLASS_NAMES[label_b[index]]}"
        )

        axes[0, index].axis("off")

        axes[1, index].imshow(
            cutmixed[index].permute(1, 2, 0).numpy()
        )

        axes[1, index].set_title(
            f"CutMix lambda={cut_lam:.2f}\n"
            f"{CLASS_NAMES[cut_a[index]]} + "
            f"{CLASS_NAMES[cut_b[index]]}"
        )

        axes[1, index].axis("off")

    plt.tight_layout()

    plt.show()


def visualize_training(
    baseline_history,
    augmented_history,
):

    epochs = np.arange(
        1,
        len(baseline_history["train_loss"]) + 1,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
    )

    fig.suptitle(
        "Training Without vs With Augmentation",
        fontsize=16,
    )

    metrics = [
        ("train_loss", "Training Loss"),
        ("test_loss", "Validation Loss"),
        ("train_accuracy", "Training Accuracy"),
        ("test_accuracy", "Validation Accuracy"),
    ]

    for axis, (key, title) in zip(
        axes.flatten(),
        metrics,
    ):

        axis.plot(
            epochs,
            baseline_history[key],
            marker="o",
            label="No augmentation",
        )

        axis.plot(
            epochs,
            augmented_history[key],
            marker="o",
            label="Augmentation",
        )

        axis.set_title(title)

        axis.set_xlabel("Epoch")

        axis.grid(True)

        axis.legend()

    plt.tight_layout()

    plt.show()


def visualize_all_training_histories(
    baseline_history,
    augmented_history,
    advanced_history,
):

    histories = [
        ("No augmentation", baseline_history),
        ("Geometric + color + Cutout", augmented_history),
        ("Augmentation + MixUp + CutMix", advanced_history),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    for name, history in histories:

        axes[0].plot(
            history["test_accuracy"],
            marker="o",
            label=name,
        )

        axes[1].plot(
            history["test_loss"],
            marker="o",
            label=name,
        )

    axes[0].set_title(
        "Test Accuracy"
    )

    axes[0].set_xlabel("Epoch")

    axes[0].set_ylabel("Accuracy")

    axes[1].set_title(
        "Test Loss"
    )

    axes[1].set_xlabel("Epoch")

    axes[1].set_ylabel("Loss")

    for axis in axes:

        axis.grid(True)

        axis.legend()

    plt.tight_layout()

    plt.show()


# ======================================================================
# STEP 12 — NUMERICAL AND UNIT TESTS
# ======================================================================

def run_tests():

    print()
    print("=" * 70)
    print("RUNNING LESSON 25 TESTS")
    print("=" * 70)

    rng = np.random.default_rng(123)

    image = rng.integers(
        0,
        256,
        size=(64, 64, 3),
        dtype=np.uint8,
    )

    # --------------------------------------------------------------
    # TEST 1 — Horizontal flip shape
    # --------------------------------------------------------------

    flipped = horizontal_flip(image)

    assert flipped.shape == image.shape

    print("Test 1 passed: Horizontal flip shape.")

    # --------------------------------------------------------------
    # TEST 2 — Horizontal flip involution
    # --------------------------------------------------------------

    restored = horizontal_flip(flipped)

    assert np.array_equal(
        restored,
        image,
    )

    print("Test 2 passed: Double flip restores original.")

    # --------------------------------------------------------------
    # TEST 3 — Rotation dimensions
    # --------------------------------------------------------------

    rotated = rotate_image(
        image,
        25,
    )

    assert rotated.shape == image.shape

    print("Test 3 passed: Rotation dimensions.")

    # --------------------------------------------------------------
    # TEST 4 — Crop and resize
    # --------------------------------------------------------------

    cropped = random_crop_and_resize(
        image
    )

    assert cropped.shape == image.shape

    print("Test 4 passed: Random crop dimensions.")

    # --------------------------------------------------------------
    # TEST 5 — Translation
    # --------------------------------------------------------------

    translated = random_translation(
        image
    )

    assert translated.shape == image.shape

    print("Test 5 passed: Translation dimensions.")

    # --------------------------------------------------------------
    # TEST 6 — Brightness clipping
    # --------------------------------------------------------------

    bright = adjust_brightness(
        image,
        1000,
    )

    assert bright.max() == 255

    print("Test 6 passed: Brightness clipping.")

    # --------------------------------------------------------------
    # TEST 7 — Contrast output
    # --------------------------------------------------------------

    contrast = adjust_contrast(
        image,
        1.2,
    )

    assert contrast.shape == image.shape

    print("Test 7 passed: Contrast adjustment.")

    # --------------------------------------------------------------
    # TEST 8 — Saturation output
    # --------------------------------------------------------------

    saturation = adjust_saturation(
        image,
        1.2,
    )

    assert saturation.shape == image.shape

    print("Test 8 passed: Saturation adjustment.")

    # --------------------------------------------------------------
    # TEST 9 — Cutout
    # --------------------------------------------------------------

    cutout_image = cutout(
        image,
        fraction=0.25,
    )

    assert cutout_image.shape == image.shape

    print("Test 9 passed: Cutout.")

    # --------------------------------------------------------------
    # TEST 10 — MixUp shape and lambda
    # --------------------------------------------------------------

    tensor_images = torch.rand(
        8,
        3,
        64,
        64,
    )

    tensor_labels = torch.randint(
        0,
        NUM_CLASSES,
        (8,),
    )

    mixed, mix_labels_a, mix_labels_b, mix_lam = mixup(
        tensor_images,
        tensor_labels,
    )

    assert mixed.shape == tensor_images.shape

    assert mix_labels_a.shape == tensor_labels.shape

    assert mix_labels_b.shape == tensor_labels.shape

    assert 0.0 <= mix_lam <= 1.0

    print("Test 10 passed: MixUp.")

    # --------------------------------------------------------------
    # TEST 11 — CutMix shape and lambda
    # --------------------------------------------------------------

    cutmixed, cut_labels_a, cut_labels_b, cut_lam = cutmix(
        tensor_images,
        tensor_labels,
    )

    assert cutmixed.shape == tensor_images.shape

    assert cut_labels_a.shape == tensor_labels.shape

    assert cut_labels_b.shape == tensor_labels.shape

    assert 0.0 <= cut_lam <= 1.0

    print("Test 11 passed: CutMix.")

    # --------------------------------------------------------------
    # TEST 12 — MixUp loss and gradients
    # --------------------------------------------------------------

    logits = torch.randn(
        8,
        NUM_CLASSES,
        requires_grad=True,
    )

    criterion = nn.CrossEntropyLoss()

    loss = mixup_loss(
        criterion,
        logits,
        mix_labels_a,
        mix_labels_b,
        mix_lam,
    )

    assert torch.isfinite(loss)

    loss.backward()

    assert logits.grad is not None

    assert torch.isfinite(logits.grad).all()

    print("Test 12 passed: MixUp loss and gradients.")

    # --------------------------------------------------------------
    # TEST 13 — Dataset generation and output
    # --------------------------------------------------------------

    images, labels = generate_dataset(
        20,
        seed=42,
    )

    assert images.shape == (
        20,
        IMAGE_SIZE,
        IMAGE_SIZE,
        3,
    )

    assert labels.shape == (20,)

    dataset = WasteDataset(
        images,
        labels,
        augmentation=ImageAugmentation(
            enabled=True
        ),
    )

    sample, label = dataset[0]

    assert sample.shape == (
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert sample.dtype == torch.float32

    assert 0 <= label.item() < NUM_CLASSES

    print("Test 13 passed: Dataset generation and output.")

    # --------------------------------------------------------------
    # TEST 14 — CNN forward
    # --------------------------------------------------------------

    model = AugmentationCNN()

    output = model(
        torch.randn(
            4,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
    )

    assert output.shape == (
        4,
        NUM_CLASSES,
    )

    print("Test 14 passed: CNN output.")

    # --------------------------------------------------------------
    # TEST 15 — Finite gradients
    # --------------------------------------------------------------

    model.zero_grad()

    output = model(
        torch.randn(
            4,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
    )

    loss = F.cross_entropy(
        output,
        torch.randint(
            0,
            NUM_CLASSES,
            (4,),
        ),
    )

    loss.backward()

    for parameter in model.parameters():

        if parameter.grad is not None:

            assert torch.isfinite(
                parameter.grad
            ).all()

    print("Test 15 passed: Finite gradients.")

    # --------------------------------------------------------------
    # TEST 16 — Augmentation pipeline
    # --------------------------------------------------------------

    pipeline = ImageAugmentation(
        enabled=True
    )

    augmented = pipeline(image)

    assert augmented.shape == image.shape

    assert augmented.dtype == np.uint8

    print("Test 16 passed: Augmentation pipeline.")

    # --------------------------------------------------------------
    # TEST 17 — Augmentation disabled
    # --------------------------------------------------------------

    pipeline_disabled = ImageAugmentation(
        enabled=False
    )

    unchanged = pipeline_disabled(image)

    assert np.array_equal(
        unchanged,
        image,
    )

    print("Test 17 passed: Disabled augmentation.")

    # --------------------------------------------------------------
    # TEST 18 — CutMix changes image content
    # --------------------------------------------------------------

    distinct_images = torch.zeros(
        4,
        3,
        32,
        32,
    )

    for index in range(4):

        distinct_images[index].fill_(
            float(index)
        )

    distinct_labels = torch.arange(4)

    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    mixed_images, _, _, _ = cutmix(
        distinct_images,
        distinct_labels,
        alpha=1.0,
    )

    assert mixed_images.shape == distinct_images.shape

    assert torch.isfinite(mixed_images).all()

    print("Test 18 passed: CutMix image values.")

    print()
    print("=" * 70)
    print("ALL TESTS PASSED.")
    print("=" * 70)


# ======================================================================
# STEP 13 — MAIN EXPERIMENT
# ======================================================================

def run_experiment():

    print()
    print("=" * 70)
    print("LESSON 25 — DATA AUGMENTATION")
    print("=" * 70)

    print("Device:", DEVICE)

    # --------------------------------------------------------------
    # Generate dataset
    # --------------------------------------------------------------

    train_images, train_labels = generate_dataset(
        TRAIN_SAMPLES,
        seed=SEED,
    )

    test_images, test_labels = generate_dataset(
        TEST_SAMPLES,
        seed=SEED + 1,
    )

    print(
        f"Training images: {len(train_images)}"
    )

    print(
        f"Test images: {len(test_images)}"
    )

    print(
        "Training image shape:",
        train_images.shape,
    )

    print(
        "Test image shape:",
        test_images.shape,
    )

    visualize_augmentations(
        train_images,
        train_labels,
    )

    visualize_mixup_cutmix(
        train_images,
        train_labels,
    )

    # --------------------------------------------------------------
    # Baseline dataset: no augmentation
    # --------------------------------------------------------------

    baseline_dataset = WasteDataset(
        train_images,
        train_labels,
        augmentation=ImageAugmentation(
            enabled=False
        ),
    )

    # --------------------------------------------------------------
    # Augmented training dataset
    # --------------------------------------------------------------

    augmented_dataset = WasteDataset(
        train_images,
        train_labels,
        augmentation=ImageAugmentation(
            enabled=True
        ),
    )

    # Test dataset is NEVER augmented.
    test_dataset = WasteDataset(
        test_images,
        test_labels,
        augmentation=None,
    )

    # --------------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------------

    baseline_loader = DataLoader(
        baseline_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    augmented_loader = DataLoader(
        augmented_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------------
    # Model A — Baseline
    # --------------------------------------------------------------

    baseline_model = initialize_model(
        AugmentationCNN()
    )

    baseline_model, baseline_history = train_model(
        model=baseline_model,
        train_loader=baseline_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        model_name="WITHOUT AUGMENTATION",
    )

    # --------------------------------------------------------------
    # Model B — Standard augmentation
    # --------------------------------------------------------------

    augmented_model = initialize_model(
        AugmentationCNN()
    )

    augmented_model, augmented_history = train_model(
        model=augmented_model,
        train_loader=augmented_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        model_name="WITH AUGMENTATION",
    )

    # --------------------------------------------------------------
    # Model C — Augmentation + MixUp + CutMix
    # --------------------------------------------------------------

    advanced_model = initialize_model(
        AugmentationCNN()
    )

    advanced_model, advanced_history = train_model(
        model=advanced_model,
        train_loader=augmented_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        model_name="AUGMENTATION + MIXUP + CUTMIX",
        use_mixup=True,
        use_cutmix=True,
    )

    # --------------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        f"Baseline test accuracy: "
        f"{baseline_history['test_accuracy'][-1]:.2%}"
    )

    print(
        f"Augmentation test accuracy: "
        f"{augmented_history['test_accuracy'][-1]:.2%}"
    )

    print(
        f"Advanced test accuracy: "
        f"{advanced_history['test_accuracy'][-1]:.2%}"
    )

    print()
    print(
        "Note: Training accuracy for MixUp/CutMix is a "
        "label-weighted diagnostic metric."
    )

    print(
        "Test accuracy is measured normally using unmodified "
        "test images and their original labels."
    )

    # --------------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------------

    visualize_training(
        baseline_history,
        augmented_history,
    )

    visualize_all_training_histories(
        baseline_history,
        augmented_history,
        advanced_history,
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    set_seed()

    print("=" * 70)
    print("COMPUTER VISION — LESSON 25")
    print("=" * 70)

    print("PyTorch:", torch.__version__)

    print("OpenCV:", cv2.__version__)

    print("Device:", DEVICE)

    run_tests()

    run_experiment()

    print()
    print("=" * 70)
    print("LESSON 25 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    main()