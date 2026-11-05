"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 2 — DEEP LEARNING FOR COMPUTER VISION
LESSON 21 — IMAGE CLASSIFICATION FROM SCRATCH
======================================================================

COURSE RULE
-----------
No pretrained models.
No pretrained weights.
No pretrained feature extractors.

This lesson implements a complete image classification pipeline
using NumPy.

OpenCV is used only to generate synthetic images and visualize results.

NEURAL NETWORK:
    Input layer
        |
    Fully Connected Layer
        |
    ReLU
        |
    Fully Connected Layer
        |
    Softmax
        |
    Classification

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1 — IMAGE CLASSIFICATION
    Understand the difference between image classification,
    object detection, and image segmentation.

STEP 2 — DATASET GENERATION
    Generate a synthetic image dataset containing circles,
    squares, and triangles.

STEP 3 — DATA PREPROCESSING
    Convert images into normalized numerical tensors.

STEP 4 — TRAIN / VALIDATION / TEST SPLIT
    Separate data to measure generalization.

STEP 5 — FORWARD PROPAGATION
    Calculate predictions through a neural network.

STEP 6 — SOFTMAX
    Convert logits into class probabilities.

STEP 7 — CROSS-ENTROPY LOSS
    Measure the difference between predictions and labels.

STEP 8 — BACKPROPAGATION
    Calculate gradients using the chain rule.

STEP 9 — ADAM OPTIMIZER
    Update model parameters using adaptive gradient estimates.

STEP 10 — EVALUATION
    Calculate accuracy, precision, recall, and confusion matrix.

STEP 11 — VISUALIZATION
    Plot training loss, accuracy, predictions, and errors.

STEP 12 — GENERALIZATION
    Test the model on images it has never seen during training.

======================================================================
MATHEMATICAL FOUNDATION
======================================================================

1. IMAGE REPRESENTATION
-----------------------

A grayscale image of size 32 x 32 is represented by:

    X ∈ R^(32 x 32)

Flatten it into a vector:

    x ∈ R^1024

Normalize pixel values:

    x_normalized = x / 255

A batch of N images becomes:

    X ∈ R^(N x 1024)

2. FULLY CONNECTED LAYER
------------------------

For input X:

    Z = XW + b

where:

    X = input matrix
    W = weight matrix
    b = bias vector

3. RELU
-------

    ReLU(x) = max(0, x)

Derivative:

    ReLU'(x) = 1 if x > 0
               0 otherwise

4. SOFTMAX
----------

Given logits z:

    P(class=i) = exp(z_i) / sum(exp(z_j))

For numerical stability:

    softmax(z) = exp(z - max(z)) / sum(exp(z - max(z)))

5. CROSS-ENTROPY LOSS
---------------------

For one-hot target y and prediction p:

    L = -sum(y_i * log(p_i))

For a batch:

    L = -mean(sum(y * log(p)))

6. BACKPROPAGATION
------------------

For softmax combined with cross-entropy:

    dL/dlogits = (probabilities - targets) / batch_size

For the output layer:

    dW2 = A1.T @ dZ2
    db2 = sum(dZ2)

For the hidden layer:

    dA1 = dZ2 @ W2.T

    dZ1 = dA1 * ReLU'(Z1)

    dW1 = X.T @ dZ1
    db1 = sum(dZ1)

7. ADAM OPTIMIZER
-----------------

First moment:

    m_t = beta1*m_(t-1) + (1-beta1)*g_t

Second moment:

    v_t = beta2*v_(t-1) + (1-beta2)*g_t^2

Bias correction:

    m_hat = m_t / (1-beta1^t)

    v_hat = v_t / (1-beta2^t)

Parameter update:

    theta = theta - learning_rate *
            m_hat / (sqrt(v_hat) + epsilon)

======================================================================
IMPORTANT LIMITATION
======================================================================

This lesson uses a fully connected network.

It does NOT yet use convolution.

A fully connected network treats all flattened pixels as
independent input features.

It does not naturally exploit local image structure.

This is one reason CNNs are important.

In Lesson 22, we will implement a convolutional neural network
and compare its behavior with this baseline.

======================================================================
REAL-WORLD ROBOTICS APPLICATION
======================================================================

OPENARM V2 — WASTE CLASSIFICATION

A simple classifier could distinguish:

    0: Plastic cup
    1: Bottle
    2: Paper waste

Pipeline:

    Camera image
         |
    Image preprocessing
         |
    Neural network
         |
    Class probabilities
         |
    Waste category
         |
    Sorting decision

For a real waste-sorting system, a classifier alone is not enough
when multiple objects appear in one image.

That requires object detection or segmentation.

This lesson establishes the neural-network foundation.

======================================================================
DEPENDENCIES
======================================================================

    pip install numpy matplotlib opencv-python

======================================================================
RUN
======================================================================

    python lesson21_image_classification_from_scratch.py

======================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass


# =====================================================================
# CONFIGURATION
# =====================================================================

SEED = 42

IMAGE_SIZE = 32
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE

NUM_CLASSES = 3

CLASS_NAMES = [
    "Circle",
    "Square",
    "Triangle"
]

HIDDEN_DIM = 128

NUM_SAMPLES = 1500

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 64
EPOCHS = 25

LEARNING_RATE = 0.001

BETA1 = 0.9
BETA2 = 0.999
ADAM_EPSILON = 1e-8

WEIGHT_DECAY = 1e-5


# =====================================================================
# STEP 1 — SYNTHETIC DATASET GENERATION
# =====================================================================

def draw_circle(image, rng):

    center = (
        int(rng.integers(10, 22)),
        int(rng.integers(10, 22))
    )

    radius = int(
        rng.integers(5, 10)
    )

    cv2.circle(
        image,
        center,
        radius,
        255,
        -1
    )


def draw_square(image, rng):

    size = int(
        rng.integers(10, 19)
    )

    x = int(
        rng.integers(3, IMAGE_SIZE - size - 2)
    )

    y = int(
        rng.integers(3, IMAGE_SIZE - size - 2)
    )

    cv2.rectangle(
        image,
        (x, y),
        (x + size, y + size),
        255,
        -1
    )


def draw_triangle(image, rng):

    center_x = int(
        rng.integers(11, 21)
    )

    top_y = int(
        rng.integers(3, 10)
    )

    half_width = int(
        rng.integers(6, 10)
    )

    bottom_y = int(
        min(top_y + rng.integers(13, 20), IMAGE_SIZE - 2)
    )

    points = np.array([
        [center_x, top_y],
        [center_x - half_width, bottom_y],
        [center_x + half_width, bottom_y]
    ], dtype=np.int32)

    cv2.fillPoly(
        image,
        [points],
        255
    )


def generate_single_image(class_id, rng):

    image = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    if class_id == 0:

        draw_circle(image, rng)

    elif class_id == 1:

        draw_square(image, rng)

    elif class_id == 2:

        draw_triangle(image, rng)

    else:

        raise ValueError("Unknown class ID.")

    # Random small translation.
    shift_x = int(
        rng.integers(-2, 3)
    )

    shift_y = int(
        rng.integers(-2, 3)
    )

    translation = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    image = cv2.warpAffine(
        image,
        translation,
        (IMAGE_SIZE, IMAGE_SIZE),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Random image intensity.
    contrast = rng.uniform(0.75, 1.15)

    image = np.clip(
        image.astype(np.float32) * contrast,
        0,
        255
    )

    # Add Gaussian noise.
    noise = rng.normal(
        0,
        12,
        image.shape
    )

    image = np.clip(
        image + noise,
        0,
        255
    )

    return image.astype(np.uint8)


def generate_dataset(
    num_samples=NUM_SAMPLES,
    seed=SEED
):

    rng = np.random.default_rng(seed)

    images = []

    labels = []

    for index in range(num_samples):

        class_id = index % NUM_CLASSES

        image = generate_single_image(
            class_id,
            rng
        )

        images.append(image)

        labels.append(class_id)

    images = np.asarray(
        images,
        dtype=np.uint8
    )

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    # Shuffle dataset.
    indices = rng.permutation(
        len(images)
    )

    images = images[indices]

    labels = labels[indices]

    return images, labels


# =====================================================================
# STEP 2 — PREPROCESSING
# =====================================================================

def preprocess_images(images):

    """
    Convert:

        N x H x W

    into:

        N x (H*W)

    and normalize pixel values into [0,1].
    """

    images = images.astype(
        np.float32
    )

    images = images / 255.0

    images = images.reshape(
        len(images),
        -1
    )

    return images


def one_hot_encode(labels, num_classes):

    one_hot = np.zeros(
        (len(labels), num_classes),
        dtype=np.float32
    )

    one_hot[
        np.arange(len(labels)),
        labels
    ] = 1.0

    return one_hot


# =====================================================================
# STEP 3 — TRAIN / VALIDATION / TEST SPLIT
# =====================================================================

@dataclass
class DatasetSplit:

    X_train: np.ndarray
    y_train: np.ndarray

    X_validation: np.ndarray
    y_validation: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray


def split_dataset(
    X,
    y,
    train_ratio=0.70,
    validation_ratio=0.15,
    seed=SEED
):

    if not np.isclose(
        train_ratio + validation_ratio + TEST_RATIO,
        1.0
    ):

        raise ValueError(
            "Dataset split ratios must sum to 1."
        )

    rng = np.random.default_rng(seed)

    indices = rng.permutation(
        len(X)
    )

    X = X[indices]

    y = y[indices]

    n = len(X)

    train_end = int(
        n * train_ratio
    )

    validation_end = train_end + int(
        n * validation_ratio
    )

    return DatasetSplit(
        X_train=X[:train_end],
        y_train=y[:train_end],

        X_validation=X[train_end:validation_end],
        y_validation=y[train_end:validation_end],

        X_test=X[validation_end:],
        y_test=y[validation_end:]
    )


# =====================================================================
# STEP 4 — ACTIVATION FUNCTIONS
# =====================================================================

def relu(x):

    return np.maximum(
        0.0,
        x
    )


def relu_derivative(x):

    return (
        x > 0
    ).astype(np.float32)


def softmax(logits):

    """
    Numerically stable softmax.
    """

    shifted = (
        logits -
        np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    exponentials = np.exp(
        shifted
    )

    probabilities = exponentials / np.sum(
        exponentials,
        axis=1,
        keepdims=True
    )

    return probabilities


# =====================================================================
# STEP 5 — CROSS-ENTROPY LOSS
# =====================================================================

def cross_entropy_loss(
    probabilities,
    labels
):

    batch_size = len(labels)

    selected_probabilities = probabilities[
        np.arange(batch_size),
        labels
    ]

    selected_probabilities = np.clip(
        selected_probabilities,
        1e-12,
        1.0
    )

    return -np.mean(
        np.log(selected_probabilities)
    )


# =====================================================================
# STEP 6 — ACCURACY
# =====================================================================

def accuracy_score(
    probabilities,
    labels
):

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    return np.mean(
        predictions == labels
    )


# =====================================================================
# STEP 7 — NEURAL NETWORK
# =====================================================================

class MLPClassifier:

    """
    Two-layer fully connected neural network.

        Input
          |
        Dense
          |
        ReLU
          |
        Dense
          |
        Softmax

    All parameters are randomly initialized.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        seed=SEED
    ):

        rng = np.random.default_rng(
            seed
        )

        # He initialization for ReLU layer.
        self.W1 = (
            rng.standard_normal(
                (input_dim, hidden_dim)
            ) *
            np.sqrt(2.0 / input_dim)
        ).astype(np.float32)

        self.b1 = np.zeros(
            (1, hidden_dim),
            dtype=np.float32
        )

        # Xavier-style initialization for output layer.
        self.W2 = (
            rng.standard_normal(
                (hidden_dim, num_classes)
            ) *
            np.sqrt(1.0 / hidden_dim)
        ).astype(np.float32)

        self.b2 = np.zeros(
            (1, num_classes),
            dtype=np.float32
        )

        self.parameters = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2
        }

        self.adam_m = {}

        self.adam_v = {}

        for name, parameter in self.parameters.items():

            self.adam_m[name] = np.zeros_like(
                parameter
            )

            self.adam_v[name] = np.zeros_like(
                parameter
            )

        self.optimizer_step = 0

    # -----------------------------------------------------------------
    # FORWARD PROPAGATION
    # -----------------------------------------------------------------

    def forward(self, X):

        Z1 = X @ self.W1 + self.b1

        A1 = relu(Z1)

        Z2 = A1 @ self.W2 + self.b2

        probabilities = softmax(Z2)

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "probabilities": probabilities
        }

        return probabilities, cache

    # -----------------------------------------------------------------
    # BACKPROPAGATION
    # -----------------------------------------------------------------

    def backward(
        self,
        cache,
        labels
    ):

        X = cache["X"]

        Z1 = cache["Z1"]

        A1 = cache["A1"]

        probabilities = cache["probabilities"]

        batch_size = len(X)

        targets = one_hot_encode(
            labels,
            probabilities.shape[1]
        )

        # Softmax + cross-entropy derivative.
        dZ2 = (
            probabilities - targets
        ) / batch_size

        dW2 = A1.T @ dZ2

        db2 = np.sum(
            dZ2,
            axis=0,
            keepdims=True
        )

        dA1 = dZ2 @ self.W2.T

        dZ1 = (
            dA1 *
            relu_derivative(Z1)
        )

        dW1 = X.T @ dZ1

        db1 = np.sum(
            dZ1,
            axis=0,
            keepdims=True
        )

        gradients = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2
        }

        return gradients

    # -----------------------------------------------------------------
    # ADAM OPTIMIZER
    # -----------------------------------------------------------------

    def update_parameters(
        self,
        gradients,
        learning_rate=LEARNING_RATE
    ):

        self.optimizer_step += 1

        t = self.optimizer_step

        for name in self.parameters:

            parameter = self.parameters[name]

            gradient = gradients[name]

            # Weight decay only on weight matrices.
            if name.startswith("W"):

                gradient = (
                    gradient +
                    WEIGHT_DECAY * parameter
                )

            self.adam_m[name] = (
                BETA1 * self.adam_m[name] +
                (1.0 - BETA1) * gradient
            )

            self.adam_v[name] = (
                BETA2 * self.adam_v[name] +
                (1.0 - BETA2) * gradient ** 2
            )

            m_hat = (
                self.adam_m[name] /
                (1.0 - BETA1 ** t)
            )

            v_hat = (
                self.adam_v[name] /
                (1.0 - BETA2 ** t)
            )

            parameter -= (
                learning_rate *
                m_hat /
                (
                    np.sqrt(v_hat) +
                    ADAM_EPSILON
                )
            )

    # -----------------------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------------------

    def predict_proba(self, X):

        probabilities, _ = self.forward(X)

        return probabilities

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return np.argmax(
            probabilities,
            axis=1
        )


# =====================================================================
# STEP 8 — TRAINING
# =====================================================================

@dataclass
class TrainingHistory:

    train_loss: list

    validation_loss: list

    train_accuracy: list

    validation_accuracy: list


def evaluate_model(model, X, y):

    probabilities = model.predict_proba(X)

    loss = cross_entropy_loss(
        probabilities,
        y
    )

    accuracy = accuracy_score(
        probabilities,
        y
    )

    return loss, accuracy


def train_model(
    model,
    dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
):

    history = TrainingHistory(
        train_loss=[],
        validation_loss=[],
        train_accuracy=[],
        validation_accuracy=[]
    )

    rng = np.random.default_rng(
        SEED + 10
    )

    print("\nSTARTING TRAINING")

    print("=" * 70)

    for epoch in range(epochs):

        indices = rng.permutation(
            len(dataset.X_train)
        )

        X_train = dataset.X_train[indices]

        y_train = dataset.y_train[indices]

        for start in range(
            0,
            len(X_train),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(X_train)
            )

            X_batch = X_train[start:end]

            y_batch = y_train[start:end]

            probabilities, cache = model.forward(
                X_batch
            )

            gradients = model.backward(
                cache,
                y_batch
            )

            model.update_parameters(
                gradients
            )

        # Evaluate after each epoch.
        train_loss, train_accuracy = evaluate_model(
            model,
            dataset.X_train,
            dataset.y_train
        )

        validation_loss, validation_accuracy = evaluate_model(
            model,
            dataset.X_validation,
            dataset.y_validation
        )

        history.train_loss.append(
            train_loss
        )

        history.validation_loss.append(
            validation_loss
        )

        history.train_accuracy.append(
            train_accuracy
        )

        history.validation_accuracy.append(
            validation_accuracy
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f}"
        )

    return history


# =====================================================================
# STEP 9 — CONFUSION MATRIX
# =====================================================================

def confusion_matrix(
    true_labels,
    predicted_labels,
    num_classes
):

    matrix = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64
    )

    for true, predicted in zip(
        true_labels,
        predicted_labels
    ):

        matrix[
            true,
            predicted
        ] += 1

    return matrix


def classification_metrics(matrix):

    metrics = {}

    for class_id in range(
        matrix.shape[0]
    ):

        true_positive = matrix[
            class_id,
            class_id
        ]

        false_positive = (
            np.sum(matrix[:, class_id]) -
            true_positive
        )

        false_negative = (
            np.sum(matrix[class_id, :]) -
            true_positive
        )

        precision = (
            true_positive /
            max(
                true_positive + false_positive,
                1
            )
        )

        recall = (
            true_positive /
            max(
                true_positive + false_negative,
                1
            )
        )

        f1 = (
            2 * precision * recall /
            max(
                precision + recall,
                1e-12
            )
        )

        metrics[CLASS_NAMES[class_id]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return metrics


# =====================================================================
# STEP 10 — GRADIENT CHECK
# =====================================================================

def numerical_gradient_check():

    """
    Verify analytical gradients against finite differences.

    This test checks the derivative of one selected parameter.

    Numerical gradient:

        dL/dw approximately equals
        [L(w+epsilon) - L(w-epsilon)] / (2*epsilon)
    """

    print("\nGRADIENT CHECK")

    rng = np.random.default_rng(123)

    X = rng.normal(
        0,
        0.2,
        size=(4, 5)
    ).astype(np.float32)

    y = np.array(
        [0, 1, 2, 1]
    )

    model = MLPClassifier(
        input_dim=5,
        hidden_dim=6,
        num_classes=3,
        seed=123
    )

    probabilities, cache = model.forward(X)

    gradients = model.backward(
        cache,
        y
    )

    analytical = gradients["W2"][0, 0]

    epsilon = 1e-3

    original = model.W2[0, 0]

    model.W2[0, 0] = original + epsilon

    plus_probabilities = model.predict_proba(X)

    loss_plus = cross_entropy_loss(
        plus_probabilities,
        y
    )

    model.W2[0, 0] = original - epsilon

    minus_probabilities = model.predict_proba(X)

    loss_minus = cross_entropy_loss(
        minus_probabilities,
        y
    )

    model.W2[0, 0] = original

    numerical = (
        loss_plus - loss_minus
    ) / (
        2 * epsilon
    )

    relative_error = abs(
        analytical - numerical
    ) / max(
        abs(analytical) + abs(numerical),
        1e-8
    )

    print(
        f"Analytical gradient: {analytical:.8f}"
    )

    print(
        f"Numerical gradient:  {numerical:.8f}"
    )

    print(
        f"Relative error:      {relative_error:.8e}"
    )

    assert relative_error < 1e-2

    print("Gradient check passed.")


# =====================================================================
# STEP 11 — NUMERICAL TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print("RUNNING LESSON 21 TESTS")

    print("=" * 70)

    # ---------------------------------------------------------------
    # TEST 1 — Dataset generation
    # ---------------------------------------------------------------

    images, labels = generate_dataset(
        num_samples=90,
        seed=123
    )

    assert images.shape == (
        90,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    assert labels.shape == (90,)

    assert set(labels) == {0, 1, 2}

    print("Test 1 passed: synthetic image dataset.")

    # ---------------------------------------------------------------
    # TEST 2 — Preprocessing
    # ---------------------------------------------------------------

    X = preprocess_images(images)

    assert X.shape == (
        90,
        INPUT_DIM
    )

    assert np.min(X) >= 0

    assert np.max(X) <= 1

    print("Test 2 passed: image preprocessing.")

    # ---------------------------------------------------------------
    # TEST 3 — One-hot encoding
    # ---------------------------------------------------------------

    encoded = one_hot_encode(
        labels,
        NUM_CLASSES
    )

    assert encoded.shape == (
        90,
        NUM_CLASSES
    )

    assert np.allclose(
        np.sum(encoded, axis=1),
        1.0
    )

    print("Test 3 passed: one-hot encoding.")

    # ---------------------------------------------------------------
    # TEST 4 — Softmax
    # ---------------------------------------------------------------

    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, 1.0]
    ])

    probabilities = softmax(logits)

    assert np.allclose(
        np.sum(probabilities, axis=1),
        1.0
    )

    assert np.all(
        probabilities >= 0
    )

    print("Test 4 passed: softmax normalization.")

    # ---------------------------------------------------------------
    # TEST 5 — Cross-entropy
    # ---------------------------------------------------------------

    probabilities = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2]
    ])

    labels_test = np.array([
        0,
        1
    ])

    loss = cross_entropy_loss(
        probabilities,
        labels_test
    )

    assert loss > 0

    assert np.isfinite(loss)

    print("Test 5 passed: cross-entropy loss.")

    # ---------------------------------------------------------------
    # TEST 6 — Network output
    # ---------------------------------------------------------------

    model = MLPClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    )

    output = model.predict_proba(
        X[:8]
    )

    assert output.shape == (
        8,
        NUM_CLASSES
    )

    assert np.allclose(
        output.sum(axis=1),
        1.0,
        atol=1e-6
    )

    print("Test 6 passed: neural network forward pass.")

    # ---------------------------------------------------------------
    # TEST 7 — Backpropagation
    # ---------------------------------------------------------------

    _, cache = model.forward(
        X[:8]
    )

    gradients = model.backward(
        cache,
        labels[:8]
    )

    for gradient in gradients.values():

        assert np.all(
            np.isfinite(gradient)
        )

    print("Test 7 passed: backpropagation.")

    # ---------------------------------------------------------------
    # TEST 8 — Parameter update
    # ---------------------------------------------------------------

    original = model.W1.copy()

    model.update_parameters(
        gradients
    )

    assert not np.allclose(
        original,
        model.W1
    )

    print("Test 8 passed: Adam parameter update.")

    # ---------------------------------------------------------------
    # TEST 9 — Confusion matrix
    # ---------------------------------------------------------------

    true = np.array([
        0, 0, 1, 1, 2, 2
    ])

    predicted = np.array([
        0, 1, 1, 1, 2, 0
    ])

    matrix = confusion_matrix(
        true,
        predicted,
        3
    )

    assert matrix.shape == (3, 3)

    assert matrix[0, 0] == 1

    assert matrix[1, 1] == 2

    assert matrix[2, 2] == 1

    print("Test 9 passed: confusion matrix.")

    # ---------------------------------------------------------------
    # TEST 10 — Gradient check
    # ---------------------------------------------------------------

    numerical_gradient_check()

    print("\nALL LESSON 21 TESTS PASSED.")


# =====================================================================
# STEP 12 — VISUALIZATION
# =====================================================================

def visualize_dataset(images, labels):

    fig, axes = plt.subplots(
        3,
        6,
        figsize=(12, 7)
    )

    for index, ax in enumerate(
        axes.ravel()
    ):

        ax.imshow(
            images[index],
            cmap="gray",
            vmin=0,
            vmax=255
        )

        ax.set_title(
            CLASS_NAMES[labels[index]]
        )

        ax.axis("off")

    plt.suptitle(
        "Synthetic Image Classification Dataset"
    )

    plt.tight_layout()

    plt.show()


def visualize_training(history):

    epochs = np.arange(
        1,
        len(history.train_loss) + 1
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5)
    )

    axes[0].plot(
        epochs,
        history.train_loss,
        label="Training"
    )

    axes[0].plot(
        epochs,
        history.validation_loss,
        label="Validation"
    )

    axes[0].set_title(
        "Cross-Entropy Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].legend()

    axes[0].grid(
        True
    )

    axes[1].plot(
        epochs,
        history.train_accuracy,
        label="Training"
    )

    axes[1].plot(
        epochs,
        history.validation_accuracy,
        label="Validation"
    )

    axes[1].set_title(
        "Classification Accuracy"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Accuracy"
    )

    axes[1].set_ylim(
        0,
        1.05
    )

    axes[1].legend()

    axes[1].grid(
        True
    )

    plt.tight_layout()

    plt.show()


def visualize_confusion_matrix(matrix):

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        matrix,
        cmap="Blues"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.xticks(
        range(NUM_CLASSES),
        CLASS_NAMES
    )

    plt.yticks(
        range(NUM_CLASSES),
        CLASS_NAMES
    )

    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    plt.show()


def visualize_predictions(
    images,
    labels,
    probabilities
):

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    fig, axes = plt.subplots(
        3,
        5,
        figsize=(13, 8)
    )

    for index, ax in enumerate(
        axes.ravel()
    ):

        ax.imshow(
            images[index],
            cmap="gray",
            vmin=0,
            vmax=255
        )

        true_name = CLASS_NAMES[
            labels[index]
        ]

        predicted_name = CLASS_NAMES[
            predictions[index]
        ]

        confidence = probabilities[
            index,
            predictions[index]
        ]

        ax.set_title(
            f"True: {true_name}\n"
            f"Pred: {predicted_name}\n"
            f"Confidence: {confidence:.2f}"
        )

        ax.axis("off")

    plt.suptitle(
        "Test Set Predictions"
    )

    plt.tight_layout()

    plt.show()


# =====================================================================
# STEP 13 — REAL-WORLD IMAGE INFERENCE
# =====================================================================

def predict_external_image(
    model,
    image_path
):

    """
    Optional: classify a real image.

    The model has only learned synthetic circles, squares,
    and triangles. It is NOT trained to recognize real-world
    waste categories.

    This function demonstrates image loading and preprocessing.
    """

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise FileNotFoundError(
            f"Cannot load image: {image_path}"
        )

    resized = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    X = preprocess_images(
        resized[None, ...]
    )

    probabilities = model.predict_proba(
        X
    )[0]

    prediction = int(
        np.argmax(probabilities)
    )

    print(
        "\nEXTERNAL IMAGE INFERENCE"
    )

    print(
        f"Image: {image_path}"
    )

    print(
        f"Predicted shape: {CLASS_NAMES[prediction]}"
    )

    for index, name in enumerate(CLASS_NAMES):

        print(
            f"{name}: {probabilities[index]:.4f}"
        )


# =====================================================================
# STEP 14 — COMPLETE EXPERIMENT
# =====================================================================

def run_experiment():

    print("\n" + "=" * 70)

    print("GENERATING IMAGE DATASET")

    print("=" * 70)

    images, labels = generate_dataset()

    print(
        f"Dataset size: {len(images)}"
    )

    print(
        f"Image shape: {images.shape[1:]}"
    )

    print(
        f"Number of classes: {NUM_CLASSES}"
    )

    # ---------------------------------------------------------------
    # Visualize dataset
    # ---------------------------------------------------------------

    visualize_dataset(
        images,
        labels
    )

    # ---------------------------------------------------------------
    # Preprocess
    # ---------------------------------------------------------------

    X = preprocess_images(
        images
    )

    # ---------------------------------------------------------------
    # Split
    # ---------------------------------------------------------------

    dataset = split_dataset(
        X,
        labels
    )

    print("\nDATASET SPLIT")

    print(
        f"Training:   {len(dataset.X_train)}"
    )

    print(
        f"Validation: {len(dataset.X_validation)}"
    )

    print(
        f"Testing:    {len(dataset.X_test)}"
    )

    # ---------------------------------------------------------------
    # Initialize model
    # ---------------------------------------------------------------

    model = MLPClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES
    )

    parameter_count = sum(
        parameter.size
        for parameter in model.parameters.values()
    )

    print(
        f"\nModel parameters: {parameter_count:,}"
    )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    history = train_model(
        model,
        dataset
    )

    # ---------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------

    test_loss, test_accuracy = evaluate_model(
        model,
        dataset.X_test,
        dataset.y_test
    )

    print("\n" + "=" * 70)

    print("FINAL TEST RESULTS")

    print("=" * 70)

    print(
        f"Test loss: {test_loss:.4f}"
    )

    print(
        f"Test accuracy: {test_accuracy:.4f}"
    )

    # ---------------------------------------------------------------
    # Classification metrics
    # ---------------------------------------------------------------

    predictions = model.predict(
        dataset.X_test
    )

    matrix = confusion_matrix(
        dataset.y_test,
        predictions,
        NUM_CLASSES
    )

    metrics = classification_metrics(
        matrix
    )

    print("\nCLASSIFICATION METRICS")

    for name, values in metrics.items():

        print(f"\n{name}")

        print(
            f"Precision: {values['precision']:.4f}"
        )

        print(
            f"Recall:    {values['recall']:.4f}"
        )

        print(
            f"F1 score:  {values['f1']:.4f}"
        )

    # ---------------------------------------------------------------
    # Visualizations
    # ---------------------------------------------------------------

    visualize_training(
        history
    )

    visualize_confusion_matrix(
        matrix
    )

    test_probabilities = model.predict_proba(
        dataset.X_test
    )

    visualize_predictions(
        images[
            np.arange(len(images))[
                -len(dataset.X_test):
            ]
        ],
        dataset.y_test,
        test_probabilities
    )

    return model, dataset, history


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)

    print("COMPUTER VISION — PHASE 2 — LESSON 21")

    print("IMAGE CLASSIFICATION FROM SCRATCH")

    print("=" * 70)

    run_tests()

    model, dataset, history = run_experiment()

    print("\n" + "=" * 70)

    print("LESSON 21 COMPLETED")

    print("=" * 70)

    print(
        "\nNext lesson:"
    )

    print(
        "LESSON 22 — CONVOLUTIONAL NEURAL NETWORK FROM SCRATCH"
    )


if __name__ == "__main__":

    main()