"""
======================================================================
COMPUTER VISION FROM SCRATCH
PHASE 2 — DEEP LEARNING FOR COMPUTER VISION
LESSON 22 — CONVOLUTIONAL NEURAL NETWORK FROM SCRATCH
======================================================================

COURSE RULE
-----------
No pretrained models.
No pretrained weights.
No pretrained feature extractors.

The CNN is initialized randomly and trained from scratch.

Libraries:
    NumPy       : neural-network mathematics and training
    OpenCV      : synthetic image generation and convolution reference
    Matplotlib  : visualization

======================================================================
WHAT YOU WILL LEARN
======================================================================

STEP 1 — WHY CNN?
    Understand why convolution is more suitable for image data
    than flattening every pixel into an independent feature.

STEP 2 — CONVOLUTION
    Implement 2D convolution using NumPy.

STEP 3 — CONVOLUTION BACKPROPAGATION
    Calculate gradients for input images and convolution kernels.

STEP 4 — RELU
    Implement activation and its derivative.

STEP 5 — MAX POOLING
    Reduce spatial dimensions while retaining strong activations.

STEP 6 — MAX POOLING BACKPROPAGATION
    Send gradients to the positions that produced maximum values.

STEP 7 — FLATTEN
    Convert convolutional feature maps into vectors.

STEP 8 — FULLY CONNECTED CLASSIFIER
    Implement the classification head.

STEP 9 — SOFTMAX AND CROSS-ENTROPY
    Convert logits to probabilities and calculate classification loss.

STEP 10 — BACKPROPAGATION
    Train the complete network using gradients from every layer.

STEP 11 — ADAM
    Update parameters with adaptive gradient optimization.

STEP 12 — EVALUATION
    Evaluate accuracy and confusion matrix.

======================================================================
NETWORK ARCHITECTURE
======================================================================

Input:
    32 x 32 x 1

Conv1:
    8 filters
    3 x 3 kernel
    padding = 1
    output = 32 x 32 x 8

ReLU:

MaxPool:
    2 x 2
    stride = 2
    output = 16 x 16 x 8

Conv2:
    16 filters
    3 x 3 kernel
    padding = 1
    output = 16 x 16 x 16

ReLU:

MaxPool:
    2 x 2
    stride = 2
    output = 8 x 8 x 16

Flatten:
    1024 features

Fully Connected:
    128 neurons

ReLU:

Output:
    3 classes

Softmax:

======================================================================
MATHEMATICAL FOUNDATION
======================================================================

1. CONVOLUTION
--------------

For an input image X and kernel K:

    Y[i,j] = sum_a sum_b X[i+a,j+b] * K[a,b]

For multiple channels:

    Y[i,j,o] =
        sum_c sum_a sum_b
        X[i+a,j+b,c] * K[a,b,c,o]

where:
    c = input channel
    o = output channel

2. PADDING
----------

For a 3x3 kernel and padding 1:

    H_out = H + 2P - K + 1

    W_out = W + 2P - K + 1

The spatial dimensions remain unchanged.

3. CONVOLUTION BACKPROPAGATION
------------------------------

Given upstream gradient dY:

    dK[a,b,c,o] =
        sum_i sum_j
        X_padded[i+a,j+b,c] * dY[i,j,o]

    dX_padded[i+a,j+b,c] +=
        sum_o K[a,b,c,o] * dY[i,j,o]

4. RELU
-------

    ReLU(x) = max(0,x)

    dReLU/dx = 1 if x > 0
               0 otherwise

5. MAX POOLING
--------------

For each 2x2 window:

    Y[i,j] = max(window)

Backward propagation:
    Send the gradient to the maximum position.

6. FLATTEN
----------

    (N,H,W,C) -> (N,H*W*C)

Backward:
    Restore the original tensor shape.

7. CROSS-ENTROPY
----------------

    L = -mean(log(P_true_class))

For softmax + cross-entropy:

    dLogits = (P - Y_one_hot) / N

8. CNN TRAINING
---------------

    Forward pass
         |
    Cross-entropy loss
         |
    Backward pass
         |
    Gradients
         |
    Adam optimizer
         |
    Updated weights

======================================================================
IMPORTANT IMPLEMENTATION NOTE
======================================================================

The convolution implementation uses NumPy sliding-window views
and matrix multiplication.

This makes the forward operation substantially faster than
four nested Python loops over every pixel.

The backward pass calculates gradients explicitly.

The implementation is educational, not a replacement for
optimized production libraries such as cuDNN.

======================================================================
REAL-WORLD ROBOTICS APPLICATION
======================================================================

OPENARM V2 — WASTE CLASSIFICATION

A CNN can learn visual features such as:

    - Bottle contours
    - Cup rims
    - Paper edges
    - Surface texture
    - Object appearance

A possible system:

    Camera
       |
    RGB image
       |
    CNN feature extractor
       |
    Classification head
       |
    Waste category

However, if several waste objects appear in one frame,
classification alone is insufficient.

Object detection or segmentation will be needed later.

======================================================================
RUN
======================================================================

    python lesson22_cnn_from_scratch.py

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

NUM_CLASSES = 3

CLASS_NAMES = [
    "Circle",
    "Square",
    "Triangle"
]

NUM_SAMPLES = 1200

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

EPOCHS = 12

BATCH_SIZE = 32

LEARNING_RATE = 0.001

BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8

WEIGHT_DECAY = 1e-5


# =====================================================================
# STEP 1 — SYNTHETIC DATASET
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

    bottom_y = min(
        top_y + int(rng.integers(13, 20)),
        IMAGE_SIZE - 2
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

    else:

        draw_triangle(image, rng)

    shift_x = int(
        rng.integers(-2, 3)
    )

    shift_y = int(
        rng.integers(-2, 3)
    )

    matrix = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    image = cv2.warpAffine(
        image,
        matrix,
        (IMAGE_SIZE, IMAGE_SIZE),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    contrast = rng.uniform(
        0.75,
        1.15
    )

    image = image.astype(
        np.float32
    ) * contrast

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

    return image.astype(
        np.uint8
    )


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
        dtype=np.float32
    )

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    indices = rng.permutation(
        num_samples
    )

    images = images[indices]

    labels = labels[indices]

    # N,H,W -> N,H,W,C
    images = images[..., None]

    # Normalize to [0,1].
    images /= 255.0

    return images, labels


# =====================================================================
# STEP 2 — CONVOLUTION
# =====================================================================

def im2col(x, kernel_size, padding=0):

    """
    Convert image neighborhoods into a matrix.

    Input:
        x: N,H,W,C

    Output:
        columns: N*H_out*W_out, K*K*C

    This is a vectorized implementation of sliding image windows.
    """

    n, h, w, c = x.shape

    k = kernel_size

    if padding > 0:

        x = np.pad(
            x,
            (
                (0, 0),
                (padding, padding),
                (padding, padding),
                (0, 0)
            ),
            mode="constant"
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        x,
        (k, k),
        axis=(1, 2)
    )

    # Shape:
    # N,H_out,W_out,C,K,K

    windows = windows.transpose(
        0, 1, 2, 4, 5, 3
    )

    columns = windows.reshape(
        -1,
        k * k * c
    )

    return columns, windows.shape


def conv2d_forward(
    x,
    weights,
    bias,
    padding=1
):

    """
    x:
        N,H,W,C_in

    weights:
        K,K,C_in,C_out

    bias:
        C_out

    output:
        N,H_out,W_out,C_out
    """

    n, h, w, c_in = x.shape

    k, _, _, c_out = weights.shape

    columns, window_shape = im2col(
        x,
        k,
        padding
    )

    weight_matrix = weights.reshape(
        -1,
        c_out
    )

    output_matrix = (
        columns @ weight_matrix +
        bias
    )

    h_out = window_shape[1]

    w_out = window_shape[2]

    output = output_matrix.reshape(
        n,
        h_out,
        w_out,
        c_out
    )

    cache = {
        "x": x,
        "weights": weights,
        "columns": columns,
        "window_shape": window_shape,
        "padding": padding
    }

    return output, cache


# =====================================================================
# STEP 3 — CONVOLUTION BACKPROPAGATION
# =====================================================================

def conv2d_backward(
    dout,
    cache
):

    x = cache["x"]

    weights = cache["weights"]

    columns = cache["columns"]

    window_shape = cache["window_shape"]

    padding = cache["padding"]

    n, h, w, c_in = x.shape

    k, _, _, c_out = weights.shape

    h_out = window_shape[1]

    w_out = window_shape[2]

    dout_matrix = dout.reshape(
        -1,
        c_out
    )

    # Gradient of weights.
    dw = (
        columns.T @ dout_matrix
    ).reshape(
        weights.shape
    )

    # Gradient of bias.
    db = np.sum(
        dout_matrix,
        axis=0
    )

    # Gradient of im2col matrix.
    dcolumns = (
        dout_matrix @
        weights.reshape(-1, c_out).T
    )

    dcolumns = dcolumns.reshape(
        n,
        h_out,
        w_out,
        k,
        k,
        c_in
    )

    # Accumulate overlapping window gradients.
    dx_padded = np.zeros(
        (
            n,
            h + 2 * padding,
            w + 2 * padding,
            c_in
        ),
        dtype=np.float32
    )

    for kernel_y in range(k):

        for kernel_x in range(k):

            dx_padded[
                :,
                kernel_y:kernel_y + h_out,
                kernel_x:kernel_x + w_out,
                :
            ] += dcolumns[
                :,
                :,
                :,
                kernel_y,
                kernel_x,
                :
            ]

    if padding > 0:

        dx = dx_padded[
            :,
            padding:-padding,
            padding:-padding,
            :
        ]

    else:

        dx = dx_padded

    return dx, dw, db


# =====================================================================
# STEP 4 — ACTIVATION
# =====================================================================

def relu_forward(x):

    output = np.maximum(
        x,
        0
    )

    cache = x

    return output, cache


def relu_backward(dout, cache):

    return dout * (
        cache > 0
    )


# =====================================================================
# STEP 5 — MAX POOLING
# =====================================================================

def max_pool_forward(
    x,
    pool_size=2,
    stride=2
):

    """
    Input:
        N,H,W,C

    Output:
        N,H_out,W_out,C

    Only non-overlapping windows are used in this lesson.
    """

    n, h, w, c = x.shape

    p = pool_size

    h_out = (
        h - p
    ) // stride + 1

    w_out = (
        w - p
    ) // stride + 1

    output = np.zeros(
        (n, h_out, w_out, c),
        dtype=np.float32
    )

    max_indices = np.zeros(
        (n, h_out, w_out, c),
        dtype=np.int64
    )

    for y in range(h_out):

        for x_pos in range(w_out):

            patch = x[
                :,
                y * stride:y * stride + p,
                x_pos * stride:x_pos * stride + p,
                :
            ]

            flattened = patch.reshape(
                n,
                p * p,
                c
            )

            indices = np.argmax(
                flattened,
                axis=1
            )

            maxima = np.max(
                flattened,
                axis=1
            )

            output[:, y, x_pos, :] = maxima

            max_indices[:, y, x_pos, :] = indices

    cache = {
        "input_shape": x.shape,
        "max_indices": max_indices,
        "pool_size": pool_size,
        "stride": stride
    }

    return output, cache


# =====================================================================
# STEP 6 — MAX POOLING BACKPROPAGATION
# =====================================================================

def max_pool_backward(
    dout,
    cache
):

    input_shape = cache["input_shape"]

    max_indices = cache["max_indices"]

    pool_size = cache["pool_size"]

    stride = cache["stride"]

    n, h, w, c = input_shape

    h_out = dout.shape[1]

    w_out = dout.shape[2]

    dx = np.zeros(
        input_shape,
        dtype=np.float32
    )

    for y in range(h_out):

        for x_pos in range(w_out):

            indices = max_indices[
                :,
                y,
                x_pos,
                :
            ]

            for sample in range(n):

                for channel in range(c):

                    index = indices[
                        sample,
                        channel
                    ]

                    offset_y = index // pool_size

                    offset_x = index % pool_size

                    target_y = (
                        y * stride + offset_y
                    )

                    target_x = (
                        x_pos * stride + offset_x
                    )

                    dx[
                        sample,
                        target_y,
                        target_x,
                        channel
                    ] += dout[
                        sample,
                        y,
                        x_pos,
                        channel
                    ]

    return dx


# =====================================================================
# STEP 7 — SOFTMAX AND LOSS
# =====================================================================

def softmax(logits):

    shifted = (
        logits -
        np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    exp_values = np.exp(
        shifted
    )

    return exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )


def cross_entropy_loss(
    probabilities,
    labels
):

    batch_size = len(labels)

    selected = probabilities[
        np.arange(batch_size),
        labels
    ]

    selected = np.clip(
        selected,
        1e-12,
        1.0
    )

    return -np.mean(
        np.log(selected)
    )


# =====================================================================
# STEP 8 — FULLY CONNECTED LAYER
# =====================================================================

def dense_forward(x, weights, bias):

    return x @ weights + bias


def dense_backward(
    dout,
    x,
    weights
):

    dx = dout @ weights.T

    dw = x.T @ dout

    db = np.sum(
        dout,
        axis=0
    )

    return dx, dw, db


# =====================================================================
# STEP 9 — CNN MODEL
# =====================================================================

class CNNClassifier:

    def __init__(
        self,
        seed=SEED
    ):

        rng = np.random.default_rng(
            seed
        )

        # Conv1: 1 -> 8 channels.
        self.W1 = (
            rng.standard_normal(
                (3, 3, 1, 8)
            ) *
            np.sqrt(2.0 / 9)
        ).astype(np.float32)

        self.b1 = np.zeros(
            8,
            dtype=np.float32
        )

        # Conv2: 8 -> 16 channels.
        self.W2 = (
            rng.standard_normal(
                (3, 3, 8, 16)
            ) *
            np.sqrt(2.0 / (3 * 3 * 8))
        ).astype(np.float32)

        self.b2 = np.zeros(
            16,
            dtype=np.float32
        )

        # 8 x 8 x 16 = 1024 features.
        self.W3 = (
            rng.standard_normal(
                (1024, 128)
            ) *
            np.sqrt(2.0 / 1024)
        ).astype(np.float32)

        self.b3 = np.zeros(
            128,
            dtype=np.float32
        )

        self.W4 = (
            rng.standard_normal(
                (128, NUM_CLASSES)
            ) *
            np.sqrt(1.0 / 128)
        ).astype(np.float32)

        self.b4 = np.zeros(
            NUM_CLASSES,
            dtype=np.float32
        )

        self.parameters = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3,
            "W4": self.W4,
            "b4": self.b4
        }

        self.m = {
            key: np.zeros_like(value)
            for key, value in self.parameters.items()
        }

        self.v = {
            key: np.zeros_like(value)
            for key, value in self.parameters.items()
        }

        self.step = 0

    # -----------------------------------------------------------------
    # FORWARD
    # -----------------------------------------------------------------

    def forward(self, x):

        # Input: N,32,32,1

        z1, conv1_cache = conv2d_forward(
            x,
            self.W1,
            self.b1,
            padding=1
        )

        a1, relu1_cache = relu_forward(
            z1
        )

        p1, pool1_cache = max_pool_forward(
            a1
        )

        # N,16,16,8

        z2, conv2_cache = conv2d_forward(
            p1,
            self.W2,
            self.b2,
            padding=1
        )

        a2, relu2_cache = relu_forward(
            z2
        )

        p2, pool2_cache = max_pool_forward(
            a2
        )

        # N,8,8,16

        flattened = p2.reshape(
            len(p2),
            -1
        )

        z3 = dense_forward(
            flattened,
            self.W3,
            self.b3
        )

        a3, relu3_cache = relu_forward(
            z3
        )

        logits = dense_forward(
            a3,
            self.W4,
            self.b4
        )

        probabilities = softmax(
            logits
        )

        cache = {
            "input": x,

            "conv1": conv1_cache,
            "relu1": relu1_cache,
            "pool1": pool1_cache,

            "conv2": conv2_cache,
            "relu2": relu2_cache,
            "pool2": pool2_cache,

            "pool2_output": p2,
            "flattened": flattened,

            "relu3": relu3_cache,
            "dense_input": a3
        }

        return probabilities, cache

    # -----------------------------------------------------------------
    # BACKWARD
    # -----------------------------------------------------------------

    def backward(
        self,
        cache,
        labels,
        probabilities
    ):

        batch_size = len(labels)

        targets = np.zeros_like(
            probabilities
        )

        targets[
            np.arange(batch_size),
            labels
        ] = 1.0

        # Softmax + cross-entropy.
        dlogits = (
            probabilities - targets
        ) / batch_size

        # Output dense layer.
        da3, dW4, db4 = dense_backward(
            dlogits,
            cache["dense_input"],
            self.W4
        )

        # ReLU.
        dz3 = relu_backward(
            da3,
            cache["relu3"]
        )

        # Hidden dense layer.
        dflat, dW3, db3 = dense_backward(
            dz3,
            cache["flattened"],
            self.W3
        )

        # Restore pooled feature map dimensions.
        dp2 = dflat.reshape(
            cache["pool2_output"].shape
        )

        # Pool2 backward.
        da2 = max_pool_backward(
            dp2,
            cache["pool2"]
        )

        # ReLU2 backward.
        dz2 = relu_backward(
            da2,
            cache["relu2"]
        )

        # Conv2 backward.
        dp1, dW2, db2 = conv2d_backward(
            dz2,
            cache["conv2"]
        )

        # Pool1 backward.
        da1 = max_pool_backward(
            dp1,
            cache["pool1"]
        )

        # ReLU1 backward.
        dz1 = relu_backward(
            da1,
            cache["relu1"]
        )

        # Conv1 backward.
        dx, dW1, db1 = conv2d_backward(
            dz1,
            cache["conv1"]
        )

        gradients = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
            "W4": dW4,
            "b4": db4
        }

        return gradients

    # -----------------------------------------------------------------
    # ADAM
    # -----------------------------------------------------------------

    def update(
        self,
        gradients,
        learning_rate=LEARNING_RATE
    ):

        self.step += 1

        t = self.step

        for name, parameter in self.parameters.items():

            gradient = gradients[name]

            if name.startswith("W"):

                gradient = (
                    gradient +
                    WEIGHT_DECAY * parameter
                )

            self.m[name] = (
                BETA1 * self.m[name] +
                (1.0 - BETA1) * gradient
            )

            self.v[name] = (
                BETA2 * self.v[name] +
                (1.0 - BETA2) * gradient ** 2
            )

            m_hat = self.m[name] / (
                1.0 - BETA1 ** t
            )

            v_hat = self.v[name] / (
                1.0 - BETA2 ** t
            )

            parameter -= (
                learning_rate *
                m_hat /
                (
                    np.sqrt(v_hat) +
                    EPSILON
                )
            )

    # -----------------------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------------------

    def predict_proba(self, x):

        probabilities, _ = self.forward(x)

        return probabilities

    def predict(self, x):

        return np.argmax(
            self.predict_proba(x),
            axis=1
        )


# =====================================================================
# STEP 10 — TRAINING AND EVALUATION
# =====================================================================

@dataclass
class History:

    train_loss: list

    validation_loss: list

    train_accuracy: list

    validation_accuracy: list


def evaluate(model, X, y):

    probabilities = model.predict_proba(X)

    loss = cross_entropy_loss(
        probabilities,
        y
    )

    accuracy = np.mean(
        np.argmax(
            probabilities,
            axis=1
        ) == y
    )

    return loss, accuracy


def train(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation
):

    history = History(
        train_loss=[],
        validation_loss=[],
        train_accuracy=[],
        validation_accuracy=[]
    )

    rng = np.random.default_rng(
        SEED + 100
    )

    print("\nTRAINING CNN")

    print("=" * 75)

    for epoch in range(EPOCHS):

        indices = rng.permutation(
            len(X_train)
        )

        X_shuffled = X_train[indices]

        y_shuffled = y_train[indices]

        for start in range(
            0,
            len(X_train),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(X_train)
            )

            X_batch = X_shuffled[start:end]

            y_batch = y_shuffled[start:end]

            probabilities, cache = model.forward(
                X_batch
            )

            gradients = model.backward(
                cache,
                y_batch,
                probabilities
            )

            model.update(
                gradients
            )

        train_loss, train_accuracy = evaluate(
            model,
            X_train,
            y_train
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            X_validation,
            y_validation
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
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Validation loss: {validation_loss:.4f} | "
            f"Validation accuracy: {validation_accuracy:.4f}"
        )

    return history


# =====================================================================
# STEP 11 — METRICS
# =====================================================================

def calculate_confusion_matrix(
    true_labels,
    predictions
):

    matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    for true, predicted in zip(
        true_labels,
        predictions
    ):

        matrix[true, predicted] += 1

    return matrix


# =====================================================================
# STEP 12 — TESTS
# =====================================================================

def run_tests():

    print("\n" + "=" * 70)

    print("RUNNING LESSON 22 TESTS")

    print("=" * 70)

    rng = np.random.default_rng(123)

    # ---------------------------------------------------------------
    # TEST 1 — Convolution output shape
    # ---------------------------------------------------------------

    x = rng.normal(
        size=(2, 8, 8, 3)
    ).astype(np.float32)

    weights = rng.normal(
        size=(3, 3, 3, 4)
    ).astype(np.float32)

    bias = np.zeros(
        4,
        dtype=np.float32
    )

    output, cache = conv2d_forward(
        x,
        weights,
        bias,
        padding=1
    )

    assert output.shape == (
        2,
        8,
        8,
        4
    )

    print("Test 1 passed: convolution output shape.")

    # ---------------------------------------------------------------
    # TEST 2 — OpenCV convolution comparison
    # ---------------------------------------------------------------

    image = rng.normal(
        size=(12, 12)
    ).astype(np.float32)

    kernel = rng.normal(
        size=(3, 3)
    ).astype(np.float32)

    numpy_output, _ = conv2d_forward(
        image[None, ..., None],
        kernel[..., None, None],
        np.zeros(1, dtype=np.float32),
        padding=1
    )

    # OpenCV filter2D performs correlation, not mathematical
    # kernel-flipped convolution. Our operation also uses
    # cross-correlation, so the kernel is not flipped.
    cv_output = cv2.filter2D(
        image,
        -1,
        kernel,
        borderType=cv2.BORDER_CONSTANT
    )

    assert np.allclose(
        numpy_output[0, :, :, 0],
        cv_output,
        atol=1e-5
    )

    print("Test 2 passed: NumPy convolution vs OpenCV.")

    # ---------------------------------------------------------------
    # TEST 3 — Convolution backward shapes
    # ---------------------------------------------------------------

    dout = rng.normal(
        size=output.shape
    ).astype(np.float32)

    dx, dw, db = conv2d_backward(
        dout,
        cache
    )

    assert dx.shape == x.shape

    assert dw.shape == weights.shape

    assert db.shape == bias.shape

    print("Test 3 passed: convolution backward shapes.")

    # ---------------------------------------------------------------
    # TEST 4 — Convolution numerical gradient
    # ---------------------------------------------------------------

    small_x = rng.normal(
        size=(1, 4, 4, 1)
    ).astype(np.float32)

    small_w = rng.normal(
        size=(3, 3, 1, 1)
    ).astype(np.float32)

    small_b = np.zeros(
        1,
        dtype=np.float32
    )

    small_output, small_cache = conv2d_forward(
        small_x,
        small_w,
        small_b,
        padding=1
    )

    small_dout = rng.normal(
        size=small_output.shape
    ).astype(np.float32)

    _, analytical_dw, _ = conv2d_backward(
        small_dout,
        small_cache
    )

    epsilon = 1e-3

    original = small_w[1, 1, 0, 0]

    small_w[1, 1, 0, 0] = original + epsilon

    plus_output, _ = conv2d_forward(
        small_x,
        small_w,
        small_b,
        padding=1
    )

    loss_plus = np.sum(
        plus_output * small_dout
    )

    small_w[1, 1, 0, 0] = original - epsilon

    minus_output, _ = conv2d_forward(
        small_x,
        small_w,
        small_b,
        padding=1
    )

    loss_minus = np.sum(
        minus_output * small_dout
    )

    small_w[1, 1, 0, 0] = original

    numerical_dw = (
        loss_plus - loss_minus
    ) / (2 * epsilon)

    assert np.isclose(
        analytical_dw[1, 1, 0, 0],
        numerical_dw,
        atol=1e-3
    )

    print("Test 4 passed: convolution gradient check.")

    # ---------------------------------------------------------------
    # TEST 5 — ReLU
    # ---------------------------------------------------------------

    values = np.array([
        [-2.0, 0.0, 3.0]
    ])

    activated, _ = relu_forward(values)

    assert np.allclose(
        activated,
        [[0.0, 0.0, 3.0]]
    )

    print("Test 5 passed: ReLU.")

    # ---------------------------------------------------------------
    # TEST 6 — Max pooling
    # ---------------------------------------------------------------

    pool_input = np.array([
        [
            [
                [1.0],
                [2.0]
            ],
            [
                [3.0],
                [4.0]
            ]
        ]
    ])

    pool_output, pool_cache = max_pool_forward(
        pool_input
    )

    assert pool_output.shape == (
        1,
        1,
        1,
        1
    )

    assert pool_output[0, 0, 0, 0] == 4.0

    print("Test 6 passed: max pooling.")

    # ---------------------------------------------------------------
    # TEST 7 — Max pooling backward
    # ---------------------------------------------------------------

    pool_gradient = np.ones_like(
        pool_output
    )

    pool_dx = max_pool_backward(
        pool_gradient,
        pool_cache
    )

    assert pool_dx[0, 1, 1, 0] == 1.0

    assert np.sum(pool_dx) == 1.0

    print("Test 7 passed: max pooling backward.")

    # ---------------------------------------------------------------
    # TEST 8 — Softmax
    # ---------------------------------------------------------------

    probabilities = softmax(
        np.array([
            [1.0, 2.0, 3.0],
            [3.0, 1.0, 0.0]
        ])
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0
    )

    print("Test 8 passed: softmax.")

    # ---------------------------------------------------------------
    # TEST 9 — CNN forward
    # ---------------------------------------------------------------

    model = CNNClassifier()

    batch = rng.random(
        size=(2, 32, 32, 1)
    ).astype(np.float32)

    probabilities, cache = model.forward(
        batch
    )

    assert probabilities.shape == (
        2,
        NUM_CLASSES
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6
    )

    print("Test 9 passed: complete CNN forward pass.")

    # ---------------------------------------------------------------
    # TEST 10 — CNN backward
    # ---------------------------------------------------------------

    gradients = model.backward(
        cache,
        np.array([0, 1]),
        probabilities
    )

    for gradient in gradients.values():

        assert np.all(
            np.isfinite(gradient)
        )

    print("Test 10 passed: complete CNN backpropagation.")

    # ---------------------------------------------------------------
    # TEST 11 — Adam update
    # ---------------------------------------------------------------

    original = model.W1.copy()

    model.update(
        gradients
    )

    assert not np.allclose(
        original,
        model.W1
    )

    print("Test 11 passed: Adam update.")

    print("\nALL LESSON 22 TESTS PASSED.")


# =====================================================================
# STEP 13 — VISUALIZATION
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
            images[index, :, :, 0],
            cmap="gray"
        )

        ax.set_title(
            CLASS_NAMES[labels[index]]
        )

        ax.axis("off")

    plt.suptitle(
        "Synthetic CNN Training Dataset"
    )

    plt.tight_layout()

    plt.show()


def visualize_history(history):

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
        label="Train"
    )

    axes[0].plot(
        epochs,
        history.validation_loss,
        label="Validation"
    )

    axes[0].set_title(
        "Training and Validation Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Cross-Entropy"
    )

    axes[0].legend()

    axes[0].grid(
        True
    )

    axes[1].plot(
        epochs,
        history.train_accuracy,
        label="Train"
    )

    axes[1].plot(
        epochs,
        history.validation_accuracy,
        label="Validation"
    )

    axes[1].set_title(
        "Training and Validation Accuracy"
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


def visualize_confusion(matrix):

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        matrix,
        cmap="Blues"
    )

    plt.title(
        "CNN Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
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
            images[index, :, :, 0],
            cmap="gray"
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
        "CNN Test Predictions"
    )

    plt.tight_layout()

    plt.show()


# =====================================================================
# STEP 14 — EXPERIMENT
# =====================================================================

def run_experiment():

    print("\nGENERATING DATASET")

    images, labels = generate_dataset()

    print(
        f"Images: {images.shape}"
    )

    print(
        f"Labels: {labels.shape}"
    )

    visualize_dataset(
        images,
        labels
    )

    rng = np.random.default_rng(
        SEED
    )

    indices = rng.permutation(
        len(images)
    )

    images = images[indices]

    labels = labels[indices]

    train_end = int(
        len(images) * TRAIN_RATIO
    )

    validation_end = train_end + int(
        len(images) * VALIDATION_RATIO
    )

    X_train = images[:train_end]

    y_train = labels[:train_end]

    X_validation = images[
        train_end:validation_end
    ]

    y_validation = labels[
        train_end:validation_end
    ]

    X_test = images[
        validation_end:
    ]

    y_test = labels[
        validation_end:
    ]

    print("\nDATASET SPLIT")

    print(
        f"Training: {len(X_train)}"
    )

    print(
        f"Validation: {len(X_validation)}"
    )

    print(
        f"Test: {len(X_test)}"
    )

    model = CNNClassifier()

    parameter_count = sum(
        parameter.size
        for parameter in model.parameters.values()
    )

    print(
        f"\nCNN parameters: {parameter_count:,}"
    )

    history = train(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    test_loss, test_accuracy = evaluate(
        model,
        X_test,
        y_test
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

    predictions = model.predict(
        X_test
    )

    matrix = calculate_confusion_matrix(
        y_test,
        predictions
    )

    print("\nCONFUSION MATRIX")

    print(matrix)

    visualize_history(
        history
    )

    visualize_confusion(
        matrix
    )

    probabilities = model.predict_proba(
        X_test
    )

    visualize_predictions(
        X_test,
        y_test,
        probabilities
    )

    return model


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)

    print("COMPUTER VISION — PHASE 2 — LESSON 22")

    print("CONVOLUTIONAL NEURAL NETWORK FROM SCRATCH")

    print("=" * 70)

    run_tests()

    run_experiment()

    print("\n" + "=" * 70)

    print("LESSON 22 COMPLETED")

    print("=" * 70)

    print(
        "NEXT: LESSON 23 — CNN OPERATIONS AND FEATURE MAPS"
    )


if __name__ == "__main__":

    main()