# Computer Vision From Scratch — Beginner to Advanced

> **Core rule:** This course does not use pretrained models. Every major algorithm and neural-network architecture is implemented, trained, tested, and visualized from scratch.


## Course Goal

Build deep practical understanding of Computer Vision from pixels and image mathematics through classical CV, CNNs, detection, segmentation, tracking, 3-D robotics vision, transformers, vision-language models, and production systems.


## From-Scratch Policy

- No pretrained weights.

- No pretrained foundation models as the primary teaching implementation.

- Neural networks are initialized randomly and trained by us.

- Core algorithms are implemented from their mathematical definitions.

- NumPy is used extensively for numerical implementations.

- PyTorch is introduced for training larger neural networks, but architectures are built by us.

- OpenCV may be used for image I/O, visualization, camera interfaces, and comparison/verification, but the learning implementation is written by us whenever practical.


## Learning Pattern

Every substantial lesson follows:

```text
Theory
  ↓
Mathematical Derivation
  ↓
From-Scratch Implementation
  ↓
Visualization
  ↓
Numerical / Unit Tests
  ↓
Real-World Example
  ↓
Exercise / Extension
```


## Prerequisites

- Python fundamentals
- NumPy
- Basic linear algebra
- Basic probability
- Basic calculus
- Basic PyTorch concepts


## Recommended Environment

```bash
python >= 3.10
pip install numpy opencv-python matplotlib scipy torch torchvision pillow
```


## Course Index


### phase0_foundations

#### Lesson 1 — Introduction to Computer Vision Engineering

**Implementation:** `phase0_foundations/lesson01_introduction.py`


**Real-world context:** Image pixels, channels, image display, and a simple visual inspection pipeline.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 2 — Digital Images and Pixel Representation

**Implementation:** `phase0_foundations/lesson02_digital_images.py`


**Real-world context:** Represent images as arrays; inspect shape, dtype, range, indexing, and channel order.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 3 — Color Spaces: RGB, HSV, LAB

**Implementation:** `phase0_foundations/lesson03_color_spaces.py`


**Real-world context:** Convert and compare color representations; demonstrate color thresholding.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 4 — Image Filtering and Convolution

**Implementation:** `phase0_foundations/lesson04_image_filtering.py`


**Real-world context:** Implement 2-D convolution and compare smoothing filters.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 5 — Gaussian, Median, and Bilateral Filters

**Implementation:** `phase0_foundations/lesson05_gaussian_median_bilateral.py`


**Real-world context:** Compare noise removal methods and edge preservation.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 6 — Edge Detection: Sobel and Canny

**Implementation:** `phase0_foundations/lesson06_edge_detection.py`


**Real-world context:** Compute image gradients and visualize edge maps.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 7 — Image Thresholding and Segmentation

**Implementation:** `phase0_foundations/lesson07_thresholding.py`


**Real-world context:** Global, adaptive, and Otsu-style thresholding.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 8 — Morphological Operations

**Implementation:** `phase0_foundations/lesson08_morphology.py`


**Real-world context:** Erosion, dilation, opening, and closing on binary masks.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 9 — Image Transformation and Perspective

**Implementation:** `phase0_foundations/lesson09_image_transform.py`


**Real-world context:** Affine transforms, homography concepts, and coordinate mapping.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 10 — Feature Detection: Harris, SIFT, and ORB

**Implementation:** `phase0_foundations/lesson10_feature_detection.py`


**Real-world context:** Detect corners and explain local feature descriptors.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.


### phase1_classical_cv

#### Lesson 11 — Contour Detection

**Implementation:** `phase1_classical_cv/lesson11_contours.py`


**Real-world context:** Extract object boundaries from binary images.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 12 — Connected Component Analysis

**Implementation:** `phase1_classical_cv/lesson12_connected_components.py`


**Real-world context:** Label and measure disconnected regions.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 13 — Template Matching

**Implementation:** `phase1_classical_cv/lesson13_template_matching.py`


**Real-world context:** Locate a template in a larger image with normalized correlation.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 14 — Feature Matching

**Implementation:** `phase1_classical_cv/lesson14_feature_matching.py`


**Real-world context:** Match local descriptors conceptually; use ORB when OpenCV contrib is available.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 15 — Optical Flow: Lucas–Kanade

**Implementation:** `phase1_classical_cv/lesson15_optical_flow_lk.py`


**Real-world context:** Estimate sparse motion from image point correspondences.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 16 — Dense Optical Flow

**Implementation:** `phase1_classical_cv/lesson16_optical_flow_dense.py`


**Real-world context:** Visualize a synthetic dense motion field.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 17 — Background Subtraction

**Implementation:** `phase1_classical_cv/lesson17_background_subtraction.py`


**Real-world context:** Build a running-average background model and foreground mask.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 18 — Camera Calibration

**Implementation:** `phase1_classical_cv/lesson18_camera_calibration.py`


**Real-world context:** Understand intrinsics, distortion, and calibrate with a checkerboard when available.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 19 — Perspective Transformation

**Implementation:** `phase1_classical_cv/lesson19_perspective.py`


**Real-world context:** Apply a homography to rectify a quadrilateral.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 20 — Stereo Vision and Depth Estimation

**Implementation:** `phase1_classical_cv/lesson20_stereo_depth.py`


**Real-world context:** Compute disparity and depth from a simplified stereo geometry model.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.


### phase2_deep_learning

#### Lesson 21 — Neural Networks for Image Classification

**Implementation:** `phase2_deep_learning/lesson21_image_classification.py`


**Real-world context:** Train a small classifier on synthetic image patterns with PyTorch if installed.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 22 — CNN Architecture from Scratch

**Implementation:** `phase2_deep_learning/lesson22_cnn_from_scratch.py`


**Real-world context:** Inspect convolution, activation, pooling, and feature maps.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 23 — Convolution, Pooling, and Feature Maps

**Implementation:** `phase2_deep_learning/lesson23_cnn_operations.py`


**Real-world context:** Implement convolution and max pooling on small arrays.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 24 — ResNet and Residual Connections

**Implementation:** `phase2_deep_learning/lesson24_resnet.py`


**Real-world context:** Demonstrate residual blocks and gradient-friendly identity paths.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 25 — Transfer Learning

**Implementation:** `phase2_deep_learning/lesson25_transfer_learning.py`


**Real-world context:** Load a pretrained torchvision model when weights are available.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 26 — Data Augmentation

**Implementation:** `phase2_deep_learning/lesson26_augmentation.py`


**Real-world context:** Apply flips, rotations, brightness changes, and crops.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 27 — Object Detection Fundamentals

**Implementation:** `phase2_deep_learning/lesson27_detection_fundamentals.py`


**Real-world context:** Represent boxes, IoU, precision, recall, and average precision concepts.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 28 — YOLO Architecture and Training

**Implementation:** `phase2_deep_learning/lesson28_yolo_training.py`


**Real-world context:** Train a small custom detector using Ultralytics when installed and a dataset is supplied.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 29 — Faster R-CNN

**Implementation:** `phase2_deep_learning/lesson29_faster_rcnn.py`


**Real-world context:** Inspect a torchvision detector and run inference on a sample image.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 30 — Semantic and Instance Segmentation

**Implementation:** `phase2_deep_learning/lesson30_segmentation.py`


**Real-world context:** Create masks, calculate IoU/Dice, and inspect segmentation outputs.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.


### phase3_detection_segmentation

#### Lesson 31 — Advanced YOLO Training

**Implementation:** `phase3_detection_segmentation/lesson31_yolo_advanced.py`


**Real-world context:** Configure training, validation, augmentation, and experiment tracking.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 32 — Anchor-Based vs Anchor-Free Detection

**Implementation:** `phase3_detection_segmentation/lesson32_anchor_free.py`


**Real-world context:** Compare anchor assignment and center-based predictions conceptually.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 33 — Non-Maximum Suppression

**Implementation:** `phase3_detection_segmentation/lesson33_nms.py`


**Real-world context:** Implement IoU-based greedy NMS from scratch.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 34 — Detection Confidence Calibration

**Implementation:** `phase3_detection_segmentation/lesson34_confidence_calibration.py`


**Real-world context:** Inspect reliability diagrams and expected calibration error.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 35 — Instance Segmentation with Mask R-CNN

**Implementation:** `phase3_detection_segmentation/lesson35_mask_rcnn.py`


**Real-world context:** Run torchvision Mask R-CNN inference when dependencies/weights are available.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 36 — YOLO Segmentation

**Implementation:** `phase3_detection_segmentation/lesson36_yolo_segmentation.py`


**Real-world context:** Train or infer a YOLO segmentation model using Ultralytics.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 37 — Vision Transformers

**Implementation:** `phase3_detection_segmentation/lesson37_vision_transformer.py`


**Real-world context:** Patchify an image and demonstrate self-attention numerically.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 38 — DETR and Transformer Detection

**Implementation:** `phase3_detection_segmentation/lesson38_detr.py`


**Real-world context:** Explain set prediction and Hungarian matching with a toy assignment.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 39 — Segment Anything Model

**Implementation:** `phase3_detection_segmentation/lesson39_sam.py`


**Real-world context:** Run SAM when the optional package and checkpoint are supplied.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 40 — Open-Vocabulary Detection

**Implementation:** `phase3_detection_segmentation/lesson40_open_vocabulary.py`


**Real-world context:** Explore text-conditioned detection with Grounding DINO when configured.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.


### phase4_tracking_memory

#### Lesson 41 — Object Tracking Fundamentals

**Implementation:** `phase4_tracking_memory/lesson41_tracking_fundamentals.py`


**Real-world context:** Track synthetic detections and distinguish detection from identity tracking.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 42 — Kalman Filter from Scratch

**Implementation:** `phase4_tracking_memory/lesson42_kalman_filter.py`


**Real-world context:** Implement a constant-velocity Kalman filter with prediction and correction.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 43 — Hungarian Algorithm

**Implementation:** `phase4_tracking_memory/lesson43_hungarian.py`


**Real-world context:** Solve a data-association cost matrix with SciPy or a small fallback.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 44 — SORT Tracking

**Implementation:** `phase4_tracking_memory/lesson44_sort.py`


**Real-world context:** Combine Kalman prediction, IoU association, and track lifecycle management.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 45 — DeepSORT

**Implementation:** `phase4_tracking_memory/lesson45_deepsort.py`


**Real-world context:** Add appearance embeddings to motion-based association; demonstrate a toy tracker.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 46 — ByteTrack

**Implementation:** `phase4_tracking_memory/lesson46_bytetrack.py`


**Real-world context:** Associate high-confidence detections first, then recover tracks with low-confidence detections.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 47 — BoT-SORT and OC-SORT

**Implementation:** `phase4_tracking_memory/lesson47_botsort_ocs​​ort.py`


**Real-world context:** Compare motion compensation and observation-centric updates conceptually.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 48 — Person Re-Identification

**Implementation:** `phase4_tracking_memory/lesson48_reid.py`


**Real-world context:** Extract/compare normalized appearance embeddings and discuss evaluation.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 49 — Appearance-Based Tracking

**Implementation:** `phase4_tracking_memory/lesson49_appearance_tracking.py`


**Real-world context:** Combine color histograms and motion costs for identity association.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 50 — Tracklet Management

**Implementation:** `phase4_tracking_memory/lesson50_tracklet_management.py`


**Real-world context:** Handle tentative, confirmed, lost, and removed track states.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 51 — Long-Term Object Tracking

**Implementation:** `phase4_tracking_memory/lesson51_long_term_tracking.py`


**Real-world context:** Reacquire objects after long absence using a gallery of appearance descriptors.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 52 — Memory-Based Object Tracking

**Implementation:** `phase4_tracking_memory/lesson52_memory_based_tracking.py`


**Real-world context:** Build short-term, long-term, temporal, and spatial memory for tracked objects.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 53 — Occlusion Handling

**Implementation:** `phase4_tracking_memory/lesson53_occlusion.py`


**Real-world context:** Predict through occlusion and recover identity after reappearance.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 54 — Multi-Camera Tracking

**Implementation:** `phase4_tracking_memory/lesson54_multi_camera.py`


**Real-world context:** Associate identities across camera views using appearance and time constraints.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 55 — Video Object Segmentation

**Implementation:** `phase4_tracking_memory/lesson55_video_segmentation.py`


**Real-world context:** Propagate a mask using optical-flow-like coordinate warping.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 56 — SAM 2 Video Tracking

**Implementation:** `phase4_tracking_memory/lesson56_sam2_tracking.py`


**Real-world context:** Prepare a video predictor workflow and explain memory-conditioned mask propagation.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.


### phase5_robotics_vision

#### Lesson 57 — RGB-D Computer Vision

**Implementation:** `phase5_robotics_vision/lesson57_rgbd.py`


**Real-world context:** Align color and depth concepts; back-project pixels into 3-D points.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 58 — Point Clouds

**Implementation:** `phase5_robotics_vision/lesson58_point_clouds.py`


**Real-world context:** Generate, transform, filter, and visualize a synthetic point cloud.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 59 — 3-D Object Detection

**Implementation:** `phase5_robotics_vision/lesson59_3d_detection.py`


**Real-world context:** Estimate 3-D bounding boxes from point clusters.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 60 — 6-D Object Pose Estimation

**Implementation:** `phase5_robotics_vision/lesson60_pose6d.py`


**Real-world context:** Represent object pose with rotation and translation; project 3-D points.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 61 — PnP and Perspective Geometry

**Implementation:** `phase5_robotics_vision/lesson61_pnp.py`


**Real-world context:** Estimate pose from 2-D/3-D correspondences using OpenCV when available.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 62 — Hand-Eye Calibration

**Implementation:** `phase5_robotics_vision/lesson62_hand_eye.py`


**Real-world context:** Explain AX=XB and solve a synthetic calibration example.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 63 — Visual Servoing

**Implementation:** `phase5_robotics_vision/lesson63_visual_servoing.py`


**Real-world context:** Implement image-based visual servoing with a simple interaction matrix.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 64 — Visual Odometry

**Implementation:** `phase5_robotics_vision/lesson64_visual_odometry.py`


**Real-world context:** Estimate relative camera motion from feature correspondences conceptually.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 65 — Visual SLAM

**Implementation:** `phase5_robotics_vision/lesson65_visual_slam.py`


**Real-world context:** Build a minimal feature/map/pose pipeline simulation.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 66 — ORB-SLAM Fundamentals

**Implementation:** `phase5_robotics_vision/lesson66_orb_slam.py`


**Real-world context:** Understand tracking, local mapping, loop closing, and ORB feature roles.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 67 — Loop Closure Detection

**Implementation:** `phase5_robotics_vision/lesson67_loop_closure.py`


**Real-world context:** Compare visual place descriptors and demonstrate drift correction.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 68 — Scene Understanding

**Implementation:** `phase5_robotics_vision/lesson68_scene_understanding.py`


**Real-world context:** Combine object labels, spatial relations, and scene graphs.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.


### phase6_model_optimization

#### Lesson 69 — Model Profiling

**Implementation:** `phase6_model_optimization/lesson69_profiling.py`


**Real-world context:** Measure warm-up, latency distribution, throughput, and memory where possible.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 70 — CPU vs GPU Inference

**Implementation:** `phase6_model_optimization/lesson70_cpu_gpu.py`


**Real-world context:** Benchmark tensor operations on CPU and CUDA when available.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 71 — Batch Size Optimization

**Implementation:** `phase6_model_optimization/lesson71_batching.py`


**Real-world context:** Measure throughput and latency across batch sizes.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 72 — Mixed Precision FP16

**Implementation:** `phase6_model_optimization/lesson72_fp16.py`


**Real-world context:** Compare FP32 and FP16 tensor behavior and numerical error.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 73 — INT8 Quantization

**Implementation:** `phase6_model_optimization/lesson73_int8.py`


**Real-world context:** Demonstrate dynamic quantization and calibration concepts.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 74 — Model Pruning

**Implementation:** `phase6_model_optimization/lesson74_pruning.py`


**Real-world context:** Prune weights and measure sparsity and accuracy impact on a toy model.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 75 — Knowledge Distillation

**Implementation:** `phase6_model_optimization/lesson75_distillation.py`


**Real-world context:** Train a student network using teacher soft targets.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 76 — ONNX Model Export

**Implementation:** `phase6_model_optimization/lesson76_onnx_export.py`


**Real-world context:** Export a small PyTorch model to ONNX when dependencies are installed.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 77 — ONNX Runtime Optimization

**Implementation:** `phase6_model_optimization/lesson77_onnx_runtime.py`


**Real-world context:** Run an ONNX model and inspect available execution providers.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 78 — NVIDIA TensorRT

**Implementation:** `phase6_model_optimization/lesson78_tensorrt.py`


**Real-world context:** Prepare an export/build/inference workflow and check TensorRT availability.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 79 — Dynamic Batching

**Implementation:** `phase6_model_optimization/lesson79_dynamic_batching.py`


**Real-world context:** Implement a simple request queue and batch scheduler.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 80 — Asynchronous Inference

**Implementation:** `phase6_model_optimization/lesson80_async_inference.py`


**Real-world context:** Build a threaded producer/consumer image pipeline.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 81 — Memory Optimization

**Implementation:** `phase6_model_optimization/lesson81_memory_optimization.py`


**Real-world context:** Measure tensor allocation and demonstrate reuse/preallocation.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 82 — Model Benchmarking

**Implementation:** `phase6_model_optimization/lesson82_benchmarking.py`


**Real-world context:** Report FPS, mean, median, P95 latency, model size, and accuracy.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 83 — Edge AI Deployment

**Implementation:** `phase6_model_optimization/lesson83_edge_ai.py`


**Real-world context:** Inspect device constraints and create an edge deployment checklist.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.


### phase7_foundation_models

#### Lesson 84 — Vision Foundation Models

**Implementation:** `phase7_foundation_models/lesson84_foundation_models.py`


**Real-world context:** Compare task-specific models with reusable pretrained visual representations.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 85 — CLIP

**Implementation:** `phase7_foundation_models/lesson85_clip.py`


**Real-world context:** Perform image/text similarity with a pretrained CLIP model when available.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 86 — DINOv2

**Implementation:** `phase7_foundation_models/lesson86_dinov2.py`


**Real-world context:** Extract self-supervised visual embeddings when Transformers is installed.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 87 — Self-Supervised Learning

**Implementation:** `phase7_foundation_models/lesson87_self_supervised.py`


**Real-world context:** Demonstrate contrastive learning with augmented views in a toy embedding space.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 88 — Vision-Language Models

**Implementation:** `phase7_foundation_models/lesson88_vlm.py`


**Real-world context:** Run an image-question pipeline with a compatible Transformers model.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 89 — Florence-2

**Implementation:** `phase7_foundation_models/lesson89_florence2.py`


**Real-world context:** Explore captioning, grounding, and region tasks using a supported checkpoint.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 90 — Grounding DINO

**Implementation:** `phase7_foundation_models/lesson90_grounding_dino.py`


**Real-world context:** Ground text prompts in images using the optional Grounding DINO package.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 91 — SAM 2 Advanced

**Implementation:** `phase7_foundation_models/lesson91_sam2_advanced.py`


**Real-world context:** Use prompts and memory propagation for video segmentation.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 92 — Video Understanding Transformers

**Implementation:** `phase7_foundation_models/lesson92_video_transformers.py`


**Real-world context:** Build frame clips and discuss temporal attention.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 93 — Anomaly Detection

**Implementation:** `phase7_foundation_models/lesson93_anomaly_detection.py`


**Real-world context:** Train a simple one-class feature-distance detector.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 94 — Few-Shot and Zero-Shot Learning

**Implementation:** `phase7_foundation_models/lesson94_few_zero_shot.py`


**Real-world context:** Compare nearest-prototype and text-embedding classification.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 95 — Visual Question Answering

**Implementation:** `phase7_foundation_models/lesson95_vqa.py`


**Real-world context:** Use an image-question model when a compatible checkpoint is supplied.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 96 — Vision-Language-Action

**Implementation:** `phase7_foundation_models/lesson96_vla.py`


**Real-world context:** Connect visual observations, language goals, and action representations.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.


### phase8_production

#### Lesson 97 — Real-Time Multi-Camera Architecture

**Implementation:** `phase8_production/lesson97_multicamera.py`


**Real-world context:** Design independent capture, inference, tracking, and output workers.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 98 — Video Streaming with OpenCV

**Implementation:** `phase8_production/lesson98_video_streaming.py`


**Real-world context:** Read, process, and write video with graceful camera fallback.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 99 — RTSP Camera Processing

**Implementation:** `phase8_production/lesson99_rtsp.py`


**Real-world context:** Configure a robust RTSP capture loop with reconnect behavior.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 100 — Multi-Threaded Vision Pipelines

**Implementation:** `phase8_production/lesson100_multithreaded.py`


**Real-world context:** Use bounded queues, backpressure, and worker shutdown.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 101 — GPU Pipeline Optimization

**Implementation:** `phase8_production/lesson101_gpu_pipeline.py`


**Real-world context:** Reduce copies and synchronize inference stages correctly.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 102 — FastAPI Vision Services

**Implementation:** `phase8_production/lesson102_fastapi.py`


**Real-world context:** Expose a model inference endpoint when FastAPI is installed.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 103 — Docker Deployment

**Implementation:** `phase8_production/lesson103_docker.py`


**Real-world context:** Generate and explain a reproducible container configuration.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 104 — Computer Vision Monitoring

**Implementation:** `phase8_production/lesson104_monitoring.py`


**Real-world context:** Track FPS, latency, dropped frames, errors, and health.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 105 — Dataset Management and Versioning

**Implementation:** `phase8_production/lesson105_dataset_versioning.py`


**Real-world context:** Validate dataset manifests, hashes, splits, and metadata.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.

#### Lesson 106 — Model Evaluation and Regression Testing

**Implementation:** `phase8_production/lesson106_regression_testing.py`


**Real-world context:** Build deterministic checks for model outputs and metrics.


**Required approach:** derive the idea, implement the core mechanism from scratch, visualize the result, test it numerically, and explain the failure cases.


## Technology Progression

```text
NumPy
  ↓
OpenCV primitives / verification
  ↓
Mathematical CV algorithms
  ↓
PyTorch from random initialization
  ↓
CNNs
  ↓
Detection / Segmentation
  ↓
Tracking
  ↓
3-D Vision
  ↓
Transformers
  ↓
Vision-Language Models
  ↓
Vision → Action
```


## Examples of What 'From Scratch' Means

### CNN

Do not load a pretrained ResNet. Implement convolution, activation, pooling, residual blocks, forward propagation, loss, backpropagation, and optimization, then train the network from random initialization.


### Object Detection

Do not load pretrained YOLO weights. Build a small detector with a backbone, detection head, box parameterization, objectness/class losses, IoU, and NMS, then train it yourself.


### Vision Transformer

Implement patch extraction, embeddings, positional encoding, scaled dot-product attention, multi-head attention, MLP blocks, residual connections, normalization, and the training loop.


### Tracking

Implement the Kalman filter, prediction/update equations, IoU association, Hungarian matching, track lifecycle, and then combine them into a tracker.


### Robotics Vision

Derive camera projection, calibration, stereo depth, PnP, coordinate transforms, hand-eye calibration, and visual servoing rather than hiding these concepts behind a library call.


## Final Robotics Project

The final project is a from-scratch visual perception pipeline for robotic waste sorting:

```text
Multi-Camera Input
      ↓
Image Processing
      ↓
Object Detection
      ↓
Segmentation
      ↓
Tracking
      ↓
Depth / 3-D Localization
      ↓
6-D Pose / Robot Coordinates
      ↓
Task Understanding
      ↓
Vision → Action
      ↓
OpenArm V2
      ↓
Pick + Sort
```


## Definition of Completion

You have completed the course when you can explain the mathematics behind the major methods, implement simplified versions without pretrained weights, train neural networks from random initialization, diagnose failures, benchmark inference, and connect visual perception to a real robotic manipulation task.