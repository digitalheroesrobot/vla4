# Computer Vision — Beginner to Advanced

## Complete Practical Course

A structured Computer Vision roadmap progressing from image fundamentals and classical OpenCV techniques to deep learning, detection, segmentation, tracking, 3-D robotics vision, vision foundation models, VLM/VLA, optimization, and production deployment.

## Course Philosophy

The course follows the progression:

`Digital Images → Classical CV → Deep Learning → Detection → Segmentation → Tracking → 3-D Vision → Robotics Vision → Foundation Models → VLM/VLA → Production`

## Prerequisites

- Python fundamentals
- NumPy basics
- Basic linear algebra
- Basic probability and calculus
- Basic machine-learning concepts
- Basic PyTorch familiarity


## Recommended Environment

```bash
python >= 3.10
pip install numpy opencv-python matplotlib scipy torch torchvision pillow
```

Optional advanced packages:
```bash
pip install ultralytics transformers
```

## Course Index


### phase0_foundations

- **Lesson 1 —** Introduction to Computer Vision Engineering
- **Lesson 2 —** Digital Images and Pixel Representation
- **Lesson 3 —** Color Spaces: RGB, HSV, LAB
- **Lesson 4 —** Image Filtering and Convolution
- **Lesson 5 —** Gaussian, Median, and Bilateral Filters
- **Lesson 6 —** Edge Detection: Sobel and Canny
- **Lesson 7 —** Image Thresholding and Segmentation
- **Lesson 8 —** Morphological Operations
- **Lesson 9 —** Image Transformation and Perspective
- **Lesson 10 —** Feature Detection: Harris, SIFT, and ORB

### phase1_classical_cv

- **Lesson 11 —** Contour Detection
- **Lesson 12 —** Connected Component Analysis
- **Lesson 13 —** Template Matching
- **Lesson 14 —** Feature Matching
- **Lesson 15 —** Optical Flow: Lucas–Kanade
- **Lesson 16 —** Dense Optical Flow
- **Lesson 17 —** Background Subtraction
- **Lesson 18 —** Camera Calibration
- **Lesson 19 —** Perspective Transformation
- **Lesson 20 —** Stereo Vision and Depth Estimation

### phase2_deep_learning

- **Lesson 21 —** Neural Networks for Image Classification
- **Lesson 22 —** CNN Architecture from Scratch
- **Lesson 23 —** Convolution, Pooling, and Feature Maps
- **Lesson 24 —** ResNet and Residual Connections
- **Lesson 25 —** Transfer Learning
- **Lesson 26 —** Data Augmentation
- **Lesson 27 —** Object Detection Fundamentals
- **Lesson 28 —** YOLO Architecture and Training
- **Lesson 29 —** Faster R-CNN
- **Lesson 30 —** Semantic and Instance Segmentation

### phase3_detection_segmentation

- **Lesson 31 —** Advanced YOLO Training
- **Lesson 32 —** Anchor-Based vs Anchor-Free Detection
- **Lesson 33 —** Non-Maximum Suppression
- **Lesson 34 —** Detection Confidence Calibration
- **Lesson 35 —** Instance Segmentation with Mask R-CNN
- **Lesson 36 —** YOLO Segmentation
- **Lesson 37 —** Vision Transformers
- **Lesson 38 —** DETR and Transformer Detection
- **Lesson 39 —** Segment Anything Model
- **Lesson 40 —** Open-Vocabulary Detection

### phase4_tracking_memory

- **Lesson 41 —** Object Tracking Fundamentals
- **Lesson 42 —** Kalman Filter from Scratch
- **Lesson 43 —** Hungarian Algorithm
- **Lesson 44 —** SORT Tracking
- **Lesson 45 —** DeepSORT
- **Lesson 46 —** ByteTrack
- **Lesson 47 —** BoT-SORT and OC-SORT
- **Lesson 48 —** Person Re-Identification
- **Lesson 49 —** Appearance-Based Tracking
- **Lesson 50 —** Tracklet Management
- **Lesson 51 —** Long-Term Object Tracking
- **Lesson 52 —** Memory-Based Object Tracking
- **Lesson 53 —** Occlusion Handling
- **Lesson 54 —** Multi-Camera Tracking
- **Lesson 55 —** Video Object Segmentation
- **Lesson 56 —** SAM 2 Video Tracking

### phase5_robotics_vision

- **Lesson 57 —** RGB-D Computer Vision
- **Lesson 58 —** Point Clouds
- **Lesson 59 —** 3-D Object Detection
- **Lesson 60 —** 6-D Object Pose Estimation
- **Lesson 61 —** PnP and Perspective Geometry
- **Lesson 62 —** Hand-Eye Calibration
- **Lesson 63 —** Visual Servoing
- **Lesson 64 —** Visual Odometry
- **Lesson 65 —** Visual SLAM
- **Lesson 66 —** ORB-SLAM Fundamentals
- **Lesson 67 —** Loop Closure Detection
- **Lesson 68 —** Scene Understanding

### phase6_model_optimization

- **Lesson 69 —** Model Profiling
- **Lesson 70 —** CPU vs GPU Inference
- **Lesson 71 —** Batch Size Optimization
- **Lesson 72 —** Mixed Precision FP16
- **Lesson 73 —** INT8 Quantization
- **Lesson 74 —** Model Pruning
- **Lesson 75 —** Knowledge Distillation
- **Lesson 76 —** ONNX Model Export
- **Lesson 77 —** ONNX Runtime Optimization
- **Lesson 78 —** NVIDIA TensorRT
- **Lesson 79 —** Dynamic Batching
- **Lesson 80 —** Asynchronous Inference
- **Lesson 81 —** Memory Optimization
- **Lesson 82 —** Model Benchmarking
- **Lesson 83 —** Edge AI Deployment

### phase7_foundation_models

- **Lesson 84 —** Vision Foundation Models
- **Lesson 85 —** CLIP
- **Lesson 86 —** DINOv2
- **Lesson 87 —** Self-Supervised Learning
- **Lesson 88 —** Vision-Language Models
- **Lesson 89 —** Florence-2
- **Lesson 90 —** Grounding DINO
- **Lesson 91 —** SAM 2 Advanced
- **Lesson 92 —** Video Understanding Transformers
- **Lesson 93 —** Anomaly Detection
- **Lesson 94 —** Few-Shot and Zero-Shot Learning
- **Lesson 95 —** Visual Question Answering
- **Lesson 96 —** Vision-Language-Action

### phase8_production

- **Lesson 97 —** Real-Time Multi-Camera Architecture
- **Lesson 98 —** Video Streaming with OpenCV
- **Lesson 99 —** RTSP Camera Processing
- **Lesson 100 —** Multi-Threaded Vision Pipelines
- **Lesson 101 —** GPU Pipeline Optimization
- **Lesson 102 —** FastAPI Vision Services
- **Lesson 103 —** Docker Deployment
- **Lesson 104 —** Computer Vision Monitoring
- **Lesson 105 —** Dataset Management and Versioning
- **Lesson 106 —** Model Evaluation and Regression Testing

## Core Learning Path

```text
01–10  Image Fundamentals
   ↓
11–20  Classical Computer Vision
   ↓
21–30  CNNs + Detection
   ↓
31–40  Modern Detection + Segmentation
   ↓
41–56  Tracking + Visual Memory
   ↓
57–68  3-D + Robotics Vision
   ↓
69–83  Optimization + Deployment
   ↓
84–96  Foundation Models + VLM/VLA
   ↓
97–106 Production Computer Vision
```

## Recommended Projects

### Project 1 — Visual Inspection
**Lessons 1–10**

`Camera → preprocessing → color/edge detection → measurement`

### Project 2 — Classical Object Tracker
**Lessons 11–20**

`Segmentation → contours → optical flow → tracking`

### Project 3 — AI Object Detector
**Lessons 21–30**

`Dataset → CNN → YOLO → object detection`

### Project 4 — Multi-Object Tracking
**Lessons 31–56**

`YOLO → Kalman Filter → association → modern tracking`

### Project 5 — 3-D Robot Perception
**Lessons 57–68**

`RGB-D → point cloud → 6-D pose → robot coordinates`

### Project 6 — Vision Foundation Model
**Lessons 84–96**

`CLIP/DINO/SAM → visual representation → language grounding`

### Project 7 — Robot Vision System
**Lessons 63–68 + 84–96**

`Camera → perception → 3-D localization → robot policy`

### Project 8 — Production Vision System
**Lessons 97–106**

`Multi-camera → RTSP → GPU pipeline → API → monitoring`

## Final Robotics Project — AI Waste Sorting

```text
RGB Cameras
    ↓
Visual Perception
    ├── Detection
    ├── Segmentation
    └── Tracking
    ↓
3-D Position / Object Pose
    ↓
Robot Coordinates
    ↓
Task Planning
    ↓
ACT / VLA
    ↓
OpenArm V2
    ↓
Pick + Sort
```

## Final Competency

By the end of the course, you should be able to design a multi-camera AI perception system that detects, segments, tracks, localizes, and classifies objects in 3-D, then supplies target information to a robot manipulation policy in real time.

Key competencies:
- 2-D computer vision
- OpenCV and classical vision
- CNNs and Vision Transformers
- Object detection and segmentation
- Multi-object tracking
- Camera calibration and geometry
- RGB-D and point clouds
- 6-D pose estimation
- Visual servoing and SLAM
- Vision foundation models
- VLM and VLA concepts
- GPU optimization and deployment
- Real-time multi-camera systems
- Production monitoring and testing
