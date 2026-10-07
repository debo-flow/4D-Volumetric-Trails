# 🌌 4D-Volumetric-Trails

An open-source computer vision pipeline designed to extract moving subjects from 2D monocular video and project them into a reconstructed 4D spacetime volumetric space (3D Space + Time). 

![Version](https://img.shields.io/badge/version-v3.0.0--Boss--Level-purple)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/debo-flow/4D-Volumetric-Trails/blob/main/4D_Volumetric_Trails.ipynb)

## 🚀 Project Overview

The goal of this project is to recreate the "Spacetime reconstruction in 4D" effect computationally. By processing a standard moving-camera video, the pipeline tracks the camera's 3D path, reconstructs the static environment, isolates the moving subject, generates a 2.5D volumetric depth map, applies cinematic camera smoothing, and prepares datasets for True 4D Gaussian Splatting (Dynamic 3DGS).

### 🧠 Complete Pipeline Architecture
1. **Video Processor (`src/video_processor.py`):** Lossless extraction of high-quality PNG frames from standard `.mp4` video inputs.
2. **Camera Tracker (`src/camera_tracker.py`):** Automated Structure-from-Motion (SfM) using COLMAP to extract precise 3D camera coordinates.
3. **Subject Segmenter (`src/subject_segmenter.py`):** AI-powered isolation (YOLOv8) of the moving subject into transparent `.png` layers.
4. **Depth Estimator (`src/depth_estimator.py`):** Uses Intel's MiDaS AI (HuggingFace Transformers) to generate 2.5D depth maps from the segmented frames.
5. **Background Reconstructor (`src/background_reconstructor.py`):** Automated 3D Gaussian Splatting (3DGS) training via Nerfstudio.
6. **Camera Smoother (`src/camera_smoother.py`):** Applies a Savitzky-Golay filter to the camera trajectory to remove handheld jitter and create a cinematic drone-like flow.
7. **4D Data Preparer (`src/prepare_4d_data.py`):** Injects normalized temporal metadata (T) into spatial coordinates for True 4D Dynamic Gaussian Splatting training.
8. **Space-Time Projector (`src/spacetime_projector.py`):** Maps the 2.5D segmented frames into 3D space using smoothed camera coordinates, exporting a `.glb` 3D trail.
9. **Interactive Web Viewer (`src/web_viewer.py`):** A Gradio-based local web application to visualize, rotate, and interact with the 4D Spacetime trail.

---

## 🛠️ Installation

**Prerequisites:**
- Python 3.10+
- Nvidia GPU (Highly recommended for YOLOv8, COLMAP, MiDaS, and Nerfstudio)
- [COLMAP](https://colmap.github.io/install.html) installed and added to system PATH.
- [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) installed for 3D Background Reconstruction.

**Setup:**

```bash
git clone https://github.com/debo-flow/4D-Volumetric-Trails.git
cd 4D-Volumetric-Trails
pip install -r requirements.txt
```

---

## 💻 Usage (Step-by-Step)

Ensure you have your input video placed in `data/input/`. You can copy and run the following commands sequentially:

```bash
# Step 1: Extract Frames from Video
python src/video_processor.py --video data/input/your_video.mp4 --output data/frames

# Step 2: Track Camera Poses using COLMAP
python src/camera_tracker.py --images data/frames --output data/colmap

# Step 3: Isolate Moving Subjects using YOLOv8
python src/subject_segmenter.py --frames data/frames --output data/segmented

# Step 4: Extract 2.5D Depth Maps using MiDaS AI
python src/depth_estimator.py --segmented data/segmented --output data/depth

# Step 5: Reconstruct 3D Background using Nerfstudio (Optional)
python src/background_reconstructor.py --images data/frames --colmap data/colmap --output data/3d_model

# Step 6: Apply Cinematic Camera Smoothing (Drone Effect)
python src/camera_smoother.py --input data/3d_model/ns_data/transforms.json --output data/3d_model/ns_data/transforms_smoothed.json

# Step 7: Inject Time Dimension for True 4D GS (Boss Level)
python src/prepare_4d_data.py --input data/3d_model/ns_data/transforms_smoothed.json --output data/3d_model/ns_data/transforms_4d.json

# Step 8: Generate 4D Spacetime Trail
python src/spacetime_projector.py --transforms data/3d_model/ns_data/transforms_4d.json --segmented data/segmented --depth data/depth --output data/output/spacetime_trail.glb

# Step 9: Launch Interactive Web Viewer
python src/web_viewer.py --model data/output/spacetime_trail.glb
```

---

## 🗺️ Project Milestones
- [x] **v0.1.0-alpha:** Core Data Processing Pipeline (Frame extraction, COLMAP tracking, YOLOv8 segmentation).
- [x] **v0.2.0-alpha:** Background Environment Reconstruction using 3DGS.
- [x] **v0.3.0-alpha:** Space-Time projection and dynamic `.glb` generation.
- [x] **v1.0.0:** Final interactive 4D visualization UI via web browser.
- [x] **v2.0.0-alpha:** 2.5D Volumetric Depth Estimation (MiDaS integration).
- [x] **v2.1.0-alpha:** Cinematic Camera Trajectory Smoothing (Savitzky-Golay filter).
- [x] **v2.2.0-alpha:** Google Colab integration for mass audience cloud execution.
- [x] **v3.0.0-Boss-Level:** Temporal Data Injection (T-dimension) for True 4D Gaussian Splatting capabilities.

## 📝 License
Distributed under the MIT License.
