# 🌌 4D-Volumetric-Trails

An open-source computer vision and spatiotemporal neural rendering suite designed to extract multiple moving subjects from 2D monocular video and project them into a reconstructed 4D spacetime volumetric space (3D Space + Time). 

![Version](https://img.shields.io/badge/version-v4.0.0--Ultimate--Studio-blueviolet)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/debo-flow/4D-Volumetric-Trails/blob/main/4D_Volumetric_Trails.ipynb)

---

## 🌟 What's New in v4.0.0 (The Ultimate Studio Edition)?

This monumental release transforms **4D-Volumetric-Trails** from a command-line research pipeline into a complete, production-ready spatiotemporal studio. You can now reconstruct multi-person scenes, compress heavy 3D meshes automatically, and run the entire workflow through an interactive **No-Code Web GUI**.

* **🖥️ 1-Click No-Code Web Studio (`src/web_viewer.py`):** Upgraded into a full **3-Tab Gradio Studio** (*Pipeline Auto-Builder*, *Interactive 4D Viewer*, and *Cinematic MP4 Exporter*) with real-time terminal logs streaming in the browser.
* **👥 Multi-Subject Tracking (`src/subject_segmenter.py`):** Integrated **ByteTrack** with **YOLOv8-seg** to isolate and track multiple moving subjects simultaneously with persistent IDs (`subject_1`, `subject_2`, etc.) into a shared 4D scene.
* **⚡ AI Mesh Optimizer (`src/mesh_optimizer.py`):** Implemented **Open3D Quadric Error Metric Decimation** to compress high-poly 4D `.glb` meshes by **75%–80%** without sacrificing visual quality.
* **🎬 Cinematic 1080p MP4 Turntable Exporter (`src/video_exporter.py`):** Headless Open3D + OpenCV rendering engine that orbits a virtual camera $360^\circ$ around your 4D trail to export a ready-to-share **1080p MP4 showcase**.
* **🐳 Enterprise Docker Containerization (`Dockerfile` & `.dockerignore`):** Full containerization for reproducible one-command local or cloud deployment.

---

## 🚀 Project Overview

The goal of this project is to recreate the "Spacetime reconstruction in 4D" effect computationally. By processing a standard moving-camera video, the pipeline tracks the camera's 3D path, reconstructs the static environment, isolates and tracks multiple moving subjects, generates 2.5D volumetric depth maps, applies cinematic camera smoothing, prepares datasets for True 4D Gaussian Splatting (Dynamic 3DGS), optimizes 3D polygon counts, and exports interactive `.glb` models alongside 1080p MP4 showcases.

### 🧠 Complete Pipeline Architecture
1. **Video Processor (`src/video_processor.py`):** Lossless extraction of high-quality PNG frames from standard `.mp4` video inputs.
2. **Camera Tracker (`src/camera_tracker.py`):** Automated Structure-from-Motion (SfM) using COLMAP to extract precise 3D camera coordinates.
3. **Multi-Subject Segmenter (`src/subject_segmenter.py`):** AI-powered multi-object isolation and persistent ID tracking using YOLOv8 + ByteTrack into transparent `.png` layers.
4. **Depth Estimator (`src/depth_estimator.py`):** Uses Intel's MiDaS AI (HuggingFace Transformers) to generate multi-subject 2.5D depth maps from segmented frames.
5. **Background Reconstructor (`src/background_reconstructor.py`):** Automated 3D Gaussian Splatting (3DGS) training via Nerfstudio.
6. **Camera Smoother (`src/camera_smoother.py`):** Applies a Savitzky-Golay filter to the camera trajectory to remove handheld jitter and create a cinematic drone-like flow.
7. **4D Data Preparer (`src/prepare_4d_data.py`):** Injects normalized temporal metadata ($T$) into spatial coordinates for True 4D Dynamic Gaussian Splatting training.
8. **Space-Time Projector (`src/spacetime_projector.py`):** Maps multiple 2.5D segmented subjects into a shared 3D space using smoothed camera coordinates, exporting a combined `.glb` 3D trail.
9. **Mesh Optimizer (`src/mesh_optimizer.py`):** Uses Open3D Quadric Decimation to reduce polygon counts by 75%–80% for smooth browser rendering.
10. **Video Exporter (`src/video_exporter.py`):** Renders a cinematic $360^\circ$ 1080p MP4 turntable animation of the 4D trail.
11. **No-Code Web Studio (`src/web_viewer.py`):** A full Gradio GUI to execute the pipeline, inspect 3D models, and render MP4 videos with one click.

---

## 🛠️ Installation

**Prerequisites:**
- Python 3.10+
- Nvidia GPU (Highly recommended for YOLOv8, COLMAP, MiDaS, and Nerfstudio)
- [COLMAP](https://colmap.github.io/install.html) installed and added to system PATH.
- [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) installed for 3D Background Reconstruction.

**Standard Setup:**

```bash
git clone https://github.com/debo-flow/4D-Volumetric-Trails.git
cd 4D-Volumetric-Trails
pip install -r requirements.txt
```

**Docker Setup (For Enterprise/Isolated Environments):**

```bash
docker build -t 4d-volumetric-trails .
docker run --gpus all -p 7860:7860 -v $(pwd)/data:/app/data 4d-volumetric-trails
```

---

## 💻 Usage

### ⚡ Option A: One-Click No-Code Web Studio (Recommended)
Launch the interactive web studio to upload videos, run the automated pipeline, view 3D models, and export MP4s directly from your browser without typing CLI commands:

```bash
python src/web_viewer.py
```
*(Open `http://127.0.0.1:7860` in your browser).*

### 🔧 Option B: Step-by-Step CLI Execution
Ensure you have your input video placed in `data/input/`. You can copy and run the following commands sequentially:

```bash
# Step 1: Extract Frames from Video
python src/video_processor.py --video data/input/your_video.mp4 --output data/frames

# Step 2: Track Camera Poses using COLMAP
python src/camera_tracker.py --images data/frames --output data/colmap

# Step 3: Isolate & Track Multiple Moving Subjects using YOLOv8 + ByteTrack
python src/subject_segmenter.py --frames data/frames --output data/segmented

# Step 4: Extract Multi-Subject 2.5D Depth Maps using MiDaS AI
python src/depth_estimator.py --segmented data/segmented --output data/depth

# Step 5: Reconstruct 3D Background using Nerfstudio (Optional)
python src/background_reconstructor.py --images data/frames --colmap data/colmap --output data/3d_model

# Step 6: Apply Cinematic Camera Smoothing (Drone Effect)
python src/camera_smoother.py --input data/3d_model/ns_data/transforms.json --output data/3d_model/ns_data/transforms_smoothed.json

# Step 7: Inject Time Dimension for True 4D GS (Boss Level)
python src/prepare_4d_data.py --input data/3d_model/ns_data/transforms_smoothed.json --output data/3d_model/ns_data/transforms_4d.json

# Step 8: Generate Raw Multi-Subject 4D Spacetime Trail
python src/spacetime_projector.py --transforms data/3d_model/ns_data/transforms_4d.json --segmented data/segmented --depth data/depth --output data/output/spacetime_trail_raw.glb

# Step 9: Optimize & Compress 3D Mesh for Smooth Viewing
python src/mesh_optimizer.py --input data/output/spacetime_trail_raw.glb --output data/output/spacetime_trail.glb --reduction 0.75

# Step 10: Export Cinematic 1080p MP4 Turntable Video
python src/video_exporter.py --model data/output/spacetime_trail.glb --output data/output/final_showcase.mp4

# Step 11: Launch Interactive Web Studio
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
- [x] **v3.0.0-Boss-Level:** Temporal Data Injection ($T$-dimension) for True 4D Gaussian Splatting capabilities.
- [x] **v4.0.0-Ultimate-Studio:** Multi-Subject Tracking (ByteTrack), Quadric Mesh Optimization, 1080p MP4 Turntable Exporter, Docker Containerization, and Full No-Code Gradio Web Studio.

## 📝 License
Distributed under the MIT License.