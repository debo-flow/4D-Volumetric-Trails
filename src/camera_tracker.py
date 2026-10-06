import os
import subprocess
import argparse
import sys

def run_command(cmd, desc):
    """Executes a system shell command with proper error handling."""
    print(f"\n[INFO] {desc}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Failed during: {desc}")
        sys.exit(1)

def track_cameras(images_dir, output_dir, colmap_bin="colmap"):
    """
    Automates the COLMAP Structure-from-Motion (SfM) pipeline:
    1. Feature Extraction (SIFT)
    2. Sequential Feature Matching (optimized for video frames)
    3. Sparse 3D Reconstruction / Mapping
    4. Model conversion to plain text (.txt) for easy 3D rendering
    """
    if not os.path.exists(images_dir):
        print(f"[ERROR] Image directory not found: {images_dir}")
        return

    # Create workspace directories
    sparse_dir = os.path.join(output_dir, "sparse")
    database_path = os.path.join(output_dir, "database.db")
    os.makedirs(sparse_dir, exist_ok=True)

    # 1. Feature Extraction
    # Single camera shared across all video frames for stability
    extract_cmd = (
        f"{colmap_bin} feature_extractor "
        f"--database_path {database_path} "
        f"--image_path {images_dir} "
        f"--ImageReader.single_camera 1 "
        f"--ImageReader.camera_model SIMPLE_RADIAL"
    )
    run_command(extract_cmd, "Extracting visual features (SIFT)")

    # 2. Sequential Matching (Video frames are sequential, much faster than exhaustive)
    match_cmd = (
        f"{colmap_bin} sequential_matcher "
        f"--database_path {database_path} "
        f"--SequentialMatching.overlap 20"
    )
    run_command(match_cmd, "Matching features between adjacent frames")

    # 3. Sparse Reconstruction (Mapper)
    map_cmd = (
        f"{colmap_bin} mapper "
        f"--database_path {database_path} "
        f"--image_path {images_dir} "
        f"--output_path {sparse_dir}"
    )
    run_command(map_cmd, "Calculating 3D camera poses and sparse point cloud")

    # 4. Convert Binary model to Text format (so our visualizer can read camera coordinates)
    model_dir = os.path.join(sparse_dir, "0")
    if os.path.exists(model_dir):
        txt_output_dir = os.path.join(output_dir, "sparse_txt")
        os.makedirs(txt_output_dir, exist_ok=True)
        convert_cmd = (
            f"{colmap_bin} model_converter "
            f"--input_path {model_dir} "
            f"--output_path {txt_output_dir} "
            f"--output_type TXT"
        )
        run_command(convert_cmd, "Converting camera poses to human-readable format")
        print(f"\n[SUCCESS] Camera poses successfully generated in: {txt_output_dir}")
        print(f"[INFO] 'images.txt' contains exact 3D coordinates (x, y, z) and orientation (quaternions) of the camera!")
    else:
        print("[WARNING] Reconstruction finished, but no model found in sparse/0.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated COLMAP Camera Tracker for 4D Pipeline.")
    parser.add_argument("--images", type=str, default="data/frames", help="Path to extracted frames folder")
    parser.add_argument("--output", type=str, default="data/colmap", help="Directory to save camera poses and database")
    parser.add_argument("--colmap_bin", type=str, default="colmap", help="COLMAP binary executable name or path")

    args = parser.parse_args()
    track_cameras(args.images, args.output, args.colmap_bin)
