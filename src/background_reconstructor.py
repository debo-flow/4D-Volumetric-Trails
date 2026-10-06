import os
import subprocess
import argparse
import sys

def run_command(cmd, desc):
    """Executes a shell command and handles errors."""
    print(f"\n[INFO] {desc}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Failed during: {desc}")
        print("[HINT] Ensure Nerfstudio is installed and activated in your environment.")
        sys.exit(1)

def reconstruct_3d_background(images_dir, colmap_dir, output_dir):
    """
    Automates the 3D Gaussian Splatting training process using Nerfstudio.
    1. Converts COLMAP data to Nerfstudio format.
    2. Trains a 3D Gaussian Splatting (splatfacto) model.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if COLMAP data exists from Milestone 1
    if not os.path.exists(colmap_dir):
        print(f"[ERROR] COLMAP data not found at {colmap_dir}. Please run camera_tracker.py first.")
        return

    # 1. Format Data for Nerfstudio
    # Nerfstudio needs the data in a specific JSON format based on COLMAP
    nerfstudio_data_dir = os.path.join(output_dir, "ns_data")
    ns_process_cmd = (
        f"ns-process-data images "
        f"--data {images_dir} "
        f"--output-dir {nerfstudio_data_dir}"
    )
    # Note: If ns-process-data runs COLMAP again, we can skip it, but for standard pipelines, 
    # letting Nerfstudio process the images ensures perfect alignment.
    run_command(ns_process_cmd, "Formatting data for Nerfstudio (Transforms JSON)")

    # 2. Train the 3D Gaussian Splatting Model (splatfacto)
    # This will train the 3D environment and save the output.
    print(f"\n[INFO] Starting 3D Gaussian Splatting Training...")
    print(f"[INFO] This requires a strong GPU and may take 15-30 minutes depending on hardware.")
    
    train_cmd = (
        f"ns-train splatfacto "
        f"--data {nerfstudio_data_dir} "
        f"--viewer.quit-on-train-completion True"
    )
    run_command(train_cmd, "Training 3D Background Model")

    print("\n[SUCCESS] 3D Background Reconstruction Complete!")
    print(f"[INFO] The trained 3D model is saved in the Nerfstudio outputs directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct 3D Background using Gaussian Splatting.")
    parser.add_argument("--images", type=str, default="data/frames", help="Path to extracted frames directory")
    parser.add_argument("--colmap", type=str, default="data/colmap", help="Path to existing COLMAP tracking data")
    parser.add_argument("--output", type=str, default="data/3d_model", help="Directory to save 3D reconstruction data")
    
    args = parser.parse_args()
    
    reconstruct_3d_background(args.images, args.colmap, args.output)
