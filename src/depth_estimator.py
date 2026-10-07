import os
import cv2
import torch
import numpy as np
import argparse
from transformers import pipeline
from PIL import Image

def estimate_depth(input_dir, output_dir):
    """
    Generates depth maps from segmented 2D images using MiDaS AI.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("[INFO] Loading Depth Estimation Model (Intel/dpt-large)...")
    # Automatically use GPU if available
    device = 0 if torch.cuda.is_available() else -1
    
    # Load HuggingFace Depth Estimation Pipeline
    depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large", device=device)
    
    frames = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    if not frames:
        print("[ERROR] No .png frames found in the input directory.")
        return
        
    print(f"[INFO] Processing {len(frames)} frames for depth extraction...")
    
    for frame_name in frames:
        img_path = os.path.join(input_dir, frame_name)
        
        # Read image with alpha channel (RGBA)
        image = Image.open(img_path).convert("RGBA")
        
        # Convert to RGB for the depth model
        rgb_image = image.convert("RGB")
        
        # Estimate depth
        depth_output = depth_estimator(rgb_image)
        depth_map = depth_output["depth"]
        
        # Convert output to numpy array
        depth_array = np.array(depth_map)
        
        # Normalize depth map to 0-255 (Grayscale)
        depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Mask out the background using the original alpha channel
        # We only want depth on the subject, background should be flat (0)
        alpha_channel = np.array(image)[:, :, 3]
        depth_normalized[alpha_channel == 0] = 0 
        
        # Save the depth map
        out_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(out_path, depth_normalized)
        print(f"  -> Generated depth map for: {frame_name}")
        
    print("[INFO] 2.5D Depth estimation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract depth maps from segmented images.")
    parser.add_argument("--segmented", type=str, required=True, help="Path to segmented input frames")
    parser.add_argument("--output", type=str, required=True, help="Path to save generated depth maps")
    
    args = parser.parse_args()
    estimate_depth(args.segmented, args.output)
