import os
import cv2
import torch
import numpy as np
import argparse
from transformers import pipeline
from PIL import Image

def estimate_depth(input_base, output_base):
    device = 0 if torch.cuda.is_available() else -1
    print("[INFO] Loading Depth Estimation Model (Intel/dpt-large)...")
    depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large", device=device)
    
    # Find all subject directories
    subjects = [d for d in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, d))]
    
    if not subjects:
        print(f"[ERROR] No subject directories found in {input_base}")
        return
        
    for subject in subjects:
        print(f"\n[INFO] Processing Depth Maps for: {subject}")
        subject_in_dir = os.path.join(input_base, subject)
        subject_out_dir = os.path.join(output_base, subject)
        os.makedirs(subject_out_dir, exist_ok=True)
        
        frames = sorted([f for f in os.listdir(subject_in_dir) if f.endswith('.png')])
        for frame_name in frames:
            img_path = os.path.join(subject_in_dir, frame_name)
            
            image = Image.open(img_path).convert("RGBA")
            rgb_image = image.convert("RGB")
            
            depth_output = depth_estimator(rgb_image)
            depth_map = depth_output["depth"]
            depth_array = np.array(depth_map)
            
            depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            alpha_channel = np.array(image)[:, :, 3]
            depth_normalized[alpha_channel == 0] = 0 
            
            out_path = os.path.join(subject_out_dir, frame_name)
            cv2.imwrite(out_path, depth_normalized)
            print(f"  -> Generated depth for: {subject}/{frame_name}")
            
    print("\n[SUCCESS] Multi-Subject Depth Estimation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract depth maps for multiple segmented subjects.")
    parser.add_argument("--segmented", required=True, help="Base path to segmented subject folders")
    parser.add_argument("--output", required=True, help="Base path to save generated depth maps")
    
    args = parser.parse_args()
    estimate_depth(args.segmented, args.output)
