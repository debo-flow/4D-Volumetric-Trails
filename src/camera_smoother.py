import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
import os

def smooth_camera_poses(input_json, output_json, window_size=11, poly_order=3):
    print(f"[INFO] Loading camera transforms from: {input_json}")
    with open(input_json, 'r') as f:
        data = json.load(f)
        
    frames = data.get("frames", [])
    if len(frames) < window_size:
        print(f"[WARNING] Not enough frames ({len(frames)}) to smooth. Saving original.")
        with open(output_json, 'w') as f:
            json.dump(data, f, indent=4)
        return

    # Extract translations (positions) and rotations (angles)
    translations = []
    rotations = []
    
    for frame in frames:
        matrix = np.array(frame["transform_matrix"])
        translations.append(matrix[:3, 3])
        rotations.append(R.from_matrix(matrix[:3, :3]).as_quat())
        
    translations = np.array(translations)
    rotations = np.array(rotations)
    
    print("[INFO] Applying Savitzky-Golay filter to smooth camera trajectory...")
    
    # Smooth XYZ Positions
    smooth_tx = savgol_filter(translations[:, 0], window_size, poly_order)
    smooth_ty = savgol_filter(translations[:, 1], window_size, poly_order)
    smooth_tz = savgol_filter(translations[:, 2], window_size, poly_order)
    
    # Smooth Rotations (Quaternions)
    print("[INFO] Stabilizing camera rotations...")
    smooth_rx = savgol_filter(rotations[:, 0], window_size, poly_order)
    smooth_ry = savgol_filter(rotations[:, 1], window_size, poly_order)
    smooth_rz = savgol_filter(rotations[:, 2], window_size, poly_order)
    smooth_rw = savgol_filter(rotations[:, 3], window_size, poly_order)
    
    smooth_rotations = np.vstack((smooth_rx, smooth_ry, smooth_rz, smooth_rw)).T
    
    # Normalize quaternions to keep rotations mathematically valid
    norms = np.linalg.norm(smooth_rotations, axis=1, keepdims=True)
    smooth_rotations = smooth_rotations / norms
    
    print(f"[INFO] Saving cinematic (smoothed) transforms to: {output_json}")
    
    # Apply smoothed data back to the matrices
    for i, frame in enumerate(frames):
        rot_matrix = R.from_quat(smooth_rotations[i]).as_matrix()
        trans = [smooth_tx[i], smooth_ty[i], smooth_tz[i]]
        
        new_matrix = np.eye(4)
        new_matrix[:3, :3] = rot_matrix
        new_matrix[:3, 3] = trans
        frame["transform_matrix"] = new_matrix.tolist()
        
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)
        
    print("[SUCCESS] Cinematic camera smoothing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smooth COLMAP/Nerfstudio camera poses for cinematic drone effect.")
    parser.add_argument("--input", required=True, help="Input transforms.json")
    parser.add_argument("--output", required=True, help="Output smoothed_transforms.json")
    parser.add_argument("--window", type=int, default=11, help="Smoothing window size (must be odd, e.g., 11, 15, 21)")
    
    args = parser.parse_args()
    w_size = args.window if args.window % 2 != 0 else args.window + 1
    smooth_camera_poses(args.input, args.output, window_size=w_size)
