import json
import argparse
import os

def add_time_to_transforms(input_json, output_json):
    print(f"[INFO] Loading 3D transforms from: {input_json}")
    with open(input_json, 'r') as f:
        data = json.load(f)
        
    frames = data.get("frames", [])
    total_frames = len(frames)
    
    if total_frames == 0:
        print("[ERROR] No frames found in transforms.json")
        return
        
    # Sort frames by filename to ensure temporal order
    frames = sorted(frames, key=lambda x: x["file_path"])
    
    print(f"[INFO] Injecting Time (T) dimension into {total_frames} frames...")
    
    for i, frame in enumerate(frames):
        # Normalize time between 0.0 and 1.0 for the 4D Neural Network
        time_val = i / float(total_frames - 1)
        frame["time"] = time_val
        frame["frame_index"] = i
        
    data["frames"] = frames
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"[SUCCESS] 4D-ready transforms saved to: {output_json}")
    print("[INFO] Your dataset is now ready for True 4D Gaussian Splatting (e.g., K-Planes or Dynamic 3DGS)!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for True 4D Gaussian Splatting by injecting time metadata.")
    parser.add_argument("--input", required=True, help="Path to original transforms.json")
    parser.add_argument("--output", required=True, help="Path to save 4D transforms_time.json")
    
    args = parser.parse_args()
    add_time_to_transforms(args.input, args.output)
