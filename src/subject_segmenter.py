import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

def segment_and_track(frames_dir, output_dir):
    print("[INFO] Loading YOLOv8 Segmentation & ByteTrack Model...")
    # Using the segmentation model
    model = YOLO('yolov8n-seg.pt') 

    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    if not frames:
        print("[ERROR] No frames found in input directory.")
        return
        
    print(f"[INFO] Processing and tracking {len(frames)} frames...")
    
    for frame_name in frames:
        img_path = os.path.join(frames_dir, frame_name)
        img = cv2.imread(img_path)
        
        # Run YOLOv8 tracking (persist=True keeps memory of objects across frames)
        # classes=[0] ensures it only tracks people (class 0). Remove this to track cars, animals, etc.
        results = model.track(img, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)
        
        # Check if anything was detected and tracked in the current frame
        if results[0].boxes is not None and results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            h, w = img.shape[:2]
            
            for mask, track_id in zip(masks, track_ids):
                # Resize YOLO's low-res mask back to original image size
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Create the transparent alpha channel
                alpha = (mask_resized * 255).astype(np.uint8)
                
                # Combine original RGB with Alpha to make RGBA
                b, g, r = cv2.split(img)
                rgba = cv2.merge((b, g, r, alpha))
                
                # Create a specific folder for this Tracked ID (e.g., subject_1, subject_2)
                id_dir = os.path.join(output_dir, f"subject_{track_id}")
                os.makedirs(id_dir, exist_ok=True)
                
                # Save the frame
                out_path = os.path.join(id_dir, frame_name)
                cv2.imwrite(out_path, rgba)
                
        print(f"  -> Tracked and extracted: {frame_name}")
        
    print(f"\n[SUCCESS] Multi-Subject Tracking complete! Check the '{output_dir}' folder for individual subject directories.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolate and track multiple subjects using YOLOv8 + ByteTrack.")
    parser.add_argument("--frames", required=True, help="Path to input frames")
    parser.add_argument("--output", required=True, help="Path to save tracked subject folders")
    
    args = parser.parse_args()
    segment_and_track(args.frames, args.output)
