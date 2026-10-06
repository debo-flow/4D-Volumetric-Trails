import cv2
import os
import argparse
import numpy as np
from ultralytics import YOLO

def segment_subject(frames_dir, output_dir, model_path="yolov8x-seg.pt", confidence_threshold=0.5):
    """
    Uses YOLOv8 Segmentation to isolate moving subjects from video frames.
    The output will be transparent PNGs containing only the subject.
    """
    if not os.path.exists(frames_dir):
        print(f"[ERROR] Frames directory not found: {frames_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLOv8 segmentation model
    print(f"[INFO] Loading YOLOv8 Segmentation Model ({model_path})...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Get sorted list of frames
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    if not frame_files:
        print(f"[ERROR] No frames found in {frames_dir}")
        return

    print(f"[INFO] Found {len(frame_files)} frames. Starting segmentation...")

    for count, frame_name in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        
        # Run inference on the frame
        results = model(frame, conf=confidence_threshold, verbose=False)
        
        # Create a transparent background (alpha channel)
        transparent_bg = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
        
        if results[0].masks is not None:
            # Get the mask of the primary subject (assuming largest/most confident detection)
            # You might need to adjust logic here if there are multiple subjects you want to track
            mask = results[0].masks.data[0].cpu().numpy()
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            
            # Convert mask to 3 channels for easier masking of original frame
            mask_3ch = np.stack([mask]*3, axis=2)
            
            # Extract subject pixels
            subject_rgb = np.where(mask_3ch > 0.5, frame, 0)
            
            # Create alpha channel where mask > 0.5 is solid (255), else transparent (0)
            alpha_channel = np.where(mask > 0.5, 255, 0).astype(np.uint8)
            
            # Combine RGB and Alpha
            extracted_subject = np.dstack((subject_rgb, alpha_channel))
        else:
            # If no subject found, save a completely transparent frame
            extracted_subject = transparent_bg
            
        # Save as PNG to preserve transparency
        output_path = os.path.join(output_dir, f"seg_{frame_name}")
        # Note: We save as PNG even if input was JPG to keep alpha channel
        if not output_path.endswith('.png'):
             output_path = os.path.splitext(output_path)[0] + ".png"
        cv2.imwrite(output_path, extracted_subject)
        
        if (count + 1) % 50 == 0:
            print(f"[INFO] Segmented {count + 1} frames...")
            
    print(f"[SUCCESS] Subject segmentation complete. Output saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract moving subjects using YOLOv8 Segmentation.")
    parser.add_argument("--frames", type=str, default="data/frames", help="Path to input frames directory")
    parser.add_argument("--output", type=str, default="data/segmented", help="Directory to save segmented subject PNGs")
    parser.add_argument("--model", type=str, default="yolov8x-seg.pt", help="YOLO segmentation model path/name (e.g., yolov8n-seg.pt for faster, yolov8x-seg.pt for accurate)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detection")
    
    args = parser.parse_args()
    
    segment_subject(args.frames, args.output, args.model, args.conf)
