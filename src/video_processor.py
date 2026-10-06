import cv2
import os
import argparse

def extract_frames(video_path, output_dir):
    """
    Extracts frames from a video file and saves them as high-quality PNGs.
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Input video not found at: {video_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the video
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    
    print(f"[INFO] Starting frame extraction from '{video_path}'...")
    print(f"[INFO] Saving frames to '{output_dir}'...")
    
    while success:
        # Save as PNG for lossless quality (crucial for COLMAP & 3DGS)
        frame_path = os.path.join(output_dir, f"frame_{count:05d}.png")
        cv2.imwrite(frame_path, image)
        
        success, image = vidcap.read()
        count += 1
        
        if count % 50 == 0:
            print(f"[INFO] Processed {count} frames...")
            
    vidcap.release()
    print(f"[SUCCESS] Extracted a total of {count} frames.")

if __name__ == "__main__":
    # Command-line argument setup for professional execution
    parser = argparse.ArgumentParser(description="Extract frames for Spacetime Reconstruction Pipeline.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file (e.g., data/input/video.mp4)")
    parser.add_argument("--output", type=str, default="data/frames", help="Directory to save extracted frames")
    
    args = parser.parse_args()
    
    extract_frames(args.video, args.output)
