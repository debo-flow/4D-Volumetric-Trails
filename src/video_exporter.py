import os
import cv2
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm

def export_cinematic_video(glb_path, output_mp4, frames=120, fps=30):
    if not os.path.exists(glb_path):
        print(f"[ERROR] Model not found: {glb_path}")
        return

    print(f"[INFO] Loading 3D Model: {glb_path}")
    mesh = o3d.io.read_triangle_mesh(glb_path)
    mesh.compute_vertex_normals()

    # Set up headless visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    vis.add_geometry(mesh)

    # Set background to black (cinematic style)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    print(f"[INFO] Rendering {frames} frames for 360-degree turntable animation...")
    
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_mp4, fourcc, fps, (1920, 1080))

    ctr = vis.get_view_control()
    
    # Render loop
    for _ in tqdm(range(frames), desc="Rendering Video"):
        # Rotate camera smoothly around the subject
        ctr.rotate(10.0, 0.0) 
        vis.poll_events()
        vis.update_renderer()
        
        # Capture frame buffer
        image = vis.capture_screen_float_buffer(False)
        image = (np.asarray(image) * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image_bgr)

    vis.destroy_window()
    video_writer.release()
    print(f"[SUCCESS] Cinematic video saved to: {output_mp4}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export 3D Spacetime trail to a cinematic MP4 video.")
    parser.add_argument("--model", required=True, help="Path to input .glb file")
    parser.add_argument("--output", required=True, help="Path to output .mp4 file")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames for 360-degree rotation")
    
    args = parser.parse_args()
    export_cinematic_video(args.model, args.output, frames=args.frames)
