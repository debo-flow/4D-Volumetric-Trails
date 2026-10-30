import os
import shutil
import argparse
import subprocess
import gradio as gr

# Ensure base directories exist
DIRS = [
    "data/input", "data/frames", "data/colmap", 
    "data/segmented", "data/depth", "data/3d_model/ns_data", "data/output"
]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

def run_cmd(cmd_list, step_name):
    """Helper to execute pipeline scripts and capture logs."""
    log = f"=== [STARTING] {step_name} ===\nCommand: {' '.join(cmd_list)}\n"
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        log += result.stdout + "\n"
        log += f"✅ [SUCCESS] {step_name} completed!\n\n"
        return True, log
    except subprocess.CalledProcessError as e:
        log += f"❌ [ERROR] {step_name} failed!\n{e.stdout}\n{e.stderr}\n\n"
        return False, log

def run_full_pipeline(video_path, smooth_window, include_nerfstudio):
    if not video_path:
        return "⚠️ Error: Please upload an input video first!", None

    logs = ""
    target_video = "data/input/uploaded_video.mp4"
    shutil.copy(video_path, target_video)
    logs += f"📁 Video uploaded successfully to {target_video}\n\n"
    yield logs, None

    # Step 1: Extract Frames
    ok, out = run_cmd(["python", "src/video_processor.py", "--video", target_video, "--output", "data/frames"], "Step 1: Video Frame Extraction")
    logs += out
    yield logs, None
    if not ok: return

    # Step 2: Camera Tracking (COLMAP)
    ok, out = run_cmd(["python", "src/camera_tracker.py", "--images", "data/frames", "--output", "data/colmap"], "Step 2: COLMAP Camera Tracking")
    logs += out
    yield logs, None
    if not ok: return

    # Step 3: Multi-Subject Segmentation & Tracking (YOLOv8 + ByteTrack)
    ok, out = run_cmd(["python", "src/subject_segmenter.py", "--frames", "data/frames", "--output", "data/segmented"], "Step 3: Multi-Subject YOLOv8 Tracking")
    logs += out
    yield logs, None
    if not ok: return

    # Step 4: 2.5D Depth Estimation (MiDaS)
    ok, out = run_cmd(["python", "src/depth_estimator.py", "--segmented", "data/segmented", "--output", "data/depth"], "Step 4: MiDaS 2.5D Depth Estimation")
    logs += out
    yield logs, None
    if not ok: return

    # Step 5: Optional Nerfstudio 3D Background Reconstruction
    if include_nerfstudio:
        ok, out = run_cmd(["python", "src/background_reconstructor.py", "--images", "data/frames", "--colmap", "data/colmap", "--output", "data/3d_model"], "Step 5: Nerfstudio 3DGS Reconstruction")
        logs += out
        yield logs, None
        if not ok: return

    transforms_in = "data/3d_model/ns_data/transforms.json"
    if not os.path.exists(transforms_in):
        logs += f"⚠️ Warning: {transforms_in} not found. Ensure COLMAP/Nerfstudio generated transforms.json.\n"
        yield logs, None
        return

    # Step 6: Cinematic Camera Smoothing
    transforms_smooth = "data/3d_model/ns_data/transforms_smoothed.json"
    ok, out = run_cmd(["python", "src/camera_smoother.py", "--input", transforms_in, "--output", transforms_smooth, "--window", str(int(smooth_window))], "Step 6: Cinematic Camera Smoothing")
    logs += out
    yield logs, None
    if not ok: return

    # Step 7: True 4D Time Injection
    transforms_4d = "data/3d_model/ns_data/transforms_4d.json"
    ok, out = run_cmd(["python", "src/prepare_4d_data.py", "--input", transforms_smooth, "--output", transforms_4d], "Step 7: 4D Temporal Data Injection")
    logs += out
    yield logs, None
    if not ok: return

    # Step 8: 2.5D Spacetime Projection
    output_glb = "data/output/spacetime_trail.glb"
    ok, out = run_cmd(["python", "src/spacetime_projector.py", "--transforms", transforms_4d, "--segmented", "data/segmented", "--depth", "data/depth", "--output", output_glb], "Step 8: 4D Volumetric Mesh Projection")
    logs += out
    if ok and os.path.exists(output_glb):
        logs += "🎉 FULL PIPELINE COMPLETE! Your 4D model is loaded on the right."
        yield logs, output_glb
    else:
        yield logs, None

def render_mp4_showcase(glb_path, num_frames):
    if not glb_path or not os.path.exists(glb_path):
        return "⚠️ Error: .glb model not found! Run the pipeline or provide a valid path.", None
    
    output_mp4 = "data/output/final_showcase.mp4"
    ok, log = run_cmd(["python", "src/video_exporter.py", "--model", glb_path, "--output", output_mp4, "--frames", str(int(num_frames))], "Step 10: Cinematic MP4 Video Export")
    if ok and os.path.exists(output_mp4):
        return log, output_mp4
    return log, None

def launch_studio(default_model):
    with gr.Blocks(title="4D-Volumetric-Trails Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🌌 4D-Volumetric-Trails: No-Code Web Studio
            **End-to-End Spatiotemporal 4D Reconstruction, Multi-Subject Tracking & Volumetric Rendering Suite**
            """
        )

        with gr.Tabs():
            # TAB 1: ONE-CLICK PIPELINE STUDIO
            with gr.Tab("🚀 1. Pipeline Studio (Auto-Builder)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Monocular Input Video (.mp4)")
                        smooth_slider = gr.Slider(minimum=5, maximum=31, step=2, value=11, label="Cinematic Camera Smoothing Window (Odd Numbers)")
                        nerf_checkbox = gr.Checkbox(value=True, label="Run Step 5: Nerfstudio 3DGS Background Training")
                        run_btn = gr.Button("🔥 Run Full 4D Spacetime Pipeline", variant="primary")
                    
                    with gr.Column(scale=1):
                         studio_model_out = gr.Model3D(label="Generated 4D Spacetime Trail (.glb)", value=default_model if os.path.exists(default_model) else None)
                
                pipeline_logs = gr.Textbox(label="Live Pipeline Execution Logs", lines=14, interactive=False)
                
                run_btn.click(
                    fn=run_full_pipeline,
                    inputs=[video_input, smooth_slider, nerf_checkbox],
                    outputs=[pipeline_logs, studio_model_out]
                )

            # TAB 2: STANDALONE 3D VIEWER
            with gr.Tab("🌐 2. Interactive 4D Viewer"):
                gr.Markdown("Inspect any existing `.glb` Spacetime Trail directly in your browser.")
                with gr.Row():
                    model_path_in = gr.Textbox(value=default_model, label="Path to .glb Model")
                    load_btn = gr.Button("🔄 Load / Refresh 3D Model")
                viewer_3d = gr.Model3D(value=default_model if os.path.exists(default_model) else None, label="Interactive 4D Viewport", height=600)
                
                load_btn.click(fn=lambda p: p if os.path.exists(p) else None, inputs=[model_path_in], outputs=[viewer_3d])

            # TAB 3: CINEMATIC MP4 EXPORTER
            with gr.Tab("🎬 3. Cinematic MP4 Exporter"):
                gr.Markdown("Render a 1080p 360-degree turntable animation of your 4D trail for social media or presentations.")
                with gr.Row():
                    with gr.Column():
                        export_glb_in = gr.Textbox(value=default_model, label="Input .glb Model Path")
                        frames_slider = gr.Slider(minimum=60, maximum=360, step=30, value=120, label="Total Animation Frames (30 FPS)")
                        export_btn = gr.Button("🎥 Render 1080p MP4 Video", variant="primary")
                        export_logs = gr.Textbox(label="Render Logs", lines=8, interactive=False)
                    with gr.Column():
                        video_output = gr.Video(label="Rendered Cinematic Showcase (.mp4)")
                
                export_btn.click(
                    fn=render_mp4_showcase,
                    inputs=[export_glb_in, frames_slider],
                    outputs=[export_logs, video_output]
                )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the 4D-Volumetric-Trails No-Code Web Studio.")
    parser.add_argument("--model", required=False, default="data/output/spacetime_trail.glb", help="Default .glb model path")
    args = parser.parse_args()
    
    launch_studio(args.model)
