import gradio as gr
import argparse
import os

def create_ui(default_model="data/output/spacetime_trail.glb"):
    """
    Creates an interactive web interface to view the 4D Space-Time Trail.
    """
    # Check if the model exists to load it by default
    initial_model = default_model if os.path.exists(default_model) else None

    with gr.Blocks(title="4D Spacetime Viewer", theme=gr.themes.Monochrome()) as app:
        gr.Markdown("<h1 style='text-align: center;'>🌌 4D-Volumetric-Trails Viewer</h1>")
        gr.Markdown("<p style='text-align: center;'>Interactive web visualizer for your generated space-time projections.</p>")
        
        with gr.Row():
            with gr.Column(scale=3):
                # 3D Model Viewer Component
                model_viewer = gr.Model3D(
                    value=initial_model, 
                    clear_color=[0.1, 0.1, 0.1, 1], # Dark background for scientific look
                    label="4D Temporal Trail Viewer"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎮 Controls")
                gr.Markdown("- **Rotate:** Left Click + Drag")
                gr.Markdown("- **Pan:** Right Click + Drag")
                gr.Markdown("- **Zoom:** Scroll Wheel")
                
                gr.Markdown("---")
                gr.Markdown("### 📂 Load Custom Trail")
                file_input = gr.File(label="Upload .glb or .obj file", file_types=[".glb", ".obj"])
                
                # Function to update viewer when a new file is uploaded
                file_input.change(
                    fn=lambda x: x.name if x else None, 
                    inputs=file_input, 
                    outputs=model_viewer
                )
                
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the 4D Spacetime Web Viewer.")
    parser.add_argument("--model", type=str, default="data/output/spacetime_trail.glb", help="Path to the default .glb model")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the web server on")
    
    args = parser.parse_args()
    
    print("[INFO] Starting the 4D Web Viewer...")
    print(f"[INFO] Look for the local URL below (usually http://127.0.0.1:{args.port})")
    
    app = create_ui(args.model)
    app.launch(server_port=args.port, share=False)
