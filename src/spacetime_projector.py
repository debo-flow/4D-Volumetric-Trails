import os
import json
import argparse
import numpy as np
import trimesh
from PIL import Image

def build_spacetime_scene(transforms_path, segmented_dir, depth_dir, output_path):
    # Load camera poses (transforms.json from Nerfstudio/COLMAP)
    print(f"[INFO] Loading camera transforms from: {transforms_path}")
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
        
    scene = trimesh.Scene()
    frames = transforms.get("frames", [])
    
    print(f"[INFO] Projecting {len(frames)} frames into 4D space...")
    
    for frame in frames:
        # Resolve file names
        file_path = frame["file_path"]
        base_name = os.path.basename(file_path)
        if not base_name.endswith('.png'):
            base_name = base_name.split('.')[0] + '.png'
            
        img_path = os.path.join(segmented_dir, base_name)
        
        if not os.path.exists(img_path):
            continue
            
        # Get Camera Pose Matrix
        pose = np.array(frame["transform_matrix"])
        
        try:
            # 1. Load Segmented Image
            img = Image.open(img_path).convert("RGBA")
            w, h = img.size
            
            # 2. Create a base 3D Grid Mesh (Plane)
            # We use a lower resolution grid (e.g., 100x100 max) to keep the final .glb file lightweight
            res_w, res_h = min(w, 100), min(h, 100)
            mesh = trimesh.creation.grid((2.0, 2.0 * (h/w)), resolution=(res_w, res_h))
            
            # 3. Calculate UV mapping for textures
            uvs = mesh.visual.uv
            px = np.clip((uvs[:, 0] * w).astype(int), 0, w - 1)
            py = np.clip(((1.0 - uvs[:, 1]) * h).astype(int), 0, h - 1)
            
            # 4. Apply 2.5D Depth Displacement (If depth map is provided)
            if depth_dir:
                depth_path = os.path.join(depth_dir, base_name)
                if os.path.exists(depth_path):
                    depth_img = Image.open(depth_path).convert("L")
                    depth_arr = np.array(depth_img)
                    
                    # Normalize depth and push the Z-axis vertices (0.5 is the extrusion scale)
                    z_displacement = (depth_arr[py, px] / 255.0) * 0.5 
                    mesh.vertices[:, 2] += z_displacement
            
            # 5. Apply the RGBA Image as a Texture Material
            material = trimesh.visual.material.SimpleMaterial(image=img)
            mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=img, material=material)
            
            # 6. Move the 2.5D Mesh to the correct Camera location in 3D Space
            mesh.apply_transform(pose)
            
            # 7. Add to the global scene
            scene.add_geometry(mesh, geom_name=base_name)
            print(f"  -> Projected 2.5D mesh for: {base_name}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {base_name}: {e}")
            
    # Export the final 4D Trail
    print(f"\n[INFO] Exporting 4D Spacetime trail to {output_path} ...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene.export(output_path)
    print("[SUCCESS] 2.5D Volumetric Export complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project 2.5D subjects into 4D space.")
    parser.add_argument("--transforms", required=True, help="Path to Nerfstudio transforms.json")
    parser.add_argument("--segmented", required=True, help="Directory of segmented PNGs")
    parser.add_argument("--depth", required=False, default=None, help="Directory of depth map PNGs (Optional, enables 2.5D)")
    parser.add_argument("--output", required=True, help="Output .glb path")
    
    args = parser.parse_args()
    build_spacetime_scene(args.transforms, args.segmented, args.depth, args.output)
