import os
import json
import argparse
import numpy as np
import trimesh
from PIL import Image

def build_spacetime_scene(transforms_path, segmented_base, depth_base, output_path):
    print(f"[INFO] Loading transforms from: {transforms_path}")
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
        
    scene = trimesh.Scene()
    frames = transforms.get("frames", [])
    
    # Find all tracked subjects
    subjects = [d for d in os.listdir(segmented_base) if os.path.isdir(os.path.join(segmented_base, d))]
    
    if not subjects:
        print(f"[ERROR] No subjects found in {segmented_base}")
        return
    
    print(f"[INFO] Found {len(subjects)} tracked subjects. Projecting into a shared 4D space...")
    
    for subject in subjects:
        print(f"\n[INFO] Processing 4D Trail for: {subject}")
        subject_seg_dir = os.path.join(segmented_base, subject)
        subject_depth_dir = os.path.join(depth_base, subject) if depth_base else None
        
        for frame in frames:
            file_path = frame["file_path"]
            base_name = os.path.basename(file_path)
            if not base_name.endswith('.png'):
                base_name = base_name.split('.')[0] + '.png'
                
            img_path = os.path.join(subject_seg_dir, base_name)
            if not os.path.exists(img_path):
                continue
                
            pose = np.array(frame["transform_matrix"])
            
            try:
                img = Image.open(img_path).convert("RGBA")
                w, h = img.size
                
                res_w, res_h = min(w, 100), min(h, 100)
                mesh = trimesh.creation.grid((2.0, 2.0 * (h/w)), resolution=(res_w, res_h))
                
                uvs = mesh.visual.uv
                px = np.clip((uvs[:, 0] * w).astype(int), 0, w - 1)
                py = np.clip(((1.0 - uvs[:, 1]) * h).astype(int), 0, h - 1)
                
                if subject_depth_dir:
                    depth_path = os.path.join(subject_depth_dir, base_name)
                    if os.path.exists(depth_path):
                        depth_img = Image.open(depth_path).convert("L")
                        depth_arr = np.array(depth_img)
                        z_displacement = (depth_arr[py, px] / 255.0) * 0.5 
                        mesh.vertices[:, 2] += z_displacement
                
                material = trimesh.visual.material.SimpleMaterial(image=img)
                mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=img, material=material)
                mesh.apply_transform(pose)
                
                # Combine subject name and frame name for unique geometry ID
                geom_name = f"{subject}_{base_name}"
                scene.add_geometry(mesh, geom_name=geom_name)
                
            except Exception as e:
                print(f"[ERROR] Failed to process {subject}/{base_name}: {e}")
                
    print(f"\n[INFO] Exporting combined Multi-Subject 4D trail to {output_path} ...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene.export(output_path)
    print("[SUCCESS] Export complete! All subjects are now in the same 4D Space.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project multiple 2.5D subjects into a shared 4D space.")
    parser.add_argument("--transforms", required=True, help="Path to transforms_4d.json")
    parser.add_argument("--segmented", required=True, help="Base directory containing subject folders")
    parser.add_argument("--depth", required=False, default=None, help="Base directory containing depth map folders")
    parser.add_argument("--output", required=True, help="Output .glb path")
    
    args = parser.parse_args()
    build_spacetime_scene(args.transforms, args.segmented, args.depth, args.output)
