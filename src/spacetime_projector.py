import os
import json
import argparse
import numpy as np
import trimesh
from PIL import Image

def create_textured_plane(image_path, transform_matrix, depth=5.0):
    """
    Creates a 3D plane (quad) and applies the segmented image as a texture.
    Positions the plane in 3D space based on the camera's transform matrix.
    """
    # Load image to get aspect ratio
    img = Image.open(image_path)
    width, height = img.size
    aspect_ratio = width / height

    # Create a 2D plane (quad)
    # Scaling it based on aspect ratio and depth
    plane_width = depth * aspect_ratio
    plane_height = depth
    
    # Define vertices for the plane
    vertices = np.array([
        [-plane_width/2, -plane_height/2, -depth],
        [ plane_width/2, -plane_height/2, -depth],
        [ plane_width/2,  plane_height/2, -depth],
        [-plane_width/2,  plane_height/2, -depth]
    ])
    
    # Define faces (two triangles make a square/rectangle)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    
    # Texture coordinates mapping
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    # Create the Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Apply the camera's 3D transformation (rotation + translation)
    mesh.apply_transform(transform_matrix)
    
    return mesh

def project_spacetime_trails(transforms_path, segmented_dir, output_path):
    """
    Reads camera coordinates and segmented images, then builds a unified 3D 
    scene containing the entire 4D temporal trail.
    """
    if not os.path.exists(transforms_path):
        print(f"[ERROR] Transforms file not found: {transforms_path}")
        return
        
    if not os.path.exists(segmented_dir):
        print(f"[ERROR] Segmented images directory not found: {segmented_dir}")
        return

    print("[INFO] Loading camera transforms...")
    with open(transforms_path, 'r') as f:
        data = json.load(f)

    frames = data.get('frames', [])
    if not frames:
        print("[ERROR] No camera frames found in transforms.json")
        return

    print(f"[INFO] Found {len(frames)} camera poses. Generating 4D trails...")
    
    scene = trimesh.Scene()
    
    # Loop through each frame recorded in time
    for i, frame in enumerate(frames):
        # Find matching segmented image
        base_name = os.path.basename(frame['file_path'])
        # Depending on naming convention, ensure we find the right seg_frame
        seg_img_name = f"seg_{base_name}"
        if not seg_img_name.endswith('.png'):
            seg_img_name = os.path.splitext(seg_img_name)[0] + ".png"
            
        img_path = os.path.join(segmented_dir, seg_img_name)
        
        if os.path.exists(img_path):
            transform_matrix = np.array(frame['transform_matrix'])
            
            # Create the 3D plane for this specific time-step
            # depth=5.0 means the subject is placed 5 units away from the camera
            trail_mesh = create_textured_plane(img_path, transform_matrix, depth=5.0)
            scene.add_geometry(trail_mesh, node_name=f"frame_{i}")
            
        if (i + 1) % 50 == 0:
            print(f"[INFO] Projected {i + 1} frames into 3D space...")

    # Export the entire 4D trail as a 3D object file (.glb or .obj)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene.export(output_path)
    print(f"\n[SUCCESS] 4D Spacetime Trail generated successfully!")
    print(f"[INFO] 3D Model saved to: {output_path}")
    print("[INFO] You can view this file in Blender or any 3D viewer.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project segmented 2D frames into a 3D Spacetime Trail.")
    parser.add_argument("--transforms", type=str, default="data/3d_model/ns_data/transforms.json", help="Path to Nerfstudio transforms.json")
    parser.add_argument("--segmented", type=str, default="data/segmented", help="Path to segmented PNGs")
    parser.add_argument("--output", type=str, default="data/output/spacetime_trail.glb", help="Output 3D model path")
    
    args = parser.parse_args()
    
    project_spacetime_trails(args.transforms, args.segmented, args.output)
