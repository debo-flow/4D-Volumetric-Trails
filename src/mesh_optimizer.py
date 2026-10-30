import os
import argparse
import open3d as o3d

def optimize_glb_mesh(input_path, output_path, reduction_ratio=0.75):
    if not os.path.exists(input_path):
        print(f"[ERROR] Input model not found: {input_path}")
        return

    print(f"[INFO] Loading high-poly 4D Mesh from: {input_path}")
    mesh = o3d.io.read_triangle_mesh(input_path, enable_post_processing=True)

    initial_triangles = len(mesh.triangles)
    if initial_triangles == 0:
        print("[WARNING] No triangles found in mesh. Copying original file.")
        o3d.io.write_triangle_mesh(output_path, mesh)
        return

    target_triangles = int(initial_triangles * (1.0 - reduction_ratio))
    print(f"[INFO] Initial Polygons: {initial_triangles} | Target Polygons: {target_triangles} ({int(reduction_ratio*100)}% reduction)")

    # Clean up redundant vertices and degenerate triangles
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    # Apply Quadric Decimation to compress geometry while preserving shape
    print("[INFO] Running Quadric Error Metric Decimation...")
    optimized_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    optimized_mesh.compute_vertex_normals()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_triangle_mesh(output_path, optimized_mesh)

    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[SUCCESS] Optimization complete! Saved to: {output_path}")
    print(f"[STATS] File Size reduced from {orig_size:.2f} MB to {new_size:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize and compress heavy 4D .glb meshes for smooth web viewing.")
    parser.add_argument("--input", required=True, help="Path to heavy input .glb file")
    parser.add_argument("--output", required=True, help="Path to save lightweight .glb file")
    parser.add_argument("--reduction", type=float, default=0.75, help="Polygon reduction ratio (0.1 to 0.9, default 0.75)")

    args = parser.parse_args()
    optimize_glb_mesh(args.input, args.output, args.reduction)
