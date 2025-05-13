import numpy as np
import open3d as o3d

def create_point_cloud(xyz, rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    return pcd

def estimate_normals(pcd, radius=0.05, max_nn=30):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.normalize_normals()
    return pcd

def poisson_mesh(pcd, depth=9):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.01)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh

def save_mesh(mesh, filename):
    o3d.io.write_triangle_mesh(filename, mesh)
    print(f"Mesh saved to {filename}")

def save_mesh_ply(path,xyz,rgb):
	pcd = create_point_cloud(xyz, rgb)
	pcd = estimate_normals(pcd)
	mesh = poisson_mesh(pcd)
	save_mesh(mesh, path)

