import torch
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render, integrate
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra
import hydra
from omegaconf import DictConfig, OmegaConf

@torch.no_grad()
def evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, return_color=False):
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    if return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")
    
    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            ret = integrate(points, view, gaussians, pipeline, background, kernel_size=kernel_size)
            alpha_integrated = ret["alpha_integrated"]
            if return_color:
                color_integrated = ret["color_integrated"]    
                final_color = torch.where((alpha_integrated < final_alpha).reshape(-1, 1), color_integrated, final_color)
            final_alpha = torch.min(final_alpha, alpha_integrated)
            
        alpha = 1 - final_alpha
    if return_color:
        return alpha, final_color
    return alpha

@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, filter_mesh : bool, texture_mesh : bool, near : float, far : float):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fusion")

    makedirs(render_path, exist_ok=True)
    
    # generate tetra points here
    print(render_path, views, near, far)
    near = 0.05
    far = 5.0
    points, points_scale = gaussians.get_tetra_points(views, near, far)
    # load cell if exists
    if os.path.exists(os.path.join(render_path, "cells.pt")):
        print("load existing cells")
        cells = torch.load(os.path.join(render_path, "cells.pt"))
    else:
        # create cell and save cells
        print("create cells and save")
        cells = cpp.triangulate(points)
        # we should filter the cell if it is larger than the gaussians
        torch.save(cells, os.path.join(render_path, "cells.pt"))
    
    # evaluate alpha
    if points.shape[0] == 0:
        print("⚠️ No points were generated — check views/near/far")
        return
    alpha = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size)
    print("alpha range:", alpha.min().item(), alpha.max().item(), alpha.mean().item())

    vertices = points.cuda()[None]
    tets = cells.cuda().long()

    print(vertices.shape, tets.shape, alpha.shape)
    def alpha_to_sdf(alpha):    
        sdf = alpha - 0.5
        sdf = sdf[None]
        return sdf
    
    sdf = alpha_to_sdf(alpha)
    
    torch.cuda.empty_cache()
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf, points_scale[None])
    print("number of surfaces:", len(faces_list[0]))
    torch.cuda.empty_cache()
    
    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    
    faces=faces_list[0].cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.
        
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale
    
    n_binary_steps = 8
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        alpha = evaluage_alpha(mid_points, views, gaussians, pipeline, background, kernel_size)
        mid_sdf = alpha_to_sdf(alpha).squeeze().unsqueeze(-1)
        
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
    
        points = (left_points + right_points) / 2
        if step not in [7]:
            continue
        
        if texture_mesh:
            _, color = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, return_color=True)
            vertex_colors=(color.cpu().numpy() * 255).astype(np.uint8)
        else:
            vertex_colors=None
        mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, vertex_colors=vertex_colors, process=False)
        
        # filter
        if filter_mesh:
            mask = (distance <= scale).cpu().numpy()
            face_mask = mask[faces].all(axis=1)
            mesh.update_vertices(mask)
            mesh.update_faces(face_mask)
            print("vertex mask keep ratio:", mask.sum(), "/", len(mask))
            print("face mask keep ratio:", face_mask.sum(), "/", len(face_mask))
        mesh.export(os.path.join(render_path, f"mesh_binary_search_{step}.ply"))

    # linear interpolation
    # right_sdf *= -1
    # points = (left_points * left_sdf + right_points * right_sdf) / (left_sdf + right_sdf)
    # mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces)
    # mesh.export(os.path.join(render_path, f"mesh_binary_search_interp.ply"))
    

def extract_mesh(cfg, dataset : ModelParams, iteration : int, pipeline : PipelineParams, filter_mesh : bool, texture_mesh : bool, near : float, far : float):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(cfg, dataset, gaussians)
        
        gaussians.load_ply(os.path.join(cfg.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        
        from random import randint
        data_stack = None
        if not data_stack:
            data_stack = list(range(len(scene.train_dataset)))
        data_idx = data_stack.pop(randint(0, len(data_stack)-1))
        data = scene.train_dataset[data_idx]
        # gaussians.compute_3D_filter(cameras=[data])    
        cams = [data]
        marching_tetrahedra_with_binary_search(cfg.model_path, "test", iteration, cams, gaussians, pipeline, background, kernel_size, filter_mesh, texture_mesh, near, far)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=2000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter_mesh", action="store_true")
    parser.add_argument("--texture_mesh", action="store_true")
    parser.add_argument("--near", default=0.02, type=float)
    parser.add_argument("--far", default=1e6, type=float)
    parser.add_argument("--model_path", default="output/11fc0e3b-7", type=str)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    extract_mesh(config, model.extract(args), args.iteration, pipeline.extract(args), args.filter_mesh, args.texture_mesh, args.near, args.far)
if __name__ == "__main__":
    main()