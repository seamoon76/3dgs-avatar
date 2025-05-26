#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from scene import Scene
import os
from tqdm import tqdm, trange
from os import makedirs
from gaussian_renderer import render
import torchvision
import pdb
from utils.general_utils import fix_random
from utils.dataset_utils import fetchPly, storePly
from utils.sh_utils import RGB2SH
from scene import GaussianModel
from utils.general_utils import Evaluator, PSEvaluator
from utils.camera_utils import clone_cameras
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from argparse import ArgumentParser
# from arguments import ModelParams, PipelineParams, get_combined_args
import hydra
from omegaconf import OmegaConf
import wandb
import open3d as o3d

from scene.cameras import Camera

def predict(config):
    with torch.set_grad_enabled(False):
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        times = []
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            iter_start.record()

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
                                compute_loss=False, return_opacity=False)
            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),]
            wandb.log({'test_images': wandb_img})

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))

            # evaluate
            times.append(elapsed)

        _time = np.mean(times[1:])
        wandb.log({'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 time=_time)

def sample_hemisphere_cameras(base_camera, center, n_views=23, uniform=True):
    K = base_camera.K
    FoVx = base_camera.FoVx
    FoVy = base_camera.FoVy
    data_device = base_camera.data_device
    radius = np.linalg.norm(base_camera.T - center)
    cameras = []

    for i in range(n_views):
        # --- Step 1: Sample a point on the upper hemisphere ---
        phi=None
        if uniform:
            # Fibonacci sampling on hemisphere (evenly distributed)
            # phi = 2 * np.pi * i / ((1 + np.sqrt(5)) / 2)  # golden angle
            phi = 2 * np.pi * i / n_views
            y = np.random.uniform(-1.0, 1.0)  # y in (0, 1]
            # y = 0.
            r = np.sqrt(1 - y * y)
            x = r * np.cos(phi)
            z = r * np.sin(phi)
        else:
            # Pure random sampling
            vec = np.random.normal(size=3)
            vec[1] = np.abs(vec[1])  # make it "upward"
            vec /= np.linalg.norm(vec)
            x, y, z = vec

        cam_pos = center + radius * np.array([x, y, z], dtype=np.float32)
        # --- Step 2: Compute rotation matrix to look at center ---
        forward = center - cam_pos
        forward /= np.linalg.norm(forward)

        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(up, forward)
        if np.linalg.norm(right) < 1e-6:  # avoid degenerate case
            up = np.array([0, 0, 1], dtype=np.float32)
            right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)

        R = np.stack([right, up, forward], axis=0)  # camera-to-world
        T = -R @ cam_pos  # world-to-camera

        cam = Camera(
            frame_id=base_camera.frame_id,
            cam_id=i,
            K=K,
            R=R.astype(np.float32),
            T=T.astype(np.float32),
            FoVx=FoVx,
            FoVy=FoVy,
            image=base_camera.image,
            mask=base_camera.mask,
            gt_alpha_mask=base_camera.gt_alpha_mask,
            image_name=f"hemi_view_{i:03d}",
            data_device=base_camera.data_device,
            rots=base_camera.rots,
            Jtrs=base_camera.Jtrs,
            bone_transforms=base_camera.bone_transforms,
            all_cameras=None
        )
        cameras.append(cam)

    return cameras


def generate_orbiting_cameras(base_camera, center):
    """
    base_camera: Camera from scene.test_dataset[idx]
    center: (3,) numpy array, usually the mean of gaussians.get_xyz()
    """
    # Get camera intrinsics from base_camera
    K = base_camera.K
    FoVx = base_camera.FoVx
    FoVy = base_camera.FoVy
    n_views=23
    radius = np.linalg.norm(base_camera.T - center)

    cameras = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        cam_pos = center + np.array([
            radius * np.cos(theta),
            radius * 0.0,
            radius * np.sin(theta)
        ])

        # Camera looks at center: compute R
        forward = center - cam_pos
        forward /= np.linalg.norm(forward)

        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)

        R = np.stack([right, up, forward], axis=0)  # camera-to-world
        T = -R @ cam_pos  # world-to-camera

        # Convert to float32 numpy
        R = R.astype(np.float32)
        T = T.astype(np.float32)

        # Construct Camera
        cam = Camera(
            frame_id=base_camera.frame_id,
            cam_id=i,
            K=K,
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            image=base_camera.image,
            mask=base_camera.mask,
            gt_alpha_mask=base_camera.gt_alpha_mask,
            image_name=f"orbit_view_{i:03d}",
            data_device=base_camera.data_device,
            rots=base_camera.rots,
            Jtrs=base_camera.Jtrs,
            bone_transforms=base_camera.bone_transforms,
            all_cameras=None
        )
        cameras.append(cam)
    # cameras_dict = {"all_cameras": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]}
    # for i in range(n_views):
    #    cameras_dict[str(i+1)] = cameras[i]

    return cameras


def test(config):
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        evaluator = PSEvaluator() if config.dataset.name == 'people_snapshot' else Evaluator()

        psnrs = []
        ssims = []
        lpipss = []
        times = []
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            if view.cam_id != 2:
                continue
            iter_start.record()
            view = scene.test_dataset[idx]
            centers = gaussians.get_xyz.detach().cpu().numpy()
            human_center = centers.mean(axis=0)

            # orbit_cams = generate_orbiting_cameras(view, center=human_center)
            hemi_cams = sample_hemisphere_cameras(view, center=human_center)

            render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background, compute_loss=False, return_opacity=False)
            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]
            gt = view.original_image[:3, :, :]

            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),
                         wandb.Image(gt[None], caption='gt_{}'.format(view.image_name))]

            wandb.log({'test_images': wandb_img})

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))

            gaussExtractor = GaussianExtractor(scene, render, config.opt.iterations, config.pipeline, background=background)
            # set the active_sh to 0 to export only diffuse texture
            gaussExtractor.gaussians.active_sh_degree = 0

            # views_clone = clone_cameras(view.all_cameras, config, view)
            # views_clone = clone_cameras(orbit_cams, config, view)
            gaussExtractor.reconstruction(hemi_cams)
            name = 'single_fuse_idx_{}.ply'.format(idx)
            mesh_res = 1024
            depth_trunc = 5
            voxel_size = 0.004 #depth_trunc / mesh_res
            sdf_trunc = 5.0 * voxel_size
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

            o3d.io.write_triangle_mesh(os.path.join(render_path, name), mesh)
            print("mesh saved at {}".format(os.path.join(render_path, name)))
            # post-process the mesh and save, saving the largest N clusters
            mesh_post = post_process_mesh(mesh, cluster_to_keep=100)
            o3d.io.write_triangle_mesh(os.path.join(render_path, name.replace('.ply', '_post.ply')), mesh_post)
            print("mesh post processed saved at {}".format(os.path.join(render_path, name.replace('.ply', '_post.ply'))))
            # evaluate
            if config.evaluate:
                metrics = evaluator(rendering, gt)
                psnrs.append(metrics['psnr'])
                ssims.append(metrics['ssim'])
                lpipss.append(metrics['lpips'])
            else:
                psnrs.append(torch.tensor([0.], device='cuda'))
                ssims.append(torch.tensor([0.], device='cuda'))
                lpipss.append(torch.tensor([0.], device='cuda'))
            times.append(elapsed)


        _psnr = torch.mean(torch.stack(psnrs))
        _ssim = torch.mean(torch.stack(ssims))
        _lpips = torch.mean(torch.stack(lpipss))
        _time = np.mean(times[1:])
        wandb.log({'metrics/psnr': _psnr,
                   'metrics/ssim': _ssim,
                   'metrics/lpips': _lpips,
                   'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 psnr=_psnr.cpu().numpy(),
                 ssim=_ssim.cpu().numpy(),
                 lpips=_lpips.cpu().numpy(),
                 time=_time)
        # scene.save(0)
        # first_xyz = first_frame_render_pkg["deformed_gaussian"]._xyz.detach().cpu().numpy()
        # first_rgb = (first_frame_render_pkg["colors_precomp"].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        # pdb.set_trace()
        # storePly("./pc.ply", first_xyz, first_frame_render_pkg["colors_precomp"].detach().cpu().numpy())
        # pcd = fetchPly("./pc.ply")
        # gs = first_frame_render_pkg["deformed_gaussian"]
        # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.points)).float().cuda())
        # features = torch.zeros((fused_color.shape[0], 3, (3 + 1) ** 2)).float().cuda()
        # features[:, :3, 0 ] = fused_color
        # features[:, 3:, 1:] = 0.0
        # print(features.shape)
        # gs._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # gs._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # gs.save_ply("./gs_sh.ply")
        # save_mesh_ply("./mesh.ply", first_xyz, first_rgb)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False
    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)

    # set wandb logger
    if config.mode == 'test':
        config.suffix = config.mode + '-' + config.dataset.test_mode
    elif config.mode == 'predict':
        predict_seq = config.dataset.predict_seq
        if config.dataset.name == 'zjumocap':
            predict_dict = {
                0: 'dance0',
                1: 'dance1',
                2: 'flipping',
                3: 'canonical'
            }
        else:
            predict_dict = {
                0: 'rotation',
                1: 'dance2',
            }
        predict_mode = predict_dict[predict_seq]
        config.suffix = config.mode + '-' + predict_mode
    else:
        raise ValueError
    if config.dataset.freeview:
        config.suffix = config.suffix + '-freeview'
    wandb_name = config.name + '-' + config.suffix
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='gaussian-splatting-avatar-test',
        entity='seamoon2020-eth-z-rich',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    fix_random(config.seed)

    if config.mode == 'test':
        test(config)
    elif config.mode == 'predict':
        predict(config)
    else:
        raise ValueError

if __name__ == "__main__":
    main()
