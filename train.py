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

import os
import numpy as np
import open3d as o3d
import cv2
import torch
import torchvision
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import fix_random, Evaluator, PSEvaluator
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import os
import matplotlib.pyplot as plt

import time
from utils.vis_utils import apply_depth_colormap, save_points, colormap
from utils.depth_utils import depths_to_points, depth_to_normal
import hydra
from omegaconf import DictConfig, OmegaConf
import torch, random, numpy as np
import lpips

def C(iteration, value):
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = OmegaConf.to_container(value)
        if not isinstance(value, list):
            raise TypeError('Scalar specification only supports list, got', type(value))
        value_list = [0] + value
        i = 0
        current_step = iteration
        while i < len(value_list):
            if current_step >= value_list[i]:
                i += 2
            else:
                break
        value = value_list[i - 1]
    return value

@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image
    
def get_edge_aware_distortion_map(gt_image, distortion_map):
    grad_img_left = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2]), 0)
    grad_img_right = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:]), 0)
    grad_img_top = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1]), 0)
    grad_img_bottom = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1]), 0)
    max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0]
    # pad
    max_grad = torch.exp(-max_grad)
    max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
    return distortion_map * max_grad
    

def L1_loss_appearance(image, gt_image, gaussians, view_idx, return_transformed_image=False):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]
    
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
    
    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    transformed_image = mapping_image * crop_image
    if not return_transformed_image:
        return l1_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
        return transformed_image
    
def training(cfg, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    pipe.debug = True
    pipe.convert_SHs_python = True
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(cfg)
    testing_interval = cfg.test_interval
    gaussians = GaussianModel(dataset.sh_degree)
    
    lpips_type = cfg.opt.get('lpips_type', 'vgg')
    loss_fn_vgg = lpips.LPIPS(net=lpips_type).cuda() # for training
    evaluator = Evaluator()
    scene = Scene(cfg, dataset, gaussians)
    scene.train()

    gaussians.training_setup(opt)
    if checkpoint:
        scene.load_checkpoint(checkpoint)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    data_stack = None
    if not data_stack:
        data_stack = list(range(len(scene.train_dataset)))
    data_idx = data_stack.pop(randint(0, len(data_stack)-1))
    data = scene.train_dataset[data_idx]
    gaussians.compute_3D_filter(cameras=[data])    

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random data point
        if not data_stack:
            data_stack = list(range(len(scene.train_dataset)))
        data_idx = data_stack.pop(randint(0, len(data_stack)-1))
        data = scene.train_dataset[data_idx]

        # gaussians.compute_3D_filter(cameras=[data])
        # print("===================================")
        # print(data)
        # print("===================================")
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        config = cfg
        lambda_mask = C(iteration, config.opt.lambda_mask)
        use_mask = lambda_mask > 0.
        render_pkg = render(data, gaussians, pipe, background, kernel_size=3)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image = render_pkg["render"][:3, :, :]
        gt_image = data.original_image.cuda()

        # ✅ 调试输出
        # print(f"\n[DEBUG Iter {iteration}] ------------------------------")
        # print(f"Rendered image min/max: {image.min().item():.4f} / {image.max().item():.4f}")
        # print(f"GT image min/max: {gt_image.min().item():.4f} / {gt_image.max().item():.4f}")
        # print(f"Visibility filter sum: {visibility_filter.sum().item()}")
        # print(f"Opacity min/max: {gaussians.get_opacity.min().item():.4f} / {gaussians.get_opacity.max().item():.4f}")
        # print(f"XYZ bounds: {gaussians.get_xyz.min(dim=0).values.tolist()} - {gaussians.get_xyz.max(dim=0).values.tolist()}")
        # 计算损失
        Ll1 = l1_loss(image, gt_image)
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if iteration % 100 == 0:  # 可调节频率
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].imshow(gt_image.permute(1, 2, 0).detach().cpu().numpy())
            axs[0].set_title("GT Image")
            axs[0].axis('off')
            axs[1].imshow(image.permute(1, 2, 0).detach().cpu().numpy())
            axs[1].set_title("Rendered Image")
            axs[1].axis('off')
            plt.suptitle(f"Train Iteration {iteration}")
            plt.tight_layout()
            plt.savefig(f"debug_vis/train/iter_{iteration}.png")
            plt.close()
        # 如果你有额外损失，比如depth_normal_loss、distortion_loss，可以在这里计算

        loss = rgb_loss
        loss.backward()

        iter_end.record()
        torch.cuda.synchronize()

        # 记录和显示
        with torch.no_grad():
            elapsed = iter_start.elapsed_time(iter_end)
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 这里写日志函数，如 tensorboard 或 wandb

            # 定期保存模型
            # os.makedirs("debug_vis/train", exist_ok=True)
            # if iteration % 100 == 0:  # 可调节频率
            #     fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            #     axs[0].imshow(gt_image.permute(1, 2, 0).detach().cpu().numpy())
            #     axs[0].set_title("GT Image")
            #     axs[0].axis('off')

            #     axs[1].imshow(image.permute(1, 2, 0).detach().cpu().numpy())
            #     axs[1].set_title("Rendered Image")
            #     axs[1].axis('off')

            #     plt.suptitle(f"Train Iteration {iteration}")
            #     plt.tight_layout()
            #     plt.savefig(f"debug_vis/train/iter_{iteration}.png")
            #     plt.close()
            dataset.kernel_size = 1.0
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size))
            # validation(iteration, testing_iterations, testing_interval, scene, evaluator,(pipe, background))
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")

            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=[data]) 

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=[data]) 
            # 优化器更新
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

def training_3dgs(config, dataset):
    model = config.model
    # dataset = config.dataset
    opt = config.opt
    pipe = config.pipeline
    testing_iterations = config.test_iterations
    testing_interval = config.test_interval
    saving_iterations = config.save_iterations
    checkpoint_iterations = config.checkpoint_iterations
    checkpoint = config.start_checkpoint
    debug_from = config.debug_from

    # define lpips
    lpips_type = config.opt.get('lpips_type', 'vgg')
    loss_fn_vgg = lpips.LPIPS(net=lpips_type).cuda() # for training
    evaluator = Evaluator()

    first_iter = 0
    gaussians = GaussianModel(model.gaussian)
    # scene = Scene(config, gaussians, config.exp_dir)
    scene = Scene(config, dataset, gaussians)
    scene.train()

    gaussians.training_setup(opt)
    if checkpoint:
        scene.load_checkpoint(checkpoint)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    data_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random data point
        if not data_stack:
            data_stack = list(range(len(scene.train_dataset)))
        data_idx = data_stack.pop(randint(0, len(data_stack)-1))
        data = scene.train_dataset[data_idx]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        lambda_mask = C(iteration, config.opt.lambda_mask)
        use_mask = lambda_mask > 0.
        render_pkg = render(data, iteration, scene, pipe, background, compute_loss=True, return_opacity=use_mask)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        opacity = render_pkg["opacity_render"] if use_mask else None
        depth = render_pkg["depth"]
        # Loss
        gt_image = data.original_image.cuda()

        lambda_l1 = C(iteration, config.opt.lambda_l1)
        lambda_dssim = C(iteration, config.opt.lambda_dssim)
        loss_l1 = torch.tensor(0.).cuda()
        loss_dssim = torch.tensor(0.).cuda()
        if lambda_l1 > 0.:
            loss_l1 = l1_loss(image, gt_image)
        if lambda_dssim > 0.:
            loss_dssim = 1.0 - ssim(image, gt_image)
        loss = lambda_l1 * loss_l1 + lambda_dssim * loss_dssim

        # perceptual loss
        lambda_perceptual = C(iteration, config.opt.get('lambda_perceptual', 0.))
        if lambda_perceptual > 0:
            # crop the foreground
            mask = data.original_mask.cpu().numpy()
            mask = np.where(mask)
            y1, y2 = mask[1].min(), mask[1].max() + 1
            x1, x2 = mask[2].min(), mask[2].max() + 1
            fg_image = image[:, y1:y2, x1:x2]
            gt_fg_image = gt_image[:, y1:y2, x1:x2]

            loss_perceptual = loss_fn_vgg(fg_image, gt_fg_image, normalize=True).mean()
            loss += lambda_perceptual * loss_perceptual
        else:
            loss_perceptual = torch.tensor(0.)

        # mask loss
        gt_mask = data.original_mask.cuda()
        if not use_mask:
            loss_mask = torch.tensor(0.).cuda()
        elif config.opt.mask_loss_type == 'bce':
            opacity = torch.clamp(opacity, 1.e-3, 1.-1.e-3)
            loss_mask = F.binary_cross_entropy(opacity, gt_mask)
        elif config.opt.mask_loss_type == 'l1':
            loss_mask = F.l1_loss(opacity, gt_mask)
        else:
            raise ValueError
        loss += lambda_mask * loss_mask

        # skinning loss
        lambda_skinning = C(iteration, config.opt.lambda_skinning)
        if lambda_skinning > 0:
            loss_skinning = scene.get_skinning_loss()
            loss += lambda_skinning * loss_skinning
        else:
            loss_skinning = torch.tensor(0.).cuda()

        lambda_aiap_xyz = C(iteration, config.opt.get('lambda_aiap_xyz', 0.))
        lambda_aiap_cov = C(iteration, config.opt.get('lambda_aiap_cov', 0.))
        if lambda_aiap_xyz > 0. or lambda_aiap_cov > 0.:
            loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(scene.gaussians, render_pkg["deformed_gaussian"])
        else:
            loss_aiap_xyz = torch.tensor(0.).cuda()
            loss_aiap_cov = torch.tensor(0.).cuda()
        loss += lambda_aiap_xyz * loss_aiap_xyz
        loss += lambda_aiap_cov * loss_aiap_cov

        # regularization
        loss_reg = render_pkg["loss_reg"]
        for name, value in loss_reg.items():
            lbd = opt.get(f"lambda_{name}", 0.)
            lbd = C(iteration, lbd)
            loss += lbd * value
        loss.backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            elapsed = iter_start.elapsed_time(iter_end)
            log_loss = {
                'loss/l1_loss': loss_l1.item(),
                'loss/ssim_loss': loss_dssim.item(),
                'loss/perceptual_loss': loss_perceptual.item(),
                'loss/mask_loss': loss_mask.item(),
                'loss/loss_skinning': loss_skinning.item(),
                'loss/xyz_aiap_loss': loss_aiap_xyz.item(),
                'loss/cov_aiap_loss': loss_aiap_cov.item(),
                'loss/total_loss': loss.item(),
                'iter_time': elapsed,
            }
            log_loss.update({
                'loss/loss_' + k: v for k, v in loss_reg.items()
            })
            wandb.log(log_loss)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            validation(iteration, testing_iterations, testing_interval, scene, evaluator,(pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and iteration > model.gaussian.delay:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt, scene, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                scene.optimize(iteration)

            if iteration in checkpoint_iterations:
                scene.save_checkpoint(iteration)            
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : list(range(len(scene.test_dataset)))},
                              {'name': 'train', 'cameras' : [idx for idx in range(0, len(scene.train_dataset), len(scene.train_dataset) // 10)]})
        for config in validation_configs:
            # print(config, config['cameras'])
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, camera_idex in enumerate(config['cameras']):
                    viewpoint = getattr(scene, config['name'] + '_dataset')[camera_idex]
                    rendering = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]
                    image = rendering[:3, :, :]
                    normal = rendering[3:6, :, :]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    if idx < 3:
                        import matplotlib.pyplot as plt
                        import os
                        os.makedirs("debug_vis", exist_ok=True)
                        print("Camera transform:\n", viewpoint.world_view_transform)
                        print("Projection matrix:\n", viewpoint.full_proj_transform)
                        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                        axs[0].imshow(gt_image.permute(1, 2, 0).detach().cpu().numpy())
                        axs[0].set_title(f"GT: {viewpoint.image_name}")
                        axs[0].axis('off')
                    
                        axs[1].imshow(image.permute(1, 2, 0).detach().cpu().numpy())
                        axs[1].set_title(f"Render: {viewpoint.image_name}")
                        axs[1].axis('off')
                    
                        plt.suptitle(f"{config['name']} - Iter {iteration} - View {idx}")
                        plt.tight_layout()
                        # ✅ 保存图片到本地文件夹
                        plt.savefig(f"debug_vis/{config['name']}_iter{iteration}_view{idx}.png")
                        plt.close()
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

        torch.cuda.empty_cache()

def validation(iteration, testing_iterations, testing_interval, scene : Scene, evaluator, renderArgs):
    # Report test and samples of training set
    if testing_interval > 0:
        if not iteration % testing_interval == 0:
            return
    else:
        if not iteration in testing_iterations:
            return

    scene.eval()
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : list(range(len(scene.test_dataset)))},
                          {'name': 'train', 'cameras' : [idx for idx in range(0, len(scene.train_dataset), len(scene.train_dataset) // 10)]})

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            examples = []
            for idx, data_idx in enumerate(config['cameras']):
                data = getattr(scene, config['name'] + '_dataset')[data_idx]
                render_pkg = render(data, iteration, scene, *renderArgs, compute_loss=False, return_opacity=True)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(data.original_image.to("cuda"), 0.0, 1.0)
                opacity_image = torch.clamp(render_pkg["opacity_render"], 0.0, 1.0)


                wandb_img = wandb.Image(opacity_image[None],
                                        caption=config['name'] + "_view_{}/render_opacity".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(image[None], caption=config['name'] + "_view_{}/render".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(gt_image[None], caption=config['name'] + "_view_{}/ground_truth".format(
                    data.image_name))
                examples.append(wandb_img)

                l1_test += l1_loss(image, gt_image).mean().double()
                metrics_test = evaluator(image, gt_image)
                psnr_test += metrics_test["psnr"]
                ssim_test += metrics_test["ssim"]
                lpips_test += metrics_test["lpips"]

                wandb.log({config['name'] + "_images": examples})
                examples.clear()

            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
            wandb.log({
                config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                config['name'] + '/loss_viewpoint - psnr': psnr_test,
                config['name'] + '/loss_viewpoint - ssim': ssim_test,
                config['name'] + '/loss_viewpoint - lpips': lpips_test,
            })

    wandb.log({'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu())})
    wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]})
    torch.cuda.empty_cache()
    scene.train()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 2_000, 5_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 2_000, 5_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args, unknown = parser.parse_known_args()

    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + config.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    # # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(config, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    # training_3dgs(config, lp.extract(args))
    # All done
    print("\nTraining complete.")
if __name__ == "__main__":
    main()
