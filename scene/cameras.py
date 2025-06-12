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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera:
    def __init__(self, camera=None, **kwargs):
        if camera is not None:
            # 拷贝已有camera的数据字典，实现copy功能
            self.data = camera.data.copy()
            return
        
        # 接收所有参数，存入data字典
        self.data = kwargs

        # 设默认值，防止缺少关键参数报错
        self.data['trans'] = self.data.get('trans', np.array([0.0, 0.0, 0.0]))
        self.data['scale'] = self.data.get('scale', 1.0)
        self.data_device = torch.device(self.data.get('data_device', 'cuda'))

        # 图像相关
        self.data['original_image'] = self.data['image'].clamp(0.0, 1.0).to(self.data_device)
        self.data['image_width'] = self.data['original_image'].shape[2]
        self.data['image_height'] = self.data['original_image'].shape[1]

        # mask处理，若无则设为1全白mask
        if 'mask' in self.data and self.data['mask'] is not None:
            self.data['original_mask'] = self.data['mask'].float().to(self.data_device)
        else:
            self.data['original_mask'] = torch.ones(
                (1, self.data['image_height'], self.data['image_width']), device=self.data_device
            )

        # 相机远近裁剪面
        self.data['zfar'] = 100.0
        self.data['znear'] = 0.01

        # world-view矩阵和投影矩阵
        self.data['world_view_transform'] = torch.tensor(
            getWorld2View2(self.data['R'], self.data['T'], self.data['trans'], self.data['scale'])
        ).transpose(0, 1).to(self.data_device)

        self.data['projection_matrix'] = getProjectionMatrix(
            znear=self.data['znear'], zfar=self.data['zfar'], fovX=self.data['FoVx'], fovY=self.data['FoVy']
        ).transpose(0, 1).to(self.data_device)

        self.data['full_proj_transform'] = (
            self.data['world_view_transform'].unsqueeze(0).bmm(self.data['projection_matrix'].unsqueeze(0))
        ).squeeze(0)

        self.data['camera_center'] = self.data['world_view_transform'].inverse()[3, :3]

        # 骨骼相关，可能为空
        self.data['rots'] = self.data.get('rots', None)
        self.data['Jtrs'] = self.data.get('Jtrs', None)
        self.data['bone_transforms'] = self.data.get('bone_transforms', None)

    def __getattr__(self, item):
        # 支持用camera.xxx访问self.data里的字段
        if item in self.data:
            return self.data[item]
        raise AttributeError(f"'Camera' object has no attribute '{item}'")

    def update(self, **kwargs):
        # 更新数据字典里的参数
        self.data.update(kwargs)

    def copy(self):
        # 复制Camera对象
        return Camera(camera=self)

    def merge(self, cam):
        # 合并另一个Camera的部分参数，常用于帧级更新
        self.data['frame_id'] = cam.frame_id
        self.data['rots'] = cam.rots.detach()
        self.data['Jtrs'] = cam.Jtrs.detach()
        self.data['bone_transforms'] = cam.bone_transforms.detach()
         
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

