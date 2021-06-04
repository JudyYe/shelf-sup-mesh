# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import numpy as np
from typing import Optional

import absl.flags as flags
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    # TexturesVertex,
    MeshRenderer, SoftSilhouetteShader, SoftGouraudShader,
    BlendParams, MeshRasterizer, RasterizationSettings,
    DirectionalLights, )
from nnutils.shading import SoftFeatureGouraudShader
import nnutils.mesh_utils as mesh_utils
FLAGS = flags.FLAGS


class MeshRenderWrapper(nn.Module):
    def __init__(self,
                 out_size=64,
                 shader='feat',
                 sigma: float = 1e-4,
                 blur_radius: float = 0,
                 faces_per_pixel: int = 1,
                 bin_size: Optional[int] = None,
                 max_faces_per_bin: Optional[int] = None,
                 ):
        super().__init__()
        raster_settings = RasterizationSettings(
            image_size=out_size,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            bin_size=bin_size,
            max_faces_per_bin=max_faces_per_bin
        )
        rasterizer = MeshRasterizer(cameras=None, raster_settings=raster_settings)

        shader_dict = {'sil': SoftSilhouetteShader,
                       'feat': SoftFeatureGouraudShader,
                       'gouraud': SoftGouraudShader}
        if shader == 'feat':
            self.my_shader = True
            shader = shader_dict[shader](device='cuda:0', blend_params=BlendParams(sigma=sigma, background_color=1))
        else:
            self.my_shader = False
            shader = shader_dict[shader](device='cuda:0', blend_params=BlendParams(sigma=sigma, ))

        self.renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=shader
        )

    def forward(self, meshes: Meshes, view_params, verts_feat=None, **kwargs):
        """
        :param meshes: (N, 7)
        :param view_params:
        :param feats: (sum(V), D)
        :return: (N, C=3, H, W), (N, C=1, H, W) in range [-1, 1]?? which ever the verts_feat gives
        """
        render_kwargs = {}

        if isinstance(view_params, torch.Tensor):
            cameras = mesh_utils.param7dof_to_camera(view_params)
            z_range = view_params[:, -2]
            znear=z_range - 1
            zfar=z_range + 1
            render_kwargs['znear'] = znear
            render_kwargs['zfar'] = zfar
        else:
            cameras = view_params
        if verts_feat is None:
            if not self.my_shader:
                verts_feat = torch.ones_like(meshes.verts_padded())
            else:
                verts_feat = torch.ones_like(meshes.verts_packed())

        if not self.my_shader:
            meshes.textures = mesh_utils.pad_texture(meshes, verts_feat)

        m = kwargs.get('light_direction', np.array([[0, -0.6, 0.8]]))
        color = kwargs.get('light_color', np.array([0.65, 0.3, 0.0]))
        am, df, sp = color
        lights = DirectionalLights(
            device=meshes.device,
            direction=m,
            ambient_color=((am, am, am), ),
            diffuse_color=((df, df, df),),
            specular_color=((sp, sp, sp), ),
        )
        image = self.renderer(meshes,
                              cameras=cameras,
                              lights=lights,
                              feature=verts_feat,
                              **render_kwargs,
                              )  # (N, H, W, 4?) in range(-1, 1) and [0, 1] for mask??

        image = torch.flip(image, dims=[-3])
        image = image.transpose(-1, -2).transpose(-2, -3)  # H, 4, W --> 4, H, W
        image, mask = torch.split(image, [image.size(1) - 1, 1], dim=1)  # [-1, 1???]

        res = {'image': image, 'mask': mask}
        return res
