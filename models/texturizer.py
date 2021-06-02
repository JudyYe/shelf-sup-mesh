# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
import nnutils.mesh_utils as mesh_utils
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
)
from nnutils.layers import get_norm_layer, linear_block, linear
import absl.flags as flags
FLAGS = flags.FLAGS


def TextureFactory(name, **kwargs):
    """convert feature to RGB for each vertex """
    name_dict = {
        'pix': PixTexturizer,
    }
    return name_dict[name](**kwargs)


### Decouple Sampler
class PixTexturizer(nn.Module):
    def __init__(self, out_size, **kwargs):
        super().__init__()
        self.out_dim = 3
        raster_settings = RasterizationSettings(
            image_size=out_size,
            blur_radius=0,
            faces_per_pixel=1,
        )
        self.rasterizer = MeshRasterizer(cameras=None, raster_settings=raster_settings)

    def forward(self, meshes: Meshes, **kwargs):
        """
        :param meshes:
        :param kwargs:
        :return: packed verts? (sumV, D)
        """
        local_feat = kwargs.get('image', None)
        sym = kwargs.get('sym', FLAGS.text_sym)
        vis = kwargs.get('vis', FLAGS.text_vis)
        view = kwargs.get('view', None)

        local_feat, local_vis = mesh_utils.get_local_feat(meshes, self.rasterizer, local_feat, view, sym, vis)

        bg_color = kwargs.get('bg', [1, -.5, -.5])
        if isinstance(bg_color, list):
            bg_color = torch.FloatTensor([bg_color]).to(local_feat)
        # todo
        verts_pix = local_feat * local_vis + bg_color * (1 - local_vis)
        # verts_pix = local_feat
        return verts_pix

