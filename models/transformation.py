# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import absl.flags as flags
FLAGS = flags.FLAGS

from nnutils import geom_utils


#   input: 3D voxel (X), view param (az,el,s,xyz,f)
#   output: transformed 3D voxel
#     view param -> world-to-view mat transformation (P), intrinsic mat (K)
#     Y = KPX
#     K:= (x,y,z) -> (fx, fy, z)

### Projective transformation ###
class Perspective3d(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.z_range = 1
        with open(FLAGS.cfg_file) as fp:
            self.cfg = json.load(fp)

        self.f = self.cfg['f']

    def ray2camera_grid(self, out_size, out_depth, f, N, device):
        """
        :param out_size:
        :param out_depth:
        :param f: (N, )
        :param device:
        :return: shoot out ray through image pixels at multiple depth. coordinate in camera space.
        """
        height = width = out_size
        z0 = geom_utils.calc_rho(f)  # (N, )

        x_t = torch.linspace(-0.5, 0.5, width, dtype=torch.float32, device=device)  # image space
        y_t = torch.linspace(-0.5, 0.5, height, dtype=torch.float32, device=device)  # image space
        z_t = torch.linspace(-self.z_range / 2, self.z_range / 2, out_depth, dtype=torch.float32, device=device)  # depth step

        z_t, y_t, x_t = torch.meshgrid(z_t, y_t, x_t)  # [D, W, H]  # cmt: this must be in ZYX order
        x_t = x_t.unsqueeze(0).repeat(N, 1, 1, 1)
        y_t = y_t.unsqueeze(0).repeat(N, 1, 1, 1)
        z_t = z_t.unsqueeze(0).repeat(N, 1, 1, 1)
        z_t = z0 + z_t
        # f = f.view(N, 1, 1, 1)

        # back project
        x_t = x_t * z_t / f # image space in 3D?
        y_t = y_t * z_t / f # image space in 3D

        ones = torch.ones_like(x_t)
        grid = torch.stack([x_t, y_t, z_t, ones], dim=-1)

        return grid

    def camera2ray_grid(self, grid, f):
        grid = grid[..., 0:3] / grid[..., 3:4]
        N, D, H, W, _ = grid.size()

        x, y, z = torch.split(grid, 1, dim=-1)
        x = x.squeeze(-1)
        y = y.squeeze(-1)
        z = z.squeeze(-1)

        f = f.view(N, 1, 1, 1)
        x = f * x / z
        y = f * y / z
        z = z - geom_utils.calc_rho(f)

        x = x * 2
        y = y * 2
        z = z * 2 / self.z_range
        grid = torch.stack([x, y, z], dim=-1).view(N, D, H, W, 3)
        return grid

    def world_grid(self, N, out_size, out_depth, device):
        height = width = out_size

        x_t = torch.linspace(-0.5, 0.5, width, dtype=torch.float32, device=device)
        y_t = torch.linspace(-0.5, 0.5, height, dtype=torch.float32, device=device)
        z_t = torch.linspace(-0.5, 0.5, out_depth, dtype=torch.float32, device=device)
        z_t, y_t, x_t = torch.meshgrid(z_t, y_t, x_t)

        x_t = x_t.unsqueeze(0).expand(N, out_depth, out_size, out_size)
        y_t = y_t.unsqueeze(0).expand(N, out_depth, out_size, out_size)
        z_t = z_t.unsqueeze(0).expand(N, out_depth, out_size, out_size)

        ones = torch.ones_like(x_t)
        grid = torch.stack([x_t, y_t, z_t, ones], dim=-1)
        return grid

    def project(self, voxels, views, f, out_size=32, out_depth=32):
        N = voxels.size(0)
        wTc = get_camera2world(views, f)  # camera-to-world matrix
        cGrid = self.ray2camera_grid(out_size, out_depth, f, N, device=voxels.device)
        wGrid = torch.matmul(cGrid.view(N, -1, 4), wTc.transpose(1, 2)).view(N, out_depth, out_size, out_size, 4)
        wGrid = 2 * wGrid[..., 0:3] / wGrid[..., 3:4]  # scale from [0.5, 0.5] to [-1, 1]
        voxels = F.grid_sample(voxels, wGrid, align_corners=True)
        return voxels

    def backproj(self, voxels, views, f, out_size=32, out_depth=32):
        N = voxels.size(0)
        cTw = get_world2camera(views, f)  # world-to-camera matrix
        wGrid = self.world_grid(N, out_size, out_depth, device=voxels.device)
        cGrid = torch.matmul(wGrid.view(N, -1, 4), cTw.transpose(1, 2)).view(N, out_depth, out_size, out_size, 4)
        rGrid = self.camera2ray_grid(cGrid, f)
        voxels = F.grid_sample(voxels, rGrid, align_corners=True)
        return voxels

    def forward(self,  voxels, views, out_size=32, out_depth=32, inv=False, f=None):
        f = self.f if f is None else f
        if inv:
            return self.project(voxels, views, f, out_size, out_depth)
        else:
            return self.backproj(voxels, views, f, out_size, out_depth)


#### Pose to world-to-view-mat ####
def get_world2camera(param, f):
    """
    :param param:
    :return: [sR, t]
    """
    # scale, rot, trans, = geom_utils.param_to_srtf(param)
    scale, trans, rot = param
    trans[..., -1] = geom_utils.calc_rho(f)

    # legacy: rot is rot^T
    rot = geom_utils.homo_to_3x3(rot)
    rot = rot.transpose(1, 2)
    rt = geom_utils.rt_to_homo(rot, trans)
    scale = geom_utils.diag_to_homo(scale)
    cTw = torch.matmul(rt, scale)
    return cTw


def get_camera2world(param, f):
    # scale, rot, trans, = geom_utils.param_to_srtf(param)
    scale, trans, rot = param
    trans[..., -1] = geom_utils.calc_rho(f)

    # legacy: rot is rot^T
    rot_inv = geom_utils.homo_to_3x3(rot)  # N, 3, 3
    rt_inv = geom_utils.rt_to_homo(rot_inv, -torch.matmul(rot_inv, trans.unsqueeze(-1)))
    scale_inv = geom_utils.diag_to_homo(1 / scale)
    wTc = torch.matmul(scale_inv, rt_inv)
    return wTc


if __name__ == '__main__':
    device = 'cuda'
    layer = Perspective3d().to(device)
    H = W = D = 16
    N = 1
    vox = torch.zeros([N, 1, D, H, W], device=device)

    param = torch.FloatTensor([0, 0, 1, 0, 0, 2])
    view = geom_utils.azel2uni(param)
    f = 375