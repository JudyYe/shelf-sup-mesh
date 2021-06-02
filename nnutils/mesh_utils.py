# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import pytorch3d.ops as ops_3d
from pytorch3d.renderer import MeshRasterizer, SfMPerspectiveCameras, TexturesVertex, MeshRenderer, SoftGouraudShader, \
    DirectionalLights, RasterizationSettings

from pytorch3d.structures import Meshes
import pytorch3d.structures.utils as struct_utils

from nnutils import geom_utils


def cubify(vox_world, th=0.1, detach_vox=True) -> Meshes:
    """
    scale range from -0.5, 0.5
    :param vox_world: （N， C， D, H, W)
    :param th:
    :return:
    """
    if not torch.is_tensor(vox_world):
        W = vox_world.shape[-1]; N = vox_world.shape[0]
        vox_world = torch.FloatTensor(vox_world).view(N, 1, W, W, W).cuda()
    if detach_vox:
        vox_world = vox_world.detach()
    meshes = ops_3d.cubify(vox_world.squeeze(1), th, align='corner')
    meshes = meshes.scale_verts_(0.5)
    return meshes

def param_to_7dof_batcch(param, f=375, use_scale=False, use_rho=False):
    """
    :param param: (N, 6)
    :param f: scaler
    :param use_scale:
    :param use_rho:
    :return:
    """
    N, C = param.size()
    azel, scale, trans = torch.split(param, [2, 1, 3], dim=1)
    zeros = torch.zeros_like(scale)
    if not use_scale:
        scale = zeros + 1
    if not use_rho:
        trans = torch.cat([zeros, zeros, zeros + calc_rho(f)], dim=1)
    f = zeros + f
    new_param = torch.cat([azel, scale, trans, f], dim=1)
    return new_param


def calc_rho(f):
    base_f = 1.875
    base_rho = 2
    rho = base_rho * f / base_f
    return rho


def view_vox2mesh_py3d(view):
    """
    :param view: (N, 7)
    :return: (N, ), (N, 3, 3), (N, 3)
    """
    view = view.clone()
    view, f = torch.split(view, [6, 1], dim=1)
    view[:, 0] = -view[:, 0]
    f = (f * 2).squeeze(1)

    scale, trans, rot = geom_utils.azel2uni(view, homo=False)
    return f, rot, trans


def param7dof_to_camera(view_params) -> SfMPerspectiveCameras:
    """
    :param view_params: (N, 7)
    :return: SfM cameras
    """
    f, rot, trans = view_vox2mesh_py3d(view_params)
    cameras = SfMPerspectiveCameras(focal_length=f, R=rot, T=trans, device=view_params.device)
    return cameras


def render_meshify_voxel(voxels, out_size, view_param, th=0.05):
    meshes = cubify(voxels, th)
    try:
        recon = render_mesh(meshes, out_size, view_param)
    except:
        print('No mesh')
        N = voxels.size(0)
        recon = {'image': torch.zeros(N, 3, out_size, out_size)}
    return recon


def render_mesh(meshes: Meshes, out_size, view_param, texture=None, **kwargs):
    N, V, _ = meshes.verts_padded().size()
    if meshes.textures is None:
        if texture is None:
            texture = torch.zeros([N, V, 3]).to(view_param) + 1 # torch.FloatTensor([[[175., 175., 175.]]]).to(view_param) / 255
        meshes.textures = pad_texture(meshes, texture)
    cameras = param7dof_to_camera(view_param)

    raster_settings = kwargs.get('raster_settings', RasterizationSettings(image_size=out_size))
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    shader = SoftGouraudShader(device=meshes.device, lights=ambient_light(meshes.device, view_param))

    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader
    )
    if 'zfar' not in kwargs:
        kwargs['zfar']= view_param[:, -2].view(N, 1, 1, 1) + 1
    if 'znear' not in kwargs:
        kwargs['znear'] = view_param[:, -2].view(N, 1, 1, 1) - 1

    image = renderer(meshes, cameras=cameras,  **kwargs)

    image = torch.flip(image, dims=[-3])
    image = image.transpose(-1, -2).transpose(-2, -3)  # H, 4, W --> 4, H, W
    rgb, mask = torch.split(image, [image.size(1) - 1, 1], dim=1)  # [0-1]

    image = image * 2 - 1
    # rgb, mask = torch.split(output, [3, 1], dim=1)
    return {'image': rgb, 'mask': mask, 'rgba': image}


def render_normals(meshes: Meshes, out_size, view_param, **kwargs):
    N, V, _ = meshes.verts_padded().size()
    # clone mesh to and replace texture with normals in camera space
    meshes = meshes.clone()
    world_normals = meshes.verts_normals_padded()
    cameras = param7dof_to_camera(view_param)  # real camera
    trans_world_to_view = cameras.get_world_to_view_transform()
    view_normals = trans_world_to_view.transform_normals(world_normals)

    # place view normal as textures
    meshes.textures = pad_texture(meshes, view_normals)

    raster_settings = kwargs.get('raster_settings', RasterizationSettings(image_size=out_size))
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    # make the ambient color full range
    shader = SoftGouraudShader(device=meshes.device, lights=ambient_light(meshes.device, view_param, color=(1, 0, 0)))

    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader
    )
    if 'zfar' not in kwargs:
        kwargs['zfar']= view_param[:, -2].view(N, 1, 1, 1) + 1
    if 'znear' not in kwargs:
        kwargs['znear'] = view_param[:, -2].view(N, 1, 1, 1) - 1

    image = renderer(meshes, cameras=cameras,  **kwargs)

    image = torch.flip(image, dims=[-3])
    image = image.transpose(-1, -2).transpose(-2, -3)  # H, 4, W --> 4, H, W
    rgb, mask = torch.split(image, [image.size(1) - 1, 1], dim=1)  # [0-1]

    # align w/ my def
    # flip r (x), b (z)
    rgb[:, 0] *= -1
    rgb[:, 2] *= -1
    # mask out bg
    rgb = rgb * mask
    # and normalize rgb to unit vector.
    rgb = F.normalize(rgb, dim=1)  # N, 3, H, W

    # rgb, mask = torch.split(output, [3, 1], dim=1)
    return {'normal': rgb, 'mask': mask, 'rgba': image}


def pad_texture(meshes: Meshes, feature: torch.Tensor) -> TexturesVertex:
    """
    :param meshes:
    :param feature: (sumV, C)
    :return:
    """
    if isinstance(feature, TexturesVertex):
        return feature
    if feature.dim() == 2:
        feature = struct_utils.packed_to_list(feature, meshes.num_verts_per_mesh().tolist())
        # feature = struct_utils.list_to_padded(feature, pad_value=-1)

    texture = TexturesVertex(feature)
    texture._num_faces_per_mesh = meshes.num_faces_per_mesh().tolist()
    texture._num_verts_per_mesh = meshes.num_verts_per_mesh().tolist()
    texture._N = meshes._N
    texture.valid = meshes.valid
    return texture



def ambient_light(device='cpu', param_view=None, **kwargs):
    amb = 0.6
    if param_view is None:
        d = get_light_direction(param_view)
    else:
        d = ((0, -0.6, 0.8), )

    color = kwargs.get('color', np.array([0.65, 0.3, 0.0]))
    am, df, sp = color
    ambient_color=((am, am, am), ),
    diffuse_color=((df, df, df),),
    specular_color=((sp, sp, sp), ),

    lights = DirectionalLights(
        device=device,
        ambient_color=ambient_color,
        diffuse_color=diffuse_color,
        specular_color=specular_color,
        direction=d,
    )
    return lights


def get_light_direction(view_params):
    """same el, opposite az"""
    N = view_params.size(0)
    az, el, _ = torch.split(view_params, [1, 1, view_params.size(-1) - 2], dim=-1)
    az = -az # np.pi

    rot = geom_utils.azel2rot(az, el, False)  # (N, 3, 3)
    unit = torch.zeros([N, 3, 1]).to(az)
    unit[:, 2] += 1  # z += 1
    unit = torch.matmul(rot, unit).squeeze(-1)
    return -unit


def get_soft_rasterizer_setting(**kwargs):
    sigma = kwargs.get('sigma', 1e-4)
    raster_settings_soft = RasterizationSettings(
        image_size=kwargs.get('image_size', 224),
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
        faces_per_pixel=kwargs.get('faces_per_pixel', 10),
        perspective_correct=False,
    )
    return raster_settings_soft

