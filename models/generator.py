# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
import numpy as np
import json
from typing import Dict, List
import absl.flags as flags
import torch.nn.functional as F
from pytorch3d.structures import Meshes

from models.render_wrapper import MeshRenderWrapper
from nnutils import mesh_utils, geom_utils
from nnutils.layers import *
from .transformation import Perspective3d
from .encoder import EncoderFactory

FLAGS = flags.FLAGS


class ReconstructModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = EncoderFactory(FLAGS.enc, FLAGS.z_dim, c_dim=3, norm=FLAGS.e_norm)

        self.decoder = Decoder(FLAGS.infer, FLAGS.vol_render,
                               FLAGS.gf_dim, FLAGS.c_dim, FLAGS.z_dim, FLAGS.vol_norm)

    def forward_image(self, fg):
        (batch_z, feat_img), (view_list, para_list, param_u) = self.encoder(fg)
        feat_word, vox_world = self.decoder.reconstruct_can(batch_z, view_list)
        pred_view = mesh_utils.param_to_7dof_batcch(para_list, self.decoder.cfg['f'])
        return vox_world, pred_view

    def encode_content(self, image):
        (recon_z, img_feat), (view_v, param_v, param_u) = self.encoder(image)
        return (recon_z, img_feat), (view_v, param_v, param_u)


def Infer3DFactory(name, num_layers, gf_dim, z_dim, norm):
    infer_dict = {
        'style': StyleInfer3D,
    }
    if name in infer_dict:
        return infer_dict[name](num_layers, gf_dim, z_dim, norm)


def ReaderFactory(name, num_layers, gf_dim, z_dim, c_dim, norm):
    reader = {
        'struct': StyleStructRender,
        'sim': SimStructRender,
        'rgb': RgbStructRender,
    }
    if name in reader:
        return reader[name](num_layers, gf_dim, z_dim, c_dim, norm)


class Decoder(nn.Module):
    def __init__(self, infer, reader, gf_dim=64, c_dim=3, z_dim=128, norm='instance', df_dim=64):
        super().__init__()
        self.output_keys = ['image', 'mask', 'flow', 'mesh', 'mesh_delta', 'verts_pix']
        with open(FLAGS.cfg_file) as fp:
            self.cfg = json.load(fp)

        vox_total_layers = int(math.log2(FLAGS.low_reso // 4))
        can_layers = int(math.log2(FLAGS.reso_3d // 4))

        self.infer3d_net = Infer3DFactory(infer, can_layers, gf_dim, z_dim, norm)
        self.view_sampler = ViewSampler()

        self.render_net = ReaderFactory(reader, vox_total_layers - can_layers, self.infer3d_net.out_dim,
                                        z_dim, c_dim, norm)
        self.mesh_renderer = MeshRenderWrapper(
            FLAGS.high_reso, FLAGS.mesh_render,
            FLAGS.render_sigma, np.log(1. / 1e-4 - 1.) * FLAGS.render_sigma , FLAGS.render_faces,
        )  # for backward loss, thus radius > 0

        self.mesh_renderer_eval = self.mesh_renderer

        # Unet for local feature
        num = int(math.log2(FLAGS.high_reso // FLAGS.reso_local))
        dims_list = [df_dim * 2 ** i for i in range(num)]
        dims_list.insert(0, c_dim)
        block = dis_block(num, dims_list, norm=FLAGS.e_norm)
        block.append(conv2d(dims_list[-1], self.infer3d_net.out_dim, k=1, d=1))
        self.feat_tower = nn.Sequential(*block)

    def forward(self, *input, **kwargs):
        """
        :param batch_z: (N, Dz)
        :param batch_view: (N, 1), (N, 3), (N, 4, 4)
        :param kwargs:
        :return:
        """
        (batch_z, feat_img), batch_view= input

        # 1. generate view independent 3d representation
        #  some 3D representation: voxel (N, C+1, D, H, W)
        feat_world, vox_world = self.infer3d_net(batch_z, batch_view)

        # 3. render a image
        #  w/ diff. aggregation
        results = {'3d': vox_world}
        render = self.render_net((feat_world, feat_img), vox_world, batch_view, batch_z)
        for each in self.output_keys:
            if each in render:
                results[each] = render[each]
        return results

    def reconstruct_can(self, *input):
        feat_world, vox_world = self.infer3d_net(*input)
        return feat_world, vox_world

    def render_k_views(self, feat_world, vox_world, batch_z, batch_view):
        render = self.render_net(feat_world, vox_world, batch_view, batch_z)
        return render


# ##################### Infer 3D ###########################
class StyleInfer3D(nn.Module):
    def __init__(self, num_layers, gf_dim, z_dim, norm):
        super().__init__()
        self.base = AdaBlock(None, gf_dim * 4, gf_dim * 4, z_dim, add_conv=False, relu='relu')

        dims_list = [gf_dim * 4 // 2**i for i in range(num_layers + 1)]
        self.out_dim = dims_list[-1]
        self.gblock3d_1 = build_gblock(3, num_layers, dims_list, k=3, d=2, p=1, op=1, adain=True,
                                       relu='leaky', z_dim=z_dim)
        std_dev = 0.02;
        w = 4
        self.const_w = nn.Parameter(torch.randn(1, gf_dim * 4, w, w, w) * std_dev)

        dims = [dims_list[-1] for i in range(int(math.log2(FLAGS.reso_vox // FLAGS.reso_3d)) + 1)]
        if len(dims) > 1:
            layers = build_gblock(3, 1, dims, False, norm=norm, relu='leaky', k=3, d=2, p=1, op=1, last_relu=False)
        else:
            layers = []
        layers.extend(build_gblock(3, 1, [dims[-1], 1], False, norm='none', relu='leaky', k=3, d=1, p=1, last_relu=False))
        self.voxel_net = nn.Sequential(
            *layers,
            nn.Sigmoid(),
        )

    def forward(self, *input, **kwargs):
        """
        :param input: batch_z (N, Dz)
        :param kwargs:
        :return: some 3D representation: voxel (N, C+1, D, H, W)
        """
        batch_z, views = input
        _, C, D, H, W = self.const_w.size()
        N = batch_z.size(0)
        x = self.const_w.expand(N, C, D, H, W)
        x = self.base(batch_z, x)

        # View-Independent 3D feature
        for gnet in self.gblock3d_1:
            x = gnet(batch_z, x)
        x = self.apply_symmetry(x)

        head = self.head_feat(x, views)
        mask = self.voxel_net(x)

        return head, mask

    def apply_symmetry(self, x):
        """
        apply reflection on W-axis
        :param x: (N, C, D, H, W)
        :return:
        """
        if FLAGS.apply_sym > 0:
            x_flip = torch.flip(x, dims=[-1])
            x = (x + x_flip) / 2
        return x

    def head_feat(self, x, views):
        return x



# ##################### Renderer ###########################
class StyleStructRender(nn.Module):
    def __init__(self, num_layers, inp_dim, z_dim, c_dim, norm):
        super().__init__()
        self.inp_dim = inp_dim
        self.inp_size = inp_size = FLAGS.reso_3d
        # 16 -> 128 3

        self.transformer = Perspective3d()

        dims_list = [inp_dim] * 3
        self.gblock3d_2 = nn.Sequential(
            *build_gblock(3, 2, dims_list, adain=False, norm=norm, relu='leaky', k=3, d=1, p=1))

        dims_list = [int(inp_dim // 2**i) for i in range(num_layers + 1)]
        self.gblock2d_2 = build_gblock(2, num_layers, dims_list, k=4, d=2, p=1,
                                       adain=True, z_dim=z_dim, relu='leaky')
        self.head = nn.Sequential(
            deconv2d(dims_list[-1], c_dim, k=3, d=1, p=1),
            nn.Tanh())

    def forward(self, *input, **kwargs):
        """
        :param x: feat3d (N, C, D, H, W)
        :param voxel: occupancy (N, 1, D, H, W)
        :param view: (N, 7)
        :param z: (N, D)
        :return:
        """
        (x, _), voxel, view, batch_z = input

        x = self.transformer.project(x, view, FLAGS.reso_3d)
        x = self.gblock3d_2(x)

        # Sample for feature
        voxel_trans = self.transformer.project(voxel, view, FLAGS.reso_3d)
        x = expected_wrt_occupancy(x, voxel_trans)
        # Sample for mask
        voxel_trans = self.transformer.project(voxel, view, FLAGS.reso_vox)
        prob = occupancy_to_prob(voxel_trans)
        mask = expected_wrt_prob(1, prob)

        # Appearance
        for gnet in self.gblock2d_2:
            x = gnet(batch_z, x)
        rgb = self.head(x)

        # Normal
        pred_normal = grad_occ(voxel_trans)  # (N, 3, D, H, W)
        exp_normal = expected_wrt_prob(pred_normal, prob)
        exp_normal = exp_normal / exp_normal.norm(dim=1, keepdim=True).clamp(min=1e-8) # normalize
        return {'mask': mask, 'image': rgb, 'normal': exp_normal, '3d_view': voxel_trans, }


class SimStructRender(StyleStructRender):
    def __init__(self, num_layers, inp_dim, z_dim, c_dim, norm):
        super().__init__(num_layers, inp_dim, z_dim, c_dim, norm)
        dims_list = [int(inp_dim // 2**i) for i in range(num_layers + 1)]
        self.gblock2d_2 = build_gblock(2, num_layers, dims_list, k=4, d=2, p=1,
                                       adain=False, norm=norm, relu='leaky')

    def forward(self, *input, **kwargs):
        (x, _), voxel, view, _ = input

        x = self.transformer(x, view, FLAGS.reso_3d)
        x = self.gblock3d_2(x)

        # Sample for feature
        voxel_trans = self.transformer(voxel, view, FLAGS.reso_3d)
        x = expected_wrt_occupancy(x, voxel_trans)
        # Sample for mask
        voxel_trans = self.transformer(voxel, view, FLAGS.reso_vox)
        prob = occupancy_to_prob(voxel_trans)
        mask = expected_wrt_prob(1, prob)

        feat2d = x
        # Appearance
        for gnet in self.gblock2d_2:
            x = gnet(x)
        x = self.head(x)

        # Normal
        pred_normal = grad_occ(voxel_trans)  # (N, 3, D, H, W)
        exp_normal = expected_wrt_prob(pred_normal, prob)
        exp_normal = exp_normal / exp_normal.norm(dim=1, keepdim=True).clamp(min=1e-8) # normalize

        return {'mask': mask, 'image': x, '3d_view': voxel_trans, 'normal': exp_normal, 'feat2d': feat2d}


class RgbStructRender(nn.Module):
    def __init__(self, num_layers, inp_dim, z_dim, c_dim, norm):
        super().__init__()
        self.aggr_net = Aggregator('net', inp_dim=inp_dim * FLAGS.reso_vox, out_dim=inp_dim, norm=norm)
        self.aggr_mask = Aggregator('net', inp_dim=1 * FLAGS.reso_vox, out_dim=1)

        self.inp_dim = inp_dim
        self.inp_size  = FLAGS.reso_3d
        # 16 -> 128 3

        self.transformer = Perspective3d()

        dims_list = [inp_dim] * 3
        self.gblock3d_2 = nn.Sequential(
            *build_gblock(3, 2, dims_list, adain=False, norm=norm, relu='leaky', k=3, d=1, p=1))

        dims_list = [int(inp_dim // 2**i) for i in range(num_layers + 1)]
        self.gblock2d_2 = build_gblock(2, num_layers, dims_list, k=4, d=2, p=1,
                                       adain=True, z_dim=z_dim, relu='leaky')
        self.head = nn.Sequential(
            deconv2d(dims_list[-1], c_dim, k=3, d=1, p=1),
            nn.Tanh())

    def forward(self, *input, **kwargs):
        (x, _), voxel, view, batch_z = input

        x = self.transformer(x, view, x.size(-1))
        x = self.gblock3d_2(x)

        # Sample for feature
        x = self.aggr_net(x)

        # Sample for mask
        voxel_trans = self.transformer(voxel, view, FLAGS.reso_vox)
        # mask = F.sigmoid(self.aggr_mask(voxel_trans))
        mask = self.aggr_mask(voxel_trans)

        # Appearance
        for gnet in self.gblock2d_2:
            x = gnet(batch_z, x)
        x = self.head(x)

        # Normal
        pred_normal = grad_occ(voxel_trans)  # (N, 3, D, H, W)
        exp_normal = expected_wrt_occupancy(pred_normal, voxel_trans)
        exp_normal = exp_normal / exp_normal.norm(dim=1, keepdim=True).clamp(min=1e-8)  # normalize

        return {'mask': mask, 'image': x, '3d_view': voxel_trans, 'normal': exp_normal}


class Aggregator(nn.Module):
    def __init__(self, aggr, inp_dim=0, out_dim=0, norm='none'):
        super().__init__()
        self.aggr = aggr
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        if aggr == 'net':
            self.gblock2d_1 = build_gblock(2, 1, [inp_dim, out_dim], k=1, d=1, adain=False, norm=norm, relu='leaky')

    def forward(self, *input, **kwargs):
        """
        :param voxels: (N, Cin, D, H, W)
        :param kwargs:
        :return: (N, Cout, H, W)
        """
        if len(input) == 1:
            x, = input
        elif len(input) == 2:
            rgb, vox = input

        if self.aggr == 'net':
            N, C, D, H, W = x.size()
            # x = x.transpose(1, 2).contiguous()
            x = x.view(N, C * D, H, W)
            for gnet in self.gblock2d_1:
                x = gnet(x)
        else:
            # legacy: previously an abstraction for explicit / implict ray marching
            raise NotImplementedError
        return x


class ViewSampler(nn.Module):
    def __init__(self):
        super().__init__()
        with open(FLAGS.cfg_file) as fp:
            self.cfg = json.load(fp)
            self.cfg['azi_low'] = self.cfg['azi_low'] / 180 * np.pi
            self.cfg['azi_high'] = self.cfg['azi_high'] / 180 * np.pi
            self.cfg['ele_low'] = self.cfg['ele_low'] / 180 * np.pi
            self.cfg['ele_high'] = self.cfg['ele_high'] / 180 * np.pi

        self.sample_method = FLAGS.sample_view

    def forward(self, N, device, para_u=None, view_u=None):
        """
        :param N:
        :param device:
        :param para_u:
        :param view_u:
        :return: triplet: (rot,azel,u6). tri_new, tri_orig, delta in (azel)
        """
        if para_u is None:
            view_u, para_u = geom_utils.sample_view(self.sample_method, N, device, cfg=self.cfg)

        delta = torch.zeros_like(para_u[:, 0:2])
        view_v, para_v = view_u, para_u

        if FLAGS.so3_enc == 6:
            v6 = geom_utils.azel2u6d(para_v)
            u6 = geom_utils.azel2u6d(para_u)
        else:
            # v6, u6 = para_v[:, 0:2], para_u[:, 0:2]
            v6, u6 = para_v, para_u

        return (view_v, para_v, v6), (view_u, para_u, u6), delta
