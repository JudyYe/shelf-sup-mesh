# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import os.path as osp
import pickle

import absl.flags as flags
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import trimesh
from pytorch3d.loss import mesh_edge_loss, mesh_normal_consistency
from pytorch3d.structures import Meshes

from models.render_wrapper import MeshRenderWrapper
from models.texturizer import TextureFactory
from nnutils import geom_utils, image_utils, utils, mesh_utils
from nnutils.laplacian_loss import mesh_laplacian_smoothing

FLAGS = flags.FLAGS

max_num = 4


class Evaluator(object):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        print(cfg)
        if cfg is None:
            self.cfg = {
                "ele_low": -60,
                "ele_high": 60,
                "azi_low": -180,
                "azi_high": 180,
                "f": 375,
            }

        self.mesh_render = MeshRenderWrapper(224, 'gouraud', sigma=1e-5, faces_per_pixel=1).cuda()
        self.vis_texture = TextureFactory('pix', out_size=224).cuda()

    def quali_vox(self, model, dataloader, save_dir, prefix='model', datapoint=None):
        save_dir = os.path.join(save_dir, prefix)

        if datapoint is not None:
            dataloader = [datapoint]
        with torch.no_grad():
            for i, datapoint in enumerate(dataloader):
                datapoint = utils.to_cuda(datapoint)
                image_utils.save_images(datapoint['image'], osp.join(save_dir, '%d_input' % i), save_idv=True, save_merge=True)
                (batch_z, feat_img), (view_list, para_list, _) = model.encode_content(datapoint['image'])

                N = batch_z.size(0)
                device = batch_z.device
                feat_word, vox_world = model.decoder.reconstruct_can(batch_z, view_list)
                recon = model.decoder.render_k_views((feat_word, feat_img), vox_world, batch_z, view_list)
                image_utils.save_images(recon['mask'] * 2 - 1, osp.join(save_dir, '%d_vox' % i))

                vox_mesh = mesh_utils.cubify(vox_world, FLAGS.mesh_th)

                can_pose_list = []
                th_list = [0, np.pi, np.pi / 4, -np.pi / 4, np.pi / 4 + np.pi, -np.pi / 4 + np.pi]
                for th in th_list:
                    sample_view = torch.FloatTensor([[th, np.pi / 6, 1, 0, 0, 2]]).to(device).expand(N, 6)
                    can_pose_list.append(mesh_utils.param_to_7dof_batcch(sample_view, self.cfg['f']))

                image_utils.save_images(datapoint['mask'] * 2 - 1, osp.join(save_dir, '%d_inputM' % i), save_idv=True)
                image_utils.save_images(datapoint['image'], osp.join(save_dir, '%d_input' % i), save_idv=True)

                self.snapshot_mesh(vox_mesh, can_pose_list, None, save_dir, '%d' % i, 'voxCan', 2)

                if i > 1:
                    break

    def quali_opt(self, model, dataloader, save_dir, prefix='model', datapoint=None):
        name = 'n%gl%g' % (FLAGS.cyc_normal_loss, FLAGS.lap_norm_loss)
        save_dir = os.path.join(save_dir, prefix, name)
        os.makedirs(save_dir, exist_ok=True)
        # os.system('rm -r %s/*' % save_dir)
        self.save_dir = save_dir
        flag_str = FLAGS.flags_into_string()
        with open(os.path.join(self.save_dir, 'flags.txt'), 'w') as fp:
            fp.write(flag_str)

        if datapoint is not None:
            dataloader = [datapoint]
        for i, datapoint in enumerate(dataloader):
            datapoint = utils.to_cuda(datapoint)
            image_utils.save_images(datapoint['image'] * .5 + datapoint['bg'] * .5,
                            osp.join(save_dir, '%d_inputO' % i), save_idv=True)
            (recon_z, img_feat), (view_list, para_list, _) = model.encoder(datapoint['image'])
            if FLAGS.detach_view:
                para_list = para_list.detach()
            device = para_list.device;
            N = recon_z.size(0)
            pose_list = []
            th_list = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3]
            for th in th_list:
                sample_view = torch.FloatTensor([[th, 0, 1, 0, 0, 2]]).to(device).expand(N, 6)
                pose_list.append(mesh_utils.param_to_7dof_batcch(sample_view, self.cfg['f']))

            can_pose_list = []
            th_list = [np.pi / 4, -np.pi / 4, ]
            for th in th_list:
                sample_view = torch.FloatTensor([[th, np.pi / 6, 1, 0, 0, 2]]).to(device).expand(N, 6).clone()
                if FLAGS.dataset[0:2] in ['im'] or FLAGS.dataset == 'snCar':
                    sample_view[:, 0] = np.pi + sample_view[:, 0]
                sample_view = mesh_utils.param_to_7dof_batcch(sample_view, self.cfg['f'])
                can_pose_list.append(sample_view)

            feat_world, vox_world = model.decoder.reconstruct_can(recon_z, view_list)
            vox_mesh = mesh_utils.cubify(vox_world).clone()
            camera_param = mesh_utils.param_to_7dof_batcch(para_list, self.cfg['f'], use_scale=False,
                                                           use_rho=False).clone()

            ones = torch.ones_like(datapoint['image'])
            verts_rgb = self.vis_texture(vox_mesh, image=ones, view=camera_param, vis=1, sym=0)
            vox_mesh.textures = mesh_utils.pad_texture(vox_mesh, verts_rgb)

            self.snapshot_mesh(vox_mesh, can_pose_list, vox_mesh.textures, save_dir, '%d' % i, 'voxCan', 2)

            mesh_inputs = {'mesh': vox_mesh, 'view': camera_param}
            with torch.enable_grad():
                mesh_outputs, record = self.opt_mask(model, mesh_inputs, datapoint, True, 300)

            self.snapshot_mesh(mesh_outputs['mesh'], can_pose_list, mesh_outputs['mesh'].textures,
                               save_dir, '%d' % i, 'meshCan', 2)
            self.snapshot_mesh(mesh_outputs['mesh'], can_pose_list,
                               None, save_dir, '%d' % i, 'meshShapeCan', 2)

            image_list = {'pred': [], 'can': []}
            for t in range(len(record['mesh'])):
                image = mesh_utils.render_mesh(record['mesh'][t].cuda(), 224, camera_param)
                image_list['pred'].append(image['image'])
                image = mesh_utils.render_mesh(record['mesh'][t].cuda(), 224, can_pose_list[0])
                image_list['can'].append(image['image'])

            del record
            if i > 10:
                break

    def get_view_list(self, view_mod, **kwargs):
        """
        :param view_mod:
        :param x:
        :return: (T, 6)
        """
        view_dict = {
            'az': (self.cfg['azi_low'], self.cfg['azi_high'], 10 / 180 * np.pi),
            'el': (self.cfg['ele_low'], self.cfg['ele_high'], 5 / 180 * np.pi),
            'vaz': (0, 2 * np.pi, 10 / 180 * np.pi),
            'pred': (0, 2 * np.pi, 10 / 180 * np.pi),
        }
        time_len = int((view_dict[view_mod][1] - view_dict[view_mod][0]) / view_dict[view_mod][2])
        zeros = torch.zeros([1, time_len])
        if 'az' in view_mod or 'pred' in view_mod:
            vary_az = torch.linspace(view_dict[view_mod][0], view_dict[view_mod][1], time_len).unsqueeze(0)
            vary_el = torch.zeros([1, 1]).expand(1, time_len) + (self.cfg['ele_low'] + self.cfg['ele_high']) / 2
        elif 'el' in view_mod:
            vary_az = torch.zeros([1, 1]).expand(1, time_len)
            vary_el = torch.linspace(self.cfg['ele_low'], self.cfg['ele_high'], time_len).unsqueeze(0)
        elif view_mod == 'cfg':
            vary_az = torch.linspace(self.cfg['azi_low'], self.cfg['azi_highs'], time_len).unsqueeze(0)
            vary_el = torch.linspace(self.cfg['ele_low'], self.cfg['ele_high'], time_len).unsqueeze(0)
        else:
            raise NotImplementedError

        delta = torch.cat([vary_az, vary_el, zeros + 1, zeros, zeros, zeros], dim=0)  # (6, time_len)
        delta = delta.transpose(0, 1)  # (time_len, 6)
        return delta

    def render_mesh_rot(self, view_mod, mesh: Meshes, mesh_texture=None, view_param=None, **kwargs):

        delta = self.get_view_list(view_mod).to(mesh.verts_padded())
        time_len = delta.size(0)
        num = len(mesh)
        image_list = []
        render = kwargs.get('render', self.mesh_render)
        for t in range(time_len):
            if 'pred' in view_mod:
                view = delta[t: t + 1].expand(num, 6).clone()
                view[..., 0:2] = view_param[..., 0:2] + view[..., 0:2]
            else:
                view = delta[t:t + 1].expand(num, 6)
            param = mesh_utils.param_to_7dof_batcch(view, self.cfg['f'])
            vox_recon = render(mesh, param, mesh_texture, light_direction=mesh_utils.get_light_direction(param))
            # todo
            image_list.append(vox_recon['image'])
        return image_list

    def vis_view(self, model, data, num=16, view_mod='az', save_dir=None, prefix='model', random_z=False):
        real_image = data['image'][0: num]
        # batch_z, batch_view = model.encoder(real_image)
        batch_z, batch_view = model.encode_content(real_image)
        batch_z, feat = batch_z

        recon = model.decoder((batch_z, feat), batch_view[0])

        if random_z:
            batch_z = torch.randn_like(batch_z) + batch_z
        else:
            images = image_utils.merge_to_numpy(real_image, n_col=min(4, num))
            imageio.imsave(save_dir + '/%s_gt.jpg' % prefix, images)
            images = image_utils.merge_to_numpy(data['mask'][0:num], n_col=min(4, num))
            imageio.imsave(save_dir + '/%s_gt_m.jpg' % prefix, images)
            images = image_utils.merge_to_numpy(recon['image'], n_col=min(4, num))
            imageio.imsave(save_dir + '/%s_recon.jpg' % prefix, images)
            images = image_utils.merge_to_numpy(recon['mask'], n_col=min(4, num))
            imageio.imsave(save_dir + '/%s_recon_m.jpg' % prefix, images)

        delta = self.get_view_list(view_mod).to(batch_z)
        time_len = delta.size(0)

        vary_view = geom_utils.azel2uni(delta)
        vary_view = geom_utils.expand_uni(vary_view, num)  # [T, N, 6]
        image_list = [];
        mean_list = []
        mask_list = [];
        vox_list = []
        for t in range(time_len):
            _, vox_world = model.decoder.reconstruct_can(batch_z, [vary_view[t]])
            param = mesh_utils.param_to_7dof_batcch(delta[t: t + 1].expand(num, 6), self.cfg['f'])
            vox_recon = mesh_utils.render_meshify_voxel(vox_world, 224, param)
            recon = model.decoder((batch_z, feat), vary_view[t])
            image_list.append(vox_recon['image'])
            mask_list.append(recon['mask'])
            vox_list.append(recon['image'])
            if FLAGS.know_vox:
                mean_recon = mesh_utils.render_meshify_voxel(data['vox'][0: num], 224, param)
                mean_list.append(mean_recon['image'])

        if save_dir is not None:
            image_utils.save_gifs(image_list, osp.join(save_dir, '%s_%s_meshify_vox' % (prefix, view_mod)))
            image_utils.save_gifs(mean_list, osp.join(save_dir, '%s_%s_meshify_mean' % (prefix, view_mod)))
            image_utils.save_gifs(vox_list, osp.join(save_dir, '%s_%s_vox' % (prefix, view_mod)))
            image_utils.save_gifs(mask_list, osp.join(save_dir, '%s_%s_voxM' % (prefix, view_mod)))

        return image_list

    def scatter_view(self, model, dataloader, save_dir, prefix):
        view_list = np.empty([0, 6])
        for data in dataloader:
            data = utils.to_cuda(data)
            real_image = data['image']
            # batch_z, batch_view = model.encoder(real_image)
            batch_z, batch_view = model.encode_content(real_image)

            batch_view = batch_view[1].cpu().detach().numpy()
            view_list = np.vstack([view_list, batch_view])
            if view_list.shape[0] > 200:
                break
        fpath = os.path.join(save_dir, '%s_%s' % (prefix, 'viewpoint'))
        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(os.path.dirname(fpath))
        image_utils.scatter_sphere(view_list[:, 0], view_list[:, 1], fpath)

    def scatter_view_param(self, param, save_dir, prefix):
        az, el, _ = torch.split(param, [1, 1, param.size(-1) - 2], dim=-1)
        az = az.cpu().detach().numpy().reshape([-1])
        el = el.cpu().detach().numpy().reshape([-1])
        fpath = os.path.join(save_dir, '%s_%s' % (prefix, 'viewpoint'))
        image_utils.scatter_sphere(az, el, fpath)

    def dump_mesh(self, meshes: Meshes, camera_para, index_list, save_dir, prefix, name):
        verts = meshes.verts_list()
        faces = meshes.faces_list()
        textures = meshes.textures.verts_features_list()
        f, rot, trans = mesh_utils.view_vox2mesh_py3d(camera_para)
        f, rot, trans = f.cpu().detach().numpy(), rot.cpu().detach().numpy(), trans.cpu().detach().numpy()
        for n in range(len(meshes)):
            fname = os.path.join(save_dir, '%s_%d_%s.ply' % (prefix, n, name))
            text = ((textures[n].cpu().detach().numpy() / 2 + 0.5) * 255).astype(np.uint8)
            trimesh.Trimesh(verts[n].cpu().detach().numpy(),
                            faces[n].cpu().detach().numpy(),
                            vertex_colors=text).export(fname)
            # io3d().save_mesh(mesh, fname)
            print(fname)
            fname = os.path.join(save_dir, '%s_%d_%s.pickle' % (prefix, n, name))
            with open(fname, 'wb') as fp:
                pickle.dump({'pose': camera_para[n], 'f': f[n], 'rot': rot, 'trans': trans, 'index': index_list[n]}, fp)
            print(fname)

    def snapshot_mesh(self, meshes: Meshes, view_list, texture, save_dir, prefix, name, merge=0, **kwargs):
        mode = kwargs.get('mode', 'az')
        if 'pred_view' in kwargs:
            # if pred_view is set, then it's in that coordinate.
            pred_view = kwargs.get('pred_view', None)
            meshes = mesh_utils.transform_meshes(meshes, pred_view)
            mode = 'v' + mode

        render = kwargs.get('render', self.mesh_render)
        l_c = kwargs.get('l_c', np.array([0.65, 0.3, 0.0]))
        for v, view_param in enumerate(view_list):
            if view_param.size(-1) == 6:
                view_param = mesh_utils.param_to_7dof_batcch(view_param, self.cfg['f'])
            l_u = mesh_utils.get_light_direction(view_param)
            image = render(meshes, view_param, texture, light_direction=l_u, light_color=l_c)
            image_utils.save_images(image['image'], osp.join(save_dir, '%s_%s_v%d' % (prefix, name, v)), scale=True)

        image = self.render_mesh_rot(mode, meshes, texture, render=render)
        image_utils.save_gifs(image, osp.join(save_dir, '%s_%s_az' % (prefix, name)), scale=True)

    def opt_mask(self, model, vox_inp, real_datapoint, opt_view=False, nstep=100):
        vox_mesh, camera_param = vox_inp['mesh'], vox_inp['view']
        device = camera_param.device

        verts_shape = vox_mesh.verts_packed().shape
        deform_verts = torch.nn.Parameter(torch.full(verts_shape, 0.0, device=device, requires_grad=True))

        # The optimizer
        params = [deform_verts]
        if opt_view:
            camera_param = torch.nn.Parameter(camera_param, True)
            params += [camera_param]
        optimizer = torch.optim.Adam(params, lr=1e-4, )

        loop = tqdm.tqdm(range(nstep))
        record = {'image': [], 'mask': [], 'soft': [], 'normal': [], 'mesh': []}

        for i in loop:
            # Initialize optimizer
            optimizer.zero_grad()
            model.zero_grad()

            # Deform the mesh
            new_vox_mesh = vox_mesh.offset_verts(deform_verts)
            # verts_rgb = self.vis_texture(new_vox_mesh, image=real_datapoint['image'], view=camera_param)
            verts_rgb = self.vis_texture(new_vox_mesh, image=real_datapoint['bg'], view=camera_param)
            new_vox_mesh.textures = mesh_utils.pad_texture(new_vox_mesh, verts_rgb)

            loss, deform_verts, render = self.fw_bw(new_vox_mesh, camera_param, deform_verts, real_datapoint)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optimization step
            # save
            if i % 5 == 0:
                record['mesh'].append(new_vox_mesh.cpu())
                new_vox_mesh.cpu()
        return {'mesh': new_vox_mesh, 'view': camera_param}, record

    def fw_bw(self, new_vox_mesh, camera_param, deform_verts, real_datapoint):
        raster_settings = mesh_utils.get_soft_rasterizer_setting(image_size=224)

        normals = mesh_utils.render_normals(new_vox_mesh, 224, camera_param, raster_settings=raster_settings)
        render = mesh_utils.render_mesh(new_vox_mesh, 224, camera_param, raster_settings=raster_settings)
        render['normal'] = normals['normal']

        # laplacian
        loss = FLAGS.lap_loss * mesh_laplacian_smoothing(new_vox_mesh, None, method=FLAGS.lap_method)
        loss += FLAGS.lap_loss * mesh_edge_loss(new_vox_mesh)
        loss += FLAGS.lap_norm_loss * FLAGS.lap_loss * mesh_normal_consistency(new_vox_mesh)

        if deform_verts is not None:
            loss += FLAGS.delta_loss * deform_verts.pow(2).mean()
        # print('lap', loss)
        loss += self.cyc_loss(render, real_datapoint)
        return loss, deform_verts, render

    def cyc_loss(self, render, real, reduction='mean'):
        N = real['image'].size(0)
        loss = 0

        loss = loss + FLAGS.cyc_mask_loss * (render['mask'] - real['mask']).abs().reshape(N, -1).mean(dim=-1)
        loss = loss + FLAGS.cyc_loss * (render['image'] - real['image']).abs().reshape(N, -1).mean(dim=-1)

        if reduction == 'mean':
            loss = loss.mean()
            # print('cyc', loss)
        return loss
