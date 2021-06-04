# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import absl.flags as flags
import torch
import torch.nn.functional as F
import pytorch3d.ops as op_3d

from nnutils import geom_utils
from .encoder import sample_z as enc_sample_z
from .trainer import BaseTrainer
FLAGS = flags.FLAGS


class VoxTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

    def train_step(self, real_datapoint):
        counter = self.counter
        real_img = real_datapoint['image']
        real_mask = real_datapoint['mask']
        balance = 1. / (FLAGS.d_loss_recon + FLAGS.d_loss_holo + FLAGS.d_loss_hallc + 1e-6)
        N = real_mask.size(0)
        device = real_img.device
        # =======================================================================================================
        #   (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # =======================================================================================================
        # Compute adversarial loss toward discriminator
        for key in self.D_list:
            self.D_list[key].zero_grad()

        # 1. real
        d_loss = self.summerize_d_real_loss({'image': real_img, 'mask': real_mask, 'normal': None},
                                            'real', FLAGS.d_loss_real)

        # 2. generate fake image
        key = 'image_inp' if FLAGS.cyc_loss else 'mask_inp'
        (recon_z, img_feat), (view_v, param_v, param_u) = self.G.encode_content(real_datapoint[key])

        feat_world, vox_world = self.G.decoder.reconstruct_can(recon_z, view_v)
        recon = self.G.decoder.render_k_views((feat_world, img_feat), vox_world, recon_z, view_v)
        recon['3d'] = vox_world
        recon['v_mu'] = param_v
        self.recon = recon

        if bool(FLAGS.d_loss_recon):
            d_loss = d_loss + self.summerize_d_fake_loss(recon, 'recon', FLAGS.d_loss_recon * balance)

        if bool(FLAGS.d_loss_holo):
            (holo_view, holo_v, holo_u), (_, holo_v_base, _), delta_view1 = self.G.decoder.view_sampler(N, device)

            holo = self.G.decoder.render_k_views((feat_world, img_feat), vox_world, recon_z, holo_view)
            d_loss = d_loss + self.summerize_d_fake_loss(holo, 'holo', FLAGS.d_loss_holo * balance)
            self.holo = holo

        if bool(FLAGS.d_loss_hallc):
            (sample_view2, _), (_, param_u2), delta_view2 = self.G.decoder.view_sampler(N, device)
            sample_z = enc_sample_z(recon_z)
            samp = self.G.decoder((sample_z, None), sample_view2)
            d_loss = d_loss + self.summerize_d_fake_loss(samp, 'samp', FLAGS.d_loss_hallc * balance)
            self.samp = samp

        if counter % 2 == 0:
            d_loss.backward()
            self.d_opt.step()

        # =======================================================================================================
        #   (2) Update G network: maximize log(D(G(z)))
        # =======================================================================================================
        self.G.encoder.zero_grad()
        self.G.decoder.zero_grad()
        g_loss, cyc_loss = torch.zeros([1]).cuda(), torch.zeros([1]).cuda()

        # loss on G
        if bool(FLAGS.d_loss_recon):
            g_loss_k = self.summerize_g_loss(recon, 'recon', FLAGS.d_loss_recon * balance)
            g_loss += g_loss_k

        cyc_loss += self.summerize_cyc_loss(recon['mask'], real_mask, FLAGS.mask_loss_type, 'mask', FLAGS.cyc_mask_loss)  # (N,)
        if bool(FLAGS.cyc_loss):
            cyc_loss += self.summerize_cyc_loss(recon['image'], real_img, 'l1', 'rgb', FLAGS.cyc_loss)  # (N,)
            if FLAGS.cyc_perc_loss:
                cyc_loss += self.summerize_cyc_loss(recon['image'], real_img, 'perc', 'perc', FLAGS.cyc_perc_loss)  # (N,)

        if bool(FLAGS.d_loss_holo):
            g_loss += self.summerize_g_loss(holo, 'holo', FLAGS.d_loss_holo * balance)

        if bool(FLAGS.d_loss_hallc):
            g_loss += self.summerize_g_loss(samp, 'samp', FLAGS.d_loss_hallc * balance)

        if bool(FLAGS.know_mean):
            cyc_loss += self.summerize_cyc_loss(recon['3d'], real_datapoint['mean_shape'], 'mse', 'vox', FLAGS.vox_loss)

        r_loss = self.summerize_reg_loss(recon_z, param_u, FLAGS.reg_loss)
        prior_loss = self.summerize_prior_loss(recon['3d_view'], 'prior')

        # VAE-GAN: Decoder opt for GAN and reconstruct loss
        g_loss = g_loss + cyc_loss + prior_loss
        e_loss = r_loss + cyc_loss + prior_loss
        if FLAGS.content_loss:
            key = 'image' if FLAGS.cyc_loss else 'mask'
            hat_u = geom_utils.azel2u6d(param_v) if FLAGS.so3_enc == 6 else param_u
            e_loss += self.summerize_content_loss(
                 self.G.encoder(recon[key].detach()), recon_z, hat_u, 'recon', FLAGS.content_loss)
            if bool(FLAGS.d_loss_holo):
                e_loss += self.summerize_content_loss(
                    self.G.encoder(holo[key].detach()), recon_z, holo_u, 'holo', FLAGS.content_loss)
            if bool(FLAGS.d_loss_hallc):
                e_loss += self.summerize_content_loss(
                    self.G.encoder(samp[key].detach()), sample_z, param_u2, 'hallc', FLAGS.content_loss)

        # Decoder step
        g_loss.backward(retain_graph=True)
        self.gen_opt.step()
        # Encoder step
        self.G.encoder.zero_grad()
        e_loss.backward()
        self.e_opt.step()

        # register some output
        self.view = {'recon': param_u, 'holo': holo_u}
        self.enc_z = recon_z
        self.loss = {'D': d_loss, 'G': g_loss, 'E': e_loss}
        self.real = real_datapoint

    def print_every(self):
        print(self.lossMng.to_str())
        print('D %.2f, G: %.2f, E: %.2f\n' % (self.loss['D'], self.loss['G'], self.loss['E']))
        counter = self.counter
        logger = self.logger
        # log loss
        logger.add_loss(counter, self.loss, pref='Total/')
        logger.add_loss(counter, self.lossMng.losses)

        logger.add_hist_by_dim(counter, self.enc_z, 'enc_z')

    def vis_every(self):
        logger = self.logger
        counter = self.counter

        logger.add_images(counter, self.recon['image'], 'recon')
        logger.add_images(counter, self.real['image'], 'real')
        self.save_vis(counter, self.real, 'real', self.log_dir)

        if bool(FLAGS.d_loss_recon):
            self.save_vis(counter, self.recon, 'rec%d' % 0, self.log_dir)
        if bool(FLAGS.d_loss_holo):
            logger.add_images(counter, self.holo['image'], 'holo')
            self.save_vis(counter, self.holo, 'sampv', self.log_dir)

            device = self.recon['image'].device
            view, param = geom_utils.sample_view(FLAGS.sample_view, 1000, device, cfg=self.cfg)
            self.evaluator.scatter_view_param(param, self.log_dir, prefix='%d_holo0' % self.counter)
            (_, param, _), _, _ = self.G.decoder.view_sampler(1000, device, param,)
            self.evaluator.scatter_view_param(param, self.log_dir, prefix='%d_holo1' % self.counter)

            _, _, nn_param = op_3d.knn_points(self.recon['v_mu'].unsqueeze(0), param.unsqueeze(0), return_nn=True)
            self.evaluator.scatter_view_param(nn_param, self.log_dir, prefix='%d_holo2' % self.counter)
        if bool(FLAGS.d_loss_hallc):
            logger.add_images(counter, self.samp['image'], 'hall')
            self.save_vis(counter, self.samp, 'sampz', self.log_dir)

        self.evaluator.vis_view(self.G, self.fix_val_data, num=8, view_mod='az',
                                prefix='%d' % self.counter, save_dir=self.log_dir)
        self.evaluator.scatter_view(self.G, self.val_dataloader, self.log_dir, prefix='%d' % self.counter)
