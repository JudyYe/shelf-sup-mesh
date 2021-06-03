# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import json
import os
from typing import Dict, Union

import imageio
import torch.optim as optim
import tqdm
from torch.optim import lr_scheduler

import nnutils.utils as utils
from models.evaluator import Evaluator
from nnutils import image_utils
from nnutils.layers import *
from nnutils.logger import Logger
from datasets.dataset import build_dataloader
from .discriminator import buildD
from .generator import ReconstructModel
from .loss import LossMng

FLAGS = flags.FLAGS


def TrainerFactory(mod):
    from .vox_trainer import VoxTrainer
    name_dict = {
        'kCam': VoxTrainer,
    }
    return name_dict[mod]()


class BaseTrainer(object):
    def __init__(self):
        self.dataloader = None
        self.val_dataloader = None

        self.lossMng = None
        self.logger = None
        self.log_dir = ''

        with open(FLAGS.cfg_file) as fp:
            self.cfg = json.load(fp)
        self.cfg['azi_low'] = self.cfg['azi_low'] / 180 * np.pi
        self.cfg['azi_high'] = self.cfg['azi_high'] / 180 * np.pi
        self.cfg['ele_low'] = self.cfg['ele_low'] / 180 * np.pi
        self.cfg['ele_high'] = self.cfg['ele_high'] / 180 * np.pi

        self.model_name = utils.get_model_name(FLAGS)
        self.build_dataset()
        return

    def build_dataset(self):
        self.val_dataloader = build_dataloader(FLAGS, 'test', False)
        if FLAGS.train == 1:
            self.dataloader = build_dataloader(FLAGS, 'train')
        for data in self.val_dataloader:
            self.fix_val_data = utils.to_cuda(data)
            break

    def build_val(self):
        # save_dir = "outputs/"
        save_dir = "/glusterfs/yufeiy2/transfer/HoloGAN"
        checkpoint = FLAGS.checkpoint

        print('Init...', checkpoint)
        check_path = os.path.join(save_dir, checkpoint)
        pretrained_dict = torch.load(check_path)
        self.cfg = pretrained_dict['cfg']

        log_dir = os.path.dirname(checkpoint)
        self.log_dir = os.path.join(save_dir, log_dir)
        print('save image to', self.log_dir)

        self.G = ReconstructModel()
        utils.load_my_state_dict(self.G, pretrained_dict['G'])

        self.G.eval()
        self.G.cuda()

    def build_train(self, checkpoint=None, clear_eval=True):
        OUTPUTDIR = "outputs/"
        self.G = ReconstructModel()

        if checkpoint:
            # set up model
            print('Init...', checkpoint)
            check_path = os.path.join(OUTPUTDIR, checkpoint)
            pretrained_dict = torch.load(check_path)
            utils.load_my_state_dict(self.G, pretrained_dict['G'])

        self.G.train()
        self.G.cuda()

        self.D = buildD(FLAGS.d_mod, FLAGS)
        self.D_mask = buildD(FLAGS.d_mod, FLAGS)
        self.D_norm = buildD(FLAGS.d_mod, FLAGS)
        self.D_list = nn.ModuleDict()
        if FLAGS.d_loss_rgb:
            self.D_list.add_module('image', self.D)
        if FLAGS.d_loss_mask:
            self.D_list.add_module('mask', self.D_mask)
        if FLAGS.d_loss_normal:
            self.D_list.add_module('normal', self.D_norm)

        self.D_list.train()
        self.D_list.cuda()

        a_ep = max(0, FLAGS.min_iters // len(self.dataloader))
        b_ep = (FLAGS.max_iters - FLAGS.min_iters) // len(self.dataloader)
        # set up optmizer
        self.e_opt = optim.Adam(self.G.encoder.parameters(), lr=FLAGS.lr)
        self.scheduler_e = get_scheduler(self.e_opt, FLAGS.scheduler, (a_ep, b_ep))

        self.gen_opt = optim.Adam(self.G.decoder.parameters(), lr=FLAGS.lr)
        self.scheduler_gen = get_scheduler(self.gen_opt, FLAGS.scheduler, (a_ep, b_ep))

        self.d_opt = optim.Adam(self.D_list.parameters(), lr=FLAGS.lr)
        self.scheduler_D = get_scheduler(self.d_opt, FLAGS.scheduler, (a_ep, b_ep))

        # set up loss manager
        self.lossMng = LossMng(FLAGS)

        # set up logger
        self.logger = Logger(self.model_name, OUTPUTDIR)
        self.log_dir = self.logger.save_dir
        self.save_dir = os.path.join(OUTPUTDIR, self.model_name)
        self.evaluator = Evaluator(self.cfg)

        flag_str = FLAGS.flags_into_string()
        with open(os.path.join(self.save_dir, 'flags.txt'), 'w') as fp:
            fp.write(flag_str)

        print(self.G)

    def train(self):
        save_point = [1, 5000, 10000, 30000, 50000]
        vis_point = [10, 100, 200, 500, 1000, 2000, 3000, 4000]
        # Save model
        self.counter = 0
        for epoch in range(10000):
            bar = tqdm.tqdm(self.dataloader, )

            for i, (real_datapoint) in enumerate(bar):
                self.lossMng.clear_loss()
                real_datapoint = utils.to_cuda(real_datapoint)

                self.train_step(real_datapoint)

                bar.set_description("%s Epoch %d [%d, %d]"
                                    % (self.model_name, epoch, self.counter, len(self.dataloader)))

                if np.mod(self.counter, FLAGS.print_every) == 0:
                    # Output training stats
                    self.print_every()

                # Check how the generator is doing by saving G's output on fixed_noise
                if np.mod(self.counter, FLAGS.vis_every) == 0 or self.counter in vis_point:
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        self.G.eval()
                        self.vis_every()
                        torch.cuda.empty_cache()
                        self.G.train()
                        torch.cuda.empty_cache()

                self.counter += 1
                if np.mod(self.counter, FLAGS.save_every) == 0 or self.counter in save_point:
                    self.save(self.counter)
                if self.counter > FLAGS.max_iters:
                    break
            if self.counter > FLAGS.max_iters:
                break
            # if len(self.dataloader) > 200:
            self.scheduler_D.step()
            self.scheduler_gen.step()
            self.scheduler_e.step()

    def vis_every(self):
        return

    def train_step(self, real_datapoint):
        raise NotImplementedError

    def print_every(self):
        return

    def save(self, epoch):
        # Save model
        state = {
            'G': self.G.state_dict(),
            'D': self.D_list.state_dict(),
            'start_epoch': epoch,
            'iters': self.counter,
            'cfg': self.cfg,
        }
        torch.save(state, os.path.join(self.save_dir, 'latest.pth'))
        torch.save(state, os.path.join(self.save_dir, '%d.pth' % epoch))

    # ######## Loss ########
    def summerize_content_loss(self, enc_output, z_target, view_target, name, wgt):
        (z_out, _), (_, _, para_u,) = enc_output
        z_loss = self.lossMng.calc_cyc_loss(z_out, z_target, 'mse', FLAGS.content_z * wgt, pref='E:z:' + name)

        if FLAGS.content_enc and FLAGS.so3_enc == 2:
            para_u = torch.cat([torch.sin(para_u), torch.cos(para_u)], dim=-1)
            view_target = torch.cat([torch.sin(view_target), torch.cos(view_target)], dim=-1)

        v_loss = self.lossMng.calc_cyc_loss(para_u, view_target, 'mse', wgt, pref='E:v:' + name)
        return z_loss + v_loss

    def summerize_cyc_loss(self, fake: torch.Tensor, real, ltype, name, wgt):
        """
        :param wgt: (might be scaler, or (N, )
        :return:
        """
        if fake.ndim == 4 and fake.size(-1) != real.size(-1):
            real = F.interpolate(real, fake.size(-1))
        return self.lossMng.calc_cyc_loss(fake, real, ltype=ltype, wgt=wgt, pref='cyc:' + name)

    def summerize_prior_loss(self, vox: torch.Tensor, name):
        """
        :param vox: (N, 1, D, H, W) in view frame
        :param name:
        :return:
        """
        prior_loss = 0
        if FLAGS.prior_thin > 0:
            loss = FLAGS.prior_thin * vox.abs().mean()
            prior_loss = prior_loss + loss
            self.lossMng.add_losses(loss, '%s:thin' % name)

        if FLAGS.prior_blob > 0:
            loss = FLAGS.prior_blob * (1 - vox).abs().mean()
            prior_loss = prior_loss + loss
            self.lossMng.add_losses(loss, '%s:blob' % name)

        if FLAGS.prior_same > 0:
            loss = FLAGS.prior_same * (vox[:, :, 1:] - vox[:, :, :-1]).abs().mean()
            prior_loss = prior_loss + loss
            self.lossMng.add_losses(loss, '%s:smooth' % name)
        return prior_loss

    def summerize_reg_loss(self, z_mu, v_mu, wgt):
        zeros = torch.zeros_like(z_mu)

        kl_z = wgt * self.lossMng.kl_loss(z_mu, zeros)
        self.lossMng.add_losses(kl_z, 'kl:z')

        zeros = torch.zeros_like(v_mu)
        kl_v1 = self.lossMng.kl_loss(v_mu, zeros, np.pi / 2, np.pi / 4, None)
        kl_v2 = self.lossMng.kl_loss(v_mu, zeros, -np.pi / 2, np.pi / 4, None)
        kl_v = wgt * torch.mean(torch.min(kl_v1, kl_v2), dim=-1)
        self.lossMng.add_losses(kl_v, 'kl_v')

        return kl_z + kl_v

    def summerize_g_loss(self, fake, name, wgt=1.):
        return self._summerize_gan_loss(fake, True, 'G:%s' % name, wgt, detach=False)

    def summerize_d_real_loss(self, real, name, wgt=1.):
        return self._summerize_gan_loss(real, True, 'D:%s' % name, wgt, detach=True)

    def summerize_d_fake_loss(self, fake, name, wgt=1.):
        return self._summerize_gan_loss(fake, False, 'D:%s' % name, wgt, detach=True)

    def _summerize_gan_loss(self, pred, target, name, wgt=1., detach=True):
        d_loss = 0
        if bool(FLAGS.d_loss_rgb):
            image = pred['image']
            if detach:
                image = image.detach()
            rgb_loss = self.lossMng.calc_gan_loss(self.D(image), target, wgt, pref=name)
            d_loss = d_loss + rgb_loss

        if bool(FLAGS.d_loss_mask):
            mask = pred['mask']
            if detach:
                mask = mask.detach()
            mask_loss = self.lossMng.calc_gan_loss(self.D_mask(mask), target, wgt, pref='%s_mask' % name)
            d_loss = d_loss + mask_loss

        if bool(FLAGS.d_loss_normal):
            image = pred['normal']
            if detach:
                image = image.detach()
            mask_loss = self.lossMng.calc_gan_loss(self.D_norm(image), target, wgt, pref='%s_norm' % name)
            d_loss = d_loss + mask_loss
        return d_loss

    # ##### visualization
    def save_vis(self, counter, recon: Union[Dict, torch.Tensor], name, save_dir):
        if torch.is_tensor(recon):
            merge_image = image_utils.merge_to_numpy(recon)
            imageio.imsave(save_dir + '/%d_%s.png' % (counter, name), merge_image)
        else:
            if recon.get('image', None) is not None:
                merge_image = image_utils.merge_to_numpy(recon['image'])
                imageio.imsave(save_dir + '/%d_%s.png' % (counter, name), merge_image)
            if recon.get('mask', None) is not None:
                merge_image = image_utils.merge_to_numpy(recon['mask'])
                imageio.imsave(save_dir + '/%d_fm_%s.png' % (counter, name), merge_image)
            if recon.get('normal', None) is not None:
                recon['normal'] / recon ['normal'].norm(dim=1, keepdim=True).clamp(min=1e-8)
                merge_image = image_utils.merge_to_numpy(recon['normal'])
                imageio.imsave(save_dir + '/%d_n_%s.png' % (counter, name), merge_image)


def get_scheduler(optimizer, lr_policy, eps):
    """Return a learning rate scheduler
    Copy from Jun-Yan's Pix2Pix-CycleGAN
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    min_ep, decay_ep = eps
    if lr_policy == 'linear':
        print('start decay at %d epo for %d epo' % (min_ep, decay_ep))
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - min_ep) / float(decay_ep)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=FLAGS.lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=min_ep, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler

