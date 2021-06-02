# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PerceptualSimilarity.models import dist_model


class LossMng():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        self.criterionGAN = GANLoss()
        self.perc_loss = PerceptualLoss()
        self.mse_loss =  MseLoss()
        self.losses = {}
        self.total_loss = None
        return

    def kl_loss(self, mu, logvar, target_mu=0, target_logvar=0, reduction='mean'):
        """https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians"""
        target_logvar = torch.zeros_like(logvar) + target_logvar
        kl_loss = -0.5 * torch.mean(1 + logvar - target_logvar - (mu - target_mu).pow(2) - logvar.exp() + target_logvar.exp(), dim=-1)
        if reduction == 'mean':
            kl_loss = torch.mean(kl_loss)
        return kl_loss

    # ##### gan loss interface #####
    def calc_gan_loss(self, logit_list, target, wgt, pref):
        """
        :param logit_list: [(N, )] list of disc output, scaled to [0, 1]
        :param target: True / False
        :return:
        """
        self.total_loss = None
        loss = wgt * self.criterionGAN(logit_list[0], target)
        self.add_losses(loss, pref)

        if self.FLAGS.d_perc_loss:
            for i in range(1, len(logit_list)):
                avg = 1. / (len(logit_list) - 1) if self.FLAGS.loss_avg else 1
                loss = wgt * avg * self.criterionGAN(logit_list[i], target)
                self.add_losses(loss, pref + '_h')
        # self.total_loss should be the sum of all losses defined in this function.
        return self.total_loss

    # ##### cycle loss interface #####
    def calc_cyc_loss(self, fake_img, real_img, ltype='l1', wgt=None, pref=''):
        self.total_loss = None
        cyc_loss = wgt * self._recon_image_loss(ltype, fake_img, real_img, 'mean')
        self.add_losses(cyc_loss, pref)
        return self.total_loss

    def _recon_image_loss(self, ltype, output, target, reduction):
        loss_dict = {'l1': F.l1_loss,
                     'bce': self.safe_bce,
                     'iou': self.iou_loss,
                     'perc': self.perc_loss,
                     'mse': self.mse_loss, }
        return loss_dict[ltype](output, target.detach(), reduction=reduction)

    def safe_bce(self, a, b, reduction):
        """a: input, b: target"""
        if a.min() < 0:
            print(a.min())
            pass
        if a.max() > 1:
            print(a.max())
            pass
        a = torch.clamp(a, 0, 1)
        return F.binary_cross_entropy(a.view(-1), b.view(-1), reduction=reduction)

    def iou_loss(self, a, b, eps=1e-6, reduction='mean'):
        dims = tuple(range(a.ndimension())[1:])
        intersect = (a * b).sum(dims)
        union = (a + b - a * b).sum(dims) + eps
        loss = 1 - (intersect / union).sum() / intersect.nelement()
        return loss

    ###### loss utils #####
    def add_losses(self, loss, name):
        if self.total_loss is None:
            self.total_loss = loss
        else:
            self.total_loss = self.total_loss + loss

        if name in self.losses:
            self.losses[name] = self.losses[name] + loss
        else:
            self.losses[name] = loss

    def to_str(self):
        s = 'Loss:\n'
        for key in self.losses:
            s += '\t%s: %.4f\n' % (key, self.losses[key])
        return s

    def clear_loss(self):
        self.losses = {}
        self.total_loss = None


class MseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, reduction='mean'):
        loss = torch.mean((pred - target.detach()) ** 2)
        return loss


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            # target_tensor = target_tensor.to(prediction)
            loss = self.loss(prediction, target_tensor.cuda())
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class PerceptualLoss(object):
    """
    Calls Richard's Perceptual Loss.
    """
    def __init__(self, model='net', net='alex', use_gpu=True):
        print('Setting up Perceptual loss..')
        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_gpu)
        print('Done')

    def __call__(self, pred, target, normalize=False, reduction='mean'):
        """
        Pred and target are Variables.
        If normalize is on, scales images between [-1, 1] assuming the inputs are in range [0, 1].
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        dist = self.model.forward(pred, target.detach())
        if reduction == 'mean':
            # feature number = 5...
            dist = dist.mean()
        return dist