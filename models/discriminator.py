# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import absl.flags as flags
FLAGS = flags.FLAGS

from nnutils.layers import dis_block, linear_block, dis_block_mean_std, conv2d

class PoolDisc(nn.Module):
    def __init__(self, cont_dim=128, df_dim=128, img_hw=64, norm='batch', add_D_noise=True, num_layers=4, c_dim=3):
        super().__init__()
        self.img_hw = img_hw
        self.add_noise = add_D_noise
        self.num_layer = num_layers

        self.base = nn.Sequential(*dis_block(1, [c_dim, df_dim], norm='none'))
        self.mask_base = nn.Sequential(*dis_block(1, [1, df_dim], norm='none'))

        df_dim_list = [df_dim * 2**(i) for i in range(num_layers + 1)]
        self.hidden = nn.ModuleList(dis_block_mean_std(num_layers, df_dim_list, FLAGS.disc_conv))

        out_dim = df_dim * 2**(num_layers)
        feat_hw = img_hw // (2**(num_layers + 1))
        assert feat_hw > 1
        flat_dim = feat_hw * feat_hw * out_dim
        self.clf_head = linear_block([out_dim, 1])
        self.cont_head = nn.Sequential(linear_block([out_dim, 128, cont_dim]), nn.Tanh())

        # each hidden head
        self.hidden_head = nn.ModuleList()
        for n in range(num_layers):
            out_dim = df_dim_list[n + 1]  # mean, std independently
            self.hidden_head.append(linear_block([out_dim, 1]))

    def forward(self, *input, **kwargs):
        """
        :param input (N, C, H, W):
        :param kwargs:
        :return: [logit(N, 1) ]
        """
        image, = input
        if image.size(-1) != self.img_hw:
            image = F.interpolate(image, self.img_hw)

        if self.add_noise:
            stddev = 0.02
            image = image + torch.randn_like(image) * stddev
        if image.size(1) == 1:
            hidden = self.mask_base(image)
        elif image.size(1) == 3:
            hidden = self.base(image)
        hidden_list, h_logits_list = [], []
        for i, net in enumerate(self.hidden):
            hidden, mean, std = net(hidden)  # hidden
            N, C, _, _ = hidden.size()
            h_style = torch.cat([mean, std], dim=0).view(2 * N, C)
            h_logits = self.hidden_head[i](h_style)
            h_logits_list.append(h_logits)
            hidden_list.append(hidden)

        N, C, h, w = hidden.size()
        hidden = F.avg_pool2d(hidden, h)
        hidden = hidden.view(N, -1)

        logits = self.clf_head(hidden)
        h_logits_list.insert(0, logits)

        return h_logits_list # , hidden_list


def buildD(mod, FLAGS):
    gf_d_z_dict = {
        'pool': PoolDisc,
    }
    if mod in gf_d_z_dict:
        model = gf_d_z_dict[mod](norm=FLAGS.d_norm, cont_dim=FLAGS.z_dim, img_hw=FLAGS.low_reso)
    else:
        raise NotImplementedError
    return model
