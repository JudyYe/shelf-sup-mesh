# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

from torchvision.models import resnet18

from nnutils import geom_utils
from nnutils.layers import *

FLAGS = flags.FLAGS


class Scale(nn.Module):
    def __init__(self, inp_dim, bias=1):
        super().__init__()
        if FLAGS.use_scale > 0:
            self.body = linear(inp_dim, 1)
            nn.init.constant_(self.body.bias, bias)
        self.bias = bias

    def forward(self, x):
        if FLAGS.use_scale > 0:
            return self.body(x)
        else:
            return torch.zeros([x.size(0), 1]).to(x) + self.bias


class Trans(nn.Module):
    def __init__(self, inp_dim, bias=(0, 0, 2)):
        super().__init__()
        self.body = linear(inp_dim, 3)
        self.register_buffer('bias', torch.FloatTensor([bias]))

    def forward(self, x):
        if FLAGS.use_trans > 0:
            return self.body(x)
        else:
            return torch.zeros([x.size(0), 3]).to(x) + self.bias


class SO2(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.body = linear(inp_dim, 2)

    def forward(self, x):
        """v = u."""
        u = self.body(x)
        azel = u
        return azel, u


class SO2_6dof(nn.Module):
    def __init__(self, inp_dim):
        super().__init__()
        self.body = linear(inp_dim, 6)

    def forward(self, x):
        """return: (N, 2) (N, 6)"""
        u = self.body(x)
        azel = geom_utils.u6d2azel(u)
        return azel, u


class Camera(nn.Module):
    """scale, trans, rot"""
    def __init__(self, inp_dim, hid_dim=128):
        super().__init__()
        self.body = linear_block([inp_dim, hid_dim], last_relu=True)

        # todo
        self.scale_head = Scale(hid_dim)
        self.trans_head = Trans(hid_dim)

        if FLAGS.so3_enc == 2:
            self.rot_head = SO2(hid_dim)
        elif FLAGS.so3_enc == 6:
            self.rot_head = SO2_6dof(hid_dim)

    def forward(self, feat):
        """
        :param feat: (N, D)
        :return: [(N, 6)], (N, k)
        """
        feat = self.body(feat.flatten(1))

        scale = self.scale_head(feat)  # (N, 1)
        trans = self.trans_head(feat)  # (N, 3)

        azel, u = self.rot_head(feat)  # (N, 2)

        para = torch.cat([azel, scale, trans], dim=1)
        if FLAGS.so3_enc == 2:
            u = para
        return para, u


def EncoderFactory(name, z_dim, df_dim=64, norm='batch', num_layers=4, c_dim=3):
    infer_dict = {
        'kCam': Encoder,
        'res': ResnetEnc,
    }
    return infer_dict[name](z_dim, df_dim, norm, num_layers, c_dim)


def sample_z(mean_z):
    """
    VAE trick, reample z
    :param mu: (N, Dz)
    :param varlog:
    :return: (N, Dz)
    """
    if FLAGS.sample_z == 'mean':
        sample_z = torch.randn_like(mean_z) + mean_z
    elif FLAGS.sample_z == 'prior':
        sample_z = torch.randn_like(mean_z)
    return sample_z


class Encoder(nn.Module):
    def __init__(self, z_dim, df_dim=64, norm='batch', num_layers=4, c_dim=3):
        super().__init__()
        self.img_hw = img_hw = FLAGS.low_reso

        self.z_dim = z_dim
        self.z_method = FLAGS.sample_z

        base = dis_block(1, [c_dim, df_dim], norm='none')
        num1 = int(math.log2(FLAGS.low_reso // FLAGS.reso_feat))
        dims1_list = [df_dim * 2 ** i for i in range(num1)]
        block1 = dis_block(num1 - 1, dims1_list, norm=norm)
        self.hidden1 = nn.Sequential(*(base + block1))
        out_dim = dims1_list[-1]

        num2 = num_layers - num1
        self.hidden2 = None
        if num2 > 0:
            dims2_list = [dims1_list[-1] * 2 ** i for i in range(num2 + 1)]
            block2 = dis_block(num2, dims2_list, norm=norm)
            self.hidden2 = nn.Sequential(*block2)
            out_dim = dims2_list[-1]

        feat_hw = img_hw // (2**(num_layers))
        self.z_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv_block([out_dim, 128, z_dim], last_relu=False, k=1))

        self.view_head = Camera(out_dim * feat_hw ** 2)
        self.logvar_head = Camera(out_dim * feat_hw ** 2)

        self.feat_head = conv2d(dims1_list[-1], FLAGS.gf_dim, k=3, d=1, p=1)
        # self.feat_head.weight.grad

    def forward(self, *input, **kwargs):
        """
        :param input: (N, C, H, W)
        :param kwargs:
        :return: (N, Dz) (N, Dv)
        """
        x, = input
        if x.size(-1) != self.img_hw:
            x = F.interpolate(x, self.img_hw)
        hidden = self.hidden1(x)
        # if FLAGS.ft:
        #     feat = hidden.detach()
        # else:
        #     feat = self.feat_head(hidden)
        if self.hidden2 is not None:
            hidden = self.hidden2(hidden)
        N, D, H, W = hidden.size()

        z_mu = self.z_head(hidden).flatten(1)  # (N, D) scale: [-1, 1]

        view_param = self.view_head(hidden.view(N, D * H * W))  # (N, 4)
        view_logvar = self.logvar_head(hidden.view(N, D * H * W))
        # build params in

        if FLAGS.sample_view == 'vae':
            device = view_param.device
            views, _ = geom_utils.sample_view('vae', N, device, mean=view_param, logvar=view_logvar)
        else:
            views = geom_utils.azel2uni(view_param)

        return [z_mu, x], [views, view_param, view_logvar]


class ResnetEnc(nn.Module):
    def __init__(self, z_dim, df_dim=64, norm='batch', num_layers=4, c_dim=3):
        super().__init__()
        self.img_hw = FLAGS.high_reso
        self.c_dim = c_dim
        model = resnet18(pretrained=False, norm_layer=get_norm_layer(norm))
        self.nets = nn.ModuleList(list(model.children())[:-2])
        out_dim = 512
        feat_hw = 4
        self.view_head = nn.Sequential(nn.AdaptiveAvgPool2d(feat_hw), Camera(out_dim * feat_hw ** 2))
        self.z_head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                    conv_block([out_dim, 128, z_dim], last_relu=False, k=1))

    def forward(self, *input, **kwargs):
        x, = input
        if x.size(-1) != self.img_hw:
            x = F.interpolate(x, self.img_hw)
        if x.size(1) != self.c_dim:
            x = x.expand(x.size(0), self.c_dim, self.img_hw, self.img_hw)

        hidden_list = [x]
        for i, net in enumerate(self.nets):
            x = net(x)
            if i >= 4:
                hidden_list.append(x)

        z = self.z_head(x)
        z = z.flatten(1)

        v, u = self.view_head(x)
        # build params in

        view = geom_utils.azel2uni(v)

        return [z, hidden_list], [view, v, u]


def sample_z(mean_z):
    """
    VAE trick, reample z
    :param mu: (N, Dz)
    :param varlog:
    :return: (N, Dz)
    """
    if FLAGS.sample_z == 'mean':
        sample_z = torch.randn_like(mean_z) + mean_z
    elif FLAGS.sample_z == 'prior':
        sample_z = torch.randn_like(mean_z)
    return sample_z

