# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import numpy as np
import torch
from PIL import Image

from models.generator import ReconstructModel
from models.evaluator import Evaluator
from nnutils import mesh_utils
from nnutils.utils import load_my_state_dict

from absl import app
from config.config_flag import *

flags.DEFINE_string("demo_image", "examples/allChair_0.png", "path to input")
flags.DEFINE_string("demo_out", "outputs/demo_out", "dir of output")

FLAGS = flags.FLAGS

# optimization lambda
FLAGS.lap_loss = 100
FLAGS.lap_norm_loss = .5
FLAGS.cyc_mask_loss = 10
FLAGS.cyc_perc_loss = 0


def demo(_):
    # os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    # load pretrianed model
    model, cfg = load_model()
    # load demo data and preprocess.
    data = load_image()
    # for visualization utils
    evaluator = Evaluator(cfg)

    # step1: infer coarse shape and camera pose
    vox_world, camera_param = model.forward_image(data['image'])
    # init meshes
    vox_mesh = mesh_utils.cubify(vox_world).clone()
    # step2: optimize meshes
    mesh_inputs = {'mesh': vox_mesh, 'view': camera_param}
    with torch.enable_grad():
        mesh_outputs, record = evaluator.opt_mask(model, mesh_inputs, data, True, 300)

    # visualize mesh.
    vis_mesh(mesh_outputs, camera_param, evaluator.snapshot_mesh)


def load_image():
    image = np.asarray(Image.open(FLAGS.demo_image))
    image = image[:,:, 0: 3]  # in case of RGBA
    image = image[:, :, :3] / 127.5 - 1  # [-1, 1]

    mask_path = FLAGS.demo_image.replace('.', '_m.')
    if os.path.exists(mask_path):
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim < 3:
            mask = mask[..., np.newaxis]
        mask = (mask > 0).astype(np.float)
        fg = image * mask + (1 - mask)  # white bg
    else:
        fg = image
    fg = to_tensor(fg)
    image = to_tensor(image)
    mask = to_tensor(mask)
    return {'bg': image, 'image': fg, 'mask': mask}


def to_tensor(image):
    image = np.transpose(image, [2, 0, 1])
    image = image[np.newaxis]
    return torch.FloatTensor(image).cuda()


def load_model():
    checkpoint = FLAGS.checkpoint
    print('Init...', checkpoint)
    pretrained_dict = torch.load(checkpoint)
    cfg = pretrained_dict['cfg']

    model = ReconstructModel()
    load_my_state_dict(model, pretrained_dict['G'])

    model.eval()
    model.cuda()
    return model, cfg


def vis_mesh(cano_mesh, pred_view, snapshot_func, f=375):
    """
    :param cano_mesh:
    :param pred_view:
    :param renderer:
    :param snapshot_func: snapshot given pose_list, and generate gif.
    :return:
    """
    # a novel view
    N = pred_view.size(0)
    can_view = torch.FloatTensor([np.pi / 4 + np.pi, np.pi / 6, 1, 0, 0, 2]).to(pred_view).unsqueeze(0).expand(N, 6)
    can_view = mesh_utils.param_to_7dof_batcch(can_view, f)
    pose_list = [can_view, pred_view]

    prefix = os.path.basename(FLAGS.demo_image).split('.')[0]
    snapshot_func(cano_mesh['mesh'][-1], pose_list, None,
                  FLAGS.demo_out, prefix, 'mesh', pred_view=pred_view)
    snapshot_func(cano_mesh['mesh'][-1], pose_list, cano_mesh['mesh'].textures,
                            FLAGS.demo_out, prefix, 'meshTexture', pred_view=pred_view)


if __name__ == '__main__':
    app.run(demo)
