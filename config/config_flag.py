# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import absl.flags as flags

flags.DEFINE_string("exp", "default", "name of exp")
flags.DEFINE_string("cfg_file", "config/cub.json", "")

# encoder
flags.DEFINE_string("enc", "res", "kCam or res")
flags.DEFINE_string("e_norm", "instance", "batch / instance / none / myin")
flags.DEFINE_string("z_map", "regular", "regular / equal")
flags.DEFINE_string("sample_view", "cfg", "[cfg, side]")
flags.DEFINE_integer("use_scale", 0, "")
flags.DEFINE_integer("use_trans", 0, "")

# Decoder
flags.DEFINE_string("g_mod", "kCam", "G")

# volumetric decoder
flags.DEFINE_string("vol_render", "struct", "")
flags.DEFINE_string("infer", "style", "")
flags.DEFINE_string("vol_norm", "none", "instance / batch / none")
flags.DEFINE_integer('text_sym', 2, '')
flags.DEFINE_integer('text_vis', 0, '')


# disc
flags.DEFINE_string("d_mod", "pool", "D")
flags.DEFINE_string("disc_conv", "spec", " conv / spec")
flags.DEFINE_string("d_norm", "myin", "batch / instance / none / myin")

# dimension
flags.DEFINE_integer("gf_dim", 64, "")
flags.DEFINE_integer("c_dim", 3, "")
flags.DEFINE_integer('z_dim', 128, '')
flags.DEFINE_integer('num_z', 1, '')


# reso
flags.DEFINE_integer('reso_3d', 16, '')
flags.DEFINE_integer('reso_feat', 8, '')
flags.DEFINE_integer('reso_local', 32, '')
flags.DEFINE_integer('reso_vox', 32, '')
flags.DEFINE_integer("high_reso", 224, "resolution of input")
flags.DEFINE_integer("low_reso", 64, "resolution of input")


# train supervision
flags.DEFINE_integer("know_fg", 3, "")
flags.DEFINE_integer("know_pose", 0, "")
flags.DEFINE_integer("know_vox", 0, "")
flags.DEFINE_integer("know_mean", 0, "")

# train
flags.DEFINE_integer("train", 1, "True for training, False for testing [False]")
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_string('scheduler', 'linear', '')
flags.DEFINE_float('lr', 1e-4, '')
flags.DEFINE_integer('lr_decay_iters', 50, '')
flags.DEFINE_float('max_iters', 100000, '')
flags.DEFINE_float('min_iters', 10000, '')
flags.DEFINE_integer('print_every', 100, '')
flags.DEFINE_integer('vis_every', 5000, '')
flags.DEFINE_integer('save_every', 20000, '')

# loss
flags.DEFINE_float('d_loss_mask', 0 , '')
flags.DEFINE_float('d_loss_rgb', 1 , '')
flags.DEFINE_float('d_loss_real', 1. , '')
flags.DEFINE_float('d_loss_recon', 1. , '')
flags.DEFINE_float('d_loss_holo', 1. , '')
flags.DEFINE_float('d_perc_loss', 1 , '')
flags.DEFINE_float('cyc_loss', 10. , '')
flags.DEFINE_float('cyc_perc_loss', 1 , '')
flags.DEFINE_float('cyc_mask_loss', 50. , '')
flags.DEFINE_string('cum_occ', 'prod', '')
flags.DEFINE_float('lap_loss', 100. , '')
flags.DEFINE_float('lap_norm_loss', .01, '')
flags.DEFINE_float('reg_loss', .1 , '')
flags.DEFINE_float('delta_loss', 1000 , '')
flags.DEFINE_float('vox_loss', 10, '')
flags.DEFINE_float('kl_loss', 1 , '')
flags.DEFINE_float('content_loss', 100. , '')
flags.DEFINE_float('content_z', 1. , '')
flags.DEFINE_integer('content_enc', 1, 'encode azel as sin cos')
flags.DEFINE_integer('so3_enc', 2, '2 / 6')

flags.DEFINE_integer('loss_avg', 1 , '')
flags.DEFINE_string('mask_loss_type', 'iou', 'l1, bce, iou')
flags.DEFINE_string('lap_method', 'uniform', 'uniform / cot')

# prior
flags.DEFINE_float('prior_thin', 0, '')
flags.DEFINE_float('prior_blob', 0, '')
flags.DEFINE_float('prior_same', 0, '')

# data
flags.DEFINE_string("dataset", "cub", "The name of dataset [celebA, lsun, chairs, shoes, cars, cats]")
flags.DEFINE_integer("n_view", 20, "")
flags.DEFINE_float('PAD_FRAC', 0.1, 'bbox is increased by this fraction of max_dim')
flags.DEFINE_float('JITTER_TRANS', 0, 'bbox is increased by this fraction of max_dim')
flags.DEFINE_float('JITTER_COLOR', 0.1, 'COLOR')
flags.DEFINE_float('JITTER_SCALE', 0, 'bbox is increased by this fraction of max_dim')
flags.DEFINE_float('filter_trunc', 0.5, '')
flags.DEFINE_string('filter_model', 'occBal', '')
flags.DEFINE_integer('rdn_lr', 1, '')
flags.DEFINE_integer('seed', 123, '')

flags.DEFINE_integer('apply_sym', 1 , '')
flags.DEFINE_integer('delta_sym', 0 , '')


# test
flags.DEFINE_string('checkpoint', None, '')
flags.DEFINE_string('test_mod', 'default', '')
flags.DEFINE_integer('test_size', 8, '')

# train
flags.DEFINE_string("gpu", "3", "gpu")


# ################ deprecated ################
flags.DEFINE_float('d_loss_hallc', 0. , 'deprecated')
flags.DEFINE_float('flow_loss', 0. , 'deprecated')
flags.DEFINE_float('ft', 0. , 'deprecated')
flags.DEFINE_float('ft_disc', 0. , 'deprecated')
flags.DEFINE_float('share_disc', 0. , 'deprecated')
flags.DEFINE_float('tight_crop', 0. , 'deprecated')
flags.DEFINE_float('noise', 0. , 'deprecated')
flags.DEFINE_string('noise_type', 'n' , 'deprecated')
flags.DEFINE_float('cyc_normal_loss', 0, '')
flags.DEFINE_integer('num_stages', 5, '')
flags.DEFINE_integer('every', 0, '')
flags.DEFINE_integer('detach_tg', 0, '')
flags.DEFINE_integer('sample_output', 1, '')
flags.DEFINE_string('gcn_text', 'sum', '')
flags.DEFINE_string('render_text', 'sum', '')
flags.DEFINE_integer("view_feat", "1", "")
# mesh decoder
flags.DEFINE_string("mesh_render", "feat", "")
flags.DEFINE_string("mesh_norm", "instance", "batch / instance / none / myin")
flags.DEFINE_float('mesh_th', 0.05, '')
flags.DEFINE_integer('render_faces', 10, '')
flags.DEFINE_integer('detach_vox', 1, 'Flow, GwoL')
flags.DEFINE_integer('detach_view', 1, 'Flow, GwoL')
flags.DEFINE_integer('detach_img_feat', 1, '')
flags.DEFINE_integer('detach_text_feat', 1, '')
flags.DEFINE_float('render_sigma', 1e-5, 'follow default setting in SoftRas')
flags.DEFINE_float('render_blur', 0, 'follow default setting in SoftRas')
flags.DEFINE_string("sample_z", "prior", "[mean, resammple]")
flags.DEFINE_integer('align_corners', 1, '')

flags.DEFINE_float('d_loss_normal', 0 , '')
flags.DEFINE_string('detach_enc', 'Flow', 'Flow, GwoL')
flags.DEFINE_float('contour_loss', 1000. , '')
flags.DEFINE_integer('balance_loss', 1, '')
flags.DEFINE_string('normal_loss_type', 'expLoss', '')
flags.DEFINE_float('vvp_loss', 0. , '')
flags.DEFINE_float('biharm_loss', 00., '' )
flags.DEFINE_integer("learn_view_prior", 0, "")
# ################ ~deprecated ################
