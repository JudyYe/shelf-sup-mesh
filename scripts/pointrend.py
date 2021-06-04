    # --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np
import scipy.io as sio
import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch

from projects.PointRend.point_rend import ColorAugSSDTransform, add_pointrend_config

import pycocotools.coco as coco

from detectron2.engine import DefaultPredictor

data_dir = 'datasets/pascal3d/'
anno_file = 'datasets/coco/annotations/instances_val2017.json'
coco_data = coco.COCO(anno_file)
coco_classes = coco_data.loadCats(coco_data.getCatIds())
coco_classes = [cat['name'] for cat in coco_classes]

# import datasets.dummy_datasets as dummy_datasets

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def coco_to_pascal_name(category):
    if category == 'airplane':
        return 'aeroplane'
    if category == 'dining table':
        return 'diningtable'
    if category == 'motorcycle':
        return 'motorbike'
    if category == 'couch':
        return 'sofa'
    if category == 'tv':
        return 'tvmonitor'
    else:
        return category


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument("--config-file",
                        default="projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml",
                        metavar="FILE", help="path to config file")
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='output_pascal',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--im_or_folder', help='image or folder of images', default='output'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = "models/pointRend_model_final_3c3198.pkl"
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    logger = logging.getLogger(__name__)
    cfg = setup(args)

    predictor = DefaultPredictor(cfg)

    # if os.path.isdir(args.im_or_folder):
    #     im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    # else:
    #     im_list = [args.im_or_folder]

    im_list = get_cat_list(['chair'])
    print(len(im_list))
    loop = tqdm.tqdm(im_list)

    for i, im_name in enumerate(loop):
        loop.set_description('%d/%d' % (i, len(im_list)))
        # if i > 10:
        #    continue
        out_name = os.path.join(
            args.output_dir, '{}'.format(im_name[:-len(args.image_ext)] + 'mat')
        )
        if not os.path.exists(os.path.dirname(out_name)):
            os.makedirs(os.path.dirname(out_name))

        im = cv2.imread(os.path.join(data_dir, 'Images', im_name))
        t = time.time()
        # https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        output = predictor(im)
        boxes, segms, classes = convert_from_pred_output(output)
        if boxes is None:
            continue

        valid_inds = np.greater(boxes[:, 4], 0.5)
        boxes = boxes[valid_inds]
        segms = segms[valid_inds]
        classes = np.array(classes)[valid_inds]
        class_names = np.asarray(([coco_to_pascal_name(coco_classes[c]) for c in classes]), dtype='object')
        sio.savemat(out_name, {'masks': segms, 'boxes': boxes, 'classes': class_names});


def convert_from_pred_output(output):
    """
    :param output:
    :return: bbox (N, 5). segms: (N, H, W), classes: (N, nC)
    """
    output = output['instances']
    score = output.scores.cpu().detach().numpy()
    bbox = output.pred_boxes.tensor.cpu().detach().numpy()
    classes = output.pred_classes.cpu().detach().numpy()
    masks = output.pred_masks.cpu().detach().numpy()

    bbox = np.hstack([bbox, score[..., None]])

    return bbox, masks, classes


def get_cat_list(cls_list):
    set_list = ['val', 'train']
    set_dir = 'datasets/pascal3d/Image_sets'
    index_list = []
    for cls in cls_list:
        cls = cls + '_imagenet'
        for s in set_list:
            fname = os.path.join(set_dir, cls + '_%s.txt' % s)
            print(fname)
            with open(fname) as fp:
                for line in fp:
                    index_list.append(os.path.join(cls, '%s.%s' % (line.strip(), args.image_ext)))
    return index_list

if __name__ == '__main__':
    args = parse_args()
    # utils.logging.setup_logging(__name__)
    main(args)