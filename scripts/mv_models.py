# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import json
import os
import os.path as osp
import numpy as np
import pandas
from absl import app, flags
from tqdm import tqdm
from datasets.dataset import build_dataloader

from config import config_flag
from nnutils import image_utils

FLAGS = flags.FLAGS
FLAGS.batch_size = 8
FLAGS.exp = 'occBal,occ05,rib100Trunc,occBlob'

def move_models():
    with open('data/openimages/Models/cat2model.json') as fp:
        cat2model = json.load(fp)
    model_dir = '/glusterfs/yufeiy2/transfer/HoloGAN'
    dst_dir = 'weights/'
    for cat, model_path in tqdm(cat2model.items()):
        if not osp.exists(osp.join(model_dir, model_path)):
            print(cat, model_path)
            continue
        os.makedirs(osp.join(dst_dir, cat), exist_ok=True)

        src_model = osp.join(model_dir, model_path)
        dst_file = osp.join(dst_dir, cat, 'model.pth')
        if os.path.exists(dst_file):
            continue

        cmd = 'cp %s %s' % (src_model, dst_file)
        os.system(cmd)

        src_model = osp.join(model_dir, osp.dirname(model_path), 'flags.txt')
        dst_file = osp.join(dst_dir, cat, 'flags.txt')
        cmd = 'cp %s %s' % (src_model, dst_file)
        os.system(cmd)
        print(cat)

def move_images(_):
    df = pandas.read_csv('examples/oiCherry.csv')
    for index, row in tqdm(df.iterrows(), total=len(df)):
        cls = row['category']
        cls_name = 'oi%s' % cls
        print(cls_name)

        exp_list = {}
        for exp in FLAGS.exp.split(','):
            if not pandas.isna(row[exp]) and len(row[exp].split('+')) >= 2:
                index_list =[e.split('_') for e in row[exp].split('+')]
                index_list = [(int(e[0]), int(e[1])) for e in index_list]
                for (b_idx, idv_idx) in index_list:
                    if b_idx not in exp_list:
                        exp_list[b_idx] = []
                    exp_list[b_idx].append(idv_idx)
                    print(exp, cls)

        dataloader = build_dataloader(FLAGS, 'test', False, name=cls_name)

        cnt = 0
        for b_idx, data in enumerate(dataloader):

            idv_list = exp_list.get(b_idx, [])
            if len(idv_list) == 0:
                continue

            for idv_idx in idv_list:
                save_image(data['mask_inp'][idv_idx], '%s_%d_m' % (cls, cnt))
                save_image(data['bg'][idv_idx] / 2 + 0.5, '%s_%d' % (cls, cnt))
                cnt += 1
            if b_idx >= 2:
                break

def move_images_quads(_):
    with open('examples/quads_demo_8.json') as fp:
        cls2example = json.load(fp)
    for index, cls in tqdm(enumerate(cls2example), total=len(cls2example)):
        cls_name = 'im%s' % cls
        print(cls_name)

        exp_list = {}
        index_list = [e.split('_') for e in cls2example[cls] ]

        index_list = [(int(e[0]), int(e[1])) for e in index_list]
        for (b_idx, idv_idx) in index_list:
            if b_idx not in exp_list:
                exp_list[b_idx] = []
            exp_list[b_idx].append(idv_idx)

        dataloader = build_dataloader(FLAGS, 'test', False, name=cls_name)

        cnt = 0
        for b_idx, data in enumerate(dataloader):

            idv_list = exp_list.get(b_idx, [])
            if len(idv_list) == 0:
                continue

            for idv_idx in idv_list:
                save_image(data['mask_inp'][idv_idx], '%s_%d_m' % (cls, cnt))
                save_image(data['bg'][idv_idx] / 2 + 0.5, '%s_%d' % (cls, cnt))
                cnt += 1
            if b_idx >= 20:
                break


def move_images_chairs(_):
    with open('examples/allchair_demo_8.txt') as fp:
        index_list = [line.strip() for line in fp]
        cls = cls_name = 'allChair'

        exp_list = {}

        index_list = [e.split('_') for e in index_list]
        index_list = [(int(e[0]), int(e[1])) for e in index_list]
        for (b_idx, idv_idx) in index_list:
            if b_idx not in exp_list:
                exp_list[b_idx] = []
            exp_list[b_idx].append(idv_idx)

        dataloader = build_dataloader(FLAGS, 'test', False, name=cls_name)

        cnt = 0
        for b_idx, data in enumerate(dataloader):

            idv_list = exp_list.get(b_idx, [])
            if len(idv_list) == 0:
                continue

            for idv_idx in idv_list:
                save_image(data['mask_inp'][idv_idx], '%s_%d_m' % (cls, cnt))
                save_image(data['bg'][idv_idx] / 2 + 0.5, '%s_%d' % (cls, cnt))
                cnt += 1
            if b_idx >= 20:
                break

def save_image(image, fname):
    image_file = osp.join(save_dir, fname)
    image_utils.save_images(image[None], image_file)




if __name__ == '__main__':
    # move_models()
    save_dir = 'outputs/demo_images_quads'
    app.run(move_images_chairs)