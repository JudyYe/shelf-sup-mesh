# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import numpy as np
from typing import Dict
import torch
from PIL import Image
from torch.utils.data import Dataset
import nnutils.image_utils as image_utils


class BaseData(Dataset):
    def __init__(self, cfg, dataset, split, train):
        self.cfg = cfg
        self.dataset = dataset
        self.split = split
        self.train = train

        self.inp_size = cfg.high_reso
        self.out_size = cfg.low_reso

        # transformation
        self.pad_frac = cfg.PAD_FRAC
        self.jitter_trans = cfg.JITTER_TRANS
        self.jitter_scale = cfg.JITTER_SCALE
        self.jitter_color = cfg.JITTER_COLOR
        self.rdn_lr = cfg.rdn_lr
        self.know_mean = cfg.know_mean

        self.image_dir = ''
        self.anno = {'rel_path': [], 'mask': [], 'bbox': [], 'cad_index': [], }

        self.vox_list = []
        self.mesh_list = []
        self.map = None

    def __len__(self):
        return len(self.anno['rel_path'])

    def preload_anno(self):
        """
        fill self.anno: rel_path, mask, bbox, cad_index, correspondingly
        :return:
        """
        raise NotImplementedError

    def __getitem__(self, index):
        if self.map is not None:
            index = self.map[index]

        sample = self.get_datapoint(index)

        # get center of image & scale of image
        box_center, box_size = self.get_center_size(sample['bbox'], self.pad_frac)
        # jitter box
        box_center, box_size = self.jitter_box(box_center, box_size)
        # crop image
        sample = self.crop_sample(sample, box_center, box_size)

        # apply augmentation on image / masks
        sample = self.rand_flip_lr(sample)
        sample = self.color_jitter(sample)
        sample = self.resize_image(sample)

        sample = self.clear_sample(sample)

        return sample

    def get_datapoint(self, index):
        sample = {}
        sample['image'] = self.get_image(index)
        sample['mask'] = self.get_mask(index)
        sample['bbox'] = self.get_bbox(index)
        sample['index'] = self.anno['rel_path'][index]

        if self.know_mean > 0:
            sample['mean_shape'] = self.get_mean_shape(index)

        return sample

    def get_mean_shape(self, index):
        return None

    def get_image(self, index):
        img_path = self.anno['rel_path'][index]
        img_path = os.path.join(self.image_dir, img_path)
        image = Image.open(img_path)
        image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        if image.shape[2] == 1:
            image = np.tile(image, [1, 1, 3])
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]
        return image

    def get_bbox(self, index):
        bbox = self.anno['bbox'][index]
        return bbox

    def get_mask(self, index):
        """
        :param index:
        :return: (H, W)
        """
        mask = self.anno['mask'][index]
        if isinstance(mask, str):
            mask = Image.open(mask)

        mask = (np.array(mask) > 0).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask

    def get_center_size(self, bbox, pad):
        """
        :return: center pixel in int, side length
        """
        scale_factor = 1 + 2 * pad
        x1, y1, x2, y2 = bbox
        center = np.array([int((x1 + x2) / 2), int(y1 + y2) / 2])

        delta_x = x2 - x1
        delta_y = y2 - y1
        max_delta = max(delta_x, delta_y)
        scale = int(max_delta * scale_factor)
        return center, scale

    def jitter_box(self, box_center, box_size):
        """
        random data augmentation
        :param center:
        :param scale:
        :return: new center, new scale, rotation
        """
        if self.train and self.jitter_trans > 0:
            jit_trans = self.jitter_trans * box_size * np.random.uniform(-1, 1, size=2)
            box_center += jit_trans
        if self.train and self.jitter_scale > 0:
            jit_scale = self.jitter_scale * np.random.randn() + 1
            jit_scale = np.clip(jit_scale, 1 - self.jitter_scale, 1 + self.jitter_scale)
            box_size *= jit_scale

        return box_center, box_size

    def crop_sample(self, sample, box_center, box_size):
        """crop image, mask, normal if exist"""
        x1, y1 = box_center - box_size / 2
        x2, y2 = x1 + box_size, y1 + box_size
        bbox = [x1, y1, x2, y2]
        sample['image'] = image_utils.crop(sample['image'], bbox, mode='reflect')
        sample['mask'] = image_utils.crop(sample['mask'], bbox, mode='const')
        return sample

    def color_jitter(self, sample):
        if self.train and self.jitter_color > 0:
            sample['image'] = image_utils.color_jitter(
                sample['image'], self.jitter_color, self.jitter_color, self.jitter_color, self.jitter_color)
        return sample

    def resize_image(self, sample):
        for key in ['image', 'mask']:
            image = sample[key]
            sample[key] = image_utils.resize(image, self.out_size)
            sample[key + '_inp'] = image_utils.resize(image, self.inp_size)
        return sample

    def rand_flip_lr(self, sample):
        if self.train and np.random.rand() > 0.5 and self.rdn_lr:
            for key in ['image', 'mask']:
                sample[key] = sample[key][:, ::-1].copy()
        return sample

    def clear_sample(self, sample: Dict):
        """
        :param sample:
        :return: mask: (1, 64, 64),
                 image: (3, 64, 64),
                 image_inp: (3, 224, 224),
                 index: str
        """
        out = {}
        # rescale
        sample['mask_img'] = (sample['mask'] > 0).astype(np.float32)
        sample['mask'] = (image_utils.resize(sample['mask'], self.cfg.reso_vox) > 0).astype(np.float32)
        sample['mask_inp'] = (sample['mask_inp'] > 0).astype(np.float32)
        sample['image'] = sample['image'] / 255 * 2 - 1
        sample['image_inp'] = sample['image_inp'] / 255 * 2 - 1

        # mask foreground
        sample['image'] = sample['image'] * sample['mask_img'] + (1 - sample['mask_img'])
        sample['bg'] = sample['image_inp']
        sample['image_inp'] = sample['image_inp'] * sample['mask_inp'] + (1 - sample['mask_inp'])

        # to tensor
        for key in ['image', 'mask', 'image_inp', 'mask_inp', 'bg']:
            if key in sample:
                out[key] = torch.FloatTensor(sample[key].copy().transpose([2, 0, 1]))
        if self.know_mean:
            out['mean_shape'] = torch.FloatTensor(sample['mean_shape'])

        return out