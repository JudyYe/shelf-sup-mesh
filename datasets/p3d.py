# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os

import numpy as np
from mat4py import loadmat
from .basedata import BaseData


class P3dPlus(BaseData):
    def __init__(self, cfg, dataset, split, train):
        if split == 'test':
            split = 'val'
        super().__init__(cfg, dataset, split, train)
        name_dict = {
            'pmAero': 'aeroplane',
            'pmCar': 'car',
            'pmChair': 'chair',
            'pmMotor': 'motorbike'
        }
        self.cls = name_dict[self.dataset]
        self.root_dir = 'data/p3d'
        self.image_dir = os.path.join(self.root_dir, 'Images/')
        self.preload_ann()
        print('p3d %s: ' % split, len(self))

    def preload_ann(self):
        """
        fill self.anno: rel_path, mask, bbox, cad_index, correspondingly
        :return:
        """
        ann_path = os.path.join(self.root_dir, 'Det/%s_%s.mat' % (self.cls, self.split))

        anno = loadmat(ann_path)
        anno_list = anno['images']

        self.anno['rel_path'] = anno_list['rel_path']
        self.anno['mask'] = anno_list['mask']
        for bbox in anno_list['bbox']:
            bbox = np.array([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
            self.anno['bbox'].append(bbox)
        self.anno['cad_index'] = anno_list['cad_index']
