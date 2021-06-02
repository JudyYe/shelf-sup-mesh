# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import numpy as np
import scipy.io as sio

from datasets.basedata import BaseData

class CUB(BaseData):
    def __init__(self, cfg, dataset, split, train):
        super().__init__(cfg, dataset, split, train)
        self.suf = 'cleaned'
        self.root_dir = 'data/cub'
        self.preload_anno()
        np.random.seed(123)
        self.map = np.random.permutation(len(self))

    def preload_anno(self):
        """
        fill self.anno: rel_path, mask, bbox, cad_index, correspondingly
        :return:
        """
        anno_train_data_path = os.path.join(
            self.root_dir, 'cachedir/cub/data', '%s_cub_%s.mat' % (self.split, self.suf))
        self.image_dir = os.path.join(self.root_dir, 'CUB_200_2011/images')

        anno = sio.loadmat(anno_train_data_path, struct_as_record=False, squeeze_me=True)

        for i in range(len(anno['images'])):
            self.anno['rel_path'].append(anno['images'][i].rel_path)
            self.anno['mask'].append(anno['images'][i].mask)
            b = anno['images'][i].bbox
            self.anno['bbox'].append(np.array([b.x1, b.y1, b.x2, b.y2]))
            self.anno['cad_index'].append(-1)
        del anno
