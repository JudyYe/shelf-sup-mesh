# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import numpy as np
import os

import tqdm
from PIL import Image

from .basedata import BaseData


class Ebay(BaseData):
    def __init__(self, cfg, dataset, split, train):
        if split == 'test':
            split = 'val'
        super().__init__(cfg, dataset, split, train)
        self.root_dir = 'data/ebay'
        self.image_dir = os.path.join(self.root_dir, 'images/chair_final')
        self.preload_anno()
        # 1028 train 24 test
        print('ebay', len(self))

    def preload_anno(self):
        findex = os.path.join(self.root_dir, self.split + '.txt')
        with open(findex) as fp:
            for line in tqdm.tqdm(fp):
                index, x, y = line.split()
                self.anno['rel_path'].append(index)
                mask_file = os.path.join(self.root_dir, 'segs', 'chair', index)
                # not too large dataset. Preload
                self.anno['mask'].append(np.array(Image.open(mask_file)))
                self.anno['bbox'].append(np.array([0, 0, int(x), int(y)]))