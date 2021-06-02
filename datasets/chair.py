# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
from .basedata import BaseData
from .p3d import P3dPlus
from .ebay import Ebay


class AllChair(BaseData):
    def __init__(self, args, dataset, split, train):
        if split == 'test':
            split = 'val'
        super().__init__(args, dataset, split, train)

        p3d = P3dPlus(args, 'pmChair', split, train)
        ebay = Ebay(args, 'ebay', split, train)
        self.data_list = [ebay, p3d]

        self.preload_anno()

    def preload_anno(self,):
        for data in self.data_list:
            index_list = [os.path.join(data.image_dir, e) for e in data.anno['rel_path']]
            self.anno['rel_path'].extend(index_list)
            self.anno['bbox'].extend(data.anno['bbox'])
            self.anno['mask'].extend(data.anno['mask'])