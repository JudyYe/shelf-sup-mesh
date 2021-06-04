# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import numpy as np
import pandas
import tqdm

from nnutils import image_utils
from .basedata import BaseData


class OpenImage(BaseData):
    def __init__(self, cfg, dataset, split, train):
        if split in ['val', 'test']:
            split = 'validation'
        super().__init__(cfg, dataset, split, train)
        self.data_dir = 'data/openimages/'
        self.filter_trunc = self.cfg.filter_trunc
        self.cls2id, self.id2cls = self.get_wnid()
        self.cats = self.parse_name(dataset[2:])
        self.image_dir = os.path.join(self.data_dir, 'JPEGImages')

        self.preload_anno()
        print(split, dataset, len(self))

    def preload_anno(self):
        if self.filter_trunc:
            ft_model = 'occBal'
            anno = pandas.read_csv(os.path.join(self.data_dir, 'Segs', '%s-obj-seg-%s.csv') % (self.split, ft_model))
        else:
            anno = pandas.read_csv(os.path.join(self.data_dir, 'Segs', '%s-annotations-object-segmentation.csv') % self.split)
        for cls_name in self.cats:
            df = self.filter_df(anno, cls_name)

            for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
                findex = row['ImageID']
                fmask = os.path.join(self.data_dir, 'Segs/Masks', row['MaskPath'])

                x1 = row['BoxXMin']
                x2 = row['BoxXMax']
                y1 = row['BoxYMin']
                y2 = row['BoxYMax']
                box = np.array([x1, y1, x2, y2])

                self.anno['rel_path'].append(findex + '.jpg')
                self.anno['mask'].append(fmask)
                self.anno['bbox'].append(box)

    def parse_name(self, dataset: str):
        cats = dataset.split('+')
        return cats

    def filter_df(self, anno, cls_name):
        wnid = self.cls2id[cls_name.replace(' ', '-')]
        df = anno.loc[anno['LabelName'] == wnid]
        if self.filter_trunc > 0:
            cnt = len(df)
            df = df.sort_values(by=['truncated'], ascending=False)
            valid_df = df.loc[df['truncated'] > self.filter_trunc]
            min_num = 1000
            if len(valid_df) > min_num:
                df = valid_df
            else:
                df = df.head(min_num)
            print('filter [%s] %d -> %d' % (cls_name, cnt, len(df)))
        return df

    def get_wnid(self):
        fname = os.path.join(self.data_dir, 'class-descriptions-boxable.csv')
        dg = pandas.read_csv(fname, header=None)
        id2cls = dict(zip(list(dg[0]), list(dg[1])))
        cls2id = dict(zip(list(dg[1]), list(dg[0])))

        id2cls = {e: id2cls[e].replace(' ', '-') for e in id2cls}
        cls2id = {e.replace(' ', '-'): cls2id[e] for e in cls2id}
        return cls2id, id2cls

    def align_images(self, sample):
        """In OpenImages, sometimes masks and images are not in the same resolution"""
        im = sample['mask']
        height, width = im.shape[0: 2]

        for key in ['image']:
            if sample[key].shape[0] != height or sample[key].shape[1] != width:
                sample[key] = image_utils.resize(sample[key], [width, height])
        return sample

    def get_datapoint(self, index):
        sample = {}
        sample['image'] = self.get_image(index)
        sample['mask'] = self.get_mask(index)
        sample = self.align_images(sample)

        H, W = sample['mask'].shape[0: 2]
        bbox = self.get_bbox(index)
        bbox[0] = bbox[0] * W
        bbox[1] = bbox[1] * H
        bbox[2] = bbox[2] * W
        bbox[3] = bbox[3] * H
        sample['bbox'] = bbox

        return sample