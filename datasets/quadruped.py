# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import numpy as np
import os
import scipy.io as sio

from nnutils import image_utils
from .basedata import BaseData


class QuadImNet(BaseData):
    imnet2wnid = {
        'rhino': ['n02391994'], 'giraffe': ['n02439033'], 'camel': ['n02437312'], 'hippo': ['n02398521'],
        'fox': ['n02119022', 'n02119789', 'n02120079', 'n02120505', ], 'bear': ['n02132136', 'n02133161', 'n02131653'],
        'leopard': ['n02128385'], 'bison': ['n02410509'], 'buffalo': ['n02408429', 'n02410702'],
        'donkey': ['n02390640', 'n02390738'], 'goat': ['n02416519', 'n02417070'], 'beest': ['n02421449', 'n02422106'],
        'kangaroo': ['n01877812'], 'german-shepherd': ['n02106662', 'n02107574', 'n02109047'],
        'pig': ['n02396427', 'n02395406', 'n02397096'], 'lion': ['n02129165', ], 'llama': ['n02437616', 'n02437971'],
        'tapir': ['n02393580', 'n02393940'], 'tiger': ['n02129604'], 'warthog': ['n02397096'],
        'wolf': ['n02114367', 'n02114548', 'n02114712'], 'horse': ['n02381460'],
        'sheep': ['n10588074'],
        'cow': ['n01887787'], 'dog': ['n02381460'], 'elephant': ['n02504013'],
    }

    def __init__(self, args, dataset, split, train):
        if split == 'test':
            split = 'val'
        super().__init__(args, dataset, split, train)
        # todo:
        self.data_dir = 'data/quads'
        self.image_dir = self.data_dir

        self.cats = self.parse_name(dataset[2:])
        self.preload_anno()
        np.random.seed(123)
        self.map = np.random.permutation(len(self))

    def parse_name(self, dataset: str):
        if 'All' in dataset:
            cats = self.imnet2wnid.keys()
        else:
            cats = dataset.split('-')
        return cats

    def preload_anno(self):
        # load mean shape
        vox_path = os.path.join(self.data_dir, 'Voxs', 'mean_%d.npz' % self.cfg.reso_vox)
        vox = np.load(vox_path)['vox']
        vox = np.flip(vox, [-2])
        vox = vox.transpose([2, 1, 0])  # x-zswip
        self.vox_list.append(vox[None].copy())

        # # load pascal first
        pascal_list = ['cow', 'dog', 'horse', 'sheep']
        for cls in self.cats:
            if cls not in pascal_list:
                continue
            anno = os.path.join(self.data_dir, 'Annos', '%s_%s.mat' % (cls, self.split))
            if not os.path.exists(anno):
                print('PASCAL annotation not exist!!! ', anno)
                continue
            anno = sio.loadmat(anno, struct_as_record=False, squeeze_me=True)['images']
            for i in range(len(anno)):
                self.anno['rel_path'].append(os.path.join('pascal', anno[i].rel_path))
                self.anno['mask'].append(anno[i].mask)
                bbox = anno[i].bbox
                self.anno['bbox'].append(np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2]))
            print('loaded pascal %s' % cls)

        # load ImageNet
        for key in self.cats:
            if key not in self.imnet2wnid:
                continue
            for cls in self.imnet2wnid[key]:
                mat_file = os.path.join(self.data_dir, 'Annos', '%s_%s.mat' % (cls, self.split))
                if not os.path.exists(mat_file):
                    print('not exist annotation !!! ', mat_file)
                    continue
                anno = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)['images']

                for i in range(len(anno)):
                    rel_path = anno[i].rel_path
                    if not '/' in rel_path:
                        # heterogeneous nameing: wnid/wnid_num.JPEG, wnid_num.JPEG
                        rel_path = os.path.join(rel_path.split('_')[0], rel_path)
                    self.anno['rel_path'].append(os.path.join('imnet', rel_path))
                    self.anno['mask'].append(anno[i].mask)
                    bbox = anno[i].bbox
                    self.anno['bbox'].append(np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2]))
                print('loaded %s' % cls)

        print('Done preload!!', len(self.anno['rel_path']))

    # todo: neccessary?
    def align_images(self, sample):
        im = sample['image']
        height, width = im.shape[0: 2]

        for key in ['mask']:
            if sample[key].shape[0] != height or sample[key].shape[1] != width:
                sample[key] = image_utils.resize(sample[key], [width, height])
        return sample

    def get_mean_shape(self, index):
        return self.vox_list[0]