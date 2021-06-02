# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import numpy as np
import tqdm
from PIL import Image

from nnutils import mesh_utils
from .basedata import BaseData


class SN(BaseData):
    name_dict = {
        'Aero': '02691156',
        'Car': '02958343',
        'Chair': '03001627',
        'Mug': '03797390',
    }

    def __init__(self, cfg, dataset, split, train):
        if split == 'val':
            split = 'test'
        super().__init__(cfg, dataset, split, train)
        self.prior, self.cls = self.parse_name(dataset[2:])
        self.data_dir = 'data/shapenet/'
        self.n_view = cfg.DB.N_VIEW
        self.vox_size = cfg.SIZE.VOXEL

        self.image_dir = os.path.join(self.data_dir, 'Images', '{0}/{1}/{2}/{4}/{3}')  # cls, self.prior, shape, '%02d.png' % i
        self.vox_dir = os.path.join(self.data_dir, 'Voxels')
        self.mesh_dir = os.path.join(self.data_dir, 'Meshes')

        self.preload_anno()

    def parse_name(self, dataset: str):
        if '-' not in dataset:
            prior = 'uniform'
        else:
            dataset, prior = dataset.split('-')
        return prior, self.name_dict[dataset]

    def preload_anno(self):
        cls = self.cls
        list_file = os.path.join(self.data_dir, 'ImageSets', '%s_%s_old.lst' % (cls, self.split))
        if self.name_dict['Mug'] == cls:
            list_file = os.path.join(self.data_dir, 'ImageSets', '%s_%s.lst' % (cls, self.split))

        with open(list_file) as fp:
            for line in tqdm.tqdm(fp):

                shape = line.split()[0]
                if not self.train:
                    voxel_file = os.path.join(self.vox_dir, cls, '%s/vox%d.npz' % (shape, self.vox_size))
                    assert os.path.exists(voxel_file)
                    voxel = np.load(voxel_file)['vox']
                    voxel = voxel[np.newaxis].astype(np.float32)
                    self.vox_list.append(voxel)

                    mesh_file = os.path.join(self.mesh_dir, cls, '%s/model.obj' % (shape))
                    meshes = mesh_utils.load_mesh(mesh_file)
                    self.mesh_list.append(meshes)

                for i in range(self.n_view):
                    rel_path = (cls, self.prior, shape, '%02d.png' % i)
                    self.anno['rel_path'].append(rel_path)
                    self.anno['bbox'].append([0, 0, 224, 224])
                    self.anno['cad_index'].append(len(self.vox_list) - 1)

                    # for fast testing, only load one view per shape
                    if self.train == 0 or self.split != 'train':
                        break
                # if len(self) > 100 and self.train == 0:
                #     break

    def get_datapoint(self, index):
        sample = {}
        sample['image'], sample['mask'] = self.get_image(index)
        sample['bbox'] = self.get_bbox(index)
        sample['index'] = self.anno['rel_path'][index]

        if not self.train:
            cad_index = self.anno['cad_index'][index]
            sample['vox'] = self.vox_list[cad_index]
            sample['mesh'] = self.mesh_list[cad_index]
        return sample

    def get_image(self, index):
        """read mask on the fly"""
        img_path = self.anno['rel_path'][index]
        image_path = self.image_dir.format(*img_path, 'old_image')
        if self.name_dict['Mug'] == self.cls:
            image_path = self.image_dir.format(*img_path, 'image')

        image = np.array(Image.open(image_path))
        rgb = image[:, :, 0:3]
        mask = image[:, :, 3]
        mask = (mask > 0).astype(np.uint8)
        return rgb, mask