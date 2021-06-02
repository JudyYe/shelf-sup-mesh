# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import re

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from pytorch3d.structures import Meshes, join_meshes_as_batch
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data import DataLoader


def build_dataloader(cfg, split, is_train=True, shuffle=None, name=None):
    if shuffle is None:
        shuffle = is_train
    # test: for vis sakes, limit batch size within 16.
    bs = cfg.SOLVER.BATCH_SIZE if is_train else min(16, cfg.SOLVER.BATCH_SIZE)

    if name is None:
        name = cfg.DB.NAME
    dataset = [build_dataset(cfg, each_name, split, is_train) for each_name in name.split('+')]
    dataset = ConcatDataset(dataset)
    loader = DataLoader(dataset, batch_size=bs, collate_fn=collate_meshes,
                        shuffle=shuffle, drop_last=is_train, num_workers=8)
    return loader


def build_dataset(args, name, split, is_train=True):
    # Curated-3
    if name.startswith('cub'):
        from .cub import CUB as dset
    elif name == 'chair':
        from .chair import AllChair as dset
    elif name[0:2] == 'im':
        from .quadruped import QuadImNet as dset
    # shapenet-3
    elif name.startswith('sn'):
        from .shapenet import SN as dset
    # Openimages
    elif name[0:2] == 'oi':
        from .openimages import OpenImage as dset
    else:
        raise NotImplementedError('not implemented %s' % name)
    dset = dset(args, name, split, is_train)
    return dset


# def dataloader(dataset, args, split):
#     if dataset[0:3] == 'cub':
#         from data.cub import CUB as dset
#     elif dataset[0:2] == 'sn':
#         from data.shape_net import SN as dset
#         # else:
#         #     from data.shapenet import SN as dset
#     elif dataset[0:2] == 'im':
#         from data.imnet import QuadImNet as dset
#     elif dataset[0:2] == 'oi':
#         from data.open_image import OpenImage as dset
#     elif dataset == 'allChair':
#         from data.all_chair import AllChair as dset
#     else:
#         raise NotImplementedError('%s ' % dataset)
#     bs = args.batch_size
#     dset = dset(args, dataset, split)
#     print('dataset len in total: ', len(dset))
#
#     loader_kwargs = {
#         'batch_size': bs,
#         'num_workers': 8,
#         'shuffle': (split=='train'),
#         'drop_last': split == 'train',
#         'collate_fn': collate_meshes,
#     }
#     loader = DataLoader(dset, **loader_kwargs)
#     return loader


def collate_meshes(batch):
    """
    collate function specifiying Meshes collation
    :return:
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")


    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, Meshes):
        return join_meshes_as_batch(batch)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_meshes([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_meshes([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_meshes(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_meshes(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
