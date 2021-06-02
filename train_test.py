# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os

import torch

from models.trainer import TrainerFactory
from models.evaluator import Evaluator

from absl import app
from config.config_flag import *
FLAGS = flags.FLAGS

def main(_):
    train_test()

def train_test():
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    if not FLAGS.train and FLAGS.checkpoint is not None:
        # test
        print('loading from flag file', FLAGS.exp, FLAGS.cyc_normal_loss, FLAGS.filter_trunc)
        print('test!!!!')
        trainer = TrainerFactory(FLAGS.g_mod)
        evaluator = Evaluator(trainer.cfg)
        trainer.build_val()

        val_dataloader = trainer.val_dataloader
        model_name = os.path.basename(FLAGS.checkpoint).split('.')[0]
        model_name = model_name + '_%s' % FLAGS.dataset + '_%s' % FLAGS.test_mod
        if hasattr(evaluator, FLAGS.test_mod):
            with torch.no_grad():
                test_func = eval('evaluator.' + FLAGS.test_mod)
                test_func(trainer.G, val_dataloader, trainer.log_dir, prefix=model_name)
        else:
            raise NotImplementedError('Not implement evaluation mode')
    else:
        # train
        trainer = TrainerFactory(FLAGS.g_mod)
        print('train ####')
        trainer.build_train()
        trainer.train()


if __name__ == '__main__':
    app.run(main)