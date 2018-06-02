# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Zheng Zhang
# --------------------------------------------------------

import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

config.MXNET_VERSION = ''
config.output_path = ''
config.symbol = ''
config.gpus = ''
config.CLASS_AGNOSTIC = True
config.SCALES = [(360, 600)]  # first is scale (the shorter side); second is max size

# default training
config.default = edict()
config.default.frequent = 1000
config.default.kvstore = 'device'

# network related params
config.network = edict()
config.network.pretrained = '../model/pretrained_model/resnet_v1-101'
config.network.pretrained_epoch = 0
config.network.COLOR_SCALE = -1
config.network.PIXEL_MEANS = np.array([103.06, 115.90, 123.15])
config.network.PIXEL_STDS= None

config.network.LABEL_STRIDE = 1
config.network.FIXED_PARAMS = []
config.network.FIXED_PARAMS_PATTERN = []
config.network.use_context = False
config.network.use_mult_label = False
config.network.use_concat = False
config.network.use_dropout = False
config.network.use_l2scale = False
config.network.use_l2reg = False
config.network.use_fusion = False
config.network.use_mult_label_weight = np.array([1.0,1.0,1.0,1.0,1.0,1.0])
config.network.ratio = 1
config.network.mult_loss = False
config.network.use_crop_context = True
config.network.use_metric = False
config.network.use_sigmoid_metric = False
config.network.scale_list = [1,2,4]
config.network.crop_context_scale = 0.7
config.network.use_weight = False
config.network.use_origin = False

# dataset related params
config.dataset = edict()
config.dataset.dataset = 'cityscapes'
config.dataset.image_set = 'leftImg8bit_train'
config.dataset.test_image_set = 'leftImg8bit_val'
config.dataset.root_path = '../data'
config.dataset.dataset_path = '../data/cityscapes'
config.dataset.NUM_CLASSES = 19
config.dataset.annotation_prefix = 'gtFine'


config.TRAIN = edict()
config.TRAIN.use_center = False
config.TRAIN.use_one_center = False
config.TRAIN.lr = 0
config.TRAIN.lr_step = ''
config.TRAIN.warmup = False
config.TRAIN.warmup_lr = 0
config.TRAIN.warmup_step = 0
config.TRAIN.momentum = 0.9
config.TRAIN.wd = 0.0005
config.TRAIN.alpha = 0.05
config.TRAIN.optimizer = "sgd"
config.TRAIN.begin_epoch = 0
config.TRAIN.end_epoch = 0
config.TRAIN.model_prefix = 'deeplab'
config.TRAIN.lr_type = 'MultiStage'
config.TRAIN.FIXED_PARAMS_PATTERN = ''
config.TRAIN.FIXED_PARAMS_PATTERN_LR_MULT = 1
config.TRAIN.eval_data_frequency = 1
config.TRAIN.use_dynamic= False
config.TRAIN.use_mult_metric= False
config.TRAIN.use_crl_ses = False

config.TRAIN.enable_crop = True
config.TRAIN.crop_size = [1024,768]
config.TRAIN.enable_scale =True
config.TRAIN.scale_range =[0.5,2]
config.TRAIN.enable_rotation= True
config.TRAIN.rotation_range =[-10,10]
config.TRAIN.flipped_ratio = 0.5
config.TRAIN.use_balance = False
config.TRAIN.enable_ignore_border = False
config.TRAIN.ignore_border_size = 0
# whether resume training
config.TRAIN.RESUME = False
config.TRAIN.SHUFFLE = True
config.TRAIN.FINTUNE = False
config.TRAIN.momentum = 0.9
config.TRAIN.enable_ohem = False
#loss_type "CE",and "OHEM",and "FocalLoss"
config.TRAIN.loss_type = "CE"
config.TRAIN.BATCH_IMAGES = 1
config.TRAIN.use_global_stats= True
config.TRAIN.use_thread = True

# test config
config.TEST = edict()
config.TEST.BATCH_IMAGES = 1
config.TEST.test_epoch = 0

# close the data augruments
config.TEST.enable_crop = False
config.TEST.crop_size = [1024,768]
config.TEST.enable_scale = False
config.TEST.scale_range =[0.5,2]
config.TEST.enable_rotation= False
config.TEST.rotation_range =[-10,10]
config.TEST.flipped_ratio = 0

# test the flipping or mulit-stage test
config.TEST.use_flipping = False
config.TEST.num_steps = 1
config.TEST.save_h5py = False
config.TEST.apply_crf = False
config.TEST.ms_array = np.ones((1))

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    if k == 'TRAIN':
                        if 'BBOX_WEIGHTS' in v:
                            v['BBOX_WEIGHTS'] = np.array(v['BBOX_WEIGHTS'])
                    elif k == 'network':
                        if 'PIXEL_MEANS' in v:
                            v['PIXEL_MEANS'] = np.array(v['PIXEL_MEANS'])
                        if 'PIXEL_STDS' in v:
                            if v['PIXEL_STDS'] is not None:
                                v['PIXEL_STDS'] = np.array(v['PIXEL_STDS'])
                            else:
                                v['PIXEL_STDS'] = None
                        if 'use_mult_label_weight' in v:
                            v['use_mult_label_weight'] = np.array(v['use_mult_label_weight'])
                    elif k == 'TEST':
                        if 'ms_array' in v:
                            v['ms_array'] = np.array(v['ms_array'])
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    if k == 'SCALES':
                        config[k] = [tuple(i) for i in v]
                    elif k == 'RANGE_SCALES':
                        config[k] = tuple(v)
                    else:
                        config[k] = v

            else:
                raise ValueError("key must exist in config.py")
