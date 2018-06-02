# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zheng Zhang
# --------------------------------------------------------

import _init_paths

import time
import argparse
import logging
import pprint
import os
import sys
from config.config import config, update_config
from utils.sutil import make_divisible

def parse_args():
    parser = argparse.ArgumentParser(description='Train deeplab network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))


import shutil
import mxnet as mx
import re
from symbols import *
from mxnet.optimizer import SGD
from core import callback, metric
from core.loader_v1 import TrainDataLoader
from core.module import MutableModule
from utils.load_data import load_gt_segdb, merge_segdb
from utils.load_model import load_param,load_preload_opt_states
from utils.PrefetchingIter import PrefetchingIter
from utils.create_logger import create_env
from utils.lr_scheduler import *
from dataset import *

Debug = 0

def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):


    logger, final_output_path, _, tensorboard_path = create_env(config.output_path, args.cfg, config.dataset.image_set)
    prefix = os.path.join(final_output_path, prefix)

    # print config
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    print "config.symbol",config.symbol
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=True)

    # setup multi-gpu
    input_batch_size = config.TRAIN.BATCH_IMAGES * len(ctx)
    NUM_GPUS = len(ctx)

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in config.dataset.image_set.split('+')]
    segdbs = [load_gt_segdb(config.dataset.dataset, image_set, config.dataset.root_path, config.dataset.dataset_path,
                            result_path=final_output_path,flip=True)
              for image_set in image_sets]
    segdb = merge_segdb(segdbs)

    segdb = segdb[:200]

    # load training data
    train_data = TrainDataLoader(sym, segdb, config, batch_size=input_batch_size,
                                 shuffle=config.TRAIN.SHUFFLE, ctx=ctx,use_context=config.network.use_context,
                                 use_mult_label=config.network.use_mult_label)

    # loading val data
    val_image_set = config.dataset.test_image_set
    val_root_path = config.dataset.root_path
    val_dataset = config.dataset.dataset
    val_dataset_path = config.dataset.dataset_path
    val_imdb = eval(val_dataset)(val_image_set, val_root_path, val_dataset_path, result_path=final_output_path)
    val_segdb = val_imdb.gt_segdb()

    val_segdb = val_segdb[:50]

    val_data = TrainDataLoader(sym, val_segdb, config, batch_size=input_batch_size,
                                 shuffle=config.TRAIN.SHUFFLE, ctx=ctx,use_context=config.network.use_context,
                               use_mult_label=config.network.use_mult_label)

    # infer max shape
    scales = [(config.TRAIN.crop_size[0], config.TRAIN.crop_size[1])] if config.TRAIN.enable_crop else config.SCALES
    label_stride = config.network.LABEL_STRIDE
    if config.network.use_context:
        max_data_shape = [ ('data', (config.TRAIN.BATCH_IMAGES, 3,
                                    config.TRAIN.crop_size[0], config.TRAIN.crop_size[1])),
                            ('origin_data', (config.TRAIN.BATCH_IMAGES, 3,
                                           config.SCALES[0][0]/4, config.SCALES[0][1]/4)),
                          ('rois', (config.TRAIN.BATCH_IMAGES,5))]
    else:
        max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3,
                                    config.TRAIN.crop_size[0], config.TRAIN.crop_size[1]))]
    if config.network.use_mult_label:
        max_label_shape = [('label', (config.TRAIN.BATCH_IMAGES, 1,
                                      make_divisible(config.TRAIN.crop_size[0], label_stride)//config.network.LABEL_STRIDE,
                                      make_divisible(config.TRAIN.crop_size[1],
                                                     label_stride) // config.network.LABEL_STRIDE)),
                           ('origin_label', (config.TRAIN.BATCH_IMAGES, 1,
                                              config.SCALES[0][0] / 4, config.SCALES[0][1] / 4))]

    else:
        max_label_shape = [('label', (config.TRAIN.BATCH_IMAGES, 1,
                                      max([make_divisible(v[0], label_stride) for v in scales])//config.network.LABEL_STRIDE,
                                      max([make_divisible(v[1], label_stride) for v in scales])//config.network.LABEL_STRIDE))]

    print sym.list_arguments()
    print 'providing maximum shape', max_data_shape, max_label_shape
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape, max_label_shape)
    print 'providing maximum shape', max_data_shape, max_label_shape

    # infer shape
    data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)
    pprint.pprint(data_shape_dict)
    sym_instance.infer_shape(data_shape_dict)

    nset = set()
    for nm in sym.list_arguments():
        if nm in nset:
            raise ValueError('Duplicate names detected, %s' % str(nm))
        nset.add(nm)

    # load and initialize params
    if config.TRAIN.RESUME:
        print 'continue training from ', begin_epoch
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
        preload_opt_states = load_preload_opt_states(prefix, begin_epoch)
    else:
        print pretrained
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        preload_opt_states = None
        if not config.TRAIN.FINTUNE:
           fixed_param_names=sym_instance.init_weights(config, arg_params, aux_params)

    # check parameter shapes
    # sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

    # create solver
    fixed_param_prefix = config.network.FIXED_PARAMS
    data_names = [k[0] for k in train_data.provide_data_single]
    label_names = [k[0] for k in train_data.provide_label_single]

    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, max_data_shapes=[max_data_shape for _ in xrange(NUM_GPUS)],
                        max_label_shapes=[max_label_shape for _ in xrange(NUM_GPUS)], fixed_param_prefix=fixed_param_prefix)

    # metric
    imagecrossentropylossmetric=metric.ImageCrossEntropyLossMetric()
    localmetric = metric.LocalImageCrossEntropyLossMetric()
    globalmetric = metric.GlobalImageCrossEntropyLossMetric()
    pixcelAccMetric = metric.PixcelAccMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    if config.network.use_mult_label:
        metric_list = [imagecrossentropylossmetric, localmetric,globalmetric,pixcelAccMetric]
    else:
        metric_list = [imagecrossentropylossmetric,pixcelAccMetric]
    # rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric
    for child_metric in metric_list:
        eval_metrics.add(child_metric)

    # callback
    batch_end_callback = [callback.Speedometer(train_data.batch_size, frequent=args.frequent),
                          callback.TensorboardCallback(tensorboard_path,prefix="train/batch")]
    epoch_end_callback = mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True)
    shared_tensorboard = batch_end_callback[1]

    epoch_end_metric_callback= callback.TensorboardCallback(tensorboard_path,shared_tensorboard=shared_tensorboard,
                                                            prefix="train/epoch")
    eval_end_callback = callback.TensorboardCallback(tensorboard_path,shared_tensorboard=shared_tensorboard,
                                                     prefix="val/epoch")
    lr_callback = callback.LrCallback(tensorboard_path,shared_tensorboard=shared_tensorboard,prefix='train/batch')

    #decide learning rate
    base_lr = lr
    lr_factor = 0.1
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(segdb) / input_batch_size) for epoch in lr_epoch_diff]
    print 'lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters

    if config.TRAIN.lr_type == "MultiStage":
        lr_scheduler = LinearWarmupMultiStageScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr,
                                                   config.TRAIN.warmup_step,args.frequent,stop_lr=lr*0.01)
    elif config.TRAIN.lr_type == "MultiFactor":
        lr_scheduler = LinearWarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr,
                                                   config.TRAIN.warmup_step,args.frequent)

    optimizer_params = {'momentum': config.TRAIN.momentum,
                        'wd': config.TRAIN.wd,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0,
                        'clip_gradient': None}
    optimizer = SGD(**optimizer_params)

    freeze_layer_pattern = config.TRAIN.FIXED_PARAMS_PATTERN
    if freeze_layer_pattern.strip():
        args_lr_mult = {}
        re_prog = re.compile(freeze_layer_pattern)
        if freeze_layer_pattern:
            fixed_param_names = [name for name in sym.list_arguments() if re_prog.match(name)]
        print "fixed_params_names:"
        print(fixed_param_names)
        for name in fixed_param_names:
            args_lr_mult[name] = config.TRAIN.FIXED_PARAMS_PATTERN_LR_MULT
    else:
        args_lr_mult = {}
    optimizer.set_lr_mult(args_lr_mult)

    if not isinstance(train_data, PrefetchingIter):
        train_data = PrefetchingIter(train_data)

    if not isinstance(val_data,PrefetchingIter):
        val_data = PrefetchingIter(val_data)

    if Debug:
        monitor = mx.monitor.Monitor(1)
    else:
        monitor = None

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
            eval_end_callback=eval_end_callback,epoch_end_metric_callback=epoch_end_metric_callback,
            optimizer=optimizer,eval_data=val_data,arg_params=arg_params, aux_params=aux_params,
            begin_epoch=begin_epoch, num_epoch=end_epoch,allow_missing=begin_epoch==0,allow_extra=True,
            monitor=monitor,preload_opt_states=preload_opt_states,eval_data_frequency=config.TRAIN.eval_data_frequency)

def main():
    print 'Called with argument:', args
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    train_net(args, ctx, config.network.pretrained, config.network.pretrained_epoch, config.TRAIN.model_prefix,
              config.TRAIN.begin_epoch, config.TRAIN.end_epoch, config.TRAIN.lr, config.TRAIN.lr_step)

if __name__ == '__main__':
    main()
