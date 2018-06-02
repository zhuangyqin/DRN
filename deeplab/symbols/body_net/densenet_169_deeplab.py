#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/21 14:06
# @Author  : yueqing.zhuang
# @Site    : 
# @File    : resnet_v2_38_deeplab.py
# @Software: PyCharm

import cPickle
import mxnet as mx
from densenet_base import DenseNet_Base

class densenet_169_deeplab(DenseNet_Base):

    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.units = [6, 12, 32, 32]
        self.num_stage = 4
        self.growth_rate = 32
        self.reduction = 0.5
        self.drop_out = 0.2
        self.bottle_neck = True
        self.bn_mom = 0.9
        self.eps = 2e-5
        self.workspace = 512

    def get_densenet_conv(self,data):

        conv = self.DenseNet(data=data,units=self.units,
                             num_stage=self.num_stage, growth_rate=self.growth_rate,
                             reduction=self.reduction, drop_out=self.drop_out,
                             bottle_neck=self.bottle_neck,bn_mom=self.bn_mom,
                             workspace=self.workspace,eps=self.eps)

        return conv


    def get_fcn_top(self,conv_feature,label,num_classes,bootstrapping):

        conv6a_bias =  mx.symbol.Variable(name="conv6a_bias", lr_mult=2.0, wd_mult=0.0)
        conv6a = mx.symbol.Convolution(name='conv6a', bias=conv6a_bias,data=conv_feature, dilate=(12, 12),
                                     kernel=(3, 3),num_filter=512,num_group=1, pad=(12, 12),
                                       stride=(1, 1), workspace=self.workspace)
        conv6a_relu = mx.symbol.Activation(name="conv6a_relu", data=conv6a, act_type='relu')

        linear19 = mx.symbol.Convolution(name='linear19', data=conv6a_relu, dilate=(12, 12),
                                     kernel=(3, 3),num_filter=num_classes,num_group=1, pad=(12, 12),
                                       stride=(1, 1), workspace=self.workspace)
        if bootstrapping:
            from layers.OhemSoftmax import OhemSoftmax, OhemSoftmaxProp
            softmax = mx.symbol.Custom(data=linear19, label=label, name='softmax', op_type='ohem_softmax',
                                       ignore_label=255, thresh=0.6, min_kept=256, margin=-1)
        else:
            softmax = mx.symbol.SoftmaxOutput(data=linear19, label=label,normalization='valid', multi_output=True, use_ignore=True,
                                              ignore_label=255, name="softmax")
        return softmax

    def get_train_symbol(self, num_classes,bootstrapping=True):
        """
        get symbol for training
        :param num_classes: num of classes
        :return: the symbol for training
        """
        data = mx.sym.Variable('data')
        seg_cls_gt = mx.symbol.Variable(name='label')

        # shared convolutional symbols
        conv_feat = self.get_densenet_conv(data)
        top = self.get_fcn_top(conv_feat,seg_cls_gt,num_classes=num_classes,bootstrapping=bootstrapping)

        return top

    def get_test_symbol(self, num_classes,bootstrapping=False):
        """
        get symbol for training
        :param num_classes: num of classes
        :return: the symbol for training
        """
        # shared convolutional symbols
        data = mx.sym.Variable('data')
        seg_cls_gt = mx.symbol.Variable(name='label')

        conv_feat = self.get_densenet_conv(data)
        top = self.get_fcn_top(conv_feat,seg_cls_gt,num_classes=num_classes,bootstrapping=bootstrapping)

        return top

    def get_symbol(self, cfg, is_train=True):
        """
        return a generated symbol, it also need to be assigned to self.sym
        """

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES

        if is_train:
            bootstrapping = cfg.TRAIN.get('bootstrapping', False)
            self.sym = self.get_train_symbol(num_classes=num_classes,bootstrapping=bootstrapping)
        else:
            self.sym = self.get_test_symbol(num_classes=num_classes,bootstrapping=False)

        return self.sym

    def init_weights(self, cfg, arg_params, aux_params):

        arg_params['conv6a_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['conv6a_weight'])
        arg_params['conv6a_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['conv6a_bias'])

        arg_params['linear19_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['linear19_weight'])
        arg_params['linear19_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['linear19_bias'])

        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
        # init._init_bilinear('upsample_weight', arg_params['upsampling_weight'])
        for k in ['conv6a_weight','linear19_weight']:
            initializer(mx.init.InitDesc(k), arg_params[k])


