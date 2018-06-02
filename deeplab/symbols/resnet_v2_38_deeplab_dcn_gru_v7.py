#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/21 14:06
# @Author  : yueqing.zhuang
# @Site    : 
# @File    : resnet_v2_38_deeplab.py
# @Software: PyCharm

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from layers.RelationMap import RnnMap_cudnn
from body_net import resnet38_v5_dcn
from layers.OhemSoftmax import OhemSoftmax, OhemSoftmaxProp
from layers.weightedlogistic import WeightedLogisticRegression,WeightedLogisticRegressionProp

class resnet_v2_38_deeplab_dcn_gru_v7(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1.001e-5
        self.use_global_stats = True
        self.fix_gamma= False
        self.workspace = 2048
        self.rnn_type  ="GRU"
        self.patch_size = (1,1)
        self.num_hidden = 512
        self.activation ="relu"

    def get_fcn_top(self, conv_feature, label, name, num_classes, cfg, is_train,grad_scale=1.0):

        conv6a_bias =  mx.symbol.Variable(name="%s_conv_bias"%name, lr_mult=2.0, wd_mult=0.0)
        conv6a = mx.symbol.Convolution(name='%s_conv'%name, bias=conv6a_bias,data=conv_feature, dilate=(12, 12),
                                     kernel=(3, 3),num_filter=512,num_group=1, pad=(12, 12),
                                       stride=(1, 1), workspace=self.workspace)
        conv6a_relu = mx.symbol.Activation(name="%s_conv_relu"%name, data=conv6a, act_type='relu')

        linear19 = mx.symbol.Convolution(name='%s_linear'%name, data=conv6a_relu, dilate=(12, 12),
                                     kernel=(3, 3),num_filter=num_classes,num_group=1, pad=(12, 12),
                                       stride=(1, 1), workspace=self.workspace)

        #default use the ce to test
        if cfg.TRAIN.loss_type == "CE" or not is_train:
            softmax = mx.symbol.SoftmaxOutput(data=linear19, label=label, normalization='valid',
                                              multi_output=True,
                                              use_ignore=True, ignore_label=255, name="%s_softmax"%name,grad_scale=grad_scale)
        elif cfg.TRAIN.loss_type == "OHEM":
            from layers.OhemSoftmax import OhemSoftmax, OhemSoftmaxProp
            # softmax = mx.symbol.Custom(data=linear19, label=label, name='softmax', op_type='ohem_softmax',
            #                            ignore_label=255, thresh=0.6, min_kept=256, margin=-1)
            softmax = mx.symbol.SoftmaxOHEMOutput(data=linear19, label=label, multi_output=True,name="%s_softmax"%name,
                                       ignore_label=255, thresh=0.6, min_keep=256,grad_scale=grad_scale)
        elif cfg.TRAIN.loss_type == "FocalLoss":
            softmax = mx.symbol.SoftmaxFocalOutput(data=linear19, label=label,
                                                   # alphas=(0.1,0.1,0.1,0.3,0.3,0.2,0.5,0.3,0.1,0.2,0.1,0.2,0.7,0.1,0.5,0.5,0.5,1,0.5),
                                                   alphas = tuple([1]*19),
                                                   gamma=2, multi_output=True,ignore_label=255,grad_scale=grad_scale)
        else:
            return NotImplementedError
        return softmax,conv6a_relu

    def get_one_metric(self,conv_feature,compare_feature,name):

        dist_feature =  mx.symbol.concat(*[conv_feature,compare_feature])

        logistic = mx.symbol.Convolution(name='%s_conv_lateral' % name, data=dist_feature,
                                         kernel=(1, 1), num_filter=128, num_group=1,
                                         stride=(1, 1),workspace=self.workspace)

        logistic =  mx.symbol.Activation(name="%s_conv_lateral_relu"%name, data=logistic, act_type='relu')

        logistic = mx.symbol.Convolution(name='%s_conv' % name, data=logistic,
                                       kernel=(1, 1), num_filter=1, num_group=1,
                                       stride=(1, 1), workspace=self.workspace)

        return logistic


    def get_metric_top(self, conv_feature, metric_label,grad_scale=1.0):

        archor_feature = mx.symbol.Pad(conv_feature,mode="constant",pad_width=(0,0,0,0,1,1,1,1))

        metric_list = []
        for ix in range(0,3):
            for iy in range(0,3):
                compare_feature = mx.symbol.Pad(conv_feature,mode="constant",pad_width=(0,0,0,0,ix,2-ix,
                                                                                        iy,2-iy))
                archor_feature_crop = mx.symbol.Crop(*[archor_feature, conv_feature],offset=(ix,iy))
                compare_feature_crop = mx.symbol.Crop(*[compare_feature, conv_feature],offset=(ix,iy))
                metric_list.append(self.get_one_metric(archor_feature_crop, compare_feature_crop, "metric_"+str(ix-1)+"_"+str(iy-1)))

        metric_feature = mx.symbol.stack(*metric_list,axis=1)
        log_top = mx.sym.Custom(metric_feature,metric_label, grad_scale=grad_scale,clip_grad = 5.0,name='metric',
                            op_type='weighted_logistic_regression')

        return log_top

    def get_train_symbol(self, num_classes, cfg):
        """
        get symbol for training
        :param num_classes: num of classes
        :return: the symbol for training
        """
        if cfg.TRAIN.enable_crop:
            data_shape = (cfg.TRAIN.BATCH_IMAGES, 3, cfg.TRAIN.crop_size[0], cfg.TRAIN.crop_size[1])
        else:
            data_shape = (cfg.TRAIN.BATCH_IMAGES, 3, cfg.SCALES[0][0], cfg.SCALES[0][1])
        data = mx.sym.Variable('data', shape=data_shape)
        seg_cls_gt = mx.symbol.Variable(name='label')
        metric_gt = mx.symbol.Variable(name='metric_label')

        array = cfg.network.use_mult_label_weight
        num_instance = data_shape[0] * data_shape[2] * data_shape[3]
        metric_grad_scale = float(array[1]) / num_instance / 9

        # shared convolutional symbols
        P_feature = resnet38_v5_dcn.get_conv_feature(data, is_train=True, workspace=self.workspace,
                                                      fix_gamma=self.fix_gamma, use_global_stats=self.use_global_stats,
                                                      eps=self.eps)

        rnn_feature = RnnMap_cudnn(P_feature[3], name="RNNRelation", type=self.rnn_type, PatchSize=self.patch_size,
                       num_hidden=self.num_hidden, use_memory=True)

        array = cfg.network.use_mult_label_weight
        fcn_fusion, metric_feature =  self.get_fcn_top(rnn_feature, seg_cls_gt, 'FUSION', num_classes=num_classes, cfg=cfg,is_train=True,grad_scale=array[0])
        metric_top = self.get_metric_top(conv_feature=metric_feature, metric_label=metric_gt,grad_scale=metric_grad_scale)

        sym =  mx.symbol.Group([fcn_fusion,metric_top])
        return sym

    def get_test_symbol(self, num_classes,cfg):
        """
        get symbol for training
        :param num_classes: num of classes
        :return: the symbol for training
        """
        # shared convolutional symbols
        data_shape = (cfg.TEST.BATCH_IMAGES, 3, cfg.SCALES[0][0], cfg.SCALES[0][1])
        data = mx.sym.Variable('data', shape=data_shape)
        seg_cls_gt = mx.symbol.Variable(name='label')

        # shared convolutional symbols
        P_feature = resnet38_v5_dcn.get_conv_feature(data, is_train=True, workspace=self.workspace,
                                                     fix_gamma=self.fix_gamma, use_global_stats=self.use_global_stats,
                                                     eps=self.eps)



        rnn_feature = RnnMap_cudnn(P_feature[3], name="RNNRelation", type=self.rnn_type, PatchSize=self.patch_size,
                                   num_hidden=self.num_hidden, use_memory=True)

        fcn_fusion,_ = self.get_fcn_top(rnn_feature, seg_cls_gt, 'FUSION', num_classes=num_classes, cfg=cfg,
                                      is_train=False)

        return mx.sym.Group([fcn_fusion,P_feature[3],rnn_feature])

    def get_symbol(self, cfg, is_train=True):
        """
        return a generated symbol, it also need to be assigned to self.sym
        """

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES

        if is_train:
            self.sym = self.get_train_symbol(num_classes=num_classes,cfg=cfg)
        else:
            self.sym = self.get_test_symbol(num_classes=num_classes,cfg=cfg)

        return self.sym


    def init_weights(self, cfg, arg_params, aux_params):

        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=1)
        for k in self.symbol.list_arguments():
            if (k in self.data_shape_dict) or (k == 'label') :
                continue
            if k not in arg_params:
                print "Init arg_params:", k
                if k.endswith("linear_weight"):
                    print "_init_bilinear: ",k
                    arg_params[k] = mx.nd.zeros(shape=self.arg_shape_dict[k])
                    initializer._init_bilinear(k, arg_params[k])
                elif k.endswith("bias") or k.endswith("weight"):
                        arg_params[k] = mx.nd.zeros(shape=self.arg_shape_dict[k])
                        initializer(mx.init.InitDesc(k), arg_params[k])
                else:
                    print "normal ", k
                    if k.endswith('state'):
                        arg_params[k] = mx.nd.zeros(shape=self.arg_shape_dict[k])
                    else:
                        init_base =  mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=1)
                        rnn_init = mx.init.FusedRNN(init_base, num_layers=1, num_hidden=self.num_hidden, mode="gru", bidirectional=True)
                        arg_params[k] = mx.nd.zeros(shape=self.arg_shape_dict[k])
                        rnn_init._init_weight(mx.init.InitDesc(k), arg_params[k])

        for k in self.symbol.list_auxiliary_states():
            if k in self.data_shape_dict:
                continue
            if k not in aux_params:
                print "Init aux_params:", k
                aux_params[k] = mx.nd.zeros(shape=self.aux_shape_dict[k])

        # deconv

        # arg_params['res6a_branch2b1_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res6a_branch2b1_offset_bias'])
        # arg_params['res6a_branch2b1_offset_weight'] = mx.nd.zeros(
        #     shape=self.arg_shape_dict['res6a_branch2b1_offset_weight'])
        # arg_params['res7a_branch2b1_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['res7a_branch2b1_offset_bias'])
        # arg_params['res7a_branch2b1_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['res7a_branch2b1_offset_weight'])
