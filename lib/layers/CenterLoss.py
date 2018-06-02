#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/19 20:32
# @Author  : yueqing.zhuang
# @Site    : 
# @File    : CenterLoss.py
# @Software: PyCharm

import cv2
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np



class CenterLoss(mx.operator.CustomOp):
    def __init__(self,num_class, alpha, grad_scale, clip_grad, ignore_label):

        self.num_class = num_class
        self.alpha = float(alpha)
        self.clip_grad = float(clip_grad)
        self.grad_scale = float(grad_scale)
        self.ignore_label = ignore_label

    def forward(self, is_train, req, in_data, out_data, aux):

        # 1.reshape to 1 dims
        feature = in_data[0].transpose((0,2,3,1))
        self.diff_reshape = feature.shape
        dim = feature.shape[3]
        feature = feature.reshape((-1,dim))
        self.label = in_data[1].reshape((-1,))
        self.batch_size = self.label.shape[0]

        # last shape
        self.center = aux[0]
        self.center[self.center.shape[0] - 1, :] = 0

        # 2.calculate diff
        hist = nd.array(np.bincount(self.label.asnumpy().astype(int)))
        centers_selected = nd.take(self.center, self.label)
        self.centers_count = nd.take(hist, self.label)

        label_valid =  (self.label!=255).reshape((-1,1))
        self.diff =  label_valid*(feature - centers_selected)/ self.centers_count.reshape((-1,1))

        # 3.calculate output
        loss = mx.nd.sum(mx.nd.square(self.diff, axis=1), 1)
        out_data[0][:] = loss

        # 4.reshape diff
        self.diff =  self.diff

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):


        # 1.update class center
        for class_id in xrange(int(self.num_class)):
            center_diff = nd.where(nd.array((self.label == class_id).astype(np.int8)),
                                   x=self.diff, y=nd.zeros_like(self.diff))
            # print nd.where(nd.array(self.label==class_id),diff,nd.zeros(diff))
            # print "class id", class_id,
            self.center[class_id] += self.alpha * nd.sum(center_diff, axis=0)

        # print self.center
        # 2. assign grad to former layer
        self.assign(in_grad[0], req[0], self.diff.reshape(self.diff_reshape).transpose((0,3,1,2)) * self.grad_scale)



@mx.operator.register("centerloss")
class CenterLossProp(mx.operator.CustomOpProp):
    def __init__(self, num_class, alpha =0.05, grad_scale = 1.0, clip_grad=5.0, ignore_label = -1):
        super(CenterLossProp, self).__init__(need_top_grad=False)
        self.grad_scale = grad_scale
        self.clip_grad = clip_grad
        self.num_class = num_class
        self.alpha = alpha
        self.ignore_label = ignore_label

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def list_auxiliary_states(self):
        return ['center_states']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],1,in_shape[0][2],in_shape[0][3])

        batch_size = label_shape[0]*label_shape[1]*label_shape[2]*label_shape[3]
        # store diff , same shape as input batch

        # #Batch_size, # Channel #H #W
        # diff_shape = in_shape[0]
        center_shape = [int(self.num_class)+1, data_shape[1]]
        output_shape = [batch_size]

        return [data_shape, label_shape], [output_shape], [center_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return CenterLoss(self.num_class,self.alpha, self.grad_scale, self.clip_grad, self.ignore_label)