#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: weightedlogistic.py
@time: 2017/12/2 16:24
"""
import mxnet as mx
from utils.image import plot_border

class WeightedLogisticRegression(mx.operator.CustomOp):
    def __init__(self, grad_scale, clip_grad):
        self.grad_scale = float(grad_scale)
        self.clip_grad = float(clip_grad)

    def forward(self, is_train, req, in_data, out_data, aux):

        self.assign(out_data[0], req[0], mx.nd.divide(1, (1 + mx.nd.exp(- in_data[0]))))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):

        total_num =  out_data[0].size
        neg_grad_scale = mx.nd.sum(in_data[1])/total_num
        pos_grad_scale = 1 - neg_grad_scale
        # print "pos_scale",pos_grad_scale.asnumpy()[0],"neg_scale",neg_grad_scale.asnumpy()[0]

        in_grad[0][:] = ((out_data[0] - 1) * in_data[1] * pos_grad_scale + out_data[0] * (
            1 - in_data[1]) * neg_grad_scale) * self.grad_scale
        if self.clip_grad>0:
            in_grad[0][:]= mx.nd.clip(in_grad[0],-self.clip_grad,self.clip_grad)


@mx.operator.register("weighted_logistic_regression")
class WeightedLogisticRegressionProp(mx.operator.CustomOpProp):
    def __init__(self,  grad_scale=1.0,clip_grad=-1.0):
        self.grad_scale = grad_scale
        self.clip_grad = clip_grad
        super(WeightedLogisticRegressionProp, self).__init__(False)

    def list_arguments(self):
        return ['data', 'label']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        shape = in_shape[0]
        return [shape, shape], [shape]
    def create_operator(self, ctx, shapes, dtypes):
        return WeightedLogisticRegression(self.grad_scale,self.clip_grad)