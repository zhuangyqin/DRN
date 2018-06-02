#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: resnet38_v3.py
@time: 2017/11/17 19:22
"""

import mxnet as mx
no_bias = True
use_global_stats = True
fix_gamma = False
eps = 1.001e-5

def Conv(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                                 dilate=dilate, no_bias=no_bias, name=('%s' % name), workspace=2048)
    return conv

def ReLU(data, name):
    return mx.symbol.Activation(data=data, act_type='relu', name=name)

def BN(data,name=None, suffix=''):
    bn = mx.symbol.BatchNorm(data=data, name=('bn%s' % suffix), eps=eps,
                             use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    return bn

def AC_CONV(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None, suffix=''):
    act = ReLU(data,name=('%s_relu' % name))
    conv = Conv(data=act, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, dilate=dilate,
                name=name)
    return conv

def BN_AC(data,name=None, suffix=''):
    bn = mx.symbol.BatchNorm(data=data, name=('bn%s' % suffix), eps=eps, use_global_stats=use_global_stats, fix_gamma=fix_gamma)
    act = ReLU(data=bn, name=('%s_relu' % name))
    return act

def BN_AC_CONV(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=None, suffix=''):
    act = BN_AC(data=data, name=('%s_relu' % name),suffix=suffix)
    conv = Conv(data=act, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, dilate=dilate,
                   name=name)
    return conv

def ResidualFactory_d(data, num_3x3_a, num_1x1_b, num_3x3_c, suffix, dilate):

    bn_ac = BN_AC(data,name=('res%s_branch2a' % suffix),suffix=('%s_branch2a' % suffix))
    convb3 = Conv(bn_ac, num_filter=num_3x3_a, kernel=(3, 3),  stride=(2, 2), pad=dilate, dilate=dilate, name=('res%s_branch2a' % suffix),)
    convb1 = Conv(bn_ac, num_filter=num_1x1_b, kernel=(1, 1),  stride=(2, 2), pad=(0, 0), dilate=(1, 1), name=('res%s_branch1' % suffix), )
    bn_ac = BN_AC_CONV(convb3, num_filter=num_3x3_c, kernel=(3,3), stride=(1, 1), pad=dilate, dilate=dilate,name=('res%s_branch2b1' % suffix),
                       suffix='%s_branch2b1' % suffix)
    summ = mx.symbol.ElementWiseSum(*[bn_ac, convb1], name=('res%s' % suffix))
    return summ


def ResidualFactory_d2(data, num_3x3_a, num_1x1_b, num_3x3_c, suffix):

    bn_ac = BN_AC(data,name=('res%s_branch2a' % suffix),suffix=('%s_branch2a' % suffix))
    convb3 = Conv(bn_ac, num_filter=num_3x3_a, kernel=(3, 3),  stride=(1, 1), pad=(1, 1), dilate=(1, 1), name=('res%s_branch2a' % suffix),)
    convb1 = Conv(bn_ac, num_filter=num_1x1_b, kernel=(1, 1),  stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=('res%s_branch1' % suffix), )
    bn_ac = BN_AC_CONV(convb3, num_filter=num_3x3_c, kernel=(3,3), stride=(1, 1), pad=(2, 2), dilate=(2, 2),name=('res%s_branch2b1' % suffix),
                       suffix='%s_branch2b1' % suffix)
    summ = mx.symbol.ElementWiseSum(*[bn_ac, convb1], name=('res%s' % suffix))
    return summ

def ResidualFactory_a(data, num_1x1_a, num_3x3_b, num_1x1_c,num_1x1_d, suffix):

    bn_ac = BN_AC(data,name=('res%s_branch2a' % suffix),suffix=('%s_branch2a' % suffix))
    convb3 = Conv(bn_ac, num_filter=num_1x1_a, kernel=(1, 1),  stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=('res%s_branch2a' % suffix))

    convb1 = Conv(bn_ac, num_filter=num_1x1_d, kernel=(1, 1),  stride=(1, 1), pad=(0, 0), dilate=(1, 1), name=('res%s_branch1' % suffix))
    bn_ac = BN_AC_CONV(convb3, num_filter=num_3x3_b, kernel=(3,3), stride=(1, 1), pad=(4, 4), dilate=(4, 4),name=('res%s_branch2b1' % suffix),
                       suffix='%s_branch2b1'% suffix)
    bn_ac = BN_AC_CONV(bn_ac, num_filter=num_1x1_c, kernel=(1, 1), stride=(1, 1), pad=(0, 0), dilate=(1, 1),
                       name=('res%s_branch2b2' % suffix),
                       suffix='%s_branch2b2'% suffix)
    summ = mx.symbol.ElementWiseSum(*[bn_ac, convb1], name=('res%s' % suffix))
    return summ


def ResidualFactory_c(data, num_3x3_a, num_3x3_b, dilate, suffix):

    branch2a = BN_AC_CONV(data=data, num_filter=num_3x3_a, kernel=(3, 3),stride=(1, 1), pad=dilate, dilate=dilate, name=('res%s_branch2a' % suffix),
                        suffix=('%s_branch2a' % suffix))
    branch2b = BN_AC_CONV(data=branch2a, num_filter=num_3x3_b, kernel=(3, 3),stride=(1, 1), pad=dilate, dilate=dilate, name=('res%s_branch2b1' % suffix),
                        suffix=('%s_branch2b1' % suffix))
    summ = mx.symbol.ElementWiseSum(*[data, branch2b], name=('res%s' % suffix))

    return summ

def ResidualFactory_x(data, num_3x3_a, num_3x3_b, dilate, suffix):

    bn = BN(data,suffix=('%s_branch2a' % suffix))

    branch2a = AC_CONV(data=bn, num_filter=num_3x3_a, kernel=(3, 3),stride=(1, 1), pad=(1, 1), dilate=(1, 1), name=('res%s_branch2a' % suffix),
                        suffix=('%s_branch2a' % suffix))
    branch2b = BN_AC_CONV(data=branch2a, num_filter=num_3x3_b, kernel=(3, 3),stride=(1, 1), pad=dilate, dilate=dilate, name=('res%s_branch2b1' % suffix),
                        suffix=('%s_branch2b1' % suffix))
    summ = mx.symbol.ElementWiseSum(*[bn, branch2b], name=('res%s' % suffix))

    return summ

def get_conv_feature(data,is_train=True,workspace=2048,fix_gamma=True,use_global_stats=True,eps =1.001e-5):

    ## group 1
    conv1=Conv(data,name='conv1a',num_filter=64,pad=(1,1),kernel=(3,3),stride=(1,1))

    ## group 2
    res2a =  ResidualFactory_d(conv1,num_3x3_a=128,num_1x1_b=128,num_3x3_c=128,dilate=(1, 1),suffix='2a')

    res2b1 = ResidualFactory_c(res2a, num_3x3_a=128,  num_3x3_b=128, dilate=(1, 1), suffix='2b1')
    res2b2 =  ResidualFactory_c(res2b1,num_3x3_a=128, num_3x3_b=128, dilate=(2, 2), suffix='2b2')

    ##group 3
    res3a =  ResidualFactory_d(res2b2, num_3x3_a=256,  num_1x1_b=256, num_3x3_c=256,
                               dilate= (5, 5),suffix='3a')
    res3b1 = ResidualFactory_c(res3a, num_3x3_a=256,  num_3x3_b=256, dilate= (9, 9), suffix='3b1')
    res3b2 = ResidualFactory_c(res3b1, num_3x3_a=256, num_3x3_b=256, dilate=(1, 1), suffix='3b2')

    ##group 4
    res4a = ResidualFactory_d(res3b2, num_3x3_a=512,  num_1x1_b=512, dilate=(2, 2), num_3x3_c=512, suffix='4a')
    res4b1 = ResidualFactory_c(res4a, num_3x3_a=512,  num_3x3_b=512, dilate=(5, 5), suffix='4b1')
    res4b2 = ResidualFactory_c(res4b1, num_3x3_a=512, num_3x3_b=512, dilate=(9, 9), suffix='4b2')
    res4b3 = ResidualFactory_c(res4b2, num_3x3_a=512, num_3x3_b=512, dilate=(1, 1), suffix='4b3')
    res4b4 = ResidualFactory_c(res4b3, num_3x3_a=512, num_3x3_b=512, dilate=(2, 2), suffix='4b4')
    res4b5 = ResidualFactory_c(res4b4, num_3x3_a=512, num_3x3_b=512, dilate=(5, 5), suffix='4b5')

    ##group 5
    res5a = ResidualFactory_d2(res4b5, num_3x3_a=512, num_1x1_b=1024, num_3x3_c=1024, suffix='5a')
    res5b1 = ResidualFactory_c(res5a, num_3x3_a=512,  num_3x3_b=1024, dilate=(9, 9), suffix='5b1')
    res5b2 = ResidualFactory_c(res5b1,num_3x3_a=512,  num_3x3_b=1024, dilate=(2, 2), suffix='5b2')

    ##group 6
    res6a = ResidualFactory_a(res5b2,num_1x1_a=512, num_3x3_b=1024, num_1x1_c=2048,
                              num_1x1_d=2048, suffix="6a")
    res6a = mx.sym.Dropout(res6a, name="dropout_res6a", p=0.5)
    res7a = ResidualFactory_a(res6a,num_1x1_a=1024, num_3x3_b=2048, num_1x1_c=4096,
                              num_1x1_d=4096, suffix="7a")
    res7a = mx.sym.Dropout(res7a, name="dropout_res7a", p=0.5)
    relu7 = BN_AC(res7a,suffix='7')

    return relu7