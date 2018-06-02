#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: resnet38_v2_base.py
@time: 2017/9/28 13:58
"""

import mxnet as mx

def get_share_feature(data,origin_data,is_train=True,workspace=2048,fix_gamma=True,use_global_stats=True,eps =1.001e-5):

    conv1a_weight = mx.symbol.Variable(name="conv1a_weight")
    conv1a = mx.symbol.Convolution(name='conv1a', data=data,weight=conv1a_weight,num_filter=64, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                   no_bias=True, workspace=workspace)


    ##############################################################res2a #############

    # res2a for downsampling
    bn2a_branch2a_beta = mx.symbol.Variable(name="bn2a_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn2a_branch2a_gamma = mx.symbol.Variable(name="bn2a_branch2a_gamma")
    bn2a_branch2a_moving_var = mx.symbol.Variable(name="bn2a_branch2a_moving_var")
    bn2a_branch2a_moving_mean = mx.symbol.Variable(name="bn2a_branch2a_moving_mean")
    res2a_branch2a_weight = mx.symbol.Variable(name="res2a_branch2a_weight")
    res2a_branch1_weight = mx.symbol.Variable(name="res2a_branch1_weight")

    bn2a_branch2b1_beta = mx.symbol.Variable(name="bn2a_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn2a_branch2b1_gamma = mx.symbol.Variable(name="bn2a_branch2b1_gamma")
    bn2a_branch2b1_moving_var = mx.symbol.Variable(name="bn2a_branch2b1_moving_var")
    bn2a_branch2b1_moving_mean = mx.symbol.Variable(name="bn2a_branch2b1_moving_mean")
    res2a_branch2b1_weight = mx.symbol.Variable(name="res2a_branch2b1_weight")

    bn2a_branch2a = mx.symbol.BatchNorm(name="bn2a_branch2a", data=conv1a, gamma=bn2a_branch2a_gamma,
                                        beta=bn2a_branch2a_beta,
                                        moving_var = bn2a_branch2a_moving_var,
                                        moving_mean= bn2a_branch2a_moving_mean,
                                        fix_gamma=fix_gamma, use_global_stats=use_global_stats, eps=eps)
    res2a_branch2a_relu = mx.symbol.Activation(name="res2a_branch2a_relu", data=bn2a_branch2a, act_type='relu')

    res2a_branch2a = mx.symbol.Convolution(name="res2a_branch2a", data=res2a_branch2a_relu, dilate=(1, 1),
                                           kernel=(3, 3),weight = res2a_branch2a_weight,
                                           no_bias=True, num_filter=128, num_group=1, pad=(1, 1), stride=(2, 2),
                                           workspace=workspace)

    res2a_branch1 = mx.symbol.Convolution(name="res2a_branch1", data=res2a_branch2a_relu, dilate=(1, 1), kernel=(1, 1),
                                          no_bias=True,weight = res2a_branch1_weight,
                                          num_filter=128, num_group=1, pad=(0, 0), stride=(2, 2),
                                          workspace=workspace)

    bn2a_branch2b1 = mx.symbol.BatchNorm(name="bn2a_branch2b1",
                                         gamma=bn2a_branch2b1_gamma,
                                         beta=bn2a_branch2b1_beta,
                                         moving_var=bn2a_branch2b1_moving_var,
                                         moving_mean=bn2a_branch2b1_moving_mean,
                                         data=res2a_branch2a,
                                         fix_gamma=fix_gamma, use_global_stats=use_global_stats
                                         , eps=eps)

    res2a_branch2b1_relu = mx.symbol.Activation(name="res2a_branch2b1_relu", data=bn2a_branch2b1, act_type='relu')

    res2a_branch2b1 = mx.symbol.Convolution(name="res2a_branch2b1", data=res2a_branch2b1_relu, dilate=(1, 1),
                                            kernel=(3, 3),weight=res2a_branch2b1_weight,
                                            no_bias=True, num_filter=128, num_group=1, pad=(1, 1), stride=(1, 1),
                                            workspace=workspace)

    res2a_out = res2a_branch2b1 + res2a_branch1

    # print "level 2 down"
    # print res2a_out.infer_shape()[1]
    # res2b1
    # res2a for downsampling
    bn2b1_branch2a_beta = mx.symbol.Variable(name="bn2b1_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn2b1_branch2a_gamma = mx.symbol.Variable(name="bn2b1_branch2a_gamma")
    bn2b1_branch2a_moving_var = mx.symbol.Variable(name="bn2b1_branch2a_moving_var")
    bn2b1_branch2a_moving_mean = mx.symbol.Variable(name="bn2b1_branch2a_moving_mean")
    res2b1_branch2a_weight = mx.symbol.Variable(name="res2b1_branch2a_weight")

    bn2b1_branch2b1_beta = mx.symbol.Variable(name="bn2b1_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn2b1_branch2b1_gamma = mx.symbol.Variable(name="bn2b1_branch2b1_gamma")
    bn2b1_branch2b1_moving_var = mx.symbol.Variable(name="bn2b1_branch2b1_moving_var")
    bn2b1_branch2b1_moving_mean = mx.symbol.Variable(name="bn2b1_branch2b1_moving_mean")
    res2b1_branch2b1_weight = mx.symbol.Variable(name="res2b1_branch2b1_weight")

    bn2b1_branch2a = mx.symbol.BatchNorm(name="bn2b1_branch2a", data=res2a_out, gamma=bn2b1_branch2a_gamma,
                                         beta=bn2b1_branch2a_beta,moving_mean=bn2b1_branch2a_moving_mean,
                                         moving_var=bn2b1_branch2a_moving_var,
                                         fix_gamma=fix_gamma, use_global_stats=use_global_stats
                                         , eps=eps)

    res2b1_branch2a_relu = mx.symbol.Activation(name="res2b1_branch2a_relu", data=bn2b1_branch2a, act_type='relu')

    res2b1_branch2a = mx.symbol.Convolution(name="res2b1_branch2a", data=res2b1_branch2a_relu, dilate=(1, 1),
                                            kernel=(3, 3), no_bias=True,weight = res2b1_branch2a_weight,
                                            num_filter=128, num_group=1, pad=(1, 1),
                                            stride=(1, 1), workspace=workspace)

    bn2b1_branch2b1 = mx.symbol.BatchNorm(name="bn2b1_branch2b1",data=res2b1_branch2a,
                                          gamma = bn2b1_branch2b1_gamma,
                                          beta =  bn2b1_branch2b1_beta,
                                          moving_mean=  bn2b1_branch2b1_moving_mean,
                                          moving_var=   bn2b1_branch2b1_moving_var,
                                          fix_gamma=fix_gamma,
                                          use_global_stats=use_global_stats, eps=eps)

    res2b1_branch2b1_relu = mx.symbol.Activation(name="res2b1_branch2b1_relu", data=bn2b1_branch2b1, act_type='relu')

    res2b1_branch2b1 = mx.symbol.Convolution(name="res2b1_branch2b1", data=res2b1_branch2b1_relu, dilate=(1, 1),
                                             kernel=(3, 3), no_bias=True,weight = res2b1_branch2b1_weight,
                                             num_filter=128, num_group=1, pad=(1, 1),
                                             stride=(1, 1), workspace=workspace)

    res2a_plus2 = res2a_out + res2b1_branch2b1

    bn2b2_branch2a_beta = mx.symbol.Variable(name="bn2b2_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn2b2_branch2a_gamma = mx.symbol.Variable(name="bn2b2_branch2a_gamma")
    bn2b2_branch2a_moving_var = mx.symbol.Variable(name="bn2b2_branch2a_moving_var")
    bn2b2_branch2a_moving_mean = mx.symbol.Variable(name="bn2b2_branch2a_moving_mean")
    res2b2_branch2a_weight = mx.symbol.Variable(name="res2b2_branch2a_weight")

    bn2b2_branch2b1_beta = mx.symbol.Variable(name="bn2b2_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn2b2_branch2b1_gamma = mx.symbol.Variable(name="bn2b2_branch2b1_gamma")
    bn2b2_branch2b1_moving_var = mx.symbol.Variable(name="bn2b2_branch2b1_moving_var")
    bn2b2_branch2b1_moving_mean = mx.symbol.Variable(name="bn2b2_branch2b1_moving_mean")
    res2b2_branch2b1_weight = mx.symbol.Variable(name="res2b2_branch2b1_weight")

    bn2b2_branch2a = mx.symbol.BatchNorm(name="bn2b2_branch2a", data=res2a_plus2,
                                         gamma=bn2b2_branch2a_gamma,beta=bn2b2_branch2a_beta,
                                         moving_mean = bn2b2_branch2a_moving_mean,moving_var = bn2b2_branch2a_moving_var,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res2b2_branch2a_relu = mx.symbol.Activation(name="res2b2_branch2a_relu", data=bn2b2_branch2a, act_type='relu')

    res2b2_branch2a = mx.symbol.Convolution(name="res2b2_branch2a", data=res2b2_branch2a_relu, dilate=(1, 1),
                                            kernel=(3, 3), no_bias=True,weight = res2b2_branch2a_weight,
                                            num_filter=128, num_group=1, pad=(1, 1),
                                            stride=(1, 1), workspace=workspace)

    bn2b2_branch2b1 = mx.symbol.BatchNorm(name="bn2b2_branch2b1", data=res2b2_branch2a,
                                          gamma=bn2b2_branch2b1_gamma,beta=bn2b2_branch2b1_beta,
                                          moving_mean=bn2b2_branch2b1_moving_mean,moving_var= bn2b2_branch2b1_moving_var,
                                          fix_gamma=fix_gamma, use_global_stats=use_global_stats,
                                          eps=eps)

    res2b2_branch2b1_relu = mx.symbol.Activation(name="res2b2_branch2b1_relu", data=bn2b2_branch2b1, act_type='relu')

    res2b2_branch2b1 = mx.symbol.Convolution(name="res2b2_branch2b1", data=res2b2_branch2b1_relu, dilate=(1, 1),
                                             kernel=(3, 3), no_bias=True,weight = res2b2_branch2b1_weight,
                                             num_filter=128, num_group=1, pad=(1, 1),
                                             stride=(1, 1), workspace=workspace)

    res2a_plus3 = res2a_plus2 + res2b2_branch2b1

    # print "level 2a"
    # print res2a_plus2.infer_shape()[1]

    # res3a for downsampling
    bn3a_branch2a_beta = mx.symbol.Variable(name="bn3a_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn3a_branch2a_gamma = mx.symbol.Variable(name="bn3a_branch2a_gamma")
    bn3a_branch2a_moving_var = mx.symbol.Variable(name="bn3a_branch2a_moving_var")
    bn3a_branch2a_moving_mean = mx.symbol.Variable(name="bn3a_branch2a_moving_mean")
    res3a_branch1_weight = mx.symbol.Variable(name="res3a_branch1_weight")

    res3a_branch2a_weight = mx.symbol.Variable(name="res3a_branch2a_weight")

    bn3a_branch2b1_beta = mx.symbol.Variable(name="bn3a_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn3a_branch2b1_gamma = mx.symbol.Variable(name="bn3a_branch2b1_gamma")
    bn3a_branch2b1_moving_var = mx.symbol.Variable(name="bn3a_branch2b1_moving_var")
    bn3a_branch2b1_moving_mean = mx.symbol.Variable(name="bn3a_branch2b1_moving_mean")
    res3a_branch2b1_weight = mx.symbol.Variable(name="res3a_branch2b1_weight")

    bn3a_branch2a = mx.symbol.BatchNorm(name="bn3a_branch2a", data=res2a_plus3,
                                        gamma=bn3a_branch2a_gamma,beta=bn3a_branch2a_beta,
                                        moving_mean =bn3a_branch2a_moving_mean,moving_var =bn3a_branch2a_moving_var,
                                        fix_gamma=fix_gamma, use_global_stats=use_global_stats, eps=eps)

    res3a_branch2a_relu = mx.symbol.Activation(name="res3a_branch2a_relu", data=bn3a_branch2a, act_type='relu')

    res3a_branch1 = mx.symbol.Convolution(name="res3a_branch1", data=res3a_branch2a_relu, dilate=(1, 1), kernel=(1, 1),
                                          no_bias=True, num_filter=256,weight =res3a_branch1_weight,
                                          num_group=1, pad=(0, 0), stride=(2, 2), workspace=workspace)

    res3a_branch2a = mx.symbol.Convolution(name="res3a_branch2a", data=res3a_branch2a_relu, dilate=(1, 1),
                                           kernel=(3, 3), no_bias=True, num_filter=256,weight= res3a_branch2a_weight,
                                           num_group=1, pad=(1, 1), stride=(2, 2), workspace=workspace)

    bn3a_branch2b1 = mx.symbol.BatchNorm(name="bn3a_branch2b1", data=res3a_branch2a,
                                         gamma=bn3a_branch2b1_gamma,
                                         beta=bn3a_branch2b1_beta,
                                         moving_mean=bn3a_branch2b1_moving_mean,
                                         moving_var = bn3a_branch2b1_moving_var,
                                         fix_gamma=fix_gamma, use_global_stats=use_global_stats, eps=eps)
    res3a_branch2b1_relu = mx.symbol.Activation(name="res3a_branch2b1_relu", data=bn3a_branch2b1, act_type='relu')
    res3a_branch2b1 = mx.symbol.Convolution(name="res3a_branch2b1", data=res3a_branch2b1_relu, dilate=(1, 1),
                                            kernel=(3, 3), no_bias=True, num_filter=256,weight = res3a_branch2b1_weight,
                                            num_group=1, pad=(1, 1), stride=(1, 1), workspace=workspace)

    res3a_out = res3a_branch1 + res3a_branch2b1

    # print "level 3a"
    # print res3a_out.infer_shape()[1]

    ##res3a b1
    bn3b1_branch2a_beta = mx.symbol.Variable(name="bn3b1_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn3b1_branch2a_gamma = mx.symbol.Variable(name="bn3b1_branch2a_gamma")
    bn3b1_branch2a_moving_var = mx.symbol.Variable(name="bn3b1_branch2a_moving_var")
    bn3b1_branch2a_moving_mean = mx.symbol.Variable(name="bn3b1_branch2a_moving_mean")
    res3b1_branch2a_weight = mx.symbol.Variable(name="res3b1_branch2a_weight")

    bn3b1_branch2b1_beta = mx.symbol.Variable(name="bn3b1_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn3b1_branch2b1_gamma = mx.symbol.Variable(name="bn3b1_branch2b1_gamma")
    bn3b1_branch2b1_moving_var = mx.symbol.Variable(name="bn3b1_branch2b1_moving_var")
    bn3b1_branch2b1_moving_mean = mx.symbol.Variable(name="bn3b1_branch2b1_moving_mean")
    res3b1_branch2b1_weight = mx.symbol.Variable(name="res3b1_branch2b1_weight")

    bn3b1_branch2a = mx.symbol.BatchNorm(name="bn3b1_branch2a", data=res3a_out,
                                         gamma =bn3b1_branch2a_gamma,
                                         beta=bn3b1_branch2a_beta,
                                         moving_mean=bn3b1_branch2a_moving_mean,
                                         moving_var=bn3b1_branch2a_moving_var,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res3b1_branch2a_relu = mx.symbol.Activation(name="res3b1_branch2a_relu", data=bn3b1_branch2a, act_type='relu')

    res3b1_branch2a = mx.symbol.Convolution(name="res3b1_branch2a", data=res3b1_branch2a_relu, dilate=(1, 1),
                                            kernel=(3, 3), no_bias=True, num_filter=256,weight=res3b1_branch2a_weight,
                                            num_group=1, pad=(1, 1), stride=(1, 1), workspace=workspace)


    bn3b1_branch2b1 = mx.symbol.BatchNorm(name="bn3b1_branch2b1", data=res3b1_branch2a,
                                          gamma =bn3b1_branch2b1_gamma,
                                          beta=bn3b1_branch2b1_beta,
                                          moving_mean=bn3b1_branch2b1_moving_mean,
                                          moving_var =bn3b1_branch2b1_moving_var,
                                          fix_gamma=fix_gamma,
                                          use_global_stats=use_global_stats, eps=eps)

    res3b1_branch2b1_relu = mx.symbol.Activation(name="res3b1_branch2b1_relu", data=bn3b1_branch2b1, act_type='relu')

    res3b1_branch2b1 = mx.symbol.Convolution(name="res3b1_branch2b1", data=res3b1_branch2b1_relu, dilate=(1, 1),
                                             kernel=(3, 3), no_bias=True, num_filter=256,weight = res3b1_branch2b1_weight,
                                             num_group=1, pad=(1, 1), stride=(1, 1), workspace=workspace)

    res3b1_out = res3a_out + res3b1_branch2b1

    # print "level 3b1"
    # print res3b1_out.infer_shape()[1]

    ##res3a b2
    ##res3a b1
    bn3b2_branch2a_beta = mx.symbol.Variable(name="bn3b2_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn3b2_branch2a_gamma = mx.symbol.Variable(name="bn3b2_branch2a_gamma")
    bn3b2_branch2a_moving_var = mx.symbol.Variable(name="bn3b2_branch2a_moving_var")
    bn3b2_branch2a_moving_mean = mx.symbol.Variable(name="bn3b2_branch2a_moving_mean")
    res3b2_branch2a_weight = mx.symbol.Variable(name="res3b2_branch2a_weight")

    bn3b2_branch2b1_beta = mx.symbol.Variable(name="bn3b2_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn3b2_branch2b1_gamma = mx.symbol.Variable(name="bn3b2_branch2b1_gamma")
    bn3b2_branch2b1_moving_var = mx.symbol.Variable(name="bn3b2_branch2b1_moving_var")
    bn3b2_branch2b1_moving_mean = mx.symbol.Variable(name="bn3b2_branch2b1_moving_mean")
    res3b2_branch2b1_weight = mx.symbol.Variable(name="res3b2_branch2b1_weight")

    bn3b2_branch2a = mx.symbol.BatchNorm(name="bn3b2_branch2a", data=res3b1_out,
                                         gamma = bn3b2_branch2a_gamma,
                                         beta=bn3b2_branch2a_beta,
                                         moving_mean=bn3b2_branch2a_moving_mean,
                                         moving_var = bn3b2_branch2a_moving_var,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res3b2_branch2a_relu = mx.symbol.Activation(data=bn3b2_branch2a, name="res3b2_branch2a_relu",
                                                act_type='relu')
    res3b2_branch2a = mx.symbol.Convolution(name="res3b2_branch2a", data=res3b2_branch2a_relu, dilate=(1, 1),
                                            kernel=(3, 3), no_bias=True, num_filter=256,weight=res3b2_branch2a_weight,
                                            num_group=1, pad=(1, 1), stride=(1, 1), workspace=workspace)

    bn3b2_branch2b1 = mx.symbol.BatchNorm(name="bn3b2_branch2b1", data=res3b2_branch2a,
                                          gamma =bn3b2_branch2b1_gamma,
                                          beta=bn3b2_branch2b1_beta,
                                          moving_mean=bn3b2_branch2b1_moving_mean,
                                          moving_var=bn3b2_branch2b1_moving_var,
                                          fix_gamma=fix_gamma,
                                          use_global_stats=use_global_stats, eps=eps)

    res3b2_branch2b1_relu = mx.symbol.Activation(name="res3b2_branch2b1_relu", data=bn3b2_branch2b1, act_type='relu')

    res3b2_branch2b1 = mx.symbol.Convolution(name="res3b2_branch2b1", data=res3b2_branch2b1_relu, dilate=(1, 1),
                                             kernel=(3, 3), no_bias=True, num_filter=256,weight=res3b2_branch2b1_weight,
                                             num_group=1, pad=(1, 1), stride=(1, 1), workspace=workspace)

    res3b2_out = res3b1_out + res3b2_branch2b1

    # print "level 3b2"
    # print res3b2_out.infer_shape()[1]

    # res4a for downsampling
    bn4a_branch2a_beta = mx.symbol.Variable(name="bn4a_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn4a_branch2a_gamma = mx.symbol.Variable(name="bn4a_branch2a_gamma")
    bn4a_branch2a_moving_var = mx.symbol.Variable(name="bn4a_branch2a_moving_var")
    bn4a_branch2a_moving_mean = mx.symbol.Variable(name="bn4a_branch2a_moving_mean")
    res4a_branch1_weight = mx.symbol.Variable(name="res4a_branch1_weight")

    res4a_branch2a_weight = mx.symbol.Variable(name="res4a_branch2a_weight")
    bn4a_branch2b1_beta = mx.symbol.Variable(name="bn4a_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn4a_branch2b1_gamma = mx.symbol.Variable(name="bn4a_branch2b1_gamma")
    bn4a_branch2b1_moving_var = mx.symbol.Variable(name="bn4a_branch2b1_moving_var")
    bn4a_branch2b1_moving_mean = mx.symbol.Variable(name="bn4a_branch2b1_moving_mean")
    res4a_branch2b1_weight = mx.symbol.Variable(name="res4a_branch2b1_weight")

    bn4a_branch2a = mx.symbol.BatchNorm(name="bn4a_branch2a", data=res3b2_out,
                                        gamma = bn4a_branch2a_gamma,
                                        beta=bn4a_branch2a_beta,
                                        moving_mean=bn4a_branch2a_moving_mean,
                                        moving_var =bn4a_branch2a_moving_var,
                                        fix_gamma=fix_gamma,
                                        use_global_stats=use_global_stats, eps=eps)

    res4a_branch2a_relu = mx.symbol.Activation(name="res4a_branch2a_relu", data=bn4a_branch2a, act_type='relu')

    res4a_branch1 = mx.symbol.Convolution(name="res4a_branch1", data=res4a_branch2a_relu, dilate=(1, 1), kernel=(1, 1),
                                          no_bias=True,weight = res4a_branch1_weight,
                                          num_filter=512,
                                          num_group=1, pad=(0, 0), stride=(2, 2), workspace=workspace)

    res4a_branch2a = mx.symbol.Convolution(name="res4a_branch2a", data=res4a_branch2a_relu, dilate=(1, 1),
                                           kernel=(3, 3), no_bias=True,weight = res4a_branch2a_weight,
                                           num_filter=512,
                                           num_group=1, pad=(1, 1), stride=(2, 2), workspace=workspace)


    bn4a_branch2b1 = mx.symbol.BatchNorm(name="bn4a_branch2b1", data=res4a_branch2a,
                                         gamma = bn4a_branch2b1_gamma,
                                         beta=bn4a_branch2b1_beta,
                                         moving_mean=bn4a_branch2b1_moving_mean,
                                         moving_var=bn4a_branch2b1_moving_var,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res4a_branch2b1_relu = mx.symbol.Activation(name="res4a_branch2b1_relu", data=bn4a_branch2b1, act_type='relu')

    res4a_branch2b1 = mx.symbol.Convolution(name="res4a_branch2b1", data=res4a_branch2b1_relu, dilate=(1, 1),
                                            kernel=(3, 3), no_bias=True,weight =res4a_branch2b1_weight,
                                            num_filter=512,
                                            num_group=1, pad=(1, 1), stride=(1, 1), workspace=workspace)

    res4a_out = res4a_branch1 + res4a_branch2b1

    # print "level 4a"
    # print res4a_out.infer_shape()[1]

    ##res4a b1

    ##res3a b1
    bn4b1_branch2a_beta = mx.symbol.Variable(name="bn4b1_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn4b1_branch2a_gamma = mx.symbol.Variable(name="bn4b1_branch2a_gamma")
    bn4b1_branch2a_moving_var = mx.symbol.Variable(name="bn4b1_branch2a_moving_var")
    bn4b1_branch2a_moving_mean = mx.symbol.Variable(name="bn4b1_branch2a_moving_mean")
    res4b1_branch2a_weight = mx.symbol.Variable(name="res4b1_branch2a_weight")

    bn4b1_branch2b1_beta = mx.symbol.Variable(name="bn4b1_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn4b1_branch2b1_gamma = mx.symbol.Variable(name="bn4b1_branch2b1_gamma")
    bn4b1_branch2b1_moving_var = mx.symbol.Variable(name="bn4b1_branch2b1_moving_var")
    bn4b1_branch2b1_moving_mean = mx.symbol.Variable(name="bn4b1_branch2b1_moving_mean")
    res4b1_branch2b1_weight = mx.symbol.Variable(name="res4b1_branch2b1_weight")

    bn4b1_branch2a = mx.symbol.BatchNorm(name="bn4b1_branch2a", data=res4a_out,
                                         gamma = bn4b1_branch2a_gamma,
                                         beta=bn4b1_branch2a_beta,
                                         moving_mean=bn4b1_branch2a_moving_mean,
                                         moving_var =bn4b1_branch2a_moving_var,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res4b1_branch2a_relu = mx.symbol.Activation(name="res4b1_branch2a_relu", data=bn4b1_branch2a, act_type='relu')

    res4b1_branch2a = mx.symbol.Convolution(name="res4b1_branch2a", data=res4b1_branch2a_relu, dilate=(1, 1),
                                            kernel=(3, 3), no_bias=True,weight =res4b1_branch2a_weight,
                                            num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                            workspace=workspace)

    bn4b1_branch2b1 = mx.symbol.BatchNorm(name="bn4b1_branch2b1", data=res4b1_branch2a,
                                          gamma = bn4b1_branch2b1_gamma,
                                          beta= bn4b1_branch2b1_beta,
                                          moving_mean = bn4b1_branch2b1_moving_mean,
                                          moving_var= bn4b1_branch2b1_moving_var,
                                          fix_gamma=fix_gamma,
                                          use_global_stats=use_global_stats, eps=eps)

    res4b1_branch2b1_relu = mx.symbol.Activation(name="res4b1_branch2b1_relu", data=bn4b1_branch2b1, act_type='relu')

    res4b1_branch2b1 = mx.symbol.Convolution(name="res4b1_branch2b1", data=res4b1_branch2b1_relu, dilate=(1, 1),
                                             kernel=(3, 3), no_bias=True,weight =res4b1_branch2b1_weight,
                                             num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                             workspace=workspace)

    res4b1_out = res4a_out + res4b1_branch2b1

    # print "level 4b1"
    # print res4b1_out.infer_shape()[1]

    bn4b2_branch2a_beta = mx.symbol.Variable(name="bn4b2_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn4b2_branch2a_gamma = mx.symbol.Variable(name="bn4b2_branch2a_gamma")
    bn4b2_branch2a_moving_var = mx.symbol.Variable(name="bn4b2_branch2a_moving_var")
    bn4b2_branch2a_moving_mean = mx.symbol.Variable(name="bn4b2_branch2a_moving_mean")
    res4b2_branch2a_weight = mx.symbol.Variable(name="res4b2_branch2a_weight")

    bn4b2_branch2b1_beta = mx.symbol.Variable(name="bn4b2_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn4b2_branch2b1_gamma = mx.symbol.Variable(name="bn4b2_branch2b1_gamma")
    bn4b2_branch2b1_moving_var = mx.symbol.Variable(name="bn4b2_branch2b1_moving_var")
    bn4b2_branch2b1_moving_mean = mx.symbol.Variable(name="bn4b2_branch2b1_moving_mean")
    res4b2_branch2b1_weight = mx.symbol.Variable(name="res4b2_branch2b1_weight")

    bn4b2_branch2a = mx.symbol.BatchNorm(name="bn4b2_branch2a", data=res4b1_out,
                                         gamma =bn4b2_branch2a_gamma,
                                         beta=bn4b2_branch2a_beta,
                                         moving_mean = bn4b2_branch2a_moving_mean,
                                         moving_var = bn4b2_branch2a_moving_var,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res4b2_branch2a_relu = mx.symbol.Activation(name="res4b2_branch2a_relu", data=bn4b2_branch2a, act_type='relu')

    res4b2_branch2a = mx.symbol.Convolution(name="res4b2_branch2a", data=res4b2_branch2a_relu, dilate=(1, 1),
                                            kernel=(3, 3), no_bias=True,weight=res4b2_branch2a_weight,
                                            num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                            workspace=workspace)
    bn4b2_branch2b1 = mx.symbol.BatchNorm(name="bn4b2_branch2b1",
                                          data=res4b2_branch2a,
                                          gamma=bn4b2_branch2b1_gamma,
                                          beta=bn4b2_branch2b1_beta,
                                          moving_mean=bn4b2_branch2b1_moving_mean,
                                          moving_var=bn4b2_branch2b1_moving_var,
                                          fix_gamma=fix_gamma,
                                          use_global_stats=use_global_stats, eps=eps)

    res4b2_branch2b1_relu = mx.symbol.Activation(name="res4b2_branch2b1_relu", data=bn4b2_branch2b1, act_type='relu')

    res4b2_branch2b1 = mx.symbol.Convolution(name="res4b2_branch2b1", data=res4b2_branch2b1_relu, dilate=(1, 1),
                                             kernel=(3, 3), no_bias=True,weight = res4b2_branch2b1_weight,
                                             num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                             workspace=workspace)

    res4b2_out = res4b1_out + res4b2_branch2b1

    # print "level 4b2"
    # print res4b2_out.infer_shape()[1]

    ##res4a b3
    bn4b3_branch2a_beta = mx.symbol.Variable(name="bn4b3_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn4b3_branch2a_gamma = mx.symbol.Variable(name="bn4b3_branch2a_gamma")
    bn4b3_branch2a_moving_var = mx.symbol.Variable(name="bn4b3_branch2a_moving_var")
    bn4b3_branch2a_moving_mean = mx.symbol.Variable(name="bn4b3_branch2a_moving_mean")
    res4b3_branch2a_weight = mx.symbol.Variable(name="res4b3_branch2a_weight")

    bn4b3_branch2b1_beta = mx.symbol.Variable(name="bn4b3_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn4b3_branch2b1_gamma = mx.symbol.Variable(name="bn4b3_branch2b1_gamma")
    bn4b3_branch2b1_moving_var = mx.symbol.Variable(name="bn4b3_branch2b1_moving_var")
    bn4b3_branch2b1_moving_mean = mx.symbol.Variable(name="bn4b3_branch2b1_moving_mean")
    res4b3_branch2b1_weight = mx.symbol.Variable(name="res4b3_branch2b1_weight")

    bn4b3_branch2a = mx.symbol.BatchNorm(name="bn4b3_branch2a", data=res4b2_out,
                                         gamma=bn4b3_branch2a_gamma,
                                         beta=bn4b3_branch2a_beta,
                                         moving_mean=bn4b3_branch2a_moving_mean,
                                         moving_var=bn4b3_branch2a_moving_var,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res4b3_branch2a_relu = mx.symbol.Activation(name="res4b3_branch2a_relu", data=bn4b3_branch2a, act_type='relu')

    res4b3_branch2a = mx.symbol.Convolution(name="res4b3_branch2a", data=res4b3_branch2a_relu, dilate=(1, 1),
                                            kernel=(3, 3), no_bias=True,weight=res4b3_branch2a_weight,
                                            num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                            workspace=workspace)

    bn4b3_branch2b1 = mx.symbol.BatchNorm(name="bn4b3_branch2b1", data=res4b3_branch2a,
                                          gamma=bn4b3_branch2b1_gamma,
                                          beta=bn4b3_branch2b1_beta,
                                          moving_mean=bn4b3_branch2b1_moving_mean,
                                          moving_var=bn4b3_branch2b1_moving_var,
                                          fix_gamma=fix_gamma,
                                          use_global_stats=use_global_stats, eps=eps)

    res4b3_branch2b1_relu = mx.symbol.Activation(name="res4b3_branch2b1_relu", data=bn4b3_branch2b1, act_type='relu')

    res4b3_branch2b1 = mx.symbol.Convolution(name="res4b3_branch2b1", data=res4b3_branch2b1_relu, dilate=(1, 1),
                                             kernel=(3, 3), no_bias=True,weight = res4b3_branch2b1_weight,
                                             num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                             workspace=workspace)

    res4b3_out = res4b2_out + res4b3_branch2b1

    # print "level 4b3"
    # print res4b3_out.infer_shape()[1]

    ##res4a b4
    bn4b4_branch2a_beta = mx.symbol.Variable(name="bn4b4_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn4b4_branch2a_gamma = mx.symbol.Variable(name="bn4b4_branch2a_gamma")
    bn4b4_branch2a_moving_var = mx.symbol.Variable(name="bn4b4_branch2a_moving_var")
    bn4b4_branch2a_moving_mean = mx.symbol.Variable(name="bn4b4_branch2a_moving_mean")
    res4b4_branch2a_weight = mx.symbol.Variable(name="res4b4_branch2a_weight")

    bn4b4_branch2b1_beta = mx.symbol.Variable(name="bn4b4_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn4b4_branch2b1_gamma = mx.symbol.Variable(name="bn4b4_branch2b1_gamma")
    bn4b4_branch2b1_moving_var = mx.symbol.Variable(name="bn4b4_branch2b1_moving_var")
    bn4b4_branch2b1_moving_mean = mx.symbol.Variable(name="bn4b4_branch2b1_moving_mean")
    res4b4_branch2b1_weight = mx.symbol.Variable(name="res4b4_branch2b1_weight")

    bn4b4_branch2a = mx.symbol.BatchNorm(name="bn4b4_branch2a", data=res4b3_out,
                                         gamma=bn4b4_branch2a_gamma,
                                         beta=bn4b4_branch2a_beta,
                                         moving_mean=bn4b4_branch2a_moving_mean,
                                         moving_var=bn4b4_branch2a_moving_var,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res4b4_branch2a_relu = mx.symbol.Activation(name="res4b4_branch2a_relu", data=bn4b4_branch2a, act_type='relu')

    res4b4_branch2a = mx.symbol.Convolution(name="res4b4_branch2a", data=res4b4_branch2a_relu, dilate=(1, 1),
                                            kernel=(3, 3), no_bias=True,weight = res4b4_branch2a_weight,
                                            num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                            workspace=workspace)

    bn4b4_branch2b1 = mx.symbol.BatchNorm(name="bn4b4_branch2b1", data=res4b4_branch2a,
                                          gamma=bn4b4_branch2b1_gamma,
                                          beta=bn4b4_branch2b1_beta,
                                          moving_mean=bn4b4_branch2b1_moving_mean,
                                          moving_var=bn4b4_branch2b1_moving_var,
                                          fix_gamma=fix_gamma,
                                          use_global_stats=use_global_stats, eps=eps)

    res4b4_branch2b1_relu = mx.symbol.Activation(name="res4b4_branch2b1_relu", data=bn4b4_branch2b1,
                                                 act_type='relu')

    res4b4_branch2b1 = mx.symbol.Convolution(name="res4b4_branch2b1", data=res4b4_branch2b1_relu, dilate=(1, 1),
                                             kernel=(3, 3), no_bias=True,weight=res4b4_branch2b1_weight,
                                             num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                             workspace=workspace)

    res4b4_out = res4b3_out + res4b4_branch2b1

    # print "level 4b4"
    # print res4b4_out.infer_shape()[1]

    ##res4a b5
    bn4b5_branch2a_beta = mx.symbol.Variable(name="bn4b5_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn4b5_branch2a_gamma = mx.symbol.Variable(name="bn4b5_branch2a_gamma")
    bn4b5_branch2a_moving_var = mx.symbol.Variable(name="bn4b5_branch2a_moving_var")
    bn4b5_branch2a_moving_mean = mx.symbol.Variable(name="bn4b5_branch2a_moving_mean")
    res4b5_branch2a_weight = mx.symbol.Variable(name="res4b5_branch2a_weight")

    bn4b5_branch2b1_beta = mx.symbol.Variable(name="bn4b5_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn4b5_branch2b1_gamma = mx.symbol.Variable(name="bn4b5_branch2b1_gamma")
    bn4b5_branch2b1_moving_var = mx.symbol.Variable(name="bn4b5_branch2b1_moving_var")
    bn4b5_branch2b1_moving_mean = mx.symbol.Variable(name="bn4b5_branch2b1_moving_mean")
    res4b5_branch2b1_weight = mx.symbol.Variable(name="res4b5_branch2b1_weight")

    bn4b5_branch2a = mx.symbol.BatchNorm(name="bn4b5_branch2a", data=res4b4_out,
                                         gamma=bn4b5_branch2a_gamma,
                                         beta=bn4b5_branch2a_beta,
                                         moving_mean=bn4b5_branch2a_moving_mean,
                                         moving_var=bn4b5_branch2a_moving_var,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)

    res4b5_branch2a_relu = mx.symbol.Activation(name="res4b5_branch2a_relu", data=bn4b5_branch2a, act_type='relu')
    res4b5_branch2a = mx.symbol.Convolution(name="res4b5_branch2a", data=res4b5_branch2a_relu, dilate=(1, 1),
                                            kernel=(3, 3), no_bias=True,weight =res4b5_branch2a_weight,
                                            num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                            workspace=workspace)

    bn4b5_branch2b1 = mx.symbol.BatchNorm(name="bn4b5_branch2b1", data=res4b5_branch2a,
                                          gamma=bn4b5_branch2b1_gamma,
                                          beta=bn4b5_branch2b1_beta,
                                          moving_mean=bn4b5_branch2b1_moving_mean,
                                          moving_var=bn4b5_branch2b1_moving_var,
                                          fix_gamma=fix_gamma,
                                          use_global_stats=use_global_stats, eps=eps)

    res4b5_branch2b1_relu = mx.symbol.Activation(name="res4b5_branch2b1_relu", data=bn4b5_branch2b1,
                                                 act_type='relu')

    res4b5_branch2b1 = mx.symbol.Convolution(name="res4b5_branch2b1", data=res4b5_branch2b1_relu, dilate=(1, 1),
                                             kernel=(3, 3), no_bias=True, weight =res4b5_branch2b1_weight,
                                             num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                             workspace=workspace)

    res4b5_out = res4b4_out + res4b5_branch2b1

    context_conv1a = mx.symbol.Convolution(name="context_conv1a", data=origin_data, weight=conv1a_weight, num_filter=64,
                                           pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1),
                                           no_bias=True, workspace=workspace)

    context_bn2a_branch2a = mx.symbol.BatchNorm(name="context_bn2a_branch2a", data=context_conv1a,
                                                gamma=bn2a_branch2a_gamma,
                                                beta=bn2a_branch2a_beta,
                                                moving_var=bn2a_branch2a_moving_var,
                                                moving_mean=bn2a_branch2a_moving_mean,
                                                fix_gamma=fix_gamma, use_global_stats=use_global_stats, eps=eps)
    context_res2a_branch2a_relu = mx.symbol.Activation(name="context_res2a_branch2a_relu", data=context_bn2a_branch2a,
                                                       act_type='relu')

    context_res2a_branch2a = mx.symbol.Convolution(name="context_res2a_branch2a", data=context_res2a_branch2a_relu,
                                                   dilate=(1, 1),
                                                   kernel=(3, 3), weight=res2a_branch2a_weight,
                                                   no_bias=True, num_filter=128, num_group=1, pad=(1, 1), stride=(2, 2),
                                                   workspace=workspace)

    context_res2a_branch1 = mx.symbol.Convolution(name="context_res2a_branch1", data=context_res2a_branch2a_relu,
                                                  dilate=(1, 1), kernel=(1, 1),
                                                  no_bias=True, weight=res2a_branch1_weight,
                                                  num_filter=128, num_group=1, pad=(0, 0), stride=(2, 2),
                                                  workspace=workspace)

    context_bn2a_branch2b1 = mx.symbol.BatchNorm(name="context_bn2a_branch2b1",
                                                 gamma=bn2a_branch2b1_gamma,
                                                 beta=bn2a_branch2b1_beta,
                                                 moving_var=bn2a_branch2b1_moving_var,
                                                 moving_mean=bn2a_branch2b1_moving_mean,
                                                 data=context_res2a_branch2a,
                                                 fix_gamma=fix_gamma, use_global_stats=use_global_stats
                                                 , eps=eps)

    context_res2a_branch2b1_relu = mx.symbol.Activation(name="context_res2a_branch2b1_relu", data=context_bn2a_branch2b1,
                                                        act_type='relu')

    context_res2a_branch2b1 = mx.symbol.Convolution(name="context_res2a_branch2b1", data=context_res2a_branch2b1_relu,
                                                    dilate=(1, 1),
                                                    kernel=(3, 3), weight=res2a_branch2b1_weight,
                                                    no_bias=True, num_filter=128, num_group=1, pad=(1, 1), stride=(1, 1),
                                                    workspace=workspace)

    context_res2a_out = context_res2a_branch2b1 + context_res2a_branch1

    context_bn2b1_branch2a = mx.symbol.BatchNorm(name="context_bn2b1_branch2a", data=context_res2a_out,
                                                 gamma=bn2b1_branch2a_gamma,
                                                 beta=bn2b1_branch2a_beta,
                                                 moving_mean=bn2b1_branch2a_moving_mean,
                                                 moving_var=bn2b1_branch2a_moving_var,
                                                 fix_gamma=fix_gamma, use_global_stats=use_global_stats
                                                 , eps=eps)

    context_res2b1_branch2a_relu = mx.symbol.Activation(name="context_res2b1_branch2a_relu", data=context_bn2b1_branch2a,
                                                        act_type='relu')

    context_res2b1_branch2a = mx.symbol.Convolution(name="context_res2b1_branch2a", data=context_res2b1_branch2a_relu,
                                                    dilate=(1, 1),
                                                    kernel=(3, 3), no_bias=True, weight=res2b1_branch2a_weight,
                                                    num_filter=128, num_group=1, pad=(1, 1),
                                                    stride=(1, 1), workspace=workspace)

    context_bn2b1_branch2b1 = mx.symbol.BatchNorm(name="context_bn2b1_branch2b1", data=context_res2b1_branch2a,
                                                  gamma=bn2b1_branch2b1_gamma,
                                                  beta=bn2b1_branch2b1_beta,
                                                  moving_mean=bn2b1_branch2b1_moving_mean,
                                                  moving_var=bn2b1_branch2b1_moving_var,
                                                  fix_gamma=fix_gamma,
                                                  use_global_stats=use_global_stats, eps=eps)

    context_res2b1_branch2b1_relu = mx.symbol.Activation(name="context_res2b1_branch2b1_relu", data=context_bn2b1_branch2b1,
                                                         act_type='relu')

    context_res2b1_branch2b1 = mx.symbol.Convolution(name="context_res2b1_branch2b1", data=context_res2b1_branch2b1_relu,
                                                     dilate=(1, 1),
                                                     kernel=(3, 3), no_bias=True, weight=res2b1_branch2b1_weight,
                                                     num_filter=128, num_group=1, pad=(1, 1),
                                                     stride=(1, 1), workspace=workspace)

    context_res2a_plus2 = context_res2a_out + context_res2b1_branch2b1

    context_bn2b2_branch2a = mx.symbol.BatchNorm(name="context_bn2b2_branch2a", data=context_res2a_plus2,
                                                 gamma=bn2b2_branch2a_gamma, beta=bn2b2_branch2a_beta,
                                                 moving_mean=bn2b2_branch2a_moving_mean,
                                                 moving_var=bn2b2_branch2a_moving_var,
                                                 fix_gamma=fix_gamma,
                                                 use_global_stats=use_global_stats, eps=eps)
    context_res2b2_branch2a_relu = mx.symbol.Activation(name="context_res2b2_branch2a_relu", data=context_bn2b2_branch2a,
                                                        act_type='relu')

    context_res2b2_branch2a = mx.symbol.Convolution(name="context_res2b2_branch2a", data=context_res2b2_branch2a_relu,
                                                    dilate=(1, 1),
                                                    kernel=(3, 3), no_bias=True, weight=res2b2_branch2a_weight,
                                                    num_filter=128, num_group=1, pad=(1, 1),
                                                    stride=(1, 1), workspace=workspace)

    context_bn2b2_branch2b1 = mx.symbol.BatchNorm(name="context_bn2b2_branch2b1", data=context_res2b2_branch2a,
                                                  gamma=bn2b2_branch2b1_gamma, beta=bn2b2_branch2b1_beta,
                                                  moving_mean=bn2b2_branch2b1_moving_mean,
                                                  moving_var=bn2b2_branch2b1_moving_var,
                                                  fix_gamma=fix_gamma, use_global_stats=use_global_stats,
                                                  eps=eps)

    context_res2b2_branch2b1_relu = mx.symbol.Activation(name="context_res2b2_branch2b1_relu", data=context_bn2b2_branch2b1,
                                                         act_type='relu')

    context_res2b2_branch2b1 = mx.symbol.Convolution(name="context_res2b2_branch2b1", data=context_res2b2_branch2b1_relu,
                                                     dilate=(1, 1),
                                                     kernel=(3, 3), no_bias=True, weight=res2b2_branch2b1_weight,
                                                     num_filter=128, num_group=1, pad=(1, 1),
                                                     stride=(1, 1), workspace=workspace)

    context_res2a_plus3 = context_res2a_plus2 + context_res2b2_branch2b1

    context_bn3a_branch2a = mx.symbol.BatchNorm(name="context_bn3a_branch2a", data=context_res2a_plus3,
                                                gamma=bn3a_branch2a_gamma, beta=bn3a_branch2a_beta,
                                                moving_mean=bn3a_branch2a_moving_mean,
                                                moving_var=bn3a_branch2a_moving_var,
                                                fix_gamma=fix_gamma, use_global_stats=use_global_stats, eps=eps)

    context_res3a_branch2a_relu = mx.symbol.Activation(name="context_res3a_branch2a_relu", data=context_bn3a_branch2a,
                                                       act_type='relu')

    context_res3a_branch1 = mx.symbol.Convolution(name="context_res3a_branch1", data=context_res3a_branch2a_relu,
                                                  dilate=(1, 1), kernel=(1, 1),
                                                  no_bias=True, num_filter=256, weight=res3a_branch1_weight,
                                                  num_group=1, pad=(0, 0), stride=(2, 2), workspace=workspace)

    context_res3a_branch2a = mx.symbol.Convolution(name="context_res3a_branch2a", data=context_res3a_branch2a_relu,
                                                   dilate=(1, 1),
                                                   kernel=(3, 3), no_bias=True, num_filter=256,
                                                   weight=res3a_branch2a_weight,
                                                   num_group=1, pad=(1, 1), stride=(2, 2), workspace=workspace)

    context_bn3a_branch2b1 = mx.symbol.BatchNorm(name="context_bn3a_branch2b1", data=context_res3a_branch2a,
                                                 gamma=bn3a_branch2b1_gamma,
                                                 beta=bn3a_branch2b1_beta,
                                                 moving_mean=bn3a_branch2b1_moving_mean,
                                                 moving_var=bn3a_branch2b1_moving_var,
                                                 fix_gamma=fix_gamma, use_global_stats=use_global_stats, eps=eps)
    context_res3a_branch2b1_relu = mx.symbol.Activation(name="context_res3a_branch2b1_relu", data=context_bn3a_branch2b1,
                                                        act_type='relu')
    context_res3a_branch2b1 = mx.symbol.Convolution(name="context_res3a_branch2b1", data=context_res3a_branch2b1_relu,
                                                    dilate=(1, 1),
                                                    kernel=(3, 3), no_bias=True, num_filter=256,
                                                    weight=res3a_branch2b1_weight,
                                                    num_group=1, pad=(1, 1), stride=(1, 1), workspace=workspace)

    context_res3a_out = context_res3a_branch1 + context_res3a_branch2b1

    # print "level 3a"
    # print context_res3a_out.infer_shape()[1]

    context_bn3b1_branch2a = mx.symbol.BatchNorm(name="context_bn3b1_branch2a", data=context_res3a_out,
                                                 gamma=bn3b1_branch2a_gamma,
                                                 beta=bn3b1_branch2a_beta,
                                                 moving_mean=bn3b1_branch2a_moving_mean,
                                                 moving_var=bn3b1_branch2a_moving_var,
                                                 fix_gamma=fix_gamma,
                                                 use_global_stats=use_global_stats, eps=eps)
    context_res3b1_branch2a_relu = mx.symbol.Activation(name="context_res3b1_branch2a_relu", data=context_bn3b1_branch2a,
                                                        act_type='relu')

    context_res3b1_branch2a = mx.symbol.Convolution(name="context_res3b1_branch2a", data=context_res3b1_branch2a_relu,
                                                    dilate=(1, 1),
                                                    kernel=(3, 3), no_bias=True, num_filter=256,
                                                    weight=res3b1_branch2a_weight,
                                                    num_group=1, pad=(1, 1), stride=(1, 1), workspace=workspace)

    context_bn3b1_branch2b1 = mx.symbol.BatchNorm(name="context_bn3b1_branch2b1", data=context_res3b1_branch2a,
                                                  gamma=bn3b1_branch2b1_gamma,
                                                  beta=bn3b1_branch2b1_beta,
                                                  moving_mean=bn3b1_branch2b1_moving_mean,
                                                  moving_var=bn3b1_branch2b1_moving_var,
                                                  fix_gamma=fix_gamma,
                                                  use_global_stats=use_global_stats, eps=eps)

    context_res3b1_branch2b1_relu = mx.symbol.Activation(name="context_res3b1_branch2b1_relu", data=context_bn3b1_branch2b1,
                                                         act_type='relu')

    context_res3b1_branch2b1 = mx.symbol.Convolution(name="context_res3b1_branch2b1", data=context_res3b1_branch2b1_relu,
                                                     dilate=(1, 1),
                                                     kernel=(3, 3), no_bias=True, num_filter=256,
                                                     weight=res3b1_branch2b1_weight,
                                                     num_group=1, pad=(1, 1), stride=(1, 1), workspace=workspace)

    context_res3b1_out = context_res3a_out + context_res3b1_branch2b1

    # print "level 3b1"
    # print context_res3b1_out.infer_shape()[1]


    context_bn3b2_branch2a = mx.symbol.BatchNorm(name="context_bn3b2_branch2a", data=context_res3b1_out,
                                                 gamma=bn3b2_branch2a_gamma,
                                                 beta=bn3b2_branch2a_beta,
                                                 moving_mean=bn3b2_branch2a_moving_mean,
                                                 moving_var=bn3b2_branch2a_moving_var,
                                                 fix_gamma=fix_gamma,
                                                 use_global_stats=use_global_stats, eps=eps)
    context_res3b2_branch2a_relu = mx.symbol.Activation(data=context_bn3b2_branch2a, name="context_res3b2_branch2a_relu",
                                                        act_type='relu')
    context_res3b2_branch2a = mx.symbol.Convolution(name="context_res3b2_branch2a", data=context_res3b2_branch2a_relu,
                                                    dilate=(1, 1),
                                                    kernel=(3, 3), no_bias=True, num_filter=256,
                                                    weight=res3b2_branch2a_weight,
                                                    num_group=1, pad=(1, 1), stride=(1, 1), workspace=workspace)

    context_bn3b2_branch2b1 = mx.symbol.BatchNorm(name="context_bn3b2_branch2b1", data=context_res3b2_branch2a,
                                                  gamma=bn3b2_branch2b1_gamma,
                                                  beta=bn3b2_branch2b1_beta,
                                                  moving_mean=bn3b2_branch2b1_moving_mean,
                                                  moving_var=bn3b2_branch2b1_moving_var,
                                                  fix_gamma=fix_gamma,
                                                  use_global_stats=use_global_stats, eps=eps)

    context_res3b2_branch2b1_relu = mx.symbol.Activation(name="context_res3b2_branch2b1_relu", data=context_bn3b2_branch2b1,
                                                         act_type='relu')

    context_res3b2_branch2b1 = mx.symbol.Convolution(name="context_res3b2_branch2b1", data=context_res3b2_branch2b1_relu,
                                                     dilate=(1, 1),
                                                     kernel=(3, 3), no_bias=True, num_filter=256,
                                                     weight=res3b2_branch2b1_weight,
                                                     num_group=1, pad=(1, 1), stride=(1, 1), workspace=workspace)

    context_res3b2_out = context_res3b1_out + context_res3b2_branch2b1

    # print "level 3b2"
    # print context_res3b2_out.infer_shape()[1]

    context_bn4a_branch2a = mx.symbol.BatchNorm(name="context_bn4a_branch2a", data=context_res3b2_out,
                                                gamma=bn4a_branch2a_gamma,
                                                beta=bn4a_branch2a_beta,
                                                moving_mean=bn4a_branch2a_moving_mean,
                                                moving_var=bn4a_branch2a_moving_var,
                                                fix_gamma=fix_gamma,
                                                use_global_stats=use_global_stats, eps=eps)

    context_res4a_branch2a_relu = mx.symbol.Activation(name="context_res4a_branch2a_relu", data=context_bn4a_branch2a,
                                                       act_type='relu')

    context_res4a_branch1 = mx.symbol.Convolution(name="context_res4a_branch1", data=context_res4a_branch2a_relu,
                                                  dilate=(1, 1), kernel=(1, 1),
                                                  no_bias=True, weight=res4a_branch1_weight,
                                                  num_filter=512,
                                                  num_group=1, pad=(0, 0), stride=(2, 2), workspace=workspace)

    context_res4a_branch2a = mx.symbol.Convolution(name="context_res4a_branch2a", data=context_res4a_branch2a_relu,
                                                   dilate=(1, 1),
                                                   kernel=(3, 3), no_bias=True, weight=res4a_branch2a_weight,
                                                   num_filter=512,
                                                   num_group=1, pad=(1, 1), stride=(2, 2), workspace=workspace)

    context_bn4a_branch2b1 = mx.symbol.BatchNorm(name="context_bn4a_branch2b1", data=context_res4a_branch2a,
                                                 gamma=bn4a_branch2b1_gamma,
                                                 beta=bn4a_branch2b1_beta,
                                                 moving_mean=bn4a_branch2b1_moving_mean,
                                                 moving_var=bn4a_branch2b1_moving_var,
                                                 fix_gamma=fix_gamma,
                                                 use_global_stats=use_global_stats, eps=eps)
    context_res4a_branch2b1_relu = mx.symbol.Activation(name="context_res4a_branch2b1_relu", data=context_bn4a_branch2b1,
                                                        act_type='relu')

    context_res4a_branch2b1 = mx.symbol.Convolution(name="context_res4a_branch2b1", data=context_res4a_branch2b1_relu,
                                                    dilate=(1, 1),
                                                    kernel=(3, 3), no_bias=True, weight=res4a_branch2b1_weight,
                                                    num_filter=512,
                                                    num_group=1, pad=(1, 1), stride=(1, 1), workspace=workspace)

    context_res4a_out = context_res4a_branch1 + context_res4a_branch2b1

    # print "level 4a"
    # print context_res4a_out.infer_shape()[1]

    ##context_res4a b1

    context_bn4b1_branch2a = mx.symbol.BatchNorm(name="context_bn4b1_branch2a", data=context_res4a_out,
                                                 gamma=bn4b1_branch2a_gamma,
                                                 beta=bn4b1_branch2a_beta,
                                                 moving_mean=bn4b1_branch2a_moving_mean,
                                                 moving_var=bn4b1_branch2a_moving_var,
                                                 fix_gamma=fix_gamma,
                                                 use_global_stats=use_global_stats, eps=eps)
    context_res4b1_branch2a_relu = mx.symbol.Activation(name="context_res4b1_branch2a_relu", data=context_bn4b1_branch2a,
                                                        act_type='relu')

    context_res4b1_branch2a = mx.symbol.Convolution(name="context_res4b1_branch2a", data=context_res4b1_branch2a_relu,
                                                    dilate=(1, 1),
                                                    kernel=(3, 3), no_bias=True, weight=res4b1_branch2a_weight,
                                                    num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                                    workspace=workspace)

    context_bn4b1_branch2b1 = mx.symbol.BatchNorm(name="context_bn4b1_branch2b1", data=context_res4b1_branch2a,
                                                  gamma=bn4b1_branch2b1_gamma,
                                                  beta=bn4b1_branch2b1_beta,
                                                  moving_mean=bn4b1_branch2b1_moving_mean,
                                                  moving_var=bn4b1_branch2b1_moving_var,
                                                  fix_gamma=fix_gamma,
                                                  use_global_stats=use_global_stats, eps=eps)

    context_res4b1_branch2b1_relu = mx.symbol.Activation(name="context_res4b1_branch2b1_relu", data=context_bn4b1_branch2b1,
                                                         act_type='relu')

    context_res4b1_branch2b1 = mx.symbol.Convolution(name="context_res4b1_branch2b1", data=context_res4b1_branch2b1_relu,
                                                     dilate=(1, 1),
                                                     kernel=(3, 3), no_bias=True, weight=res4b1_branch2b1_weight,
                                                     num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                                     workspace=workspace)

    context_res4b1_out = context_res4a_out + context_res4b1_branch2b1

    # print "level 4b1"
    # print context_res4b1_out.infer_shape()[1]

    context_bn4b2_branch2a = mx.symbol.BatchNorm(name="context_bn4b2_branch2a", data=context_res4b1_out,
                                                 gamma=bn4b2_branch2a_gamma,
                                                 beta=bn4b2_branch2a_beta,
                                                 moving_mean=bn4b2_branch2a_moving_mean,
                                                 moving_var=bn4b2_branch2a_moving_var,
                                                 fix_gamma=fix_gamma,
                                                 use_global_stats=use_global_stats, eps=eps)
    context_res4b2_branch2a_relu = mx.symbol.Activation(name="context_res4b2_branch2a_relu", data=context_bn4b2_branch2a,
                                                        act_type='relu')

    context_res4b2_branch2a = mx.symbol.Convolution(name="context_res4b2_branch2a", data=context_res4b2_branch2a_relu,
                                                    dilate=(1, 1),
                                                    kernel=(3, 3), no_bias=True, weight=res4b2_branch2a_weight,
                                                    num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                                    workspace=workspace)
    context_bn4b2_branch2b1 = mx.symbol.BatchNorm(name="context_bn4b2_branch2b1",
                                                  data=context_res4b2_branch2a,
                                                  gamma=bn4b2_branch2b1_gamma,
                                                  beta=bn4b2_branch2b1_beta,
                                                  moving_mean=bn4b2_branch2b1_moving_mean,
                                                  moving_var=bn4b2_branch2b1_moving_var,
                                                  fix_gamma=fix_gamma,
                                                  use_global_stats=use_global_stats, eps=eps)

    context_res4b2_branch2b1_relu = mx.symbol.Activation(name="context_res4b2_branch2b1_relu", data=context_bn4b2_branch2b1,
                                                         act_type='relu')

    context_res4b2_branch2b1 = mx.symbol.Convolution(name="context_res4b2_branch2b1", data=context_res4b2_branch2b1_relu,
                                                     dilate=(1, 1),
                                                     kernel=(3, 3), no_bias=True, weight=res4b2_branch2b1_weight,
                                                     num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                                     workspace=workspace)

    context_res4b2_out = context_res4b1_out + context_res4b2_branch2b1

    # print "level 4b2"
    # print context_res4b2_out.infer_shape()[1]


    context_bn4b3_branch2a = mx.symbol.BatchNorm(name="context_bn4b3_branch2a", data=context_res4b2_out,
                                                 gamma=bn4b3_branch2a_gamma,
                                                 beta=bn4b3_branch2a_beta,
                                                 moving_mean=bn4b3_branch2a_moving_mean,
                                                 moving_var=bn4b3_branch2a_moving_var,
                                                 fix_gamma=fix_gamma,
                                                 use_global_stats=use_global_stats, eps=eps)
    context_res4b3_branch2a_relu = mx.symbol.Activation(name="context_res4b3_branch2a_relu", data=context_bn4b3_branch2a,
                                                        act_type='relu')

    context_res4b3_branch2a = mx.symbol.Convolution(name="context_res4b3_branch2a", data=context_res4b3_branch2a_relu,
                                                    dilate=(1, 1),
                                                    kernel=(3, 3), no_bias=True, weight=res4b3_branch2a_weight,
                                                    num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                                    workspace=workspace)

    context_bn4b3_branch2b1 = mx.symbol.BatchNorm(name="context_bn4b3_branch2b1", data=context_res4b3_branch2a,
                                                  gamma=bn4b3_branch2b1_gamma,
                                                  beta=bn4b3_branch2b1_beta,
                                                  moving_mean=bn4b3_branch2b1_moving_mean,
                                                  moving_var=bn4b3_branch2b1_moving_var,
                                                  fix_gamma=fix_gamma,
                                                  use_global_stats=use_global_stats, eps=eps)

    context_res4b3_branch2b1_relu = mx.symbol.Activation(name="context_res4b3_branch2b1_relu", data=context_bn4b3_branch2b1,
                                                         act_type='relu')

    context_res4b3_branch2b1 = mx.symbol.Convolution(name="context_res4b3_branch2b1", data=context_res4b3_branch2b1_relu,
                                                     dilate=(1, 1),
                                                     kernel=(3, 3), no_bias=True, weight=res4b3_branch2b1_weight,
                                                     num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                                     workspace=workspace)

    context_res4b3_out = context_res4b2_out + context_res4b3_branch2b1

    context_bn4b4_branch2a = mx.symbol.BatchNorm(name="context_bn4b4_branch2a", data=context_res4b3_out,
                                                 gamma=bn4b4_branch2a_gamma,
                                                 beta=bn4b4_branch2a_beta,
                                                 moving_mean=bn4b4_branch2a_moving_mean,
                                                 moving_var=bn4b4_branch2a_moving_var,
                                                 fix_gamma=fix_gamma,
                                                 use_global_stats=use_global_stats, eps=eps)
    context_res4b4_branch2a_relu = mx.symbol.Activation(name="context_res4b4_branch2a_relu", data=context_bn4b4_branch2a,
                                                        act_type='relu')

    context_res4b4_branch2a = mx.symbol.Convolution(name="context_res4b4_branch2a", data=context_res4b4_branch2a_relu,
                                                    dilate=(1, 1),
                                                    kernel=(3, 3), no_bias=True, weight=res4b4_branch2a_weight,
                                                    num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                                    workspace=workspace)

    context_bn4b4_branch2b1 = mx.symbol.BatchNorm(name="context_bn4b4_branch2b1", data=context_res4b4_branch2a,
                                                  gamma=bn4b4_branch2b1_gamma,
                                                  beta=bn4b4_branch2b1_beta,
                                                  moving_mean=bn4b4_branch2b1_moving_mean,
                                                  moving_var=bn4b4_branch2b1_moving_var,
                                                  fix_gamma=fix_gamma,
                                                  use_global_stats=use_global_stats, eps=eps)

    context_res4b4_branch2b1_relu = mx.symbol.Activation(name="context_res4b4_branch2b1_relu", data=context_bn4b4_branch2b1,
                                                         act_type='relu')

    context_res4b4_branch2b1 = mx.symbol.Convolution(name="context_res4b4_branch2b1", data=context_res4b4_branch2b1_relu,
                                                     dilate=(1, 1),
                                                     kernel=(3, 3), no_bias=True, weight=res4b4_branch2b1_weight,
                                                     num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                                     workspace=workspace)

    context_res4b4_out = context_res4b3_out + context_res4b4_branch2b1

    context_bn4b5_branch2a = mx.symbol.BatchNorm(name="context_bn4b5_branch2a", data=context_res4b4_out,
                                                 gamma=bn4b5_branch2a_gamma,
                                                 beta=bn4b5_branch2a_beta,
                                                 moving_mean=bn4b5_branch2a_moving_mean,
                                                 moving_var=bn4b5_branch2a_moving_var,
                                                 fix_gamma=fix_gamma,
                                                 use_global_stats=use_global_stats, eps=eps)

    context_res4b5_branch2a_relu = mx.symbol.Activation(name="context_res4b5_branch2a_relu", data=context_bn4b5_branch2a,
                                                        act_type='relu')
    context_res4b5_branch2a = mx.symbol.Convolution(name="context_res4b5_branch2a", data=context_res4b5_branch2a_relu,
                                                    dilate=(1, 1),
                                                    kernel=(3, 3), no_bias=True, weight=res4b5_branch2a_weight,
                                                    num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                                    workspace=workspace)

    context_bn4b5_branch2b1 = mx.symbol.BatchNorm(name="context_bn4b5_branch2b1", data=context_res4b5_branch2a,
                                                  gamma=bn4b5_branch2b1_gamma,
                                                  beta=bn4b5_branch2b1_beta,
                                                  moving_mean=bn4b5_branch2b1_moving_mean,
                                                  moving_var=bn4b5_branch2b1_moving_var,
                                                  fix_gamma=fix_gamma,
                                                  use_global_stats=use_global_stats, eps=eps)

    context_res4b5_branch2b1_relu = mx.symbol.Activation(name="context_res4b5_branch2b1_relu", data=context_bn4b5_branch2b1,
                                                         act_type='relu')

    context_res4b5_branch2b1 = mx.symbol.Convolution(name="context_res4b5_branch2b1", data=context_res4b5_branch2b1_relu,
                                                     dilate=(1, 1),
                                                     kernel=(3, 3), no_bias=True, weight=res4b5_branch2b1_weight,
                                                     num_filter=512, num_group=1, pad=(1, 1), stride=(1, 1),
                                                     workspace=workspace)

    context_res4b5_out = context_res4b4_out + context_res4b5_branch2b1

    return res4b5_out,  context_res4b5_out



def get_context_feature(in_data,is_train=True,workspace=2048,fix_gamma=True,use_global_stats=True,eps =1.001e-5):


    bn5a_branch2a_beta = mx.symbol.Variable(name="context_bn5a_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn5a_branch2a = mx.symbol.BatchNorm(name="context_bn5a_branch2a", data=in_data, beta=bn5a_branch2a_beta,
                                        fix_gamma=fix_gamma,
                                        use_global_stats=use_global_stats, eps=eps)
    res5a_branch2a_relu = mx.symbol.Activation(name="context_res5a_branch2a_relu", data=bn5a_branch2a, act_type='relu')

    res5a_branch1 = mx.symbol.Convolution(name="context_res5a_branch1", data=res5a_branch2a_relu, dilate=(1, 1), kernel=(1, 1),
                                          no_bias=True,
                                          num_filter=1024,
                                          num_group=1, pad=(0, 0), stride=(1, 1), workspace=workspace)
    res5a_branch2a = mx.symbol.Convolution(name="context_res5a_branch2a", data=res5a_branch2a_relu, dilate=(1, 1),
                                           kernel=(3, 3), no_bias=True,
                                           num_filter=512,
                                           num_group=1, pad=(1, 1), stride=(1, 1), workspace=workspace)

    bn5a_branch2b1_beta = mx.symbol.Variable(name="context_bn5a_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn5a_branch2b1 = mx.symbol.BatchNorm(name="context_bn5a_branch2b1", data=res5a_branch2a, beta=bn5a_branch2b1_beta,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res5a_branch2b1_relu = mx.symbol.Activation(name="context_res5a_branch2b1_relu", data=bn5a_branch2b1, act_type='relu')
    res5a_branch2b1 = mx.symbol.Convolution(name="context_res5a_branch2b1", data=res5a_branch2b1_relu, dilate=(2, 2),
                                            kernel=(3, 3), no_bias=True,
                                            num_filter=1024,
                                            num_group=1, pad=(2, 2), stride=(1, 1), workspace=workspace)

    res5a_out = res5a_branch1 + res5a_branch2b1

    # print "level 5a"
    # print res5a_out.infer_shape()[1]

    ##res5a b1
    bn5b1_branch2a_beta = mx.symbol.Variable(name="context_bn5b1_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn5b1_branch2a = mx.symbol.BatchNorm(name="context_bn5b1_branch2a", data=res5a_out, beta=bn5b1_branch2a_beta,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res5b1_branch2a_relu = mx.symbol.Activation(name="context_res5b1_branch2a_relu", data=bn5b1_branch2a, act_type='relu')

    res5b1_branch2a = mx.symbol.Convolution(name="context_res5b1_branch2a", data=res5b1_branch2a_relu, dilate=(2, 2),
                                            kernel=(3, 3), no_bias=True,
                                            num_filter=512, num_group=1, pad=(2, 2), stride=(1, 1),
                                            workspace=workspace)
    bn5b1_branch2b1_beta = mx.symbol.Variable(name="context_bn5b1_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn5b1_branch2b1 = mx.symbol.BatchNorm(name="context_bn5b1_branch2b1", data=res5b1_branch2a, beta=bn5b1_branch2b1_beta,
                                          fix_gamma=fix_gamma,
                                          use_global_stats=use_global_stats, eps=eps)

    res5b1_branch2b1_relu = mx.symbol.Activation(name="context_res5b1_branch2b1_relu", data=bn5b1_branch2b1,
                                                 act_type='relu')

    res5b1_branch2b1 = mx.symbol.Convolution(name="context_res5b1_branch2b1", data=res5b1_branch2b1_relu, dilate=(2, 2),
                                             kernel=(3, 3), no_bias=True,
                                             num_filter=1024, num_group=1, pad=(2, 2), stride=(1, 1),
                                             workspace=workspace)

    res5b1_out = res5a_out + res5b1_branch2b1

    # print "level 5b1"
    # print res5b1_out.infer_shape()[1]


    ##res5a b2
    bn5b2_branch2a_beta = mx.symbol.Variable(name="context_bn5b2_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn5b2_branch2a = mx.symbol.BatchNorm(name="context_bn5b2_branch2a", data=res5b1_out, beta=bn5b2_branch2a_beta,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res5b2_branch2a_relu = mx.symbol.Activation(name="context_res5b2_branch2a_relu", data=bn5b2_branch2a, act_type='relu')

    res5b2_branch2a = mx.symbol.Convolution(name="context_res5b2_branch2a", data=res5b2_branch2a_relu, dilate=(2, 2),
                                            kernel=(3, 3), no_bias=True,
                                            num_filter=512, num_group=1, pad=(2, 2), stride=(1, 1),
                                            workspace=workspace)

    bn5b2_branch2b1_beta = mx.symbol.Variable(name="context_bn5b2_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn5b2_branch2b1 = mx.symbol.BatchNorm(name="context_bn5b2_branch2b1", data=res5b2_branch2a, beta=bn5b2_branch2b1_beta,
                                          fix_gamma=fix_gamma,
                                          use_global_stats=use_global_stats, eps=eps)

    res5b2_branch2b1_relu = mx.symbol.Activation(name="context_res5b2_branch2b1_relu", data=bn5b2_branch2b1,
                                                 act_type='relu')

    res5b2_branch2b1 = mx.symbol.Convolution(name="context_res5b2_branch2b1", data=res5b2_branch2b1_relu, dilate=(2, 2),
                                             kernel=(3, 3), no_bias=True,
                                             num_filter=1024, num_group=1, pad=(2, 2), stride=(1, 1),
                                             workspace=workspace)

    res5b2_out = res5b1_out + res5b2_branch2b1

    # print "level 5b2"
    # print res5b2_out.infer_shape()[1]

    # res6a for downsampling
    bn6a_branch2a_beta = mx.symbol.Variable(name="context_bn6a_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn6a_branch2a = mx.symbol.BatchNorm(name="context_bn6a_branch2a", data=res5b2_out, beta=bn6a_branch2a_beta,
                                        fix_gamma=fix_gamma,
                                        use_global_stats=use_global_stats, eps=eps)
    res6a_branch2a_relu = mx.symbol.Activation(name="context_res6a_branch2a_relu", data=bn6a_branch2a, act_type='relu')

    res6a_branch1 = mx.symbol.Convolution(name="context_res6a_branch1", data=res6a_branch2a_relu, dilate=(1, 1),
                                          kernel=(1, 1), no_bias=True,
                                          num_filter=2048,
                                          num_group=1, pad=(0, 0), stride=(1, 1), workspace=workspace)
    res6a_branch2a = mx.symbol.Convolution(name="context_res6a_branch2a", data=res6a_branch2a_relu, dilate=(1, 1),
                                           kernel=(1, 1), no_bias=True,
                                           num_filter=512,
                                           num_group=1, pad=(0, 0), stride=(1, 1), workspace=workspace)

    bn6a_branch2b1_beta = mx.symbol.Variable(name="context_bn6a_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn6a_branch2b1 = mx.symbol.BatchNorm(name="context_bn6a_branch2b1", data=res6a_branch2a, beta=bn6a_branch2b1_beta,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res6a_branch2b1_relu = mx.symbol.Activation(name="context_res6a_branch2b1_relu", data=bn6a_branch2b1, act_type='relu')

    res6a_branch2b1 = mx.symbol.Convolution(name="context_res6a_branch2b1", data=res6a_branch2b1_relu,
                                            num_filter=1024, pad=(4, 4), kernel=(3, 3),
                                            num_group=1,
                                            stride=(1, 1), dilate=(4, 4), no_bias=True)

    bn6a_branch2b2_beta = mx.symbol.Variable(name="context_bn6a_branch2b2_beta", lr_mult=2.0, wd_mult=0.0)
    bn6a_branch2b2 = mx.symbol.BatchNorm(name="context_bn6a_branch2b2", data=res6a_branch2b1, beta=bn6a_branch2b2_beta,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res6a_branch2b2_relu = mx.symbol.Activation(name="context_res6a_branch2b2_relu", data=bn6a_branch2b2, act_type='relu')

    res6a_branch2b2 = mx.symbol.Convolution(name="context_res6a_branch2b2", data=res6a_branch2b2_relu, dilate=(1, 1),
                                            kernel=(1, 1), no_bias=True,
                                            num_filter=2048,
                                            num_group=1, pad=(0, 0), stride=(1, 1), workspace=workspace)

    if is_train:
        res6a_branch2b2 = mx.sym.Dropout(res6a_branch2b2, name="context_dropout_res6a_branch2b2", p=0.3)

    res6a_out = res6a_branch1 + res6a_branch2b2

    # print "level 6a"
    # print res6a_out.infer_shape()[1]

    # res7a
    bn7a_branch2a_beta = mx.symbol.Variable(name="context_bn7a_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn7a_branch2a = mx.symbol.BatchNorm(name="context_bn7a_branch2a", data=res6a_out, beta=bn7a_branch2a_beta,
                                        fix_gamma=fix_gamma,
                                        use_global_stats=use_global_stats, eps=eps)
    res7a_branch2a_relu = mx.symbol.Activation(data=bn7a_branch2a, name="context_res7a_branch2a_relu", act_type='relu')

    res7a_branch1 = mx.symbol.Convolution(name="context_res7a_branch1", data=res7a_branch2a_relu, dilate=(1, 1),
                                          kernel=(1, 1), no_bias=True,
                                          num_filter=4096,
                                          num_group=1, pad=(0, 0), stride=(1, 1), workspace=workspace)

    res7a_branch2a = mx.symbol.Convolution(name="context_res7a_branch2a", data=res7a_branch2a_relu, dilate=(1, 1),
                                           kernel=(1, 1), no_bias=True,
                                           num_filter=1024,
                                           num_group=1, pad=(0, 0), stride=(1, 1), workspace=workspace)

    bn7a_branch2b1_beta = mx.symbol.Variable(name="context_bn7a_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn7a_branch2b1 = mx.symbol.BatchNorm(name="context_bn7a_branch2b1", data=res7a_branch2a, beta=bn7a_branch2b1_beta,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res7a_branch2b1_relu = mx.symbol.Activation(name="context_res7a_branch2b1_relu", data=bn7a_branch2b1, act_type='relu')

    res7a_branch2b1 = mx.symbol.Convolution(name="context_res7a_branch2b1", data=res7a_branch2b1_relu,
                                            num_filter=2048, pad=(4, 4), kernel=(3, 3),
                                            num_group=1,
                                            stride=(1, 1), dilate=(4, 4), no_bias=True)

    bn7a_branch2b2_beta = mx.symbol.Variable(name="context_bn7a_branch2b2_beta", lr_mult=2.0, wd_mult=0.0)
    bn7a_branch2b2 = mx.symbol.BatchNorm(name="context_bn7a_branch2b2", data=res7a_branch2b1, beta=bn7a_branch2b2_beta,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)

    res7a_branch2b2_relu = mx.symbol.Activation(name="context_res7a_branch2b2_relu", data=bn7a_branch2b2, act_type='relu')
    res7a_branch2b2 = mx.symbol.Convolution(name="context_res7a_branch2b2", data=res7a_branch2b2_relu, dilate=(1, 1),
                                            kernel=(1, 1), no_bias=True,
                                            num_filter=4096,
                                            num_group=1, pad=(0, 0), stride=(1, 1), workspace=workspace)
    if is_train:
        res7a_branch2b2 = mx.sym.Dropout(res7a_branch2b2, name="context_dropout_res7a_branch2b2", p=0.5)

    res7a_out = res7a_branch1 + res7a_branch2b2

    bn7_beta = mx.symbol.Variable(name="context_bn7_beta", lr_mult=2.0, wd_mult=0.0)
    bn7 = mx.symbol.BatchNorm(name="context_bn7", data=res7a_out, beta=bn7_beta, fix_gamma=fix_gamma,
                              use_global_stats=use_global_stats, eps=eps)
    relu7 = mx.symbol.Activation(name="context_relu7", data=bn7, act_type='relu')

    # print "level 7"
    # print relu7.infer_shape()[1]
    return relu7


def get_conv_feature(in_data, is_train=True, workspace=2048, fix_gamma=True, use_global_stats=True, eps=1.001e-5):

    # print "level 4b5"
    # print res4b4_out.infer_shape()[1]

    # res5a for downsampling

    bn5a_branch2a_beta = mx.symbol.Variable(name="bn5a_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn5a_branch2a = mx.symbol.BatchNorm(name="bn5a_branch2a", data=in_data, beta=bn5a_branch2a_beta,
                                        fix_gamma=fix_gamma,
                                        use_global_stats=use_global_stats, eps=eps)
    res5a_branch2a_relu = mx.symbol.Activation(name="res5a_branch2a_relu", data=bn5a_branch2a, act_type='relu')

    res5a_branch1 = mx.symbol.Convolution(name="res5a_branch1", data=res5a_branch2a_relu, dilate=(1, 1), kernel=(1, 1),
                                          no_bias=True,
                                          num_filter=1024,
                                          num_group=1, pad=(0, 0), stride=(1, 1), workspace=workspace)
    res5a_branch2a = mx.symbol.Convolution(name="res5a_branch2a", data=res5a_branch2a_relu, dilate=(1, 1),
                                           kernel=(3, 3), no_bias=True,
                                           num_filter=512,
                                           num_group=1, pad=(1, 1), stride=(1, 1), workspace=workspace)

    bn5a_branch2b1_beta = mx.symbol.Variable(name="bn5a_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn5a_branch2b1 = mx.symbol.BatchNorm(name="bn5a_branch2b1", data=res5a_branch2a, beta=bn5a_branch2b1_beta,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res5a_branch2b1_relu = mx.symbol.Activation(name="res5a_branch2b1_relu", data=bn5a_branch2b1, act_type='relu')
    res5a_branch2b1 = mx.symbol.Convolution(name="res5a_branch2b1", data=res5a_branch2b1_relu, dilate=(2, 2),
                                            kernel=(3, 3), no_bias=True,
                                            num_filter=1024,
                                            num_group=1, pad=(2, 2), stride=(1, 1), workspace=workspace)

    res5a_out = res5a_branch1 + res5a_branch2b1

    # print "level 5a"
    # print res5a_out.infer_shape()[1]

    ##res5a b1
    bn5b1_branch2a_beta = mx.symbol.Variable(name="bn5b1_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn5b1_branch2a = mx.symbol.BatchNorm(name="bn5b1_branch2a", data=res5a_out, beta=bn5b1_branch2a_beta,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res5b1_branch2a_relu = mx.symbol.Activation(name="res5b1_branch2a_relu", data=bn5b1_branch2a, act_type='relu')

    res5b1_branch2a = mx.symbol.Convolution(name="res5b1_branch2a", data=res5b1_branch2a_relu, dilate=(2, 2),
                                            kernel=(3, 3), no_bias=True,
                                            num_filter=512, num_group=1, pad=(2, 2), stride=(1, 1),
                                            workspace=workspace)
    bn5b1_branch2b1_beta = mx.symbol.Variable(name="bn5b1_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn5b1_branch2b1 = mx.symbol.BatchNorm(name="bn5b1_branch2b1", data=res5b1_branch2a, beta=bn5b1_branch2b1_beta,
                                          fix_gamma=fix_gamma,
                                          use_global_stats=use_global_stats, eps=eps)

    res5b1_branch2b1_relu = mx.symbol.Activation(name="res5b1_branch2b1_relu", data=bn5b1_branch2b1,
                                                 act_type='relu')

    res5b1_branch2b1 = mx.symbol.Convolution(name="res5b1_branch2b1", data=res5b1_branch2b1_relu, dilate=(2, 2),
                                             kernel=(3, 3), no_bias=True,
                                             num_filter=1024, num_group=1, pad=(2, 2), stride=(1, 1),
                                             workspace=workspace)

    res5b1_out = res5a_out + res5b1_branch2b1

    # print "level 5b1"
    # print res5b1_out.infer_shape()[1]


    ##res5a b2
    bn5b2_branch2a_beta = mx.symbol.Variable(name="bn5b2_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn5b2_branch2a = mx.symbol.BatchNorm(name="bn5b2_branch2a", data=res5b1_out, beta=bn5b2_branch2a_beta,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res5b2_branch2a_relu = mx.symbol.Activation(name="res5b2_branch2a_relu", data=bn5b2_branch2a, act_type='relu')

    res5b2_branch2a = mx.symbol.Convolution(name="res5b2_branch2a", data=res5b2_branch2a_relu, dilate=(2, 2),
                                            kernel=(3, 3), no_bias=True,
                                            num_filter=512, num_group=1, pad=(2, 2), stride=(1, 1),
                                            workspace=workspace)

    bn5b2_branch2b1_beta = mx.symbol.Variable(name="bn5b2_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn5b2_branch2b1 = mx.symbol.BatchNorm(name="bn5b2_branch2b1", data=res5b2_branch2a, beta=bn5b2_branch2b1_beta,
                                          fix_gamma=fix_gamma,
                                          use_global_stats=use_global_stats, eps=eps)

    res5b2_branch2b1_relu = mx.symbol.Activation(name="res5b2_branch2b1_relu", data=bn5b2_branch2b1,
                                                 act_type='relu')

    res5b2_branch2b1 = mx.symbol.Convolution(name="res5b2_branch2b1", data=res5b2_branch2b1_relu, dilate=(2, 2),
                                             kernel=(3, 3), no_bias=True,
                                             num_filter=1024, num_group=1, pad=(2, 2), stride=(1, 1),
                                             workspace=workspace)

    res5b2_out = res5b1_out + res5b2_branch2b1

    # print "level 5b2"
    # print res5b2_out.infer_shape()[1]

    # res6a for downsampling
    bn6a_branch2a_beta = mx.symbol.Variable(name="bn6a_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn6a_branch2a = mx.symbol.BatchNorm(name="bn6a_branch2a", data=res5b2_out, beta=bn6a_branch2a_beta,
                                        fix_gamma=fix_gamma,
                                        use_global_stats=use_global_stats, eps=eps)
    res6a_branch2a_relu = mx.symbol.Activation(name="res6a_branch2a_relu", data=bn6a_branch2a, act_type='relu')

    res6a_branch1 = mx.symbol.Convolution(name="res6a_branch1", data=res6a_branch2a_relu, dilate=(1, 1),
                                          kernel=(1, 1), no_bias=True,
                                          num_filter=2048,
                                          num_group=1, pad=(0, 0), stride=(1, 1), workspace=workspace)
    res6a_branch2a = mx.symbol.Convolution(name="res6a_branch2a", data=res6a_branch2a_relu, dilate=(1, 1),
                                           kernel=(1, 1), no_bias=True,
                                           num_filter=512,
                                           num_group=1, pad=(0, 0), stride=(1, 1), workspace=workspace)

    bn6a_branch2b1_beta = mx.symbol.Variable(name="bn6a_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn6a_branch2b1 = mx.symbol.BatchNorm(name="bn6a_branch2b1", data=res6a_branch2a, beta=bn6a_branch2b1_beta,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res6a_branch2b1_relu = mx.symbol.Activation(name="res6a_branch2b1_relu", data=bn6a_branch2b1, act_type='relu')

    res6a_branch2b1 = mx.symbol.Convolution(name='res6a_branch2b1', data=res6a_branch2b1_relu,
                                            num_filter=1024, pad=(4, 4), kernel=(3, 3),
                                            num_group=1,
                                            stride=(1, 1), dilate=(4, 4), no_bias=True)

    bn6a_branch2b2_beta = mx.symbol.Variable(name="bn6a_branch2b2_beta", lr_mult=2.0, wd_mult=0.0)
    bn6a_branch2b2 = mx.symbol.BatchNorm(name="bn6a_branch2b2", data=res6a_branch2b1, beta=bn6a_branch2b2_beta,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res6a_branch2b2_relu = mx.symbol.Activation(name="res6a_branch2b2_relu", data=bn6a_branch2b2, act_type='relu')

    res6a_branch2b2 = mx.symbol.Convolution(name='res6a_branch2b2', data=res6a_branch2b2_relu, dilate=(1, 1),
                                            kernel=(1, 1), no_bias=True,
                                            num_filter=2048,
                                            num_group=1, pad=(0, 0), stride=(1, 1), workspace=workspace)

    if is_train:
        res6a_branch2b2 = mx.sym.Dropout(res6a_branch2b2, name="dropout_res6a_branch2b2", p=0.3)

    res6a_out = res6a_branch1 + res6a_branch2b2

    # print "level 6a"
    # print res6a_out.infer_shape()[1]

    # res7a
    bn7a_branch2a_beta = mx.symbol.Variable(name="bn7a_branch2a_beta", lr_mult=2.0, wd_mult=0.0)
    bn7a_branch2a = mx.symbol.BatchNorm(name="bn7a_branch2a", data=res6a_out, beta=bn7a_branch2a_beta,
                                        fix_gamma=fix_gamma,
                                        use_global_stats=use_global_stats, eps=eps)
    res7a_branch2a_relu = mx.symbol.Activation(data=bn7a_branch2a, name="res7a_branch2a_relu", act_type='relu')

    res7a_branch1 = mx.symbol.Convolution(name="res7a_branch1", data=res7a_branch2a_relu, dilate=(1, 1),
                                          kernel=(1, 1), no_bias=True,
                                          num_filter=4096,
                                          num_group=1, pad=(0, 0), stride=(1, 1), workspace=workspace)

    res7a_branch2a = mx.symbol.Convolution(name="res7a_branch2a", data=res7a_branch2a_relu, dilate=(1, 1),
                                           kernel=(1, 1), no_bias=True,
                                           num_filter=1024,
                                           num_group=1, pad=(0, 0), stride=(1, 1), workspace=workspace)

    bn7a_branch2b1_beta = mx.symbol.Variable(name="bn7a_branch2b1_beta", lr_mult=2.0, wd_mult=0.0)
    bn7a_branch2b1 = mx.symbol.BatchNorm(name="bn7a_branch2b1", data=res7a_branch2a, beta=bn7a_branch2b1_beta,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)
    res7a_branch2b1_relu = mx.symbol.Activation(name="res7a_branch2b1_relu", data=bn7a_branch2b1, act_type='relu')

    res7a_branch2b1 = mx.symbol.Convolution(name='res7a_branch2b1', data=res7a_branch2b1_relu,
                                            num_filter=2048, pad=(4, 4), kernel=(3, 3),
                                            num_group=1,
                                            stride=(1, 1), dilate=(4, 4), no_bias=True)

    bn7a_branch2b2_beta = mx.symbol.Variable(name="bn7a_branch2b2_beta", lr_mult=2.0, wd_mult=0.0)
    bn7a_branch2b2 = mx.symbol.BatchNorm(name="bn7a_branch2b2", data=res7a_branch2b1, beta=bn7a_branch2b2_beta,
                                         fix_gamma=fix_gamma,
                                         use_global_stats=use_global_stats, eps=eps)

    res7a_branch2b2_relu = mx.symbol.Activation(name="res7a_branch2b2_relu", data=bn7a_branch2b2, act_type='relu')
    res7a_branch2b2 = mx.symbol.Convolution(name='res7a_branch2b2', data=res7a_branch2b2_relu, dilate=(1, 1),
                                            kernel=(1, 1), no_bias=True,
                                            num_filter=4096,
                                            num_group=1, pad=(0, 0), stride=(1, 1), workspace=workspace)
    if is_train:
        res7a_branch2b2 = mx.sym.Dropout(res7a_branch2b2, name="dropout_res7a_branch2b2", p=0.5)

    res7a_out = res7a_branch1 + res7a_branch2b2

    bn7_beta = mx.symbol.Variable(name="bn7_beta", lr_mult=2.0, wd_mult=0.0)
    bn7 = mx.symbol.BatchNorm(name="bn7", data=res7a_out, beta=bn7_beta, fix_gamma=fix_gamma,
                              use_global_stats=use_global_stats, eps=eps)
    relu7 = mx.symbol.Activation(name="relu7", data=bn7, act_type='relu')

    # print "level 7"
    # print relu7.infer_shape()[1]
    return relu7

def get_feature(data,origin_data,is_train=True, workspace=2048, fix_gamma=True, use_global_stats=True, eps=1.001e-5):

    local_feature_share,global_feature_share = get_share_feature(data,origin_data,is_train, workspace, fix_gamma, use_global_stats, eps)

    local_feature = get_conv_feature(local_feature_share, is_train, workspace, fix_gamma, use_global_stats, eps)

    global_feature = get_context_feature(global_feature_share, is_train, workspace, fix_gamma, use_global_stats, eps)

    return local_feature,global_feature,local_feature_share,global_feature_share