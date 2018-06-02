#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: calculate_rf.py
@time: 2017/11/19 12:28
"""
# !/usr/bin/env python
from math import floor,ceil


# # fsize, stride, pad, dilate
# net_struct = {
#     'alexnet': {'net': [[11, 4, 0, 1], [3, 2, 0, 1], [5, 1, 2, 1], [3, 2, 0, 1], [3, 1, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1], [3, 2, 0, 1]],
#                 'name': ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5', 'pool5']},
#     'vgg16': {'net': [[3, 1, 1, 1], [3, 1, 1, 1], [2, 2, 0, 1], [3, 1, 1, 1], [3, 1, 1, 1], [2, 2, 0, 1], [3, 1, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1],
#                       [2, 2, 0, 1], [3, 1, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1], [2, 2, 0, 1], [3, 1, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1],
#                       [2, 2, 0, 1]],
#               'name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2',
#                        'conv3_3', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3',
#                        'pool5']},
#     'zf-5': {'net': [[7, 2, 3, 1], [3, 2, 1, 1], [5, 2, 2, 1], [3, 2, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1]],
#              'name': ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5']}}
#
#

net_struct= {
    'resnet-38':{'net': [ [3, 1, 1, 1],
                          # res2a
                          [3, 2, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1],[3, 1, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1],
                          # res3a
                          [3, 2, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1],[3, 1, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1],
                          # res4a
                          [3, 2, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1],[3, 1, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1],
                          [3, 1, 1, 1], [3, 1, 1, 1],  [3, 1, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1],
                          # res5a
                          [3, 1, 2, 2], [3, 1, 2, 2], [3, 1, 2, 2],[3, 1, 2, 2], [3, 1, 2, 2], [3, 1, 2, 2],
                          # res6
                          # res5a
                          [1, 1, 0, 1], [3, 1, 4, 4], [1, 1, 0, 1],
                          [1, 1, 0, 1], [3, 1, 4, 4], [1, 1, 0, 1],
                          ],
                 'name':['conv1a',
                         'res2a_branch2a','res2a_branch2b1','res2b1_branch2a', 'res2b1_branch2b1', 'res2b2_branch2a', 'res2b2_branch2a',
                         'res3a_branch2a','res3a_branch2b1','res3b1_branch2a', 'res3b1_branch2b1', 'res3b2_branch2a', 'res3b2_branch2a',
                         'res4a_branch2a', 'res4a_branch2b1', 'res4b1_branch2a', 'res4b1_branch2b1', 'res4b2_branch2a','res4b2_branch2a',
                         'res4b3_branch2a', 'res4b3_branch2b1', 'res4b4_branch2a', 'res4b4_branch2b1', 'res4b5_branch2a','res4b5_branch2a',
                         'res5a_branch2a','res5a_branch2b1','res5b1_branch2a', 'res5b1_branch2b1', 'res5b2_branch2a', 'res5b2_branch2a',
                         'res6a_branch2a','res6a_branch2b1','res6a_branch2b2',
                         'res7a_branch2a', 'res7a_branch2b1', 'res7a_branch2b2']}
}


# net_struct= {
#     'resnet-38':{'net': [ [3, 1, 1, 1],
#                           # res2a
#                           [3, 2, 1, 1], [3, 1, 1, 1], [3, 1, 1, 1],[3, 1, 1, 1], [3, 1, 2, 2], [3, 1, 2, 2],
#                           # res3a
#                           [3, 2, 5, 5], [3, 1, 5, 5], [3, 1, 9, 9],[3, 1, 9, 9], [3, 1, 1, 1], [3, 1, 1, 1],
#                           # res4a
#                           [3, 2, 2, 2], [3, 1, 2, 2], [3, 1, 5, 5],[3, 1, 5, 5], [3, 1, 9, 9], [3, 1, 9, 9],
#                           [3, 1, 1, 1], [3, 1, 1, 1],  [3, 1, 2, 2], [3, 1, 2, 2], [3, 1, 5, 5], [3, 1, 5, 5],
#                           # res5a
#                           [3, 1, 1, 1], [3, 1, 1, 1], [3, 1, 9, 9],[3, 1, 9, 9], [3, 1, 2, 2], [3, 1, 2, 2],
#                           # res6
#                           # res5a
#                           [1, 1, 0, 1], [3, 1, 4, 4], [1, 1, 0, 1],
#                           [1, 1, 0, 1], [3, 1, 4, 4], [1, 1, 0, 1],
#                           ],
#                  'name':['conv1a',
#                          'res2a_branch2a','res2a_branch2b1','res2b1_branch2a', 'res2b1_branch2b1', 'res2b2_branch2a', 'res2b2_branch2a',
#                          'res3a_branch2a','res3a_branch2b1','res3b1_branch2a', 'res3b1_branch2b1', 'res3b2_branch2a', 'res3b2_branch2a',
#                          'res4a_branch2a', 'res4a_branch2b1', 'res4b1_branch2a', 'res4b1_branch2b1', 'res4b2_branch2a','res4b2_branch2a',
#                          'res4b3_branch2a', 'res4b3_branch2b1', 'res4b4_branch2a', 'res4b4_branch2b1', 'res4b5_branch2a','res4b5_branch2a',
#                          'res5a_branch2a','res5a_branch2b1','res5b1_branch2a', 'res5b1_branch2b1', 'res5b2_branch2a', 'res5b2_branch2a',
#                          'res6a_branch2a','res6a_branch2b1','res6a_branch2b2',
#                          'res7a_branch2a', 'res7a_branch2b1', 'res7a_branch2b2']}
# }

imsize = 512


def outFromIn(isz, net, layernum):
    totstride = 1
    insize = isz
    for layer in range(layernum):
        fsize, stride, pad, dilate = net[layer]
        outsize = floor(float(insize + 2 * pad - (dilate*(fsize-1)+1)) / stride)+ 1
        insize = outsize
        totstride = totstride * stride
    return outsize, totstride


def inFromOut(net, layernum):
    RF = 1
    for layer in reversed(range(layernum)):
        fsize, stride, pad, dilate = net[layer]
        RF = ((RF - 1) * stride* dilate) + fsize
    return RF


if __name__ == '__main__':
    print "layer output sizes given image = %dx%d" % (imsize, imsize)

    for net in net_struct.keys():
        print '************net structrue name is %s**************' % net
        for i in range(len(net_struct[net]['net'])):
            p = outFromIn(imsize, net_struct[net]['net'], i + 1)
            rf = inFromOut(net_struct[net]['net'], i + 1)
            print "Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d" % (
            net_struct[net]['name'][i], p[0], p[1], rf)