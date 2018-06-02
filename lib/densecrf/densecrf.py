#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: densecrf.py
@time: 2017/10/31 17:47
"""
import numpy as np
import cv2

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

def ApplyCRF_batch(segdb, softmax_outputs, interation=5):
    pass


def ApplyCRF_one_from_softmax(img,softmax_output,interation=5):


    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)