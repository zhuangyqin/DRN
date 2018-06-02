#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: sutil.py
@time: 2017/11/5 21:21
"""
import numpy as np

def make_divisible(v, divider):
    return int(np.ceil(float(v) / divider) * divider)