#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/28 14:09
# @Author  : yueqing.zhuang
# @Site    : 
# @File    : Gen_cityscape_list.py
# @Software: PyCharm

"""
    this is used to generate the list of the cityscapes for training the time through 
    output list line format is as blow
    ['img_current','img_s_ago','img_l_ago','img_gt_label']
"""

def gen_list(data_root,image_set,listoutput_path):
    image_set_main_folder, image_set_sub_folder = image_set.split('_', 1)

