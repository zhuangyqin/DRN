#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: apply_post_crf.py
@time: 2017/11/5 16:49
"""
import os
import itertools
from cv2 import imread, imwrite
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

def ApplyCRF_cityscape_one(image_path,prediction_path,save_path,gt_path):
    pass

def ApplyCRF_cityscape(image_folder,prediction_folder,save_folder,gt_folder):
    image_name_set = [filename for parent, dirname, filename in os.walk(image_folder)]
    image_name_set = list(itertools.chain.from_iterable(image_name_set))


if __name__ == "__main__":
    image_folder=""
    prediction_folder =""
    save_folder = ""
    gt_folder= ""
    ApplyCRF_cityscape(image_folder, prediction_folder,save_folder,gt_folder)
