#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: diff_pascal_context.py
@time: 2018/1/18 9:47
"""

from PIL import Image
import os
import numpy as np

def main():
    workspace_dir= "../workspace"
    list_file = os.path.join(workspace_dir,'val.txt')
    with open(list_file) as f:
        val_names = f.readlines()

    print "start"
    for file_name in val_names:
        file_name = file_name.strip()
        print file_name
        pred_label = np.array(Image.open(os.path.join(workspace_dir,'prediction',file_name+'.png')))
        gt_label = np.array(Image.open(os.path.join(workspace_dir,'label',file_name+'.png')))

        Invalid_Mask = np.zeros(gt_label.shape)
        Invalid_Mask[np.where(gt_label == 255)] = 1
        Image_Sub = Invalid_Mask * 255 + (gt_label == pred_label) * 127
        diff_dir = os.path.join(workspace_dir,'difference')
        if not os.path.exists(diff_dir):
            os.makedirs(diff_dir)
        diff_path = os.path.join(diff_dir,file_name + '.png')
        Image.fromarray(Image_Sub.astype(np.uint8)).save(diff_path)

if __name__ =="__main__":
    main()