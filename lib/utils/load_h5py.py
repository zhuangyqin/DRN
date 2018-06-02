#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: load_h5py.py
@time: 2017/11/5 10:50
"""
import os
import os.path as osp
import h5py
import numpy as np

def save_batch_softmax_ouputs(result_folder,segdb,batch_softmax_outputs):
    softmax_outputs = []
    for batch_softmax_output in batch_softmax_outputs:
        for isoftmax_output in batch_softmax_output:
            softmax_outputs.append(isoftmax_output)

    pred_save_folder = osp.join(result_folder, 'softmax_outputs')
    if not osp.exists(pred_save_folder):
        os.makedirs(pred_save_folder)

    for isegdb,isoftmax_output in zip(segdb,softmax_outputs):
        image_pathes = osp.split(isegdb['image'])
        res_image_name = image_pathes[1][:-len('_leftImg8bit.png')]
        data = (isoftmax_output.astype(np.float32), isegdb['height'], isegdb['width'])
        pred_save_path = osp.join(pred_save_folder, '{}.h5'.format(res_image_name))
        h5py_save(pred_save_path,*data)

def h5py_save(to_path, *data):
    with h5py.File(to_path, 'w') as f:
        for i, datum in enumerate(data):
            f.create_dataset('d{}'.format(i), data=datum)

def h5py_load(from_path):
    data = []
    if osp.isfile(from_path):
        with h5py.File(from_path) as f:
            for k in f.keys():
                data.append(f[k][()])
    return tuple(data)