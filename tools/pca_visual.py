#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: pca_visual.py
@time: 2017/12/14 11:40
"""
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    norm = False
    baseline = True
    if baseline:
        res38_npy_origin = np.load('./numpy/resnet38-base.npy')
        print res38_npy_origin.shape
        dim = 4096
    else:
        res38_npy_origin = np.load('./numpy/Relation.npy')
        print res38_npy_origin.shape
        res38_npy_origin = res38_npy_origin[:,:,::-1]
        dim = 1024
    print res38_npy_origin.shape
    pca = PCA(n_components=3)
    res38_npy_flatte = res38_npy_origin.transpose((1,2,0)).reshape((-1,dim))
    pca.fit(res38_npy_flatte)
    res38_npy_flatte_pca = pca.transform(res38_npy_flatte)
    print res38_npy_flatte_pca.shape
    res38_npy_flatte_pca_image = res38_npy_flatte_pca.transpose((1,0)).reshape((3,res38_npy_origin.shape[1],res38_npy_origin.shape[2])).transpose((1,2,0))

    if norm:
        amin, amax = res38_npy_flatte_pca_image.min(), res38_npy_flatte_pca_image.max()  # 求最大最小值
        res38_npy_flatte_pca_image = (res38_npy_flatte_pca_image - amin) / (amax - amin)*255
        res38_npy_flatte_pca_image = res38_npy_flatte_pca_image.astype(np.uint8)
    plt.imshow(res38_npy_flatte_pca_image)
    plt.show()

    # res38_dcn_gru_npy  = np.load('./numpy/resnet38_dcn_gru.npy')
    # print res38_dcn_gru_npy.shape
