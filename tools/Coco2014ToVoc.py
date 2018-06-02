#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: Coco2014ToVoc.py
@time: 2017/12/8 12:08
"""

#!/usr/bin/python

__author__ = 'hcaesar'

# Converts COCO segmentation .json files (GT or results) to one .png file per image.
#
# This script can be used for visualization of ground-truth and result files.
# Furthermore it can convert the ground-truth annotations to a more easily
# accessible .png format that is supported by many semantic segmentation methods.
#
# Note: To convert a result file to .png, we need to have both a valid GT file
# and the result file and set isAnnotation=False.
#
# The .png images are stored as indexed images, which means they contain both the
# segmentation map, as well as a color palette for visualization.
#
# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Licensed under the Simplified BSD License [see coco/license.txt]

import sys
import os
sys.path.append(os.path.join(".","lib"))
from dataset.pycocotools import mask
from dataset.pycocotools.cocostuffhelper import cocoSegmentationToPng
from dataset.pycocotools.coco import COCO
import skimage.io
import matplotlib.pyplot as plt

def cocoSegmentationToPngDemo(dataDir='/home/zyq/data/coco', dataTypeAnn='train2014', dataTypeRes='examples', \
        pngFolderName='export_png', isAnnotation=True, exportImageLimit=10):
    '''
    Converts COCO segmentation .json files (GT or results) to one .png file per image.
    :param dataDir: location of the COCO root folder
    :param dataTypeAnn: identifier of the ground-truth annotation file
    :param dataTypeRes: identifier of the result annotation file (if any)
    :param pngFolderName: the name of the subfolder where we store .png images
    :param isAnnotation: whether the COCO file is a GT annotation or a result file
    :return: None
    '''

    # Define paths
    annPath = '%s/annotations/instances_%s.json' % (dataDir, dataTypeAnn)
    pngFolder = '%s/annotations/%s' % (dataDir, pngFolderName)

    # Initialize COCO ground-truth API
    coco = COCO(annPath)
    imgIds = coco.getImgIds()

    # if exportImageLimit < len(imgIds):
    #     imgIds = imgIds[0:exportImageLimit]
    txt_path  = '%s/annotations/train_coco_seg.txt' % (dataDir)
    with open(txt_path,"w") as f:
        # Convert each image to a png
        imgCount = len(imgIds)
        for i in xrange(0, imgCount):
            imgId = imgIds[i]
            imgName = coco.loadImgs(ids=imgId)[0]['file_name'].replace('.jpg', '')
            print('Exporting image %d of %d: %s' % (i+1, imgCount, imgName))
            segmentationPath = '%s/%s.png' % (pngFolder, imgName)
            if cocoSegmentationToPng(coco, imgId, segmentationPath):
                f.write(imgName+"\n")

if __name__ == "__main__":

    cocoSegmentationToPngDemo()