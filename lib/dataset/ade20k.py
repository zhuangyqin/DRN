#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: ade20k.py
@time: 2018/2/5 20:25
"""
import cv2
import cPickle
import os
import numpy as np
import itertools

from imdb import IMDB
from PIL import Image
from collections import namedtuple

class ADE20K(IMDB):

    def __init__(self,image_set, root_path, dataset_path, result_path=None):
        self.image_set = image_set
        super(ADE20K, self).__init__('ADE20K', image_set, root_path, dataset_path, result_path)
        self.root_path = root_path
        self.data_path = dataset_path
        self.num_classes = 150
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)

        self.has_label = True
        if self.image_set_index == 'testing':
            self.has_label = False
        print 'num_images', self.num_images


    def load_image_set_index(self):
        image_set_folder_path = os.path.join(self.data_path,'images',self.image_set)
        image_name_set = [filename for parent, dirname, filename in os.walk(image_set_folder_path)]
        image_name_set = list(itertools.chain.from_iterable(image_name_set))

        return image_name_set

    def image_path_from_index(self, index):
        """
        find the image path from given index
        :param index: the given index
        :return: the image path
        """
        index = index.split('.')[0]
        image_file = os.path.join(self.data_path, 'images', self.image_set,  index + '.jpg')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def annotation_path_from_index(self, index):
        """
        find the gt path from given index
        :param index: the given index
        :return: the image path
        """
        index = index.split('.')[0]
        image_file = os.path.join(self.data_path,'annotations',  self.image_set, index+'.png')
        if self.image_set == 'testing':
            image_file = None
        else:
            assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file


    def load_segdb_from_index(self, index):

        seg_rec = dict()
        seg_rec['image'] = self.image_path_from_index(index)

        im = cv2.imread(seg_rec['image'])
        size = im.shape

        seg_rec['height'] = size[0]
        seg_rec['width'] = size[1]
        if self.has_label:
            seg_rec['seg_cls_path'] = self.annotation_path_from_index(index)
        seg_rec['flipped'] = False
        seg_rec['ID2trainID'] = self.ID2trainID()

        return seg_rec


    def ID2trainID(self):
        ID2trainID_ARRAY = 255 * np.ones((256,))
        ID2trainID_ARRAY[:np.arange(-1,150).shape[0]] = np.arange(-1,150)
        ID2trainID_ARRAY[0] = 255
        return ID2trainID_ARRAY.astype(np.int32)

    def trainID2ID(self):
        trainID2ID_ARRAY = np.arange(1,256+1)
        trainID2ID_ARRAY[255] = 0

        return trainID2ID_ARRAY.astype(np.int32)


    def gt_segdb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_segdb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                segdb = cPickle.load(fid)
            print '{} gt segdb loaded from {}'.format(self.name, cache_file)
            return segdb

        gt_segdb = [self.load_segdb_from_index(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_segdb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt segdb to {}'.format(cache_file)

        return gt_segdb

    def getpallete(self):
        """
        this function is to get the colormap for visualizing the segmentation mask
        :param num_cls: the number of visulized class
        :return: the pallete
        """
        n = self.num_classes
        pallete = [0] * (n * 3)
        for j in xrange(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete

    def write_segmentation_result(self, segmentation_results):
        """
        Write the segmentation result to result_file_folder
        :param segmentation_results: the prediction result
        :param result_file_folder: the saving folder
        :return: [None]
        """
        res_file_folder = os.path.join(self.result_path, 'results', 'prediction')
        if not os.path.exists(res_file_folder):
            os.makedirs(res_file_folder)

        pallete = self.getpallete()
        trainID2ID = self.trainID2ID()

        for i, index in enumerate(self.image_set_index):
            index = index.split('.')[0]
            # seg_image_info =self.image_path_from_index(index)
            res_save_path = os.path.join(res_file_folder, index + '.png')

            if trainID2ID is not None:
                segmentation_result = np.uint8(trainID2ID[np.squeeze(np.copy(segmentation_results[i]))])
            else:
                segmentation_result = np.uint8(np.squeeze(np.copy(segmentation_results[i])))

            segmentation_result = Image.fromarray(segmentation_result)
            segmentation_result.putpalette(pallete)
            segmentation_result.save(res_save_path)


    def evaluate_segmentations(self, pred_segmentations = None):
        """
        top level evaluations
        :param pred_segmentations: the pred segmentation result
        :return: the evaluation results
        """
        if not (pred_segmentations is None):
            self.write_segmentation_result(pred_segmentations)

        return None



