#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/7 11:44
# @Author  : yueqing.zhuang
# @Site    : 
# @File    : camvid.py
# @Software: PyCharm

import cv2
import cPickle
import os
import itertools
from imdb import IMDB

class CamVID(IMDB):
    def __init__(self,image_set,root_path,dataset_path,result_path=None):
        self.vid_list = image_set.split('_')
        super(CamVID,self).__init__('camvid',image_set,root_path,dataset_path,result_path)
        self.root_path = root_path
        self.data_path = dataset_path
        self.num_classes =31
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images
        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set
        :return: the indexes of given image set
        """
        image_name_set = []
        for vid_name in self.vid_list:
            #Collection all subfolders
            image_set_main_folder_path = os.path.join(self.data_path, 'images', vid_name)
            image_name_set_tmp = [os.path.join(vid_name,filename) for parent, dirname, filename in os.walk(image_set_main_folder_path)]
            image_name_set.extend(list(itertools.chain.from_iterable(image_name_set_tmp)))
        return image_name_set

    def image_path_from_index(self, index):
        """
        find the image path from given index
        :param index: the given index
        :return: the image path
        """
        image_file = os.path.join(self.data_path,'images',index)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file


    def annotation_path_from_index(self, index):
        """
        find the gt path from given index
        :param index: the given index
        :return: the image path
        """
        index = os.path.splitext(index) + '_L.png'
        image_file = os.path.join(self.data_path, 'gts', index)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def load_segdb_from_index(self, index):
        """
        load segdb from given index
        :param index: given index
        :return: segdb
        """
        seg_rec = dict()
        seg_rec['image'] = self.image_path_from_index(index)
        im = cv2.imread(seg_rec['image'])
        size = im.shape

        seg_rec['height'] = size[0]
        seg_rec['width'] = size[1]

        seg_rec['seg_cls_path'] = self.annotation_path_from_index(index)
        seg_rec['flipped'] = False

        return seg_rec

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

