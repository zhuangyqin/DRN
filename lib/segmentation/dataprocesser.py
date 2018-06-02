#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: dataprocesser.py
@time: 2017/9/13 13:35
"""
import os
import cv2
import random
import numpy as np

from PIL import Image
from math import floor, ceil
from utils.image import tensor_vstack,resize_im_target,rotation,transform,transform_seg_gt,flip,\
    showImwithGt,resize_target,showGt,resize_seg_target,resize,resize_seg

Debug = 0

class SegDataProcesser(object):

    def __init__(self,enable_crop, crop_size, enable_scale,scale_range,enable_rotation,
                 rotation_range,color_scale, pixel_means,pixel_stds,flipped_ratio,label_stride=1,has_label=True):

        self.enable_crop = enable_crop
        self.crop_height,self.crop_width= crop_size
        self.enable_scale = enable_scale
        self.min_scale,self.max_scale = scale_range
        self.enable_rotation = enable_rotation
        self.min_rotation,self.max_rotation = rotation_range
        self.color_scale = color_scale
        self.pixel_stds = pixel_stds
        self.pixel_means = pixel_means
        self.flipped_ratio = flipped_ratio
        self.label_stride = label_stride
        self.has_label = has_label

    def make_divisible(self, v, divider):
        return int(np.ceil(float(v) / divider) * divider)

    def get_segmentation_batch(self,segdb):
        num_images = len(segdb)
        assert num_images>0,'no images'

        processed_ims = []
        if self.has_label:
            processed_seg_cls_gt = []
        processed_segdb = []

        multiprocess_results = []

        for i in range(num_images):
            seg_rec = segdb[i]
            im,seg_cls_gt,new_rec = self.get_an_sample(seg_rec,)
            processed_ims.append(im)
            if self.has_label:
                processed_seg_cls_gt.append(seg_cls_gt)
            processed_segdb.append(new_rec)


        im_arrays = tensor_vstack(processed_ims)
        seg_cls_gts = tensor_vstack(processed_seg_cls_gt)
        im_infos = tensor_vstack([np.array([isegdb['im_info']], dtype=np.float32) for isegdb in processed_segdb])

        #pack in data and label and then together
        # print(im_arrays.shape,seg_cls_gts.shape)
        if Debug:
            print "im_arrays.shape",im_arrays.shape
            print "seg_cls_gt.shape",seg_cls_gts.shape
        data = {'data': im_arrays,
                'im_info': im_infos}
        label = {'label': seg_cls_gts}

        return {'total_data': data, 'total_label': label}


    def get_an_sample(self,seg_rec):

        new_rec = seg_rec.copy()

        assert os.path.exists(seg_rec['image']), '%s does not exist'.format(seg_rec['image'])
        im = np.array(cv2.imread(seg_rec['image']))
        if self.has_label:
            assert os.path.exists(seg_rec['seg_cls_path']), '%s does not exist'.format(seg_rec['seg_cls_path'])
            seg_cls_gt = np.array(Image.open(seg_rec['seg_cls_path']))
            ID2trainID = seg_rec.get("ID2trainID", None)
            if ID2trainID is not None:
                seg_cls_gt = np.array(ID2trainID[seg_cls_gt]).astype(np.uint8)

        im_height, im_width = im.shape[:2]
        if Debug:
            showImwithGt(im, seg_cls_gt)

        # config
        if self.enable_crop:
            target_h, target_w = self.crop_height, self.crop_width
        else:
            target_h, target_w = im_height, im_width

        # rotation operation
        im_scale = 1
        if self.enable_rotation:
            if Debug:
                print "rotation:", self.min_rotation,self.max_rotation
            random_degree = random.uniform(self.min_rotation, self.max_rotation)
            im, im_scale = rotation(im, random_degree, fixed_scale=False, borderValue=tuple(self.pixel_means))
            if self.has_label:
                seg_cls_gt, im_scale = rotation(seg_cls_gt, random_degree, interpolation=cv2.INTER_NEAREST,
                                                fixed_scale=False, borderValue=(255,))
            if Debug:
                showImwithGt(im, seg_cls_gt)

        # scale and crop operation
        # 1. scale the image
        if self.enable_scale:

            im_size_short = min(im.shape[0],im.shape[1])
            im_size_long = max(im.shape[0],im.shape[1])
            ratio = float(im_size_long)/im_size_short
            target_size = random.randrange(ceil(self.min_scale * im_size_short),
                                           floor(self.max_scale * im_size_short))

            target_size = self.make_divisible(target_size,self.label_stride)
            assert target_size>=target_h and target_size>=target_w

            max_size = floor(target_size * ratio)
            im, scale = resize(im, target_size, max_size, stride=self.label_stride)
            # print "scale:",scale
            if self.has_label:
                seg_cls_gt, seg_scale = resize_seg(
                    seg_cls_gt, target_size, max_size, self.label_stride)
                assert seg_scale==scale,"the scale must be same"
                if Debug:
                    showGt(seg_cls_gt)

            im_scale *= scale

            if Debug:
                print "target_size,max_size:", target_size, max_size
                showImwithGt(im,seg_cls_gt)


        # 2. make sure the start
        sx = int(floor(random.random() * (im.shape[1] - target_w + 1)))
        sy = int(floor(random.random() * (im.shape[0] - target_h + 1)))

        assert (sx >= 0 and sx < im.shape[1] - target_w + 1)
        assert (sy >= 0 and sy < im.shape[0] - target_h  + 1)

        ex = (int)(sx + target_w - 1)
        ey = (int)(sy + target_h - 1)

        # 3. crop the image
        im = im[sy:ey + 1, sx:ex + 1, ...]
        if self.has_label:
            seg_cls_gt = seg_cls_gt[sy:ey + 1, sx:ex + 1, ...]
        if Debug:
            showImwithGt(im, seg_cls_gt)

        # 4.ramdom flipped
        if random.uniform(0, 1) < self.flipped_ratio:
            im = flip(im)
            if self.has_label:
                seg_cls_gt = flip(seg_cls_gt)
        if Debug:
            showImwithGt(im, seg_cls_gt)

        # 5.resize to the target size
        im = resize_im_target(im, target_h=target_h, target_w=target_w)

        # 6.final transform the pixel_mean
        im = transform(im, pixel_means=self.pixel_means,color_scale=self.color_scale,
                       pixel_stds=self.pixel_stds)

        if self.has_label:
            if self.label_stride > 1:
                seg_target_h = target_h // self.label_stride
                seg_target_w = target_w // self.label_stride

                # seg_cls_gt = resize_seg_target(seg_cls_gt, target_h, target_w, self.label_stride)
                # use the pillow seg cls
                seg_cls_gt = np.array(Image.fromarray(np.squeeze(seg_cls_gt.astype(np.uint8, copy=False))).resize(
                    (target_w // self.label_stride,
                     target_h // self.label_stride),
                    Image.NEAREST))

            if Debug:
                showGt(seg_cls_gt)
            seg_cls_gt = transform_seg_gt(seg_cls_gt)

        im_info = [target_h, target_w, im_scale]
        new_rec['im_info'] = im_info

        return im,seg_cls_gt,new_rec