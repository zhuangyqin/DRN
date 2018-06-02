"""
Segmentation:
data =
    {'data': [num_images, c, h, w],
     'im_info': [num_images, 4] (optional)}
label =
    {'label': [batch_size, 1] <- [batch_size, c, h, w]}
"""

import numpy as np
import math
import random
import cv2
import os
from PIL import Image
from utils.sutil import make_divisible
from utils.image import tensor_vstack,resize,transform,clip_boxes,transform_seg_gt,resize_seg
import numpy as np
import numpy.random as npr


def get_interp_method(imh_src, imw_src, imh_dst, imw_dst, default=Image.CUBIC):
    if imh_dst < imh_src and imw_dst < imw_src:
        return Image.ANTIALIAS
    elif imh_dst > imh_src and imw_dst > imw_src:
        return Image.CUBIC
    else:
        return Image.LINEAR


def get_segmentation_image_voc(segdb, config, is_train = True,has_label=True):

    num_images = len(segdb)
    max_h, max_w = config.SCALES[0]

    crop_height, crop_width = config.TRAIN.crop_size
    label_stride = config.network.LABEL_STRIDE

    for _ in xrange(num_images):
        output_data = np.zeros((num_images, 3, crop_height, crop_width), np.single)
        output_label = np.zeros((num_images,1 ,crop_height//label_stride, crop_width//label_stride), np.single)

    for i in range(num_images):
        seg_rec = segdb[i]
        # read the image
        assert os.path.exists(seg_rec['image']), '%s does not exist'.format(seg_rec['image'])
        im = np.array(cv2.imread(seg_rec['image']),np.uint8)
        label = np.array(Image.open(seg_rec['seg_cls_path']),np.uint8)

        # flipped the image or not
        if seg_rec.get('flipped', False):
            im = im[:, ::-1, :]
            label = label[:, ::-1]

        h, w = im.shape[:2]
        assert h <= max_h and w <= max_w

        d0 = max(1, int(label_stride // 2))
        d1 = max(0, int(label_stride - d0))
        target_size_range = [int(_ * max(h, w)) for _ in config.TRAIN.scale_range]
        min_rate = 1. * target_size_range[0] / max(max_h, max_w)
        max_crop_size = int(max(crop_width, crop_height) / min_rate)
        nim_size = max(max_crop_size, max_h, max_w) + d1 + d0

        # image
        nim = np.zeros((nim_size, nim_size, 3), np.single)
        nim += config.network.PIXEL_MEANS
        nim[d0:d0 + h, d0:d0 + w, :] = im

        # label
        nlabel = 255 * np.ones((nim_size, nim_size), np.uint8)
        label_2_id = seg_rec.get("ID2trainID", None)
        if label_2_id is not None:
            label = label_2_id[label]
        nlabel[d0:d0 + h, d0:d0 + w] = label

        target_size = npr.randint(target_size_range[0], target_size_range[1] + 1)
        random_rate = 1. * target_size / max(h, w)
        window_crop_height = int(crop_height*random_rate)
        window_crop_width = int(crop_width*random_rate)

        sy = npr.randint(0, max(1, label_stride, d0 + h - window_crop_height + 1))
        sx = npr.randint(0, max(1, label_stride, d0 + w - window_crop_width + 1))

        tim = nim[sy:sy + window_crop_height, sx :sx  + window_crop_width, :].astype(np.uint8)
        interp_method = get_interp_method(window_crop_height, window_crop_width, crop_height, crop_width)

        rim = Image.fromarray(tim).resize((crop_width,crop_height), interp_method)
        rim = np.array(rim)

        rim = transform(rim,color_scale=config.network.COLOR_SCALE,
                                pixel_means=config.network.PIXEL_MEANS,
                                pixel_stds=config.network.PIXEL_STDS)

        lsy = label_stride / 2
        lsx = label_stride / 2

        tlabel = nlabel[sy:sy + window_crop_height, sx :sx  + window_crop_width]
        rlabel = Image.fromarray(tlabel).resize((crop_width, crop_height), Image.NEAREST)
        rlabel = np.array(rlabel)[lsy: crop_height: label_stride, lsx: crop_width: label_stride]
        rlabel = transform_seg_gt(rlabel)


        output_data[i][:] = rim
        output_label[i][:] = rlabel

    return output_data, output_label

def get_segmentation_image(segdb, config,is_train = True,has_label=True,scale=1.0):
    """
    propocess image and return segdb
    :param segdb: a list of segdb
    :return: list of img as mxnet format
    """
    num_images = len(segdb)
    assert num_images > 0, 'No images'
    processed_ims = []
    origin_ims = []
    origin_labels = []
    processed_segdb = []
    processed_seg_cls_gt = []
    for i in range(num_images):
        seg_rec = segdb[i]

        assert os.path.exists(seg_rec['image']), '%s does not exist'.format(seg_rec['image'])
        im = np.array(cv2.imread(seg_rec['image']))
        if seg_rec.get('flipped',False):
            im = im[:,::-1,:]
        new_rec = seg_rec.copy()

        stride = config.network.LABEL_STRIDE
        network_ratio = config.network.ratio

        im_size_short = min(im.shape[0], im.shape[1])
        im_size_long = max(im.shape[0], im.shape[1])
        origin_target_size = im_size_short*network_ratio
        origin_max_size =  im_size_long*network_ratio
        origin_im, im_scale = resize(im, origin_target_size, origin_max_size, stride=stride)
        origin_im_tensor = transform(origin_im, color_scale=config.network.COLOR_SCALE,
                              pixel_means=config.network.PIXEL_MEANS,
                              pixel_stds=config.network.PIXEL_STDS)
        origin_ims.append(origin_im_tensor)

        ## ramdom scale for the image size

        enable_range_scale = config.TRAIN.enable_scale
        if  is_train and enable_range_scale:
            range_scale = config.TRAIN.scale_range
            im_size_short = min(im.shape[0],im.shape[1])
            im_size_long = max(im.shape[0],im.shape[1])
            ratio = float(im_size_long)/im_size_short
            target_size = random.randrange(math.ceil(range_scale[0]*im_size_short),math.floor(range_scale[1]*im_size_short))
            if config.TRAIN.enable_crop and target_size < min(config.TRAIN.crop_size[:2]):
                target_size = min(config.TRAIN.crop_size[:2])
            max_size =  math.floor(target_size*ratio)
            # print "random"
            im, im_scale = resize(im, target_size, max_size, stride=stride)
        elif is_train:
            scale_ind = random.randrange(len(config.SCALES))
            target_size = config.SCALES[scale_ind][0]
            max_size = config.SCALES[scale_ind][1]
            im, im_scale = resize(im, target_size, max_size, stride=stride)
        else:
            target_size = min(im.shape[:2])
            max_size = max(im.shape[:2])
            # print "-----"
            # print "origin imsize:",im.shape[:20]
            stride = 0
            origin_im = im
            im, im_scale = resize(im, target_size*scale, max_size*scale, stride=stride)
            # print "resize imsize:", im.shape[:2]
            # print "-----"

        im_tensor = transform(im,color_scale=config.network.COLOR_SCALE,
                                pixel_means=config.network.PIXEL_MEANS,
                                pixel_stds=config.network.PIXEL_STDS)
        if is_train:
            im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        else:
            im_info = [origin_im.shape[0], origin_im.shape[1], 1]

        new_rec['im_info'] = im_info
        processed_ims.append(im_tensor)
        processed_segdb.append(new_rec)
        # label infomation
        if has_label:
            seg_cls_gt = np.array(Image.open(seg_rec['seg_cls_path']))
            if seg_rec['flipped']:
                seg_cls_gt = seg_cls_gt[:, ::-1]
            label_2_id = seg_rec.get("ID2trainID", None)
            if label_2_id is not None:
                seg_cls_gt = label_2_id[seg_cls_gt]

            origin_seg_cls_gt,_ = resize_seg(seg_cls_gt,origin_target_size,
                                           origin_max_size,stride)
            origin_labels.append(transform_seg_gt(origin_seg_cls_gt))

            seg_cls_gt, seg_cls_gt_scale = resize_seg(
                    seg_cls_gt, target_size, max_size, stride=stride)
            seg_cls_gt_tensor = transform_seg_gt(seg_cls_gt)
            processed_seg_cls_gt.append(seg_cls_gt_tensor)
        else:
            processed_seg_cls_gt = []

    return processed_ims, processed_seg_cls_gt, processed_segdb,origin_ims,origin_labels

def get_segmentation_test_batch(segdb, config,is_train=False,has_label=True,scale=1.0):
    """
    return a dict of train batch
    :param segdb: ['image', 'flipped']
    :param config: the config setting
    :return: data, label, im_info
    """
    imgs, seg_cls_gts, segdb, origin_ims, origin_labels = get_segmentation_image(segdb, config,is_train=is_train,has_label=has_label,scale=scale)

    im_array =  tensor_vstack(imgs)
    if has_label:
        seg_cls_gt = tensor_vstack(seg_cls_gts)
    else :
        seg_cls_gt = []
    im_info = tensor_vstack([np.array([isegdb['im_info']], dtype=np.float32) for isegdb in segdb])
    origin_im = tensor_vstack(origin_ims)
    rois = []

    if config.network.use_crop_context:
        crop_context_scale = config.network.crop_context_scale
        crop_height, crop_width =config.TRAIN.crop_size
        feature_stride = config.network.LABEL_STRIDE
        scale_width = make_divisible(int(float(crop_width) / crop_context_scale), feature_stride)
        scale_height = make_divisible(int(float(crop_height) / crop_context_scale), feature_stride)
        pad_width = int(scale_width - crop_width) / 2
        pad_height = int(scale_height - crop_height) / 2

        origin_data = np.zeros((im_array.shape[0], im_array.shape[1],
                                im_array.shape[2] + 2 * int(pad_height),
                                im_array.shape[3] + 2 * int(pad_width)))
        origin_data[:, :, int(pad_height):im_array.shape[2] + int(pad_height),
            int(pad_width):im_array.shape[3]+int(pad_width)] = im_array

        for i, im_info in enumerate(im_info):
            im_size = im_info[:2]
            rois.append(np.array([i, pad_width, pad_height, pad_width+im_size[1], pad_width+im_size[0]]).reshape((1, 5)))
        rois = tensor_vstack(rois)
        # print rois

    else:
        network_ratio = config.network.ratio
        for i, im_info in enumerate(im_info):
            im_size = im_info[:2]
            rois.append(np.array([i, 0, 0, im_size[1] * network_ratio, im_size[0] * network_ratio]).reshape((1, 5)))
        rois = tensor_vstack(rois)
        print rois

    data = {'data': im_array,
            'im_info': im_info,
            'origin_data': origin_im,
            'rois':rois}

    label = {'label':seg_cls_gt}

    return {'data': data, 'label': label}

def get_segmentation_train_batch(segdb, config,is_train=True):
    """
    return a dict of train batch
    :param segdb: ['image', 'flipped']
    :param config: the config setting
    :return: data, label, im_info
    """

    imgs, seg_cls_gts, segdb,origin_ims,origin_labels = get_segmentation_image(segdb, config,is_train=is_train,has_label=True)

    im_array =  tensor_vstack(imgs)
    seg_cls_gt = tensor_vstack(seg_cls_gts,pad=255)
    origin_im = tensor_vstack(origin_ims)
    origin_label = tensor_vstack(origin_labels,pad=255)
    im_info = tensor_vstack([np.array([isegdb['im_info']], dtype=np.float32) for isegdb in segdb])

    data = {'data': im_array,
            'im_info': im_info,
            'origin_data':origin_im}
    label = {'label': seg_cls_gt,
             'origin_label': origin_label}

    return data, label