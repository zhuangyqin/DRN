import numpy as np
import os
import cv2
import random
import math
from PIL import Image
from bbox.bbox_transform import clip_boxes
from functools import partial
from multiprocessing import Pool
from matplotlib import pyplot as plt

Debug = 0

def showImwithGt(im, gt):
    print type(gt)
    print gt.shape, im.shape
    gt_color = Image.fromarray(gt, mode='P')
    # n = num_cls
    pallete_raw = np.zeros((256, 3)).astype('uint8')
    # pallete = np.zeros((n, 3)).astype('uint8')

    pallete_raw[0, :] = [128, 64, 128]
    pallete_raw[1, :] = [244, 35, 232]
    pallete_raw[2, :] = [70, 70, 70]
    pallete_raw[3, :] = [102, 102, 156]
    pallete_raw[4, :] = [190, 153, 153]
    pallete_raw[5, :] = [153, 153, 153]
    pallete_raw[6, :] = [250, 170, 30]
    pallete_raw[7, :] = [220, 220, 0]
    pallete_raw[8, :] = [107, 142, 35]
    pallete_raw[9, :] = [152, 251, 152]
    pallete_raw[10, :] = [70, 130, 180]
    pallete_raw[11, :] = [220, 20, 60]
    pallete_raw[12, :] = [255, 0, 0]
    pallete_raw[13, :] = [0, 0, 142]
    pallete_raw[14, :] = [0, 0, 70]
    pallete_raw[15, :] = [0, 60, 100]
    pallete_raw[16, :] = [0, 80, 100]
    pallete_raw[17, :] = [0, 0, 230]
    pallete_raw[18, :] = [119, 11, 32]
    pallete_raw[255, :] = [224, 224, 224]

    pallete_raw = pallete_raw.reshape(-1)
    gt_color.putpalette(pallete_raw)
    gt_color = np.array(gt_color.convert('RGB'))

    im_blending = cv2.addWeighted(im, 0.5, gt_color, 0.5, 0)
    # plt.subplot(1, 3, 1)
    # plt.imshow(im)
    # plt.subplot(1, 3, 2)
    # plt.imshow(gt_color)
    # plt.subplot(1, 3, 3)
    plt.imshow(im_blending)
    plt.show()


def showGt(gt):

    # print gt.shape
    # gt = gt.setmode('P');
    # n = num_cls
    gt = Image.fromarray(gt, mode='P')
    pallete_raw = np.zeros((256, 3)).astype('uint8')
    # pallete = np.zeros((n, 3)).astype('uint8')

    pallete_raw[0, :] = [128, 64, 128]
    pallete_raw[1, :] = [244, 35, 232]
    pallete_raw[2, :] = [70, 70, 70]
    pallete_raw[3, :] = [102, 102, 156]
    pallete_raw[4, :] = [190, 153, 153]
    pallete_raw[5, :] = [153, 153, 153]
    pallete_raw[6, :] = [250, 170, 30]
    pallete_raw[7, :] = [220, 220, 0]
    pallete_raw[8, :] = [107, 142, 35]
    pallete_raw[9, :] = [152, 251, 152]
    pallete_raw[10, :] = [70, 130, 180]
    pallete_raw[11, :] = [220, 20, 60]
    pallete_raw[12, :] = [255, 0, 0]
    pallete_raw[13, :] = [0, 0, 142]
    pallete_raw[14, :] = [0, 0, 70]
    pallete_raw[15, :] = [0, 60, 100]
    pallete_raw[16, :] = [0, 80, 100]
    pallete_raw[17, :] = [0, 0, 230]
    pallete_raw[18, :] = [119, 11, 32]
    pallete_raw[255, :] = [224, 224, 224]

    pallete_raw = pallete_raw.reshape(-1)

    gt.putpalette(pallete_raw)
    gt_color = np.array(gt.convert('RGB'))
    # plt.imshow(im)
    # plt.subplot(1, 3, 2)
    # plt.imshow(gt_color)
    # plt.subplot(1, 3, 3)
    plt.imshow(gt_color)
    plt.show()

def resize_softmax_output_one_channel(target_size,softmax_one_channel):
    """
    :param target_size: [target_height,target_width] 
    :param softmax_one: [H,W] two channel 
    :return: 
    """
    softmax_one_channel_result = np.array(Image.fromarray(softmax_one_channel).
                                         resize((target_size[1], target_size[0]),
                                                Image.CUBIC))

    return softmax_one_channel_result

def resize_an_softmax_output(target_size, softmax_output, num_threads=4):
    """
    :param target_size: [imh,imw] 
    :param softmax_output: [num_class,imh,imw]
    :param num_threads: 
    :return: 
    """
    worker = partial(resize_softmax_output_one_channel, target_size)

    if num_threads == 1:
        ret = [worker(_) for _ in softmax_output]
    else:
        pool = Pool(num_threads)
        ret = pool.map(worker, [_ for _ in softmax_output])
        pool.close()
        result = [np.expand_dims(np.array(_),0) for _ in ret ]
    return  tensor_vstack(result)

def resize_batch_softmax_output(softmax_outputs, target_size):
    """
    :param softmax_outputs: [batch, num_class, imh, imw]
    :param target_size: [imh, imw]
    :return: softmax_outputs: [batch, num_class, target_size[0], target_size[1]]
    """

    softmax_resize_outputs = []
    for softmax_output in softmax_outputs:
        softmax_resize_outputs.append(np.expand_dims(resize_an_softmax_output(target_size=target_size, softmax_output=softmax_output),0))

    return tensor_vstack(softmax_resize_outputs)

def resize_seg_pillow_target(im,target_w,target_h):
    Image.fromarray(im.astype(np.uint8,copy=False))

def resize_seg_target(im,target_w,target_h,stride):

    im, _ = resize(im,target_w,target_h,interpolation=cv2.INTER_NEAREST)
    im = im[::stride,::stride]

    assert im.shape[0]== target_h//stride and im.shape[1] == target_w//stride
    return im

def resize_target(im, target_w, target_h,interpolation = cv2.INTER_LINEAR):
    """
    :param im: image 
    :param target_w:  target_w
    :param target_h:  target_h
    :param interpolation:  interpolation: if given, 
                    using given interpolation method to resize image
    :return: 
    """
    return cv2.resize(im,(target_w,target_h),None,interpolation = interpolation)

def resize_im_target(im,target_w,target_h, use_random=False):
    """
    :param im:  origin image 
    :param target_w: target w 
    :param target_h:  target h
    :return: 
    """
    origin_h,origin_w = im.shape[:2]
    if origin_h > target_h and origin_w>target_w:
        interpolation = cv2.INTER_AREA
    elif origin_h < target_h and origin_w<target_w:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_LINEAR

    if use_random:
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, \
                              cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interpolation = random.choice(interp_methods)

    return resize_target(im, target_w, target_h,interpolation)


def resize_seg(im, target_size, max_size, stride=0):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = np.array(cv2.resize(im, None, None, fx=im_scale, fy=im_scale,interpolation=cv2.INTER_NEAREST))
    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        padded_im = np.ones((im_height, im_width),np.uint8)* 255
        padded_im[:im.shape[0], :im.shape[1]] = im
        return padded_im, im_scale


def resize(im, target_size, max_size, stride=0, interpolation = cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = np.array(cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation))

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel)).astype(np.uint8)
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale


def resize_one_target(im, target_w, target_h, interpolation=cv2.INTER_LINEAR):
    im_tensor = np.zeros((im.shape[1], im.shape[2], 3))
    for i in range(3):
        im_tensor[:, :, i] = im[i, :, :]
    im_tensor = cv2.resize(im_tensor, (target_w,target_h), None, interpolation=interpolation)
    im = np.zeros((3,im_tensor.shape[0], im_tensor.shape[1]))
    for i in range(3):
        im[i, :, :] = im_tensor[:, :,i]
    return im

def resize_batch_target(im_tensor,target_w,target_h, interpolation=cv2.INTER_LINEAR):
    res = []
    for im in im_tensor:
        res.append(np.expand_dims(resize_one_target(im, target_w, target_h, interpolation), 0))
    res = tensor_vstack(res)
    return res


def rotation(im,target_degree,interpolation=cv2.INTER_LINEAR,fixed_scale=True,borderValue=(255,255,255)):
    height,width = im.shape[:2]

    if not fixed_scale:
        if target_degree%180 == 0:
            scale = 1
        elif target_degree%90 == 0:
            scale = float(max(height, width)) / min(height, width)
        else:
            scale = math.sqrt(pow(height,2)+pow(width,2))/min(height, width)
    else:
        scale = 1

    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), target_degree, scale)
    im = cv2.warpAffine(im,rotateMat,(width,height),flags=interpolation,borderValue=borderValue)

    return im, scale

def flip(im):
    if len(im.shape)==2:
        return im[:,::-1]
    elif len(im.shape)==3:
        return im[:,::-1,:]
    else:
        return NotImplementedError

def transform(im, pixel_means,color_scale=0,pixel_stds = None):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    if color_scale > 0:
        im = im * color_scale
    im = im - pixel_means

    if pixel_stds is not None:
        im /= pixel_stds

    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i]
    return im_tensor

def transform_seg_gt(gt):
    """
    transform segmentation gt image into mxnet tensor
    :param gt: [height, width, channel = 1]
    :return: [channel = 1, height, width]
    """
    gt_tensor = np.zeros((1, 1, gt.shape[0], gt.shape[1]))
    gt_tensor[0, 0, :, :] = gt[:, :]

    return gt_tensor

def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im

def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)

    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)

    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor

def generate_metric_label(label,skip_step=1):

    label_batchsize,label_channel,label_h,label_w = label.shape
    anchor_image = np.full((label_batchsize, label_channel, label_h+2*skip_step,label_w+2*skip_step),-1)
    anchor_image[:,:,skip_step:skip_step+label_h,skip_step:skip_step+label_w] = label
    metric_labels = []
    for ix in xrange(-1,2,1):
        for iy in xrange(-1,2,1):
            label_image = np.full((label_batchsize, label_channel, label_h + 2*skip_step, label_w + 2*skip_step), -1)
            label_image[:,:,skip_step*(1+ix):skip_step*(1+ix)+label_h,skip_step*(1+iy):skip_step*(1+iy)+label_w] = label
            # metric_label = ((label_image == anchor_image)*((label_image != 255)*(anchor_image != 255)))
            metric_label = (label_image == anchor_image)
            metric_label = metric_label[:,:,skip_step*(1+ix):skip_step*(1+ix)+label_h,
                           skip_step*(1+iy):skip_step*(1+iy)+label_w]
            metric_labels.append(metric_label.astype(np.uint8))
            if Debug:
                print ix,iy
                print metric_label.shape
                for one_metric_label,one_label_image,one_anchor_image in zip(metric_label,label_image,anchor_image):
                    plot_border(one_metric_label.astype(np.uint8),one_label_image,one_anchor_image)

    result = np.stack(metric_labels,1)
    return result


def border_ignore_label(label,ignore_size,pad_value= 255.0):

    label_batchsize, label_channel, label_h, label_w = label.shape
    result_label = np.full((label_batchsize, label_channel, label_h, label_w), pad_value)

    result_label[:,:,ignore_size:label_h-ignore_size,
    ignore_size:label_w-ignore_size]= label[:,:,ignore_size:label_h-ignore_size,
    ignore_size:label_w-ignore_size]
    return result_label

def plot_border(metric_label,one_label_image,one_anchor_image):

    pallete_raw = np.zeros((256, 3)).astype('uint8')
    pallete_raw[0, :] = [128, 64, 128]
    pallete_raw[1, :] = [244, 35, 232]
    pallete_raw[2, :] = [70, 70, 70]
    pallete_raw[3, :] = [102, 102, 156]
    pallete_raw[4, :] = [190, 153, 153]
    pallete_raw[5, :] = [153, 153, 153]
    pallete_raw[6, :] = [250, 170, 30]
    pallete_raw[7, :] = [220, 220, 0]
    pallete_raw[8, :] = [107, 142, 35]
    pallete_raw[9, :] = [152, 251, 152]
    pallete_raw[10, :] = [70, 130, 180]
    pallete_raw[11, :] = [220, 20, 60]
    pallete_raw[12, :] = [255, 0, 0]
    pallete_raw[13, :] = [0, 0, 142]
    pallete_raw[14, :] = [0, 0, 70]
    pallete_raw[15, :] = [0, 60, 100]
    pallete_raw[16, :] = [0, 80, 100]
    pallete_raw[17, :] = [0, 0, 230]
    pallete_raw[18, :] = [119, 11, 32]
    pallete_raw[255, :] = [224, 224, 224]
    plt.subplot(131)
    plt.imshow(np.squeeze(metric_label))
    plt.subplot(132)
    plt.imshow(pallete_raw[np.squeeze(one_label_image)])
    plt.subplot(133)
    plt.imshow(pallete_raw[np.squeeze(one_anchor_image)])
    plt.show()