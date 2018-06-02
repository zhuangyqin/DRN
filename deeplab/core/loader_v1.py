# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Zheng Zhang
# --------------------------------------------------------

import numpy as np
import mxnet as mx
import random
import math

from mxnet.executor_manager import _split_input_slice
from utils.image import tensor_vstack
from segmentation.segmentation import get_segmentation_train_batch, get_segmentation_test_batch,get_segmentation_image_voc
from PIL import Image
from multiprocessing import Pool
from utils.sutil import make_divisible
from utils.image import generate_metric_label,border_ignore_label

class TestDataLoader(mx.io.DataIter):
    def __init__(self, segdb, config, batch_size=1, shuffle=False,ctx=[mx.cpu()],has_label = True):
        super(TestDataLoader, self).__init__()

        # save parameters as properties
        self.segdb = segdb
        last = segdb[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.config = config
        self.ctx = ctx

        # infer properties from roidb
        self.size = len(self.segdb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        if config.network.use_context:
            self.data_name = ['data','origin_data','rois']
        else:
            self.data_name = ['data']
        self.label_name = ['label']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = []
        self.im_info = None
        self.has_label = has_label

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        res = [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]
        return res

    @property
    def provide_label(self):
        return [None for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        this=[(k, v.shape) for k, v in zip(self.data_name, self.data[0])]
        return this

    @property
    def provide_label_single(self):
        return None

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur > self.size:
            return self.cur - self.size
        else:
            return 0

    def get_batch(self, scale=1.0, back=False):
        if back:
            cur_from = self.cur - self.batch_size
        else:
            cur_from = self.cur

        cur_to = min(cur_from + self.batch_size, self.size)
        segdb = [self.segdb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = [1] * len(self.ctx)

        slices = _split_input_slice(self.batch_size, work_load_list)
        rst = []

        for idx, islice in enumerate(slices):
            isegdb = [segdb[i] for i in range(islice.start, islice.stop) if i < len(segdb)]
            if not isegdb==[]:
                rst.append(get_segmentation_test_batch(isegdb,self.config,is_train=False,has_label=self.has_label,scale = scale))

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]

        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]

        if self.has_label:
            self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]
            self.labels_gt = {'label':tensor_vstack([label[key] for key in self.label_name for label in all_label])}



class TrainDataLoader(mx.io.DataIter):
    def __init__(self, sym, segdb, config, batch_size=1,
                 shuffle=False, ctx=None, work_load_list=None,use_context=False,
                 use_mult_label=False,use_metric = False):
        """
        This Iter will provide seg data to Deeplab network
        :param sym: to infer shape
        :param segdb: must be preprocessed
        :param config: config file
        :param batch_size: must divide BATCH_SIZE(128)
        :param crop_height: the height of cropped image
        :param crop_width: the width of cropped image
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :return: DataLoader
        """
        super(TrainDataLoader, self).__init__()

        # save parameters as properties
        self.sym = sym
        self.segdb = segdb
        self.config = config
        self.batch_size = batch_size

        if self.config.TRAIN.enable_crop:
            self.crop_height = config.TRAIN.crop_size[0]
            self.crop_width = config.TRAIN.crop_size[1]
        else:
            self.crop_height = None
            self.crop_width = None

        self.shuffle = shuffle
        self.ctx = ctx
        self.cfg = config

        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list

        # infer properties from segdb
        self.size = len(segdb)
        self.index = np.arange(self.size)

        # decide data and label names
        if use_context:
            self.data_name = ['data','origin_data','rois']
        else:
            self.data_name = ['data']

        if use_mult_label:
            self.label_name = ['label','origin_label']
        elif use_metric:
            scale = config.network.scale_list
            scale_name = ['a', 'b', 'c']
            if config.network.scale_list == [1, 2, 4]:
                scale_name = ['', '', '']
            if config.TRAIN.use_mult_metric:
                self.label_name = ['label', 'metric_label_'+str(scale[0])+scale_name[0],'metric_label_'+str(scale[1])+scale_name[1],'metric_label_'+str(scale[2])+scale_name[2]]
            else:
                self.label_name = ['label','metric_label']
        else:
            self.label_name = ['label']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None
        print "dataset size ",self.size
        # init multi-process pool
        # get first batch to fill in provide_data and provide_label
        self.reset()
        if config.TRAIN.use_thread:
            self.pool = Pool(processes=len(ctx))
            self.get_batch_parallel()
        else:
            self.get_batch()

        random.seed()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            if self.cfg.TRAIN.use_thread:
                self.get_batch_parallel()
            else:
                self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            print "pading",self.cur + self.batch_size - self.size
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []

        max_shapes = dict(max_data_shape + max_label_shape)
        _, label_shape, _ = self.sym.infer_shape(**max_shapes)
        label_shape = [(self.label_name[0], max_shapes['label'])]
        return max_data_shape, label_shape

    def __getitem__(self, item):
        """
        get the i-batch of the train data 
        :param item:  the i-th batch in the train data
        :return: 
        """

        def getindex(cur):
            return cur / self.batch_size

        def getpad(cur):
            if cur + self.batch_size > self.size:
                return cur + self.batch_size - self.size
            else:
                return 0

        cur_from = item*self.batch_size
        cur_to = min(cur_from + self.batch_size, self.size)

        segdb = [self.segdb[self.index[i]] for i in range(cur_from,cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        assert work_load_list==None,"the work_load_list must be the None"

        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        multiprocess_results = []

        for idx, islice in enumerate(slices):
            isegdb = [segdb[i] for i in range(islice.start, islice.stop)]
            multiprocess_results.append(self.pool.apply_async(parfetch, (self.config, self.crop_width, self.crop_height, isegdb)))

        rst = [multiprocess_result.get() for multiprocess_result in multiprocess_results]

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]

        return mx.io.DataBatch(data=[[mx.nd.array(data[key]) for key in self.data_name] for data in all_data],
                               label=[[mx.nd.array(label[key]) for key in self.label_name] for label in all_label],
                               pad=getpad(cur_from), index=getindex(cur_from),
                               provide_data=self.provide_data, provide_label=self.provide_label)

    def get_batch_parallel(self):

        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        segdb = [self.segdb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        multiprocess_results = []

        for idx, islice in enumerate(slices):
            isegdb = [segdb[i] for i in range(islice.start, islice.stop) if i < len(segdb)]
            multiprocess_results.append(self.pool.apply_async(parfetch, (self.config, self.crop_width, self.crop_height, isegdb)))
            # multiprocess_results.append(
            # parfetch(self.config, self.crop_width, self.crop_height, isegdb))

        rst = [multiprocess_result.get() for multiprocess_result in multiprocess_results]

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]

        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]

    def get_batch(self):

        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        segdb = [self.segdb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        rst = []

        for idx, islice in enumerate(slices):
            isegdb = [segdb[i] for i in range(islice.start, islice.stop) if i < len(segdb)]
            rst.append(parfetch(self.config, self.crop_width, self.crop_height, isegdb))
            # multiprocess_results.append(
            # parfetch(self.config, self.crop_width, self.crop_height, isegdb))

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]

        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]


def parfetch(config, crop_width, crop_height, isegdb):
    # get testing data for multigpu

    if config.dataset.dataset == "PascalVOC" or config.dataset.dataset == "ADE20K" :
        datas = {}
        labels = {}
        datas['data'] , labels['label'] = get_segmentation_image_voc(isegdb, config)
        if config.network.use_metric:
            labels['metric_label'] = generate_metric_label(labels['label'])
        if config.TRAIN.use_mult_metric:
            for i in [1,2,4]:
                labels['metric_label_'+str(i)] = generate_metric_label(labels['label'],skip_step=i)

        return {'data': datas, 'label': labels}
    else:
        datas, labels = get_segmentation_train_batch(isegdb, config)
        feature_stride = config.network.LABEL_STRIDE
        network_ratio = config.network.ratio
        if config.TRAIN.enable_crop:
            datas_internal = datas['data']
            labels_internal = labels['label']
            sx = math.floor(random.random() * (datas_internal.shape[3] - crop_width + 1))
            sy = math.floor(random.random() * (datas_internal.shape[2] - crop_height + 1))
            sx = (int)(sx)
            sy = (int)(sy)

            assert(sx >= 0 and sx < datas_internal.shape[3] - crop_width + 1)
            assert(sy >= 0 and sy < datas_internal.shape[2] - crop_height + 1)

            ex = (int)(sx + crop_width - 1)
            ey = (int)(sy + crop_height - 1)

            datas_internal = datas_internal[:, :, sy : ey + 1, sx : ex + 1]
            labels_internal = labels_internal[:, :, sy : ey + 1, sx : ex + 1]

            if config.network.use_crop_context:
                crop_context_scale = config.network.crop_context_scale

                scale_width = make_divisible(int(float(crop_width)/crop_context_scale),feature_stride)
                scale_height = make_divisible(int(float(crop_height)/crop_context_scale),feature_stride)
                pad_width =  int(scale_width- crop_width)/2
                pad_height = int(scale_height- crop_height)/2

                datas['origin_data'] = np.zeros((datas['data'].shape[0],datas['data'].shape[1],
                                                 datas['data'].shape[2]+2*int(pad_height),
                                                 datas['data'].shape[3]+2*int(pad_width)))
                datas['origin_data'][:,:,int(pad_height):datas['data'].shape[2]+int(pad_height),
                int(pad_width):datas['data'].shape[3]+int(pad_width)] = datas['data']

                labels['origin_label'] = np.full((labels['label'].shape[0],
                                                 labels['label'].shape[1],
                                                 labels['label'].shape[2]+2*int(pad_height),
                                                 labels['label'].shape[3]+2*int(pad_width)),255)
                labels['origin_label'][:, :, int(pad_height):labels['label'].shape[2] + int(pad_height),
                int(pad_width):labels['label'].shape[3] + int(pad_width)] = labels['label']

                datas_origin = datas['origin_data'][:, :, sy: sy + scale_height,
                                 sx: sx + scale_width]

                labels_origin = labels['origin_label'][:, :, sy: sy + scale_height,
                                  sx: sx  + scale_width]

                datas['origin_data'] = datas_origin
                labels['origin_label'] = labels_origin

                # labels_origin_in = np.zeros((labels['origin_label'].shape[0],labels['origin_label'].shape[1],
                #                   labels['origin_label'].shape[2]//feature_stride,labels['origin_label'].shape[3]//feature_stride))
                # for i, label in enumerate(labels['origin_label']):
                #     label_im = Image.fromarray(np.squeeze(label.astype(np.uint8, copy=False))).resize(
                #         (labels['origin_label'].shape[3] // feature_stride,
                #          labels['origin_label'].shape[2] // feature_stride), Image.NEAREST)
                #     label = np.array(label_im)
                #     labels_origin_in[i, 0, :, :] = label
                #
                # labels['origin_label']=labels_origin_in


                rois = []
                for i,im_info in zip(xrange(datas_internal.shape[0]),datas['im_info']):
                    rois.append(np.array([i,pad_width,pad_height,pad_width+crop_width,
                                          pad_height+crop_height]).reshape((1,5)))
                datas['rois']= tensor_vstack(rois)
                # print rois

                datas['data'] = datas_internal
                labels['label'] = labels_internal

            else:
                rois = []
                for i,im_info in zip(xrange(datas_internal.shape[0]),datas['im_info']):
                    scale = im_info[2]
                    rois.append(np.array([i,sx*network_ratio/scale,sy*network_ratio/scale,(ex+1)*network_ratio/scale,(ey+1)*network_ratio/scale]).reshape((1,5)))
                datas['rois']= tensor_vstack(rois)

                datas['data'] = datas_internal
                labels['label'] = labels_internal
                assert (datas['data'].shape[2] == crop_height) and (datas['data'].shape[3] == crop_width)
        else:
            datas_internal = datas['data']
            rois = []
            for i,im_info in zip(xrange(datas_internal.shape[0]),datas['im_info']):
                im_size = im_info[:2]
                rois.append(np.array([i, 0, 0, im_size[1]*network_ratio ,im_size[0]*network_ratio]).reshape((1,5)))
            datas['rois'] = tensor_vstack(rois)


        # if feature_stride == 1:
        #     assert (labels['label'].shape[2] == crop_height) and (labels['label'].shape[3] == crop_width)
        # else:

        labels_in = dict()
        labels_in['origin_label']=labels['origin_label']
        labels_in['label']  = np.zeros((labels['label'].shape[0],labels['label'].shape[1],
                              labels['label'].shape[2]//feature_stride,labels['label'].shape[3]//feature_stride))

        # to reshape the label to the network label
        for i,label in enumerate(labels['label']):
            label_im  = Image.fromarray(np.squeeze(label.astype(np.uint8, copy=False))).resize((labels['label'].shape[3]//feature_stride,
                                                                         labels['label'].shape[2]//feature_stride),Image.NEAREST)
            label  = np.array(label_im)
            labels_in['label'][i,0,:,:] = label

        labels = labels_in

        if config.TRAIN.enable_ignore_border:
            labels['label']= border_ignore_label(labels['label'],config.TRAIN.ignore_border_size,255.0)

        if config.network.use_metric:
            labels['metric_label'] = generate_metric_label(labels['label'])

        if config.TRAIN.use_mult_metric:
            scale_name = ['a', 'b', 'c']
            if config.network.scale_list == [1, 2, 4]:
                scale_name = ['', '', '']
            for idx,i in enumerate(config.network.scale_list):
                labels['metric_label_'+str(i)+scale_name[idx]] = generate_metric_label(labels['label'],skip_step=i)

        return {'data': datas, 'label': labels}
