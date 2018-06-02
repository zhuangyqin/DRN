# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yueqing.zhuang
# --------------------------------------------------------

import cPickle
import os
import time
import mxnet as mx
import numpy as np
from math import ceil
from mxnet.executor_manager import _split_input_slice
from module import MutableModule
from config.config import config
from utils.image import tensor_vstack, resize_batch_softmax_output,resize_batch_target,transform
from segmentation.online_evaluation import ScoreUpdater
from utils.load_h5py import save_batch_softmax_ouputs

Debug = 1
output_median = False

def make_divisible(v, divider):
    return int(np.ceil(float(v) / divider) * divider)

class PredictorOneGPU(object):
    def __init__(self, symbol, data_names, label_names,
                 context_id = mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context_id, max_data_shapes=max_data_shapes)

        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict_patch(self,im_tensor,feature_stride,num_steps,output_name ="softmax"):

        origin_shape = im_tensor.shape
        data_shape = (origin_shape[0], origin_shape[1], make_divisible(origin_shape[2], feature_stride),
                      make_divisible(origin_shape[3], feature_stride))

        canva_data = np.zeros(
            (data_shape[0], data_shape[1], data_shape[2] + feature_stride, data_shape[3] + feature_stride))

        sy = sx = feature_stride // 2
        canva_data[:, :, sy:sy + data_shape[2], sx:sx + data_shape[3]] = resize_batch_target(im_tensor, data_shape[3],
                                                                                             data_shape[2])

        canva_softmax = np.zeros((data_shape[0], data_shape[1], data_shape[2]//feature_stride*num_steps, data_shape[3]//feature_stride*num_steps))

        # prepare the start of the strides
        prediction_stride = feature_stride // num_steps
        sy = sx = prediction_stride // 2 + np.arange(num_steps) * prediction_stride

        for ix in xrange(num_steps):
            for iy in xrange(num_steps):
                input_data = canva_data[:,:,sy[iy]:sy[iy]+data_shape[2],sx[ix]:sx[ix]+data_shape[3]]
                data = [[mx.nd.array(input_data)]]
                data_name = "data"
                provide_data = [[(data_name, input_data.shape)]]
                batch_data = mx.io.DataBatch(data=data,provide_data=provide_data,label=None,provide_label=None)
                self._mod.forward(batch_data, is_train=False)
                result = [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=True))]
                canva_softmax[:,:,iy::num_steps,ix::num_steps] = result[output_name][0][0].asnumpy()

        softmax_output = resize_batch_softmax_output(canva_softmax,origin_shape[2:])
        return np.squeeze(softmax_output)

    def predict(self, imarray, pixel_means, pixel_stds, crop_size=512, color_scale=-1,
                    feature_ratio= 2.0 / 3, num_steps=1, output_name="softmax",feature_stride = 8):

            im_tensor = transform(imarray, pixel_means, color_scale=color_scale, pixel_stds=pixel_stds)
            long_size = max(im_tensor[2:])
            if(long_size<crop_size):
                return self.predict_patch(im_tensor,feature_stride,num_steps,output_name)

class PredictorALLGPU(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):

        if isinstance(context,list):
            self.num_gpus = len(context)
            self.mod_list = [PredictorOneGPU(symbol,data_names,label_names,ctx,max_data_shapes,provide_data,
                                             provide_label,arg_params,aux_params) for ctx in context]

    def predict(self, imarray_list,pixel_means, pixel_stds,crop_size= 512,color_scale=-1,
                feature_ratio = 2.0/3, num_steps = 1, output_name="softmax",feature_stride=8):

        softmax_result = []
        for id, im in enumerate(imarray_list):
            assert len(imarray_list) == self.num_gpus
            softmax_result.append(self.mod_list[id].predict(im, pixel_means, pixel_stds, crop_size, color_scale,
                    feature_ratio, num_steps, output_name, feature_stride))

        return softmax_result




def pred_eval(predictor, test_data, imdb, vis=False, ignore_cache=None, logger=None):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param ignore_cache: ignore the saved cache file
    :param logger: the logger instance
    :return:
    """
    res_file = os.path.join(imdb.result_path, imdb.name + '_segmentations.pkl')

    if output_median:
        output_median_dir = os.path.join(imdb.result_path, 'numpy_output')
        if not os.path.exists(output_median_dir):
            os.makedirs(output_median_dir)

    if os.path.exists(res_file) and not ignore_cache:
        with open(res_file , 'rb') as fid:
            evaluation_results = cPickle.load(fid)
            meanIU = evaluation_results['meanIU']
            IU_array = evaluation_results['IU_array']
            logger.info('evaluate segmentation: \n')
            logger.info('class IU_array:\n')
            logger.info(str(IU_array*100))
            logger.info( 'meanIU:%.5f'%(meanIU*100))
        return

    assert vis or not test_data.shuffle

    if test_data.has_label:
        scorer = ScoreUpdater(np.arange(config.dataset.NUM_CLASSES), config.dataset.NUM_CLASSES,
                              test_data.size, logger)
        scorer.reset()

    all_segmentation_result = [[] for _ in xrange(imdb.num_images)]

    num_steps = config.TEST.num_steps
    use_flipping = config.TEST.use_flipping
    num_class = config.dataset.NUM_CLASSES
    label_stride = config.network.LABEL_STRIDE

    idx = 0
    save_feature = 0
    if config.network.use_metric:
        output_name = "FUSION_softmax_output"
    else:
        output_name = "softmax_output"
    # output_name = "FUSION_softmax_output"
    name_i = 0
    for index,data_batch in enumerate(test_data):

        origin_data_shapes = [
            (data_shape[0][1][0], data_shape[0][1][1], data_shape[0][1][2] , data_shape[0][1][3]) for
            data_shape in data_batch.provide_data]

        softmax_outputs_scales = []
        logger.info("#####################################")

        batch_size = 0
        for data_shape in origin_data_shapes:
            batch_size += data_shape[0]

        softmax_batch_predictions = predictor.predict()


        #3.get the final label prediction and save the softmax to the h5 format
        label_predictions=[]
        for batch_softmax_output in softmax_batch_predictions:
            label_predictions.extend([np.argmax(softmax_output, axis=0) for softmax_output in batch_softmax_output])

        if config.TEST.save_h5py:
            save_batch_softmax_ouputs(imdb.result_path,test_data.segdb[idx:idx++test_data.batch_size - data_batch.pad],softmax_batch_predictions)

        #4.crop the prediction and the ground truth
        label_predictions_new = []
        for j, label_prediction in zip(xrange(len(label_predictions)), label_predictions):
            seg_rec = test_data.segdb[index * test_data.batch_size + j]
            imh, imw = (seg_rec['height'], seg_rec['width'])
            label_prediction = label_prediction[:imh, :imw]
            label_predictions_new.append(label_prediction)
        label_predictions = label_predictions_new

        #5.update the online prediction
        if test_data.has_label:
            labels_gt = [label[0].asnumpy() for label in test_data.label]
            labels_gt = tensor_vstack(labels_gt)
            labels_gt = [label for label in  labels_gt]
            for j,label_prediction,label_gt in zip(xrange(len(label_predictions)),label_predictions,labels_gt):
                seg_rec = test_data.segdb[index * test_data.batch_size + j]
                imh, imw = (seg_rec['height'], seg_rec['width'])
                label_gt = np.squeeze(label_gt[:,:imh, :imw])
                if Debug:
                    print label_prediction.shape, label_gt.shape
                assert label_prediction.shape == label_gt.shape
                scorer.update(pred_label=label_prediction,label=label_gt,i=index*test_data.batch_size+j)

        all_segmentation_result[idx: idx+test_data.batch_size - data_batch.pad] = [output.astype('int8') for output in label_predictions]
        logger.info('Done {}/{}'.format(idx + batch_size, test_data.size))
        idx += test_data.batch_size - data_batch.pad

    #total results
    logger.info('-------------------------------------------------------')
    if test_data.has_label:
        evaluation_results = imdb.evaluate_segmentations(all_segmentation_result)
        with open(res_file, 'wb') as f:
            cPickle.dump(evaluation_results, f, protocol=cPickle.HIGHEST_PROTOCOL)
            meanIU = evaluation_results['meanIU']
            IU_array = evaluation_results['IU_array']
            logger.info('evaluate segmentation:')
            logger.info('class IU_array:')
            logger.info(str(IU_array*100))
            logger.info('meanIU:%.5f' % (meanIU*100))

    else:
        imdb.write_segmentation_result(all_segmentation_result)
        logger.info("write the result done!")