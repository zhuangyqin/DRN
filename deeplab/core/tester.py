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
from module import MutableModule
from config.config import config
from utils.image import tensor_vstack, resize_batch_softmax_output,resize_batch_target
from segmentation.online_evaluation import ScoreUpdater
from utils.load_h5py import save_batch_softmax_ouputs

Debug = False
output_median = False
output_median_name = "RNNRelation_ud_relu_output"

class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

    def predict(self, data_batch):
        self._mod.forward(data_batch,is_train=False)
        return [dict(zip(self._mod.output_names, _)) for _ in zip(*self._mod.get_outputs(merge_multi_context=False))]


class Predictor_Emsemble(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params_list=None, aux_params_list=None):
        # assert isinstance(context,List),"the context must be a list "
        assert len(context)==len(arg_params_list),"num of the params must be equal to the arg_params"
        assert len(context)== len(aux_params_list),"num of the params must be equal to the aux_params"

        self._mod_list = []
        for ctx,arg_params,aux_params in zip(context,arg_params_list,aux_params_list):
            mod = MutableModule(symbol, data_names, label_names,
                                      context=[ctx], max_data_shapes=max_data_shapes)
            mod.bind(provide_data, provide_label, for_training=False)
            mod.init_params(arg_params=arg_params, aux_params=aux_params)
            self._mod_list.append(mod)

        self.output_names = self._mod_list[0].output_names


    def predict(self, data_batch,num_stage=1,useflip=False):

        total_results = []

        for mod in self._mod_list:
            mod.forward(data_batch,is_train=False)
            results = [dict(zip(mod.output_names, _)) for _ in zip(*mod.get_outputs(merge_multi_context=False))]
            total_results.append(results)

        avg_results = [dict() for i in xrange(len(results))]

        for output_name in self.output_names:
            for batch_id in xrange(len(results)):
                tmp_results = 0
                for emsemble_id in xrange(len(total_results)):
                    # aggreation in mxnet cpu
                    tmp_results += total_results[emsemble_id][batch_id][output_name].as_in_context(mx.cpu())
                avg_results[batch_id][output_name] = tmp_results/len(total_results)

        return avg_results

def make_divisible(v, divider):
    return int(np.ceil(float(v) / divider) * divider)

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

    if config.dataset.dataset == 'cityscapes':
        feature_stride = label_stride
    else:
        feature_stride = label_stride*4

    idx = 0
    save_feature = 0
    if config.network.use_metric and not config.TRAIN.use_crl_ses:
        output_name = "FUSION_softmax_output"
    else:
        output_name = "softmax_output"
    #output_name = "FUSION_softmax_output"
    name_i = 0
    for index,data_batch in enumerate(test_data):

        origin_data_shapes = [
            (data_shape[0][1][0], data_shape[0][1][1], data_shape[0][1][2] , data_shape[0][1][3]) for
            data_shape in data_batch.provide_data]

        softmax_outputs_scales = []
        logger.info("#####################################")

        for scale in config.TEST.ms_array:
            logger.info("Now Scale: %.2f"%scale)

            test_data.get_batch(scale,True)

            scale_data_shapes = [
                (data_shape[0][1][0], data_shape[0][1][1], data_shape[0][1][2], data_shape[0][1][3]) for
                data_shape in test_data.provide_data]

            data_shapes = [
                (scale_data_shape[0], scale_data_shape[1],
                 make_divisible(scale_data_shape[2],feature_stride),
                 make_divisible(scale_data_shape[3],feature_stride))
                 for scale_data_shape in scale_data_shapes]

            batch_size = 0
            for data_shape in data_shapes:
                batch_size += data_shape[0]

            data_batch.provide_data = [[('data', data_shape)] for data_shape in data_shapes]

            canva_softmax_outputs = [np.zeros((data_shape[0],
                                               num_class,
                                               data_shape[2] // label_stride * num_steps,
                                               data_shape[3] // label_stride * num_steps))
                                               for data_shape in data_shapes]

            canva_datas = [np.zeros((data_shape[0],
                                     data_shape[1],
                                     data_shape[2]+label_stride,
                                     data_shape[3]+label_stride))
                           for data_shape in data_shapes]

            sy = sx = label_stride//2
            for canva_data,origin_data,data_shape in zip(canva_datas,test_data.data,data_shapes):
                canva_data[:,:,sy:sy+data_shape[2],sx:sx+data_shape[3]]= resize_batch_target(origin_data[0].asnumpy(),
                                                                                                         data_shape[3],
                                                                                                         data_shape[2])
            # prepare the start of the strides
            prediction_stride = label_stride // num_steps
            sy = sx = prediction_stride // 2 + np.arange(num_steps)*prediction_stride

            # operation of mult_step
            for ix in xrange(num_steps):
                for iy in xrange(num_steps):
                    data_batch.data = [[mx.nd.array(canva_data[:,:,sy[iy]:sy[iy]+data_shape[2],sx[ix]:sx[ix]+data_shape[3]])]
                                       for canva_data,data_shape in zip(canva_datas,data_shapes)]


                    output_all = predictor.predict(data_batch)
                    softmax_outputs = [output[output_name].asnumpy() for output in output_all]
                    for canva_softmax_output,softmax_output in zip(canva_softmax_outputs,softmax_outputs):
                        canva_softmax_output[:,:,iy::num_steps,ix::num_steps] = softmax_output

                    if use_flipping:
                        data_batch.data = [[mx.nd.array(canva_data[:, :, sy[iy]:sy[iy] + data_shape[2], sx[ix]:sx[ix] + data_shape[3]][:,:,:,::-1])]
                            for canva_data, data_shape in zip(canva_datas, data_shapes)]
                        output_all = predictor.predict(data_batch)
                        softmax_outputs = [output[output_name].asnumpy() for output in output_all]
                        for canva_softmax_output, softmax_output in zip(canva_softmax_outputs, softmax_outputs):
                            canva_softmax_output[:, :, iy::num_steps, ix::num_steps] = 0.5*(canva_softmax_output[:, :, iy::num_steps, ix::num_steps]+
                                                                                            softmax_output[:,:,:,::-1])

            # resize the data inputs and crop the scale inputs
            final_canva_softmax_outputs = [np.zeros((scale_data_shape[0],num_class,scale_data_shape[2],scale_data_shape[3])) for scale_data_shape in scale_data_shapes]
            for data_shape, scale_data_shape, canva_softmax_output,final_canva_softmax_output in zip(data_shapes,scale_data_shapes,
                                                                                                     canva_softmax_outputs,
                                                                                                     final_canva_softmax_outputs):
                final_canva_softmax_output[:,:,:,:] = resize_batch_softmax_output(canva_softmax_output,scale_data_shape[2:])
            softmax_outputs_scales.append(final_canva_softmax_outputs)

        if output_median:
            for zi,output_all_batch in enumerate(output_all):
                output_all_batch_numpy  = output_all_batch[output_median_name].asnumpy()
                for zj, output_numpy_one in enumerate(output_all_batch_numpy):
                    f = file(os.path.join(output_median_dir, output_median_name+'_'+str(name_i)+".npy"), "wb")
                    np.save(f,output_numpy_one)
            name_i +=1
            print test_data.segdb[0]
            print "name", name_i

        #1.resize the data shape
        softmax_outputs_scales_new = []
        for canva_softmax_outputs in softmax_outputs_scales:
            batch_softmax_output_list = []
            for data_shape,batch_softmax_output in zip(origin_data_shapes,canva_softmax_outputs):
                if Debug:
                    print "#1:batch_softmax_output ",batch_softmax_output.shape
                    print "#2:target shape ",data_shape[2:]
                target_size = data_shape[2:]
                batch_softmax_output = resize_batch_softmax_output(batch_softmax_output, target_size)
                batch_softmax_output_list.append(batch_softmax_output)
            softmax_outputs_scales_new.append(batch_softmax_output_list)

        #2.get the avg softmax prediction
        softmax_batch_predictions = [np.zeros((data_shape[0],num_class,data_shape[2],
                                               data_shape[3])) for data_shape in origin_data_shapes]
        for i in xrange(len(softmax_outputs_scales_new)):
            for j in xrange(len(data_batch.provide_data)):
                softmax_batch_predictions[j] += softmax_outputs_scales_new[i][j]

        for j in xrange(len(data_batch.provide_data)):
            softmax_batch_predictions[j] /= float(len(softmax_outputs_scales_new))

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