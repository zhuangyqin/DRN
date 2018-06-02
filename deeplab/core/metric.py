# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Zheng Zhang
# --------------------------------------------------------

import mxnet as mx
import numpy as np

def get_fcn_names():
    pred = ['softmax_output','local_softmax_output','global_softmax']
    label = ['label','origin_label']
    return pred, label


# define some metric of center_loss
class CenterLossMetric(mx.metric.EvalMetric):
    def __init__(self,idx):
        super(CenterLossMetric, self).__init__('center_loss_%02d'%idx)
        self.idx = idx

    def update(self, labels, preds):
        self.sum_metric += np.mean(preds[self.idx].asnumpy())
        self.num_inst += 1

class FCNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, show_interval):
        super(FCNLogLossMetric, self).__init__('FCNLogLoss')
        self.show_interval = show_interval
        self.sum_metric = 0
        self.num_inst = 0

    def update(self, labels, preds):
        pred = preds[0]
        label = labels[0]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))

        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != 255)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1*np.log(cls)
        cls_loss = np.sum(cls_loss)

        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

class ImageCrossEntropyLossMetric(mx.metric.EvalMetric):
    def __init__(self,eps=1e-14, name='cross-ImageCrossEntropyLoss'):
        super(ImageCrossEntropyLossMetric, self).__init__("ImageCrossEntropyLoss")
        self.pred, self.label = get_fcn_names()
        self.eps=eps

    def update(self, labels, preds):
        pred = preds[self.pred.index('softmax_output')].asnumpy()
        label = labels[self.label.index('label')].asnumpy()

        assert pred.shape[0]==label.shape[0],"the pred shape do not match the label shape"
        batch_size=pred.shape[0]

        for i in xrange(batch_size):
            pred_i = pred[i,:]

            label_i= np.squeeze(label[i,:])

            #change to one dimsion
            pred_i = pred_i.transpose(1,2,0)

            assert pred_i.shape[0]==label_i.shape[0] and pred_i.shape[1]==label_i.shape[1]

            pred_i = pred_i.reshape((-1,pred_i.shape[2]))
            label_i = label_i.reshape((-1,))

            assert pred_i.shape[0] == label_i.shape[0]

            #remove the  label 255
            label_index = np.where(label_i!=255)[0]

            pred_i = pred_i[label_index,:]
            label_i = label_i[label_index]

            prob=pred_i[np.arange(label_i.shape[0]), np.int64(label_i)]

            loss = (-np.log(prob + self.eps)).sum()


            self.sum_metric += loss
            self.num_inst += label_i.shape[0]

class LocalImageCrossEntropyLossMetric(mx.metric.EvalMetric):
    def __init__(self,eps=1e-14, name='cross-ImageCrossEntropyLoss'):
        super(LocalImageCrossEntropyLossMetric, self).__init__("LocalLoss")
        self.pred, self.label = get_fcn_names()
        self.eps=eps

    def update(self, labels, preds):
        pred = preds[self.pred.index('local_softmax_output')].asnumpy()
        label = labels[self.label.index('label')].asnumpy()

        assert pred.shape[0]==label.shape[0],"the pred shape do not match the label shape"
        batch_size=pred.shape[0]

        for i in xrange(batch_size):
            pred_i = pred[i,:]

            label_i= np.squeeze(label[i,:])

            #change to one dimsion
            pred_i = pred_i.transpose(1,2,0)

            assert pred_i.shape[0]==label_i.shape[0] and pred_i.shape[1]==label_i.shape[1]

            pred_i = pred_i.reshape((-1,pred_i.shape[2]))
            label_i = label_i.reshape((-1,))

            assert pred_i.shape[0] == label_i.shape[0]

            #remove the  label 255
            label_index = np.where(label_i!=255)[0]

            pred_i = pred_i[label_index,:]
            label_i = label_i[label_index]

            prob=pred_i[np.arange(label_i.shape[0]), np.int64(label_i)]

            loss = (-np.log(prob + self.eps)).sum()


            self.sum_metric += loss
            self.num_inst += label_i.shape[0]

class MImageCrossEntropyLossMetric(mx.metric.EvalMetric):
    def __init__(self,which_idx,eps=1e-14):
        super(MImageCrossEntropyLossMetric, self).__init__("LocalLoss-%02d"
                                                           %which_idx)
        self.pred, self.label = get_fcn_names()
        self.eps=eps
        self.which_idx = which_idx

    def update(self, labels, preds):
        pred = preds[self.which_idx].asnumpy()
        label = labels[self.label.index('label')].asnumpy()

        assert pred.shape[0]==label.shape[0],"the pred shape do not match the label shape"
        batch_size=pred.shape[0]

        for i in xrange(batch_size):
            pred_i = pred[i,:]

            label_i= np.squeeze(label[i,:])

            #change to one dimsion
            pred_i = pred_i.transpose(1,2,0)

            assert pred_i.shape[0]==label_i.shape[0] and pred_i.shape[1]==label_i.shape[1]

            pred_i = pred_i.reshape((-1,pred_i.shape[2]))
            label_i = label_i.reshape((-1,))

            assert pred_i.shape[0] == label_i.shape[0]

            #remove the  label 255
            label_index = np.where(label_i!=255)[0]

            pred_i = pred_i[label_index,:]
            label_i = label_i[label_index]

            prob=pred_i[np.arange(label_i.shape[0]), np.int64(label_i)]

            loss = (-np.log(prob + self.eps)).sum()


            self.sum_metric += loss
            self.num_inst += label_i.shape[0]

class GlobalImageCrossEntropyLossMetric(mx.metric.EvalMetric):
    def __init__(self,eps=1e-14, name='cross-ImageCrossEntropyLoss'):
        super(GlobalImageCrossEntropyLossMetric, self).__init__("GlobalLoss")
        self.pred, self.label = get_fcn_names()
        self.eps=eps

    def update(self, labels, preds):
        pred = preds[self.pred.index('global_softmax')].asnumpy()
        label = labels[self.label.index('origin_label')].asnumpy()

        assert pred.shape[0]==label.shape[0],"the pred shape do not match the label shape"
        batch_size=pred.shape[0]

        for i in xrange(batch_size):
            pred_i = pred[i,:]

            label_i= np.squeeze(label[i,:])

            #change to one dimsion
            pred_i = pred_i.transpose(1,2,0)

            assert pred_i.shape[0]==label_i.shape[0] and pred_i.shape[1]==label_i.shape[1]

            pred_i = pred_i.reshape((-1,pred_i.shape[2]))
            label_i = label_i.reshape((-1,))

            assert pred_i.shape[0] == label_i.shape[0]

            #remove the  label 255
            label_index = np.where(label_i!=255)[0]

            pred_i = pred_i[label_index,:]
            label_i = label_i[label_index]

            prob=pred_i[np.arange(label_i.shape[0]), np.int64(label_i)]

            loss = (-np.log(prob + self.eps)).sum()


            self.sum_metric += loss
            self.num_inst += label_i.shape[0]


class MetricLossMetric(mx.metric.EvalMetric):

    def __init__(self,index):
        super(MetricLossMetric,self).__init__("MetricLoss")
        self.pred,self.label=get_fcn_names()
        self.index = index

    def update(self, labels, preds):

        pred = preds[self.index].asnumpy()
        metric_label = labels[1].asnumpy()
        # print metric_label.shape
        # print np.bincount(metric_label.flatten().astype(np.int64))
        sum_metric = np.sum(pred)
        num_inst = pred.size
        # print "sum_metric:", sum_metric, "num_inst:", num_inst
        self.sum_metric += sum_metric
        self.num_inst += num_inst+ (num_inst == 0)


class SigmoidPixcelAccMetric(mx.metric.EvalMetric):

    def __init__(self,index=1):
        self.index = index
        super(SigmoidPixcelAccMetric,self).__init__("BinaryPixcelAcc")
        super(SigmoidPixcelAccMetric,self).__init__("BinaryPixcelAcc")
        self.pred,self.label=get_fcn_names()

    def update(self, labels, preds):

        pred = preds[self.index].asnumpy()
        label = labels[self.index].asnumpy()

        pred = pred.ravel()
        pred_label = pred > 0.5
        gt_label = label.ravel()== 1.0
        sum_metric = (gt_label == pred_label).sum()
        num_inst = label.size

        # print sum_metric,num_inst
        self.sum_metric += sum_metric
        self.num_inst += num_inst+ (num_inst == 0)


class PixcelAccMetric(mx.metric.EvalMetric):

    def __init__(self):
        super(PixcelAccMetric,self).__init__("PixcelAcc")
        self.pred,self.label=get_fcn_names()

    def update(self, labels, preds):

        pred = preds[self.pred.index('softmax_output')].asnumpy()
        label = labels[self.label.index('label')].asnumpy()

        gt_label = label.ravel()
        valid_flag = gt_label != 255
        gt_label = gt_label[valid_flag]
        pred_label = pred.argmax(1).ravel()[valid_flag]

        sum_metric = (gt_label == pred_label).sum()
        num_inst = valid_flag.sum()



        self.sum_metric += sum_metric
        self.num_inst += num_inst+ (num_inst == 0)
        #
        # assert preds.shape[0]==labels.shape[0],"the pred shape do not match the label shape"
        #
        # batch_size=preds.shape[0]
        # num_classes = preds.shape[1]
        #
        # metric = []
        # for i in xrange(batch_size):
        #     pred_i = np.squeeze(preds[i,:]).argmax(axis=0)
        #     label_i= labels[i,:]
        #
        #     pred_flags = [set(np.where((pred_i == _).ravel())[0]) for _ in xrange(num_classes)]
        #     class_flags = [set(np.where((label_i == _).ravel())[0]) for _ in xrange(num_classes)]
        #
        #     conf = np.array([len(class_flags[j].intersection(pred_flags[k]))
        #                      for j in xrange(num_classes) for k in xrange(num_classes)]
        #                     ,np.single).reshape(num_classes,num_classes)
        #
        #     pixel = np.array([len(class_flags[j]) for j in xrange(num_classes)],np.single)
        #
        #     score = conf[xrange(num_classes), xrange(num_classes)]
        #
        #     acc = score.sum() / pixel.sum()
        #
        #     metric.append(acc)




def FcnValidMetric():
    def _eval_func(label, pred):
        gt_label = label.ravel()
        valid_flag = gt_label != 255
        gt_label = gt_label[valid_flag]
        pred_label = pred.argmax(1).ravel()[valid_flag]

        sum_metric = (gt_label == pred_label).sum()
        num_inst = valid_flag.sum()
        return (sum_metric, num_inst + (num_inst == 0))

    return mx.metric.CustomMetric(_eval_func, 'Fcn-Valid',allow_extra_outputs=True)

