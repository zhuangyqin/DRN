#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: online_evaluation.py
@time: 2017/9/27 14:46
"""
import time
import numpy as np

class ScoreUpdater(object):
    def __init__(self, valid_labels, c_num, x_num, logger=None, label=None, info=None):
        self._valid_labels = valid_labels
        self._confs = np.zeros((c_num, c_num, x_num))
        self._pixels = np.zeros((c_num, x_num))
        self._logger = logger
        self._label = label
        self._info = info

    @property
    def info(self):
        return self._info

    def reset(self):
        self._start = time.time()
        self._computed = np.zeros((self._pixels.shape[1],))
        self._confs[:] = 0
        self._pixels[:] = 0

    @staticmethod
    def calc_updates(valid_labels, pred_label, label):
        num_classes = len(valid_labels)

        pred_flags = [set(np.where((pred_label == _).ravel())[0]) for _ in valid_labels]
        class_flags = [set(np.where((label == _).ravel())[0]) for _ in valid_labels]

        conf = [len(class_flags[j].intersection(pred_flags[k])) for j in xrange(num_classes) for k in xrange(num_classes)]
        pixel = [len(class_flags[j]) for j in xrange(num_classes)]
        return np.single(conf).reshape((num_classes, num_classes)), np.single(pixel)

    def do_updates(self, conf, pixel, i, computed=True):
        if computed:
            self._computed[i] = 1
        self._confs[:, :, i] = conf
        self._pixels[:, i] = pixel

    def update(self, pred_label, label, i, computed=True):
        conf, pixel = ScoreUpdater.calc_updates(self._valid_labels, pred_label, label)
        self.do_updates(conf, pixel, i, computed)
        self.scores(i)

    def scores(self, i=None, logger=None):
        confs = self._confs
        pixels = self._pixels

        num_classes = pixels.shape[0]
        x_num = pixels.shape[1]

        class_pixels = pixels.sum(1)
        class_pixels += class_pixels == 0
        scores = confs[xrange(num_classes), xrange(num_classes), :].sum(1)
        acc = scores.sum() / pixels.sum()
        cls_accs = scores / class_pixels
        class_preds = confs.sum(0).sum(1)
        ious = scores / (class_pixels + class_preds - scores)

        logger = self._logger if logger is None else logger
        if logger is not None:
            logger.info('-------------------------------------------------------')
            if i is not None:
                speed = 1.* self._computed.sum() / (time.time() - self._start)
                logger.info('Done {}/{} with speed: {:.2f} samples/s'.format(i + 1, x_num, speed))
            name = '' if self._label is None else '{}, '.format(self._label)
            # set the printoptiobnf
            np.set_printoptions(formatter={'float': '{:5.2f}'.format},linewidth=200)
            logger.info('Average       :{} pixel acc: {:.2f}%, mean acc: {:.2f}%, mean iou: {:.2f}%'. \
                        format(name, acc * 100, cls_accs.mean() * 100, ious.mean() * 100))
            logger.info('Mean class Acc:{}'.format(cls_accs * 100))
            logger.info('Mean class IoU:{}'.format(ious * 100))

        return acc, cls_accs, ious

    def overall_scores(self, logger=None):
        acc, cls_accs, ious = self.scores(None, logger)
        return acc, cls_accs.mean(), ious.mean()