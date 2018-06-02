# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Zheng Zhang
# --------------------------------------------------------

import logging
import time

def sequence_module_checkpoint(mod, prefix, period=1, save_optimizer_states=False):
    """Callback to checkpoint Module to prefix every epoch.

    Parameters
    ----------
    mod : subclass of BaseModule
        The module to checkpoint.
    prefix : str
        The file prefix for this checkpoint.
    period : int
        How many epochs to wait before checkpointing. Defaults to 1.
    save_optimizer_states : bool
        Indicates whether or not to save optimizer states for continued training.

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_end_callback to fit.
    """
    period = int(max(1, period))
    # pylint: disable=unused-argument
    def _callback(iter_no, feature_sym=None,confusion_sym=None, arg=None, aux=None):
        """The checkpoint function."""
        if (iter_no + 1) % period == 0:
            mod.save_checkpoint(prefix, iter_no + 1, save_optimizer_states)
    return _callback


class Speedometer(object):
    def __init__(self, batch_size, frequent=50,auto_reset=False):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset
        self.pre_name_value = dict()

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False


        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                s = ''
                if param.eval_metric is not None:
                    name, value = param.eval_metric.get()
                    s = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-" % (param.epoch, count, speed)
                    for n, v in zip(name, value):
                        if self.auto_reset:
                            # compute the
                            assert self.frequent!=0,"the frequent must not be zero when the auto reset be true"
                            total_value_now = v*count
                            total_value_pre = v*count-self.pre_name_value[n]*(self.last_count)
                            s += "%s=%f,\t" %(n ,
                            (total_value_now-total_value_pre)/ (count -self.last_count))

                            self.pre_name_value[n] = v
                        else:
                            s += "%s=%f,\t" % (n, v)

                else:
                    s = "Iteration [%d]\tBatch [%d]\tSpeed: %.2f samples/sec" % (param.epoch, count, speed)

                logging.info(s)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()
            names, values = param.eval_metric.get()
            for name in names:
                self.pre_name_value[name]=0

        self.last_count = count

class TensorboardCallback(object):

    def __init__(self, log_dir,shared_tensorboard=None, prefix=None):
        self.log_dir = log_dir
        self.prefix = prefix
        if shared_tensorboard==None:
            try:
                from tensorboard.writer import SummaryWriter
                self.summary_writer = SummaryWriter(log_dir)
            except ImportError:
                logging.error('You can install tensorboard via `pip install tensorboard`.')
        else:
            self.summary_writer = shared_tensorboard.summary_writer
        self.index = 0

    def __call__(self, param):
        """Callback to log training speed and metrics in TensorBoard."""
        if param.eval_metric is None:
            return
        name_value = param.eval_metric.get_name_value()
        for name, value in name_value:
            if self.prefix is not None:
                name = '%s-%s' % (self.prefix, name)
            self.summary_writer.add_scalar(name, value,global_step=self.index)

        self.index += 1


class LrCallback(object):

    def __init__(self, log_dir,shared_tensorboard=None, prefix=None):
        self.log_dir = log_dir
        self.prefix = prefix
        if shared_tensorboard==None:
            try:
                from tensorboard.writer import SummaryWriter
                self.summary_writer = SummaryWriter(log_dir)
            except ImportError:
                logging.error('You can install tensorboard via `pip install tensorboard`.')
        else:
            self.summary_writer = shared_tensorboard.summary_writer

    def __call__(self,num_update ,lr):
        """Callback to log training speed and metrics in TensorBoard."""

        name = '%s-lr' % (self.prefix)
        self.summary_writer.add_scalar(name, lr, global_step=num_update)