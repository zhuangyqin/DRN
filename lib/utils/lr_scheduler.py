# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------


import logging
from mxnet.lr_scheduler import LRScheduler

class LinearLrScheduler(LRScheduler):
    def __init__(self, end_epoch, begin_lr ,end_lr,iteration_per_epoch, begin_epoch=0,frequent=10):
        """
        :param end_epoch: 
        :param begin_epoch:  using it to resume 
        """
        super(ConstLrScheduler, self).__init__()
        self.end_epoch = end_epoch
        self.begin_epoch = begin_epoch
        self.begin_lr = begin_lr
        self.end_lr = end_lr
        self.iteration_per_epoch =iteration_per_epoch
        self.frequent = frequent
        self.pre_updates = -1
        self.type = "Linear"

    def __call__(self, num_update):

        lr_offset = (self.end_lr-self.begin_lr)/(self.end_epoch)*self.begin_epoch + self.begin_lr
        lr = (self.end_lr-self.begin_lr)/(self.end_epoch*self.iteration_per_epoch)*num_update+lr_offset

        # log
        if self.frequency > 0 and num_update % self.frequency == 0 and self.pre_updates != num_update:
            logging.info('%s, Iteration [%d]: learning rate:  %0.5e',self.type, num_update, lr)
            self.pre_updates = num_update

        return lr

class ConstLrScheduler(LRScheduler):

    def __init__(self, end_epoch,lr,begin_epoch=0,frequent=10):
        """
        :param end_epoch: 
        :param begin_epoch:  using it to resume 
        """
        super(ConstLrScheduler, self).__init__()
        self.end_epoch = end_epoch
        self.begin_epoch = begin_epoch
        self.lr = lr
        self.frequent = frequent
        self.pre_updates = -1
        self.type = "const "

    def __call__(self, num_update):
        # log
        lr = self.lr
        if self.frequency > 0 and num_update % self.frequency == 0 and self.pre_updates != num_update:
            logging.info('%s, Iteration [%d]: learning rate:  %0.5e',self.type, num_update, lr)
            self.pre_updates = num_update

        return lr


class LinearWarmupLinearScheduler(LRScheduler):
    """Reduce learning rate linearly

    Assume the weight has been updated by n times, then the learning rate will
    be

    base_lr * (1 - n/iters)

    Parameters
    ----------
    step: int
        schedule learning rate after n updates
    factor: float
        the factor for reducing the learning rate
    """

    def __init__(self, num_updates, warmup=False , warmup_lr=0, warmup_step=0,stop_lr=-1., offset=0,frequency=0):
        super(LinearWarmupLinearScheduler, self).__init__()
        if num_updates < 1:
            raise ValueError('Schedule required max number of updates to be greater than 1 round')
        self.num_updates = num_updates
        self.warmup = warmup
        self.warmup_step = warmup_step
        self.warmup_lr = warmup_lr
        self.stop_lr =stop_lr
        self.offset = offset
        self.frequency = frequency
        self.pre_updates = -1

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """

        now_update = self.offset + num_update

        # NOTE: use while rather than if  (for continuing training via load_epoch)
        if self.warmup and num_update < self.warmup_step:
            lr = float(self.base_lr -self.warmup_lr)/float(self.warmup_step)*num_update +self.warmup_lr
        else:
            if now_update > self.num_updates:
                if self.pre_updates != num_update:
                    print 'Exceeds the number of updates, {} > {}'.format(now_update, self.num_updates)
                    self.pre_updates = num_update
                now_update = self.num_updates

            lr = self.base_lr * (1 - float(now_update) / self.num_updates)
            if  lr < self.stop_lr:
                lr = self.stop_lr

        # log
        if self.frequency > 0 and now_update % self.frequency == 0 and self.pre_updates != now_update:
            logging.info('Update[%d]: Current learning rate is %0.5e',now_update, lr)
            self.pre_updates = now_update

        return lr


class WarmupMultiFactorScheduler(LRScheduler):
    """Reduce learning rate in factor at steps specified in a list

    Assume the weight has been updated by n times, then the learning rate will
    be

    base_lr * factor^(sum((step/n)<=1)) # step is an array

    Parameters
    ----------
    step: list of int
        schedule learning rate after n updates
    factor: float
        the factor for reducing the learning rate
    offset: int
        the begin epoch*train.size/train.batchsize 
        which to begin
    """
    def __init__(self, step, factor=1, warmup=False, warmup_lr=0, warmup_step=0,frequency=0):
        super(WarmupMultiFactorScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0
        self.warmup = warmup
        self.warmup_lr = warmup_lr
        self.warmup_step = warmup_step
        self.frequency = frequency
        self.pre_updates = -1

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """

        # NOTE: use while rather than if  (for continuing training via load_epoch)
        if self.warmup and num_update < self.warmup_step:
            lr  = self.warmup_lr
        else:
            while self.cur_step_ind <= len(self.step)-1:
                if num_update > self.step[self.cur_step_ind]:
                    self.count = self.step[self.cur_step_ind]
                    self.cur_step_ind += 1
                    self.base_lr *= self.factor
                    logging.info("Update[%d]: Change learning rate to %0.5e",num_update, self.base_lr)
                    break
                else:
                    lr = self.base_lr
                    break
            lr = self.base_lr
        # log
        if self.frequency > 0 and num_update % self.frequency == 0 and self.pre_updates !=num_update:
            logging.info('Iter[%d]: learning rate:  %0.5e',num_update, lr)
            self.pre_updates = num_update

        return lr


class LinearWarmupMultiFactorScheduler(LRScheduler):
    def __init__(self, step ,factor=1, warmup=False, warmup_lr=0,warmup_step=0,frequency=0,lr_callback = None):
        super(LinearWarmupMultiFactorScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0
        self.warmup = warmup
        self.warmup_lr = warmup_lr
        self.warmup_step = warmup_step
        self.frequency = frequency
        self.pre_updates = -1
        self.lr_callback = lr_callback

    def __call__(self, num_update):
        """
               Call to schedule current learning rate

               Parameters
               ----------
               num_update: int
                   the maximal number of updates applied to a weight.
               """

        # NOTE: use while rather than if  (for continuing training via load_epoch)
        if self.warmup and num_update < self.warmup_step:
            lr = float(self.base_lr -self.warmup_lr)/float(self.warmup_step)*num_update +self.warmup_lr
        else:
            while self.cur_step_ind <= len(self.step) - 1:
                if num_update > self.step[self.cur_step_ind]:
                    self.count = self.step[self.cur_step_ind]
                    self.cur_step_ind += 1
                    self.base_lr *= self.factor
                    logging.info("Update[%d]: Change learning rate to %0.5e", num_update, self.base_lr)
                    break
                else:
                    lr = self.base_lr
                    break
            lr = self.base_lr
        # log
        if self.frequency > 0 and num_update % self.frequency == 0 and self.pre_updates != num_update:
            logging.info('Iter[%d]: learning rate:  %0.5e', num_update, lr)
            if self.lr_callback != None:
                self.lr_callback(num_update,lr)
            self.pre_updates = num_update
        return lr



class LinearWarmupMultiStageScheduler(LRScheduler):
    def __init__(self, step ,factor=1, warmup=False, warmup_lr=0,warmup_step=0,frequency=0,
                 stop_lr=-1.,power = 1, tensorboard=None):
        super(LinearWarmupMultiStageScheduler, self).__init__()
        assert isinstance(step, list) and len(step) == 2,"The step must be two step"
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0
        self.warmup = warmup
        self.warmup_lr = warmup_lr
        self.warmup_step = warmup_step
        self.frequency = frequency
        self.stop_lr =stop_lr
        self.pre_updates = -1
        self.power = power
        self.tensorboard = tensorboard

    def __call__(self, num_update):
        """
        :param num_update: 
        :return: learning rate
        """
        if self.warmup and num_update < self.warmup_step:
            lr = float(self.base_lr -self.warmup_lr)/float(self.warmup_step)*num_update +self.warmup_lr
        else:

            if self.cur_step_ind < 2 and num_update > self.step[self.cur_step_ind]:
                self.cur_step_ind += 1
                logging.info("Update[%d]: Change learning rate! ")

            if self.cur_step_ind == 0:
                lr =self.base_lr
            else:
                lr = self.base_lr * ((1 - float(num_update-self.step[0]) / (self.step[1]-self.step[0])))**self.power
                if lr < self.stop_lr:
                    lr = self.stop_lr
        # log
        if self.frequency > 0 and num_update % self.frequency == 0 and self.pre_updates != num_update:
            logging.info('Iter[%d]: learning rate:  %0.5e', num_update, lr)
            self.pre_updates = num_update

        return lr