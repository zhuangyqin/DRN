#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/6 18:06
# @Author  : yueqing.zhuang
# @Site    : 
# @File    : optimizer.py
# @Software: PyCharm

import mxnet as mx
from mxnet.ndarray import zeros, NDArray
import numpy as np
import math

@mx.optimizer.Optimizer.register
class YFOptimizer(mx.optimizer.Optimizer):
  """The YF optimizer built upon SGD optimizer with momentum and weight decay.
  The optimizer updates the weight by::
    state = momentum * state + lr * rescale_grad * clip(grad, clip_gradient) + wd * weight
    weight = weight - state
  For details of the update algorithm see :class:`~mxnet.ndarray.sgd_update` and
  :class:`~mxnet.ndarray.sgd_mom_update`.
  This optimizer accepts the following parameters in addition to those accepted
  by :class:`.Optimizer`.
  Parameters
  ----------
  momentum : float, optional
     The momentum value.
  """
  def __init__(self, momentum=0.0, beta=0.999, curv_win_width=20, zero_bias=True, **kwargs):
    super(YFOptimizer, self).__init__(**kwargs)
    self.momentum = momentum
    self.beta = beta
    self.curv_win_width = 20
    self.zero_bias = zero_bias
    # The following are global states for YF tuner
    self._iter = 0
    # 1. Used in curvature estimation
    self._h_min = 0.0
    self._h_max = 0.0
    self._h_window = np.zeros(curv_win_width)
    # 3. Used in distance to opt. estimation
    self._grad_norm_avg = 0.0
    self._h_avg = 0.0
    self._dist_to_opt_avg = 0.0

  def create_state(self, index, weight):
    # 2. These two state tensors are used in grad variance estimation
    # grad_avg = None
    # grad_avg_squared = None
    momentum = zeros(weight.shape, weight.context, dtype=weight.dtype)
    grad_avg = zeros(weight.shape, weight.context, dtype=weight.dtype)
    grad_avg_squared = zeros(weight.shape, weight.context, dtype=weight.dtype)
    return momentum, grad_avg, grad_avg_squared

  def curvature_range(self, grad):
    curv_win = self._h_window
    beta = self.beta
    grad_norm = mx.ndarray.norm(grad).asscalar()
    curv_win[self._iter % self.curv_win_width] = grad_norm ** 2
    valid_end = min(self.curv_win_width, self._iter + 1)
    self._h_min = beta*self._h_min + (1-beta)*curv_win[:valid_end].min()
    self._h_max = beta*self._h_max + (1-beta)*curv_win[:valid_end].max()
    return self._h_min, self._h_max

  def zero_debias_factor(self):
    if self.zero_bias:
      return 1.0
    return 1.0 - self.beta ** (self._iter + 1)

  def grad_variance(self, grad, state):
    beta = self.beta

    grad_avg = state[1]
    grad_avg_squared = state[2]
    grad_avg[:] = self.beta * grad_avg + (1-self.beta) * grad
    grad_avg_squared[:] = self.beta * grad_avg_squared + (1-beta)* grad * grad

    debias_factor = self.zero_debias_factor()
    grad_norm = mx.ndarray.norm(grad).asscalar()
    grad_var = mx.ndarray.sum(grad_avg*grad_avg) / -(debias_factor**2) + grad_norm / debias_factor
    return grad_var.asscalar()

  def dist_to_opt(self, grad):
    beta = self.beta
    grad_norm = mx.ndarray.norm(grad)
    self._grad_norm_avg = beta*self._grad_norm_avg + (1-beta)*grad_norm
    self._h_avg = beta*self._h_avg + (1-beta)*grad_norm*grad_norm
    self._dist_to_opt_avg = beta*self._dist_to_opt_avg + (1-beta)*self._grad_norm_avg / self._h_avg
    debias_factor = self.zero_debias_factor()
    return self._dist_to_opt_avg.asscalar() / debias_factor

  def single_step_mu_lr(self, C, D, h_min, h_max):
    coef = np.array([-1.0, 3.0, 0.0, 1.0])
    # print C, D, h_min, h_max
    coef[2] = -(3 + D**2 * h_min**2 / 2.0 / C)
    roots = np.roots(coef)
    root = roots[np.logical_and(np.logical_and(np.real(roots) > 0.0,
      np.real(roots) < 1.0), np.imag(roots) < 1e-5) ]
    assert root.size == 1
    dr = h_max / h_min
    mu_t = max(np.real(root)[0]**2, ( (np.sqrt(dr) - 1) / (np.sqrt(dr) + 1) )**2 )
    lr_t = (1.0 - math.sqrt(mu_t)) ** 2 / h_min
    return mu_t, lr_t

  def after_apply(self, index, grad, state, lr, momentum):
    beta = self.beta
    h_min, h_max = self.curvature_range(grad)
    C = self.grad_variance(grad, state)
    D = self.dist_to_opt(grad)
    mu_t, lr_t = self.single_step_mu_lr(C, D, h_min, h_max)
    self.momentum = beta*momentum + (1-beta)*mu_t
    self.set_lr_mult({index: beta*lr+(1-beta)*lr_t})

  def update(self, index, weight, grad, state):
    assert(isinstance(weight, NDArray))
    assert(isinstance(grad, NDArray))
    lr = self._get_lr(index)
    wd = self._get_wd(index)
    self._update_count(index)

    kwargs = {'rescale_grad': self.rescale_grad}
    if self.momentum > 0:
      kwargs['momentum'] = self.momentum
    if self.clip_gradient:
      kwargs['clip_gradient'] = self.clip_gradient

    if state is not None:
      mx.optimizer.sgd_mom_update(weight, grad, state[0], out=weight,
                    lr=lr, wd=wd, **kwargs)
      self.after_apply(index, grad, state, lr, self.momentum)
    else:
      mx.optimizer.sgd_update(weight, grad, out=weight,
                    lr=lr, wd=wd, **kwargs)
    self._iter += 1