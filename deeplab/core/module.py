# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Zheng Zhang
# --------------------------------------------------------

"""A `MutableModule` implement the `BaseModule` API, and allows input shape
varying with training iterations. If shapes vary, executors will rebind,
using shared arrays from the initial module binded with maximum shape.
"""

import time
import logging
import warnings
import mxnet as mx
from mxnet import context as ctx
from mxnet.initializer import Uniform, InitDesc
from mxnet.module.base_module import BaseModule, _check_input_names, _parse_data_desc, _as_list
from mxnet.model import _create_kvstore, _initialize_kvstore, _update_params, _update_params_on_kvstore, \
    load_checkpoint, BatchEndParam
from mxnet import metric

from .DataParallelExecutorGroup import DataParallelExecutorGroup
from mxnet import ndarray as nd
from mxnet import optimizer as opt
Debug = 0


class Module(BaseModule):
    """Module is a basic module that wrap a `Symbol`. It is functionally the same
    as the `FeedForward` model, except under the module API.

    Parameters
    ----------
    symbol : Symbol
    data_names : list of str
        Default is `('data')` for a typical model used in image classification.
    label_names : list of str
        Default is `('softmax_label')` for a typical model used in image
        classification.
    logger : Logger
        Default is `logging`.
    context : Context or list of Context
        Default is `cpu()`.
    work_load_list : list of number
        Default `None`, indicating uniform workload.
    fixed_param_names: list of str
        Default `None`, indicating no network parameters are fixed.
    state_names : list of str
        states are similar to data and label, but not provided by data iterator.
        Instead they are initialized to 0 and can be set by set_states()
    """

    def __init__(self, symbol, data_names=('data',), label_names=('softmax_label',),
                 logger=logging, context=ctx.cpu(), work_load_list=None,
                 fixed_param_names=None, state_names=None):
        super(Module, self).__init__(logger=logger)

        if isinstance(context, ctx.Context):
            context = [context]
        self._context = context
        if work_load_list is None:
            work_load_list = [1] * len(self._context)
        assert len(work_load_list) == len(self._context)
        self._work_load_list = work_load_list

        self._symbol = symbol

        data_names = list(data_names) if data_names is not None else []
        label_names = list(label_names) if label_names is not None else []
        state_names = list(state_names) if state_names is not None else []
        fixed_param_names = list(fixed_param_names) if fixed_param_names is not None else []

        _check_input_names(symbol, data_names, "data", True)
        _check_input_names(symbol, label_names, "label", False)
        _check_input_names(symbol, state_names, "state", True)
        _check_input_names(symbol, fixed_param_names, "fixed_param", True)

        arg_names = symbol.list_arguments()
        input_names = data_names + label_names + state_names
        self._param_names = [x for x in arg_names if x not in input_names]
        self._fixed_param_names = fixed_param_names
        self._aux_names = symbol.list_auxiliary_states()
        self._data_names = data_names
        self._label_names = label_names
        self._state_names = state_names
        self._output_names = symbol.list_outputs()

        self._arg_params = None
        self._aux_params = None
        self._params_dirty = False

        self._optimizer = None
        self._kvstore = None
        self._update_on_kvstore = None
        self._updater = None
        self._preload_opt_states = None
        self._grad_req = None

        self._exec_group = None
        self._data_shapes = None
        self._label_shapes = None

    @staticmethod
    def load(prefix, epoch, load_optimizer_states=False, **kwargs):
        """Create a model from previously saved checkpoint.

        Parameters
        ----------
        prefix : str
            path prefix of saved model files. You should have
            "prefix-symbol.json", "prefix-xxxx.params", and
            optionally "prefix-xxxx.states", where xxxx is the
            epoch number.
        epoch : int
            epoch to load.
        load_optimizer_states : bool
            whether to load optimizer states. Checkpoint needs
            to have been made with save_optimizer_states=True.
        data_names : list of str
            Default is `('data')` for a typical model used in image classification.
        label_names : list of str
            Default is `('softmax_label')` for a typical model used in image
            classification.
        logger : Logger
            Default is `logging`.
        context : Context or list of Context
            Default is `cpu()`.
        work_load_list : list of number
            Default `None`, indicating uniform workload.
        fixed_param_names: list of str
            Default `None`, indicating no network parameters are fixed.
        """
        sym, args, auxs = load_checkpoint(prefix, epoch)
        mod = Module(symbol=sym, **kwargs)
        mod._arg_params = args
        mod._aux_params = auxs
        mod.params_initialized = True
        if load_optimizer_states:
            mod._preload_opt_states = '%s-%04d.states' % (prefix, epoch)
        return mod

    def save_checkpoint(self, prefix, epoch, save_optimizer_states=False):
        """Save current progress to checkpoint.
        Use mx.callback.module_checkpoint as epoch_end_callback to save during training.

        Parameters
        ----------
        prefix : str
            The file prefix to checkpoint to
        epoch : int
            The current epoch number
        save_optimizer_states : bool
            Whether to save optimizer states for continue training
        """
        self._symbol.save('%s-symbol.json' % prefix)
        param_name = '%s-%04d.params' % (prefix, epoch)
        self.save_params(param_name)
        logging.info('Saved checkpoint to \"%s\"', param_name)
        if save_optimizer_states:
            state_name = '%s-%04d.states' % (prefix, epoch)
            self.save_optimizer_states(state_name)
            logging.info('Saved optimizer state to \"%s\"', state_name)

    def _reset_bind(self):
        """Internal function to reset binded state."""
        self.binded = False
        self._exec_group = None
        self._data_shapes = None
        self._label_shapes = None

    @property
    def data_names(self):
        """A list of names for data required by this module."""
        return self._data_names

    @property
    def label_names(self):
        """A list of names for labels required by this module."""
        return self._label_names

    @property
    def output_names(self):
        """A list of names for the outputs of this module."""
        return self._output_names

    @property
    def data_shapes(self):
        """Get data shapes.
        Returns
        -------
        A list of `(name, shape)` pairs.
        """
        assert self.binded
        return self._data_shapes

    @property
    def label_shapes(self):
        """Get label shapes.
        Returns
        -------
        A list of `(name, shape)` pairs. The return value could be `None` if
        the module does not need labels, or if the module is not binded for
        training (in this case, label information is not available).
        """
        assert self.binded
        return self._label_shapes

    @property
    def output_shapes(self):
        """Get output shapes.
        Returns
        -------
        A list of `(name, shape)` pairs.
        """
        assert self.binded
        return self._exec_group.get_output_shapes()

    def get_params(self):
        """Get current parameters.
        Returns
        -------
        `(arg_params, aux_params)`, each a dictionary of name to parameters (in
        `NDArray`) mapping.
        """
        assert self.binded and self.params_initialized

        if self._params_dirty:
            self._sync_params_from_devices()
        return (self._arg_params, self._aux_params)


    def save_params(self, fname):
        """Saves model parameters to file.

        Parameters
        ----------
        fname : str
            Path to output param file.

        Examples
        --------
        >>> # An example of saving module parameters.
        >>> mod.save_params('myfile')
        """
        arg_params, aux_params = self.get_params()
        save_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
        save_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
        mx.ndarray.save(fname, save_dict)

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False,allow_extra=False):
        """Initialize the parameters and auxiliary states.

        Parameters
        ----------
        initializer : Initializer
            Called to initialize parameters if needed.
        arg_params : dict
            If not None, should be a dictionary of existing arg_params. Initialization
            will be copied from that.
        aux_params : dict
            If not None, should be a dictionary of existing aux_params. Initialization
            will be copied from that.
        allow_missing : bool
            If true, params could contain missing values, and the initializer will be
            called to fill those missing params.
        force_init : bool
            If true, will force re-initialize even if already initialized.
        """
        if self.params_initialized and not force_init:
            warnings.warn("Parameters already initialized and force_init=False. "
                          "init_params call ignored.", stacklevel=2)
            return
        assert self.binded, 'call bind before initializing the parameters'

        def _impl(name, arr, cache):
            """Internal helper for parameter initialization"""
            if cache is not None:
                if name in cache:
                    cache_arr = cache[name]

                    # just in case the cached array is just the target itself
                    if cache_arr is not arr:
                        cache_arr.copyto(arr)
                else:
                    if not allow_missing:
                        raise RuntimeError("%s is not presented" % name)
                    if initializer != None:
                        initializer(name, arr)
            else:
                initializer(name, arr)

        attrs = self._symbol.attr_dict()
        for name, arr in self._arg_params.items():
            desc = InitDesc(name, attrs.get(name, None))
            _impl(desc, arr, arg_params)

        for name, arr in self._aux_params.items():
            desc = InitDesc(name, attrs.get(name, None))
            _impl(desc, arr, aux_params)

        self.params_initialized = True
        self._params_dirty = False

        # copy the initialized parameters to devices
        self._exec_group.set_params(self._arg_params, self._aux_params,allow_extra=allow_extra)

    def set_params(self, arg_params, aux_params, allow_missing=False, force_init=True
                   ,allow_extra=False):
        """Assign parameter and aux state values.

        Parameters
        ----------
        arg_params : dict
            Dictionary of name to value (`NDArray`) mapping.
        aux_params : dict
            Dictionary of name to value (`NDArray`) mapping.
        allow_missing : bool
            If true, params could contain missing values, and the initializer will be
            called to fill those missing params.
        force_init : bool
            If true, will force re-initialize even if already initialized.

        Examples
        --------
        An example of setting module parameters::
            >>> sym, arg_params, aux_params = \
            >>>     mx.model.load_checkpoint(model_prefix, n_epoch_load)
            >>> mod.set_params(arg_params=arg_params, aux_params=aux_params)
        """
        if not allow_missing:
            self.init_params(initializer=None, arg_params=arg_params, aux_params=aux_params,
                             allow_missing=allow_missing, force_init=force_init,allow_extra=allow_extra)
            return

        if self.params_initialized and not force_init:
            warnings.warn("Parameters already initialized and force_init=False. "
                          "set_params call ignored.", stacklevel=2)
            return

        self._exec_group.set_params(arg_params, aux_params,allow_extra=allow_extra)

        # because we didn't update self._arg_params, they are dirty now.
        self._params_dirty = True
        self.params_initialized = True

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None,
             grad_req='write'):
        """Bind the symbols to construct executors. This is necessary before one
        can perform computation with the module.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is `data_iter.provide_data`.
        label_shapes : list of (str, tuple)
            Typically is `data_iter.provide_label`.
        for_training : bool
            Default is `True`. Whether the executors should be bind for training.
        inputs_need_grad : bool
            Default is `False`. Whether the gradients to the input data need to be computed.
            Typically this is not needed. But this might be needed when implementing composition
            of modules.
        force_rebind : bool
            Default is `False`. This function does nothing if the executors are already
            binded. But with this `True`, the executors will be forced to rebind.
        shared_module : Module
            Default is `None`. This is used in bucketing. When not `None`, the shared module
            essentially corresponds to a different bucket -- a module with different symbol
            but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
        """
        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True
        self._grad_req = grad_req

        if not for_training:
            assert not inputs_need_grad
        else:
            pass
            # this is not True, as some module might not contains a loss function
            # that consumes the labels
            # assert label_shapes is not None

        # self._data_shapes, self._label_shapes = _parse_data_desc(
        #     self.data_names, self.label_names, data_shapes, label_shapes)
        self._data_shapes, self._label_shapes = zip(
            *[_parse_data_desc(self.data_names, self.label_names, data_shape, label_shape)
              for data_shape, label_shape in zip(data_shapes, label_shapes)])

        if self._label_shapes.count(None) == len(self._label_shapes):
            self._label_shapes = None

        if shared_module is not None:
            assert isinstance(shared_module, Module) and \
                   shared_module.binded and shared_module.params_initialized
            shared_group = shared_module._exec_group
        else:
            shared_group = None

        self._exec_group = DataParallelExecutorGroup(self._symbol, self._context,
                                                     self._work_load_list, self._data_shapes,
                                                     self._label_shapes, self._param_names,
                                                     for_training, inputs_need_grad,
                                                     shared_group, logger=self.logger,
                                                     fixed_param_names=self._fixed_param_names,
                                                     grad_req=grad_req,
                                                     state_names=self._state_names)
        # self._total_exec_bytes = self._exec_group._total_exec_bytes
        if shared_module is not None:
            self.params_initialized = True
            self._arg_params = shared_module._arg_params
            self._aux_params = shared_module._aux_params
        elif self.params_initialized:
            # if the parameters are already initialized, we are re-binding
            # so automatically copy the already initialized params
            self._exec_group.set_params(self._arg_params, self._aux_params)
        else:
            assert self._arg_params is None and self._aux_params is None
            param_arrays = [
                nd.zeros(x[0].shape, dtype=x[0].dtype)
                for x in self._exec_group.param_arrays
            ]
            self._arg_params = {name: arr for name, arr in zip(self._param_names, param_arrays)}

            aux_arrays = [
                nd.zeros(x[0].shape, dtype=x[0].dtype)
                for x in self._exec_group.aux_arrays
            ]
            self._aux_params = {name: arr for name, arr in zip(self._aux_names, aux_arrays)}

        if shared_module is not None and shared_module.optimizer_initialized:
            self.borrow_optimizer(shared_module)

    def reshape(self, data_shapes, label_shapes=None):
        """Reshape the module for new input shapes.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is `data_iter.provide_data`.
        label_shapes : list of (str, tuple)
            Typically is `data_iter.provide_label`.
        """
        assert self.binded
        # self._data_shapes, self._label_shapes = _parse_data_desc(
        #     self.data_names, self.label_names, data_shapes, label_shapes)
        self._data_shapes, self._label_shapes = zip(
            *[_parse_data_desc(self.data_names, self.label_names, data_shape, label_shape)
              for data_shape, label_shape in zip(data_shapes, label_shapes)])

        self._exec_group.reshape(self._data_shapes, self._label_shapes)

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        """Install and initialize optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default `False`, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert self.binded and self.params_initialized

        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring...')
            return

        (kvstore, update_on_kvstore) = \
            _create_kvstore(kvstore, len(self._context), self._arg_params)

        batch_size = self._exec_group.batch_size
        if kvstore and 'dist' in kvstore.type and '_sync' in kvstore.type:
            batch_size *= kvstore.num_workers
        rescale_grad = 1.0 / batch_size

        if isinstance(optimizer, str):
            idx2name = {}
            if update_on_kvstore:
                idx2name.update(enumerate(self._exec_group.param_names))
            else:
                for k in range(len(self._context)):
                    idx2name.update({i * len(self._context) + k: n
                                     for i, n in enumerate(self._exec_group.param_names)})
            optimizer_params = dict(optimizer_params)
            if 'rescale_grad' not in optimizer_params:
                optimizer_params['rescale_grad'] = rescale_grad
            optimizer = opt.create(optimizer,
                                   sym=self.symbol, param_idx2name=idx2name,
                                   **optimizer_params)
        else:
            assert isinstance(optimizer, opt.Optimizer)
            if optimizer.rescale_grad != rescale_grad:
                # pylint: disable=no-member
                warnings.warn(
                    "Optimizer created manually outside Module but rescale_grad " +
                    "is not normalized to 1.0/batch_size/num_workers (%s vs. %s). " % (
                        optimizer.rescale_grad, rescale_grad) +
                    "Is this intended?", stacklevel=2)

        self._optimizer = optimizer
        self._kvstore = kvstore
        self._update_on_kvstore = update_on_kvstore
        self._updater = None

        if kvstore:
            # copy initialized local parameters to kvstore
            _initialize_kvstore(kvstore=kvstore,
                                param_arrays=self._exec_group.param_arrays,
                                arg_params=self._arg_params,
                                param_names=self._param_names,
                                update_on_kvstore=update_on_kvstore)
        if update_on_kvstore:
            kvstore.set_optimizer(self._optimizer)
        else:
            self._updater = opt.get_updater(optimizer)

        self.optimizer_initialized = True

        if self._preload_opt_states is not None:
            self.load_optimizer_states(self._preload_opt_states)
            self._preload_opt_states = None

    def borrow_optimizer(self, shared_module):
        """Borrow optimizer from a shared module. Used in bucketing, where exactly the same
        optimizer (esp. kvstore) is used.

        Parameters
        ----------
        shared_module : Module
        """
        assert shared_module.optimizer_initialized
        self._optimizer = shared_module._optimizer
        self._kvstore = shared_module._kvstore
        self._update_on_kvstore = shared_module._update_on_kvstore
        self._updater = shared_module._updater
        self.optimizer_initialized = True

    def forward(self, data_batch, is_train=None):
        """Forward computation.

        Parameters
        ----------
        data_batch : DataBatch
            Could be anything with similar API implemented.
        is_train : bool
            Default is `None`, which means `is_train` takes the value of `self.for_training`.
        """
        assert self.binded and self.params_initialized
        self._exec_group.forward(data_batch, is_train)

    def backward(self, out_grads=None):
        """Backward computation.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.
        """
        assert self.binded and self.params_initialized
        self._exec_group.backward(out_grads=out_grads)

    def update(self):
        """Update parameters according to the installed optimizer and the gradients computed
        in the previous forward-backward batch.
        """
        assert self.binded and self.params_initialized and self.optimizer_initialized

        self._params_dirty = True
        if self._update_on_kvstore:
            _update_params_on_kvstore(self._exec_group.param_arrays,
                                      self._exec_group.grad_arrays,
                                      self._kvstore,
                                      self._param_names)
        else:
            _update_params(self._exec_group.param_arrays,
                           self._exec_group.grad_arrays,
                           updater=self._updater,
                           num_device=len(self._context),
                           kvstore=self._kvstore)

    def get_outputs(self, merge_multi_context=True):
        """Get outputs of the previous forward computation.

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is `True`, it is like `[out1, out2]`. Otherwise, it
        is like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`. All the output
        elements are `NDArray`.
        """
        assert self.binded and self.params_initialized
        return self._exec_group.get_outputs(merge_multi_context=merge_multi_context)

    def get_input_grads(self, merge_multi_context=True):
        """Get the gradients with respect to the inputs of the module.

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the outputs
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is `True`, it is like `[grad1, grad2]`. Otherwise, it
        is like `[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]`. All the output
        elements are `NDArray`.
        """
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._exec_group.get_input_grads(merge_multi_context=merge_multi_context)

    def get_states(self, merge_multi_context=True):
        """Get states from all devices

        Parameters
        ----------
        merge_multi_context : bool
            Default is `True`. In the case when data-parallelism is used, the states
            will be collected from multiple devices. A `True` value indicate that we
            should merge the collected results so that they look like from a single
            executor.

        Returns
        -------
        If `merge_multi_context` is `True`, it is like `[out1, out2]`. Otherwise, it
        is like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`. All the output
        elements are `NDArray`.
        """
        assert self.binded and self.params_initialized
        return self._exec_group.get_states(merge_multi_context=merge_multi_context)

    def set_states(self, states=None, value=None):
        """Set value for states. Only one of states & value can be specified.

        Parameters
        ----------
        states : list of list of NDArrays
            source states arrays formatted like [[state1_dev1, state1_dev2],
            [state2_dev1, state2_dev2]].
        value : number
            a single scalar value for all state arrays.
        """
        assert self.binded and self.params_initialized
        self._exec_group.set_states(states, value)

    def update_metric(self, eval_metric, labels):
        """Evaluate and accumulate evaluation metric on outputs of the last forward computation.

        Parameters
        ----------
        eval_metric : EvalMetric
        labels : list of NDArray
            Typically `data_batch.label`.
        """
        self._exec_group.update_metric(eval_metric, labels)

    def _sync_params_from_devices(self):
        """Synchronize parameters from devices to CPU. This function should be called after
        calling `update` that updates the parameters on the devices, before one can read the
        latest parameters from `self._arg_params` and `self._aux_params`.
        """
        self._exec_group.get_params(self._arg_params, self._aux_params)
        self._params_dirty = False

    def save_optimizer_states(self, fname):
        """Save optimizer (updater) state to file

        Parameters
        ----------
        fname : str
            Path to output states file.
        """
        assert self.optimizer_initialized

        if self._update_on_kvstore:
            self._kvstore.save_optimizer_states(fname)
        else:
            with open(fname, 'wb') as fout:
                fout.write(self._updater.get_states())

    def load_optimizer_states(self, fname):
        """Load optimizer (updater) state from file

        Parameters
        ----------
        fname : str
            Path to input states file.
        """
        assert self.optimizer_initialized

        if self._update_on_kvstore:
            self._kvstore.load_optimizer_states(fname)
        else:
            self._updater.set_states(open(fname, 'rb').read())

    def install_monitor(self, mon):
        """ Install monitor on all executors """
        assert self.binded
        self._exec_group.install_monitor(mon)


class MutableModule(BaseModule):
    """A mutable module is a module that supports variable input data.

    Parameters
    ----------
    symbol : Symbol
    data_names : list of str
    label_names : list of str
    logger : Logger
    context : Context or list of Context
    work_load_list : list of number
    max_data_shapes : list of (name, shape) tuple, designating inputs whose shape vary
    max_label_shapes : list of (name, shape) tuple, designating inputs whose shape vary
    fixed_param_prefix : list of str, indicating fixed parameters
    """

    def __init__(self, symbol, data_names, label_names,
                 logger=logging, context=ctx.cpu(), work_load_list=None,
                 max_data_shapes=None, max_label_shapes=None, fixed_param_prefix=None):
        super(MutableModule, self).__init__(logger=logger)
        if isinstance(symbol,mx.symbol.Symbol):
            self._symbol = symbol
            self._sym_gen = None
        else:
            self._symbol = None
            self._sym_gen = symbol
        print self._sym_gen
        self._data_names = data_names
        self._label_names = label_names
        self._context = context
        self._work_load_list = work_load_list

        self._curr_module = None
        self._max_data_shapes = max_data_shapes
        self._max_label_shapes = max_label_shapes
        self._fixed_param_prefix = fixed_param_prefix


        fixed_param_names = list()
        if fixed_param_prefix is not None:
            for name in self._symbol.list_arguments():
                for prefix in self._fixed_param_prefix:
                    if prefix in name:
                        fixed_param_names.append(name)
        self._fixed_param_names = fixed_param_names

        self._aux_params = None
        self._arg_params = None


    def _reset_bind(self):
        self.binded = False
        self._curr_module = None

    @property
    def data_names(self):
        return self._data_names

    @property
    def output_names(self):
        return self._symbol.list_outputs()

    @property
    def data_shapes(self):
        assert self.binded
        return self._curr_module.data_shapes

    @property
    def label_shapes(self):
        assert self.binded
        return self._curr_module.label_shapes

    @property
    def output_shapes(self):
        assert self.binded
        return self._curr_module.output_shapes

    def get_params(self):
        assert self.binded and self.params_initialized
        return self._curr_module.get_params()

    def init_params(self, initializer=Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False,allow_extra=False):
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'
        self._arg_params = arg_params
        self._aux_params = aux_params
        self._curr_module.init_params(initializer=initializer, arg_params=arg_params,
                                      aux_params=aux_params, allow_missing=allow_missing,
                                      force_init=force_init,allow_extra=allow_extra)
        self.params_initialized = True

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None, grad_req='write'):
        # in case we already initialized params, keep it
        if self.params_initialized:
            arg_params, aux_params = self.get_params()

        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        assert shared_module is None, 'shared_module for MutableModule is not supported'

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True

        max_shapes_dict = dict()
        if self._max_data_shapes is not None:
            max_shapes_dict.update(dict(self._max_data_shapes[0]))
        if self._max_label_shapes is not None:
            max_shapes_dict.update(dict(self._max_label_shapes[0]))

        max_data_shapes = list()
        for name, shape in data_shapes[0]:
            if name in max_shapes_dict:
                max_data_shapes.append((name, max_shapes_dict[name]))
            else:
                max_data_shapes.append((name, shape))

        max_label_shapes = list()
        if not label_shapes.count(None) == len(label_shapes):
            for name, shape in label_shapes[0]:
                if name in max_shapes_dict:
                    max_label_shapes.append((name, max_shapes_dict[name]))
                else:
                    max_label_shapes.append((name, shape))

        if len(max_label_shapes) == 0:
            max_label_shapes = None

        if self._sym_gen is None:
            self._symbol = self._symbol
        else:
            self._symbol = self._sym_gen(data_shapes)
        module = Module(self._symbol, self._data_names, self._label_names, logger=self.logger,
                        context=self._context, work_load_list=self._work_load_list,
                        fixed_param_names=self._fixed_param_names)

        module.bind([max_data_shapes for _ in xrange(len(self._context))],
                    [max_label_shapes for _ in xrange(len(self._context))],
                    for_training, inputs_need_grad, force_rebind=False, shared_module=None)

        self._curr_module = module

        # copy back saved params, if already initialized
        if self.params_initialized:
            self.set_params(arg_params, aux_params)

    def save_checkpoint(self, prefix, epoch, save_optimizer_states=False):
        """Save current progress to checkpoint.
        Use mx.callback.module_checkpoint as epoch_end_callback to save during training.

        Parameters
        ----------
        prefix : str
            The file prefix to checkpoint to
        epoch : int
            The current epoch number
        save_optimizer_states : bool
            Whether to save optimizer states for continue training
        """
        self._curr_module.save_checkpoint(prefix, epoch, save_optimizer_states)

    @staticmethod
    def load(prefix, epoch, load_optimizer_states=False, **kwargs):
        """Create a model from previously saved checkpoint.

        Parameters
        ----------
        prefix : str
            path prefix of saved model files. You should have
            "prefix-symbol.json", "prefix-xxxx.params", and
            optionally "prefix-xxxx.states", where xxxx is the
            epoch number.
        epoch : int
            epoch to load.
        load_optimizer_states : bool
            whether to load optimizer states. Checkpoint needs
            to have been made with save_optimizer_states=True.
        data_names : list of str
            Default is `('data')` for a typical model used in image classification.
        label_names : list of str
            Default is `('softmax_label')` for a typical model used in image
            classification.
        logger : Logger
            Default is `logging`.
        context : Context or list of Context
            Default is `cpu()`.
        work_load_list : list of number
            Default `None`, indicating uniform workload.
        fixed_param_names: list of str
            Default `None`, indicating no network parameters are fixed.
        """
        sym, args, auxs = load_checkpoint(prefix, epoch)
        module = MutableModule(symbol=sym, **kwargs)
        module._arg_params = args
        module._aux_params = auxs
        module.params_initialized = True
        if load_optimizer_states:
            module._preload_opt_states = '%s-%04d.states' % (prefix, epoch)
        return module

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        assert self.binded and self.params_initialized
        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring.')
            return

        self._curr_module._preload_opt_states = self._preload_opt_states
        self._curr_module._arg_params = self._arg_params
        self._curr_module._aux_params = self._aux_params
        self._curr_module.init_optimizer(kvstore, optimizer, optimizer_params,
                                         force_init=force_init)
        self.optimizer_initialized = True

    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            epoch_end_metric_callback=None,
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None, prefix=None,allow_extra=False,
            preload_opt_states=None,eval_data_frequency=1):
        """Train the module parameters.

        Parameters
        ----------
        train_data : DataIter
        eval_data : DataIter
            If not `None`, will be used as validation set and evaluate the performance
            after each epoch.
        eval_metric : str or EvalMetric
            Default `'acc'`. The performance measure used to display during training.
        epoch_end_callback : function or list of function
            Each callback will be called with the current `epoch`, `symbol`, `arg_params`
            and `aux_params`.
        batch_end_callback : function or list of function
            Each callback will be called with a `BatchEndParam`.
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The parameters for the optimizer constructor.
            The default value is not a `dict`, just to avoid pylint warning on dangerous
            default values.
        eval_end_callback : function or list of function
            These will be called at the end of each full evaluation, with the metrics over
            the entire evaluation set.
        eval_batch_end_callback : function or list of function
            These will be called at the end of each minibatch during evaluation
        initializer : Initializer
            Will be called to initialize the module parameters if not already initialized.
        arg_params : dict
            Default `None`, if not `None`, should be existing parameters from a trained
            model or loaded from a checkpoint (previously saved model). In this case,
            the value here will be used to initialize the module parameters, unless they
            are already initialized by the user via a call to `init_params` or `fit`.
            `arg_params` has higher priority to `initializer`.
        aux_params : dict
            Default `None`. Similar to `arg_params`, except for auxiliary states.
        allow_missing : bool
            Default `False`. Indicate whether we allow missing parameters when `arg_params`
            and `aux_params` are not `None`. If this is `True`, then the missing parameters
            will be initialized via the `initializer`.
        force_rebind : bool
            Default `False`. Whether to force rebinding the executors if already binded.
        force_init : bool
            Default `False`. Indicate whether we should force initialization even if the
            parameters are already initialized.
        begin_epoch : int
            Default `0`. Indicate the starting epoch. Usually, if we are resuming from a
            checkpoint saved at a previous training phase at epoch N, then we should specify
            this value as N+1.
        num_epoch : int
            Number of epochs to run training.

        Examples
        --------
        An example of using fit for training::
            >>> #Assume training dataIter and validation dataIter are ready
            >>> mod.fit(train_data=train_dataiter, eval_data=val_dataiter,
                        optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
                        num_epoch=10)
        """
        assert num_epoch is not None, 'please specify number of epochs'
        print "train_data",train_data.provide_data
        print "train_data",train_data.provide_label
        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)

        self._preload_opt_states = preload_opt_states
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init,allow_extra=allow_extra)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            for nbatch, data_batch in enumerate(train_data):
                if monitor is not None:
                    monitor.tic()
                self.forward_backward(data_batch)
                self.update()
                self.update_metric(eval_metric, data_batch.label)

                if monitor is not None:
                    monitor.toc_print()

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)

            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            
            if epoch_end_metric_callback is not None:
                epoch_end_params = BatchEndParam(epoch=epoch,nbatch=0,eval_metric=eval_metric,locals=locals)
                for callback in _as_list(epoch_end_metric_callback):
                    callback(epoch_end_params)


            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

            # sync aux params across devices
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params,allow_extra=allow_extra)

            if epoch_end_callback is not None:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            # ----------------------------------------
            # evaluation on validation set
            if eval_data and epoch%eval_data_frequency==0:
                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                # TODO: pull this into default
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()

    def forward(self, data_batch, is_train=None):
        assert self.binded and self.params_initialized

        # get current_shapes
        if self._curr_module.label_shapes is not None:
            current_shapes = [dict(self._curr_module.data_shapes[i] + self._curr_module.label_shapes[i]) for i in
                              xrange(len(self._context))]
        else:
            current_shapes = [dict(self._curr_module.data_shapes[i]) for i in xrange(len(self._curr_module.data_shapes))]

        # get input_shapes
        if is_train or (self._curr_module.label_shapes is not None):
            input_shapes = [dict(data_batch.provide_data[i] + data_batch.provide_label[i]) for i in
                            xrange(len(self._context))]
        else:
            input_shapes = [dict(data_batch.provide_data[i]) for i in xrange(len(data_batch.provide_data))]

        # decide if shape changed
        shape_changed = len(current_shapes) != len(input_shapes)
        for pre, cur in zip(current_shapes, input_shapes):
            for k, v in pre.items():
                if v != cur[k]:
                    shape_changed = True

        if shape_changed:
            if Debug:
                print "change shape"
            # self._curr_module.reshape(data_batch.provide_data, data_batch.provide_label)
            if self._sym_gen is None:
                self._symbol = self._symbol
            else:
                self._symbol = self._sym_gen(data_batch.provide_data)

            module = Module(self._symbol, self._data_names, self._label_names,
                            logger=self.logger, context=self._context,
                            work_load_list=self._work_load_list,
                            fixed_param_names=self._fixed_param_names)
            module.bind(data_batch.provide_data, data_batch.provide_label, self._curr_module.for_training,
                        self._curr_module.inputs_need_grad, force_rebind=False,
                        shared_module=self._curr_module)
            self._curr_module = module

        self._curr_module.forward(data_batch, is_train=is_train)

    def backward(self, out_grads=None):
        assert self.binded and self.params_initialized
        self._curr_module.backward(out_grads=out_grads)

    def update(self):
        assert self.binded and self.params_initialized and self.optimizer_initialized
        self._curr_module.update()

    def get_outputs(self, merge_multi_context=True):
        assert self.binded and self.params_initialized
        return self._curr_module.get_outputs(merge_multi_context=merge_multi_context)

    def get_input_grads(self, merge_multi_context=True):
        assert self.binded and self.params_initialized and self.inputs_need_grad
        return self._curr_module.get_input_grads(merge_multi_context=merge_multi_context)

    def update_metric(self, eval_metric, labels):
        assert self.binded and self.params_initialized
        self._curr_module.update_metric(eval_metric, labels)

    def install_monitor(self, mon):
        """ Install monitor on all executors """
        assert self.binded
        self._curr_module.install_monitor(mon)


class SequenceModule(BaseModule):

    def __init__(self, feature_symbol,confusion_symbol, data_names, label_names,cfg,
                 logger=logging, feature_context=ctx.cpu(),confusion_context= ctx.cpu(),
                 max_data_shapes=None,
                 max_label_shapes=None,
                 fixed_param_prefix=None):
        super(SequenceModule, self).__init__(logger=logger)

        self.feature_symbol = feature_symbol
        self.confusion_symbol = confusion_symbol
        self._data_names = data_names
        self.label_names= label_names
        self.feature_context = feature_context
        self.confusion_context = confusion_context
        self.max_data_shapes=  max_data_shapes
        self.max_label_shapes = max_label_shapes
        self.logger = logger
        self.fixed_param_prefix = fixed_param_prefix
        self.cfg = cfg


    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None, grad_req='write'):
        # in case we already initialized params, keep it
        if self.params_initialized:
            arg_params, aux_params = self.get_params()

        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        assert shared_module is None, 'shared_module for MutableModule is not supported'

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True

        # feature model
        self.feature_model = mx.mod.Module(self.feature_symbol, data_names=self.data_names, label_names=None,
                                           context=self.feature_context)
        self.feature_model.bind(data_shapes=data_shapes, label_shapes=None,for_training=for_training,inputs_need_grad=inputs_need_grad)

        # confusion model data


        if for_training:
            sequence_length = self.cfg.TRAIN.SEQUENCE_LENGTH
            confusion_data_infer_shapes = \
            self.feature_symbol.infer_shape(data=(sequence_length, 3, self.cfg.TRAIN.CROP_HEIGHT, self.cfg.TRAIN.CROP_WIDTH))[
                1]
        else:
            sequence_length = self.cfg.TEST.SEQUENCE_LENGTH
            confusion_data_infer_shapes = \
                self.feature_symbol.infer_shape(
                    data=(sequence_length, 3, self.cfg.SCALES[0][0], self.cfg.SCALES[0][1]))[1]

        self.sequence_length = sequence_length

        confusion_data_outputs_names = self.feature_symbol.list_outputs()

        self.confusion_model = mx.mod.Module(self.confusion_symbol, data_names=self.feature_symbol.list_outputs(),
                                             label_names=self.label_names,
                                             context=self.confusion_context)
        confusion_data_shapes = [(k, (1, v[0], v[1], v[2], v[3])) for k, v in
                                 zip(confusion_data_outputs_names, confusion_data_infer_shapes)]
        # when training it need the top grads
        self.confusion_model.bind(data_shapes=confusion_data_shapes, label_shapes=label_shapes,
                                  inputs_need_grad=for_training,for_training=for_training)

        # copy back saved params, if already initialized
        if self.params_initialized:
            self.set_params(arg_params, aux_params)

    @property
    def data_names(self):
        return self._data_names

    @property
    def output_names(self):
        return self.confusion_symbol.list_outputs()

    @property
    def data_shapes(self):
        assert self.binded
        return self.feature_model.data_shapes

    @property
    def label_shapes(self):
        assert self.binded
        return self.confusion_model.label_shapes

    @property
    def output_shapes(self):
        assert self.binded
        return self.confusion_model.output_shapes

    def get_params(self):
        feature_arg_params,feature_aux_params = self.feature_model.get_params()
        confusion_arg_params,confusion_aux_params = self.confusion_model.get_params()
        arg_params = dict(feature_arg_params.items()+confusion_arg_params.items())
        aux_params = dict(feature_aux_params.items()+confusion_aux_params.items())
        return arg_params,aux_params

    def set_params(self, *args,**kwargs):
        self.feature_model.set_params(*args, **kwargs)
        self.confusion_model.set_params(*args, **kwargs)


    def save_params(self, fname):
        """Saves model parameters to file.

        Parameters
        ----------
        fname : str
            Path to output param file.

        Examples
        --------
        >>> # An example of saving module parameters.
        >>> mod.save_params('myfile')
        """
        arg_params, aux_params = self.get_params()
        save_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
        save_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
        mx.ndarray.save(fname, save_dict)


    def save_optimizer_states(self, fname):
        """Save optimizer (updater) state to file

        Parameters
        ----------
        fname : str
            Path to output states file.
        """
        assert self.optimizer_initialized

        # save the optimizer states
        if self.feature_model._update_on_kvstore :
            feature_model_states =self.feature_model._kvstore._updater.get_states()
        else:
            feature_model_states = self.feature_model._updater.get_states()

        if self.confusion_model._update_on_kvstore:
            confusion_model_states = self.confusion_model._kvstore._updater.get_states()
        else:
            confusion_model_states = self.confusion_model._updater.get_states()

        with open(fname, 'wb') as fout:
            fout.write(feature_model_states + confusion_model_states)


    def save_checkpoint(self, prefix, epoch, save_optimizer_states=False):
        """Save current progress to checkpoint.
        Use mx.callback.module_checkpoint as epoch_end_callback to save during training.

        Parameters
        ----------
        prefix : str
            The file prefix to checkpoint to
        epoch : int
            The current epoch number
        save_optimizer_states : bool
            Whether to save optimizer states for continue training
        """
        self.feature_symbol.save('%s-feature-symbol.json' % prefix)
        self.confusion_symbol.save('%s-confusion-symbol.json' % prefix)
        param_name = '%s-%04d.params' % (prefix, epoch)
        self.save_params(param_name)
        logging.info('Saved checkpoint to \"%s\"', param_name)
        if save_optimizer_states:
            state_name = '%s-%04d.states' % (prefix, epoch)
            self.save_optimizer_states(state_name)
            logging.info('Saved optimizer state to \"%s\"', state_name)


    def init_params(self, *args, **kwargs):
        self.feature_model.init_params(*args, **kwargs)
        self.confusion_model.init_params(*args, **kwargs)
        self.params_initialized = True


    def init_optimizer(self, *args, **kwargs):
        self.feature_model.init_optimizer(*args, **kwargs)
        self.confusion_model.init_optimizer(*args, **kwargs)
        self.optimizer_initialized = True

    def forward_backward(self,data_batch):
        self.forward(data_batch)
        self.backward()


    def forward(self, data_batch, is_train=None):
        feature_data_batch = mx.io.DataBatch(data=data_batch.data,label=None,pad=data_batch.pad,index=data_batch.index,
                                             provide_data=data_batch.provide_data,provide_label=None)
        self.feature_model.forward(data_batch=feature_data_batch,is_train=is_train)
        feature_model_output = [mx.ndarray.expand_dims(self.feature_model.get_outputs()[0],axis=0)]
        confusion_data_batch = mx.io.DataBatch(data=feature_model_output, label=data_batch.label, pad=data_batch.pad,
                                             index=data_batch.index,
                                             provide_data=[(self.feature_symbol.list_outputs()[0],feature_model_output[0].shape)],
                                             provide_label=data_batch.provide_label)
        self.confusion_model.forward(data_batch=confusion_data_batch,is_train=is_train)

    def backward(self):
        self.confusion_model.backward()
        feature_model_input_grads=self.confusion_model.get_input_grads()
        # rescale the grad for the bias on the samples
        feature_model_input_grads = [mx.ndarray.reshape(feature_model_input_grads[0],feature_model_input_grads[0].shape[1:])
                                     *1.0/(self.sequence_length)]
        self.feature_model.backward(feature_model_input_grads)

    def update(self):
        self.feature_model.update()
        self.confusion_model.update()

    def update_metric(self, eval_metric, labels):
        eval_metric.update(labels, self.confusion_model.get_outputs())

    def fit(self, train_data, eval_data=None, eval_metric='acc', epoch_end_callback=None, batch_end_callback=None,
            kvstore='local', optimizer='sgd', optimizer_params=(('learning_rate', 0.01),), eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01), arg_params=None, aux_params=None,
            allow_missing=False, force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None,epoch_end_metric_callback=None,monitor=None):

        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(self.max_data_shapes, label_shapes=self.max_label_shapes,
                  for_training=True, force_rebind=force_rebind)

        if monitor is not None:
            self.install_monitor(monitor)

        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            for nbatch, data_batch in enumerate(train_data):
                self.forward_backward(data_batch)
                self.update()
                self.update_metric(eval_metric=eval_metric,labels=data_batch.label)
                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)

            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)

            if epoch_end_metric_callback is not None:
                epoch_end_params = BatchEndParam(epoch=epoch, nbatch=0, eval_metric=eval_metric, locals=locals)
                for callback in _as_list(epoch_end_metric_callback):
                    callback(epoch_end_params)

            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

            # sync aux params across devices
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)

            if epoch_end_callback is not None:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.feature_symbol,self.confusion_model, arg_params, aux_params)

            # ----------------------------------------
            # evaluation on validation set
            if eval_data:
                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                # TODO: pull this into default
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()

    def get_outputs(self, merge_multi_context=True):
        assert self.binded and self.params_initialized
        return self.confusion_model.get_outputs(merge_multi_context=merge_multi_context)

    def score(self, eval_data, eval_metric, num_batch=None, batch_end_callback=None,
              score_end_callback=None,
              reset=True, epoch=0):
        """Runs prediction on ``eval_data`` and evaluates the performance according to
        the given ``eval_metric``.

        Checkout `Module Tutorial <http://mxnet.io/tutorials/basic/module.html>`_ to see
        a end-to-end use-case.

        Parameters
        ----------
        eval_data : DataIter
            Evaluation data to run prediction on.
        eval_metric : EvalMetric or list of EvalMetrics
            Evaluation metric to use.
        num_batch : int
            Number of batches to run. Defaults to ``None``, indicating run until the `DataIter`
            finishes.
        batch_end_callback : function
            Could also be a list of functions.
        reset : bool
            Defaults to ``True``. Indicates whether we should reset `eval_data` before starting
            evaluating.
        epoch : int
            Defaults to 0. For compatibility, this will be passed to callbacks (if any).
            During training, this will correspond to the training epoch number.

        Examples
        --------
        >>> # An example of using score for prediction.
        >>> # Evaluate accuracy on val_dataiter
        >>> metric = mx.metric.Accuracy()
        >>> mod.score(val_dataiter, metric)
        >>> mod.score(val_dataiter, ['mse', 'acc'])
        """
        assert self.binded and self.params_initialized

        if reset:
            eval_data.reset()

        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        eval_metric.reset()
        actual_num_batch = 0

        for nbatch, eval_batch in enumerate(eval_data):
            if num_batch is not None and nbatch == num_batch:
                break

            self.forward(eval_batch, is_train=False)
            self.update_metric(eval_metric, eval_batch.label)

            if batch_end_callback is not None:
                batch_end_params = BatchEndParam(epoch=epoch,
                                                 nbatch=nbatch,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                for callback in _as_list(batch_end_callback):
                    callback(batch_end_params)
            actual_num_batch += 1

        if score_end_callback:
            params = BatchEndParam(epoch=epoch,
                                   nbatch=actual_num_batch,
                                   eval_metric=eval_metric,
                                   locals=locals())
            for callback in _as_list(score_end_callback):
                callback(params)

        return eval_metric.get_name_value()

    def _reset_bind(self):
        self.binded = False
        self._curr_module = None

    def install_monitor(self,monitor):
        self.feature_model.install_monitor(monitor)
        self.confusion_model.intall_monitor(monitor)


