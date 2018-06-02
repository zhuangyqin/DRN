#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/14 12:46
# @Author  : yueqing.zhuang
# @Site    : 
# @File    : ConvRNNCell.py
# @Software: PyCharm

from mxnet import symbol
from mxnet import ndarray
from mxnet.base import string_types, numeric_types


def _cells_state_shape(cells):
    return sum([c.state_shape for c in cells], [])

def _cells_state_info(cells):
    return sum([c.state_info for c in cells], [])

def _cells_begin_state(cells, **kwargs):
    return sum([c.begin_state(**kwargs) for c in cells], [])

def _cells_unpack_weights(cells, args):
    for cell in cells:
        args = cell.unpack_weights(args)
    return args

def _cells_pack_weights(cells, args):
    for cell in cells:
        args = cell.pack_weights(args)
    return args



def _normalize_sequence(length, inputs, layout, merge, in_layout=None):
    """
    :param length:  the length of the RNN learn
    :param inputs:  list of symbol or a symbol
    :param layout: 
    :param merge:  the out
    :param in_layout: 
    :return: 
    """
    assert inputs is not None, \
        "unroll(inputs=None) has been deprecated. " \
        "Please create input variables outside unroll."

    axis = layout.find('T')
    in_axis = in_layout.find('T') if in_layout is not None else axis
    if isinstance(inputs, symbol.Symbol):
        if merge is False:
            assert len(inputs.list_outputs()) == 1, \
                "unroll doesn't allow grouped symbol as input. Please convert " \
                "to list with list(inputs) first or let unroll handle splitting."
            inputs = list(symbol.split(inputs, axis=in_axis, num_outputs=length,
                                       squeeze_axis=1))
    else:
        assert length is None or len(inputs) == length
        if merge is True:
            inputs = [symbol.expand_dims(i, axis=axis) for i in inputs]
            inputs = symbol.Concat(*inputs, dim=axis)
            in_axis = axis

    if isinstance(inputs, symbol.Symbol) and axis != in_axis:
        inputs = symbol.swapaxes(inputs, dim0=axis, dim1=in_axis)

    return inputs, axis


class ConvRNNParams(object):
    """Container for holding variables.
    Used by RNN cells for parameter sharing between cells.

    Parameters
    ----------
    prefix : str
        All variables' name created by this container will
        be prepended with prefix
    """
    def __init__(self, prefix=''):
        self._prefix = prefix
        self._params = {}

    def get(self, name, **kwargs):
        """Get a variable with name or create a new one if missing.

        Parameters
        ----------
        name : str
            name of the variable
        **kwargs :
            more arguments that's passed to symbol.Variable
        """
        name = self._prefix + name
        if name not in self._params:
            self._params[name] = symbol.Variable(name, **kwargs)
        return self._params[name]

class BaseConvRNNCell(object):
    """Abstract base class for RNN cells

        Parameters
        ----------
        prefix : str
            prefix for name of layers
            (and name of weight if params is None)
        params : RNNParams or None
            container for weight sharing between cells.
            created if None.
        """

    def __init__(self, prefix='', params=None,feat_size=None):
        assert feat_size!=None,"the feat size must be given!"
        if params is None:
            params = ConvRNNParams(prefix)
            self._own_params = True
        else:
            self._own_params = False
        self._prefix = prefix
        self._params = params
        self._modified = False
        self._feat_size=feat_size

        self.reset()

    def reset(self):
        """Reset before re-using the cell for another graph"""
        self._init_counter = -1
        self._counter = -1

    def __call__(self, inputs, states):
        """Construct symbol for one step of RNN.

        Parameters
        ----------
        inputs : sym.Variable
            input symbol, 2D, batch * num_units
        states : sym.Variable
            state from previous step or begin_state().

        Returns
        -------
        output : Symbol
            output symbol
        states : Symbol
            state to next step of RNN.
        """
        raise NotImplementedError()

    @property
    def params(self):
        """Parameters of this cell"""
        self._own_params = False
        return self._params

    @property
    def state_info(self):
        """shape and layout information of states"""
        raise NotImplementedError()

    @property
    def state_shape(self):
        """shape(s) of states"""
        return [ele['shape'] for ele in self.state_info]

    @property
    def _gate_names(self):
        """name(s) of gates"""
        return ()

    def begin_state(self, func=symbol.zeros, **kwargs):
        """Initial state for this cell.

        Parameters
        ----------
        func : callable, default symbol.zeros
            Function for creating initial state. Can be symbol.zeros,
            symbol.uniform, symbol.Variable etc.
            Use symbol.Variable if you want to directly
            feed input as states.
        **kwargs :
            more keyword arguments passed to func. For example
            mean, std, dtype, etc.

        Returns
        -------
        states : nested list of Symbol
            starting states for first RNN step
        """
        assert not self._modified, \
            "After applying modifier cells (e.g. DropoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        states = []
        for info in self.state_info:
            self._init_counter += 1
            if info is None:
                state = func(name='%sbegin_state_%d' % (self._prefix, self._init_counter),
                             **kwargs)
            else:
                kwargs.update(info)
                state = func(name='%sbegin_state_%d' % (self._prefix, self._init_counter),
                             **kwargs)
            states.append(state)
        return states

    def unpack_weights(self, args):
        """Unpack fused weight matrices into separate
        weight matrices

        Parameters
        ----------
        args : dict of str -> NDArray
            dictionary containing packed weights.
            usually from Module.get_output()

        Returns
        -------
        args : dict of str -> NDArray
            dictionary with weights associated to
            this cell unpacked.
        """
        args = args.copy()
        if not self._gate_names:
            return args
        h = self._num_hidden
        for group_name in ['i2h', 'h2h']:
            weight = args.pop('%s%s_weight' % (self._prefix, group_name))
            bias = args.pop('%s%s_bias' % (self._prefix, group_name))
            for j, gate in enumerate(self._gate_names):
                wname = '%s%s%s_weight' % (self._prefix, group_name, gate)
                args[wname] = weight[j * h:(j + 1) * h].copy()
                bname = '%s%s%s_bias' % (self._prefix, group_name, gate)
                args[bname] = bias[j * h:(j + 1) * h].copy()
        return args

    def pack_weights(self, args):
        """Pack separate weight matrices into fused
        weight.

        Parameters
        ----------
        args : dict of str -> NDArray
            dictionary containing unpacked weights.

        Returns
        -------
        args : dict of str -> NDArray
            dictionary with weights associated to
            this cell packed.
        """
        args = args.copy()
        if not self._gate_names:
            return args
        for group_name in ['i2h', 'h2h']:
            weight = []
            bias = []
            for gate in self._gate_names:
                wname = '%s%s%s_weight' % (self._prefix, group_name, gate)
                weight.append(args.pop(wname))
                bname = '%s%s%s_bias' % (self._prefix, group_name, gate)
                bias.append(args.pop(bname))
            args['%s%s_weight' % (self._prefix, group_name)] = ndarray.concatenate(weight)
            args['%s%s_bias' % (self._prefix, group_name)] = ndarray.concatenate(bias)
        return args

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None):
        """Unroll an RNN cell across time steps.

        Parameters
        ----------
        length : int
            number of steps to unroll
        inputs : Symbol, list of Symbol, or None
            if inputs is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.

            If inputs is a list of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...).
        begin_state : nested list of Symbol
            input states. Created by begin_state()
            or output state of another cell. Created
            from begin_state() if None.
        layout : str
            layout of input symbol. Only used if inputs
            is a single Symbol.
        merge_outputs : bool
            If False, return outputs as a list of Symbols.
            If True, concatenate output across time steps
            and return a single symbol with shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.
            If None, output whatever is faster

        Returns
        -------
        outputs : list of Symbol
            output symbols.
        states : Symbol or nested list of Symbol
            has the same structure as begin_state()
        """
        self.reset()

        inputs, _ = _normalize_sequence(length, inputs, layout, False)
        if begin_state is None:
            begin_state = self.begin_state()

        states = begin_state
        outputs = []
        for i in range(length):
            output, states = self(inputs[i], states)
            outputs.append(output)

        outputs, _ = _normalize_sequence(length, outputs, layout, merge_outputs)

        return outputs, states

    # pylint: disable=no-self-use
    def _get_activation(self, inputs, activation, **kwargs):
        """Get activation function. Convert if is string"""
        if isinstance(activation, string_types):
            return symbol.Activation(inputs, act_type=activation, **kwargs)
        else:
            return activation(inputs, **kwargs)



class ConvRNNCell(BaseConvRNNCell):
    """Simple recurrent neural network cell

        Parameters
        ----------
        num_hidden : int
            number of units in output symbol
        activation : str or Symbol, default 'tanh'
            type of activation function
        prefix : str, default 'rnn_'
            prefix for name of layers
            (and name of weight if params is None)
        params : RNNParams or None
            container for weight sharing between cells.
            created if None.
        """

    def __init__(self, num_hidden,feat_size, activation='relu',
                 prefix='rnn_',params=None, stride=(1,1),kernel=(1,1),dilate=(0,0)):
        self._stride = stride
        self._dilate = dilate
        # todo the stride and the dilate not support

        self._kernel = kernel
        super(ConvRNNCell, self).__init__(prefix=prefix, params=params,feat_size=feat_size)
        self._num_hidden = num_hidden
        self._activation = activation
        self._iW = self.params.get('i2h_weight')
        self._iB = self.params.get('i2h_bias')
        self._hW = self.params.get('h2h_weight')
        self._hB = self.params.get('h2h_bias')



    @property
    def state_info(self):
        return [{'shape': (0, self._num_hidden,self._feat_size[0],self._feat_size[1]), '__layout__': 'NC'}]

    @property
    def _gate_names(self):
        return ('',)

    def __call__(self, inputs, states):
        self._counter += 1
        name = '%st%d_' % (self._prefix, self._counter)

        #input
        i2h = symbol.Convolution(data=inputs, weight=self._iW, bias=self._iB,kernel=self._kernel,
                                    num_filter=self._num_hidden,stride=self._stride,name='%si2h' % name)

        #memory
        h2h = symbol.Convolution(data=states[0], weight=self._hW, bias=self._hB,
                                 kernel=self._kernel,num_filter=self._num_hidden,stride=self._stride,name='%sh2h' % name)

        output = self._get_activation(i2h+h2h, self._activation,
                                      name='%sout' % name)

        return output, [output]


class ConvGRUCell(BaseConvRNNCell):

    """Gated Rectified Unit (GRU) network cell.
        Note: this is an implementation of the cuDNN version of GRUs
        (slight modification compared to Cho et al. 2014).

        Parameters
        ----------
        num_hidden : int
            number of units in output symbol
        prefix : str, default 'gru_'
            prefix for name of layers
            (and name of weight if params is None)
        params : RNNParams or None
            container for weight sharing between cells.
            created if None.
        """

    def __init__(self, num_hidden, feat_size , prefix='gru_', params=None,stride=(1,1),
                 dilate=(0,0), kernel=(1,1),pad=(0,0)):
        super(ConvGRUCell, self).__init__(prefix=prefix, params=params,feat_size=feat_size)
        self._num_hidden = num_hidden
        self._stride = stride
        self._dilate = dilate
        self._kernel = kernel
        self._pad = pad

        self._iW = self.params.get("i2h_weight")
        self._iB = self.params.get("i2h_bias")
        self._hW = self.params.get("h2h_weight")
        self._hB = self.params.get("h2h_bias")

    @property
    def state_info(self):
        return [{'shape': (1, self._num_hidden,self._feat_size[0],
                           self._feat_size[1]),
                 '__layout__': 'NC'}]

    @property
    def _gate_names(self):
        return ['_r', '_z', '_o']

    def __call__(self, inputs, states):
        # pylint: disable=too-many-locals
        self._counter += 1

        seq_idx = self._counter
        name = '%st%d_' % (self._prefix, seq_idx)
        prev_state_h = states[0]

        i2h = symbol.Convolution(data=inputs,
                                 weight=self._iW,
                                 kernel  = self._kernel,
                                 bias=self._iB,
                                 num_filter=self._num_hidden * 3,
                                 dilate =self._dilate,
                                 pad = self._pad,
                                 name="%s_i2h" % name)

        h2h = symbol.Convolution(data=prev_state_h,
                                 kernel=self._kernel,
                                 weight=self._hW,
                                 bias=self._hB,
                                 num_filter=self._num_hidden * 3,
                                 dilate=self._dilate,
                                 pad = self._pad,
                                 name="%s_h2h" % name)

        i2h_r, i2h_z, i2h = symbol.SliceChannel(i2h, num_outputs=3, name="%s_i2h_slice" % name)
        h2h_r, h2h_z, h2h = symbol.SliceChannel(h2h, num_outputs=3, name="%s_h2h_slice" % name)

        reset_gate = symbol.Activation(i2h_r + h2h_r, act_type="sigmoid",
                                       name="%s_r_act" % name)
        update_gate = symbol.Activation(i2h_z + h2h_z, act_type="sigmoid",
                                        name="%s_z_act" % name)

        next_h_tmp = symbol.Activation(i2h + reset_gate * h2h, act_type="tanh",
                                       name="%s_h_act" % name)

        next_h = symbol._internal._plus((1. - update_gate) * next_h_tmp, update_gate * prev_state_h,
                                        name='%sout' % name)

        return next_h, [next_h]


class ConvBidirectionalCell(BaseConvRNNCell):
    """Bidirectional RNN cell

    Parameters
    ----------
    l_cell : BaseRNNCell
        cell for forward unrolling
    r_cell : BaseRNNCell
        cell for backward unrolling
    output_prefix : str, default 'bi_'
        prefix for name of output
    add_outputs : for the add the symbol
    """
    def __init__(self, l_cell, r_cell, params=None, output_prefix='bi_',add_outputs=False):
        super(ConvBidirectionalCell, self).__init__('', params=params)
        self._override_cell_params = params is not None
        self._cells = [l_cell, r_cell]
        self._output_prefix = output_prefix
        self._add_outputs = add_outputs


    def unpack_weights(self, args):
        return _cells_unpack_weights(self._cells, args)

    def pack_weights(self, args):
        return _cells_pack_weights(self._cells, args)

    def __call__(self, inputs, states):
        raise NotImplementedError("Bidirectional cannot be stepped. Please use unroll")

    @property
    def state_info(self):
        return _cells_state_info(self._cells)

    def begin_state(self, **kwargs):
        assert not self._modified, \
            "After applying modifier cells (e.g. DropoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        return _cells_begin_state(self._cells, **kwargs)

    def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None):
        self.reset()

        inputs, axis = _normalize_sequence(length, inputs, layout, False)
        if begin_state is None:
            begin_state = self.begin_state()

        states = begin_state
        l_cell, r_cell = self._cells
        l_outputs, l_states = l_cell.unroll(length, inputs=inputs,
                                            begin_state=states[:len(l_cell.state_info)],
                                            layout=layout, merge_outputs=merge_outputs)
        r_outputs, r_states = r_cell.unroll(length,
                                            inputs=list(reversed(inputs)),
                                            begin_state=states[len(l_cell.state_info):],
                                            layout=layout, merge_outputs=merge_outputs)

        if merge_outputs is None:
            merge_outputs = (isinstance(l_outputs, symbol.Symbol)
                             and isinstance(r_outputs, symbol.Symbol))
            if not merge_outputs:
                if isinstance(l_outputs, symbol.Symbol):
                    l_outputs = list(symbol.SliceChannel(l_outputs, axis=axis,
                                                         num_outputs=length, squeeze_axis=1))
                if isinstance(r_outputs, symbol.Symbol):
                    r_outputs = list(symbol.SliceChannel(r_outputs, axis=axis,
                                                         num_outputs=length, squeeze_axis=1))

        if merge_outputs:
            l_outputs = [l_outputs]
            r_outputs = [symbol.reverse(r_outputs, axis=axis)]
        else:
            r_outputs = list(reversed(r_outputs))


        if not self._add_outputs:
            outputs = [symbol.Concat(l_o, r_o, dim=1+merge_outputs,
                                     name=('%sout'%(self._output_prefix) if merge_outputs
                                           else '%st%d'%(self._output_prefix, i)))
                       for i, l_o, r_o in
                       zip(range(len(l_outputs)), l_outputs, r_outputs)]
        else:
            assert len(l_outputs)==len(r_outputs),"the l_outputs must be equal to the r_outputs"
            outputs = [symbol._internal._plus(l_o, r_o, dim=1+merge_outputs,
                                     name=('%st%d'%(self._output_prefix, i)))
                       for i, l_o, r_o in
                       zip(range(len(l_outputs)), l_outputs, r_outputs)]

        if merge_outputs:
            outputs = outputs[0]

        states = [l_states, r_states]
        return outputs, states



import unittest
import mxnet as mx
import numpy as np
class ConvRNNCellTest(unittest.TestCase):
    def setUp(self):
        pass

    def testConvRNNCell(self):


        input_data_1 = mx.symbol.Variable(name="data_input_1",shape=(1, 3, 4, 7))
        input_data_2 = mx.symbol.Variable(name="data_input_2",shape=(1, 3, 4, 7))
        input_data_3 = mx.symbol.Variable(name="data_input_3",shape=(1, 3, 4, 7))
        input_data_4 = mx.symbol.Variable(name="data_input_4",shape=(1, 3, 4, 7))
        inputs=[input_data_1,input_data_2,input_data_3,input_data_4]
        cell = ConvGRUCell(1, prefix='gru_', stride=(1, 1), params=None,feat_size=(4,7))
        outputs,_ = cell.unroll(length=len(inputs),inputs=inputs)
        print outputs[0].infer_shape()
        print outputs[0].get_internals().list_outputs()

        sym = mx.symbol.concat(*outputs, dim=1, name="concat")
        input_data_1_v = mx.nd.array(np.arange(4 * 7 * 3).reshape(3, 4, 7))
        input_data_2_v = mx.nd.array(np.arange(4 * 7 * 3).reshape(3, 4, 7))
        input_data_3_v = mx.nd.array(np.arange(4 * 7 * 3).reshape(3, 4, 7))
        input_data_4_v = mx.nd.array(np.arange(4 * 7 * 3).reshape(3, 4, 7))

        args = {}

        for arg_name, arg_shape in zip(sym.list_arguments(), sym.infer_shape()[0]):
            if arg_name.endswith("i2h_weight"):
                args[arg_name]= mx.nd.array(np.ones(arg_shape))
            # elif arg_name.endswith("h2h_weight"):
            #     args[arg_name]= mx.nd.array(np.ones(arg_shape))
            else:
                args[arg_name] = mx.nd.array(np.zeros(arg_shape))

        args['data_input_1'] = input_data_1_v
        args['data_input_2'] = input_data_2_v
        args['data_input_3'] = input_data_3_v
        args['data_input_4'] = input_data_4_v

        sym_group = sym.get_internals()
        excutor = sym_group.bind(mx.cpu(), args)
        exc = excutor.forward()

        for internal_out,name in zip(exc,sym_group.list_outputs()):
            print "---------------------------------"
            print name
            print internal_out.asnumpy()

if __name__=='__main__':
    unittest.main()