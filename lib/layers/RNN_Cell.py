#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/13 17:58
# @Author  : yueqing.zhuang
# @Site    : 
# @File    : RNN_Cell.py
# @Software: PyCharm

from mxnet.rnn.rnn_cell import BaseRNNCell
from mxnet import symbol
from mxnet import init

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

class RNNCell(BaseRNNCell):
    """Simple recurrent neural network cell.

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
    def __init__(self, num_hidden, activation='tanh', prefix='rnn_', params=None,use_memory=True):
        super(RNNCell, self).__init__(prefix=prefix, params=params)
        self._num_hidden = num_hidden
        self._activation = activation
        self._use_memory = use_memory
        self._iW = self.params.get('i2h_weight')
        self._iB = self.params.get('i2h_bias')
        self._hW = self.params.get('h2h_weight')
        self._hB = self.params.get('h2h_bias')

    @property
    def state_info(self):
        return [{'shape': (0, self._num_hidden), '__layout__': 'NC'}]

    @property
    def _gate_names(self):
        return ('',)

    def __call__(self, inputs, states):
        self._counter += 1
        name = '%st%d_'%(self._prefix, self._counter)
        i2h = symbol.FullyConnected(data=inputs, weight=self._iW, bias=self._iB,
                                    num_hidden=self._num_hidden,
                                    name='%si2h'%name)

        if self._use_memory:
            h2h = symbol.FullyConnected(data=states[0], weight=self._hW, bias=self._hB,
                                        num_hidden=self._num_hidden,
                                        name='%sh2h'%name)
        else:
            h2h = 0
        output = self._get_activation(i2h + h2h, self._activation,
                                      name='%sout'%name)

        return output, [output]

class GRUCell(BaseRNNCell):
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
    def __init__(self, num_hidden, prefix='gru_', params=None,use_memory=True):
        super(GRUCell, self).__init__(prefix=prefix, params=params)
        self._num_hidden = num_hidden
        self._iW = self.params.get("i2h_weight")
        self._iB = self.params.get("i2h_bias")
        self._hW = self.params.get("h2h_weight")
        self._hB = self.params.get("h2h_bias")
        self._use_memory=use_memory

    @property
    def state_info(self):
        return [{'shape': (0, self._num_hidden),
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

        i2h = symbol.FullyConnected(data=inputs,
                                    weight=self._iW,
                                    bias=self._iB,
                                    num_hidden=self._num_hidden * 3,
                                    name="%s_i2h" % name)
        h2h = symbol.FullyConnected(data=prev_state_h,
                                    weight=self._hW,
                                    bias=self._hB,
                                    num_hidden=self._num_hidden * 3,
                                    name="%s_h2h" % name)

        i2h_r, i2h_z, i2h = symbol.SliceChannel(i2h, num_outputs=3, name="%s_i2h_slice" % name)
        h2h_r, h2h_z, h2h = symbol.SliceChannel(h2h, num_outputs=3, name="%s_h2h_slice" % name)

        reset_gate = symbol.Activation(i2h_r + h2h_r, act_type="sigmoid",
                                       name="%s_r_act" % name)
        update_gate = symbol.Activation(i2h_z + h2h_z, act_type="sigmoid",
                                        name="%s_z_act" % name)
        if not self._use_memory:
            update_gate = 0
            reset_gate =0

        next_h_tmp = symbol.Activation(i2h + reset_gate * h2h, act_type="tanh",
                                       name="%s_h_act" % name)


        next_h = symbol._internal._plus((1. - update_gate) * next_h_tmp, update_gate * prev_state_h,
                                        name='%sout' % name)

        return next_h, [next_h]

class LSTMCell(BaseRNNCell):
    """Long-Short Term Memory (LSTM) network cell.

    Parameters
    ----------
    num_hidden : int
        number of units in output symbol
    prefix : str, default 'rnn_'
        prefix for name of layers
        (and name of weight if params is None)
    params : RNNParams or None
        container for weight sharing between cells.
        created if None.
    forget_bias : bias added to forget gate, default 1.0.
        Jozefowicz et al. 2015 recommends setting this to 1.0
    """
    def __init__(self, num_hidden, prefix='lstm_', params=None, forget_bias=1.0,use_memory=False):
        super(LSTMCell, self).__init__(prefix=prefix, params=params)

        self._num_hidden = num_hidden
        self._iW = self.params.get('i2h_weight')
        self._hW = self.params.get('h2h_weight')
        # we add the forget_bias to i2h_bias, this adds the bias to the forget gate activation
        self._iB = self.params.get('i2h_bias', init=init.LSTMBias(forget_bias=forget_bias))
        self._hB = self.params.get('h2h_bias')
        self._use_memory=use_memory

    @property
    def state_info(self):
        return [{'shape': (0, self._num_hidden), '__layout__': 'NC'},
                {'shape': (0, self._num_hidden), '__layout__': 'NC'}]

    @property
    def _gate_names(self):
        return ['_i', '_f', '_c', '_o']

    def __call__(self, inputs, states):
        self._counter += 1
        name = '%st%d_'%(self._prefix, self._counter)
        i2h = symbol.FullyConnected(data=inputs, weight=self._iW, bias=self._iB,
                                    num_hidden=self._num_hidden*4,
                                    name='%si2h'%name)
        h2h = symbol.FullyConnected(data=states[0], weight=self._hW, bias=self._hB,
                                    num_hidden=self._num_hidden*4,
                                    name='%sh2h'%name)
        gates = i2h + h2h
        slice_gates = symbol.SliceChannel(gates, num_outputs=4,
                                          name="%sslice"%name)
        in_gate = symbol.Activation(slice_gates[0], act_type="sigmoid",
                                    name='%si'%name)
        forget_gate = symbol.Activation(slice_gates[1], act_type="sigmoid",
                                        name='%sf'%name)
        in_transform = symbol.Activation(slice_gates[2], act_type="tanh",
                                         name='%sc'%name)
        out_gate = symbol.Activation(slice_gates[3], act_type="sigmoid",
                                     name='%so'%name)
        next_c = symbol._internal._plus(forget_gate * states[1], in_gate * in_transform,
                                        name='%sstate'%name)
        next_h = symbol._internal._mul(out_gate, symbol.Activation(next_c, act_type="tanh"),
                                       name='%sout'%name)

        return next_h, [next_h, next_c]



class BidirectionalCell(BaseRNNCell):
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
        super(BidirectionalCell, self).__init__('', params=params)
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