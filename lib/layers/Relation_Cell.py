#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/13 17:58
# @Author  : yueqing.zhuang
# @Site    : 
# @File    : RNN_Cell.py
# @Software: PyCharm

from mxnet import symbol
from mxnet import init
from mxnet import symbol, init, ndarray
from mxnet.base import string_types, numeric_types


class RNNParams(object):
    """Container for holding variables.
    Used by RNN cells for parameter sharing between cells.

    Parameters
    ----------
    prefix : str
        Names of all variables created by this container will
        be prepended with prefix.
    """
    def __init__(self, prefix=''):
        self._prefix = prefix
        self._params = {}

    def get(self, name, **kwargs):
        """Get the variable given a name if one exists or create a new one if missing.

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


class RelationBaseRNNCell(object):
    """Abstract base class for RNN cells

    Parameters
    ----------
    prefix : str, optional
        Prefix for names of layers
        (this prefix is also used for names of weights if `params` is None
        i.e. if `params` are being created and not reused)
    params : RNNParams, default None.
        Container for weight sharing between cells.
        A new RNNParams container is created if `params` is None.
    """
    def __init__(self, prefix='', params=None):
        if params is None:
            params = RNNParams(prefix)
            self._own_params = True
        else:
            self._own_params = False
        self._prefix = prefix
        self._params = params
        self._modified = False

        self.reset()

    def reset(self):
        """Reset before re-using the cell for another graph."""
        self._init_counter = -1
        self._counter = -1

    def __call__(self, inputs, states):
        """Unroll the RNN for one time step.

        Parameters
        ----------
        inputs : sym.Variable
            input symbol, 2D, batch * num_units
        states : list of sym.Variable
            RNN state from previous step or the output of begin_state().

        Returns
        -------
        output : Symbol
            Symbol corresponding to the output from the RNN when unrolling
            for a single time step.
        states : nested list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of begin_state().
            This can be used as input state to the next time step
            of this RNN.

        See Also
        --------
        begin_state: This function can provide the states for the first time step.
        unroll: This function unrolls an RNN for a given number of (>=1) time steps.
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
            Starting states for the first RNN step.
        """
        assert not self._modified, \
            "After applying modifier cells (e.g. DropoutCell) the base " \
            "cell cannot be called directly. Call the modifier cell instead."
        states = []
        for info in self.state_info:
            self._init_counter += 1
            if info is None:
                state = func(name='%sbegin_state_%d'%(self._prefix, self._init_counter),
                             **kwargs)
            else:
                kwargs.update(info)
                state = func(name='%sbegin_state_%d'%(self._prefix, self._init_counter),
                             **kwargs)
            states.append(state)
        return states

    def unpack_weights(self, args):
        """Unpack fused weight matrices into separate
        weight matrices.

        For example, say you use a module object `mod` to run a network that has an lstm cell.
        In `mod.get_params()[0]`, the lstm parameters are all represented as a single big vector.
        `cell.unpack_weights(mod.get_params()[0])` will unpack this vector into a dictionary of
        more readable lstm parameters - c, f, i, o gates for i2h (input to hidden) and
        h2h (hidden to hidden) weights.

        Parameters
        ----------
        args : dict of str -> NDArray
            Dictionary containing packed weights.
            usually from `Module.get_params()[0]`.

        Returns
        -------
        args : dict of str -> NDArray
            Dictionary with unpacked weights associated with
            this cell.

        See Also
        --------
        pack_weights: Performs the reverse operation of this function.
        """
        args = args.copy()
        if not self._gate_names:
            return args
        h = self._num_hidden
        for group_name in ['i2h', 'h2h']:
            weight = args.pop('%s%s_weight'%(self._prefix, group_name))
            bias = args.pop('%s%s_bias' % (self._prefix, group_name))
            for j, gate in enumerate(self._gate_names):
                wname = '%s%s%s_weight' % (self._prefix, group_name, gate)
                args[wname] = weight[j*h:(j+1)*h].copy()
                bname = '%s%s%s_bias' % (self._prefix, group_name, gate)
                args[bname] = bias[j*h:(j+1)*h].copy()
        return args

    def pack_weights(self, args):
        """Pack separate weight matrices into a single packed
        weight.

        Parameters
        ----------
        args : dict of str -> NDArray
            Dictionary containing unpacked weights.

        Returns
        -------
        args : dict of str -> NDArray
            Dictionary with packed weights associated with
            this cell.
        """
        args = args.copy()
        if not self._gate_names:
            return args
        for group_name in ['i2h', 'h2h']:
            weight = []
            bias = []
            for gate in self._gate_names:
                wname = '%s%s%s_weight'%(self._prefix, group_name, gate)
                weight.append(args.pop(wname))
                bname = '%s%s%s_bias'%(self._prefix, group_name, gate)
                bias.append(args.pop(bname))
            args['%s%s_weight'%(self._prefix, group_name)] = ndarray.concatenate(weight)
            args['%s%s_bias'%(self._prefix, group_name)] = ndarray.concatenate(bias)
        return args

    #pylint: disable=no-self-use
    def _get_activation(self, inputs, activation, **kwargs):
        """Get activation function. Convert if is string"""
        if isinstance(activation, string_types):
            return symbol.Activation(inputs, act_type=activation, **kwargs)
        else:
            return activation(inputs, **kwargs)

class RelationGRUCell(RelationBaseRNNCell):
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
        super(RelationGRUCell, self).__init__(prefix=prefix, params=params)
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
        return ['_o']

    def __call__(self, inputs, reset_gate, update_gate, states):
        # pylint: disable=too-many-locals
        self._counter += 1

        seq_idx = self._counter
        name = '%st%d_' % (self._prefix, seq_idx)
        prev_state_h = states[0]

        i2h = symbol.FullyConnected(data=inputs,
                                    weight=self._iW,
                                    bias=self._iB,
                                    num_hidden=self._num_hidden,
                                    name="%s_i2h" % name)
        h2h = symbol.FullyConnected(data=prev_state_h,
                                    weight=self._hW,
                                    bias=self._hB,
                                    num_hidden=self._num_hidden,
                                    name="%s_h2h" % name)

        if not self._use_memory:
            update_gate = 0
            reset_gate =0

        next_h_tmp = symbol.Activation(i2h + reset_gate * h2h, act_type="tanh",
                                       name="%s_h_act" % name)


        next_h = symbol._internal._plus((1. - update_gate) * next_h_tmp, update_gate * prev_state_h,
                                        name='%sout' % name)

        return next_h, [next_h]

    def unroll(self, length, inputs,reset_gates,update_gates, begin_state=None, layout='NTC', merge_outputs=None):
        """Unroll an RNN cell across time steps.

        Parameters
        ----------
        length : int
            Number of steps to unroll.
        inputs : Symbol, list of Symbol, or None
            If `inputs` is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.

            If `inputs` is a list of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...).
        begin_state : nested list of Symbol, default None
            Input states created by `begin_state()`
            or output state of another cell.
            Created from `begin_state()` if None.
        layout : str, optional
            `layout` of input symbol. Only used if inputs
            is a single Symbol.
        merge_outputs : bool, optional
            If False, return outputs as a list of Symbols.
            If True, concatenate output across time steps
            and return a single symbol with shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.
            If None, output whatever is faster.

        Returns
        -------
        outputs : list of Symbol or Symbol
            Symbol (if `merge_outputs` is True) or list of Symbols
            (if `merge_outputs` is False) corresponding to the output from
            the RNN from this unrolling.

        states : nested list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of begin_state().
        """
        self.reset()

        inputs, _ = _normalize_sequence(length, inputs, layout, False)
        reset_gates, _ = _normalize_sequence(length, reset_gates, layout, False)
        update_gates, _ = _normalize_sequence(length, update_gates, layout, False)

        if begin_state is None:
            begin_state = self.begin_state()

        states = begin_state
        outputs = []
        for i in range(length):
            output, states = self(inputs[i],reset_gates[i],update_gates[i], states)
            outputs.append(output)

        outputs, _ = _normalize_sequence(length, outputs, layout, merge_outputs)

        return outputs, states

class RelationGRUCell_V2(RelationBaseRNNCell):
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
        super(RelationGRUCell_V2, self).__init__(prefix=prefix, params=params)
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
        return ['_o']

    def __call__(self, inputs, i2h_r, i2h_z, states):
        # pylint: disable=too-many-locals
        self._counter += 1

        seq_idx = self._counter
        name = '%st%d_' % (self._prefix, seq_idx)
        prev_state_h = states[0]

        i2h = symbol.FullyConnected(data=inputs,
                                    weight=self._iW,
                                    bias=self._iB,
                                    num_hidden=self._num_hidden,
                                    name="%s_i2h" % name)
        h2h = symbol.FullyConnected(data=prev_state_h,
                                    weight=self._hW,
                                    bias=self._hB,
                                    num_hidden=self._num_hidden * 3,
                                    name="%s_h2h" % name)

        h2h_r, h2h_z, h2h = symbol.SliceChannel(h2h, num_outputs=3, name="%s_h2h_slice" % name)

        reset_gate = symbol.Activation(i2h_r + h2h_r, act_type="sigmoid",
                                       name="%s_r_act" % name)
        update_gate = symbol.Activation(i2h_z + h2h_z, act_type="sigmoid",
                                        name="%s_z_act" % name)

        if not self._use_memory:
            update_gate = 0
            reset_gate =0

        next_h_tmp = symbol.Activation(i2h + reset_gate * h2h, act_type="tanh", name="%s_h_act" % name)

        next_h = symbol._internal._plus((1. - update_gate) * next_h_tmp, update_gate * prev_state_h,
                                        name='%sout' % name)

        return next_h, [next_h]

    def unroll(self, length, inputs,i2h_rs, i2h_zs, begin_state=None, layout='NTC', merge_outputs=None):
        """Unroll an RNN cell across time steps.

        Parameters
        ----------
        length : int
            Number of steps to unroll.
        inputs : Symbol, list of Symbol, or None
            If `inputs` is a single Symbol (usually the output
            of Embedding symbol), it should have shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.

            If `inputs` is a list of symbols (usually output of
            previous unroll), they should all have shape
            (batch_size, ...).
        begin_state : nested list of Symbol, default None
            Input states created by `begin_state()`
            or output state of another cell.
            Created from `begin_state()` if None.
        layout : str, optional
            `layout` of input symbol. Only used if inputs
            is a single Symbol.
        merge_outputs : bool, optional
            If False, return outputs as a list of Symbols.
            If True, concatenate output across time steps
            and return a single symbol with shape
            (batch_size, length, ...) if layout == 'NTC',
            or (length, batch_size, ...) if layout == 'TNC'.
            If None, output whatever is faster.

        Returns
        -------
        outputs : list of Symbol or Symbol
            Symbol (if `merge_outputs` is True) or list of Symbols
            (if `merge_outputs` is False) corresponding to the output from
            the RNN from this unrolling.

        states : nested list of Symbol
            The new state of this RNN after this unrolling.
            The type of this symbol is same as the output of begin_state().
        """
        self.reset()

        inputs, _ = _normalize_sequence(length, inputs, layout, False)
        i2h_rs, _ = _normalize_sequence(length, i2h_rs, layout, False)
        i2h_zs, _ = _normalize_sequence(length, i2h_zs, layout, False)

        if begin_state is None:
            begin_state = self.begin_state()

        states = begin_state
        outputs = []
        for i in range(length):
            output, states = self(inputs[i],i2h_rs[i],i2h_zs[i], states)
            outputs.append(output)

        outputs, _ = _normalize_sequence(length, outputs, layout, merge_outputs)

        return outputs, states


class RelationBidirectionalCell_V2(RelationBaseRNNCell):
    """Bidirectional RNN cell

    Parameters
    ----------
    l_cell : RelationBaseRNNCell
        cell for forward unrolling
    r_cell : RelationBaseRNNCell
        cell for backward unrolling
    output_prefix : str, default 'bi_'
        prefix for name of output
    add_outputs : for the add the symbol
    """
    def __init__(self, l_cell, r_cell, params=None, output_prefix='bi_',add_outputs=False):
        super(RelationBidirectionalCell_V2, self).__init__('', params=params)
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

    def unroll(self, length, inputs,p_i2h_rs,p_i2h_zs,n_i2h_rs,n_i2h_zs, begin_state=None, layout='NTC', merge_outputs=None):
        self.reset()

        inputs, axis = _normalize_sequence(length, inputs, layout, False)
        p_i2h_rs,_ = _normalize_sequence(length, p_i2h_rs, layout, False)
        p_i2h_zs,_ = _normalize_sequence(length, p_i2h_zs, layout, False)

        n_i2h_rs,_ = _normalize_sequence(length, n_i2h_rs, layout, False)
        n_i2h_zs,_ = _normalize_sequence(length, n_i2h_zs, layout, False)

        if begin_state is None:
            begin_state = self.begin_state()

        states = begin_state
        l_cell, r_cell = self._cells
        l_outputs, l_states = l_cell.unroll(length, inputs=inputs,
                                            i2h_rs= p_i2h_rs,
                                            i2h_zs= p_i2h_zs,
                                            begin_state=states[:len(l_cell.state_info)],
                                            layout=layout, merge_outputs=merge_outputs)

        r_outputs, r_states = r_cell.unroll(length,inputs=list(reversed(inputs)),
                                            i2h_rs= n_i2h_rs,
                                            i2h_zs= n_i2h_zs,
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

class RelationBidirectionalCell(RelationBaseRNNCell):
    """Bidirectional RNN cell

    Parameters
    ----------
    l_cell : RelationBaseRNNCell
        cell for forward unrolling
    r_cell : RelationBaseRNNCell
        cell for backward unrolling
    output_prefix : str, default 'bi_'
        prefix for name of output
    add_outputs : for the add the symbol
    """
    def __init__(self, l_cell, r_cell, params=None, output_prefix='bi_',add_outputs=False):
        super(RelationBidirectionalCell, self).__init__('', params=params)
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

    def unroll(self, length, inputs,p_reset_gates,p_update_gates,n_reset_gates,n_update_gates, begin_state=None, layout='NTC', merge_outputs=None):
        self.reset()

        inputs, axis = _normalize_sequence(length, inputs, layout, False)
        p_reset_gates,_ = _normalize_sequence(length, p_reset_gates, layout, False)
        p_update_gates,_ = _normalize_sequence(length, p_update_gates, layout, False)

        n_reset_gates,_ = _normalize_sequence(length, n_reset_gates, layout, False)
        n_update_gates,_ = _normalize_sequence(length, n_update_gates, layout, False)

        if begin_state is None:
            begin_state = self.begin_state()

        states = begin_state
        l_cell, r_cell = self._cells
        l_outputs, l_states = l_cell.unroll(length, inputs=inputs,
                                            reset_gates=p_reset_gates,
                                            update_gates=p_update_gates,
                                            begin_state=states[:len(l_cell.state_info)],
                                            layout=layout, merge_outputs=merge_outputs)

        r_outputs, r_states = r_cell.unroll(length,inputs=list(reversed(inputs)),
                                            reset_gates=n_reset_gates,
                                            update_gates=n_update_gates,
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