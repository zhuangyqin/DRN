#!/usr/bin/env python
# encoding: utf-8

"""
@author: ‘zyq‘
@license: Apache Licence 
@file: RelationMap.py
@time: 2017/9/28 10:17
"""

import mxnet as mx
from math import ceil, floor
from layers.RNN_Cell import GRUCell, BidirectionalCell, LSTMCell, RNNCell
from layers.Relation_Cell import RelationGRUCell, RelationBidirectionalCell, \
    RelationGRUCell_V2,RelationBidirectionalCell_V2

Debug = 0



def _sequence_image(InData, mode="LR", PatchSize=(1, 1),is_cudnn=False,skip_step = 1):
    inshape = InData.infer_shape()[1][0]

    if inshape[2] % PatchSize[0] != 0 or inshape[3] % PatchSize[1] != 0:
        h_pad = inshape[2] % PatchSize[0]
        w_pad = inshape[3] % PatchSize[1]
        Padding_out = mx.symbol.pad(InData, mode='const', const_value=0, pad_width=(0, 0, 0, 0, floor(h_pad / 2),
                                                                                    ceil(h_pad / 2), floor(w_pad / 2),
                                                                                    ceil(w_pad / 2)))
    else:
        Padding_out = InData

    padding_out_c, padding_out_h, padding_out_w = Padding_out.infer_shape()[1][0][1:]
    patchh, patchw = PatchSize

    npatchhes = padding_out_h / patchh
    npatchwes = padding_out_w / patchw

    # bs*cc, #H, ph, #W, pw
    Padding_out_reshape = mx.symbol.reshape(Padding_out, shape=(-1, npatchhes, patchh, npatchwes, patchw))

    if Debug:
        print "Padding_out_reshape.infer_shape", Padding_out_reshape.infer_shape()[1][0]

    if mode == "LR" or mode == "RL":
        # bs*cc, #H, #W, ph, pw,
        net = mx.symbol.transpose(Padding_out_reshape, axes=(0, 1, 3, 2, 4))
        # bs,cc, #H, #W, ph*pw,
        net = mx.symbol.reshape(net, shape=(-1, padding_out_c, npatchhes, npatchwes, patchh * patchw))
        # bs, #H, #W, cc, ph*pw,
        net = mx.symbol.transpose(net, axes=(0, 2, 3, 1, 4))
        # bs*#H, #W, cc*ph*pw,
        net = mx.symbol.reshape(net, shape=(-1, npatchwes, padding_out_c * patchh * patchw))


        if is_cudnn:
            #W, bs*#H,  cc*ph*pw,
            net = mx.symbol.transpose(net, axes=(1, 0, 2))
            if skip_step > 1:
                t = net.infer_shape()[1][0][0]
                b = net.infer_shape()[1][0][1]
                c = net.infer_shape()[1][0][2]
                t_ = t//skip_step

                # #T,#batch_size,
                net = net.reshape((t_,skip_step*b,c))

        if mode == "RL":
            axis = 1
            if is_cudnn:
                axis = 0
            net = mx.symbol.flip(net, axis=axis)

            if skip_step > 1:
                t = net.infer_shape()[1][0][0]
                b = net.infer_shape()[1][0][1]
                c = net.infer_shape()[1][0][2]
                t_ = t//skip_step

                # #T,#batch_size,
                net = net.reshape((t_,skip_step*b,c))

            if Debug:
                print "RL mode"
        if Debug:
            print "LR.infer_shape", net.infer_shape()[1][0]


        return net

    elif mode == "UD" or mode == "DU":
        # bs*cc, #H, #W, ph, pw,
        net = mx.symbol.transpose(Padding_out_reshape, axes=(0, 1, 3, 2, 4))
        # bs,cc, #H, #W, ph*pw,
        net = mx.symbol.reshape(net, shape=(-1, padding_out_c, npatchhes, npatchwes, patchh * patchw))
        # bs, #W, #H, cc, ph*pw,
        net = mx.symbol.transpose(net, axes=(0, 3, 2, 1, 4))
        # bs* #W, #H, cc*ph*pw,
        net = mx.symbol.reshape(net, shape=(-1, npatchhes, padding_out_c * patchh * patchw))

        if is_cudnn:
            net = mx.symbol.transpose(net, axes=(1, 0, 2))

            if skip_step > 1:
                t = net.infer_shape()[1][0][0]
                b = net.infer_shape()[1][0][1]
                c = net.infer_shape()[1][0][2]
                t_ = t//skip_step

                # #T,#batch_size,
                net = net.reshape((t_,skip_step*b,c))

        if mode == "DU":
            axis = 1
            if is_cudnn:
                axis = 0
            net = mx.symbol.flip(net, axis=axis)
            if Debug:
                print "DU mode"

            if skip_step > 1:
                t = net.infer_shape()[1][0][0]
                b = net.infer_shape()[1][0][1]
                c = net.infer_shape()[1][0][2]
                t_ = t//skip_step

                # #T,#batch_size,
                net = net.reshape((t_,skip_step*b,c))

        if Debug:
            print "UD.infer_shape", net.infer_shape()[1][0]

        return net
    else:
        return NotImplementedError


def _unsequence_image(InData, batchsize, mode="LR",is_cudnn = False,skip_step=1):
    # assume data [num_ouputs,batchsize*#?,hidden]

    inshape = InData.infer_shape()[1][0]
    sequence_length = inshape[0]
    hidden = inshape[2]

    # sequence_length,batchsize,-1,hidden
    net = mx.symbol.reshape(InData, shape=(sequence_length*skip_step , batchsize , -1, hidden))

    if mode == "LR" or mode == "RL":
        # batchsize,hidden,-1 ,sequence_length
        net = mx.symbol.transpose(net, axes=(1, 3, 2, 0))
        if mode == "RL":
            net = mx.symbol.flip(net, axis=3)

    elif mode == "UD" or mode == "DU":
        # batchsize,hidden,sequence_length,-1
        net = mx.symbol.transpose(net, axes=(1, 3, 0, 2))
        if mode == "DU":
            net = mx.symbol.flip(net, axis=2)
    else:
        return NotImplementedError

    return net


def RnnMap(InData, name="RNNRelation", type="GRU", PatchSize=(1, 1), num_hidden=2048, use_memory=True):
    # choosing the type of the rcnn
    if type == "RNN":
        lbasecell = RNNCell(num_hidden, prefix=name + "_rnn_l_", use_memory=use_memory)
        rbasecell = RNNCell(num_hidden, prefix=name + "_rnn_r_", use_memory=use_memory)
        ubasecell = RNNCell(num_hidden, prefix=name + "_rnn_u_", use_memory=use_memory)
        dbasecell = RNNCell(num_hidden, prefix=name + "_rnn_d_", use_memory=use_memory)
    elif type == "LSTM":
        lbasecell = LSTMCell(num_hidden, prefix=name + "_lstm_l_", use_memory=use_memory)
        rbasecell = LSTMCell(num_hidden, prefix=name + "_lstm_r_", use_memory=use_memory)
        ubasecell = LSTMCell(num_hidden, prefix=name + "_lstm_u_", use_memory=use_memory)
        dbasecell = LSTMCell(num_hidden, prefix=name + "_lstm_d_", use_memory=use_memory)
    elif type == "GRU":
        lbasecell = GRUCell(num_hidden, prefix=name + "_gru_l_", use_memory=use_memory)
        rbasecell = GRUCell(num_hidden, prefix=name + "_gru_r_", use_memory=use_memory)
        ubasecell = GRUCell(num_hidden, prefix=name + "_gru_u_", use_memory=use_memory)
        dbasecell = GRUCell(num_hidden, prefix=name + "_gru_d_", use_memory=use_memory)

    LRRNN = BidirectionalCell(l_cell=lbasecell, r_cell=rbasecell, output_prefix=name + "_lr_bi_")
    UDRNN = BidirectionalCell(l_cell=ubasecell, r_cell=dbasecell, output_prefix=name + "_ud_bi_")

    batchsize = InData.infer_shape()[1][0][0]

    net = _sequence_image(InData, mode="LR", PatchSize=PatchSize)
    seq_length_lr = net.infer_shape()[1][0][1]
    rnnlrout, rnnlrstate = LRRNN.unroll(length=seq_length_lr, inputs=net)

    if Debug:
        for i, out_step in enumerate(rnnlrout):
            print "rnnlr output step", i + 1, "shape", out_step.infer_shape()[1][0]

    net = mx.symbol.stack(*rnnlrout)
    net = _unsequence_image(net, batchsize, mode="LR")
    net = mx.sym.Activation(data=net, act_type='relu', name=name + '_lr_relu')

    if Debug:
        print "--------------------------------------------------------"

    net = _sequence_image(net, mode="UD", PatchSize=PatchSize)
    seq_length_ud = net.infer_shape()[1][0][1]
    rnnudout, rnnudstate = UDRNN.unroll(length=seq_length_ud, inputs=net)

    if Debug:
        for i, out_step in enumerate(rnnudout):
            print "rnnlr output step", i + 1, "shape", out_step.infer_shape()[1][0]

    net = mx.symbol.stack(*rnnudout)
    net = _unsequence_image(net, batchsize, mode="UD")

    net = mx.sym.Activation(data=net, act_type='relu', name=name + '_ud_relu')

    print "RNN out", net.infer_shape()[1][0]

    if Debug:
        print  "network output", net.infer_shape()[1][0]

    return net


def RnnMap_cudnn(InData, name="RNNRelation", type="GRU", PatchSize=(1, 1), num_hidden=2048, use_memory=True,num_layers=1,old_type =True):

    batchsize = InData.infer_shape()[1][0][0]

    net = _sequence_image(InData, mode="LR", PatchSize=PatchSize,is_cudnn=True)

    print "net shape", net.infer_shape()[1][0]
    if old_type:
        lr_parameters = mx.symbol.var("rnn_lr_param")
        lr_state = mx.symbol.var("rnn_lr_state")
    else:
        lr_parameters = mx.symbol.var("%s_lr_param"%name)
        lr_state = mx.symbol.var("%s_lr_state"%name)
    rnnlrout = mx.symbol.RNN(net,lr_parameters,lr_state, state_size=num_hidden,num_layers=num_layers,name="rnn_lr",
                             bidirectional=True,mode="gru",state_outputs=False,p=0.5)

    print "rnnlr output shape", rnnlrout.infer_shape()[1][0]

    # bs*#W,#H,#NUM
    net = mx.sym.Activation(data=rnnlrout, act_type='relu', name=name + '_lr_relu')

    net = _unsequence_image(net, batchsize, mode="LR", is_cudnn=True)
    print "rnnlr output image shape", net.infer_shape()[1][0]
    net = _sequence_image(net, mode="UD", PatchSize=PatchSize,is_cudnn=True)

    if Debug:
        print "--------------------------------------------------------"

    if old_type:
        ud_parameters = mx.symbol.var("rnn_ud_param")
        ud_state = mx.symbol.var("rnn_ud_state")
    else:
        ud_parameters = mx.symbol.var("%s_ud_param" % name)
        ud_state = mx.symbol.var("%s_ud_state" % name)
    rnnudout = mx.symbol.RNN(net,ud_parameters,ud_state,state_size=num_hidden,num_layers=num_layers,name="rnn_ud",
                             bidirectional=True,mode="gru",state_outputs=False,p=0.5)

    net = _unsequence_image(rnnudout, batchsize, mode="UD", is_cudnn=True)
    net = mx.sym.Activation(data=net, act_type='relu', name=name + '_ud_relu')

    print "RNN out", net.infer_shape()[1][0]

    if Debug:
        print  "network output", net.infer_shape()[1][0]

    return net


def RnnMap_cudnn_v2(InData, name="RNNRelation", type="GRU", PatchSize=(1, 1), num_hidden=2048,
                    use_memory=True,num_layers=1, old_type =True, skip_step = 1, parallel = True):
    # this version is parallled the ud and the lr operation
    if Debug:
        print "==========================================="
        print "=======        skip_step: "+str(skip_step)+" ============="

    batchsize = InData.infer_shape()[1][0][0]

    net = _sequence_image(InData, mode="LR", PatchSize=PatchSize,is_cudnn=True,skip_step=skip_step)
    if Debug:
        print "rnnlr input shape", net.infer_shape()[1][0]
    if old_type:
        lr_parameters = mx.symbol.var("rnn_lr_param")
        lr_state = mx.symbol.var("rnn_lr_state")
    else:
        lr_parameters = mx.symbol.var("%s_lr_param"%name)
        lr_state = mx.symbol.var("%s_lr_state"%name)
    rnnlrout = mx.symbol.RNN(net,lr_parameters,lr_state, state_size=num_hidden,num_layers=num_layers,name="rnn_lr",
                             bidirectional=True,mode="gru",state_outputs=False,p=0.5)

    if Debug:
        print "rnnlr output shape", rnnlrout.infer_shape()[1][0]

    rnnlrout = _unsequence_image(rnnlrout, batchsize, mode="LR", is_cudnn=True,skip_step=skip_step)    # bs*#W,#H,#NUM
    rnnlrout = mx.sym.Activation(data=rnnlrout, act_type='relu', name=name + '_lr_relu')

    if Debug:
        print "rnnlr output image shape", rnnlrout.infer_shape()[1][0]

    if parallel:
        lr_indata = InData
    else:
        lr_indata = rnnlrout
    net = _sequence_image(lr_indata, mode="UD", PatchSize=PatchSize,is_cudnn=True,skip_step=skip_step)

    if Debug:
        print "--------------------------------------------------------"

    if old_type:
        ud_parameters = mx.symbol.var("rnn_ud_param")
        ud_state = mx.symbol.var("rnn_ud_state")
    else:
        ud_parameters = mx.symbol.var("%s_ud_param" % name)
        ud_state = mx.symbol.var("%s_ud_state" % name)
    rnnudout = mx.symbol.RNN(net,ud_parameters,ud_state,state_size=num_hidden,num_layers=num_layers,name="rnn_ud",
                             bidirectional=True,mode="gru",state_outputs=False,p=0.5)

    rnnudout = _unsequence_image(rnnudout, batchsize, mode="UD", is_cudnn=True,skip_step=skip_step)
    rnnudout = mx.sym.Activation(data=rnnudout, act_type='relu', name=name + '_ud_relu')

    if Debug:
        print "RNN LR out ", rnnlrout.infer_shape()[1][0]
        print "RNN UD out ", rnnudout.infer_shape()[1][0]

    if parallel:
        net = mx.sym.concat(*[rnnlrout,rnnudout])
    else:
        net = rnnudout

    if Debug:
        print "RNN out", net.infer_shape()[1][0]

    if Debug:
        print  "network output", net.infer_shape()[1][0]
    if Debug:
        print "==========================================="

    return net

def RnnMapOutRelation_V2(InData, i2h_rs, i2h_zs, name="RNNRelation", PatchSize=(1, 1),
                      num_hidden=2048, use_memory=True):

    lbasecell = RelationGRUCell_V2(num_hidden, prefix=name + "_gru_l_", use_memory=use_memory)
    rbasecell = RelationGRUCell_V2(num_hidden, prefix=name + "_gru_r_", use_memory=use_memory)
    ubasecell = RelationGRUCell_V2(num_hidden, prefix=name + "_gru_u_", use_memory=use_memory)
    dbasecell = RelationGRUCell_V2(num_hidden, prefix=name + "_gru_d_", use_memory=use_memory)

    LRRNN = RelationBidirectionalCell_V2(l_cell=lbasecell, r_cell=rbasecell, output_prefix=name + "_lr_bi_")
    UDRNN = RelationBidirectionalCell_V2(l_cell=ubasecell, r_cell=dbasecell, output_prefix=name + "_ud_bi_")

    batchsize = InData.infer_shape()[1][0][0]

    net = _sequence_image(InData, mode="LR", PatchSize=PatchSize)

    # relation for the lr mode
    l_i2h_rs, r_i2h_rs, u_i2h_rs, d_i2h_rs = mx.symbol.split(i2h_rs, axis=1, num_outputs=4)
    l_i2h_zs, r_i2h_zs, u_i2h_zs, d_i2h_zs = mx.symbol.split(i2h_zs, axis=1,num_outputs=4)
    if Debug:
        print "i2h_rs shape", i2h_rs.infer_shape()[1][0]
        print "l_i2h_rs shape", l_i2h_rs.infer_shape()[1][0]
        print "r_i2h_rs shape", r_i2h_rs.infer_shape()[1][0]
        print "u_i2h_rs shape", u_i2h_rs.infer_shape()[1][0]
        print "d_i2h_rs shape", d_i2h_rs.infer_shape()[1][0]

        print "i2h_zs shape", i2h_zs.infer_shape()[1][0]
        print "l_i2h_zs shape", l_i2h_zs.infer_shape()[1][0]
        print "r_i2h_zs shape", r_i2h_zs.infer_shape()[1][0]
        print "u_i2h_zs shape", u_i2h_zs.infer_shape()[1][0]
        print "d_i2h_zs shape", d_i2h_zs.infer_shape()[1][0]

    l_i2h_rs = _sequence_image(l_i2h_rs, mode="LR", PatchSize=PatchSize)
    l_i2h_zs = _sequence_image(l_i2h_zs, mode="LR", PatchSize=PatchSize)
    r_i2h_rs = _sequence_image(r_i2h_rs, mode="RL", PatchSize=PatchSize)
    r_i2h_zs = _sequence_image(r_i2h_zs, mode="RL", PatchSize=PatchSize)
    u_i2h_rs = _sequence_image(u_i2h_rs, mode="UD", PatchSize=PatchSize)
    u_i2h_zs = _sequence_image(u_i2h_zs, mode="UD", PatchSize=PatchSize)
    d_i2h_rs = _sequence_image(d_i2h_rs, mode="DU", PatchSize=PatchSize)
    d_i2h_zs = _sequence_image(d_i2h_zs, mode="DU", PatchSize=PatchSize)

    seq_length_lr = net.infer_shape()[1][0][1]
    rnnlrout, rnnlrstate = LRRNN.unroll(length=seq_length_lr, inputs=net,
                                        p_i2h_rs=l_i2h_rs,
                                        p_i2h_zs=l_i2h_zs,
                                        n_i2h_rs=r_i2h_rs,
                                        n_i2h_zs=r_i2h_zs,
                                        )
    net = mx.symbol.stack(*rnnlrout)
    net = _unsequence_image(net, batchsize, mode="LR")
    net = mx.sym.Activation(data=net, act_type='relu', name=name + '_lr_relu')

    if Debug:
        print "--------------------------------------------------------"

    net = _sequence_image(net, mode="UD", PatchSize=PatchSize)

    seq_length_ud = net.infer_shape()[1][0][1]

    rnnudout, rnnudstate = UDRNN.unroll(length=seq_length_ud, inputs=net,
                                        p_i2h_rs=u_i2h_rs,
                                        p_i2h_zs=u_i2h_zs,
                                        n_i2h_rs=d_i2h_rs,
                                        n_i2h_zs=d_i2h_zs)
    net = mx.symbol.stack(*rnnudout)

    net = _unsequence_image(net, batchsize, mode="UD")

    net = mx.sym.Activation(data=net, act_type='relu', name=name + '_ud_relu')

    if Debug:
        print  "network output", net.infer_shape()[1][0]

    return net


def RnnMapOutRelation(InData, reset_gates, update_gates, name="RNNRelation", PatchSize=(1, 1),
                      num_hidden=2048, use_memory=True):

    lbasecell = RelationGRUCell(num_hidden, prefix=name + "_gru_l_", use_memory=use_memory)
    rbasecell = RelationGRUCell(num_hidden, prefix=name + "_gru_r_", use_memory=use_memory)
    ubasecell = RelationGRUCell(num_hidden, prefix=name + "_gru_u_", use_memory=use_memory)
    dbasecell = RelationGRUCell(num_hidden, prefix=name + "_gru_d_", use_memory=use_memory)

    LRRNN = RelationBidirectionalCell(l_cell=lbasecell, r_cell=rbasecell, output_prefix=name + "_lr_bi_")
    UDRNN = RelationBidirectionalCell(l_cell=ubasecell, r_cell=dbasecell, output_prefix=name + "_ud_bi_")

    batchsize = InData.infer_shape()[1][0][0]

    net = _sequence_image(InData, mode="LR", PatchSize=PatchSize)

    # relation for the lr mode

    l_reset_gates, r_reset_gates, u_reset_gates, d_reset_gates = mx.symbol.split(reset_gates, axis=1, num_outputs=4)
    l_update_gates, r_update_gates, u_update_gates, d_update_gates = mx.symbol.split(update_gates, axis=1,
                                                                                     num_outputs=4)
    if Debug:
        print "reset_gates shape", reset_gates.infer_shape()[1][0]
        print "l_reset_gates shape", l_reset_gates.infer_shape()[1][0]
        print "w_reset_gates shape", r_reset_gates.infer_shape()[1][0]
        print "u_reset_gates shape", u_reset_gates.infer_shape()[1][0]
        print "d_reset_gates shape", d_reset_gates.infer_shape()[1][0]

        print "update_gates shape", update_gates.infer_shape()[1][0]
        print "l_update_gates shape", l_update_gates.infer_shape()[1][0]
        print "r_update_gates shape", r_update_gates.infer_shape()[1][0]
        print "u_update_gates shape", u_update_gates.infer_shape()[1][0]
        print "d_update_gates shape", d_update_gates.infer_shape()[1][0]

    l_reset_gates = _sequence_image(l_reset_gates, mode="LR", PatchSize=PatchSize)
    l_update_gates = _sequence_image(l_update_gates, mode="LR", PatchSize=PatchSize)
    r_reset_gates = _sequence_image(r_reset_gates, mode="RL", PatchSize=PatchSize)
    r_update_gates = _sequence_image(r_update_gates, mode="RL", PatchSize=PatchSize)
    u_reset_gates = _sequence_image(u_reset_gates, mode="UD", PatchSize=PatchSize)
    u_update_gates = _sequence_image(u_update_gates, mode="UD", PatchSize=PatchSize)
    d_reset_gates = _sequence_image(d_reset_gates, mode="DU", PatchSize=PatchSize)
    d_update_gates = _sequence_image(d_update_gates, mode="DU", PatchSize=PatchSize)

    seq_length_lr = net.infer_shape()[1][0][1]
    rnnlrout, rnnlrstate = LRRNN.unroll(length=seq_length_lr, inputs=net,
                                        p_reset_gates=l_reset_gates,
                                        p_update_gates=l_update_gates,
                                        n_reset_gates=r_reset_gates,
                                        n_update_gates=r_update_gates,
                                        )
    net = mx.symbol.stack(*rnnlrout)
    net = _unsequence_image(net, batchsize, mode="LR")
    net = mx.sym.Activation(data=net, act_type='relu', name=name + '_lr_relu')

    if Debug:
        print "--------------------------------------------------------"

    net = _sequence_image(net, mode="UD", PatchSize=PatchSize)

    seq_length_ud = net.infer_shape()[1][0][1]

    rnnudout, rnnudstate = UDRNN.unroll(length=seq_length_ud, inputs=net,
                                        p_reset_gates=u_reset_gates,
                                        p_update_gates=u_update_gates,
                                        n_reset_gates=d_reset_gates,
                                        n_update_gates=d_update_gates)
    net = mx.symbol.stack(*rnnudout)

    net = _unsequence_image(net, batchsize, mode="UD")

    net = mx.sym.Activation(data=net, act_type='relu', name=name + '_ud_relu')

    if Debug:
        print  "relation network output", net.infer_shape()[1][0]

    return net
