#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/13 17:58
# @Author  : yueqing.zhuang
# @Site    :
# @File    : RNN_Cell.py
# @Software: PyCharm

from math import ceil, floor
import unittest
import mxnet as mx

from layers.RNN_Cell import GRUCell, BidirectionalCell, LSTMCell,RNNCell

Debug = 0

def ReLayer(in_data, name="relay", type="RNN", patch_size=(1, 1), num_hidden=2048,
            stack=True,activation="relu",use_memory=True,add_outputs=False,dropout=-1):

    #choosing the type of the rcnn
    if type=="RNN":
        lbasecell= RNNCell(num_hidden,prefix=name+"_rnn_l_",activation=activation,use_memory=use_memory)
        rbasecell= RNNCell(num_hidden,prefix=name+"_rnn_r_",activation=activation,use_memory=use_memory)
        ubasecell= RNNCell(num_hidden,prefix=name+"_rnn_u_",activation=activation,use_memory=use_memory)
        dbasecell= RNNCell(num_hidden,prefix=name+"_rnn_d_",activation=activation,use_memory=use_memory)
    elif type=="LSTM":
        lbasecell= LSTMCell(num_hidden,prefix=name+"_lstm_l_",use_memory=use_memory)
        rbasecell= LSTMCell(num_hidden,prefix=name+"_lstm_r_",use_memory=use_memory)
        ubasecell= LSTMCell(num_hidden,prefix=name+"_lstm_u_",use_memory=use_memory)
        dbasecell= LSTMCell(num_hidden,prefix=name+"_lstm_d_",use_memory=use_memory)
    elif type=="GRU":
        lbasecell = GRUCell(num_hidden,prefix=name+"_gru_l_",use_memory=use_memory)
        rbasecell = GRUCell(num_hidden,prefix=name+"_gru_r_",use_memory=use_memory)
        ubasecell = GRUCell(num_hidden,prefix=name+"_gru_u_",use_memory=use_memory)
        dbasecell = GRUCell(num_hidden,prefix=name+"_gru_d_",use_memory=use_memory)

        lrbasecelldrop =  mx.rnn.DropoutCell(dropout=dropout)
        udbasecelldrop =  mx.rnn.DropoutCell(dropout=dropout)

    LRRNN=BidirectionalCell(l_cell=lbasecell,r_cell=rbasecell,add_outputs=add_outputs,output_prefix=name+"_lr_bi_")
    UDRNN=BidirectionalCell(l_cell=ubasecell,r_cell=dbasecell,add_outputs=add_outputs,output_prefix=name+"_ud_bi_")


    if dropout!=-1:

        lrbasecelldrop = mx.rnn.DropoutCell(dropout=dropout)
        stacked_rnn_cells_LR =mx.rnn.SequentialRNNCell()
        stacked_rnn_cells_LR.add(LRRNN)
        stacked_rnn_cells_LR.add(lrbasecelldrop)

        udbasecelldrop = mx.rnn.DropoutCell(dropout=dropout)
        stacked_rnn_cells_UD =mx.rnn.SequentialRNNCell()
        stacked_rnn_cells_UD.add(UDRNN)
        stacked_rnn_cells_UD.add(udbasecelldrop)

        LRRNN = stacked_rnn_cells_LR
        UDRNN = stacked_rnn_cells_UD



    #1. reshape the input
    inshape=in_data.infer_shape()[1][0]

    if Debug:
        print "input shape:",inshape

    if inshape[2]%patch_size[0]!=0 or inshape[3]%patch_size[1]!=0:
        h_pad= inshape[2]%patch_size[0]
        w_pad= inshape[3]%patch_size[1]
        pad_out = mx.symbol.pad(in_data,mode='const',const_value=0,pad_width=(0,0,0,0,floor(h_pad/2),
                            ceil(h_pad/2),floor(w_pad/2),ceil(w_pad/2)),name=name+"_input_pad")
    else:
        pad_out=in_data

    if Debug:
        print "pad_out shape:",pad_out.infer_shape()[1]

    pad_out_shape = pad_out.infer_shape()[1][0]
    pad_out_c, pad_out_h, pad_out_w=pad_out_shape[1:]

    pheight,pwidth = patch_size

    psize = pheight * pwidth * pad_out_c

    # Number of patches in each direction
    npatchesh=pad_out_h/pheight
    npatchesw=pad_out_w/pwidth

    #2. apply left to right rnn
    # reshape pad_out into patches: bs*cc, #H, ph, #W, pw
    pad_out_reshape=mx.symbol.reshape(pad_out,shape=(-1,npatchesh,pheight,npatchesw,pwidth),name=name+"_pad_out_reshape")

    #change the axies for the pad out
    # bs*cc, #H, #W, ph, pw,
    pad_out_trans= mx.symbol.transpose(pad_out_reshape,axes=(0, 1, 3, 2, 4),name=name+"_pad_out_trans")

    # bs,cc, #H, #W, ph*pw,
    pad_out_trans_reshape = mx.symbol.reshape(pad_out_trans, shape=(-1, pad_out_c, npatchesh, npatchesw, pheight*pwidth),
                                              name=name+"_pad_out_trans_reshape")

    # bs,#H, #W,cc,ph*pw,
    pad_out_in_rnn_lr = mx.symbol.transpose(pad_out_trans_reshape, axes=(0, 2, 3, 1, 4),name=name+"_pad_out_in_rnn_lr")

    if Debug:
        print "pad_out_trans shape",pad_out_in_rnn_lr.infer_shape()[1][0]

    # The Video_Segmentation Layer needs a 3D tensor input: bs*#H, #W, psize
    # bs*#H, #W, ph * pw * cc
    datarnnlr=mx.symbol.reshape(pad_out_in_rnn_lr,shape=(-1,npatchesw,psize),name=name+"_datarnnlr")

    datarnnlr_shape=datarnnlr.infer_shape()[1][0]

    if Debug:
        print "datarnnlr shape:", datarnnlr_shape

    seq_length_lr = datarnnlr_shape[1]

    rnnlrout,rnnlrstate=LRRNN.unroll(length=seq_length_lr,inputs=datarnnlr)

    if Debug:
        for i,out_step in enumerate(rnnlrout):
            print "rnnlr output step",i+1,"shape",out_step.infer_shape()[1][0]

    #bs,#H*#W*hidden*2
    rnn_lr_out = mx.symbol.concat(*rnnlrout,dim=1,name=name+"_rnn_lr_out")
    rnn_lr_out = mx.sym.Activation(data=rnn_lr_out, act_type='relu', name=name + '_rnn_lr_out_relu')

    if Debug:
        rnn_concat_shape=rnn_lr_out.infer_shape()[1][0]
        print "rnn_concat shape",rnn_concat_shape

    # 3. apply_up and dowm rnn
    if stack:
        # bs,#H,#W,hidden*2
        rnn_out_reshape = mx.symbol.reshape(rnn_lr_out,shape=(-1,npatchesh,npatchesw,2*num_hidden if not add_outputs else num_hidden),name=name+"_rnn_out_reshape")
        # bs,#W,#H,hidden*2
        rnn_out_trans= mx.symbol.transpose(rnn_out_reshape,axes=(0,2,1,3),name=name+"_rnn_out_trans")
        # bs*#W,#H,hidden*2
        rnn_ud_in=mx.symbol.reshape(rnn_out_trans,shape=(-1,npatchesh,2*num_hidden if not add_outputs else num_hidden),name=name+"_rnn_ud_input")

    else:
        # bs,#W, #H,cc,ph*pw,
        pad_out_in_rnn_ud = mx.symbol.transpose(pad_out_in_rnn_lr,axes=(0,2,1,3,4),name=name+"_pad_out_in_rnn_ud")
        # bs*#W, #H,cc*ph*pw,
        rnn_ud_in = mx.symbol.reshape(pad_out_in_rnn_ud,shape=(-1,npatchesh,psize),name=name+"_rnn_ud_input")

    rnn_ud_shape=rnn_ud_in.infer_shape()[1][0]

    if Debug:
        print "rnn_ud_in shape",rnn_ud_shape

    seq_length_ud = rnn_ud_shape[1]
    rnnudout, rnnudstate = UDRNN.unroll(length=seq_length_ud, inputs=rnn_ud_in)

    if Debug:
        for i, out_step in enumerate(rnnudout):
            print "rnnud output step", i + 1, "shape", out_step.infer_shape()[1][0]

    # bs,#W*#H*hidden*2
    rnn_ud_concat = mx.symbol.concat(*rnnudout, dim=1, name=name+"_rnn_concat_ud")

    if Debug:
        rnn_ud_concat_shape = rnn_ud_concat.infer_shape()[1][0]
        print "rnn_ud_concat shape", rnn_ud_concat_shape

    #4. reshape for the output
    # bs,#W,#H,hidden*2
    rnn_ud_reshape = mx.symbol.reshape(rnn_ud_concat,shape=(-1,npatchesw,npatchesh,
                                                            2*num_hidden if not add_outputs else num_hidden),
                                       name=name+"rnn_ud_reshape")
    # bs,hidden*2,#H,#W
    rnn_out = mx.symbol.transpose(rnn_ud_reshape,axes=(0,3,2,1),name=name+"_rnn_out")
    rnn_out = mx.sym.Activation(data=rnn_out, act_type='relu', name=name + '_rnn_out_relu')

    if Debug:
        rnn_out_shape = rnn_out.infer_shape()[1][0]
        print "output shape",rnn_out_shape

    return rnn_out

import numpy as np
class ReLayerCase(unittest.TestCase):
    def setUp(self):
       pass


    def tearDown(self):
        pass

    # def testmnist(self):
    #     # set up logger
    #     logging.basicConfig()
    #     logger = logging.getLogger()
    #     logger.setLevel(logging.INFO)
    #
    #     test_data = mx.io.MNISTIter(
    #         image="/home/zyq/workspace/mxnet_space/Video_Segmentation/data/mnist/t10k-images-idx3-ubyte",
    #         label="/home/zyq/workspace/mxnet_space/Video_Segmentation/data/mnist/t10k-labels-idx1-ubyte")
    #     data = mx.io.MNISTIter(
    #         image="/home/zyq/workspace/mxnet_space/Video_Segmentation/data/mnist/train-images-idx3-ubyte",
    #         label="/home/zyq/workspace/mxnet_space/Video_Segmentation/data/mnist/train-labels-idx1-ubyte")
    #     print data.provide_data
    #     print data.provide_label
    #     data_shape_dict = dict(data.provide_data)
    #
    #     input = mx.symbol.Variable(name='data', shape=data_shape_dict['data'])
    #     label = mx.sym.Variable('softmax_label')
    #
    #     relay = ReLayer(in_data=input, name='relayer1', type='RNN', patch_size=(2, 2), num_hidden=20,
    #                          stack=True)
    #
    #     flantten = mx.symbol.flatten(relay)
    #
    #     fc2 = mx.symbol.FullyConnected(flantten, num_hidden=200, name="fc2")
    #     fc2a = mx.symbol.Activation(fc2, act_type='relu', name="fc2_act")
    #
    #     fc3 = mx.symbol.FullyConnected(fc2a, num_hidden=10, name="fc3")
    #     fc3a = mx.symbol.Activation(fc3, act_type='relu', name="fc3_act")
    #
    #     out = mx.symbol.SoftmaxOutput(data=fc3a, label=label, name='softmax')
    #
    #     mod = mx.mod.Module(out, context=mx.gpu(0))
    #
    #     train_iter = data
    #
    #     mod.fit(train_iter,num_epoch=50,validation_metric='acc',
    #                  initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),batch_end_callback = mx.callback.Speedometer(128, 10))
    #     mod.score(test_data,eval_metric='acc',num_batch=300)


    def testconst(self):

        input_data = mx.symbol.Variable(name='data', shape=(1,3,4,4))

        input_data = mx.symbol.Convolution(input_data,kernel=(1,1),name="conv1",num_filter=3)

        sym = ReLayer(in_data=input_data, name='relayer', type='RNN', patch_size=(1, 1), num_hidden=2,
                             stack=True,use_memory=False,add_outputs=True)

        # sym = mx.symbol.Convolution(sym, kernel=(1, 1), name="conv2", num_filter=2)

        # loss = mx.sym.MakeLoss(sym,name="loss")
        # sym = mx.sym.SoftmaxOutput(sym, name='softmax')

        print "total arguments:",sym.list_arguments()
        # print "iput arguments:",input.list_arguments()


        # print test_data.asnumpy()
        # excutor = input.bind(mx.cpu(),{'data':test_data})
        #
        # input_gt=excutor.forward()

        print "relay arguments:",sym.list_arguments()
        print "relay shape:",sym.infer_shape()

        # i2h_weight = mx.nd.array(np.zeros((80,4)))
        # i2h_bias = mx.nd.array(np.zeros((80)))
        # h2h_weight = mx.nd.array(np.zeros((80, 4)))
        # h2h_bias = mx.nd.array(np.zeros((80)))

        args={}
        for arg_name,arg_shape  in zip(sym.list_arguments(),sym.infer_shape()[0]):
            if arg_name=='data':
                continue
            if arg_name.endswith("bias"):
                args[arg_name] = mx.nd.array(np.zeros(arg_shape))
            elif arg_name.endswith("h2h_weight"):
                args[arg_name] = mx.nd.array(np.zeros(arg_shape))
            else:
                args[arg_name]=mx.nd.array(np.ones(arg_shape))
        test_data = mx.nd.array(np.ones((1, 3, 4, 4)))
        # test_data = mx.nd.array(np.zeros((1,3,4,4)))
        args["data"]= test_data

        for k,v in  args.items():
            print "--------------------------"
            print k,v.asnumpy()

        # sym_group = loss.get_internals()
        print "=================== module ========================"

        # train_iter = mx.io.NDArrayIter(test_data, None, 1, shuffle=True)
        # mod = mx.mod.Module(symbol=sym_group,context=mx.cpu(),data_names=["data"],label_names=None)
        # mod.bind(data_shapes=train_iter.provide_data)
        # mod.set_params(arg_params=args,aux_params=None)

        # val_iter = mx.io.NDArrayIter(test_data, None, 1)
        executor = sym.simple_bind(ctx=mx.cpu(),grad_req='write')


        print "=================== forward ========================"

        executor.forward(**args)
        # data_batch = train_iter.next()
        # mod.forward(is_train=True,data_batch=data_batch)

        for name, output in zip(sym.list_arguments(),executor.outputs):
            print "---------------------------"
            print name
            print output.asnumpy()
        # print relay_gt[0].asnumpy()

        print "=================== backward ========================"

        executor.backward([mx.nd.ones((1, 2, 4, 4))*3])

        for name ,array in zip(sym.list_arguments(),executor.grad_arrays):
            print "--------------------"
            print name
            print array.asnumpy()

        # executor.backward([mx.nd.ones((1, 2, 4, 4))*3])
        #
        # for name ,array in zip(sym.list_arguments(),executor.grad_arrays):
        #     print "--------------------"
        #     print name
        #     print array.asnumpy()

# if __name__=='__main__':
#     unittest.main()