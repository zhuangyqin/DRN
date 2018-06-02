# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------
import mxnet as mx
import numpy as np
from copy import deepcopy
Debug = 0
class Symbol:
    def __init__(self):
        self.arg_shape_dict = None
        self.out_shape_dict = None
        self.aux_shape_dict = None
        self.sym = None

    @property
    def symbol(self):
        return self.sym

    def get_symbol(self, cfg, is_train=True):
        """
        return a generated symbol, it also need to be assigned to self.sym
        """
        raise NotImplementedError()

    def init_weights(self, cfg, arg_params, aux_params):
        raise NotImplementedError()

    def get_msra_std(self, shape):
        fan_in = float(shape[1])
        if len(shape) > 2:
            fan_in *= np.prod(shape[2:])
        print(np.sqrt(2 / fan_in))
        return np.sqrt(2 / fan_in)

    def infer_shape(self, data_shape_dict):
        # infer shape
        self.data_shape_dict = data_shape_dict
        arg_shape, out_shape, aux_shape = self.sym.infer_shape(**data_shape_dict)
        self.arg_shape_dict = dict(zip(self.sym.list_arguments(), arg_shape))
        self.out_shape_dict = dict(zip(self.sym.list_outputs(), out_shape))
        self.aux_shape_dict = dict(zip(self.sym.list_auxiliary_states(), aux_shape))
        print "========================="
        print "output shape:",self.out_shape_dict
        print "========================="


    def check_parameter_shapes(self, arg_params, aux_params, data_shape_dict, is_train=True):
        print arg_params.keys()
        print self.sym.list_arguments()
        for k in self.sym.list_arguments():
            if k in data_shape_dict or (False if is_train else 'label' in k):
                continue
            if k.endswith('label'):
                continue
            if k.endswith('state'):
                arg_params[k] = mx.nd.zeros(self.arg_shape_dict[k])
            assert k in arg_params, k + ' not initialized'
            assert arg_params[k].shape == self.arg_shape_dict[k], \
                'shape inconsistent for ' + k + ' inferred ' + str(self.arg_shape_dict[k]) + ' provided ' + str(
                    arg_params[k].shape)
        for k in self.sym.list_auxiliary_states():
            assert k in aux_params, k + ' not initialized'
            assert aux_params[k].shape == self.aux_shape_dict[k], \
                'shape inconsistent for ' + k + ' inferred ' + str(self.aux_shape_dict[k]) + ' provided ' + str(
                    aux_params[k].shape)

    def sym_gen(self, cfg, is_train = False):
        """
        get symbol for testing
        :param num_classes: num of classes
        :return: the symbol for testing
        """
        config = deepcopy(cfg)
        print config
        def sym_generator(data_shapes):
            if Debug:
                print data_shapes[0]
            input_shapes = [dict(data_shapes[i]) for i in xrange(len(data_shapes))]
            len_data_shapes = len(input_shapes)
            data_shapes_anchor = input_shapes[0]
            shape_same_flag = True
            for i in xrange(len_data_shapes):
                now_data_shape = input_shapes[i]
                for k, v in now_data_shape.items():
                    if v != data_shapes_anchor[k]:
                        shape_same_flag = False
            assert shape_same_flag == True, "now dymanic shape only support for the same input in the all gpus"
            config.SCALES = [(data_shapes_anchor['data'][2],data_shapes_anchor['data'][3])]
            if Debug:
                print "config.SCALES",config.SCALES
            return  self.get_symbol(config,is_train=is_train)


        return sym_generator
