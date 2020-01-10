#!/bin/env python3

import caffe
from caffe import layers as L, params as P
import caffe.proto.caffe_pb2 as pb2
import re

default_weight_filler = dict(type = 'msra')

def BatchNorm(ns, in_place = False, ext_scale = False):
    prefix = ns.keys()[-1]
    bn_params = dict(
        in_place = in_place,
        use_global_stats = True,
        eps = 1e-5)
    ns[f'{prefix}/bn'] = L.BatchNorm(ns[prefix], **bn_params)
    scale_ext = dict(
        filler = dict(value = 1),
        bias_filler = dict(value = 0)) if ext_scale else {}
    scale_params = dict(
        in_place = True,
        bias_term = True,
        param = [dict(lr_mult = 1, decay_mult = 0) for i in range(2)],
        **scale_ext)
    ns[f'{prefix}/scale'] = L.Scale(ns[f'{prefix}/bn'], **scale_params)

def get_conv_index(ns):
    lns = list(n for n in ns.keys() if n.startswith('conv'))
    conv_index = re.search('^\d+', max(lns).lstrip('conv')).group(0)
    return int(conv_index)

def dw(ns, c, s):
    dw_params = dict(
        num_output = c,
        group = c,
        stride = s,
        bias_term = False,
        weight_filler = default_weight_filler,
        kernel_size = 3,
        pad = 1,
        engine = pb2.ConvolutionParameter.CAFFE)
    conv_index = get_conv_index(ns) + 1
    ns[f'conv{conv_index:02d}/dw'] = L.Convolution(ns[ns.keys()[-1]], **dw_params)
    BatchNorm(ns, True, True)
    ns[f'relu{conv_index:02d}/dw'] = L.ReLU(ns[ns.keys()[-1]], in_place=True)

def sp(ns, c):
    sep_params = dict(
        num_output = c,
        bias_term = False,
        pad = 0,
        weight_filler = default_weight_filler,
        kernel_size = 1,
        stride = 1)
    conv_index = get_conv_index(ns)
    ns[f'conv{conv_index:02d}/sep'] = L.Convolution(ns[ns.keys()[-1]], **sep_params)
    BatchNorm(ns, True, True)
    ns[f'relu{conv_index:02d}/sep'] = L.ReLU(ns[ns.keys()[-1]], in_place=True)

def head(in_place_bn = False, index_width = 1, scale_ext = False):
    ns = caffe.NetSpec()
    input_params = dict(shape = dict(dim = [1, 3, 224, 224]))
    ns.data = L.Input(**input_params)
    conv1_params = dict(
        kernel_size = 3,
        num_output = 32,
        stride = 2,
        pad = 1,
        weight_filler = default_weight_filler,
        bias_term = False)
    ns[f'conv{1:0{index_width}d}'] = L.Convolution(ns.data, **conv1_params)
    BatchNorm(ns, in_place_bn, scale_ext)
    ns[f'relu{1:0{index_width}d}/relu'] = L.ReLU(ns[ns.keys()[-1]], in_place=True)
    return ns

def tail(ns, index_width = 1):
    pool_params = dict(
        pool = pb2.PoolingParameter.AVE,
        global_pooling = True)
    ns['avgpool'] = L.Pooling(ns[ns.keys()[-1]], **pool_params)
    conv10_params = dict(
        kernel_size = 1,
        num_output = 1000,
        weight_filler = default_weight_filler,
        bias_filler = dict(
            type = 'constant',
            value = 0),
        param = dict(lr_mult = 2, decay_mult = 0))
    conv_index = get_conv_index(ns) + 1
    ns[f'conv{conv_index:0{index_width}d}'] = L.Convolution(ns[ns.keys()[-1]], **conv10_params)
    ns['prob'] = L.Softmax(ns[ns.keys()[-1]])

def block(ns, in_plans, table_row):
    t, c, n, s = table_row
    conv_index = get_conv_index(ns) + 1
    for i in range(n):
        depth = in_plans * t if i == 0 else c * t
        bottom = ns.keys()[-1].replace('scale', 'bn')
        expand_params = dict(
            num_output = depth,
            bias_term = False,
            kernel_size = 1,
            weight_filler = default_weight_filler)
        ns[f'conv{conv_index}_{i+1}/expand'] = L.Convolution(ns[ns.keys()[-1]], **expand_params)
        BatchNorm(ns)
        ns[f'relu{conv_index}_{i+1}/expand'] = L.ReLU(ns[ns.keys()[-1]], in_place=True)
        dwise_params = dict(
            num_output = depth,
            bias_term = False,
            pad = 1,
            kernel_size = 3,
            group = depth,
            engine = pb2.ConvolutionParameter.CAFFE,
            weight_filler = default_weight_filler)
        if i == 0 and s != 1:
            dwise_params['stride'] = s
        ns[f'conv{conv_index}_{i+1}/dwise'] = L.Convolution(ns[ns.keys()[-1]], **dwise_params)
        BatchNorm(ns)
        ns[f'relu{conv_index}_{i+1}/dwise'] = L.ReLU(ns[ns.keys()[-1]], in_place=True)
        linear_params = dict(
            num_output = c,
            bias_term = False,
            kernel_size = 1,
            weight_filler = default_weight_filler)
        ns[f'conv{conv_index}_{i+1}/linear'] = L.Convolution(ns[ns.keys()[-1]], **linear_params)
        BatchNorm(ns)
        if i == 0 and in_plans == c and s == 1 or i != 0:
            ns[f'conv{conv_index}_{i}/plus'] = L.Eltwise(ns[ns.keys()[-1]], bottom=bottom)

def mobilenetv1():
    ns = head(in_place_bn = True, index_width = 2, scale_ext = True)
    dw(ns, 32,  1)
    sp(ns, 64)
    dw(ns, 64,  2)
    sp(ns, 128)
    dw(ns, 128, 1)
    sp(ns, 128)
    dw(ns, 128, 2)
    sp(ns, 256)
    dw(ns, 256, 1)
    sp(ns, 256)
    dw(ns, 256, 2)
    for i in range(5):
        sp(ns, 512)
        dw(ns, 512, 1)
    sp(ns, 512)
    dw(ns, 512, 2)
    sp(ns, 1024)
    dw(ns, 1024, 1)
    sp(ns, 1024)
    tail(ns)
    return ns

def mobilenetv2():
    ns = head()
    block(ns, 32,  (1,  16, 1, 1)) # conv2
    block(ns, 16,  (6,  24, 2, 2)) # conv3
    block(ns, 24,  (6,  32, 3, 2)) # conv4
    # block(ns, 32,  (6,  64, 4, 2))
    block(ns, 32,  (6,  64, 4, 1)) # conv5, caffe implementation diffs from paper
    # block(ns, 64,  (6,  96, 3, 1))
    block(ns, 64,  (6,  96, 3, 2)) # conv6, caffe implementation diffs from paper
    block(ns, 96,  (6, 160, 3, 2)) # conv7
    block(ns, 160, (6, 320, 1, 1)) # conv8
    conv9_params = dict(
        kernel_size = 1,
        num_output = 1280,
        weight_filler = default_weight_filler,
        bias_term = False)
    ns.conv9 = L.Convolution(ns[ns.keys()[-1]], **conv9_params)
    BatchNorm(ns)
    ns['conv9/relu'] = L.ReLU(ns[ns.keys()[-1]], in_place=True)
    tail(ns)
    return ns

from google.protobuf import text_format
def generate(v = 2):
    with open(f'mobilenetv{v}.prototxt', 'w') as f:
        if v == 1:
            net = mobilenetv1().to_proto()
        elif v == 2:
            net = mobilenetv2().to_proto()
        else:
            raise Excpetion('Yet to come')
        f.write(text_format.MessageToString(net))

def copy_weight(v):
    import sys
    import os
    if len(sys.argv) != 3:
        print(f'{sys.argv[0]} <pretrained proto> <pretrained weights>')
        return
    pretrained_files = sys.argv[1:]
    for f in pretrained_files:
        if not os.path.isfile(f):
            print(f'file {f} not exist')
            return
    pretrained = caffe.Net(*pretrained_files, caffe.TEST)
    net = caffe.Net(f'mobilenetv{v}.prototxt', caffe.TEST)
    if len(pretrained.params) != len(net.params):
        print(f'layer param num dismatch')
        return
    for pt_ln, ln in zip(pretrained.params, net.params):
        print(f'copy params {pt_ln:24} into {ln}')
        if len(net.params[ln]) != len(pretrained.params[pt_ln]):
            print(f'bias term existence dismatch')
            return
        for i in range(len(net.params[ln])):
            net.params[ln][i].data[...] = pretrained.params[pt_ln][i].data[...]
    net.save(f'mobilenetv{v}.caffemodel')

if __name__ == '__main__':
    v = 2
    generate(v)
    copy_weight(v)
