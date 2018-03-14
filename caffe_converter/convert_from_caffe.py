#!/usr/bin/env python

import os.path as osp
import caffe
import torch
import numpy as np
import torchvision
from models import Deeplab
from collections import OrderedDict
import caffe.proto.caffe_pb2 as caffe_pb2
model = open('/home/laughtervv/Downloads/deeplab_global_baseline_vgg_37.08/train_large_iter_25000_vgg.caffemodel', 'rb')
net_param = caffe_pb2.NetParameter()
net_param.ParseFromString(model.read())
assert (net_param.layer[42].name =='pool6_1x1_norm')

caffe_prototxt = 'train.prototxt'  # NOQA
caffe_model_path = 'train_iter_20000.caffemodel'

caffe.set_mode_cpu()
caffe_model = caffe.Net(caffe_prototxt, caffe_model_path, caffe.TEST)
caffe_model.forward()

torch_model = Deeplab.Deeplab_SS(None, 20, False, vgg=True) # torchvision.models.vgg16()
torch_model_params = torch_model.named_parameters()
W = caffe_model.params['conv1_1'][0].data
print np.mean(W)
newdict  = OrderedDict()
for name, p1 in caffe_model.params.iteritems():
    try:
        p2 = torch_model_params.next()
        print('%s: %s -> %s %s' % (name, p1[0].data.shape, p2[0], p2[1].data.size()))
        p2[1].data = torch.from_numpy(p1[0].data).float()
        print(np.mean(p1[0].data))
        if len(p1) == 2:
            p2 = torch_model_params.next()
            print('%s: %s ->%s %s' % (name, p1[1].data.shape, p2[0], p2[1].data.size()))
            p2[1].data = torch.from_numpy(p1[1].data)
            print(np.mean(p1[1].data),)
    except StopIteration:
        break

torch_model_path = 'DeepLab_VGG_caffe.pth'
torch.save(torch_model.state_dict(), torch_model_path)

