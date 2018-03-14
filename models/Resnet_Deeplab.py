import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from .base_model import BaseModel, load_pretrained_model
import numpy as np
import shutil
from utils import util
from collections import OrderedDict
from tensorboardX import SummaryWriter
import os
from torch.autograd import Variable
from .ops.depthconv.modules import DepthConv
from .ops.depthavgpooling.modules import Depthavgpooling


affine_par = True


def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        # for i in self.bn2.parameters():
        #     i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        # for i in self.bn3.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DepthConvBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(DepthConvBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False

        padding = dilation
        self.conv2 = DepthConv(planes,planes,kernel_size=3,stride=stride,padding=padding, dilation = dilation, bias=False)
            # nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            #                    padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # for i in self.bn2.parameters():
        #     i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        # for i in self.bn3.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x_depth):
        x, depth = x_depth
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # print('d',out.size(), depth.size())
        out = self.conv2(out,depth)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes, inplanes, depthconv=False):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            if depthconv:
                conv = DepthConv(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias= True)
            else:
                conv = nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True)
            self.conv2d_list.append(conv)

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class Residual_Covolution(nn.Module):
    def __init__(self, icol, ocol, num_classes):
        super(Residual_Covolution, self).__init__()
        self.conv1 = nn.Conv2d(icol, ocol, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv2 = nn.Conv2d(ocol, num_classes, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv3 = nn.Conv2d(num_classes, ocol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv4 = nn.Conv2d(ocol, icol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dow1 = self.conv1(x)
        dow1 = self.relu(dow1)
        seg = self.conv2(dow1)
        inc1 = self.conv3(seg)
        add1 = dow1 + self.relu(inc1)
        inc2 = self.conv4(add1)
        out = x + self.relu(inc2)
        return out, seg

class Residual_Refinement_Module(nn.Module):

    def __init__(self, num_classes):
        super(Residual_Refinement_Module, self).__init__()
        self.RC1 = Residual_Covolution(2048, 512, num_classes)
        self.RC2 = Residual_Covolution(2048, 512, num_classes)

    def forward(self, x):
        x, seg1 = self.RC1(x)
        _, seg2 = self.RC2(x)
        return [seg1, seg1+seg2]

class ResNet_Refine(nn.Module):
    def __init__(self, block, layers, num_classes, depthconv=False):
        self.inplanes = 64
        super(ResNet_Refine, self).__init__()
        self.depthconv = depthconv
        if depthconv:
            self.conv1 = DepthConv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        # self.layer1 = self._make_layer(block, 64, layers[0])
        if depthconv:
            self.downsample_depth1 = nn.AvgPool2d(5,padding=1,stride=4)
            self.layer1 = self._make_layer_depthconv(64, layers[0])
        else:
            self.layer1 = self._make_layer(block, 64, layers[0])

        if depthconv:
            self.downsample_depth2 = nn.AvgPool2d(3,padding=1,stride=2)
            self.layer2 = self._make_layer_depthconv(128, layers[1], stride=2)
        else:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        if depthconv:
            # self.downsample_depth3 = nn.AvgPool2d(3,padding=1,stride=2)
            self.layer3 = self._make_layer_depthconv(256, layers[2], stride=1, dilation=2)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)

        if depthconv:
            # self.downsample_depth4 = nn.AvgPool2d(3,padding=1,stride=2)
            self.layer4 = self._make_layer_depthconv(512, layers[3], stride=1, dilation=4)
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.dropout = nn.Dropout()
        self.layer5 = Residual_Refinement_Module(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # for i in m.parameters():
                #     i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        # for i in downsample._modules['1'].parameters():
        #     i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_layer_depthconv(self, planes, blocks, stride=1, dilation = 1):
        downsample = None
        block = DepthConvBottleneck
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, depth=None):
        if self.depthconv:
            x = self.conv1(x,depth)
        else:
            x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.depthconv:
            depth = self.downsample_depth1(depth)
            x,depth = self.layer1((x,depth))
        else:
            x = self.layer1(x)

        if self.depthconv:
            depth = self.downsample_depth2(depth)
            x,depth = self.layer2((x,depth))
        else:
            x = self.layer2(x)
        if self.depthconv:
            depth = self.downsample_depth3(depth)
            x,_ = self.layer3((x,depth))
        else:
            x = self.layer3(x)
        if self.depthconv:
            x,_ = self.layer4((x,depth))
        else:
            x = self.layer4(x)
        x = self.dropout(x)
        x = self.layer5(x)

        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, depthconv=False, globalpooling=False, pretrain=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.depthconv = depthconv
        if depthconv:
            self.conv1 = DepthConv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True) # change

        if depthconv:
            self.layer1 = self._make_layer_depthconv(block, 64, layers[0])
            self.downsample_depth1 = nn.AvgPool2d(5,padding=1,stride=4)
        else:
            self.layer1 = self._make_layer(block, 64, layers[0])

        if depthconv:
            self.layer2 = self._make_layer_depthconv(block, 128, layers[1], stride=2)
            self.downsample_depth2 = nn.AvgPool2d(3,padding=1,stride=2)
        else:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)


        if depthconv:
            self.layer3 = self._make_layer_depthconv(block, 256, layers[2], stride=1, dilation=2)
            self.downsample_depth3 = nn.AvgPool2d(3,padding=1,stride=2)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)

        if depthconv:
            # self.downsample_depth4 = nn.AvgPool2d(3,padding=1,stride=2)
            self.layer4 = self._make_layer_depthconv(block, 512, layers[3], stride=1, dilation=4)
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.globalpooling = globalpooling
        if globalpooling:
            self.globalpooling = nn.AdaptiveMaxPool2d((1,1))
            self.inplanes *= 2
        self.dropout = nn.Dropout()
        self.layer5 = self._make_pred_layer(Classifier_Module, [12],[12],num_classes,self.inplanes)
        # self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes,self.inplanes)

        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1,padding=1)
        self.pool5a_d = Depthavgpooling(kernel_size=3, stride=1,padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))#(0, 0.01)#
                torch.nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if affine_par:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                for i in m.parameters():
                    i.requires_grad = False
        if pretrain:
            load_pretrained_model(self,
                                  model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'),
                                  False)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        # for i in downsample._modules['1'].parameters():
        #     i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_layer_depthconv(self, block,  planes, blocks, stride=1, dilation=1):
        downsample = None
        # block = DepthConvBottleneck
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(DepthConvBottleneck(self.inplanes, planes, stride, downsample,dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes,inplanes):
        return block(dilation_series,padding_series,num_classes,inplanes)

    def forward(self, x, depth=None):
        # print self.layer3._modules.values()[13].bn2.running_mean
        if self.depthconv:
            x = self.conv1(x,depth)
        else:
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.depthconv:
            depth = self.downsample_depth1(depth)
            x = self.layer1((x,depth))
        else:
            x = self.layer1(x)

        if self.depthconv:
            # print ('o',x.size(), depth.size())
            # depth = self.downsample_depth2(depth)
            x = self.layer2((x,depth))
        else:
            x = self.layer2(x)

        if self.depthconv:
            depth = self.downsample_depth3(depth)
            x = self.layer3((x,depth))
        else:
            x = self.layer3(x)

        if self.depthconv:
            x = self.layer4((x,depth))
        else:
            x = self.layer4(x)

        if self.globalpooling:
            x_size = x.size()
            globalpool = self.globalpooling(x).repeat(1,1,x_size[2],x_size[3])
            x = torch.cat([x,globalpool], 1)
        x = self.dropout(x)
        x = self.layer5(x)
        if self.depthconv:
            x = self.pool5a_d(x,depth)
        else:
            x = self.pool5a(x)


        return x

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                if isinstance(j, nn.Conv2d):
                    for k in j.parameters():
                        if k.requires_grad:
                            yield k

    def get_bn_params(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)
        b.append(self.layer5)

        for i in range(len(b)):
            for j in b[i].modules():
                if isinstance(j, nn.BatchNorm2d):
                    for n, k in j.named_parameters():
                        # print n
                        if k.requires_grad:
                            yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5)

        # for j in range(len(b)):
        #     for i in b[j]:
        #         yield i

        for i in range(len(b)):
            for j in b[i].modules():
                if isinstance(j, nn.Conv2d):
                    if j.weight is not None:
                        if j.weight.requires_grad:
                            yield j.weight
                # for k in j.parameters():
                #     if k.requires_grad:
                #         yield k

    def get_20x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5)

        # for j in range(len(b)):
        #     for i in b[j]:
        #         yield i

        for i in range(len(b)):
            for j in b[i].modules():
                if isinstance(j, nn.Conv2d):
                    if j.bias is not None:
                        if j.bias.requires_grad:
                            yield j.bias