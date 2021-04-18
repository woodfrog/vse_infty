import os
import torch
import torch.nn as nn
import math

import torch.utils.model_zoo as model_zoo
import logging

logger = logging.getLogger(__name__)

__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):
    def __init__(self, block, layers, width_mult, num_classes=1000):
        self.inplanes = 64 * width_mult
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64 * width_mult, layers[0])
        self.layer2 = self._make_layer(block, 128 * width_mult, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * width_mult, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * width_mult, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion * width_mult, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False, width_mult=1):
    """Constructs a ResNet-50 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], width_mult)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, width_mult=1):
    """Constructs a ResNet-101 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], width_mult)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, width_mult=1):
    """Constructs a ResNet-152 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], width_mult)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class ResnetFeatureExtractor(nn.Module):
    def __init__(self, backbone_source, weights_path, pooling_size=7, fixed_blocks=2):
        super(ResnetFeatureExtractor, self).__init__()
        self.backbone_source = backbone_source
        self.weights_path = weights_path
        self.pooling_size = pooling_size
        self.fixed_blocks = fixed_blocks

        if 'detector' in self.backbone_source:
            self.resnet = resnet101()
        elif self.backbone_source == 'imagenet':
            self.resnet = resnet101(pretrained=True)
        elif self.backbone_source == 'imagenet_res50':
            self.resnet = resnet50(pretrained=True)
        elif self.backbone_source == 'imagenet_res152':
            self.resnet = resnet152(pretrained=True)
        elif self.backbone_source == 'imagenet_resnext':
            self.resnet = torch.hub.load('pytorch/vision:v0.4.2', 'resnext101_32x8d', pretrained=True)
        elif 'wsl' in self.backbone_source:
            self.resnet = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        else:
            raise ValueError('Unknown backbone source {}'.format(self.backbone_source))

        self._init_modules()

    def _init_modules(self):
        # Build resnet.
        self.base = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu,
                                  self.resnet.maxpool, self.resnet.layer1, self.resnet.layer2, self.resnet.layer3)
        self.top = nn.Sequential(self.resnet.layer4)

        if self.weights_path != '':
            if 'detector' in self.backbone_source:
                if os.path.exists(self.weights_path):
                    logger.info(
                        'Loading pretrained backbone weights from {} for backbone source {}'.format(self.weights_path,
                                                                                                    self.backbone_source))
                    backbone_ckpt = torch.load(self.weights_path)
                    self.base.load_state_dict(backbone_ckpt['base'])
                    self.top.load_state_dict(backbone_ckpt['top'])
                else:
                    raise ValueError('Could not find weights for backbone CNN at {}'.format(self.weights_path))
            else:
                logger.info('Did not load external checkpoints')
        self.unfreeze_base()

    def set_fixed_blocks(self, fixed_blocks):
        self.fixed_blocks = fixed_blocks

    def get_fixed_blocks(self):
        return self.fixed_blocks

    def unfreeze_base(self):
        assert (0 <= self.fixed_blocks < 4)
        if self.fixed_blocks == 3:
            for p in self.base[6].parameters(): p.requires_grad = False
            for p in self.base[5].parameters(): p.requires_grad = False
            for p in self.base[4].parameters(): p.requires_grad = False
            for p in self.base[0].parameters(): p.requires_grad = False
            for p in self.base[1].parameters(): p.requires_grad = False
        if self.fixed_blocks == 2:
            for p in self.base[6].parameters(): p.requires_grad = True
            for p in self.base[5].parameters(): p.requires_grad = False
            for p in self.base[4].parameters(): p.requires_grad = False
            for p in self.base[0].parameters(): p.requires_grad = False
            for p in self.base[1].parameters(): p.requires_grad = False
        if self.fixed_blocks == 1:
            for p in self.base[6].parameters(): p.requires_grad = True
            for p in self.base[5].parameters(): p.requires_grad = True
            for p in self.base[4].parameters(): p.requires_grad = False
            for p in self.base[0].parameters(): p.requires_grad = False
            for p in self.base[1].parameters(): p.requires_grad = False
        if self.fixed_blocks == 0:
            for p in self.base[6].parameters(): p.requires_grad = True
            for p in self.base[5].parameters(): p.requires_grad = True
            for p in self.base[4].parameters(): p.requires_grad = True
            for p in self.base[0].parameters(): p.requires_grad = True
            for p in self.base[1].parameters(): p.requires_grad = True

        logger.info('Resnet backbone now has fixed blocks {}'.format(self.fixed_blocks))

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        # Override train so that the training mode is set as we want (BN does not update the running stats)
        nn.Module.train(self, mode)
        if mode:
            # fix all bn layers
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.base.apply(set_bn_eval)
            self.top.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        fc7 = self.top(pool5).mean(3).mean(2)
        return fc7

    def forward(self, im_data):
        b_s = im_data.size(0)
        base_feat = self.base(im_data)
        top_feat = self.top(base_feat)
        features = top_feat.view(b_s, top_feat.size(1), -1).permute(0, 2, 1)
        return features


if __name__ == '__main__':
    import numpy as np

    def count_params(model):
        model_parameters = model.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    model = resnet50(pretrained=False, width_mult=1)
    num_params = count_params(model)

