from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from collections import OrderedDict

from Model.DPNet.adaptive_avgmax_pool import adaptive_avgmax_pool2d


__all__ = ['DPN', 'dpn68', 'dpn68b', 'dpn131' ]


model_urls = {
    'dpn68':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn68-66bebafa7.pth',
    'dpn68b-extra':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn68b_extra-84854c156.pth',

    'dpn131':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn131-71dfe43e0.pth',

}


def dpn68(pretrained=False, test_time_pool=False, **kwargs):
    """Constructs a DPN-68 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time

        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['dpn68']))
    return model


def dpn68b(pretrained=False, test_time_pool=False, **kwargs):
    """Constructs a DPN-68b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time

        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        b=True, k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['dpn68b-extra']))
    return model


def dpn131(pretrained=False, test_time_pool=False, **kwargs):
    """Constructs a DPN-131 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time

        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        num_init_features=128, k_r=160, groups=40,
        k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['dpn131']))
    return model


class CatBnAct(nn.Module):
    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride,
                 padding=0, groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):
    def __init__(self, num_init_features, kernel_size=7,
                 padding=3, activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(
            3, num_init_features, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    def __init__(
            self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type == 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type == 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type == 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(
            in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3,
            stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        # print(resid.size())
        # print(dense.size())
        return resid, dense


class DPN(nn.Module):
    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32,
                 b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
                 num_classes=1000, test_time_pool=False):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4

        blocks = OrderedDict()

        # conv1
        if small:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=3, padding=1)
        else:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=7, padding=3)

        # conv2
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv3
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv4
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv5
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        blocks['conv5_bn_ac'] = CatBnAct(in_chs)

        self.features = nn.Sequential(blocks)

        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        self.classifier = nn.Conv2d(832, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.features(x)
        if not self.training and self.test_time_pool:
            x = F.avg_pool2d(x, kernel_size=7, stride=1)
            out = self.classifier(x)
            # The extra test time pool should be pooling an img_size//32 - 6 size patch
            out = adaptive_avgmax_pool2d(out, pool_type='avgmax')
        else:
            x = adaptive_avgmax_pool2d(x, pool_type='avg')
            out = self.classifier(x)
        return out.view(out.size(0), -1)


class TransferModel(nn.Module):

    def __init__(self, num_out_classes=2, dropout=0.0, weight_norm=False):
        super(TransferModel, self).__init__()

        self.model = dpn68(num_classes=num_out_classes)
        self.test_time_pool = self.model.test_time_pool

        weights = r"F:\HH\dpn68.pth"
        if os.path.exists(weights):
            weights_dict = torch.load(weights, map_location="cuda")
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if self.model.state_dict()[k].numel() == v.numel()}
            print(self.model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(weights))

        del self.model.classifier

        self.last_conv = nn.Conv2d(832, 832, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.last_linear = nn.Linear(832, num_out_classes)
        num_ftrs = 832

        print('Using dropout', dropout)
        if weight_norm:
            print('Using Weight_Norm')
            self.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.utils.weight_norm(
                    nn.Linear(num_ftrs, num_out_classes), name='weight')
            )

        self.last_linear = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_ftrs, num_out_classes)
        )

    def fea_conv1x(self, x):
        x = self.model.features.conv1_1(x)

        return x

    def fea_conv2x(self, x):
        x = self.model.features.conv2_1(x)
        x = self.model.features.conv2_2(x)
        x = self.model.features.conv2_3(x)

        return x

    def fea_conv3x(self, x):
        x = self.model.features.conv3_1(x)
        x = self.model.features.conv3_2(x)
        x = self.model.features.conv3_3(x)
        x = self.model.features.conv3_4(x)

        return x

    def fea_conv4x(self, x):
        x = self.model.features.conv4_1(x)
        x = self.model.features.conv4_2(x)
        x = self.model.features.conv4_3(x)
        x = self.model.features.conv4_4(x)
        x = self.model.features.conv4_5(x)
        x = self.model.features.conv4_6(x)
        x = self.model.features.conv4_7(x)
        x = self.model.features.conv4_8(x)
        x = self.model.features.conv4_9(x)
        x = self.model.features.conv4_10(x)
        x = self.model.features.conv4_11(x)
        x = self.model.features.conv4_12(x)

        return x

    def fea_conv5x(self, x):
        x = self.model.features.conv5_1(x)
        x = self.model.features.conv5_2(x)
        x = self.model.features.conv5_3(x)
        x = self.model.features.conv5_bn_ac(x)

        return x


    def features(self, x):
        x = self.fea_conv1x(x)
        x = self.fea_conv2x(x)
        x = self.fea_conv3x(x)
        x = self.fea_conv4x(x)
        fea = self.fea_conv5x(x)

        return fea

    def classifier(self, x):

        x = adaptive_avgmax_pool2d(x, pool_type='avg')
        x = self.last_conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        out = self.last_linear(x)

        return out

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, out_dim=None, add=False, ratio=8):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.add = add
        if out_dim is None:
            out_dim = in_dim
        self.out_dim = out_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X C X(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width*height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, self.out_dim, width, height)

        if self.add:
            out = self.gamma*out + x
        else:
            out = self.gamma*out
        return out  # , attention

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class Attention(nn.Module):
    def __init__(self, in_dims, out_dim=None, add=True):
        super(Attention, self).__init__()
        self.in_dims = in_dims // 2

        if out_dim is None:
            out_dim = in_dims // 2

        self.out_dim = out_dim
        self.add = add
        self.att3 = Self_Attn(self.in_dims)
        self.att2 = Self_Attn(self.in_dims)

        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        x3 = x[:, :self.in_dims, :, :]
        x2 = x[:, self.in_dims:, :, :]


        m_batchsize, C, width, height = x3.size()


        q2 = self.att2.query_conv(x2).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X C X(N)
        k2 = self.att2.query_conv(x2).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        v2 = self.att2.value_conv(x2).view(
            m_batchsize, -1, width*height)  # B X C X N

        q3 = self.att3.query_conv(x3).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X C X(N)
        k3 = self.att3.query_conv(x3).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        v3 = self.att3.value_conv(x3).view(
            m_batchsize, -1, width*height)  # B X C X N


        att32 = self.softmax(torch.bmm(q3, k2))
        out32 = torch.bmm(v3, att32.permute(0, 2, 1))
        out32 = out32.view(m_batchsize, self.out_dim, width, height)
        att33 = self.softmax(torch.bmm(q3, k3))
        out33 = torch.bmm(v3, att33.permute(0, 2, 1))
        out33 = out33.view(m_batchsize, self.out_dim, width, height)

        """tranmodelatten"""
        # if self.add:
        #     out = self.gamma3 * out3 + x3
        # else:
        #     out = self.gamma3 * out3
        # return out  # , attention

        if self.add:
            x3 = self.gamma2 * out32 + self.gamma3 * out33 + x3
        else:
            x3 = self.gamma2 * out32 + self.gamma3 * out33

        return x3

class DualCrossModalAttention(nn.Module):
    """ Dual CMA attention Layer"""

    def __init__(self, in_dim, activation=None, size=16, ratio=8, ret_att=False):
        super(DualCrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.ret_att = ret_att

        # query conv
        self.key_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv_share = nn.Conv2d(
            in_channels=in_dim//ratio, out_channels=in_dim//ratio, kernel_size=1)

        self.linear1 = nn.Linear(size*size, size*size)
        self.linear2 = nn.Linear(size*size, size*size)

        # separated value conv
        self.value_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.value_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W = x.size()

        def _get_att(a, b):
            proj_key1 = self.key_conv_share(self.key_conv1(a)).view(
                B, -1, H*W).permute(0, 2, 1)  # B, HW, C
            proj_key2 = self.key_conv_share(self.key_conv2(b)).view(
                B, -1, H*W)  # B X C x (*W*H)
            energy = torch.bmm(proj_key1, proj_key2)  # B, HW, HW

            attention1 = self.softmax(self.linear1(energy))
            attention2 = self.softmax(self.linear2(
                energy.permute(0, 2, 1)))  # BX (N) X (N)

            return attention1, attention2

        att_y_on_x, att_x_on_y = _get_att(x, y)
        proj_value_y_on_x = self.value_conv2(y).view(
            B, -1, H*W)  # B, C, HW
        out_y_on_x = torch.bmm(proj_value_y_on_x, att_y_on_x.permute(0, 2, 1))
        out_y_on_x = out_y_on_x.view(B, C, H, W)
        out_x = self.gamma1*out_y_on_x + x

        proj_value_x_on_y = self.value_conv1(x).view(
            B, -1, H*W)  # B , C , HW
        out_x_on_y = torch.bmm(proj_value_x_on_y, att_x_on_y.permute(0, 2, 1))
        out_x_on_y = out_x_on_y.view(B, C, H, W)
        out_y = self.gamma2*out_x_on_y + y

        if self.ret_att:
            return out_x, out_y, att_y_on_x, att_x_on_y

        return out_x, out_y  # , attention

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=832 * 2, out_chan=832, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, ratio=16)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class SRMConv2d_simple(nn.Module):

    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        return filters


class SRMConv2d_Separate(nn.Module):

    def __init__(self, inc, outc, num = 1,learnable=False):
        super(SRMConv2d_Separate, self).__init__()
        self.num = num
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)
        self.out_conv = nn.Sequential(
            nn.Conv2d(3 * inc, outc, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )

        for ly in self.out_conv.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''

        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        out = F.conv2d(x, self.kernel, stride=1, padding=2, groups=self.inc)
        out = self.truc(out)
        out = self.out_conv(out)
        return out






    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        # filters = np.repeat(filters, inc, axis=1)
        filters = np.repeat(filters, inc, axis=0)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        # print(filters.size())
        return filters


class Two_Stream_Net(nn.Module):
    def __init__(self, dropout=0.5, weight_norm=False):
        super(Two_Stream_Net, self).__init__()

        self.dpn_rgb = TransferModel(dropout=dropout, weight_norm=weight_norm)

        self.down144 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=320, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )
        self.atten2_3 = Attention(640)

        self.down320 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=704, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(704),
            nn.ReLU(inplace=True),
        )
        self.atten3_4 = Attention(1408)

        self.down704 = nn.Sequential(
            nn.Conv2d(in_channels=704, out_channels=832, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(832),
            nn.ReLU(inplace=True),
        )
        self.atten4_5 = Attention(1664)

        self.dpn_srm = TransferModel(dropout=dropout, weight_norm=weight_norm)

        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(32, 32)
        self.srm_conv2 = SRMConv2d_Separate(64, 64)
        self.relu = nn.ReLU(inplace=True)

        self.dual_cma0 = DualCrossModalAttention(in_dim=728, ret_att=False)

        self.fusion = FeatureFusionModule()

    def features(self, x):

        srm = self.srm_conv0(x)

        x = self.dpn_rgb.fea_conv1x(x)
        y = self.dpn_srm.fea_conv1x(srm)

        x = self.dpn_rgb.fea_conv2x(x)
        y = self.dpn_srm.fea_conv2x(y)

        x2 = torch.cat([x[0], x[1]], dim=1)
        x2 = self.down144(x2)

        x = self.dpn_rgb.fea_conv3x(x)
        y = self.dpn_srm.fea_conv3x(y)

        x3 = torch.cat([x[0], x[1]], dim=1)
        x32 = torch.cat([x3, x2], dim=1)
        x32 = self.atten2_3(x32)
        x3 = self.down320(x32)

        x = self.dpn_rgb.fea_conv4x(x)
        y = self.dpn_srm.fea_conv4x(y)

        x4 = torch.cat([x[0], x[1]], dim=1)
        x43 = torch.cat([x4, x3], dim=1)
        x43 = self.atten3_4(x43)
        x43 = self.down704(x43)

        x = self.dpn_rgb.fea_conv5x(x)
        y = self.dpn_srm.fea_conv5x(y)

        x = torch.cat([x, x43], dim=1)

        x = self.atten4_5(x)

        fea = self.fusion(x, y)

        return fea

    def classifier(self, x):
        return self.dpn_rgb.classifier(x)

    def forward(self, x):
        fea = self.features(x)
        out = self.classifier(fea)

        return out


class Two_Stream_Netv1(nn.Module):
    def __init__(self, dropout=0.5, weight_norm=False):
        super(Two_Stream_Netv1, self).__init__()

        self.dpn_rgb = TransferModel(dropout=dropout, weight_norm=weight_norm)

        self.down144 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=320, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )
        self.atten2_3 = Attention(640)

        self.down320 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=704, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(704),
            nn.ReLU(inplace=True),
        )
        self.atten3_4 = Attention(1408)

        self.down704 = nn.Sequential(
            nn.Conv2d(in_channels=704, out_channels=832, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(832),
            nn.ReLU(inplace=True),
        )
        self.atten4_5 = Attention(1664)

        self.dpn_srm = TransferModel(dropout=dropout, weight_norm=weight_norm)

        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(10, 10)
        self.srm_conv2 = SRMConv2d_Separate(inc=144, outc=144)
        self.relu = nn.ReLU(inplace=True)

        self.dual_cma0 = DualCrossModalAttention(in_dim=728, ret_att=False)

        self.fusion = FeatureFusionModule()

    def features(self, x):

        srm = self.srm_conv0(x)

        x = self.dpn_rgb.fea_conv1x(x)

        y = self.dpn_srm.fea_conv1x(srm) + self.srm_conv1(x)
        y = self.relu(y)

        x = self.dpn_rgb.fea_conv2x(x)
        y = self.dpn_srm.fea_conv2x(y)
        y = torch.cat(y, dim=1) if isinstance(y, tuple) else y
        y = y + self.srm_conv2(x)
        y = self.relu(y)

        x2 = torch.cat([x[0], x[1]], dim=1)
        x2 = self.down144(x2)

        x = self.dpn_rgb.fea_conv3x(x)
        y = self.dpn_srm.fea_conv3x(y)

        x3 = torch.cat([x[0], x[1]], dim=1)
        x32 = torch.cat([x3, x2], dim=1)
        x32 = self.atten2_3(x32)
        x3 = self.down320(x32)

        x = self.dpn_rgb.fea_conv4x(x)
        y = self.dpn_srm.fea_conv4x(y)

        x4 = torch.cat([x[0], x[1]], dim=1)
        x43 = torch.cat([x4, x3], dim=1)
        x43 = self.atten3_4(x43)
        x43 = self.down704(x43)

        x = self.dpn_rgb.fea_conv5x(x)
        y = self.dpn_srm.fea_conv5x(y)

        x = torch.cat([x, x43], dim=1)

        x = self.atten4_5(x)


        fea = self.fusion(x, y)

        return fea

    def classifier(self, x):
        return self.dpn_rgb.classifier(x)

    def forward(self, x):

        fea = self.features(x)
        out = self.classifier(fea)

        return out



if __name__=="__main__":

    # model = FeatureFusionModule()
    #
    # x = torch.rand(10, 832, 7, 7)
    # y = torch.rand(10, 832, 7, 7)
    #
    # out = model(x, y)
    model = Two_Stream_Netv1()
    x = torch.rand(10, 3, 128,128)
    out = model(x)
    print(out.size())
    print(out)