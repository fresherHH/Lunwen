from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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
        self.classifier = nn.Conv2d(in_chs, num_classes, kernel_size=1, bias=True)

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


class Attention(nn.Module):
    def __init__(self, in_dims, out_dim=None, add=False):
        super(Attention, self).__init__()
        self.in_dims = in_dims // 3

        if out_dim is None:
            out_dim = in_dims // 3
        self.out_dim = out_dim
        self.add = add
        self.att1 = Self_Attn(256)
        self.att2 = Self_Attn(256)
        self.att3 = Self_Attn(256)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        x1 = x[:, :self.in_dims, :, :]
        x2 = x[:, self.in_dims:2 * self.in_dims, :, :]
        x3 = x[:, 2*self.in_dims:, :, :]

        m_batchsize, C, width, height = x1.size()

        q1 = self.att1.query_conv(x1).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X C X(N)
        k1 = self.att1.query_conv(x1).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        v1 = self.att1.value_conv(x1).view(
            m_batchsize, -1, width*height)  # B X C X N

        q2 = self.att1.query_conv(x2).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X C X(N)
        k2 = self.att1.query_conv(x2).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        v2 = self.att1.value_conv(x2).view(
            m_batchsize, -1, width*height)  # B X C X N

        q3 = self.att1.query_conv(x3).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X C X(N)
        k3 = self.att1.query_conv(x3).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        v3 = self.att1.value_conv(x3).view(
            m_batchsize, -1, width*height)  # B X C X N

        att11 = self.softmax(torch.bmm(q1, k1))
        out11 = torch.bmm(v1, att11.permute(0, 2, 1))
        out11 = out11.view(m_batchsize, self.out_dim, width, height)
        att12 = self.softmax(torch.bmm(q1, k2))
        out12 = torch.bmm(v1, att12.permute(0, 2, 1))
        out12 = out12.view(m_batchsize, self.out_dim, width, height)
        att13 = self.softmax(torch.bmm(q1, k3))
        out13 = torch.bmm(v1, att13.permute(0, 2, 1))
        out13 = out13.view(m_batchsize, self.out_dim, width, height)
        out1 = out11 + out12 + out13

        att21 = self.softmax(torch.bmm(q2, k1))
        out21 = torch.bmm(v2, att21.permute(0, 2, 1))
        out21 = out21.view(m_batchsize, self.out_dim, width, height)
        att22 = self.softmax(torch.bmm(q2, k2))
        out22 = torch.bmm(v2, att22.permute(0, 2, 1))
        out22 = out22.view(m_batchsize, self.out_dim, width, height)
        att23 = self.softmax(torch.bmm(q2, k3))
        out23 = torch.bmm(v2, att23.permute(0, 2, 1))
        out23 = out23.view(m_batchsize, self.out_dim, width, height)
        out2 = out21 + out22 + out23

        att31 = self.softmax(torch.bmm(q3, k1))
        out31 = torch.bmm(v3, att31.permute(0, 2, 1))
        out31 = out31.view(m_batchsize, self.out_dim, width, height)
        att32 = self.softmax(torch.bmm(q3, k2))
        out32 = torch.bmm(v3, att32.permute(0, 2, 1))
        out32 = out32.view(m_batchsize, self.out_dim, width, height)
        att33 = self.softmax(torch.bmm(q3, k3))
        out33 = torch.bmm(v3, att33.permute(0, 2, 1))
        out33 = out33.view(m_batchsize, self.out_dim, width, height)
        out3 = out31 + out32 + out33

        if self.add:
            out = self.gamma1 * out1 + self.gamma2 *out2 + self.gamma3 * out3 + x1 + x2 + x3
        else:
            out = self.gamma1 * out1 + self.gamma2 *out2 + self.gamma3 * out3
        return out  # , attention


class TranModel(nn.Module):
    def __init__(self, num_out_classes=2):
        super(TranModel, self).__init__()

        model = dpn68(num_classes=num_out_classes)
        self.test_time_pool = model.test_time_pool
        weights = r"F:\HH\dpn68.pth"
        if os.path.exists(weights):
            weights_dict = torch.load(weights, map_location="cuda")
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(weights))


        self.conv1_1 = model.features.conv1_1
        self.conv2_1 = model.features.conv2_1
        self.conv2_2 = model.features.conv2_2
        self.conv2_3 = model.features.conv2_3
        self.conv3_1 = model.features.conv3_1
        self.conv3_2 = model.features.conv3_2
        self.conv3_3 = model.features.conv3_3
        self.conv3_4 = model.features.conv3_4
        self.conv4_1 = model.features.conv4_1
        self.conv4_2 = model.features.conv4_2
        self.conv4_3 = model.features.conv4_3
        self.conv4_4 = model.features.conv4_4
        self.conv4_5 = model.features.conv4_5
        self.conv4_6 = model.features.conv4_6
        self.conv4_7 = model.features.conv4_7
        self.conv4_8 = model.features.conv4_8
        self.conv4_9 = model.features.conv4_9
        self.conv4_10 = model.features.conv4_10
        self.conv4_11 = model.features.conv4_11
        self.conv4_12 = model.features.conv4_12
        self.conv5_1 = model.features.conv5_1
        self.conv5_2 = model.features.conv5_2
        self.conv5_3 = model.features.conv5_3
        self.conv5_bn_ac = model.features.conv5_bn_ac
        self.classifier = model.classifier
        self.atten = Attention(768)
        self.down64 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.down128 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def fea_conv1x(self, x):
        x = self.conv1_1(x)

        return x

    def fea_conv2x(self, x):
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        return x

    def fea_conv3x(self, x):
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        return x

    def fea_conv4x(self, x):
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)
        x = self.conv4_7(x)
        x = self.conv4_8(x)
        x = self.conv4_9(x)
        x = self.conv4_10(x)
        x = self.conv4_11(x)
        x = self.conv4_12(x)

        return x

    def fea_conv5x(self, x):
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_bn_ac(x)

        return x

    def forward(self, x):

        x = self.fea_conv1x(x)
        # print(x.size())

        x = self.fea_conv2x(x)
        # print(x[0].size())
        # print(x[1].size())

        x0 = self.down64(x[0])

        x = self.fea_conv3x(x)
        # print(x[0].size())
        # print(x[1].size())
        x1 = self.down128(x[0])

        x = self.fea_conv4x(x)
        # print(x[0].size())
        # print(x[1].size())
        X = torch.cat([x[0], x0, x1], 1)
        X = self.atten(X)
        x = torch.cat([X, x[1]], dim = 1)
        # print(x.size())
        x = self.fea_conv5x(x)
        # print(x.size())


        if not self.training and self.test_time_pool:
            x = F.avg_pool2d(x, kernel_size=7, stride=1)
            out = self.classifier(x)
            # The extra test time pool should be pooling an img_size//32 - 6 size patch
            out = adaptive_avgmax_pool2d(out, pool_type='avgmax')
        else:
            x = adaptive_avgmax_pool2d(x, pool_type='avg')
            out = self.classifier(x)
        return out.view(out.size(0), -1)




if __name__=="__main__":
    #
    # writer = SummaryWriter('model')
    #
    # model = dpn68()
    # input = torch.rand(1, 3, 224, 224)
    #
    # with SummaryWriter(comment='DPN') as w:
    #     w.add_graph(model, (input, ))
    #
    # print(model)
    # # pass

    # model = dpn68()
    model = TranModel()
    print(model)
    inp = torch.rand(1,3, 224,224)
    # summary(model, (3, 224, 224), device='cpu')
    out = model(inp)
    print(out.size())







