import torch
import torch.nn as nn
import torch.nn. functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class SRMConv2d(nn.Module):

    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

    def forward(self, x):

        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)
        return out

    def _build_kernel(self, inc ):

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

        filters = [[filter1],[filter2], [filter3]]

        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)
        return filters


class SRMConv2dFilter(nn.Module):

    def __init__(self, inc = 3, outc = 3, learnable=False):
        super(SRMConv2dFilter, self).__init__()
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
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


if __name__== "__main__":

    data_transform = transforms.Compose(
        [
         # transforms.Resize(256),
            #          # transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])

    # img_path = r"C:\Users\HH\Desktop\Au_txt_0056.jpg"
    # img_path = r"C:\Users\HH\Desktop\Sp_S_NNN_C_txt0056_txt0056_0056.jpg"
    # img_path = r"C:\Users\HH\Desktop\Au_ani_0023.jpg"
    # img_path = r"C:\Users\HH\Desktop\Sp_S_NNN_C_txt0059_txt0059_0059.jpg"
    # img_path = r"C:\Users\HH\Desktop\Sp_D_CNN_A_nat0071_ani0024_0270.jpg"
    img_path = r"C:\Users\HH\Desktop\答辩PPT\最新\561.jpg"
    img = Image.open(img_path)
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    m = SRMConv2d()
    img = m(img)
    print(img.size())
    img = torch.squeeze(img, dim=0)
    img = img.detach().numpy().transpose(1,2,0)
    # img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    # plt.subplot(1, 2, 2)
    plt.imshow(img.astype('uint8'))
    plt.show()


