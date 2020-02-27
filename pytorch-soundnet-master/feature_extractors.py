import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class HourglassExtractor(nn.Module):

    def __init__(self):
        super(HourglassExtractor, self).__init__()

        self.extendConv = nn.Conv2d(1, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inConv = nn.Conv2d(64, 128, kernel_size=(3, 1),stride=(1, 1))
        self.BN128 = nn.BatchNorm2d(128)
        self.BN64 = nn.BatchNorm2d(64)
        self.BN32 = nn.BatchNorm2d(32)
        self.BN16 = nn.BatchNorm2d(16)
        self.BN8 = nn.BatchNorm2d(8)
        self.BN100 = nn.BatchNorm2d(100)
        self.relu = nn.ReLU(inplace=True)

        #DownSample
        self.conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
        self.conv4 = nn.Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
        self.conv5 = nn.Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))

        #UpSample
        self.OutConv = nn.Conv2d(128, 100, kernel_size=(1, 1), stride=(1, 1))
        self.conv_2 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv_3 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv_4 = nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1))
        self.conv_5 = nn.Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1))




    def forward(self, x):
        b, c, w, h = x.size()
        x = self.extendConv(x) #5 64 3 72
        x = self.inConv(x)  #5 128 1 72
        x = self.BN128(x)
        x = self.relu(x)
        res1 = x

        x = self.conv1(x)
        x = self.BN64(x)
        x = self.relu(x)
        res2 = x

        x = self.conv2(x)
        x = self.BN32(x)
        x = self.relu(x)
        res3 = x

        x = self.conv3(x)
        x = self.BN16(x)
        x = self.relu(x)
        res4 = x

        x = self.conv4(x)
        x = self.BN8(x)
        x = self.relu(x)
        res5 = x

        x = self.conv5(x)
        x = self.BN8(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.BN8(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.BN8(x)
        x = self.relu(x)

        x = res5 + x

        x = self.conv_5(x)
        x = self.BN16(x)
        x = self.relu(x)

        x = res4 + x

        x = self.conv_4(x)
        x = self.BN32(x)
        x = self.relu(x)

        x = res3 + x

        x = self.conv_3(x)
        x = self.BN64(x)
        x = self.relu(x)

        x = res2 + x

        x = self.conv_2(x)
        x = self.BN128(x)
        x = self.relu(x)

        x = res1 + x

        x = self.OutConv(x)
        x = self.BN100(x)
        x = self.relu(x)

        return x

class ResNetExtractor(nn.Module):

    def __init__(self):
        super(ResNetExtractor, self).__init__()

        self.extendConv = nn.Conv2d(1, 64, kernel_size=(1, 1), stride=(1, 1))
        self.inConv = nn.Conv2d(64, 100, kernel_size=(3, 1),stride=(1, 1))
        self.BN100 = nn.BatchNorm2d(100)
        self.BN64 = nn.BatchNorm2d(64)
        self.BN32 = nn.BatchNorm2d(32)
        self.BN16 = nn.BatchNorm2d(16)


        self.relu = nn.ReLU(inplace=True)

        #DownSample
        self.conv1 = nn.Conv2d(100, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1))
        self.conv4 = nn.Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))


        #UpSample
        self.OutConv = nn.Conv2d(64, 100, kernel_size=(1, 1), stride=(1, 1))
        self.conv_1 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1))





    def forward(self, x):
        b, c, w, h = x.size()

        x = self.extendConv(x)
        x = self.inConv(x)
        x = self.BN100(x)
        x = self.relu(x)
        res_100 = x

        x = self.conv1(x)
        x = self.BN64(x)
        x = self.relu(x)
        res_64 = x

        x = self.conv2(x)
        x = self.BN32(x)
        x = self.relu(x)
        res_32 = x

        x = self.conv3(x)
        x = self.BN16(x)
        x = self.relu(x)
        res_16 = x


        x = self.conv_2(x) #32
        x = self.BN32(x)
        x = self.relu(x)

        x = res_32 + x

        x = self.conv_1(x)
        x = self.BN64(x)
        x = self.relu(x)

        x = res_64 + x

        x = self.OutConv(x)
        x = self.BN100(x)
        x = self.relu(x)
        return x
if __name__ == '__main__':

    nx = torch.randn(5, 1, 3, 72).float().cuda()

    model = ResNetExtractor().cuda()

    output = model(nx)

    print(output.shape)
