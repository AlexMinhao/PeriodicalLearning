from torch import nn
import torch
import torch.nn.functional as F
from definitions import *
class SoundNet(nn.Module):
    def __init__(self, in_channel):
        super(SoundNet, self).__init__()
        self.channel = in_channel


        self.conv1 = nn.Conv2d(self.channel, 16, kernel_size=(3, 1), stride=(1, 1),
                               padding=(2, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((3, 1), stride=(1, 1), padding = (0, 0))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(1, 1),
                               padding=(2, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((3, 1), stride=(1, 1), padding = (1, 0))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(8, 1), stride=(2, 1),
                               padding=(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1),
                               padding=(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1), padding = (2, 0))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(2, 1), stride=(1, 1),
                               padding=(0, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)

        self.conv8_objs = nn.Conv2d(1024, 1000, kernel_size=(8, 1),
                                    stride=(2, 1))
        self.conv8_scns = nn.Conv2d(1024, 256, kernel_size=(1, 1))


        self.linear = nn.Linear(256, 4)
        self.norm = nn.Softmax()

    def forward(self, waveform): #1 1 128
        x = self.conv1(waveform) #16 65 1
        x = self.batchnorm1(x) # 16 65 1
        x = self.relu1(x)
        x = self.maxpool1(x) # 16 8 1

        x = self.conv2(x)  # 32 5 1
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = self.relu7(x)

        x = self.conv8_scns(x)
        # x = x.view((-1, 256))
        # x = self.linear(x)

        # x = self.norm(x)

        return x


if __name__ == '__main__':
   nx = torch.rand(20, 1, 144, 1).float().cuda()
   model = SoundNet(in_channel=1).cuda()
   model_map = {
       0: model.conv1,
       4: model.conv2,
       8: model.conv3,
       11: model.conv4,
       14: model.conv5,
       18: model.conv6,
       21: model.conv7,
   }
   print(model_map)
   output = model(nx)
   print(output)