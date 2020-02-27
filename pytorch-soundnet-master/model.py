from soundnet import SoundNet
from Attentation import Encoder, Decoder, Gaussian_Attentation
import torch
import numpy as np
import torch.nn as nn
from definitions import Short_Length, attention, channel
from triangularLayer import TriangularLayer
from feature_extractors import HourglassExtractor, ResNetExtractor
class Model(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout, NUM_FILTERS):
        super(Model, self).__init__()

        # self.encoder = EncoderLayer(
        #     d_model=d_model, d_inner=d_inner,
        #     n_head=n_head, d_k=d_k, d_v=d_v,
        #      dropout=dropout)
        self.num_fileters = NUM_FILTERS
        self.soundnet = SoundNet(in_channel = 1)
        self.triangular = TriangularLayer(Short_Length)
        self.relu = nn.ReLU(True)
        self.fc = nn.Linear(NUM_FILTERS, 4)
        # self.atten = Self_Attentation(in_channel = 256)
        self.atten = Gaussian_Attentation(in_channel = 1)

        self.BN = nn.BatchNorm2d(1, eps=1e-5, momentum=0.1, affine=False)

    def forward(self, x):
        b, c, w, h = x.size()  # 20 1 72 1
        # x = x.view(-1, 1, Short_Length, 1)
        # x_short = x.view(-1, Short_Length, 1)
        # x_short = self.triangular(x_short)
        # x_short = x_short.view(-1, c, Short_Length, 1)

        # x_long = torch.rand(b,  w).float().cuda()
        para = None
        if attention:
           atten, para = self.atten(x)
           atten = self.BN(atten)

           #x = torch.stack((x, atten), dim = 1)
           x = torch.cat((x, atten), dim = 2)
           x = x.view(b, -1, 2*w, h)
           # a = x[0]

        x = self.soundnet(x) # 20 256 2 1

        x = x.view(b, -1)
        x = self.fc(x)
        # x_short = torch.stack([x_short[i, :, x_atten_selection[i]] for i in range(0, x_atten_selection.size()[0])])

        return x, para


class DeepConvLSTM(nn.Module):
    def __init__(self):
        super(DeepConvLSTM, self).__init__()

        self.feature_extractor = ResNetExtractor()

        self.encoder = Encoder(vocab_size=72,
                               embed_size=100,
                               enc_hidden_size=100,
                               dec_hidden_size=100,
                               dropout=0.2)

        self.out = nn.Linear(72 * 200, 4)
    def forward(self, x, x_lengths):
        b, c, w, h = x.size() # 5 1 3 72
        x = self.feature_extractor(x)  # 5 100 1 72
        x = torch.squeeze(x).transpose(1, 2)

        encoder_out, hid = self.encoder(x, x_lengths) #5 60 200
        encoder_out = encoder_out.view(b, -1)
        padding_size = 72 - encoder_out.shape[1] / 200
        padding = (torch.ones(b, int(padding_size) * 200)*10e-15).cuda()
        encoder_out = torch.cat((encoder_out, padding), dim=1)
        encoder_out = self.out(encoder_out)
        return encoder_out, hid


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size=72,
                   embed_size=100,
                   enc_hidden_size=100,
                   dec_hidden_size=100,
                   dropout=0.2)
        self.decoder = Decoder(vocab_size=72,
                   embed_size=100,
                   enc_hidden_size=100,
                   dec_hidden_size=100,
                   dropout=0.2)
        self.feature_extractor = ResNetExtractor()
        self.atten = Gaussian_Attentation(in_channel=1)
        self.fc = nn.Linear(60*72, 4)
    def forward(self, x, x_lengths, y):
        b, c, w, h = x.size()
        x = self.feature_extractor(x)
        x = torch.squeeze(x).transpose(1,2)
        # atten = x.view(b, 1, w, 1)
        # atten, para = self.atten(atten)
        # atten = atten.view(b, w, h)
        encoder_out, hid = self.encoder(x, x_lengths)
        # encoder_out_atten, hid_atten = self.encoder(atten, x_lengths)
        output, hid, attn = self.decoder(ctx = encoder_out,
                    ctx_lengths = x_lengths,
                    y = encoder_out,
                    y_lengths = x_lengths,
                    hid=hid)

        return output, attn



if __name__ == '__main__':
   # nx = torch.rand(20, 1, 3, 72).float().cuda()
   #
   # model = Model(
   #          d_model=64, d_inner=2048,
   #          n_head=8, d_k=32, d_v=32,
   #           dropout=0.1, NUM_FILTERS = 256*3).cuda()

   nx = np.zeros((5,  72))

   len_nx = []
   for i in range(len(nx)):
       for j in range((i + 1) * 12):
           nx[i][j] = 1
       len_nx.append((i + 1) * 12)

   nx = torch.rand(5, 1, 3, 72).float().cuda()
   len_nx = np.array(len_nx)
   len_nx = torch.from_numpy(len_nx).long().cuda()
   print(nx)

   model = DeepConvLSTM().cuda()
   out,atten = model(nx, len_nx)
   print(out.size())

