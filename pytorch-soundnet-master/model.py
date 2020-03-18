from soundnet import SoundNet
from Attentation import Encoder, Decoder, Gaussian_Attentation, PlainDecoder, PlainEncoder
import torch
import numpy as np
import torch.nn as nn
from definitions import *
from triangularLayer import TriangularLayer
from feature_extractors import HourglassExtractor, ResNetExtractor
from torch.autograd import Variable
import torch.nn.functional as F

class DeepConvLSTM(nn.Module):
    def __init__(self):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=NUM_FILTERS, kernel_size=(12, FILTER_SIZE), stride=(1, 3)),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(FILTER_SIZE, FILTER_SIZE)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.Dropout2d(0.5),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(FILTER_SIZE, FILTER_SIZE)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.Dropout2d(0.5),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(1, FILTER_SIZE)),
            # nn.BatchNorm2d(NUM_FILTERS),
            # nn.Dropout2d(0.5),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(1, FILTER_SIZE)),
            # nn.BatchNorm2d(NUM_FILTERS),
            # nn.Dropout2d(0.5),
            nn.ReLU())
        self.lstm = nn.LSTM(NUM_FILTERS, NUM_UNITS_LSTM, NUM_LSTM_LAYERS, batch_first=True)

        self.fc = nn.Linear(NUM_UNITS_LSTM, NUM_CLASSES)

    def forward(self, x):
        # print (x.shape)
        out = self.conv1(x)
        # print (out.shape)
        out = self.conv2(out)
        # print (out.shape)
        out = self.conv3(out)
        # print (out.shape)
        out = self.conv4(out)
        # print (out.shape)
        # out = out.view(-1, NB_SENSOR_CHANNELS, NUM_FILTERS)

        out = out.view(-1, 9 * 31, NUM_FILTERS)  # CHANNELS_NUM_50

        h0 = Variable(torch.zeros(NUM_LSTM_LAYERS, out.size(0), NUM_UNITS_LSTM))
        c0 = Variable(torch.zeros(NUM_LSTM_LAYERS, out.size(0), NUM_UNITS_LSTM))
        if torch.cuda.is_available():
            h0, c0 = h0.cuda(), c0.cuda()

        # forward propagate rnn


        out, _ = self.lstm(out, (h0, c0))
        #  out[:, -1, :] -> [100,11,128] ->[100,128]
        out = self.fc(out[:, -1, :])
        return out


class ConvLSTM(nn.Module):
    def __init__(self):
        super(DeepConvLSTM, self).__init__()

        self.feature_extractor = ResNetExtractor()

        self.encoder = Encoder(vocab_size=72,
                               embed_size=100,
                               enc_hidden_size=100,
                               dec_hidden_size=100,
                               dropout=0.2)

        self.out = nn.Linear(72 * 200, 4)
        self.out_att = nn.Linear(200, 4)
        self.hid_att = nn.Linear(100, 72)
    def forward(self, x, x_lengths, atten):
        b, c, w, h = x.size() # 5 1 3 72
        x = self.feature_extractor(x)  # 5 100 1 72
        x = torch.squeeze(x).transpose(1, 2)

        encoder_out, hid = self.encoder(x, x_lengths) #5 60 200    1 5 100

        padding_size = 72 - encoder_out.shape[1]

        if attention:
            att = atten.view(b, c, -1)
            att = self.hid_att(hid.view(b,-1)).view(b, c, -1)
            padding = (torch.ones(b, int(padding_size), 200) * 10e-15).cuda()
            encoder_out = torch.cat((encoder_out, padding), dim=1)
            out = torch.bmm(att,encoder_out).view(b,-1)
            out = self.out_att(out)
            return out, hid
        else:
            encoder_out = encoder_out.view(b, -1)
            padding = (torch.ones(b, int(padding_size)*200) * 10e-15).cuda()
            encoder_out = torch.cat((encoder_out, padding), dim=1)
            encoder_out = self.out(encoder_out)
            return encoder_out, hid


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = PlainEncoder(vocab_size= 3,
                      hidden_size= 100,
                      dropout= 0.2)
        self.decoder = PlainDecoder(vocab_size= 3,
                      hidden_size= 100,
                      dropout= 0.2)
        self.feature_extractor = ResNetExtractor()
        self.atten = Gaussian_Attentation(in_channel=1)
        self.fc = nn.Linear(60*72, 4)
    def forward(self, x, x_lengths, y):
        b, c, w, h = x.size()
        # x = self.feature_extractor(x)
        x = torch.squeeze(x).transpose(1,2)
        # atten = x.view(b, 1, w, 1)
        # atten, para = self.atten(atten)
        # atten = atten.view(b, w, h)
        encoder_out, hid = self.encoder(x, x_lengths)
        # encoder_out_atten, hid_atten = self.encoder(atten, x_lengths)
        output, hid  = self.decoder(
                    y = encoder_out,
                    y_lengths = x_lengths,
                    hid=hid)

        return output


class ae_spatial_LSTM_CNN(nn.Module):
    def __init__(self):#, args):
        super(ae_spatial_LSTM_CNN, self).__init__()
        # parameter part
        self.n_lstm_hidden = 64# args.n_lstm_hidden
        self.n_lstm_layer = 1#args.n_lstm_layer

        self.n_feature = 113 #args.n_feature
        self.len_sw = 24# args.len_sw
        self.n_class = 18# args.n_class

        # autoencoder part
        self.encoder = nn.Sequential(
            nn.Linear(self.n_feature * self.len_sw, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(32, 20))
        self.decoder = nn.Sequential(
            nn.Linear(20, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(128, self.n_feature * self.len_sw),
            nn.Tanh())
        # rnn part
        self.lstm = nn.LSTM(self.n_feature, self.n_lstm_hidden, self.n_lstm_layer, batch_first=True)
        self.lstm_spatial = nn.LSTM(self.len_sw, self.n_lstm_hidden, self.n_lstm_layer, batch_first=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_feature, out_channels=1024, kernel_size=(1, 5)),
            # nn.BatchNorm1d()
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )

        ## fc part after concat of three networks
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=(2*self.n_lstm_hidden + 20 + 64), out_features=1000),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=self.n_class)
        )

    def forward(self, x):

        out_encoder = self.encoder(x.view(x.size(0), -1)) #x = torch.zeros(100, 51, 24)
        out_decoder = self.decoder(out_encoder)

        out_rnn, _ = self.lstm(x.view(x.shape[0], -1, self.n_feature)) # (64, 100, 9)
        out_rnn = out_rnn[:, -1, :]

        out_rnn_spatial, _ = self.lstm_spatial(x.view(x.shape[0], self.n_feature, -1))  # (64, 9, 100)
        out_rnn_spatial = out_rnn_spatial[:, -1, :]

        out_conv1 = self.conv1(x.view(-1, x.shape[1], 1, x.shape[2]))
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv4 = out_conv4.reshape(-1, out_conv4.shape[1] * out_conv4.shape[3])

        out_combined = torch.cat((out_encoder.view(out_encoder.size(0), -1), out_rnn.view(out_rnn.size(0), -1),
                                  out_rnn_spatial.view(out_rnn_spatial.size(0), -1),
                                  out_conv4.view(out_conv3.size(0), -1)), dim=1)  # (64,184)
        out_combined = self.fc1(out_combined)
        out_combined = self.fc2(out_combined)
        out_combined = self.fc3(out_combined)
        out_combined = F.softmax(out_combined, dim=1)
        return out_combined, out_decoder


class DeepConvAE(nn.Module):
    def __init__(self):
        super(DeepConvAE, self).__init__()
        self.encoder = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=NUM_FILTERS, kernel_size=(FILTER_SIZE, FILTER_SIZE)),
            nn.ReLU(),

            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(FILTER_SIZE, FILTER_SIZE)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.Dropout2d(0.5),
            nn.ReLU(),

            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(FILTER_SIZE, FILTER_SIZE)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.Dropout2d(0.5),
            nn.ReLU(),

            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(1, FILTER_SIZE)),
            nn.ReLU(),

        )


        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(1, FILTER_SIZE)),
            nn.ReLU(),

            nn.ConvTranspose2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(FILTER_SIZE, FILTER_SIZE)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.Dropout2d(0.5),
            nn.ReLU(),

            nn.ConvTranspose2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(FILTER_SIZE, FILTER_SIZE)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.Dropout2d(0.5),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=NUM_FILTERS, out_channels=1, kernel_size=(FILTER_SIZE, FILTER_SIZE)),
            nn.ReLU(),

        )
        self.Dense1 = nn.Linear(64*18*105, 256)
        self.Dense2 = nn.Linear(256, 18)
    def forward(self, x):

        out_encoder = self.encoder(x)  # 100 64 18 105
        out_decoder = self.decoder(out_encoder)

        out = self.Dense1(out_encoder.view(-1, 64*18*105))
        out = self.Dense2(out)
        return out, out_decoder.view(out_decoder.size(0), -1)

if __name__ == '__main__':
   # nx = torch.rand(20, 1, 3, 72).float().cuda()
   #
   #
   #
   # nx = np.zeros((5,  72))
   #
   # len_nx = []
   # for i in range(len(nx)):
   #     for j in range((i + 1) * 12):
   #         nx[i][j] = 1
   #     len_nx.append((i + 1) * 12)
   #
   # nx = torch.rand(5, 1, 3, 72).float().cuda()
   # ny = torch.rand(5,  72).float().cuda()
   # len_nx = np.array(len_nx)
   # len_nx = torch.from_numpy(len_nx).long().cuda()
   # print(nx)
   # #
   # model = Seq2Seq(
   #          ).cuda()
   # out, atten = model(nx, len_nx, ny)
   # print(out.size())


   # x = torch.zeros(100, 1, 24, 113)
   # x = Variable(x)
   # x = x.cuda() if torch.cuda.is_available() else x
   # model = DeepConvLSTM().cuda()
   # out = model(x)
   # print(out.size())

    x = torch.zeros(100, 1, 24, 113)
    x = Variable(x)
    x = x.cuda() if torch.cuda.is_available() else x
    model = DeepConvAE().cuda()
    out, out_decoder = model(x)
    print(out.size())



   # model = ae_spatial_LSTM_CNN()
   # if torch.cuda.is_available():
   #     model.cuda()
   # # N C T V
   # x = torch.zeros(100, 113, 24)
   # x = Variable(x)
   # x = x.cuda() if torch.cuda.is_available() else x
   # out_combined, out_decoder = model(x)
   # print(out_combined.size())
   # print(out_decoder.size())