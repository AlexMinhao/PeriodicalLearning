
''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from PIL import Image

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, enc_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(self, x, lengths):
        sorted_len, sorted_idx = lengths.sort(0, descending=True) # 从大到小 排序列
        x_sorted = x[sorted_idx.long()]
        embedded = x_sorted
        # embedded = self.dropout(embedded)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(),
                                                            batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        hid = torch.cat([hid[-2], hid[-1]], dim=1)  # 正向的和反向的
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)

        return out, hid

class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention, self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size * 2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size)

    def forward(self, output, context, mask):
        # output: batch_size, output_len, dec_hidden_size  5 60 100
        # context: batch_size, context_len, 2*enc_hidden_size  # 5 60 200

        batch_size = output.size(0)  # 5
        output_len = output.size(1)  # 60
        input_len = context.size(1)  # 60

        context_in = self.linear_in(context.view(batch_size * input_len, -1)).view(
            batch_size, input_len, -1)  # batch_size, context_len, dec_hidden_size  5 60 100

        # context_in.transpose(1,2): batch_size, dec_hidden_size, context_len  5 100 60
        # output: batch_size, output_len, dec_hidden_size  5 60 100
        attn = torch.bmm(output, context_in.transpose(1, 2))
        # batch_size, output_len, context_len  5 60 60

        attn.data.masked_fill(mask, -1e6)

        attn = F.softmax(attn, dim=2)
        # batch_size, output_len, context_len

        context = torch.bmm(attn, context)    #5 60 60 * 5 60 200  = 5 60 200
        # batch_size, output_len, enc_hidden_size

        output = torch.cat((context, output), dim=2)  # batch_size, output_len, enc_hidden_size*2 + dec_hidden_size

        output = output.view(batch_size * output_len, -1)  # batch_size*output_len, enc_hidden_size*2 + dec_hidden_size 5 60 300
        output = torch.tanh(self.linear_out(output))  # batch_size*output_len, dec_hidden_size
        output = output.view(batch_size, output_len, -1)  # batch_size output_len, dec_hidden_size
        return output, attn


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        self.rnn = nn.GRU(2*enc_hidden_size, dec_hidden_size, batch_first=True)
        self.out = nn.Linear(72*dec_hidden_size, 4)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self, x_len, y_len):
        # a mask of shape x_len * y_len
        device = x_len.device
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_mask = torch.arange(max_x_len, device=x_len.device)[None, :] < x_len[:, None]  #5， 60
        y_mask = torch.arange(max_y_len, device=x_len.device)[None, :] < y_len[:, None]  #5， 60
        mask = (1 - x_mask[:, :, None] * y_mask[:, None, :]).byte()
        return mask

    def forward(self, ctx, ctx_lengths, y, y_lengths, hid):  #50 60 200   12 24 36 48 60   1 5 100
        b, c, w = ctx.size()
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]  # 1 5 100

        # y_sorted = self.dropout(y_sorted)  # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True) #50 60 100
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()  #5 60 100
        hid = hid[:, original_idx.long()].contiguous() #1 5 100

        mask = self.create_mask(y_lengths, ctx_lengths)

        output, attn = self.attention(output_seq, ctx, mask)
        output = output.view(b, -1)
        padding_size = 72 - output.shape[1]/100
        padding = torch.zeros(b, int(padding_size)*100).cuda()
        output = torch.cat((output,padding), dim= 1)
        output = self.out(output)

        return output, hid, attn
            # 5 60 72  1 500 100  5 60 60


class Gaussian_Attentation(nn.Module):
    def __init__(self, in_channel):
        super(Gaussian_Attentation, self).__init__()
        mean1 = [32.]
        mean1 = torch.FloatTensor(mean1)
        mean2 = [96.]
        mean2 = torch.FloatTensor(mean2)
        sigma = [1.]
        sigma = torch.FloatTensor(sigma)
        self.mean_1 = nn.Parameter(data = mean1, requires_grad=True)
        self.mean_2 = nn.Parameter(data=mean2, requires_grad=True)
        self.sigma_1 = nn.Parameter(data=sigma, requires_grad=True)
        self.sigma_2 = nn.Parameter(data=sigma, requires_grad=True)

        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channel,
                16,
                kernel_size=1,
                stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.Conv2d(
                16,
                32,
                kernel_size=1,
                stride=(1, 1)),
            nn.BatchNorm2d(32),
        )

        self.conv1 = nn.Conv2d(32, 64, kernel_size=(8, 1), stride=(4, 1))
        self.batchNorm1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)  # 16 65 1
        self.avgpool1 = nn.AvgPool2d((3, 1), stride=(1, 1))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(1, 1))
        self.batchNorm2 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)  # 16 65 1


        self.conv3 = nn.Conv2d(128, 32, kernel_size=(3, 1), stride=(1, 1))
        self.batchNorm3 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)  # 16 65 1


        self.conv4 = nn.Conv2d(32, 16, kernel_size=(3, 1), stride=(1, 1))
        self.batchNorm4 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)  # 16 65 1

        self.maxpool = nn.MaxPool2d((3, 1), stride=(1, 1))

        self.relu = nn.ReLU(True)
        self.fc = nn.Linear(48, 2)

    def gaussion(self, mean, sigma):
        x_index = torch.linspace(0, 71, 72, dtype=torch.float64)
        x_index = x_index.cuda() if torch.cuda.is_available() else x_index
        pi = torch.FloatTensor([np.pi]).double()
        pi = pi.cuda() if torch.cuda.is_available() else pi
        sigma = 4
        return torch.exp(-1 * ((x_index - mean) ** 2) / (2 * (sigma ** 2))) / (
            torch.sqrt(2 * pi * sigma))

    def forward(self, x):
        res = self.residual(x)  # 20 1 128 1
        x = res + x

        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.avgpool1(x)

        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = self.relu(x)  #20 16 25 1
        x = self.maxpool(x)

        b, c, h, w = x.size()
        x = x.view(b, c*h)
        x = torch.abs(self.fc(x))
        x = x.double()
        # a = self.gaussion(x[0][0], x[0][1])
        # b = self.gaussion(x[0][2], x[0][3])
        # c = a + b
        #
        # index =  np.linspace(0,127, 128)
        # a = np.array(a.detach())
        # b = np.array(b.detach())
        # c = np.array(c.detach())
        # plt.plot(index, a, 'r', label='m=0,sig=1')
        # plt.plot(index, b, 'b', label='m=1,sig=2')
        # plt.plot(index, c, 'c', label='m=3,sig=3')
        # plt.grid()
        # plt.show()

        attn = [self.gaussion(10, x[i][0]) + self.gaussion(28, x[i][1]) for i in range(len(x))]
        attn = torch.stack(attn)
        return attn.view(b, 1, -1, 1).float(), x


class PlainEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super(PlainEncoder, self).__init__()
        # self.embed = nn.Embedding(vocab_size, hidden_size)
        # print("vocab_size---->")
        # print(vocab_size)
        self.input = nn.Linear(vocab_size, hidden_size )
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):  # 最后一个hidden state
        # 把 batch里面的seq按照长度排序
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx.long()]
        # embedded = self.dropout(self.embed(x_sorted))
        # print("embedded.size()---->")
        # print(embedded.size())
        x_sorted = self.input(x_sorted)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(x_sorted, sorted_len.long().cpu().data.numpy(),
                                                            batch_first=True)
        packed_out, hid = self.rnn(packed_embedded)

        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        print("out.size()---->")
        print(out.size())
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:, original_idx.long()].contiguous()

        return out, hid[[-1]]


class PlainDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.2):
        super(PlainDecoder, self).__init__()
        # self.embed = nn.Embedding(vocab_size, hidden_size)
        self.vocab_size = vocab_size
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, y, y_lengths, hid):
        b, c, w = y.size()
        sorted_len, sorted_idx = y_lengths.sort(0, descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:, sorted_idx.long()]

        # y_sorted = self.dropout(self.embed(y_sorted))  # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted, sorted_len.long().cpu().data.numpy(), batch_first=True)
        out, hid = self.rnn(packed_seq, hid)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        _, original_idx = sorted_idx.sort(0, descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        #         print(output_seq.shape)
        hid = hid[:, original_idx.long()].contiguous()

        output = self.out(output_seq)

        padding = output.view(b, -1)
        padding_size = 72 - padding.shape[1] / self.vocab_size
        padding = torch.zeros(b, int(padding_size), 3).cuda()
        output =  torch.cat((output, padding), dim = 1)
        return output, hid

if __name__ == '__main__':
   # nx = torch.rand(100, 1, 72, 1).float()
   # # model = EncoderLayer(
   # #          d_model=64, d_inner=2048,
   # #          n_head=8, d_k=32, d_v=32,
   # #           dropout=0.1).cuda()
   # model =  Gaussian_Attentation(in_channel = 1)
   #
   # output = model(nx)
   # print(output)

   nx = np.zeros((5, 72))

   len_nx = []
   for i in range(len(nx)):
       for j in range((i + 1) * 12):
           nx[i][j] = 1
       len_nx.append((i + 1) * 12)

   nx = torch.from_numpy(nx.reshape((5, 72, 1))).float().cuda()
   len_nx = np.array(len_nx)
   len_nx = torch.from_numpy(len_nx).long().cuda()
   print(nx)
   ec = Encoder(vocab_size=72,
                   embed_size=1,
                   enc_hidden_size=100,
                   dec_hidden_size=100,
                   dropout=0.2).cuda()
   de = Decoder(vocab_size=72,
                   embed_size=1,
                   enc_hidden_size=100,
                   dec_hidden_size=100,
                   dropout=0.2).cuda()
   encoder_out, hid = ec(nx, len_nx) # 5 60 200      1 5 100
   output, hid, attn = de(ctx = encoder_out,
                    ctx_lengths = len_nx,
                    y = encoder_out,
                    y_lengths = len_nx,
                    hid=hid)
   print(output.size())