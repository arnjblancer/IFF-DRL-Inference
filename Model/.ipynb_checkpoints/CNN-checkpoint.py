import math

import torch
from torch import nn


class PreCNN(nn.Module):
    def __init__(self, c, t, outChanel1D, outChanel2D):
        """
        :param:
        :return:
        """
        super().__init__()
        self.c = c
        self.conv1 = nn.ModuleList([self.conv1_layer(outChanel1D) for i in range(c)])
        self.conv1_3_c528 = nn.Sequential(
            nn.Conv2d(c, outChanel2D, 1, padding=0),
            nn.Tanh(),
            nn.BatchNorm2d(outChanel2D),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, outChanel2D, kernel_size=c-2),
            nn.Tanh(),
            nn.BatchNorm2d(outChanel2D),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, outChanel2D, kernel_size=c),
            # B, 2D, 1, T - c + 1
            nn.Tanh(),
            nn.BatchNorm2d(outChanel2D),
            nn.Flatten(2, 3),
            #V, 2D, T-2
            nn.Linear(t - c + 1, t-2)
        )

    @staticmethod
    def conv1_layer(outChanel1D):
        """
        :return: 1*5 的一维卷积结构
        b, 1, 30
        b, 8, 26
        b, 1, 22
        """
        return nn.Sequential(
            nn.Conv1d(1, outChanel1D, kernel_size=3, padding=0),
            nn.Tanh(),
            nn.BatchNorm1d(outChanel1D),
        )

    def conv1_3(self, x):
        # input_size = [b, 1, 30]
        x_split = [x[:, i:i + 1, :] for i in range(self.c)]
        a = x_split[0].type()
        return torch.cat([torch.unsqueeze(self.conv1[index](x_split[index]), dim=1) for index in range(self.c)], dim=1)

    def conv3_3(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.conv2(x)
        return x

    def conv5_5(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.conv3(x)
        return x

    def forward(self, x):
        """
        :param:
        :return:
        """
        future1_3 = self.conv1_3_c528(self.conv1_3(x))
        # print(future1_3.size())
        # b, outChanel2D, outChanel=1D, t-2

        # x=b,1,c,t
        # b, outChanel2D, c-2, t-4
        future3_3 = self.conv3_3(x)
        future5_5 = self.conv5_5(x)
        future = torch.cat([future1_3, future3_3, torch.unsqueeze(future5_5, dim=2)], dim=2)
        # out -> b, 2d, 1d + 1 + c - 2 ->b, 2d ,1d + c - 1
        return future


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class BackBone(nn.Module):
    def __init__(self, inChannel, h, w, outChannel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d( inChannel, outChannel, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(outChannel),
            nn.MaxPool2d(2),
            nn.Conv2d(outChannel, outChannel, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(outChannel),
        )
        self.eca = eca_block(outChannel)

    def forward(self, x):
        """
        :param x: b, 2 * 2d , 1d + c - 1, t-2
        :return:
        """
        block = self.conv1(x)
        se = self.eca(block)
        return se


class reward_CNN(nn.Module):
    def __init__(self,configs,
                 preCNNOutChannel1D=8,
                 preCNNOutChannel2D=8,
                 backboneOutChannel=32,
                 linearMidChannel = 8,
                 linearMid = 1000,
                 classes=3
                 ):
        """
        :param:
        :return:
        """
        super().__init__()
        self.configs = configs
        self.C = self.configs.C
        self.T = self.configs.T
        self.preCNN = PreCNN(self.C, self.T, preCNNOutChannel1D, preCNNOutChannel2D)
        self.backbone = BackBone(preCNNOutChannel2D, preCNNOutChannel1D + self.C - 1, self.T-2, backboneOutChannel)
        self.linear = nn.Sequential(
            # nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(backboneOutChannel, linearMidChannel, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(linearMidChannel*(preCNNOutChannel1D+self.C-1)*(self.T-2)//4, linearMid),
            nn.Linear(linearMid, classes),
            nn.Softmax()
        )


    def forward(self, x):

        # B, T, C
        x = x.permute(0, 2, 1)
        #day = x[:, 0: self.C, :]
        #week = x[:, self.C:, :]
        day_pre = self.preCNN(x)
        # b, 2d ,1d + c - 1, t-2
        #week_pre = self.preCNN(week)

        data_pre = day_pre
        # b, 2 * 2d , 1d + c - 1, t-2
        future = self.backbone(data_pre)
        # b, backboneOutChannel, (1d + c - 1)/2, (t-2)/2
        res = self.linear(future)
        return res



