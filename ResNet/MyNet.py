import torch.nn as nn
import torch


class Conv2d_BN_ReLU(nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size,stride=1,padding=0,groups=1):

        super(Conv2d_BN_ReLU, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding=padding,groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        """
        瓶颈结构的残差块
        conv1x1
        gconv3x3
        conv1x1
        :param in_channels:
        """
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            Conv2d_BN_ReLU(in_channels, in_channels//2, 1,1),
            Conv2d_BN_ReLU(in_channels//2, in_channels//2, 3, 1 ,padding=1, groups=2),
            nn.Conv2d(in_channels//2, in_channels, 1, 1),
            nn.BatchNorm2d(in_channels)
        )

        self.active = nn.ReLU(True)

    def forward(self,x):# 残差是两者相加的结果再激活，而不应该是激活后再相加
        return self.active(x + self.layer(x))


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn_layer = nn.Sequential(
            # 224 x 224
            Conv2d_BN_ReLU(3,16,3,1),
            Conv2d_BN_ReLU(16,32,3,1),
            nn.MaxPool2d(2,2),#110
            ResidualBlock(32),
            ResidualBlock(32),
            Conv2d_BN_ReLU(32,64,3,1),
            nn.MaxPool2d(2,2),#54
            ResidualBlock(64),
            ResidualBlock(64),
            Conv2d_BN_ReLU(64,128,3,1),
            nn.MaxPool2d(2,2),#26
            ResidualBlock(128),
            ResidualBlock(128),

            # 先扩通道
            Conv2d_BN_ReLU(128,256,1,1,groups=2)
        )

        self.mobile_layer = nn.Sequential(
            Conv2d_BN_ReLU(256,256,3,1, groups=256),
            Conv2d_BN_ReLU(256,128,1,1,groups=2),
            nn.AvgPool2d(2, 2),
        )

        self.mobile_layer2 = nn.Sequential(
            Conv2d_BN_ReLU(128, 128, 3, 1, groups=128),
            Conv2d_BN_ReLU(128, 64, 1,1)
        )

        self.cnn_layer2 = nn.Sequential(
            Conv2d_BN_ReLU(64, 128, 10,1),
            nn.Conv2d(128,5,1,1)
        )

    def channel_shuffle(self, x, groups):
        N,C,H,W = x.shape
        x = x.reshape(N,groups, C//groups, H, W)
        x = x.permute(0,2,1,3,4)
        x = x.reshape(N,C,H,W)
        return x

    def channel_shuffle2(self,x, *channels):
        """
        自定义通道数目
        :return:
        """
        N,C,H,W = x.shape
        indices = list(range(len(channels),0,-1))
        x = x.reshape(N, *channels, H,W)
        x = x.permute(0, *indices, x.dim()-2,x.dim()-1)
        x = x.reshape(N,C,H,W)
        return  x


    def forward(self, x):
        x = self.cnn_layer(x)
        x = self.channel_shuffle(x,16)
        x = self.mobile_layer(x)
        x = self.channel_shuffle(x, 8)
        x = self.mobile_layer2(x)
        x = self.cnn_layer2(x)
        x = x.squeeze()
        category = torch.sigmoid(x[:,0])
        axes = torch.relu(x[:, 1:])

        return category, axes
