import torch.nn as nn
import torch


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn_layer = nn.Sequential(
            # 224 x 224
            nn.Conv2d(3,16,3,1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16,32,3,1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),#110
            nn.Conv2d(32,64,3,1),#108
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),#54
            nn.Conv2d(64,128,3,1),#52
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),#26

            # 先扩通道
            nn.Conv2d(128, 256, 1, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.mobile_layer = nn.Sequential(
            nn.Conv2d(256, 256, 3,1, groups=256),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256,128, 1,1, groups=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.AvgPool2d(2, 2),
        )

        self.mobile_layer2 = nn.Sequential(
            nn.Conv2d(128, 128 , 3,1, groups=128),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64,128,10,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
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
        256       16 16
        reshape   N 4 2 2 4 2 2 H W
        permute   0 6 5 4 3 2 1 7 8
        :return:
        """
        N,C,H,W = x.shape
        indices = list(range(len(channels),0,-1))# 递减
        x = x.reshape(N, *channels, H,W)
        x = x.permute(0, *indices, x.dim()-2,x.dim()-1)
        x = x.reshape(N,C,H,W)
        return  x


    def forward(self, x):
        x = self.cnn_layer(x)
        x = self.channel_shuffle(x,16)
        x = self.mobile_layer(x)
        x = self.channel_shuffle2(x, 4,4,2,4)
        x = self.mobile_layer2(x)
        x = self.cnn_layer2(x)
        x = x.squeeze()
        category = torch.sigmoid(x[:,0])
        axes = torch.relu(x[:, 1:])

        return category, axes


if __name__ == '__main__':

    a = list(range(6))
    print(a)
    print(list(range(len(a),0,-1)))

