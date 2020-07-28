import torch.nn as nn
import torch

class Conv1x1Bottleneck(nn.Module):
    """
    瓶颈结构
    conv1x1
    conv3x3
    conv1x1

    """
    def __init__(self, in_channels ,out_channels,kernel_size,stride=1,padding=0,groups=1):
        super(Conv1x1Bottleneck, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2,1,1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(True),
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size, stride, padding=padding, groups=groups),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(True),
            nn.Conv2d(in_channels//2, out_channels, 1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layer(x)



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

            Conv1x1Bottleneck(128,256,3,1),
            nn.AvgPool2d(2,2), #12
            Conv1x1Bottleneck(256,128,3,1),
            Conv1x1Bottleneck(128,64,3,1)
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64,128,8,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128,5,1,1)
        )

    def forward(self, x):
        x = self.cnn_layer(x)
        x = self.cnn_layer2(x)
        x = x.squeeze()
        category = torch.sigmoid(x[:,0])
        axes = torch.relu(x[:, 1:])

        return category, axes


