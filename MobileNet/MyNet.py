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
        )

        self.mobile_layer = nn.Sequential(
            nn.Conv2d(128,256,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3,1, groups=256),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.AvgPool2d(2,2),
            nn.Conv2d(256,128, 1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
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

    def forward(self, x):
        x = self.cnn_layer(x)
        x = self.mobile_layer(x)
        x = self.mobile_layer2(x)
        x = self.cnn_layer2(x)
        x = x.squeeze()
        category = torch.sigmoid(x[:,0])
        axes = torch.relu(x[:, 1:])

        return category, axes


