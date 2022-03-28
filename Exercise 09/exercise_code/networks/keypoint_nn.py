"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl


class ResidualBlock(nn.Module):

    def __init__(self, in_channel_num, out_channel_num, use_11conv=False, stride=1):
        super(ResidualBlock, self).__init__()

        self.main_path = nn.Sequential(
            nn.Conv2d(in_channel_num, out_channel_num, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channel_num),
            nn.ReLU(),
            nn.Conv2d(out_channel_num, out_channel_num, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel_num)
        )

        if use_11conv:
            self.side_path = nn.Conv2d(in_channel_num, out_channel_num, kernel_size=1, stride=stride)
        else:
            self.side_path = None

    def forward(self, x):
        if self.side_path:
            return F.relu(self.main_path(x) + self.side_path(x))
        else:
            return F.relu(self.main_path(x) + x)


class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hyper parameters
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super(KeypointModel, self).__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the key points.       #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # max pooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # over fitting.                                                        #
        ########################################################################

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.block2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        self.block3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        self.block4 = nn.Sequential(*self.resnet_block(128, 224, 2))

        self.net = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(224, 30)
        )

        self.initialize()

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def resnet_block(self, in_channel_num, out_channel_num, residual_block_num, first_block=False):
        blocks = []
        for i in range(0, residual_block_num):
            if i == 0 and not first_block:
                blocks.append(ResidualBlock(in_channel_num, out_channel_num, use_11conv=True, stride=2))
            else:
                blocks.append(ResidualBlock(out_channel_num, out_channel_num))
        return blocks

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted key points                                   #
        ########################################################################

        x = self.net(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the key points of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
