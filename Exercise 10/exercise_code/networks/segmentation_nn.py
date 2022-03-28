"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision import models


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.num_classes = num_classes
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        # The encoder part
        self.encoder = models.alexnet(pretrained=True).features

        # The decoder part
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 4096, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(4096),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Upsample(scale_factor=8, mode="bilinear"),
            nn.Conv2d(4096, 256, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Upsample(scale_factor=5, mode="bilinear"),
            nn.Conv2d(256, self.num_classes, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.num_classes),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(self.num_classes, self.num_classes, kernel_size=3, padding=1, stride=1),
        )

        self.initialize()

    def initialize(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.encoder(x)
        x = self.decoder(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()

        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

