import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.nn import ModuleList, MaxPool2d, ConvTranspose2d, Conv2d



class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=(3, 3), padding=1,
                 activation=nn.ReLU, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation

        self.conv1 = nn.Conv2d(self.in_features, self.out_features[0], self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(self.out_features[0], self.out_features[1], self.kernel_size, padding=self.padding)
        if self.activation is not None:
            self.activation = self.activation()

    def forward(self, x):
        # conv + relu + conv + relu increasing/decreasing channels
        #         print('before the residual up shape: ', x.shape)

        # conv + relu 1 + conv + relu 2
        out = self.conv1(x)
        if self.activation is not None:
            out = self.activation(out)
        out = self.conv2(out)
        if self.activation is not None:
            out = self.activation(out)

        #         print('after the residual up shape: ', out.shape)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1,
                 activation=nn.ReLU, start_channels=64, factor=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding

        # branches
        self.down = ModuleList([ResidualBlock(in_feats, out_feats, self.kernel_size, self.padding, self.activation)
                                for in_feats, out_feats in zip([self.in_channels, start_channels, start_channels * factor,
                                                                start_channels * factor**2],
                                                               [(start_channels * factor**i, start_channels * factor**i) for i in range(4)
                                                               ])
                               ])
        self.pool_down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ResidualBlock(start_channels * factor**3, (start_channels * factor**4, start_channels * factor**4),
                                        self.kernel_size, self.padding, self.activation)
        self.up = ModuleList([ResidualBlock(in_feats, out_feats, self.kernel_size, self.padding, self.activation)
                              for in_feats, out_feats in zip([start_channels * factor**i for i in range(4, 0, -1)],
                                                             [(start_channels * factor**i, start_channels * factor**i) for i in range(3, -1, -1)])])
        self.conv_up = ModuleList([ConvTranspose2d(in_channels, out_channels,
                                                   kernel_size=2, stride=2)
                                   for in_channels, out_channels in zip([start_channels * factor**i for i in range(4, 0, -1)],
                                                                        [start_channels * factor**i for i in range(3, -1, -1)])])
        self.head = Conv2d(start_channels, self.out_channels, self.kernel_size, padding=self.padding)

    def forward(self, x):
#         print('shape before anything: ', x.shape)
        skips = []
#         down branch saving intermediate layers
        for block in self.down:
            x = block(x)
            skips.append(x)
#             print('skip shape: ', x.shape)
            x = self.pool_down(x)
#             print('shape after pool down: ', x.shape)

        # bottleneck
        x = self.bottleneck(x)
#         print('shape after bottleneck: ', x.shape)

        # up branch
        for up, block in zip(self.conv_up, self.up):
            x = up(x)
#             print('shape after deconv: ', x.shape)
            x = torch.cat([x, skips.pop(-1)], dim=1)
            x = block(x)
#             print('shape block up: ', x.shape)

        # head
        x = self.head(x)
#         print('final shape: ', x.shape)
        return x
