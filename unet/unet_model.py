# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        self.inc = inconv(n_channels, 8)
        self.denseblock0 = DenseBlock(8, 16)
        self.down1 = down(16, 16)
        self.denseblock1 = DenseBlock(16, 32)

        self.down2 = down(32, 32)
        self.denseblock2 = DenseBlock(32, 64)
        self.down3 = down(64, 64)
        self.denseblock3 = DenseBlock(64, 128)
        self.down4 = down(128, 128)
        # self.resblock1 = ResBlock(8, 8)

        self.up1 = up(256, 32)
        self.denseblock4 = DenseBlock(32, 64)
        self.up2 = up(128, 16)
        self.denseblock5 = DenseBlock(16, 32)
        self.up3 = up(64, 8)
        self.denseblock6 = DenseBlock(8, 16)
        self.up4 = up(32, 8)
        self.outc = outconv(8, n_classes)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.denseblock0(x0)
        
        x2 = self.down1(x1)
        x2 = self.denseblock1(x2)

        x3 = self.down2(x2)
        x3 = self.denseblock2(x3)

        x4 = self.down3(x3)
        x4 = self.denseblock3(x4)
        x5 = self.down4(x4)

        outx = self.up1(x5, x4)
        outx = self.denseblock4(outx)

        outx = self.up2(outx, x3)
        outx = self.denseblock5(outx)

        outx = self.up3(outx, x2)
        outx = self.denseblock6(outx)

        outx = self.up4(outx, x1)
        outx = self.outc(outx)

        return 4*torch.sigmoid(outx)
        
