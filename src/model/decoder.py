import torch.nn as nn
from .modules.modules import *


class Decoder(nn.Module):
    def __init__(self, kernels=[8, 16, 32, 64, 256]): # all 8 norms to batch
        super().__init__()  
        # Decoder stages
        # 1st SMR block
        self.red4 = Convolution2d1x1_sel(in_channels=kernels[4], out_channels=kernels[3], norm='batch', activation='relu')
        self.upconv4 = nn.ConvTranspose2d(kernels[3], kernels[3], kernel_size=2, stride=2) # 256, 128
        self.dec4_fusion = FusionBlock(in_channels=kernels[3]*2, out_channels=kernels[3]) # 2x Convlayers
        self.dec4_1 = DoubleConvolution2d_sel(kernels[3], kernels[3], norm='batch')
        # end of 1st SMR block

        # 2nd SMR block
        self.red3 = Convolution2d1x1_sel(in_channels=kernels[3], out_channels=kernels[2], norm='batch', activation='relu')
        self.upconv3 = nn.ConvTranspose2d(kernels[2], kernels[2], kernel_size=2, stride=2) # 128, 64
        self.dec3_fusion = FusionBlock(in_channels=kernels[2]*2, out_channels=kernels[2]) # 2x Convlayers
        self.dec3_1 = DoubleConvolution2d_sel(kernels[2], kernels[2], norm='batch')
        # end of 2nd SMR block

        # 3rd SMR block
        self.red2 = Convolution2d1x1_sel(in_channels=kernels[2], out_channels=kernels[1], norm='batch', activation='relu')
        self.upconv2 = nn.ConvTranspose2d(kernels[1], kernels[1], kernel_size=2, stride=2) # 64, 32
        self.dec2_fusion = FusionBlock(in_channels=kernels[1]*3, out_channels=kernels[1]*2) # 2x Convlayers
        self.dec2_1 = DoubleConvolution2d_sel(kernels[1]*2, kernels[1], norm='batch') # 2x Convlayers
        # end of 3rd SMR block

        # 4th SMR block
        self.red1 = Convolution2d1x1_sel(in_channels=kernels[1], out_channels=kernels[0], norm='batch', activation='relu')
        self.upconv1 = nn.ConvTranspose2d(kernels[0], kernels[0], kernel_size=2, stride=2) # 32, 16
        self.dec1_fusion = FusionBlock(in_channels=kernels[0]*2, out_channels=kernels[0]) # 2x Convlayers
        self.dec1_1 = DoubleConvolution2d_sel(kernels[0], kernels[0], norm='batch') # 2x Convlayers
        # end of 4th SMR block

    def forward(self, b, s11, s12, s13, s14):
        # Up the decoder Path, and incorporating skip connections from corresponding encoder stages
        d4 = self.red4(b)
        d4 = self.upconv4(d4)
        d4 = self.dec4_fusion(s14, d4) 
        d4 = self.dec4_1(d4)

        d3 = self.red3(d4)
        d3 = self.upconv3(d3)
        d3 = self.dec3_fusion(s13, d3)
        d3 = self.dec3_1(d3)

        d2 = self.red2(d3)
        d2 = self.upconv2(d2)
        d2 = self.dec2_fusion(s12, d2)
        d2 = self.dec2_1(d2)

        d1 = self.red1(d2)
        d1 = self.upconv1(d1)
        d1 = self.dec1_fusion(s11, d1)
        d1 = self.dec1_1(d1)

        if self.training:
            return d1, d4
        else: 
            return d1