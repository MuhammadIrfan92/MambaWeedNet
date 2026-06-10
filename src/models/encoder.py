
import torch.nn as nn
from .modules.modules import Conv_Wavelet_Fusion, MultiGranularityFusion_4VSS
class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, kernels=[8, 16, 32, 64, 256]):
        super().__init__()  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder 1 Stages
        self.enc1_1 = Conv_Wavelet_Fusion(in_channels, kernels[0]) # in_channels adjusted for RGB (3) 

        self.enc1_2 = MultiGranularityFusion_4VSS(in_ch=kernels[0], mid_ch=kernels[1], mamba_in_ch=kernels[1]*2, mamba_out_ch=kernels[1]*2, res_out_ch=kernels[1]*2)
        # Stage 3: produces kernels[2]
        self.enc1_3 = MultiGranularityFusion_4VSS(in_ch=kernels[1]*2, mid_ch=kernels[2], mamba_in_ch=kernels[2]*2, mamba_out_ch=kernels[2], res_out_ch=kernels[2])
        # Stage 4: produces kernels[3]
        self.enc1_4 = MultiGranularityFusion_4VSS(in_ch=kernels[2], mid_ch=kernels[3], mamba_in_ch=kernels[3]*2, mamba_out_ch=kernels[3], res_out_ch=kernels[3])
    
    def forward(self, x):
                # Down the encoder Path 1
        s11, p11 = self.enc1_1(x)
        # p11 = self.pool(s11)

        s12 = self.enc1_2(p11)
        p12 = self.pool(s12)

        s13 = self.enc1_3(p12)
        p13 = self.pool(s13)

        s14 = self.enc1_4(p13)
        p14 = self.pool(s14)

        return s11, s12, s13, s14, p14