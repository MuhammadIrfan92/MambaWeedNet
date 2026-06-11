import torch.nn as nn
import torch
from .Mamba4ScanSS import VSSBlock_seldir
from .submodules import *




class Local_feature_extraction_module(nn.Module):
    def __init__ (self, in_channels, out_channels, groups=4):
        super().__init__()
        if in_channels % groups != 0:
            AssertionError(f"in_channels {in_channels} must be divisble by groups {groups}")
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, stride=1, groups=groups),
            nn.BatchNorm2d(in_channels),
            # nn.GroupNorm(num_groups= in_channels // 2, num_channels = in_channels),
            nn.ReLU(inplace=True)
            )
            
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels+in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=groups),
            nn.BatchNorm2d(in_channels),
            # nn.GroupNorm(num_groups= in_channels // 2, num_channels = in_channels),
            nn.ReLU(inplace=True)
            )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels+in_channels*2, in_channels, kernel_size=3, padding=1, stride=1, groups=groups),
            nn.BatchNorm2d(in_channels),
            # nn.GroupNorm(num_groups= in_channels // 2, num_channels = in_channels),
            nn.ReLU(inplace=True)
            )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels+in_channels*3, out_channels, kernel_size=1, padding=0, stride=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            # nn.GroupNorm(num_groups= out_channels // 2, num_channels = out_channels),
            nn.ReLU(inplace=True) 
            )
       
    
    def forward(self, x):
        y = self.layer1(x)
        
        x1 = torch.cat([x,y], dim=1)
        y1 = self.layer2(x1)

        x2 = torch.cat([x, y, y1], dim=1)
        y2 = self.layer3(x2)

        x3 = torch.cat([x, y, y1, y2], dim=1)
        return self.layer4(x3)




class MultiGranularityFusion_4VSS(nn.Module):
    def __init__(self, in_ch, mid_ch, mamba_in_ch, mamba_out_ch, res_out_ch=None, norm='batch', activation='gelu'):
        super().__init__()
        d = 4 # number of depth_wise chunks for each scan
        self.conv = DoubleConvolution2d_sel(in_ch, mid_ch, norm='batch')
        self.lfe = Local_feature_extraction_module(in_ch, mid_ch)

        self.conv1x1 = Convolution2d1x1_sel(mamba_in_ch, mamba_in_ch, norm='batch')
        # Mambaba4Scan block
        mamba_in_ch_d = int(mamba_in_ch/d)
        self.mamba_hw_forward = VSSBlock_seldir(mamba_in_ch_d, direction="hw_forward")
        self.mamba_wh_forward = VSSBlock_seldir(mamba_in_ch_d, direction="wh_forward")
        self.mamba_hw_reverse = VSSBlock_seldir(mamba_in_ch_d, direction="hw_reverse")
        self.mamba_wh_reverse = VSSBlock_seldir(mamba_in_ch_d, direction="wh_reverse")
        self.conv1x1_mamba = Convolution2d1x1_sel(mamba_in_ch, mamba_out_ch, norm='batch')
        # end of Mamba4Scan block
        
        # self.mamba = MultiDirectionalMambaScan(1, mamba_in_ch, mamba_out_ch)
        if res_out_ch is None:
            res_out_ch = mamba_out_ch
        self.res = Convolution2d1x1_sel(in_ch, res_out_ch, norm='batch', activation=activation)

    def forward(self, x):
        s1 = self.conv(x)
        s2 = self.lfe(x)
        s_res = self.res(x)
        cat_s = self.conv1x1(torch.cat([s1, s2], dim=1)).permute(0, 2, 3, 1)
        
        m1, m2, m3, m4 = torch.chunk(cat_s, 4, dim=-1)
        sm1 = self.mamba_hw_forward(m1)
        sm2 = self.mamba_wh_forward(m2)
        sm3 = self.mamba_hw_reverse(m3)
        sm4 = self.mamba_wh_reverse(m4)
        s = self.conv1x1_mamba(torch.cat([sm1, sm2, sm3, sm4], dim=-1).permute(0, 3, 1, 2))
        return s + s_res



class MultiScaleCrossFusion_4VSS(nn.Module):
    def __init__(self, kernels=[8, 16, 32, 64, 256]):
        super().__init__()  
            # self.fusionB = FusionBlock(kernels[3]*2, kernels[3]*2)
        d = 4 # number of depth_wise chunks for each scan
        
        self.bot1_1 = GroupedSeparableConvolution_sel(kernels[3], kernels[3], norm='batch', activation='silu', activation_inplace=True) # 1
        self.bot1_2 = GroupedSeparableConvolution_sel(kernels[3]*2, kernels[3], norm='batch', activation='silu', activation_inplace=True) # 2
        self.bot1_3 = GroupedSeparableConvolution_sel(kernels[3]*2, kernels[3], norm='batch', activation='silu', activation_inplace=True) # 3

        self.bot2_1 = GroupedSeparableConvolution_sel(kernels[3], kernels[3], norm='batch', activation='silu', activation_inplace=True) # 4
        self.bot2_2 = GroupedSeparableConvolution_sel(kernels[3]*2, kernels[3], norm='batch', activation='silu', activation_inplace=True) # 5
        self.bot2_3 = GroupedSeparableConvolution_sel(kernels[3]*2, kernels[3], norm='batch', activation='silu', activation_inplace=True) # 6
        self.bot0_conv1x1 = Convolution2d1x1_sel(in_channels= kernels[3]*2, out_channels= kernels[4], norm='batch', activation='silu', activation_inplace=True)

       # Mambaba4Scan block
        mamba_in_ch_d = int(kernels[3]/d)
        self.mamba_hw_forward = VSSBlock_seldir(mamba_in_ch_d, direction="hw_forward")
        self.mamba_wh_forward = VSSBlock_seldir(mamba_in_ch_d, direction="wh_forward")
        self.mamba_hw_reverse = VSSBlock_seldir(mamba_in_ch_d, direction="hw_reverse")
        self.mamba_wh_reverse = VSSBlock_seldir(mamba_in_ch_d, direction="wh_reverse")
        self.conv1x1_mamba = Convolution2d1x1_sel(kernels[3], kernels[4], norm='batch')
        # end of Mamba4Scan block


        self.bot_conv1x1 = Convolution2d1x1_sel(in_channels= kernels[4]*2, out_channels= kernels[4], norm='batch', activation='silu', activation_inplace=True)
        self.bottleneck_res = nn.Sequential(
            Convolution2d1x1_silu(kernels[3], kernels[4]),
            Convolution2d1x1_silu(kernels[4], kernels[4]),
        )

    def forward(self, p14):
        b1_1 = self.bot1_1(p14)
        b2_1 = self.bot2_1(p14)

        b1_2 = self.bot1_2(torch.cat([b1_1, b2_1], dim=1))
        b2_2 = self.bot1_2(torch.cat([b2_1, b1_1], dim=1))

        b1_3 = self.bot1_3(torch.cat([b1_2, b2_2], dim=1))
        b2_3 = self.bot1_3(torch.cat([b2_2, b1_2], dim=1))
        b = self.bot0_conv1x1(torch.cat([b1_3, b2_3], dim=1))
    
        # Mamba4Scan block
        m1, m2, m3, m4 = torch.chunk(p14.permute(0, 2, 3, 1), 4, dim=-1)
        sm1 = self.mamba_hw_forward(m1)
        sm2 = self.mamba_wh_forward(m2)
        sm3 = self.mamba_hw_reverse(m3)
        sm4 = self.mamba_wh_reverse(m4)
        b_mamba = self.conv1x1_mamba(torch.cat([sm1, sm2, sm3, sm4], dim=-1).permute(0, 3, 1, 2))
        # end of Mamba4Scan block

        b = torch.cat([b, b_mamba], dim=1)
        b = self.bot_conv1x1(b)
        b_res = self.bottleneck_res(p14)
        b = b + b_res

        return b


class Conv_Wavelet_Fusion(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Branch 1
        self.branch1 = nn.Sequential(
            Convolution2d_sel(in_channels, out_channels, norm='batch', activation='relu'), 
            Convolution2d_sel(out_channels, out_channels, norm='batch', activation='relu') 
        )

        # Bracnch 2
        self.branch2 = LearnableHaarDWT(out_channels, learnable_filters=True)

        self.DSP_Conv = DepthWiseSeparableConvolution_sel(out_channels*5, out_channels, norm='batch') 


    def forward(self, x):
        branch1_output = self.branch1(x)
        x1_t = self.pool(branch1_output)
        _ = self.branch2(branch1_output)

        fA_t = (x1_t+self.branch2.cAs) * self.branch2.cAs
        fH_t = (x1_t+self.branch2.cHs) * self.branch2.cHs
        fV_t = (x1_t+self.branch2.cVs) * self.branch2.cVs
        fD_t = (x1_t+self.branch2.cDs) * self.branch2.cDs

        fused = torch.cat([fA_t, fH_t, fV_t, fD_t, x1_t], dim=1)

        branch2_output = self.DSP_Conv(fused)

        return branch1_output, branch2_output