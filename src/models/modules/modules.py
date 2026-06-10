import torch.nn as nn
import torch
from .Mamba4ScanSS import VSSBlock_seldir




class GroupedSeparableConvolution_sel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=4, activation = 'relu', norm='batch', activation_inplace=True):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias='False', groups=groups), # Depthwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias='False'), # Pointwise convolution
            
        ]

        # selection normalization
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))

        elif norm == 'group':
            layers.append(nn.GroupNorm(num_groups= out_channels // 2, num_channels = out_channels))
        
        else:
            raise AssertionError(f"{norm} is not allowed choice.")

        if activation == 'relu':
            layers.append(nn.ReLU(inplace=activation_inplace))
        
        elif activation == 'silu':
            layers.append(nn.SiLU(inplace=activation_inplace))
        
        elif activation == 'gelu':
            layers.append(nn.GELU())
        
        else:
            raise AssertionError(f"{norm} is not allowed choice.")

        self.dwp_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.dwp_conv(x)
    




class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=4):
        super().__init__()

        self.layer1 = nn.Conv2d(channels, channels // ratio, kernel_size=1, bias=True)
        self.activation = nn.ReLU(inplace=True)
        self.layer2 = nn.Conv2d(channels // ratio, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)


        avg_out = self.layer2(self.activation(self.layer1(avg_pool)))
        max_out = self.layer2(self.activation(self.layer1(max_pool)))

        feats = avg_out + max_out

        attention_weights = self.sigmoid(feats)

        return x * attention_weights


class Convolution2d1x1_silu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        return self.conv(x)

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

CSF = MultiScaleCrossFusion_4VSS



class DoubleConvolution2d_sel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, norm='batch', activation='relu', activation_inplace=True):
        super().__init__()


        if norm not in ['batch', 'group']:
            conv_bias = True
        else:
            conv_bias = False


        def get_norm(channels):
            if norm not in ['batch', 'group']:
                return nn.Identity()
            if norm == 'batch':
                return nn.BatchNorm2d(channels)
            elif norm == 'group':
                return nn.GroupNorm(num_groups=channels // 2, num_channels=channels)
            else:
                raise ValueError(f"Unsupported norm_type: {norm}")

        def get_activation():
            if activation == 'relu':
                return nn.ReLU(inplace=activation_inplace)
            elif activation == 'silu':
                return nn.SiLU(inplace=activation_inplace)
            elif activation == 'gelu':
                return nn.GELU()
            elif activation == 'none':
                return nn.Identity()
            else:
                raise ValueError(f"Unsupported activation: {activation}")

        self.block = nn.Sequential(
            # First conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=conv_bias),
            get_norm(out_channels), # Was used
            get_activation(),

            # Second conv layer (pointwise)
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=conv_bias),
            get_norm(out_channels), # Was used
            get_activation()
        )

    def forward(self, x):
        return self.block(x)
    
class Convolution2d1x1_sel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, norm='batch', activation='relu', activation_inplace=True):
        super().__init__()
        
        if norm not in ['batch', 'group']:
            conv_bias = True
        else:
            conv_bias = False
        
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, stride=1, bias=conv_bias),
            # nn.ReLU(inplace=activation_inplace),
            # nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, stride=1, bias=conv_bias)
            ]


        # selection normalization
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))

        elif norm == 'group':
            layers.append(nn.GroupNorm(num_groups= out_channels // 2, num_channels = out_channels))
        

        if activation == 'relu':
            layers.append(nn.ReLU(inplace=activation_inplace))
        
        elif activation == 'silu':
            layers.append(nn.SiLU(inplace=activation_inplace))
        
        elif activation == 'gelu':
            layers.append(nn.GELU())
        
        else:
            raise AssertionError(f"{norm} is not allowed choice.")

        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.fusionblock = nn.Sequential(
            GroupedSeparableConvolution_sel(in_channels=in_channels, out_channels=in_channels, norm='batch'),
            ChannelAttention(channels=in_channels),
            Convolution2d1x1_sel(in_channels=in_channels, out_channels=out_channels, norm='batch', activation='silu')
        )

    def forward(self, x1, x2):
        return self.fusionblock(torch.cat([x1, x2], dim=1))