import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableHaarDWT(nn.Module):
    def __init__(self, in_channels=1, learnable_filters=True):
        """
        Learnable 2D Haar DWT layer for PyTorch.
        
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            learnable_filters: Whether to make wavelet filters trainable
        """
        super().__init__()
        self.in_channels = in_channels
        
        # Initialize Haar wavelet filters
        low_pass = torch.tensor([1.0, 1.0], dtype=torch.float32) / torch.sqrt(torch.tensor(2.0))
        high_pass = torch.tensor([-1.0, 1.0], dtype=torch.float32) / torch.sqrt(torch.tensor(2.0))
        
        # Make filters learnable if specified
        if learnable_filters:
            self.low_pass = nn.Parameter(low_pass).to('cuda')
            self.high_pass = nn.Parameter(high_pass).to('cuda')
        else:
            self.register_buffer('low_pass', low_pass).to('cuda')
            self.register_buffer('high_pass', high_pass).to('cuda')
        
        # Prepare filters for convolution
        self._prepare_filters()
        
        self.cAs = None
        self.cHs = None
        self.cVs = None
        self.cDs = None

    def _prepare_filters(self):
        """Reshape filters for convolution operations"""
        # For row-wise operations (last dimension)
        self.low_pass_row = self.low_pass.view(1, 1, 1, -1).to('cuda')
        self.high_pass_row = self.high_pass.view(1, 1, 1, -1).to('cuda')
        
        # For column-wise operations (second last dimension)
        self.low_pass_col = self.low_pass.view(1, 1, -1, 1).to('cuda')
        self.high_pass_col = self.high_pass.view(1, 1, -1, 1).to('cuda')

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Tuple of (cA, cH, cV, cD) subbands, each of shape (B, C, H/2, W/2)
        """
        B, C, H, W = x.shape
        
        # Pad if dimensions are odd
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='reflect')
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0), mode='reflect')
        
        # Process each channel separately
        outputs = []
        cA_list, cH_list, cV_list, cD_list = [], [], [], []
        for c in range(C):
            channel = x[:, c:c+1, :, :]
            
            # Row-wise DWT
            low_row = F.conv2d(channel, self.low_pass_row, stride=(1, 2))
            high_row = F.conv2d(channel, self.high_pass_row, stride=(1, 2))
            
            # Column-wise DWT
            cA = F.conv2d(low_row, self.low_pass_col, stride=(2, 1))
            cH = F.conv2d(low_row, self.high_pass_col, stride=(2, 1))
            cV = F.conv2d(high_row, self.low_pass_col, stride=(2, 1))
            cD = F.conv2d(high_row, self.high_pass_col, stride=(2, 1))
            
            outputs.extend([cA, cH, cV, cD])
            cA_list.append(cA)
            cH_list.append(cH)
            cV_list.append(cV)
            cD_list.append(cD)


        self.cAs = torch.cat(cA_list, dim=1)        
        self.cHs = torch.cat(cH_list, dim=1)        
        self.cVs = torch.cat(cV_list, dim=1)        
        self.cDs = torch.cat(cD_list, dim=1)

        # Stack all channels and reshape
        out = torch.cat(outputs, dim=1)
        return out

class Convolution2d_sel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm='batch', activation='relu', activation_inplace='True'):
        super().__init__()
        
        if norm not in ['batch', 'group']:
            conv_bias = True
        else:
            conv_bias = False

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=conv_bias),
            ]


        # selection normalization
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))

        elif norm == 'group':
            layers.append(nn.GroupNorm(num_groups= out_channels // 2, num_channels = out_channels))
        
        
        # Activation function selection
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


class DepthWiseSeparableConvolution_sel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm='batch', activation='relu', activation_inplace=True):
        super().__init__()
        
        if norm not in ['batch', 'group']:
            conv_bias = True
        else:
            conv_bias = False


        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=conv_bias, groups=in_channels), # Depthwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=conv_bias), # Pointwise convolution
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


        self.dwp_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.dwp_conv(x)




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
