
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .modules.proposed_modules import CSF

class MambaWeed_Net(nn.Module): # 1
    def __init__(self, in_channels=3, out_channels=1, kernels=[8, 16, 32, 64, 256]):
        super().__init__()

        # Encoder  
        self.encoder = Encoder(in_channels=in_channels, out_channels=out_channels, kernels=kernels)

        # Bottleneck
        self.bottleneck = CSF(kernels=kernels)

        # Decoder
        self.decoder = Decoder(kernels=kernels)

        # Output Layer
        self.final_convolution = nn.Conv2d(kernels[0], out_channels, kernel_size=1) # 16, 3)
    
    def forward(self, x):
        
        s11, s12, s13, s14, p14 = self.encoder(x)

        b = self.bottleneck(p14)

        if self.training:
            d1, d4 = self.decoder(b, s11, s12, s13, s14)
            return self.final_convolution(d1), d4, s14 
        else:
            d1 = self.decoder(b, s11, s12, s13, s14)
            conv = self.final_convolution(d1)
            return conv