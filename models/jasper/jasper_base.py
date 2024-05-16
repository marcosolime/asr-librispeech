import torch
import torch.nn as nn
from models.jasper.blocks import InBlock, OutBlock, Encoder, Decoder

class Jasper(nn.Module):
    def __init__(self):
        super(Jasper, self).__init__()
    
        self.prolog = InBlock(in_channels=128,
                              out_channels=256,
                              kernel_size=11,
                              stride=2,
                              padding=5,
                              drop_rate=0.2)
        
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        x = x.squeeze()

        x = self.prolog(x)
        x = self.encoder(x)
        x = self.decoder(x)
        
        x = x.transpose(1, 2).contiguous()
        return x