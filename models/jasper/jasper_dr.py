import torch
import torch.nn as nn
from models.jasper.blocks import InBlock, OutBlock, EncoderDR, Decoder

class JasperDR(nn.Module):
    def __init__(self):
        super(JasperDR, self).__init__()
    
        self.prolog = InBlock(in_channels=128,
                              out_channels=256,
                              kernel_size=11,
                              stride=2,
                              padding=5,
                              drop_rate=0.2)
        
        self.encoder = EncoderDR()
        self.decoder = Decoder()
    
    def forward(self, x):
        x = self.prolog(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x