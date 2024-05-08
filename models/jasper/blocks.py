import torch
import torch.nn as nn

class InBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 drop_rate,
                 dilation = 1):
        
        super(InBlock, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.drop(x)
        return x

class OutBlock(nn.Module):
    def __init__(self,
                 in_channels : int,
                 out_channels : int,
                 n_blocks : int,
                 kernel_size : int,
                 drop_rate : float,
                 dilation : int = 1):
        
        super(OutBlock, self).__init__()

        self.in_blocks = nn.Sequential(
            *[InBlock(in_channels=in_channels if i==0 else out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding='same',
                      drop_rate=drop_rate,
                      dilation=dilation)
              for i in range(n_blocks-1)])
        
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )

        # last inner block
        self.conv = nn.Conv1d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1,
                              padding='same')
        self.norm = nn.BatchNorm1d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)
    
    def forward(self, x):
        skip = self.skip(x)

        # 1..n_blocks-1
        x = self.in_blocks(x)
        
        # last block
        x = self.conv(x)
        x = self.norm(x)
        
        # residual
        x += skip
        x = self.relu(x)
        x = self.dropout(x)

        return x

class Encoder(nn.Module):
    def __init__(self):

        super(Encoder, self).__init__()

        self.enc_stack = nn.Sequential(
            OutBlock(in_channels=256, out_channels=256, n_blocks=3, kernel_size=11, drop_rate=0.2),
            OutBlock(in_channels=256, out_channels=384, n_blocks=3, kernel_size=13, drop_rate=0.2),
            OutBlock(in_channels=384, out_channels=512, n_blocks=3, kernel_size=17, drop_rate=0.2),
            OutBlock(in_channels=512, out_channels=640, n_blocks=3, kernel_size=21, drop_rate=0.3),
            OutBlock(in_channels=640, out_channels=768, n_blocks=3, kernel_size=25, drop_rate=0.3)
        )
    
    def forward(self, x):
        x = self.enc_stack(x)
        return x

class Decoder(nn.Module):
    def __init__(self):

        super(Decoder, self).__init__()

        self.vocab_size = 29

        self.epilog = nn.Sequential(
            InBlock(in_channels=768,
                    out_channels=896,
                    kernel_size=29,
                    stride=1,
                    padding='same',
                    drop_rate=0.4,
                    dilation=2),
            InBlock(in_channels=896,
                    out_channels=1024,
                    kernel_size=1,
                    stride=1,
                    padding='same',
                    drop_rate=0.4),
            InBlock(in_channels=1024,
                    out_channels=self.vocab_size,
                    kernel_size=1,
                    stride=1,
                    padding='same',
                    drop_rate=0.)
        )
    
    def forward(self, x):
        x = self.epilog(x)
        return x