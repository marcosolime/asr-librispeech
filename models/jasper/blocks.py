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

class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(SkipBlock, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels,out_channels=out_channels,
                              kernel_size=1,stride=1,padding='same')
        self.norm = nn.BatchNorm1d(num_features=out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class EncoderDR(nn.Module):
    def __init__(self):

        super(EncoderDR, self).__init__()

        # residual blocks
        self.skip_0_1 = SkipBlock(in_channels=256, out_channels=256)
        self.skip_0_2 = SkipBlock(in_channels=256, out_channels=384)
        self.skip_0_3 = SkipBlock(in_channels=256, out_channels=512)
        self.skip_0_4 = SkipBlock(in_channels=256, out_channels=640)
        self.skip_0_5 = SkipBlock(in_channels=256, out_channels=768)

        self.skip_1_2 = SkipBlock(in_channels=256, out_channels=384)
        self.skip_1_3 = SkipBlock(in_channels=256, out_channels=512)
        self.skip_1_4 = SkipBlock(in_channels=256, out_channels=640)
        self.skip_1_5 = SkipBlock(in_channels=256, out_channels=768)

        self.skip_2_3 = SkipBlock(in_channels=384, out_channels=512)
        self.skip_2_4 = SkipBlock(in_channels=384, out_channels=640)
        self.skip_2_5 = SkipBlock(in_channels=384, out_channels=768)

        self.skip_3_4 = SkipBlock(in_channels=512, out_channels=640)
        self.skip_3_5 = SkipBlock(in_channels=512, out_channels=768)

        self.skip_4_5 = SkipBlock(in_channels=640, out_channels=768)

        # 1st block
        self.block_1_1 = InBlock(in_channels=256,out_channels=256,kernel_size=11,
                                 stride=1,padding='same',drop_rate=0.2)
        self.block_1_2 = InBlock(in_channels=256,out_channels=256,kernel_size=11,
                                 stride=1,padding='same',drop_rate=0.2)
        self.conv_1_3 = nn.Conv1d(in_channels=256,out_channels=256,kernel_size=11,
                                  stride=1,padding='same')
        self.norm_1_3 = nn.BatchNorm1d(num_features=256)
        self.relu_1_3 = nn.ReLU()
        self.dropout_1_3 = nn.Dropout(p=0.2)
    
        # 2nd block
        self.block_2_1 = InBlock(in_channels=256,out_channels=384,kernel_size=13,
                                 stride=1,padding='same',drop_rate=0.2)
        self.block_2_2 = InBlock(in_channels=384,out_channels=384,kernel_size=13,
                                 stride=1,padding='same',drop_rate=0.2)
        self.conv_2_3 = nn.Conv1d(in_channels=384,out_channels=384,kernel_size=13,
                                  stride=1,padding='same')
        self.norm_2_3 = nn.BatchNorm1d(num_features=384)
        self.relu_2_3 = nn.ReLU()
        self.dropout_2_3 = nn.Dropout(p=0.2)

        # 3rd block
        self.block_3_1 = InBlock(in_channels=384,out_channels=512,kernel_size=17,
                                 stride=1,padding='same',drop_rate=0.2)
        self.block_3_2 = InBlock(in_channels=512,out_channels=512,kernel_size=17,
                                 stride=1,padding='same',drop_rate=0.2)
        self.conv_3_3 = nn.Conv1d(in_channels=512,out_channels=512,kernel_size=17,
                                  stride=1,padding='same')
        self.norm_3_3 = nn.BatchNorm1d(num_features=512)
        self.relu_3_3 = nn.ReLU()
        self.dropout_3_3 = nn.Dropout(p=0.2)

        # 4th block
        self.block_4_1 = InBlock(in_channels=512,out_channels=640,kernel_size=21,
                                 stride=1,padding='same',drop_rate=0.3)
        self.block_4_2 = InBlock(in_channels=640,out_channels=640,kernel_size=21,
                                 stride=1,padding='same',drop_rate=0.3)
        self.conv_4_3 = nn.Conv1d(in_channels=640,out_channels=640,kernel_size=21,
                                  stride=1,padding='same')
        self.norm_4_3 = nn.BatchNorm1d(num_features=640)
        self.relu_4_3 = nn.ReLU()
        self.dropout_4_3 = nn.Dropout(p=0.3)

        # 5th block
        self.block_5_1 = InBlock(in_channels=640,out_channels=768,kernel_size=25,
                                 stride=1,padding='same',drop_rate=0.3)
        self.block_5_2 = InBlock(in_channels=768,out_channels=768,kernel_size=25,
                                 stride=1,padding='same',drop_rate=0.3)
        self.conv_5_3 = nn.Conv1d(in_channels=768,out_channels=768,kernel_size=25,
                                  stride=1,padding='same')
        self.norm_5_3 = nn.BatchNorm1d(num_features=768)
        self.relu_5_3 = nn.ReLU()
        self.dropout_5_3 = nn.Dropout(p=0.3)

    def forward(self, x):
        skip_0 = x

        # 1st block
        x = self.block_1_1(x)
        x = self.block_1_2(x)
        x = self.conv_1_3(x)
        x = self.norm_1_3(x)

        x += self.skip_0_1(skip_0)
        x = self.relu_1_3(x)
        x = self.dropout_1_3(x)

        skip_1 = x

        # 2nd block
        x = self.block_2_1(x)
        x = self.block_2_2(x)
        x = self.conv_2_3(x)
        x = self.norm_2_3(x)

        x += self.skip_0_2(skip_0) + self.skip_1_2(skip_1)
        x = self.relu_2_3(x)
        x = self.dropout_2_3(x)

        skip_2 = x

        # 3rd block
        x = self.block_3_1(x)
        x = self.block_3_2(x)
        x = self.conv_3_3(x)
        x = self.norm_3_3(x)

        x += self.skip_0_3(skip_0) + self.skip_1_3(skip_1) + self.skip_2_3(skip_2)
        x = self.relu_3_3(x)
        x = self.dropout_3_3(x)

        skip_3 = x

        # 4th block
        x = self.block_4_1(x)
        x = self.block_4_2(x)
        x = self.conv_4_3(x)
        x = self.norm_4_3(x)

        x += self.skip_0_4(skip_0) + self.skip_1_4(skip_1) + self.skip_2_4(skip_2) + self.skip_3_4(skip_3)
        x = self.relu_4_3(x)
        x = self.dropout_4_3(x)

        skip_4 = x

        # 5th block
        x = self.block_5_1(x)
        x = self.block_5_2(x)
        x = self.conv_5_3(x)
        x = self.norm_5_3(x)

        x += self.skip_0_5(skip_0) + self.skip_1_5(skip_1) + self.skip_2_5(skip_2) + self.skip_3_5(skip_3) + self.skip_4_5(skip_4)
        x = self.relu_5_3(x)
        x = self.dropout_5_3(x)

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
            nn.Conv1d(in_channels=1024,
                      out_channels=self.vocab_size,
                      kernel_size=1)
        )
    
    def forward(self, x):
        x = self.epilog(x)
        return x