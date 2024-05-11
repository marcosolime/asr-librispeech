import torch
import torch.nn as nn
from models.conformer.blocks import Prolog, ConformerMultiheadSelfAttentionModule, ConformerConvolutionModule, ConformerFeedForwardModule, ConvSubsampling

class ConformerBlock(nn.Module):
    def __init__(self, in_features, num_heads, max_rel_pos):
        super(ConformerBlock, self).__init__()
        self.feed_forward_1 = ConformerFeedForwardModule(in_features=in_features)
        self.attention = ConformerMultiheadSelfAttentionModule(emb_dim=in_features,
                                                               num_heads=num_heads, 
                                                               max_rel_pos=max_rel_pos)
        self.convolution = ConformerConvolutionModule(num_features=in_features)
        self.feed_forward_2 = ConformerFeedForwardModule(in_features=in_features)
        self.norm = nn.LayerNorm(normalized_shape=in_features)
    
    def forward(self, x):
        x = self.feed_forward_1(x)
        x = self.attention(x)
        x = self.convolution(x)
        x = self.feed_forward_2(x)
        x = self.norm(x)

        return x

class Conformer(nn.Module):
    def __init__(self,
                 in_features, 
                 encoder_dim,
                 hidden_size=320,
                 kernel_size=11,
                 stride=2,
                 num_heads=4,
                 max_rel_pos=400,
                 n_class=29,
                 n_blocks=4):

        super(Conformer, self).__init__()

        self.prolog = Prolog(in_features=in_features,
                             out_features=encoder_dim,
                             kernel_size=kernel_size,
                             stride=stride)
        self.conformer_stack = nn.Sequential(
            *[ConformerBlock(in_features=encoder_dim,
                             num_heads=num_heads,
                             max_rel_pos=max_rel_pos)
              for _ in range(n_blocks)])
        self.lstm = nn.LSTM(input_size=encoder_dim,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True)
        self.scoring = nn.Linear(in_features=hidden_size,
                                 out_features=n_class)

    def forward(self,x):
        x = self.prolog(x)
        x = x.transpose(1, 2).contiguous() # [batch_size, seq_len, num_features]

        x = self.conformer_stack(x)
        x, _ = self.lstm(x)
        x = self.scoring(x)

        return x
