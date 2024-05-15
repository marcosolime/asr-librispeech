import torch
import torch.nn as nn
from models.conformer.blocks import Prolog, Epilog, ConformerMHSA, ConformerConv, ConformerFFN, ConvSubsampling

class ConformerBlock(nn.Module):
    def __init__(self, encoder_dim, num_heads, kernel_size, device):
        super(ConformerBlock, self).__init__()
        self.feed_forward_1 = ConformerFFN(num_features=encoder_dim)
        self.attention = ConformerMHSA(num_features=encoder_dim, device=device, num_heads=num_heads)
        self.convolution = ConformerConv(num_features=encoder_dim, kernel_size=kernel_size)
        self.feed_forward_2 = ConformerFFN(num_features=encoder_dim)
        self.norm = nn.LayerNorm(normalized_shape=encoder_dim)
    
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
                 num_heads,
                 kernel_size,
                 hidden_size,
                 n_class,
                 n_blocks,
                 device):

        super(Conformer, self).__init__()

        self.prolog = Prolog(in_features=in_features, encoder_dim=encoder_dim)
        self.conformer_stack = nn.Sequential(
            *[ConformerBlock(encoder_dim=encoder_dim,
                             num_heads=num_heads,
                             kernel_size=kernel_size,
                             device=device)
              for _ in range(n_blocks)])
        self.epilog = Epilog(encoder_dim=encoder_dim, hidden_size=hidden_size, n_class=n_class)

    def forward(self, x):
        """
        Input:  [batch_size, 1, seq_len, num_features]
        Output: [batch_size, seq_len, n_classes]
        """
        x = x.transpose(2, 3).contiguous()

        x = self.prolog(x)
        x = self.conformer_stack(x)
        x = self.epilog(x)

        return x
