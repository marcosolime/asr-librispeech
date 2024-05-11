import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConformerMultiheadSelfAttentionModule(nn.Module):
    def __init__(self, emb_dim, num_heads, max_rel_pos, device):

        super(ConformerMultiheadSelfAttentionModule, self).__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.max_rel_pos=max_rel_pos

        self.norm = nn.LayerNorm(emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.dropout = nn.Dropout()
        
        self.device = device
        self.pos_matrix = self.get_positional_matrix().to(self.device)


    def get_positional_matrix(self):
        """
        Create positional matrix of shape (2*self.max_rel_pos + 1, emb_dim)
        Only (:seq_len, emb_dim) will be summed to input tensors
        """
        matrix = torch.zeros(2*self.max_rel_pos + 1, self.emb_dim)

        pos = torch.arange(0, 2*self.max_rel_pos + 1).unsqueeze(1).float()
        divisor = torch.exp(torch.arange(0, self.emb_dim, 2).float() * -math.log(10000) / self.emb_dim)
        
        matrix[:, 0::2] = torch.sin(pos*divisor)
        matrix[:, 1::2] = torch.cos(pos*divisor)
        final_matrix = matrix.unsqueeze(0)

        return final_matrix

    def forward(self, x):
        # Input shape: [batch, seq_len, in_features]
        batch_size, seq_len, _ = x.size()

        skip = x

        x = self.norm(x)
        pos_emb = self.pos_matrix[:, :seq_len, :].expand(batch_size, seq_len, self.emb_dim)
        x += pos_emb
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)

        x += skip

        return x

class ConformerConvolutionModule(nn.Module):
    def __init__(self, num_features, exp_factor=2):

        super(ConformerConvolutionModule, self).__init__()

        self.layer_norm = nn.LayerNorm(num_features)
        self.conv_1 = nn.Conv1d(in_channels=num_features,
                                out_channels=num_features*exp_factor,
                                kernel_size=1)
        self.glu = nn.GLU(1)
        self.depth_conv = nn.Conv1d(in_channels=num_features,
                                    out_channels=num_features,
                                    kernel_size=1,
                                    groups=num_features)
        
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.swish = nn.SiLU()
        self.conv_2 = nn.Conv1d(in_channels=num_features,
                                out_channels=num_features,
                                kernel_size=1)
        self.dropout = nn.Dropout()

    def forward(self, x):

        skip = x # [batch_size, seq_len, num_features] 
        x = self.layer_norm(x)

        x = x.transpose(1, 2).contiguous() # [batch_size, num_features, seq_len] 

        x = self.conv_1(x)
        x = self.glu(x)
        x = self.depth_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.conv_2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2).contiguous() # [batch_size, seq_len, num_features]
        x += skip

        return x

class ConformerFeedForwardModule(nn.Module):

    def __init__(self, in_features, exp_factor=4):

        super(ConformerFeedForwardModule, self).__init__()

        self.norm = nn.LayerNorm(in_features)

        self.linear_1 = nn.Linear(in_features, in_features*exp_factor)
        self.swish = nn.SiLU()
        self.dropout_1 = nn.Dropout()

        self.linear_2 = nn.Linear(in_features*exp_factor, in_features)
        self.dropout_2 = nn.Dropout()
    
    def forward(self, x):
        
        skip = x # [batch_size, seq_len, in_features]

        x = self.norm(x)
        
        x = self.linear_1(x) # [batch_size, seq_len, in_features * exp_factor]
        x = self.swish(x)
        x = self.dropout_1(x)

        x = self.linear_2(x) # [batch_size, seq_len, in_features]
        x = self.dropout_2(x)

        x = skip + 1/2 * x

        return x

"""
Preprocessing - the pipeline before the Conformer stack
"""
class ConvSubsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        
        super(ConvSubsampling, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, 
                              stride=stride,
                              padding=kernel_size//stride)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Prolog(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride):

        super(Prolog, self).__init__()

        self.conv_sub = ConvSubsampling(in_channels=in_features,
                                        out_channels=in_features,
                                        kernel_size=kernel_size,
                                        stride=stride)
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.dropout = nn.Dropout()
    
    def forward(self, x):
        x = self.conv_sub(x)

        x = x.transpose(1, 2).contiguous()
        x = self.linear(x)
        x = x.transpose(1, 2).contiguous()

        x = self.dropout(x)
        return x