import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConformerMHSA(nn.Module):
    def __init__(self, num_features, device, num_heads, max_rel_pos=800, drop_rate=0.1):

        super(ConformerMHSA, self).__init__()

        self.emb_dim = num_features
        self.num_heads = num_heads
        self.max_rel_pos = max_rel_pos

        self.norm = nn.LayerNorm(num_features)
        self.attention = nn.MultiheadAttention(num_features, num_heads, batch_first=True)
        self.dropout = nn.Dropout(p=drop_rate)
        
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
        # Input shape: [batch, seq_len, num_features]
        batch_size, seq_len, _ = x.size()

        skip = x

        x = self.norm(x)
        pos_emb = self.pos_matrix[:, :seq_len, :].expand(batch_size, seq_len, self.emb_dim)
        x += pos_emb
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)

        x += skip

        return x

class ConformerConv(nn.Module):
    def __init__(self, num_features, kernel_size, exp_factor=2, drop_rate=0.1):

        super(ConformerConv, self).__init__()

        self.layer_norm = nn.LayerNorm(num_features)
        self.point_conv_1 = nn.Conv1d(in_channels=num_features,
                                      out_channels=num_features*exp_factor,
                                      kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depth_conv = nn.Conv1d(in_channels=num_features,
                                    out_channels=num_features,
                                    kernel_size=kernel_size,
                                    padding=(kernel_size-1)//2,
                                    groups=num_features)
        
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.swish = nn.SiLU()
        self.point_conv_2 = nn.Conv1d(in_channels=num_features,
                                      out_channels=num_features,
                                      kernel_size=1)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):

        skip = x # [batch_size, seq_len, num_features] 
        x = self.layer_norm(x)

        x = x.transpose(1, 2).contiguous() # [batch_size, num_features, seq_len] 

        x = self.point_conv_1(x)
        x = self.glu(x)
        x = self.depth_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.point_conv_2(x)
        x = self.dropout(x)

        x = x.transpose(1, 2).contiguous() # [batch_size, seq_len, num_features]
        x += skip

        return x

class ConformerFFN(nn.Module):

    def __init__(self, num_features, exp_factor=4, drop_rate=0.1):

        super(ConformerFFN, self).__init__()

        self.norm = nn.LayerNorm(num_features)

        self.linear_1 = nn.Linear(num_features, num_features*exp_factor)
        self.swish = nn.SiLU()
        self.dropout_1 = nn.Dropout(p=drop_rate)

        self.linear_2 = nn.Linear(num_features*exp_factor, num_features)
        self.dropout_2 = nn.Dropout(p=drop_rate)
    
    def forward(self, x):
        
        skip = x # [batch_size, seq_len, num_features]

        x = self.norm(x)
        
        x = self.linear_1(x) # [batch_size, seq_len, num_features * exp_factor]
        x = self.swish(x)
        x = self.dropout_1(x)

        x = self.linear_2(x) # [batch_size, seq_len, num_features]
        x = self.dropout_2(x)

        x = skip + 1/2 * x

        return x

class Epilog(nn.Module):

    def __init__(self, encoder_dim, hidden_size, n_class):

        super(Epilog, self).__init__()

        self.lstm = nn.LSTM(input_size=encoder_dim,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True)
        self.scoring = nn.Linear(in_features=hidden_size, out_features=n_class)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.scoring(x)
        return x

"""
Preprocessing - the pipeline before the Conformer stack
"""
class ConvSubsampling(nn.Module):

    def __init__(self, out_channels):
        
        super(ConvSubsampling, self).__init__()
    
        self.sub_stack = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=out_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.sub_stack(x)

        batch_size, channels, seq_len, num_features = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, channels*num_features)

        return x

class Prolog(nn.Module):
    def __init__(self, in_features, encoder_dim, drop_rate=0.1):

        super(Prolog, self).__init__()

        self.out_features = self.get_out_features(in_features)

        self.conv_sub = ConvSubsampling(out_channels=in_features)
        self.linear = nn.Linear(in_features=self.out_features, out_features=encoder_dim)
        self.dropout = nn.Dropout(p=drop_rate)
    
    def get_out_features(self, in_features):
        ans = in_features * in_features // 2
        return ans

    def forward(self, x):
        x = self.conv_sub(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x