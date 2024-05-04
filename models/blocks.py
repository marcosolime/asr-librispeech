import torch
import torch.nn as nn
import torch.nn.functional as F

""" BUILDING BLOCKS

    - LayerNorm
    - ResNet
    - ResNetInv
    - BiGRU
    - PrenormEncoder
        MHSA
        FFN
    - Classifier

"""

class LayerNorm(nn.Module):
    """
    Custom layer norm.
    We want to normalize on the FIXED feature dimension,
    as the time dimension can vary
    """
    def __init__(self, n_features):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_features)
    
    def forward(self, x):
        """
        Input shape: [batch, channels, n_features, time]      
        """
        x = x.transpose(2, 3).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, drop_rate):
        super(ResNet, self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=1)
        self.cnn_2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=1)
        
        self.dropout = nn.Dropout(drop_rate)

        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Input shape: [batch, channels, n_features, time]
        """
        skip = x

        x = self.cnn_1(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.cnn_2(x)
        x = self.batch_norm_2(x)

        x += skip
        x = F.relu(x)
        return x

class ResNetInv(nn.Module):
    def __init__(self, in_channels, out_channels, n_features, kernel, stride, drop_rate):
        super(ResNetInv, self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=1)
        self.cnn_2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=1)
        
        self.dropout_1 = nn.Dropout(drop_rate)
        self.dropout_2 = nn.Dropout(drop_rate)
        
        self.layer_norm_1 = LayerNorm(n_features)
        self.layer_norm_2 = LayerNorm(n_features)

    
    def forward(self, x):
        """
        Input shape: [batch, channels, n_features, time]
        """
        skip = x

        # 1st stage
        x = self.layer_norm_1(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        x = self.cnn_1(x)

        # 2nd stage
        x = self.layer_norm_2(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        x = self.cnn_2(x)
    
        x += skip
        return x

class BiGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, drop_rate, batch_first):
        super(BiGRU, self).__init__()

        self.gru = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):
        """
        Input shape: [batch, seq_len, n_features]
        """
        x = self.layer_norm(x)
        x = F.relu(x)
        x, _ = self.gru(x)
        x = self.dropout(x)
        return x

class MHSA(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(MHSA, self).__init__()
        self.attention = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        skip = x
        x = self.layer_norm(x)
        x, _ = self.attention(x, x, x)
        x += skip
        return x

class FFN(nn.Module):
    def __init__(self, emb_dim, drop_rate):
        super(FFN, self).__init__()
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.linear_1 = nn.Linear(emb_dim, emb_dim)
        self.linear_2 = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):
        skip = x
        x = self.layer_norm(x)
        
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.linear_2(x)
        x = F.relu(x)

        x += skip
        return x

class PrenormEncoder(nn.Module):
    def __init__(self, emb_dim, n_heads, drop_rate):
        super(PrenormEncoder, self).__init__()
        self.mhsa_block = MHSA(emb_dim, n_heads)
        self.ffn_block = FFN(emb_dim, drop_rate)

    def forward(self, x):
        x = self.mhsa_block(x)
        x = self.ffn_block(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_rate):
        super(Classifier, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x