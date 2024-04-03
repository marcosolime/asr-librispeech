import torch
import torch.nn as nn
import torch.nn.functional as F

######### BUILDING BLOCKS #########
# - CNNLayerNorm                  #
# - ResidualCNN                   #
# - BiGRU                         #
###################################

class CNNLayerNorm(nn.Module):
    def __init__(self, n_features):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_features)
    
    def forward(self, x):
        """
        Input shape: [batch, channels, n_features, time]      
        """
        x = x.transpose(2, 3).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()

class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, drop_rate, n_features):
        super(ResidualCNN, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.dropout = nn.Dropout(drop_rate)
        self.cnn_layer_norm = CNNLayerNorm(n_features)
    
    def forward(self, x):
        """
        Input shape: [batch, channels, n_features, time]
        """
        skip = x

        x = self.conv2d(x)
        x = self.cnn_layer_norm(x)
        x = F.relu(x)
        x = self.dropout(x)

        x += skip
        return x

class BiGRU(nn.Module):

    def __init__(self, input_size, hidden_size, drop_rate, batch_first):
        super(BiGRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size*2)
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):
        """
        Input shape: [batch, seq_len, n_features]
        """
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        skip = x
        x = self.layer_norm(x)
        x, _ = self.attention(x, x, x)
        x += skip
        return x

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_rate):
        super(FeedForwardNetwork, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
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
    def __init__(self, input_dim, hidden_dim, num_heads, drop_rate):
        super(PrenormEncoder, self).__init__()
        self.attention_block = MultiheadAttention(input_dim, num_heads)
        self.ffn_block = FeedForwardNetwork(input_dim, hidden_dim, drop_rate)

    def forward(self, x):
        x = self.attention_block(x)
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