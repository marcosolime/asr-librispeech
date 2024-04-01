import torch
import torch.nn as nn
import torch.nn.functional as F

######### BUILDING BLOCKS #########
# - MyLayerNorm                   #
# - MySkipCNN                     #
# - MyBiGRU                       #
###################################

class MyLayerNorm(nn.Module):
    def __init__(self, n_bins):
        super(MyLayerNorm, self).__init__()
        self.my_layer_norm = nn.LayerNorm(n_bins)
    
    def forward(self, x):
        """
        Assume input has shape (batch, channels, n_bins, time).
        n_bins must be trailer dimension to apply layer norm.       
        """
        x = x.transpose(2, 3).contiguous()
        x = self.my_layer_norm(x)
        return x.transpose(2, 3).contiguous()

class MySkipCNN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, drop_rate, n_bins):
        super(MySkipCNN, self).__init__()

        self.conv2d_1 = nn.Conv2d(in_ch, out_ch, kernel, stride, padding=kernel//2)
        self.conv2d_2 = nn.Conv2d(out_ch, out_ch, kernel, stride, padding=kernel//2)
        self.drop_1 = nn.Dropout(drop_rate)
        self.drop_2 = nn.Dropout(drop_rate)
        self.my_layer_norm_1 = MyLayerNorm(n_bins)
        self.my_layer_norm_2 = MyLayerNorm(n_bins)
    
    def forward(self, x):
        """
        Assume input dimension has shape (batch, channels, n_bins, time)
        """
        skip = x

        # 1st stage
        x = self.my_layer_norm_1(x)
        x = F.gelu(x)
        x = self.drop_1(x)
        x = self.conv2d_1(x)

        # 2nd stage
        x = self.my_layer_norm_2(x)
        x = F.gelu(x)
        x = self.drop_2(x)
        x = self.conv2d_2(x)

        x += skip
        return x

class MyBiGRU(nn.Module):

    def __init__(self, input_size, hidden_size, drop_rate, batch_first):
        super(MyBiGRU, self).__init__()

        self.my_bigru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(input_size)
        self.drop = nn.Dropout(drop_rate)
    
    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.my_bigru(x)
        x = self.drop(x)
        return x


######### DEEP SPEECH 2 #########

class MyDeepSpeech(nn.Module):
    
    def __init__(self, n_conv2d, n_rnns, rnn_dim, n_class, n_bins, stride=2, drop_rate=0.1):
        super(MyDeepSpeech, self).__init__()
        n_bins = n_bins // 2

        # init conv2d layer
        self.init_cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=1)

        # skip cnns layers
        self.skip_cnn_layers = nn.Sequential(
            *[MySkipCNN(32, 32, kernel=3, stride=1, drop_rate=drop_rate, n_bins=n_bins)
            for _ in range(n_conv2d)]
        )

        # linear layer adapter
        self.fc_adapter = nn.Linear(n_bins*32, rnn_dim)

        # bi-rnn layers
        self.birnn_layers = nn.Sequential(
            *[MyBiGRU(input_size=rnn_dim if l==0 else rnn_dim*2, hidden_size=rnn_dim, drop_rate=drop_rate, batch_first=(l==0))
            for l in range(n_rnns)]
        )

        # linear classifier
        self.fc_classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(rnn_dim, n_class)
        )
    
    def forward(self, x):
        x = self.init_cnn(x)

        # 1st stage
        x = self.skip_cnn_layers(x)

        # linear adapter
        shape = x.shape
        x = x.view(shape[0], shape[1]*shape[2], shape[3]) # (batch, freq, time)
        x = x.transpose(1, 2) # (batch, time, freq)
        x = self.fc_adapter(x)

        # 2nd stage
        x = self.birnn_layers(x)

        # linear classifier
        x = self.fc_classifier(x)

        return x