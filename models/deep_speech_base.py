import torch
import torch.nn as nn
from models.blocks import LayerNorm, ResNet, ResNetInv, BiGRU, Classifier

"""
Deep Speech Base
- 1st stage: stack of ResNet blocks
- 2nd stage: stack of recurrent networks (BiGRU)
- classifier: fully connected layer (2 layers)
"""

class DeepSpeechBase(nn.Module):
    
    def __init__(self, n_cnn, n_rnn, rnn_dim, n_features, n_class, stride_time=2, stride_freq=2, drop_rate=0.2):
        super(DeepSpeechBase, self).__init__()
        n_features = n_features // stride_freq

        # init conv2d layer
        self.init_cnn = nn.Conv2d(1, 32, 3, stride=stride_time, padding=1)

        # 1st stage: ResNet blocks
        self.stage_1 = nn.Sequential(
            *[ResNetInv(32, 32, n_features=n_features, kernel=3, stride=1, drop_rate=drop_rate) 
              for _ in range(n_cnn)])

        # linear adapter
        self.fc_adapter = nn.Linear(n_features*32, rnn_dim)

        # bi-rnn layers
        self.stage_2 = nn.Sequential(
            *[BiGRU(rnn_dim=rnn_dim if layer==0 else rnn_dim*2, 
                    hidden_size=rnn_dim, 
                    drop_rate=drop_rate, 
                    batch_first=True)
            for layer in range(n_rnn)])

        # linear classifier
        self.classifier = Classifier(rnn_dim*2, rnn_dim, n_class, drop_rate)

    def forward(self, x):
        x = self.init_cnn(x)

        # 1st stage
        x = self.stage_1(x)

        # linear adapter
        shape = x.shape
        x = x.view(shape[0], shape[1]*shape[2], shape[3]) # (batch, features, time)
        x = x.transpose(1, 2) # (batch, time, features)
        x = self.fc_adapter(x)

        # 2nd stage
        x = self.stage_2(x)

        # linear classifier
        x = self.classifier(x)

        return x
