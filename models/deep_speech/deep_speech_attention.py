import torch
import torch.nn as nn
from models.deep_speech.blocks import ResNetInv, PrenormEncoder, Classifier, MHSA, FFN

"""
Deep Speech Attention
- 1st stage: stack of ResNet blocks
- 2nd stage: stack of Prenorm encoders (MHSA + FFN)
- classifier: fully connected layer (2 layers) 
"""

class DeepSpeechAttention(nn.Module):

    def __init__(self, n_cnn, n_enc, n_features, n_class, 
                 emb_dim=512, n_heads=4, stride=2, drop_rate=0.2):
        super(DeepSpeechAttention, self).__init__()
        n_features = n_features // 2

        # init conv2d layer
        self.init_cnn = nn.Conv2d(1, 32, 3, stride=(2, stride), padding=1)

        # 1st stage: ResNet blocks
        self.stage_1 = nn.Sequential(
            *[ResNetInv(32, 32, n_features=n_features, kernel=3, stride=1, drop_rate=drop_rate) 
              for _ in range(n_cnn)])
        
        # linear adapter
        self.fc_adapter = nn.Linear(n_features*32, emb_dim)

        # 2nd stage: Attention blocks
        self.stage_2 = nn.Sequential(
            *[PrenormEncoder(emb_dim=emb_dim,
                             n_heads=n_heads,
                             drop_rate=drop_rate)
                             for _ in range(n_enc)])
        
        # linear classifier
        self.classifier = Classifier(emb_dim, emb_dim, n_class, drop_rate)

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