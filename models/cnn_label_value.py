import torch.nn as nn
import torch
from utils.math import *


feature_dim = 32

class CnnLabelValue(nn.Module):
    def __init__(self, activation='sigmoid'):
        super().__init__()

        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.cnn = nn.Sequential(

            nn.Tanh(),
            nn.MaxPool2d(10, stride=10),

            nn.Conv2d(1, 8, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(8, 2, kernel_size=3, stride=1),

        )

        self.value_head = nn.Sequential(
            nn.Linear(2*14*14, feature_dim), 
            nn.Tanh(),
            nn.Linear(feature_dim, 1)
            )
        
        # self.value_head.weight.data.mul_(0.1)
        # self.value_head.bias.data.mul_(0.0)


    def forward(self, arr_img):

        x = self.cnn(arr_img)
        x = x.view(-1, 2*14*14)
        value = self.value_head(x)
        value = torch.tanh(value)
        
        return value



