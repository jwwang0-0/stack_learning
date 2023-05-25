import torch.nn as nn
import torch
from utils.math import *


feature_dim = 32

class CnnLabelPolicy(nn.Module):
    def __init__(self, action_dim, activation='sigmoid', log_std=0):
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

        self.action_mean = nn.Sequential(
            nn.Linear(2*14*14, feature_dim), 
            nn.Tanh(),
            nn.Linear(feature_dim, action_dim)
            )
        
        # self.action_mean.weight.data.mul_(0.1)
        # self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, arr_img):

        x = self.cnn(arr_img)
        x = x.view(-1, 2*14*14)
        action_mean = self.action_mean(x)
        action_mean = torch.tanh(action_mean)

        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)



