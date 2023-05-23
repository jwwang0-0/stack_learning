import torch.nn as nn
import torch
from utils.math import *


feature_dim = 32

class DeepIO(nn.Module):
    def __init__(self):
        super(DeepIO, self).__init__()
        self.rnn = nn.LSTM(input_size=6, hidden_size=512,
                           num_layers=2, bidirectional=True)
        self.drop_out = nn.Dropout(0.25)
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc_out = nn.Linear(256, 7)

    def forward(self, x):
        """
        args:
        x:  a list of inputs of diemension [BxTx6]
        """
        lengths = [x_.size(0) for x_ in x]   # get the length of each sequence in the batch
        x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True)  # padd all sequences
        b, s, n = x_padded.shape
        
        # pack padded sequece
        x_padded = nn.utils.rnn.pack_padded_sequence(x_padded, lengths=lengths, batch_first=True, enforce_sorted=False)
        
        # calc the feature vector from the latent space 
        out, hidden = self.rnn(x_padded)
        breakpoint()
        
        # unpack the featrue vector
        out, lens_unpacked = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out.view(b, s, self.num_dir, self.hidden_size[0])

        # many-to-one rnn, get the last result
        y = out[:, -1, 0]

        y = F.relu(self.fc1(y), inplace=True)
        y = self.bn1(y)
        y = self.drop_out(y)

        y = self.out(y)
        return y

class RnnAC(nn.Module):
    def __init__(self, action_dim, activation='tanh', log_std=0):
        super().__init__()

        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.rnn_feature_dim = 8
        self.linear_dim = 4

        self.rnn = nn.LSTM(input_size=8, 
                           hidden_size=self.rnn_feature_dim,
                           num_layers=2,
                           batch_first=True,
                           bidirectional=False)

        self.linear = nn.Sequential(
            nn.Linear(self.rnn_feature_dim, self.linear_dim), 
            nn.Tanh()
            )
        
        self.value_head = nn.Linear(self.linear_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        self.action_mean = nn.Linear(self.linear_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, batch_seq):

        lengths = [seq.size(0) for seq in batch_seq]   # get the length of each sequence in the batch
        seq_padded = nn.utils.rnn.pad_sequence(batch_seq, batch_first=True)  # padd all sequences
        b, s, n = seq_padded.shape
        
        # pack padded sequece
        seq_padded = nn.utils.rnn.pack_padded_sequence(
            seq_padded, lengths=lengths, batch_first=True, enforce_sorted=False)
        
        # calc the feature vector from the latent space 
        out, hidden = self.rnn(seq_padded)
        
        # unpack the featrue vector
        out, lens_unpacked = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = out.view(b, s, self.num_dir, self.hidden_size[0])

        # many-to-one rnn, get the last result
        y = out[:, -1, 0]



        y = self.out(y)
        # return y

        action_mean = self.action_mean(batch_seq)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        value = self.value_head(batch_seq)

        return value, action_mean, action_log_std, action_std

    def select_action(self, x):
        _, action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action

    def get_log_prob(self, x, actions):
        _, action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)



