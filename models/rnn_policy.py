import torch.nn as nn
import torch
from utils.math import *



class RnnPolicyNet(nn.Module):
    def __init__(self, action_dim, activation='tanh', log_std=0):
        super().__init__()

        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.rnn_feature_dim = 32
        self.linear_dim = 16

        self.rnn = nn.RNN(input_size=2, 
                           hidden_size=self.rnn_feature_dim,
                           num_layers=2,
                           nonlinearity='tanh',
                           dropout=0.4,
                           batch_first=True,
                           bidirectional=False)

        self.action_mean = nn.Sequential(
            nn.Linear(self.rnn_feature_dim, self.linear_dim), 
            nn.Tanh(),
            nn.Linear(self.linear_dim, self.linear_dim),
            nn.Tanh(),
            nn.Linear(self.linear_dim, action_dim),
            )
        
        # self.action_mean.weight.data.mul_(0.1)
        # self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, batch_seq):

        if type(batch_seq) is list:

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
            out = torch.stack([item[lens_unpacked[idx]-1] 
                           for idx, item in enumerate(out)])

        else:
            out, hidden = self.rnn(batch_seq)
            out = out[0][-1].view(1, 1, 1, -1)

        action_mean = self.action_mean(out)
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



