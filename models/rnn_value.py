import torch.nn as nn
import torch
from utils.math import *


feature_dim = 32


class RnnValueNet(nn.Module):
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
        out = out.view(b, s, 1, -1)

        value = self.value_head(out)

        return value



