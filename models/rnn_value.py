import torch.nn as nn
import torch
from utils.math import *



class RnnValueNet(nn.Module):
    def __init__(self, activation='tanh', hidden_n=64, hidden_l=2):
        super().__init__()

        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.rnn_feature_dim = hidden_n
        self.linear_dim = 16

        self.rnn = nn.RNN(input_size=2, 
                           hidden_size=self.rnn_feature_dim,
                           num_layers=hidden_l,
                           nonlinearity='tanh',
                           dropout=0.2,
                           batch_first=True,
                           bidirectional=False)

        self.value_head = nn.Sequential(
            nn.Linear(self.rnn_feature_dim, 1), 
            # nn.Tanh(),
            # nn.Linear(self.linear_dim, self.linear_dim),
            # nn.Tanh(),
            # nn.Linear(self.linear_dim, 1),
            )
        

    def forward(self, batch_seq):

        lengths = [seq.size(0) for seq in batch_seq]   # get the length of each sequence in the batch
        seq_padded = nn.utils.rnn.pad_sequence(batch_seq, batch_first=True)  # padd all sequences
        b, s, n = seq_padded.shape
        
        # pack padded sequece
        seq_padded = nn.utils.rnn.pack_padded_sequence(
            seq_padded, lengths=lengths, batch_first=True, enforce_sorted=False)
        
        # calc the feature vector from the latent space 
        out, _ = self.rnn(seq_padded)
        
        # unpack the featrue vector
        out, lens_unpacked = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = torch.stack([item[lens_unpacked[idx]-1] 
                           for idx, item in enumerate(out)])

        value = self.value_head(out)
        value = torch.tanh(value)

        return value



