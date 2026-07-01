import torch
import torch.nn as nn
import numpy as np

class CharLSTM(nn.Module):
    def __init__(self, n_chars, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # Mapping from index to character and vice-versa
        self.chars = None
        self.int2char = None
        self.char2int = None
        
        # LSTM layer
        self.lstm = nn.LSTM(n_chars, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # Linear layer
        self.fc = nn.Linear(n_hidden, n_chars)

    def forward(self, x, hidden):
        # New hidden state is produced by LSTM
        r_output, hidden = self.lstm(x, hidden)
        
        # Pass output through dropout
        out = self.dropout(r_output)
        
        # Flatten for linear layer: (batch_size * seq_length, n_hidden)
        out = out.contiguous().view(-1, self.n_hidden)
        
        # Final output (batch_size * seq_length, n_chars)
        out = self.fc(out)
        
        return out, hidden

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes (n_layers, batch_size, n_hidden)
        # initialized to zero, for hidden and cell states of LSTM
        weight = next(self.parameters()).data
        
        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().zero_())
        
        return hidden
