import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embed_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.rnn = nn.LSTM(embed_dim, hid_dim, num_layers=n_layers, dropout=dropout)

    def forward(self, x):

        outputs, (hidden, cell) = self.rnn(x)
        # outputs = [sen_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]
        return hidden, cell
