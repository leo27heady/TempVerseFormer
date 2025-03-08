import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, embed_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.rnn = nn.LSTM(embed_dim, hid_dim, num_layers=n_layers, dropout=dropout)
        
    def forward(self, x, hidden, cell):
        
        # x = [batch_size, embed_dim]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]
        
        x = x.unsqueeze(0)  # [1, batch_size, embed_dim]
        
        output, (hidden, cell) = self.rnn(x, (hidden, cell))
        output = output.squeeze(0)  # seq_len and n_dir will always be 1 in the decoder
        # output = [batch_size, hid_dim * n_dir]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]
        
        return output, hidden, cell
