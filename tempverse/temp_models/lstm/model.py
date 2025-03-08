import torch
import torch.nn as nn
from einops import rearrange

from .encoder import Encoder
from .decoder import Decoder
from ..config import LSTM_Config


class Seq2SeqLSTM(nn.Module):
    def __init__(self, config: LSTM_Config):
        super().__init__()

        self.input_projection = nn.Linear(config.input_dim, config.embed_dim, bias=True)
        self.encoder = Encoder(
            config.embed_dim, config.embed_dim, config.n_layers, config.enc_dropout
        )
        self.decoder = Decoder(
            config.embed_dim, config.embed_dim, config.n_layers, config.dec_dropout
        )

        self.output_norm = nn.LayerNorm(config.embed_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(config.embed_dim, config.input_dim, bias=True)
        )
        
    def forward(self, x, t):
        
        x = rearrange(
            x, 'b n c (nh ph) (nw pw) -> (n nh nw) b (ph pw c)',
            ph=1,
            pw=1
        )

        context_size, batch_size, input_size = x.shape
        
        # tensor to store decoder outputs
        outputs = torch.zeros_like(x)
        
        if context_size - t > 0:
            for i in range(context_size - t):
                outputs[i] = x[i + t]
        
        x = self.input_projection(x)  # [sen_len, batch_size, embed_dim]
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(x)
        
        # first input to the decoder is the last context image
        input = x[-1]
        for i in range(context_size - t, context_size):
            # insert input token embedding, previous hidden and previous cell states 
            # receive output tensor (predictions) and new hidden and cell states.
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # replace predictions in a tensor holding predictions
            if i >= 0: outputs[i] = self.output_projection(self.output_norm(output))  # [batch_size, input_size]

            input = output
        
        outputs = rearrange(
            outputs, '(n nh nw) b (ph pw c) -> b n c (nh ph) (nw pw)',
            ph=1,
            pw=1,
            nw=1,
            nh=1
        )

        return outputs
