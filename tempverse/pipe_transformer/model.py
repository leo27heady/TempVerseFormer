import torch
from torch import nn
from torch.autograd import Function as Function
from einops import rearrange

from ..config import PipeTransformerConfig
from ..vanilla_transformer.block import TransformerBlock


class PipeTransformer(nn.Module):
    def __init__(
        self,
        config: PipeTransformerConfig,
        context_size: int,
        enable_amp: bool = False,
    ):
        super().__init__()

        self.input_dim = config.input_dim
        self.embed_dim = config.embed_dim
        self.n_head = config.n_head
        self.depth = config.depth

        self.input_projection = nn.Linear(self.input_dim, self.embed_dim, bias=True)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.embed_dim,
                    num_heads=self.n_head,
                    enable_amp=enable_amp,
                )
                for _ in range(self.depth)
            ]
        )

        self.pos_embeddings = nn.Parameter(
            torch.zeros(1, context_size, self.embed_dim)
        )

        self.norm = nn.LayerNorm(self.embed_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.input_dim, bias=True)
        )

    def forward(self, x, t):
        
        x = rearrange(
            x, 'b n c (nh ph) (nw pw) -> b (n nh nw) (ph pw c)',
            ph=1,
            pw=1
        )

        # Input projection
        x = self.input_projection(x)

        for i in range(t):
            x = torch.cat((x[:, 1:], x[:, -1:]), dim=1)  # git rid of the first entry and duplicate the last
            x += self.pos_embeddings
            for _, layer in enumerate(self.layers):
                x = layer(x)

        # termination fusion
        x = self.output_projection(self.norm(x))
        
        x = rearrange(
            x, 'b (n nh nw) (ph pw c) -> b n c (nh ph) (nw pw)',
            ph=1,
            pw=1,
            nw=1,
            nh=1
        )

        return x
