from torch import nn

from ..rev_transformer.block import AttentionSubBlock, MLPSubblock


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, enable_amp):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """
        super().__init__()
        self.F = AttentionSubBlock(
            dim=dim, num_heads=num_heads, enable_amp=enable_amp
        )
        self.G = MLPSubblock(dim=dim, enable_amp=enable_amp)

    def forward(self, X):
        """
        forward pass equations:
        Y = X + Attention(X), F = Attention
        Y = Y + MLP(Y), G = MLP
        """
        Y = X + self.F(X)
        Y = Y + self.G(Y)
        return Y
