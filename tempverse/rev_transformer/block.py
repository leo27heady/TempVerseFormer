import torch
from torch import nn
from torch.autograd import Function as Function
from torch.nn import MultiheadAttention as MHA

from ..rev_back_prop import ReversibleModule


class MLPSubblock(nn.Module):
    """
    This creates the function G such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4,
        enable_amp=False,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.enable_amp = enable_amp

    def forward(self, x):
        # The reason for implementing autocast inside forward loop instead
        # in the main training logic is the implicit forward pass during
        # memory efficient gradient backpropagation. In backward pass, the
        # activations need to be recomputed, and if the forward has happened
        # with mixed precision, the recomputation must also be so. This cannot
        # be handled with the autocast setup in main training logic.
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            return self.mlp(self.norm(x))


class AttentionSubBlock(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim,
        num_heads,
        enable_amp=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)

        # using vanilla attention for simplicity. To support adanced attention
        # module see pyslowfast.
        # Note that the complexity of the attention module is not a concern
        # since it is used blackbox as F block in the reversible logic and
        # can be arbitrary.
        self.attn = MHA(dim, num_heads, batch_first=True)
        self.enable_amp = enable_amp

    def forward(self, x):
        # See MLP fwd pass for explanation.
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            x = self.norm(x)
            
            # bs, sl, nc = x.shape
            # attn_mask = nn.Transformer.generate_square_subsequent_mask(sl, device=x.device, dtype=x.dtype)
            out, _ = self.attn(x, x, x)  # , attn_mask=attn_mask, is_causal=True)
            return out


class ReversibleBlock(ReversibleModule):
    """
    Reversible Blocks for Reversible Vision Transformer.
    See Section 3.3.2 in paper "Reversible Vision Transformers" for details.
    """

    def __init__(self, dim, num_heads, enable_amp):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """
        super().__init__()
        # F and G can be arbitrary functions, here they use
        # simple attwntion and MLP sub-blocks using vanilla attention.
        self.F = AttentionSubBlock(
            dim=dim, num_heads=num_heads, enable_amp=enable_amp
        )

        self.G = MLPSubblock(dim=dim, enable_amp=enable_amp)

        # note that since all functions are deterministic, and they are
        # not using any stochastic elements such as dropout, they do
        # not need to control seeds for the random number generator.
        # To see usage with controlled seeds and dropout, see pyslowfast.

    def forward(self, x):
        """
        forward pass equations:
        O_2 = I_2 + Attention(I_1), F = Attention
        O_1 = I_1 + MLP(O_2), G = MLP
        """

        I_1, I_2 = torch.chunk(x, 2, dim=-1)

        self.seed_cuda("attn")
        f_out = self.F(I_1)

        O_2 = I_2 + f_out

        # free memory
        del I_2

        self.seed_cuda("mlp")
        g_out = self.G(O_2)

        O_1 = I_1 + g_out

        del I_1

        return torch.cat([O_1, O_2], dim=-1)

    def backward_pass(self, y, dy):
        """
        equations for recovering activations:
        I_1 = O_1 - MLP(O_2), G = MLP
        I_2 = O_2 - Attention(I_1), F = Attention

        And they use pytorch native logic carefully to
        calculate gradients on F and G.
        """
        
        O_1, O_2 = torch.chunk(y, 2, dim=-1)
        dY_1, dY_2 = torch.chunk(dy, 2, dim=-1)

        # temporarily record intermediate activation for G
        # and use them for gradient calculation of G
        with torch.enable_grad():
            O_2.requires_grad = True

            # reconstructing the intermediate activations
            # and the computational graph for F.
            self.seed_cuda("mlp")
            g_out = self.G(O_2)

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            g_out.backward(dY_1, retain_graph=True)

        # activation recomputation is by design and not part of
        # the computation graph in forward pass. Hence they do not
        # need to record it in the computation graph.
        with torch.no_grad():
            # recomputing I_1 from the rev equation
            I_1 = O_1 - g_out

            # free memory since g_out is now not needed
            del g_out

            # the gradients for the previous block
            # note that it is called dY_2 but it in fact dI_2 in math.
            # reusing same variable to save memory
            dY_2 = dY_2 + O_2.grad

            # free memory since O_2.grad is now not needed
            O_2.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            I_1.requires_grad = True

            # reconstructing the intermediate activations
            # and the computational graph for F.
            self.seed_cuda("attn")
            f_out = self.F(I_1)

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            f_out.backward(dY_2, retain_graph=True)

        # propagate reverse computed activations at the start of
        # the previous block for backprop.s
        with torch.no_grad():
            # recomputing I_2 from the rev equation
            I_2 = O_2 - f_out

            del f_out, O_2
            # the gradients for the previous block
            # note that it is called dY_1 but it in fact dI_1 in math.
            # reusing same variable to save memory
            dY_1 = dY_1 + I_1.grad
            
            # free memory since I_1.grad is now not needed
            I_1.grad = None

            I_1 = I_1.detach()

        # et voila~
        return torch.cat([I_1, I_2], dim=-1), torch.cat([dY_1, dY_2], dim=-1)
