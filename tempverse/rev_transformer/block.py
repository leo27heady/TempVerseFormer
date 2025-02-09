import torch
from torch import nn
from torch.autograd import Function as Function
from torch.nn import MultiheadAttention as MHA


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


class ReversibleBlock(nn.Module):
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

    def forward(self, X_1, X_2):
        """
        forward pass equations:
        Y_1 = X_1 + Attention(X_2), F = Attention
        Y_2 = X_2 + MLP(Y_1), G = MLP
        """

        # Y_1 : attn_output
        f_X_2 = self.F(X_2)

        # Y_1 = X_1 + f(X_2)
        Y_1 = X_1 + f_X_2

        # free memory since X_1 is now not needed
        del X_1

        g_Y_1 = self.G(Y_1)

        # Y_2 = X_2 + g(Y_1)
        Y_2 = X_2 + g_Y_1

        # free memory since X_2 is now not needed
        del X_2

        return Y_1, Y_2

    def backward_pass(
        self,
        Y_1,
        Y_2,
        dY_1,
        dY_2,
    ):
        """
        equation for activation recomputation:
        X_2 = Y_2 - G(Y_1), G = MLP
        X_1 = Y_1 - F(X_2), F = Attention

        And they use pytorch native logic carefully to
        calculate gradients on F and G.
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():
            Y_1.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            g_Y_1 = self.G(Y_1)

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            g_Y_1.backward(dY_2, retain_graph=True)

        # activation recomputation is by design and not part of
        # the computation graph in forward pass. Hence they do not
        # need to record it in the computation graph.
        with torch.no_grad():
            # recomputing X_2 from the rev equation
            X_2 = Y_2 - g_Y_1

            # free memory since g_Y_1 is now not needed
            del g_Y_1

            # the gradients for the previous block
            # note that it is called dY_1 but it in fact dX_1 in math.
            # reusing same variable to save memory
            dY_1 = dY_1 + Y_1.grad

            # free memory since Y_1.grad is now not needed
            Y_1.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            f_X_2 = self.F(X_2)

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            f_X_2.backward(dY_1, retain_graph=True)

        # propagate reverse computed acitvations at the start of
        # the previou block for backprop.s
        with torch.no_grad():
            # recomputing X_1 from the rev equation
            X_1 = Y_1 - f_X_2

            del f_X_2, Y_1
            # the gradients for the previous block
            # note that it is called dY_2 but it in fact dX_2 in math.
            # reusing same variable to save memory
            dY_2 = dY_2 + X_2.grad

            # free memory since X_2.grad is now not needed
            X_2.grad = None

            X_2 = X_2.detach()

        # et voila~
        return X_1, X_2, dY_1, dY_2
