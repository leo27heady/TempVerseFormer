import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp

from tempverse.utils import BaseLogger
from tempverse.rev_back_prop import ReversibleModule
from .ms_attention import MultiScaleAttention


class ReversibleMultiScaleBlock(ReversibleModule):
    """Reversible Multiscale Transformer block, no pool residual or projection."""

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        qkv_pool_kernel=(3, 3),
        stride_q=1,
        stride_kv=1,
        residual_pooling=True,
        window_size=0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
        custom_backward=True,
        enable_amp=False,
    ):
        """
        Args:
            type (str): Either encoder or decoder.
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            num_heads (int): Number of attention heads in the MViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            qkv_pool_kernel (tuple): kernel size for qkv pooling layers.
            stride_q (int): stride size for q pooling layer.
            stride_kv (int): stride size for kv pooling layer.
            residual_pooling (bool): If true, enable residual pooling.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution.
            enable_amp (bool): If True, enable mixed precision training.
        """

        super().__init__()
        
        self.logger = BaseLogger(__name__)

        self.custom_backward = custom_backward
        self.enable_amp = enable_amp

        self.norm1 = norm_layer(dim)
        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            pool_kernel=qkv_pool_kernel,
            stride_q=stride_q,
            stride_kv=stride_kv,
            pool_padding=[k // 2 for k in qkv_pool_kernel],
            residual_pooling=residual_pooling,
            window_size=window_size,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
        )

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=int(dim_out * mlp_ratio),
            out_features=dim_out,
            act_layer=act_layer,
        )

        if stride_q > 1:
            raise ValueError(
                "stride_q > 1 is not supported for ReversibleMultiScaleBlock."
            )

    def F(self, x):
        """Attention forward pass"""
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            x_out = self.attn(self.norm1(x))
        return x_out

    def G(self, x):
        """MLP forward pass"""
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            x_out = self.mlp(self.norm2(x))
        return x_out

    def forward(self, x):
        """
        forward pass equations:
        O_2 = I_2 + Attention(I_1), F = Attention
        O_1 = I_1 + MLP(O_2), G = MLP
        """
        
        self.logger.debug("Start ReversibleMultiScaleBlock forward")

        I_1, I_2 = torch.chunk(x, 2, dim=-1)

        self.logger.debug(f"Pre Attention: {I_1.shape}")
        self.seed_cuda("attn")
        f_out = self.F(I_1)
        self.logger.debug(f"Post Attention: {f_out.shape}")

        self.seed_cuda("droppath")
        O_2 = I_2 + self.drop_path(f_out)

        # free memory
        del I_2

        self.logger.debug(f"Pre MLP: {O_2.shape}")
        self.seed_cuda("mlp")
        g_out = self.G(O_2)
        self.logger.debug(f"Pre Attention: {g_out.shape}")

        self.seed_cuda("droppath", safe=True)
        O_1 = I_1 + self.drop_path(g_out)

        del I_1

        return torch.cat([O_1, O_2], dim=-1)

    def backward_pass(self, y, dy):
        """
        equations for recovering activations:
        I_1 = O_1 - MLP(O_2), G = MLP
        I_2 = O_2 - Attention(I_1), F = Attention
        """
        
        self.logger.debug("Start ReversibleMultiScaleBlock backward")
        
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

            self.seed_cuda("droppath", safe=True)
            g_out = self.drop_path(g_out)

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

            self.seed_cuda("droppath")
            f_out = self.drop_path(f_out)

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

    def backward_pass_recover(self, Y_1, Y_2):
        """
        Use equations to recover activations and return them.
        Used for parallelizing the backward pass.
        """
        with torch.enable_grad():
            Y_1.requires_grad = True

            torch.manual_seed(self.seeds["mlp"])
            g_Y_1 = self.G(Y_1)

            torch.manual_seed(self.seeds["droppath"])
            g_Y_1 = self.drop_path(g_Y_1)

        with torch.no_grad():
            X_2 = Y_2 - g_Y_1

        with torch.enable_grad():
            X_2.requires_grad = True

            torch.manual_seed(self.seeds["attn"])
            f_X_2 = self.F(X_2)

            torch.manual_seed(self.seeds["droppath"])
            f_X_2 = self.drop_path(f_X_2)

        with torch.no_grad():
            X_1 = Y_1 - f_X_2

        # Keep tensors around to do backprop on the graph.
        ctx = [X_1, X_2, Y_1, g_Y_1, f_X_2]
        return ctx

    def backward_pass_grads(self, X_1, X_2, Y_1, g_Y_1, f_X_2, dY_1, dY_2):
        """
        Receive intermediate activations and inputs to backprop through.
        """

        with torch.enable_grad():
            g_Y_1.backward(dY_2)

        with torch.no_grad():
            dY_1 = dY_1 + Y_1.grad
            Y_1.grad = None

        with torch.enable_grad():
            f_X_2.backward(dY_1)

        with torch.no_grad():
            dY_2 = dY_2 + X_2.grad
            X_2.grad = None
            X_2.detach()

        return dY_1, dY_2
