import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp

from .reversible import NotReversibleModule
from .utils import attention_pool
from .ms_attention import MultiScaleAttention


class MultiScaleBlock(NotReversibleModule):
    """Multiscale Transformer block, specifically for Stage Transitions."""

    def __init__(
        self,
        type,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        qkv_pool_kernel=(4, 4),
        stride_q=1,
        stride_kv=1,
        residual_pooling=True,
        window_size=0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
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
            pool_padding=1,
            residual_pooling=residual_pooling,
            window_size=window_size,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
            transpose=(type == "decode")
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
        self.enable_amp = enable_amp

        # For Stage-Transition
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        if stride_q > 1:
            self.pool_skip = (nn.Conv2d if type == "encode" else nn.ConvTranspose2d)(
                dim_out,
                dim_out,
                qkv_pool_kernel, stride_q, 1
            )

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            self.seed_cuda("drop_path")

            x_norm = self.norm1(x)
            x_block = self.attn(x_norm)

            if hasattr(self, "proj"):
                x = self.proj(x_norm)
            if hasattr(self, "pool_skip"):
                x = attention_pool(x, self.pool_skip)

            x = x + self.drop_path(x_block)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def forward_for_backward(self, x):
        return self.forward(x)
