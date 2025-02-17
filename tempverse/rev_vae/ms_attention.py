import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .utils import (
    attention_pool,
    add_decomposed_rel_pos,
    window_partition,
    window_unpartition,
)


class MultiScaleAttention(nn.Module):
    """Multiscale Multi-head Attention block."""

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        pool_kernel=(3, 3),
        stride_q=1,
        stride_kv=1,
        pool_padding=1,
        residual_pooling=True,
        window_size=0,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
        transpose=False
    ):
        """
        Args:
            dim (int): Number of input channels.
            dim_out (int): Number of output channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            pool_kernel (tuple): kernel size for qkv pooling layers.
            stride_q (int): stride size for q pooling layer.
            stride_kv (int): stride size for kv pooling layer.
            residual_pooling (bool): If true, enable residual pooling.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        # qkv pooling
        dim_conv = dim_out // num_heads
        self.pool_q = (nn.ConvTranspose2d if transpose else nn.Conv2d)(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_q,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_q = norm_layer(dim_conv)
        self.pool_k = (nn.ConvTranspose2d if transpose else nn.Conv2d)(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_kv,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_k = norm_layer(dim_conv)
        self.pool_v = (nn.ConvTranspose2d if transpose else nn.Conv2d)(
            dim_conv,
            dim_conv,
            pool_kernel,
            stride=stride_kv,
            padding=pool_padding,
            groups=dim_conv,
            bias=False,
        )
        self.norm_v = norm_layer(dim_conv)

        self.window_size = window_size
        if window_size:
            self.q_win_size = window_size // stride_q
            self.kv_win_size = window_size // stride_kv

            if self.q_win_size <= 1 or self.kv_win_size <= 1:
                self.window_size = 0

        self.residual_pooling = residual_pooling

        
        assert input_size[0] == input_size[1]
        size = input_size[0]
        self.use_rel_pos = use_rel_pos and size > 1
        
        if self.use_rel_pos:
            # initialize relative positional embeddings
            rel_dim = 2 * max(size // stride_q, size // stride_kv) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_dim, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H, W, C)
        qkv = (
            self.qkv(x)
            .reshape(B, H, W, 3, self.num_heads, -1)
            .permute(3, 0, 4, 1, 2, 5)
        )
        # q, k, v with shape (B * nHead, H, W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H, W, -1).unbind(0)

        q = attention_pool(q, self.pool_q, self.norm_q)
        k = attention_pool(k, self.pool_k, self.norm_k)
        v = attention_pool(v, self.pool_v, self.norm_v)

        ori_q = q
        if self.window_size:
            q, q_hw_pad = window_partition(q, self.q_win_size)
            k, kv_hw_pad = window_partition(k, self.kv_win_size)
            v, _ = window_partition(v, self.kv_win_size)
            q_hw = (self.q_win_size, self.q_win_size)
            kv_hw = (self.kv_win_size, self.kv_win_size)
        else:
            q_hw = q.shape[1:3]
            kv_hw = k.shape[1:3]

        q = q.view(q.shape[0], np.prod(q_hw), -1)
        k = k.view(k.shape[0], np.prod(kv_hw), -1)
        v = v.view(v.shape[0], np.prod(kv_hw), -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, q_hw, kv_hw
            )

        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.view(x.shape[0], q_hw[0], q_hw[1], -1)

        if self.window_size:
            x = window_unpartition(
                x, self.q_win_size, q_hw_pad, ori_q.shape[1:3]
            )

        if self.residual_pooling:
            x += ori_q

        H, W = x.shape[1], x.shape[2]
        x = (
            x.view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)

        return x
