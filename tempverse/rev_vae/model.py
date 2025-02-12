# MViTv2 implementation taken from detectron2
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/mvit.py

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .utils import get_abs_pos
from .block import MultiScaleBlock
from .rev_block import ReversibleMultiScaleBlock
from ..rev_back_prop import RevBackProp


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size=(16, 16),
        stride=(16, 16),
        padding=(0, 0),
        in_chans=3,
        embed_dim=768,
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class ReversibleMViT(nn.Module):
    """
    This module adds reversibility on top of Multiscale Vision Transformer (MViT) from :paper:'mvitv2'.
    """

    def __init__(
        self,
        img_size=224,
        patch_kernel=(7, 7),
        patch_stride=(4, 4),
        patch_padding=(3, 3),
        in_chans=3,
        embed_dim=96,
        depth=16,
        num_heads=1,
        last_block_indexes=(0, 2, 13, 15),
        qkv_pool_kernel=(3, 3),
        adaptive_kv_stride=4,
        adaptive_window_size=56,
        residual_pooling=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=False,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        fast_backprop=False,
        enable_amp=False,
        custom_backward: bool = True,
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_kernel (tuple): kernel size for patch embedding.
            patch_stride (tuple): stride size for patch embedding.
            patch_padding (tuple): padding size for patch embedding.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of MViT.
            num_heads (int): Number of base attention heads in each MViT block.
            last_block_indexes (tuple): Block indexes for last blocks in each stage.
            qkv_pool_kernel (tuple): kernel size for qkv pooling layers.
            adaptive_kv_stride (int): adaptive stride size for kv pooling.
            adaptive_window_size (int): adaptive window size for window attention blocks.
            residual_pooling (bool): If true, enable residual pooling.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            fast_backprop (bool): If True, use fast backprop, i.e. PaReprop.
            enable_amp (bool): If True, enable automatic mixed precision.
        """
        super().__init__()
        self.patch_embed = PatchEmbed(
            kernel_size=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        img_size = img_size[0]

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (img_size // patch_stride[0]) * (
                img_size // patch_stride[1]
            )
            num_positions = num_patches
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_positions, embed_dim)
            )
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        dim_out = embed_dim
        stride_kv = adaptive_kv_stride
        window_size = adaptive_window_size
        input_size = (img_size // patch_stride[0], img_size // patch_stride[1])
        stage = 2
        stride = patch_stride[0]
        # self._out_feature_strides = {}
        # self._out_feature_channels = {}
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Multiply stride_kv by 2 if it's the last block of stage2 and stage3.
            # Here however, we modify it so that we only look at stage2 (since we only have 3 stages for a smaller CIFAR model)
            # if i == last_block_indexes[1] or i == last_block_indexes[2]:
            if i == last_block_indexes[1]:
                stride_kv_ = stride_kv * 2
            else:
                stride_kv_ = stride_kv
            # hybrid window attention: global attention in last three stages.
            window_size_ = 0 if i in last_block_indexes[1:] else window_size
            block_type = (
                MultiScaleBlock
                if i - 1 in last_block_indexes
                else ReversibleMultiScaleBlock
            )
            block = block_type(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                qkv_pool_kernel=qkv_pool_kernel,
                stride_q=2 if i - 1 in last_block_indexes else 1,
                stride_kv=stride_kv_,
                residual_pooling=residual_pooling,
                window_size=window_size_,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                input_size=input_size,
                enable_amp=enable_amp,
            )
            self.blocks.append(block)

            embed_dim = dim_out
            if i in last_block_indexes:
                embed_dim *= 2
                dim_out *= 2
                num_heads *= 2
                stride_kv = max(stride_kv // 2, 1)
                stride *= 2
                stage += 1
            if i - 1 in last_block_indexes:
                window_size = window_size // 2
                input_size = [s // 2 for s in input_size]

        # self._out_features = out_features
        self._last_block_indexes = last_block_indexes
        self.use_fast_backprop = fast_backprop

        if self.use_fast_backprop:
            # Initialize streams globally
            global s1, s2
            s1 = torch.cuda.default_stream(device=torch.cuda.current_device())
            # s1 = torch.cuda.Stream(device=torch.cuda.current_device())
            s2 = torch.cuda.Stream(device=torch.cuda.current_device())

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.custom_backward = custom_backward
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def vanilla_backward(h, t, layers):
        """
        Using rev layers without rev backward propagation. Debugging purposes only.
        Deactivated with self.custom_backward.
        """
        # split into hidden states (h) and attention_output (a)
        h, a = torch.chunk(h, 2, dim=-1)
        
        for i in range(t):
            for _, layer in enumerate(layers):
                a, h = layer(a, h)

        return torch.cat([a, h], dim=-1)

    def forward(self, x):
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, False, x.shape[1:3])

        # process layers in reversible and irreversible stacks
        stack = []
        for l_i in range(len(self.blocks)):
            if isinstance(self.blocks[l_i], MultiScaleBlock):
                stack.append(("Stage Transition", l_i))
            else:
                if len(stack) == 0 or stack[-1][0] == "Stage Transition":
                    stack.append(("Reversible", []))
                stack[-1][1].append(l_i)

        for i, substack in enumerate(stack):
            if substack[0] == "Stage Transition":
                x = self.blocks[substack[1]](x)
            else:
                # first concat two copies of x for the two streams
                x = torch.cat([x, x], dim=-1)

                if not self.training or not self.custom_backward:
                    executing_fn = ReversibleMViT.vanilla_backward
                else:
                    executing_fn = RevBackProp.apply

                x = executing_fn(
                    x, 1, self.blocks[substack[1][0] : substack[1][-1] + 1]
                )

        # B H W C -> B C H W
        x = x.permute(0, 3, 1, 2)

        return x
