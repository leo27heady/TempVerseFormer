# MViTv2 implementation taken from detectron2
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/mvit.py

import torch
import torch.nn as nn
from einops import rearrange

from .reversible import ReversibleModule, NotReversibleModule
from .block import MultiScaleBlock
from .rev_block import ReversibleMultiScaleBlock
from .rev_back_prop import EfficientMViTRevBackProp
from ..config import ReversibleVaeConfig



class UnpatchEmbed(NotReversibleModule):
    """
    Image to Unpatch Embedding to Image.
    """

    def __init__(
        self,
        kernel_size=(16, 16),
        stride=(16, 16),
        padding=(0, 0),
        norm_channels=8,
        embed_dim=768,
        im_channels=3,
        enable_amp=False
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
            im_channels (int): Number of input image channels.
        """
        super().__init__()

        self.enable_amp = enable_amp
        self.norm_out = nn.GroupNorm(norm_channels, embed_dim)
        self.non_linearity = nn.GELU()
        self.proj = nn.Conv2d(
            embed_dim,
            im_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            self.seed_cuda("unpatch")

            x = rearrange(x, "B H W C -> B C H W")
            x = self.norm_out(x)
            x = self.non_linearity(x)
            x = self.proj(x)
        return x

    def forward_for_backward(self, x):
        return self.forward(x)


class Rev_MViT_Decoder(nn.Module):
    """
    This module adds reversibility on top of Multiscale Vision Transformer (MViT) from :paper:'mvitv2'.
    """

    def __init__(
        self,
        im_channels: int,
        img_size: int,
        config: ReversibleVaeConfig,
        enable_amp: bool = False,
        custom_backward: bool = True,
    ):
        super().__init__()

        self.custom_backward = custom_backward

        embed_dim = config.z_channels
        dim_out = embed_dim
        num_heads = 16
        input_size = [1, 1]
        decoder_depth = config.decoder_stage_size * config.decoder_stages - (config.decoder_stage_size - 1 if config.decoder_halt_final_stage else 0)
        decoder_dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, decoder_depth)]  # stochastic depth decay rule
        stage = 1
        is_first_stage_block = True
        
        self.decode_blocks = nn.ModuleList()
        for block_number in range(1, decoder_depth + 1):
            is_last_stage_block = block_number % config.decoder_stage_size == 0 and stage <= config.decoder_stages

            # hybrid window attention: global attention in first stages.
            window_size_ = 0 if stage <= config.decoder_stages - 3 else input_size[0] // 4

            block_type = MultiScaleBlock if is_first_stage_block else ReversibleMultiScaleBlock
            self.decode_blocks.append(block_type(
                type="decode",
                dim=embed_dim * 2 if is_first_stage_block else embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                drop_path=decoder_dpr[block_number - 1],
                qkv_pool_kernel=config.qkv_pool_kernel if is_first_stage_block else (3, 3),
                stride_q=(2 if is_first_stage_block else 1),
                stride_kv=(2 if is_first_stage_block else 1),
                window_size=window_size_,
                residual_pooling=config.residual_pooling,
                use_rel_pos=config.use_rel_pos,
                rel_pos_zero_init=config.rel_pos_zero_init,
                input_size=input_size,
                enable_amp=enable_amp,
            ))
            
            if is_first_stage_block:
                num_heads = num_heads // 2 or 1
                embed_dim = embed_dim // 2
                dim_out = dim_out // 2
                input_size = [s * 2 for s in input_size]
                stage += 1
                is_first_stage_block = False
            
            if is_last_stage_block:
                is_first_stage_block = True
        
        assert img_size in input_size
                
        self.decode_blocks.append(UnpatchEmbed(
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            norm_channels=config.norm_channels,
            embed_dim=embed_dim*2,
            im_channels=im_channels,
            enable_amp=enable_amp
        ))


    @staticmethod
    def vanilla_backward(x, modules):
        """
        Using rev layers without rev backward propagation. Debugging purposes only.
        Deactivated with self.custom_backward.
        """
        for module in modules:
            x = module(x)
        return x

    def forward(self, x):
        x = rearrange(x, "B C H W -> B H W C")
        if not self.training or not self.custom_backward:
            x = Rev_MViT_Decoder.vanilla_backward(x, self.decode_blocks)
        else:
            x = EfficientMViTRevBackProp.apply(x, self.decode_blocks)

        return x
