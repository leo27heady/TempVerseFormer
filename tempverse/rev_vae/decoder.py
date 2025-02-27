# Based on the MViTv2 implementation taken from detectron2
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/mvit.py

import torch
import torch.nn as nn
from einops import rearrange

from ..rev_back_prop import ReversibleModule, NotReversibleModule, EfficientRevBackProp
from .res_net_block import ResNetUpBlock
from .rev_block import ReversibleMultiScaleBlock
from ..config import ReversibleVaeConfig
from ..utils import BaseLogger


class PostQuantModule(NotReversibleModule):
    """
    Postquant module
    """

    def __init__(
        self,
        z_channels,
        embed_dim,
        custom_backward=True,
        enable_amp=False
    ):
        super().__init__()

        self.custom_backward = custom_backward
        self.enable_amp = enable_amp
        
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(z_channels, embed_dim, kernel_size=3, padding=(1, 1))

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            self.seed_cuda("post_quant")
            x = self.post_quant_conv(x)
            x = self.decoder_conv_in(x)
            x = rearrange(x, "B C H W -> B H W C")
        return x

    def forward_for_backward(self, x):
        return self.forward(x)


class UnpatchEmbed(NotReversibleModule):
    """
    Image to Unpatch Embedding to Image.
    """

    def __init__(
        self,
        norm_channels=8,
        fuse_res_paths=True, 
        embed_dim=768,
        im_channels=3,
        custom_backward=True,
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

        self.custom_backward = custom_backward
        self.enable_amp = enable_amp
        self.fuse_res_paths = fuse_res_paths

        if self.fuse_res_paths:
            self.res_paths_fusion = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)
        
        self.norm_out = nn.GroupNorm(norm_channels, embed_dim)
        self.non_linearity = nn.GELU()
        self.conv_out = nn.Conv2d(
            embed_dim,
            im_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            self.seed_cuda("unpatch")
            x = rearrange(x, "B H W C -> B C H W")
            
            if self.fuse_res_paths:
                x = self.res_paths_fusion(x)

            x = self.norm_out(x)
            x = self.non_linearity(x)
            x = self.conv_out(x)
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
        
        self.logger = BaseLogger(__name__)

        up_channels = list(reversed(config.down_channels.copy()))
        attn_up = [False] + config.attn_up.copy()  # insert False as the 1st entry always doesn't has path split

        number_of_stages: int = len(up_channels) - 1
        assert number_of_stages > 1  # at least 2 stages
        assert len(attn_up) == len(up_channels)  # channels and attentions lists have same lengths
        assert config.decoder_stage_size >= 1  # if 1 only upsample remains

        self.custom_backward = custom_backward
        
        self.decode_blocks = nn.ModuleList()
        self.decode_blocks.append(PostQuantModule(
            z_channels=config.z_channels,
            embed_dim=up_channels[0],
            custom_backward=custom_backward,
            enable_amp=enable_amp
        ))

        # decoder_dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, decoder_depth)]  # stochastic depth decay rule
        num_heads = config.num_heads_max
        input_size = [1, 1]

        for stage in range(1, number_of_stages + 1):
            self.decode_blocks.append(ResNetUpBlock(
                in_channels=up_channels[stage - 1], 
                out_channels=up_channels[stage], 
                norm_channels=config.norm_channels,
                num_layers=config.up_scale_layers, 
                fuse_res_paths=attn_up[stage - 1],  # fuse if previous stage has rev blocks
                duplicate_output=attn_up[stage],  # duplicate if current stage has rev blocks
                custom_backward=custom_backward,
                enable_amp=enable_amp,
            ))
            input_size = [s * 2 for s in input_size]

            for _ in (range(config.decoder_stage_size - 1) if attn_up[stage] else []):
                self.decode_blocks.append(ReversibleMultiScaleBlock(
                    dim=up_channels[stage],
                    dim_out=up_channels[stage],
                    num_heads=num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    drop_path=0.0,
                    qkv_pool_kernel=(3, 3),
                    stride_q=1,
                    stride_kv=1,
                    window_size=(input_size[0] // 4),  # TODO: hybrid window attention: global attention in first stages.
                    residual_pooling=config.residual_pooling,
                    use_rel_pos=config.use_rel_pos,
                    rel_pos_zero_init=config.rel_pos_zero_init,
                    input_size=input_size,
                    custom_backward=custom_backward,
                    enable_amp=enable_amp,
                ))
            num_heads = max(num_heads // 2, config.num_heads_min)
        
        assert img_size in input_size

        self.decode_blocks.append(UnpatchEmbed(
            norm_channels=config.norm_channels,
            fuse_res_paths=attn_up[-1],  # fuse if last stage has rev blocks
            embed_dim=up_channels[-1],
            im_channels=im_channels,
            custom_backward=custom_backward,
            enable_amp=enable_amp
        ))

    @staticmethod
    def vanilla_backward(x, t, modules):
        """
        Using rev layers without rev backward propagation. Debugging purposes only.
        Deactivated with self.custom_backward.
        """
        for module in modules:
            x = module(x)
        return x

    def forward(self, x):
        if not self.training or not self.custom_backward:
            self.logger.debug("Start Rev_MViT_Decoder vanilla_backward")
            executing_fn = Rev_MViT_Decoder.vanilla_backward
        else:
            self.logger.debug("Start Rev_MViT_Decoder EfficientRevBackProp")
            x.requires_grad_()
            executing_fn = EfficientRevBackProp.apply
        
        x = executing_fn(x, 1, self.decode_blocks)

        return x
