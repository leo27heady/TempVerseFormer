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


class PatchEmbed(NotReversibleModule):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size=(16, 16),
        stride=(16, 16),
        padding=(0, 0),
        im_channels=3,
        embed_dim=8,
        enable_amp=False
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            im_channels (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.enable_amp = enable_amp
        self.proj = nn.Conv2d(
            im_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            if "patch" in self.seeds:
                torch.manual_seed(self.seeds.pop("patch"))
            else:
                self.seed_cuda("patch")
            
            x = self.proj(x)
            x = rearrange(x, "B C H W -> B H W C")
            x = torch.cat([x, x], dim=-1)
        return x

    def forward_for_backward(self, x):
        return self.forward(x)


class PreQuantModule(NotReversibleModule):
    """
    Prequant module
    """

    def __init__(
        self,
        norm_channels=8,
        in_dim=256,
        out_dim=1024,
        enable_amp=False
    ):
        super().__init__()

        self.enable_amp = enable_amp
        self.encoder_norm_out = nn.GroupNorm(norm_channels, in_dim)
        self.act_layer = nn.GELU()
        self.encoder_conv_out = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        # Latent Dimension is 2*Latent because they are predicting mean & variance
        self.pre_quant_conv = nn.Conv2d(out_dim, out_dim, kernel_size=1)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            if "pre_quant" in self.seeds:
                torch.manual_seed(self.seeds.pop("pre_quant"))
            else:
                self.seed_cuda("pre_quant")

            x = rearrange(x, "B H W C -> B C H W")
            x = self.encoder_norm_out(x)
            x = self.act_layer(x)
            x = self.encoder_conv_out(x)
            x = self.pre_quant_conv(x)
        
        return x

    def forward_for_backward(self, x):
        return self.forward(x)


class SamplingModule(NotReversibleModule):
    """
    Sampling module
    """

    def __init__(
        self,
        enable_amp=False
    ):
        super().__init__()

        self.enable_amp = enable_amp

    def sample_output(self, out):
        mean, logvar = torch.chunk(out, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn_like(mean)
        return sample

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            if "sample" in self.seeds:
                torch.manual_seed(self.seeds.pop("sample"))
            else:
                self.seed_cuda("sample")
            
            O1, O2 = torch.chunk(x, 2, dim=1)
            o1_sample = self.sample_output(O1)
            o2_sample = self.sample_output(O2)
            sample = torch.cat([o1_sample, o2_sample], dim=1)
        
        return sample

    def forward_for_backward(self, x):
        return self.forward(x)


class Rev_MViT_Encoder(nn.Module):
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
        
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(PatchEmbed(
            kernel_size=config.patch_kernel,
            stride=config.patch_stride,
            padding=config.patch_padding,
            im_channels=im_channels,
            embed_dim=config.patch_embed_dim,
            enable_amp=enable_amp
        ))
        embed_dim = config.patch_embed_dim

        encoder_depth = config.encoder_stage_size * config.encoder_stages
        encoder_dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, encoder_depth)]  # stochastic depth decay rule
        embed_dim *= 2  # As we duplicate the input in channels dim
        num_heads = config.num_heads
        dim_out = embed_dim
        input_size = [img_size // config.patch_stride[0], img_size // config.patch_stride[1]]
        # self._out_feature_strides = {}
        # self._out_feature_channels = {}
        
        stage = 1
        for block_number in range(config.encoder_stage_size if config.encoder_jump_first_stage else 1, encoder_depth + 1):
            is_last_stage_block = block_number % config.encoder_stage_size == 0 and stage <= config.encoder_stages

            if is_last_stage_block:
                dim_out *= 2
                num_heads = min(16, num_heads*2)

            # hybrid window attention: global attention in last stages.
            window_size_ = 0 if stage != 1 else input_size[0] // 4

            block_type = MultiScaleBlock if is_last_stage_block else ReversibleMultiScaleBlock

            self.encode_blocks.append(block_type(
                type="encode",
                dim=embed_dim if is_last_stage_block else embed_dim // 2,
                dim_out=dim_out if is_last_stage_block else dim_out // 2,
                num_heads=num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                drop_path=encoder_dpr[block_number - 1],
                qkv_pool_kernel=config.qkv_pool_kernel if is_last_stage_block else (3, 3),
                stride_q=(2 if is_last_stage_block else 1),
                stride_kv=(2 if is_last_stage_block else 1),
                window_size=window_size_,
                residual_pooling=config.residual_pooling,
                use_rel_pos=config.use_rel_pos,
                rel_pos_zero_init=config.rel_pos_zero_init,
                input_size=input_size,
                enable_amp=enable_amp,
            ))

            if is_last_stage_block:
                embed_dim = dim_out
                input_size = [s // 2 for s in input_size]
                assert 0 not in input_size
                stage += 1
        
        assert 1 in input_size
        
        self.encode_blocks.append(PreQuantModule(config.norm_channels, embed_dim, 4 * config.z_channels, enable_amp))

        self.sampling = SamplingModule(enable_amp)


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
        if not self.training or not self.custom_backward:
            x = Rev_MViT_Encoder.vanilla_backward(x, self.encode_blocks)
        else:
            x = EfficientMViTRevBackProp.apply(x, self.encode_blocks)
        
        sample = self.sampling(x)

        return sample, x
