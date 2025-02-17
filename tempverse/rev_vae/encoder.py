# MViTv2 implementation taken from detectron2
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/mvit.py

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange

from .block import MultiScaleBlock
from .rev_block import ReversibleMultiScaleBlock
from .rev_back_prop import EfficientMViTRevBackProp
from ..config import ReversibleVaeConfig


class PatchEmbed(nn.Module):
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

        self.proj = nn.Conv2d(
            im_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "B C H W -> B H W C")
        x = torch.cat([x, x], dim=-1)
        return x


class PreQuantSampling(nn.Module):
    """
    Prequant sampling module
    """

    def __init__(
        self,
        norm_channels=8,
        in_dim=256,
        out_dim=1024,
    ):
        super().__init__()

        self.encoder_norm_out = nn.GroupNorm(norm_channels, in_dim)
        self.act_layer = nn.GELU()
        self.encoder_conv_out = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        # Latent Dimension is 2*Latent because they are predicting mean & variance
        self.pre_quant_conv = nn.Conv2d(out_dim, out_dim, kernel_size=1)

    def sample_output(self, out):
        mean, logvar = torch.chunk(out, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn(mean.shape).to(device=out.device)
        return sample

    def forward(self, x):
        x = rearrange(x, "B H W C -> B C H W")
        x = self.encoder_norm_out(x)
        x = self.act_layer(x)
        x = self.encoder_conv_out(x)
        x = self.pre_quant_conv(x)
        
        O1, O2 = torch.chunk(x, 2, dim=1)
        o1_sample = self.sample_output(O1)
        o2_sample = self.sample_output(O2)
        sample = torch.cat([o1_sample, o2_sample], dim=1)
        return sample, x


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
        """
        Args:
            img_size (int): Input image size.
            patch_kernel (tuple): kernel size for patch embedding.
            patch_stride (tuple): stride size for patch embedding.
            patch_padding (tuple): padding size for patch embedding.
            im_channels (int): Number of input image channels.
            patch_embed_dim (int): Patch embedding dimension.
            depth (int): Depth of MViT.
            num_heads (int): Number of base attention heads in each MViT block.
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

        self.custom_backward = custom_backward
        
        self.patch_embed = PatchEmbed(
            kernel_size=config.patch_kernel,
            stride=config.patch_stride,
            padding=config.patch_padding,
            im_channels=im_channels,
            embed_dim=config.patch_embed_dim,
        )
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
        self.encode_blocks = nn.ModuleList()
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
        
        self.pre_quant_sample = PreQuantSampling(config.norm_channels, embed_dim, 4 * config.z_channels)

    #     self.use_fast_backprop = config.fast_backprop

    #     if self.use_fast_backprop:
    #         # Initialize streams globally
    #         global s1, s2
    #         s1 = torch.cuda.default_stream(device=torch.cuda.current_device())
    #         # s1 = torch.cuda.Stream(device=torch.cuda.current_device())
    #         s2 = torch.cuda.Stream(device=torch.cuda.current_device())

    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=0.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.GroupNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)


    @staticmethod
    def vanilla_backward(h, layers):
        """
        Using rev layers without rev backward propagation. Debugging purposes only.
        Deactivated with self.custom_backward.
        """
        # split into hidden states (h) and attention_output (a)
        h, a = torch.chunk(h, 2, dim=-1)
        
        for _, layer in enumerate(layers):
            a, h = layer(a, h)

        return torch.cat([a, h], dim=-1)
    
    def create_blocks_stack(self, blocks):
        stack = []
        for l_i in range(len(blocks)):
            if isinstance(blocks[l_i], MultiScaleBlock):
                stack.append(("Stage Transition", l_i))
            else:
                if len(stack) == 0 or stack[-1][0] == "Stage Transition":
                    stack.append(("Reversible", []))
                stack[-1][1].append(l_i)
        return stack

    def forward(self, x):
        x = self.patch_embed(x)

        # process layers in reversible and irreversible stacks
        stack = self.create_blocks_stack(self.encode_blocks)
        for i, substack in enumerate(stack):
            if substack[0] == "Stage Transition":
                x = self.encode_blocks[substack[1]](x)
            else:
                if not self.training or not self.custom_backward:
                    executing_fn = Rev_MViT_Encoder.vanilla_backward
                else:
                    executing_fn = EfficientMViTRevBackProp.apply

                x = executing_fn(
                    x, self.encode_blocks[substack[1][0] : substack[1][-1] + 1]
                )
        
        sample, x = self.pre_quant_sample(x)
        return sample, x
