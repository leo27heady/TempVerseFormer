# MViTv2 implementation taken from detectron2
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/mvit.py

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange

from .utils import get_abs_pos
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
        embed_dim=768,
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
        return x

class UnpatchEmbed(nn.Module):
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
        x = self.norm_out(x)
        x = self.non_linearity(x)
        x = self.proj(x)
        return x


class Reversible_MViT_VAE(nn.Module):
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
        
        self.patch_embed = PatchEmbed(
            kernel_size=config.patch_kernel,
            stride=config.patch_stride,
            padding=config.patch_padding,
            im_channels=im_channels,
            embed_dim=config.patch_embed_dim,
        )
        embed_dim = config.patch_embed_dim

        if config.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (img_size // config.patch_stride[0]) * (
                img_size // config.patch_stride[1]
            )
            num_positions = num_patches
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_positions, embed_dim)
            )
        else:
            self.pos_embed = None


        #######################
        ####### ENCODER #######
        #######################

        encoder_depth = config.encoder_stage_size * config.encoder_stages
        encoder_dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, encoder_depth)]  # stochastic depth decay rule
        embed_dim *= 2
        num_heads = config.num_heads
        dim_out = embed_dim
        stride_kv = config.adaptive_kv_stride
        input_size = [img_size // config.patch_stride[0], img_size // config.patch_stride[1]]
        stage = 1
        # self._out_feature_strides = {}
        # self._out_feature_channels = {}
        
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
        
        self.encoder_norm_out = nn.GroupNorm(config.norm_channels, embed_dim)
        self.act_layer = nn.GELU()
        self.encoder_conv_out = nn.Conv2d(embed_dim, 2 * embed_dim, kernel_size=3, padding=1)
        # Latent Dimension is 2*Latent because they are predicting mean & variance
        self.pre_quant_conv = nn.Conv2d(2 * embed_dim, 2 * embed_dim, kernel_size=1)

        
        #############################
        #######    DECODER    #######
        #############################

        self.post_quant_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=(1, 1))

        embed_dim = embed_dim // 2
        dim_out = embed_dim
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
                
        self.unpatch_embed = UnpatchEmbed(
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            norm_channels=config.norm_channels,
            embed_dim=embed_dim*2,
            im_channels=im_channels,
        )


        self.use_fast_backprop = config.fast_backprop

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
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


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

    def sample_output(self, out):
        mean, logvar = torch.chunk(out, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn(mean.shape).to(device=out.device)
        return sample

    def encode(self, x):
        x = self.patch_embed(x)
        x = rearrange(x, "B C H W -> B H W C")

        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, False, x.shape[1:3])
        
        x = torch.cat([x, x], dim=-1)

        # process layers in reversible and irreversible stacks
        stack = self.create_blocks_stack(self.encode_blocks)
        for i, substack in enumerate(stack):
            if substack[0] == "Stage Transition":
                x = self.encode_blocks[substack[1]](x)
            else:
                if not self.training or not self.custom_backward:
                    executing_fn = Reversible_MViT_VAE.vanilla_backward
                else:
                    executing_fn = EfficientMViTRevBackProp.apply

                x = executing_fn(
                    x, self.encode_blocks[substack[1][0] : substack[1][-1] + 1]
                )
        
        out = rearrange(x, "B H W C -> B C H W")

        out = self.encoder_norm_out(out)
        out = self.act_layer(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        
        O1, O2 = torch.chunk(out, 2, dim=1)
        o1_sample = self.sample_output(O1)
        o2_sample = self.sample_output(O2)
        sample = torch.cat([o1_sample, o2_sample], dim=1)

        return sample, out

    def decode(self, z):
        x = rearrange(z, "B C H W -> B H W C")

        stack = self.create_blocks_stack(self.decode_blocks)
        for i, substack in enumerate(stack):
            if substack[0] == "Stage Transition":
                x = self.decode_blocks[substack[1]](x)
            else:
                if not self.training or not self.custom_backward:
                    executing_fn = Reversible_MViT_VAE.vanilla_backward
                else:
                    executing_fn = EfficientMViTRevBackProp.apply

                x = executing_fn(
                    x, self.decode_blocks[substack[1][0] : substack[1][-1] + 1]
                )
        x = rearrange(x, "B H W C -> B C H W")
        x = self.unpatch_embed(x)
        
        return x

    def forward(self, x):
        z, encoder_output = self.encode(x)
        out = self.decode(z)
        return out, encoder_output
