import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from tempverse.config import ReversibleVaeConfig, GradientCalculationWays
from .encoder import Rev_MViT_Encoder
from .decoder import Rev_MViT_Decoder


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
        grad_calc_way: GradientCalculationWays = GradientCalculationWays.REVERSE_CALCULATION,
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
        
        self.encoder = Rev_MViT_Encoder(im_channels, img_size, config, grad_calc_way, enable_amp)
        self.decoder = Rev_MViT_Decoder(im_channels, img_size, config, grad_calc_way, enable_amp)

        self.use_fast_backprop = config.fast_backprop

        if self.use_fast_backprop:
            # Initialize streams globally
            global s1, s2
            s1 = torch.cuda.default_stream(device=torch.cuda.current_device())
            # s1 = torch.cuda.Stream(device=torch.cuda.current_device())
            s2 = torch.cuda.Stream(device=torch.cuda.current_device())

        self.grad_calc_way = grad_calc_way
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):  # or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # elif isinstance(m, nn.GroupNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)

    def encode(self, x):
        z, encoder_output = self.encoder(x)
        return z, encoder_output

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x):
        z, encoder_output = self.encode(x)
        out = self.decode(z)
        return out, encoder_output
