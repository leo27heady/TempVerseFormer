import torch
import torch.nn as nn
from einops import rearrange

from ..rev_back_prop import NotReversibleModule


class ResNetDownBlock(NotReversibleModule):
    r"""
    Down conv block
    Sequence of following block
    1. Resnet block
    2. Downsample
    """

    def __init__(
        self, 
        in_channels, out_channels, 
        norm_channels=8,
        num_layers=1, 
        fuse_res_paths=True, 
        duplicate_output=True,
        custom_backward=True,
        enable_amp=False
    ):

        super().__init__()

        self.num_layers = num_layers
        self.fuse_res_paths = fuse_res_paths
        self.duplicate_output = duplicate_output
        self.custom_backward = custom_backward
        self.enable_amp = enable_amp

        if self.fuse_res_paths:
            self.res_paths_fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.GELU(),
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels,
                    kernel_size=3, stride=1, padding=1
                ),
            )
            for i in range(num_layers)
        ])
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.GELU(),
                nn.Conv2d(
                    out_channels, out_channels,
                    kernel_size=3, stride=1, padding=1
                ),
            )
            for _ in range(num_layers)
        ])

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            self.seed_cuda("downsample")

            x = rearrange(x, "B H W C -> B C H W")
            
            if self.fuse_res_paths:
                x = self.res_paths_fusion(x)
            
            out = x
            for i in range(self.num_layers):
                # Resnet block
                resnet_input = out
                out = self.resnet_conv_first[i](out)
                out = self.resnet_conv_second[i](out)
                out = out + self.residual_input_conv[i](resnet_input)

            # Downsample
            out = self.down_sample_conv(out)
            
            out = rearrange(out, "B C H W -> B H W C")
            if self.duplicate_output:
                out = torch.cat([out, out], dim=-1)
        return out

    def forward_for_backward(self, x):
        return self.forward(x)


class ResNetUpBlock(NotReversibleModule):
    r"""
    Up conv block.
    Sequence of following blocks:
    1. Upsample
    2. Concatenate Down block output
    """

    def __init__(
        self, 
        in_channels, out_channels, 
        norm_channels=8,
        num_layers=1, 
        fuse_res_paths=True, 
        duplicate_output=True,
        custom_backward=True,
        enable_amp=False
    ):

        super().__init__()

        self.num_layers = num_layers
        self.fuse_res_paths = fuse_res_paths
        self.duplicate_output = duplicate_output
        self.custom_backward = custom_backward
        self.enable_amp = enable_amp

        if self.fuse_res_paths:
            self.res_paths_fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1),
            )
            for i in range(num_layers)
        ])
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            for _ in range(num_layers)
        ])

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            self.seed_cuda("upsample")
            x = rearrange(x, "B H W C -> B C H W")
            
            if self.fuse_res_paths:
                x = self.res_paths_fusion(x)

            # Upsample
            x = self.up_sample_conv(x)

            out = x
            for i in range(self.num_layers):
                # Resnet Block
                resnet_input = out
                out = self.resnet_conv_first[i](out)
                out = self.resnet_conv_second[i](out)
                out = out + self.residual_input_conv[i](resnet_input)
            
            out = rearrange(out, "B C H W -> B H W C")
            if self.duplicate_output:
                out = torch.cat([out, out], dim=-1)
        return out

    def forward_for_backward(self, x):
        return self.forward(x)
