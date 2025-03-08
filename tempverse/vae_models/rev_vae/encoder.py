# Based on the MViTv2 implementation taken from detectron2
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/mvit.py

import torch
import torch.nn as nn
from einops import rearrange

from ..rev_back_prop import ReversibleModule, NotReversibleModule, EfficientRevBackProp
from .res_net_block import ResNetDownBlock
from .rev_block import ReversibleMultiScaleBlock
from ..config import ReversibleVaeConfig, GradientCalculationWays
from ..utils import BaseLogger


class InputEmbed(NotReversibleModule):
    """
    Image RGB to Embedding.
    """

    def __init__(
        self,
        im_channels=3,
        embed_dim=8,
        duplicate_output=True,
        custom_backward=True,
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

        self.duplicate_output = duplicate_output
        self.custom_backward = custom_backward
        self.enable_amp = enable_amp
        self.conv_in = nn.Conv2d(im_channels, embed_dim, kernel_size=3, padding=(1, 1))

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            self.seed_cuda("patch")
            
            x = self.conv_in(x)
            x = rearrange(x, "B C H W -> B H W C")
            if self.duplicate_output:
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
        in_dim=1024,
        z_channels=256,
        custom_backward=True,
        enable_amp=False
    ):
        super().__init__()

        self.custom_backward = custom_backward
        self.enable_amp = enable_amp

        self.norm_1 = nn.GroupNorm(norm_channels, in_dim)
        self.conv_out = nn.Conv2d(in_dim, z_channels * 2, kernel_size=1)
        self.act_layer = nn.GELU()
        # Latent Dimension is 2*Latent because they are predicting mean & variance
        self.pre_quant_conv = nn.Conv2d(z_channels * 2, z_channels * 2, kernel_size=1)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            self.seed_cuda("pre_quant")
            x = rearrange(x, "B H W C -> B C H W")
            x = self.norm_1(x)
            x = self.act_layer(self.conv_out(x))
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
        custom_backward=True,
        enable_amp=False
    ):
        super().__init__()

        self.custom_backward = custom_backward
        self.enable_amp = enable_amp

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            self.seed_cuda("sample")

            mean, logvar = torch.chunk(x, 2, dim=1)
            std = torch.exp(0.5 * logvar)
            sample = mean + std * torch.randn_like(mean)
        
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
        grad_calc_way: GradientCalculationWays,
        enable_amp: bool = False,
    ):
        super().__init__()
        
        self.logger = BaseLogger(__name__)

        down_channels = list(config.down_channels.copy())
        attn_down = config.attn_down.copy() + [False]  # append False as the last entry always has't path split

        number_of_stages: int = len(down_channels) - 1
        assert number_of_stages > 1  # at least 2 stages
        assert len(attn_down) == len(down_channels)  # channels and attentions lists have same lengths
        assert config.encoder_stage_size >= 1  # if 1 only downsample remains

        self.grad_calc_way = grad_calc_way
        
        self.encode_blocks = nn.ModuleList()
        self.encode_blocks.append(InputEmbed(
            im_channels=im_channels,
            embed_dim=config.down_channels[0],
            duplicate_output=attn_down[0],  # duplicate if 1st stage has attention blocks
            custom_backward=grad_calc_way.is_custom_bp_for_not_reversible,
            enable_amp=enable_amp,
        ))

        # encoder_dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, encoder_depth)]  # stochastic depth decay rule
        num_heads = config.num_heads_min
        input_size = [img_size, img_size]
        # self._out_feature_strides = {}
        # self._out_feature_channels = {}

        for stage in range(number_of_stages):
            for _ in (range(config.encoder_stage_size - 1) if attn_down[stage] else []):
                self.encode_blocks.append(ReversibleMultiScaleBlock(
                    dim=config.down_channels[stage],
                    dim_out=config.down_channels[stage],
                    num_heads=num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    drop_path=0.0,
                    qkv_pool_kernel=(3, 3),
                    stride_q=1,
                    stride_kv=1,
                    window_size=(input_size[0] // 4),  # TODO: hybrid window attention: global attention in last stages.
                    residual_pooling=config.residual_pooling,
                    use_rel_pos=config.use_rel_pos,
                    rel_pos_zero_init=config.rel_pos_zero_init,
                    input_size=input_size,
                    custom_backward=grad_calc_way.is_custom_bp_for_reversible,
                    enable_amp=enable_amp,
                ))

            self.encode_blocks.append(ResNetDownBlock(
                in_channels=config.down_channels[stage], 
                out_channels=config.down_channels[stage + 1], 
                norm_channels=config.norm_channels,
                num_layers=config.down_scale_layers, 
                fuse_res_paths=attn_down[stage],  # fuse if current stage has rev blocks
                duplicate_output=attn_down[stage + 1],  # duplicate if next stage has rev blocks
                custom_backward=grad_calc_way.is_custom_bp_for_not_reversible,
                enable_amp=enable_amp,
            ))
            num_heads = min(num_heads * 2, config.num_heads_max)
            input_size = [s // 2 for s in input_size]
            assert 0 not in input_size

        self.encode_blocks.append(PreQuantModule(
            norm_channels=config.norm_channels, 
            in_dim=config.down_channels[-1], 
            z_channels=config.z_channels, 
            custom_backward=grad_calc_way.is_custom_bp_for_not_reversible,
            enable_amp=enable_amp,
        ))

        self.sampling = nn.ModuleList([SamplingModule(
            custom_backward=grad_calc_way.is_custom_bp_for_not_reversible,
            enable_amp=enable_amp
        )])
    
    def create_blocks_stack(self, blocks) -> list[tuple[bool, list[int]]]:
        stack = []
        for l_i in range(len(blocks)):
            if isinstance(blocks[l_i], ReversibleModule):
                if len(stack) == 0 or not stack[-1][0]:
                    stack.append((True, []))
            else:
                if len(stack) == 0 or stack[-1][0]:
                    stack.append((False, []))
            stack[-1][1].append(l_i)
        return stack

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
        match self.grad_calc_way:
            case GradientCalculationWays.VANILLA_BP:
                self.logger.debug("Start Rev_MViT_Encoder VANILLA_BP")
                x = self.vanilla_backward(x, 1, self.encode_blocks)
                sample = self.vanilla_backward(x, 1, self.sampling)
            case GradientCalculationWays.REVERSE_CALCULATION_FULL:
                self.logger.debug("Start Rev_MViT_Encoder REVERSE_CALCULATION_FULL")
                x.requires_grad_()
                x = EfficientRevBackProp.apply(x, 1, self.encode_blocks)
                sample = EfficientRevBackProp.apply(x, 1, self.sampling)
            case GradientCalculationWays.REVERSE_CALCULATION:
                self.logger.debug("Start Rev_MViT_Encoder REVERSE_CALCULATION")
                stack: list[tuple[bool, list[int]]] = self.create_blocks_stack(self.encode_blocks)
                for is_rev, substack in stack:
                    x = (EfficientRevBackProp.apply if is_rev else self.vanilla_backward)(
                        x, 1, self.encode_blocks[substack[0] : substack[-1] + 1]
                    )
                sample = EfficientRevBackProp.apply(x, 1, self.sampling)
        return sample, x
