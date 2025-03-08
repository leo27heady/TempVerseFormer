import torch
from torch import nn
from torch.autograd import Function as Function
from einops import rearrange
from timm.models.layers import trunc_normal_

from tempverse.rev_back_prop import NotReversibleModule, EfficientRevBackProp
from tempverse.config import ReverseTransformerConfig, GradientCalculationWays
from tempverse.utils import BaseLogger
from .block import ReversibleBlock


class InputProjectionModule(NotReversibleModule):
    """
    VAE latent to Transformer Embedding.
    """

    def __init__(
        self,
        input_dim=256,
        embed_dim=768,
        context_size=18,
        custom_backward=True,
        enable_amp=False
    ):
        super().__init__()

        self.custom_backward = custom_backward
        self.enable_amp = enable_amp
        self.in_projection = nn.Linear(input_dim, embed_dim, bias=True)
        self.pos_embeddings = nn.Parameter(
            torch.zeros(1, context_size, embed_dim)
        )

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            self.seed_cuda("projection")
            x = rearrange(
                x, 'b n c (nh ph) (nw pw) -> b (n nh nw) (ph pw c)',
                ph=1,
                pw=1
            )
            x = self.in_projection(x)
            x = x + self.pos_embeddings
            x = torch.cat([x, x], dim=-1)
        return x

    def forward_for_backward(self, x):
        return self.forward(x)


class OutputProjectionModule(NotReversibleModule):
    """
    Transformer Embedding to VAE latent.
    """

    def __init__(
        self,
        input_dim=256,
        embed_dim=768,
        custom_backward=True,
        enable_amp=False
    ):
        super().__init__()

        self.custom_backward = custom_backward
        self.enable_amp = enable_amp

        self.norm = nn.LayerNorm(2 * embed_dim)
        self.term_fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim, bias=True),
            nn.Linear(2 * embed_dim, input_dim, bias=True)
        )

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            self.seed_cuda("projection")
            
            x = self.term_fusion(self.norm(x))
            x = rearrange(
                x, 'b (n nh nw) (ph pw c) -> b n c (nh ph) (nw pw)',
                ph=1,
                pw=1,
                nw=1,
                nh=1
            )
        return x

    def forward_for_backward(self, x):
        return self.forward(x)


class RevFormer(nn.Module):
    def __init__(
        self,
        config: ReverseTransformerConfig,
        context_size: int,
        grad_calc_way: GradientCalculationWays = GradientCalculationWays.REVERSE_CALCULATION,
        enable_amp: bool = False,
    ):
        super().__init__()
        
        self.logger = BaseLogger(__name__)

        self.input_dim = config.input_dim
        self.embed_dim = config.embed_dim
        self.n_head = config.n_head
        self.depth = config.depth

        # Reversible blocks can be treated same as vanilla blocks,
        # any special treatment needed for reversible backpropagation
        # is constrained inside the block code and not exposed.
        self.layers = nn.ModuleList()

        self.input_projection = nn.ModuleList([InputProjectionModule(
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            context_size=context_size,
            custom_backward=grad_calc_way.is_custom_bp_for_not_reversible,
            enable_amp=enable_amp
        )])

        for _ in range(self.depth):
            self.layers.append(ReversibleBlock(
                dim=self.embed_dim,
                num_heads=self.n_head,
                custom_backward=grad_calc_way.is_custom_bp_for_reversible,
                enable_amp=enable_amp,
            ))

        self.output_projection = nn.ModuleList([OutputProjectionModule(
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            custom_backward=grad_calc_way.is_custom_bp_for_not_reversible,
            enable_amp=enable_amp
        )])

        # Switch between vanilla backprop and rev backprop
        self.grad_calc_way = grad_calc_way
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def vanilla_backward(x, t, layers):
        """
        Using rev layers without rev backpropagation. Debugging purposes only.
        Deactivated with self.custom_backward.
        """
        for i in range(t):
            for layer in layers:
                x = layer(x)
        return x

    def forward(self, x, t):
        if t == 0: return x
        
        match self.grad_calc_way:
            case GradientCalculationWays.VANILLA_BP:
                self.logger.debug("Start RevFormer VANILLA_BP")
                x = self.vanilla_backward(x, 1, self.input_projection)
                x = self.vanilla_backward(x, t, self.layers)
                x = self.vanilla_backward(x, 1, self.output_projection)
            case GradientCalculationWays.REVERSE_CALCULATION_FULL:
                self.logger.debug("Start RevFormer REVERSE_CALCULATION_FULL")
                x = EfficientRevBackProp.apply(x, 1, self.input_projection)
                x = EfficientRevBackProp.apply(x, t, self.layers)
                x = EfficientRevBackProp.apply(x, 1, self.output_projection)
            case GradientCalculationWays.REVERSE_CALCULATION:
                self.logger.debug("Start RevFormer REVERSE_CALCULATION")
                x = self.vanilla_backward(x, 1, self.input_projection)
                x = EfficientRevBackProp.apply(x, t, self.layers)
                x = self.vanilla_backward(x, 1, self.output_projection)
        
        return x


if __name__ == "__main__":
    # Test the backprop

    recon_criterion = nn.MSELoss()

    batch_size = 4
    context_size = 16
    model = RevFormer(context_size=context_size)

    # random input, instantiate and fixing.
    # no need for GPU for unit test, runs fine on CPU.
    x = torch.randn((batch_size, context_size, 256, 1, 1))
    t = 8

    # output of the model under reversible backward logic
    output = model(x, t)
    loss = recon_criterion(output, x)

    # computation gradients with reversible backward logic
    # using retain_graph=True to keep the computation graph.
    loss.backward(retain_graph=True)

    # gradient of the input projection layer under custom bwd logic
    rev_grad = model.input_projection.weight.grad.clone()

    # resetting the computation graph
    for param in model.parameters():
        param.grad = None

    # switching model mode to use vanilla backward logic
    model.custom_backward = False

    # computing forward with the same input and model.
    output = model(x, t)
    loss = recon_criterion(output, x)

    # backward but with vanilla logic, does not need retain_graph=True
    loss.backward()

    # looking at the gradient of the input projection layer again
    vanilla_grad = model.input_projection.weight.grad.clone()

    # difference between the two gradients is small enough.
    print("Max difference: ", (rev_grad - vanilla_grad).abs().max())
    print("Min difference: ", (rev_grad - vanilla_grad).abs().min())
    print("Mean difference: ", (rev_grad - vanilla_grad).abs().mean())
    assert (rev_grad - vanilla_grad).abs().max() < 1e-6
