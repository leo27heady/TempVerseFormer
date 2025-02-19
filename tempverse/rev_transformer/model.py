import torch
from torch import nn
from torch.autograd import Function as Function
from einops import rearrange

from .block import ReversibleBlock
from ..rev_back_prop import NotReversibleModule, EfficientRevBackProp
from ..config import ReverseTransformerConfig


class InputProjectionModule(NotReversibleModule):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        input_dim=256,
        embed_dim=768,
        context_size=18,
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
        # self.input_projection = nn.Linear(input_dim*2, embed_dim*2, bias=True)
        self.pos_embeddings = nn.Parameter(
            torch.zeros(1, context_size, embed_dim*2)
        )

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=self.enable_amp):
            self.seed_cuda("projection")
            
            # x = self.input_projection(x)
            x = x + self.pos_embeddings
        return x

    def forward_for_backward(self, x):
        return self.forward(x)


class RevFormer(nn.Module):
    def __init__(
        self,
        config: ReverseTransformerConfig,
        context_size: int,
        enable_amp: bool = False,
        custom_backward: bool = True,
    ):
        super().__init__()

        self.input_dim = config.input_dim
        self.embed_dim = config.embed_dim
        self.n_head = config.n_head
        self.depth = config.depth

        # self.input_projection = nn.Linear(self.input_dim, self.embed_dim, bias=True)

        # Reversible blocks can be treated same as vanilla blocks,
        # any special treatment needed for reversible backpropagation
        # is constrained inside the block code and not exposed.
        self.layers = nn.ModuleList()

        self.layers.append(InputProjectionModule(
            input_dim=self.input_dim,
            embed_dim=self.embed_dim,
            context_size=context_size,
            enable_amp=enable_amp
        ))

        for _ in range(self.depth):
            self.layers.append(ReversibleBlock(
                dim=self.embed_dim,
                num_heads=self.n_head,
                enable_amp=enable_amp,
            ))

        # Boolean to switch between vanilla backprop and rev backprop
        self.custom_backward = custom_backward

        # self.norm = nn.LayerNorm(2 * self.embed_dim)
        # self.term_fusion = nn.Sequential(
        #     nn.Linear(2 * self.embed_dim, 2 * self.embed_dim, bias=True),
        #     nn.Linear(2 * self.embed_dim, self.input_dim, bias=True)
        # )

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
        
        x = rearrange(
            x, 'b n c (nh ph) (nw pw) -> b (n nh nw) (ph pw c)',
            ph=1,
            pw=1
        )

        # no need for custom backprop in eval/inference phase
        if not self.training or not self.custom_backward:
            executing_fn = RevFormer.vanilla_backward
        else:
            executing_fn = EfficientRevBackProp.apply

        # This takes care of switching between vanilla backprop and rev backprop
        if t > 0: x = executing_fn(x, t, self.layers)

        # termination fusion
        # x = self.term_fusion(self.norm(x))
        
        x = rearrange(
            x, 'b (n nh nw) (ph pw c) -> b n c (nh ph) (nw pw)',
            ph=1,
            pw=1,
            nw=1,
            nh=1
        )

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
