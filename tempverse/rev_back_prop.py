import torch
from torch.autograd import Function as Function


class RevBackProp(Function):

    """
    Custom Backpropagation function to allow (A) flushing memory in forward
    and (B) activation recomputation reversibly in backward for gradient
    calculation. Inspired by
    https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(
        ctx,
        x,
        t,
        layers,
    ):
        """
        Reversible Forward pass.
        Each reversible layer implements its own forward pass pass logic.
        """

        # obtaining X_1 and X_2 from the concatenated input
        X_1, X_2 = torch.chunk(x, 2, dim=-1)

        for i in range(t):
            for layer in layers:
                X_1, X_2 = layer(X_1, X_2)
                all_tensors = [X_1.detach(), X_2.detach()]

        # saving only the final activations of the last reversible block for the last timestep
        # for backward pass, no intermediate activations are needed.
        ctx.save_for_backward(*all_tensors)
        ctx.t = t
        ctx.layers = layers

        return torch.cat([X_1, X_2], dim=-1)

    @staticmethod
    def backward(ctx, dx):
        """
        Reversible Backward pass.
        Each layer implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """
        # obtaining gradients dX_1 and dX_2 from the concatenated input
        dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)

        # retrieve the last saved activations, to start rev recomputation
        X_1, X_2 = ctx.saved_tensors
        t = ctx.t
        # layer weights
        layers = ctx.layers

        for i in range(t):
            for _, layer in enumerate(layers[::-1]):
                # this is recomputing both the activations and the gradients wrt
                # those activations.
                X_1, X_2, dX_1, dX_2 = layer.backward_pass(
                    Y_1=X_1,
                    Y_2=X_2,
                    dY_1=dX_1,
                    dY_2=dX_2,
                )
        
        # final input gradient to be passed backward to the patchification layer
        dx = torch.cat([dX_1, dX_2], dim=-1)

        del dX_1, dX_2, X_1, X_2

        return dx, None, None, None
