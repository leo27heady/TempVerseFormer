import torch
from torch.autograd import Function

from .modules import ReversibleModule, NotReversibleModule


class EfficientRevBackProp(Function):

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
        modules,
    ):
        """
        Reversible Forward pass.
        Each reversible module implements its own forward pass logic.
        """

        assert all([isinstance(m, ReversibleModule) or isinstance(m, NotReversibleModule) for m in modules]), \
            "Only instances of `ReversibleModule` and `NotReversibleModule` can be part of the grad function!"
        
        is_prev_reversible = True
        all_tensors = []
        with torch.no_grad():
            for i in range(t):
                for module in modules:
                    if isinstance(module, ReversibleModule):
                        is_prev_reversible = True
                    elif isinstance(module, NotReversibleModule) and is_prev_reversible:
                        is_prev_reversible = False
                        all_tensors.append(x.detach())

                    x = module(x)
            
            x = x.detach()
            if is_prev_reversible:
                all_tensors.append(x)

        # saving only the final activations of the last reversible block
        # for backward pass, no intermediate activations are needed.
        ctx.save_for_backward(*all_tensors)
        ctx.modules = modules
        ctx.t = t
        return x

    @staticmethod
    def backward(ctx, dx):
        """
        Reversible Backward pass.
        Each module implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """

        # retrieve the saved activations, to start rev recomputation
        saved_tensors = list(ctx.saved_tensors)
        modules = ctx.modules
        t = ctx.t
        
        # if the last layer is not reversible that means the the last saved activation not right at the end
        if isinstance(modules[-1], NotReversibleModule):
            saved_tensors.append(None)

        x = saved_tensors[-1]

        def propagate_non_reversible(x, dx, accumulated_non_reversible):
            del x, saved_tensors[-1]
            x = saved_tensors[-1]
            future_x = x

            with torch.enable_grad():
                start_index, end_index = accumulated_non_reversible[-1], accumulated_non_reversible[0]
                not_reversible_modules: list[NotReversibleModule] = modules[start_index:end_index + 1]
                for module in not_reversible_modules:
                    if future_x.is_leaf: future_x.requires_grad = True
                    future_x = module.forward_for_backward(future_x)

                future_x.backward(dx, retain_graph=True)
                out_dx = torch.autograd.grad(future_x, x, grad_outputs=dx, retain_graph=True)[0]
                future_x.grad = None
                del dx, future_x
            
            accumulated_non_reversible.clear()

            return x, out_dx

        accumulated_non_reversible: list[int] = []
        for i in range(t):
            for index in reversed(range(len(modules))):
                module = modules[index]
                if isinstance(module, NotReversibleModule):
                    accumulated_non_reversible.append(index)
                elif isinstance(module, ReversibleModule):
                    if accumulated_non_reversible:
                        x, dx = propagate_non_reversible(x, dx, accumulated_non_reversible)
                    
                    # this is recomputing both the activations and the gradients wrt those activations.
                    x, dx = module.backward_pass(y=x, dy=dx)
        
            if accumulated_non_reversible:
                x, dx = propagate_non_reversible(x, dx, accumulated_non_reversible)

        del x, saved_tensors[-1]

        return dx, None, None, None
