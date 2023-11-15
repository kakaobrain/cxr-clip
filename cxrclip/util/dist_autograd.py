import torch
import torch.distributed as dist


def DistAutogradAllGatherFunction(partial=False):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
            dist.all_gather(output, input)

            return tuple(output)

        @staticmethod
        def backward(ctx, *grads):
            (input,) = ctx.saved_tensors
            grad_out = torch.zeros_like(input)

            if partial:
                grad_out[:] = grads[dist.get_rank()]
            else:
                dist.reduce_scatter(grad_out, list(grads), dist.ReduceOp.SUM)

            return grad_out

    return F
