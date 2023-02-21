import torch
import torch.distributed as dist


def exclude_bias_and_norm(p):
    return p.ndim == 1

def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# Need to study the difference between the usage of this in VICReg and Barlow Twins all_reduce. 

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        #create tensor to copy output to
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        #gather tensors x from all GPUs and write to list.
        dist.all_gather(output, x)
        #return as tuple
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        #stack gradients all gradients
        all_gradients = torch.stack(grads)
        #reduce grads so all GPUs get the final result
        dist.all_reduce(all_gradients)
        #return gradients to each process
        return all_gradients[dist.get_rank()]
