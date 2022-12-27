import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss 


class BarlowTwinsLoss(_Loss):
    r"""The Barlow Twins loss. This is a subclass of the internal loss class used
    for the pytorch non-weighted losses, _Loss. It is intialized in the same way that the internal
    non-weighted losses are, to be consistent with their internal losses. 

    The forward method of the class computes the cross-correlation matrix of the two representations over the batch
    """
    def __init__(self, expander_output: int):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(expander_output, affine = False)
        self.off_diag_coef = off_diag_coef


    def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:

        #create cross-correlation matrix over batch
        c = self.bn(z_a).T @ self.bn(z_b)
        
        # normalize by batch_size
        c.div_(self.args['batch_size'])

        #may need to comment out
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        return on_diag, off_diag

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

