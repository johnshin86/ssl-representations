import torch
from torch import nn

from utils import off_diagonal
from projector import Projector

import timm


class BarlowTwins(nn.Module):
    r"""An implementation of the Barlow Twins SSL method. 
    The Barlow Twins method computes the cross-correlation
    matrix between a batch of views. The optimization objective
    is to have the cross-correlation matrix be the identity matrix.
    This will align the views, while making the rest of the batch
    orthogonal.

    Suppose we have two batches of views (z1, z2) where
    they are both of size batch_size, feature_dim.
    We z-score (standardize) both views over the batch, 
    and compute the cross-correlation matrix.

    C = 1/batch_size * bn(z1).T @ bn(z2)

    The on-diagonal terms must minimize:

    \sum_i (C_ii - 1)^2

    While the off-diagonal terms must minimize:

    \sum_{i \neq j} (C_ij)^2

    The off-diagonal term is down-weighted in comparison
    to the on-diagonal term:

    on_diag + \lambda * off_diag

    Parameters
    ----------
    y1: torch.Tensor

        Batch of intermediate representations, which have been randomly augmented at the input level.

    y2: torch.Tensor

        Second batch of intermediate representations, which have been randomly augmented at the input level.

    Returns
    -------
    loss: torch.Tensor
        Scalar loss value. 
    """ 
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = int(args.mlp.split("-")[-1])

        if "resnet" in args.arch:
            model = timm.create_model(args.arch, zero_init_last=True)
        else:
            model = timm.create_model(args.arch)
        
        self.rep_dim = model.fc.in_features
        model.fc = nn.Identity()
        self.backbone = model
        self.projector = Projector(args, self.rep_dim)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(self.embedding_dim, affine=False)

    def forward(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        # This works as it is over feature dims
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss

