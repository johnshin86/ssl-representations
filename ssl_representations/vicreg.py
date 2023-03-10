import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import FullGatherLayer, off_diagonal
from projector import Projector

import timm



class VICReg(nn.Module):
    r"""This is an implementation of the VICReg SSL method.
    The VICReg method builds off the Barlow Twins decorrelation
    mechanism, yet is distinct in that rather than optimizing the cross-correlation
    between views, the covariance matrix is computed within a batch of views.

    The covariance of a batch of views is independently optimized. The
    off-diagonal terms of the covariance matrix are optimized to be zero.
    Rather than optimizing the on-diagonal terms of the covariance matrix
    to be the identity, the standard deviation is optimized to be near the value
    one using the hinge loss. The VICReg paper claims that this aids training stability.

    Finally, the L2 distance between a pair of views is minimized. Suppose
    we have two batches of views, x, y. Let X and Y be the covariance matrices
    within the batch. We then have:

    1/batch_size * ||x - y||^2_2 
    + coeff_1 * \sum_ii \max(0, 1 - \sqrt{X_ii}) + \max(0, 1 - \sqrt{Y_ii}) 
    + coeff_2 * \sum_{i \neq j} (X_ij)^2 + (Y_ij)^2

    Parameters
    ----------
    x: torch.Tensor

        A batch of intermediate representations that have been randomly augmented at the input level.

    y: 
        A second batch of intermediate representations that have been randomly augmented at the input level. Each index corresponds to the same sample in the first batch.

    Returns
    -------
    loss: torch.Tensor

        A scalar valued loss.
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.embedding_dim
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.embedding_dim)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss

