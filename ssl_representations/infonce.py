import torch
from torch import nn

from utils import off_diagonal
from projector import Projector

import timm

class InfoNCE(nn.Module):
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


    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss
