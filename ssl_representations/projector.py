import torch.nn as nn


def Projector(args, embedding):
    r"""Current vision SSL methods require the use of a projector function 
    (also referred to as an expander function in some references). The projector
    function is typically a 3-layer MLP with non-linearities in intermediate layers.
    Ablation studies of the architecture of the projector function were performed
    in the Barlow Twins paper (https://arxiv.org/abs/2103.03230).

    It is hypothesized that the use of the projector function aids in downstream performance
    when the SSL task and the downstream task are not aligned (https://arxiv.org/abs/2206.13378).
    """
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    if args.tau:
        layers.append(nn.Linear(f[-2], f[-1] + 1, bias=False))
    else:
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)