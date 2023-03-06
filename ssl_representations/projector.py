import torch.nn as nn


def Projector(args, representation_dim: int) -> nn.Sequential:
    r"""Current vision SSL methods require the use of a projector function 
    (also referred to as an expander function in some references). The projector
    function is typically a 3-layer MLP with non-linearities in intermediate layers.
    Ablation studies of the architecture of the projector function were performed
    in the Barlow Twins paper (https://arxiv.org/abs/2103.03230).

    It is hypothesized that the use of the projector function aids in downstream performance
    when the SSL task and the downstream task are not aligned (https://arxiv.org/abs/2206.13378).

    Parameters
    ----------
    args: ArgParser

        ArgParser object from the train.py file
    
    representation_dim: int

        Dimension of the input of the projector

    Returns
    -------
    
    projector: nn.Sequential
    
        An nn.Sequential object containing a 3 layer MLP, with intermediate batchnorms and ReLUs.
    """
    mlp_spec = f"{representation_dim}-{args.mlp}"
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