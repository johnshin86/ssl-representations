import torch.nn as nn


def Projector(args, embedding):
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