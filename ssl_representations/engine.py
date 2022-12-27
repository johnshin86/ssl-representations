import math
import sys
import time
import torch
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
from torchvision.transforms import ToTensor

import utils

import cv2
import numpy as np
import pickle

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, loss_dict, loss_coeff, augment_policy, framework: str):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, bounds in metric_logger.log_every(data_loader, print_freq, header):
        
        # TODO: handling of non-vicreg
        images = torch.stack(images)
        images = images.to(device)

        x_a = augment_policy(images)
        x_b = augment_policy(images)

        z_a = model(x_a)
        z_b = model(x_b)

        if framework == "vicreg":

            L_inv = loss_dict["mse"](z_a, z_b)
            L_var = loss_dict["var"](z_a, z_b)
            L_cov = loss_dict["cov"](z_a, z_b)

            losses = loss_coeff["lambda"] * L_inv + loss_coeff["mu"] * L_var + loss_coeff["nu"] * L_cov
            loss_value = losses.item()
            loss_dict_print = {
            "L_inv": L_inv,
            "L_var": L_var,
            "L_cov": L_cov
            }
        else:
            raise ValueError(f"Framework "{framework}" not implemented.")

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_print)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_print)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

    












