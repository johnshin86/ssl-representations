import torch
from torch import nn
import torch.nn.functional as F

from utils import off_diagonal, FullGatherLayer
from projector import Projector

import timm

class MCInfoNCE(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self):
		pass