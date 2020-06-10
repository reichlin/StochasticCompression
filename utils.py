import torch
import sys
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import numpy as np


def bpp(mu, symbols, h, w, n, H, W):

    return float(mu * np.log2(symbols) + h * w * np.log2(n)) / float(H * W)
