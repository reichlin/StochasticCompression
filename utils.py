import torch
import sys
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import numpy as np


def bpp(h, w, n, k_values, img_size, bits):

    # number of bitsrequired to encode z masked is the number of non masked values times log_2(L): k_values * bits
    # number of bits required to encode the mask: h * w * np.log2(n)

    bpp = (k_values * bits + h * w * np.log2(n)) / img_size

    return bpp

