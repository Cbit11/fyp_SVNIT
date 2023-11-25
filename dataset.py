import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2 as cv
import math
import numpy as np
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from utils import get_img_tensor


def dataloader (idx, batch_size, dataset_path, HEIGHT, WIDTH,channel=None):
    out = [get_img_tensor(i, dataset_path, channel, HEIGHT, WIDTH) for i in range(idx, idx + batch_size)]
    return torch.cat(out, 0)