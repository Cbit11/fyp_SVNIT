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
from dataset import dataloader
from utils import img_tensor2np

d= torch.load("--path of the file--")
device = 'cuda' if torch.cuda.is_available() else 'device'

idx= 9
X = dataloader(idx,1, r"C:\Users\ratho\OneDrive\Desktop\dataset1\Validation\X").to(device)
img = d(X,1)
img_cap = img_tensor2np(0,img)

#plt.imshow(img_cap)
plt.savefig("-- path to file--")