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
from model import network
from dataset import dataloader
from utils import psnr

device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 50
x_dataset_path = "-- PATH OF HAZY DATA"
y_dataset_path = "-- PATH OF GROUND TRUTH IMAGES"
BATCH_SIZE= 4
lr= 1e-3
HEIGHT, WIDTH = 300,400

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:26])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, input_image1, input_image2):
        feature_representation1 = self.feature_extractor(input_image1)
        feature_representation2 = self.feature_extractor(input_image2)
        return nn.MSELoss()(feature_representation1, feature_representation2)
    
class VGG_L1_loss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.vgg_loss = VGGLoss()
        self.l1_loss = nn.L1Loss()
        self.beta = beta

    def forward(self, input_image1, input_image2):
        l2 = self.vgg_loss(input_image1, input_image2)
        l1 = self.l1_loss(input_image1, input_image2)
        return l1 + self.beta * l2
    
d = network().to(device)
optimizer_d = optim.Adam(d.parameters(), lr=lr, betas=(0.9, 0.999))
loss_fn = VGG_L1_loss().to(device)


num_of_training_images = len(os.listdir(x_dataset_path))
for epoch in range(EPOCHS):
    for i in range(0, num_of_training_images, BATCH_SIZE):
        batch_size = BATCH_SIZE if num_of_training_images - i > BATCH_SIZE else num_of_training_images - i
        X = dataloader(i, batch_size, x_dataset_path,HEIGHT, WIDTH)
        Y = dataloader(i, batch_size, y_dataset_path, HEIGHT, WIDTH)
        X, Y = X.to(device), Y.to(device)
#         print(batch_size)
        Y_cap = d(X, batch_size,HEIGHT,WIDTH)
        optimizer_d.zero_grad()
        loss = loss_fn(Y_cap,Y)
        loss.backward()
        optimizer_d.step()
        _psnr= psnr(Y_cap, Y)
    print(
        f"Epoch [{epoch+1}/{EPOCHS}] Loss: {loss.item()} \n"
        f"Epoch [{epoch+1}/{EPOCHS}] PSNR: {_psnr.item()} \n"
        "################################################ \n"
     )
    
torch.save(d.state_dicts(),'------')