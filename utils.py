import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2 as cv
import math
import numpy as np



def rotate_to_landscape(img):
    if img.shape[1] < img.shape[0]:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    return img

def downscale_images(img,HEIGHT, WIDTH):
    img = cv.resize(img, (WIDTH, HEIGHT), interpolation=cv.INTER_AREA)
    return img

def get_img_tensor (idx, dataset_path, HEIGHT,WIDTH,channel=None):
    file_name = os.listdir(dataset_path)[idx]
    file_path = os.path.join(dataset_path, file_name)
    img = cv.imread(file_path)
    img = rotate_to_landscape(img)
    img = downscale_images(img)
    b,g,r = cv.split(img)
    
    b = torch.tensor(b, dtype=torch.float).reshape(1,1,HEIGHT,WIDTH)
    g = torch.tensor(g, dtype=torch.float).reshape(1,1,HEIGHT,WIDTH)
    r = torch.tensor(r, dtype=torch.float).reshape(1,1,HEIGHT,WIDTH)
    
    if channel is None:
        img = torch.cat((b,g,r),1)
        img = (img * (1/127.5)) - 1
    elif channel == 0:
        img = (b * (1/127.5)) - 1
    elif channel == 1:
        img = (g * (1/127.5)) - 1
    elif channel == 2:
        img = (r * (1/127.5)) - 1
    
    return img

def img_tensor2np (idx, img_tensor):
    img_tensor = (img_tensor + 1) * 127.5
    _,c,_,_ = img_tensor.shape
    img = img_tensor.cpu().detach().numpy().astype(np.uint8)
    if c == 3:
        b,g,r = img[idx,0,:,:], img[idx,1,:,:], img[idx,2,:,:]
        img = cv.merge((b,g,r))
    elif c == 1:
        img = img[idx,0,:,:]
    return img

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

