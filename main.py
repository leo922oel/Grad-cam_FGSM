import os 
import sys

import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fixed_seed(seed: int=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_heatmap(fmap, grads, prefix=''):
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    grads = grads.reshape([grads.shape[0], -1])
    weights = np.mean(grads, axis=1)
    for i, w in enumerate(weights):
        cam += w * fmap[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (28, 28))
    cam /= cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return heatmap

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def backward_hook(module, grad_in, grad_out):
    gradient_block.append(grad_out[0].detach())

def forward_hook(module, input, output):
    feature_block.append(output)

def iterative_FGSM(img, data_grad, eps=0.00001):
    img = img.to(device)
    pertubed = img + eps * data_grad.sign()
    pertubed = torch.clamp(pertubed, 0, 1)

    return pertubed

def Grad_CAM(model):
    pass

if __name__ == '__main__':
    fixed_seed()

    model = Net().to(device)
    # model.load_state_dict(torch.load('...', map_location="cpu"))
    model.eval()

    feature_block = []
    gradient_block = []

    # TODO: other layer
    model.conv1.register_forward_hook(forward_hook)
    model.conv1.register_full_backward_hook(backward_hook)

    # TODO: Integrating FGSM & Grad CAM