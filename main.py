import os 
import sys

import cv2
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resumed_model = './mnist_model.pth'
exp_data = './data/71_data.json'

def fixed_seed(seed: int=1000) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_json2array(fname) -> np.ndarray:
    with open(fname, 'r') as f:
        dataset = json.load(f)
    dataset = np.array(dataset)

    return dataset

def get_heatmap(fmap, grads, eps=1e-8) -> np.ndarray:
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    grads = grads.reshape([grads.shape[0], -1])
    weights = np.mean(grads, axis=1)
    for i, w in enumerate(weights):
        cam += w * fmap[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (28, 28))
    cam /= (cam + eps).max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

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

def backward_hook(module, grad_in, grad_out) -> None:
    gradient_block.append(grad_out[0].detach())

def forward_hook(module, input, output) -> None:
    feature_block.append(output)

def get_perturbed(img, mask, eps) -> torch.Tensor:
    data_grad = img.grad.sign()
    perturbed_img = img + eps * data_grad * mask
    # Clip perturbed image to be within epsilon neighborhood of the original image
    # perturbed_img.data = torch.max(torch.min(perturbed_img.data, original_img + epsilon), original_img - epsilon)
    perturbed_img = torch.clamp(perturbed_img, 0, 1)
    return perturbed_img.detach()

def CAM_mask(heatmap, threshold, mode: str=['rgb', 'gray']):
    if mode == 'rgb':
        mask = heatmap[:, :, 0]
    elif mode == 'gray':
        mask = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    else: ValueError('Mode of CAM mask error.')

    mask = (mask >= threshold).astype(np.int8)
    mask = torch.from_numpy(mask).long()
    return mask

def iterative_FGSM(dataloader, model, num_iter=10, eps=0.00001, threshold=225, mode='gray', epsilon=1e-8):
    correct = 0
    examples = []
    for idx, (img, target) in enumerate(dataloader):
        original_img, target = img.to(device), target.to(device)
        original_img = original_img.view(1, 1, 28, -1)
        # original_img.requires_grad = True
        
        # original_output = model(original_img)
        # _, original_pred = torch.max(original_output, 1)
        perturbed_img = original_img
        for i in tqdm(range(num_iter)):
            perturbed_img.requires_grad = True
            output = model(perturbed_img)
            _, pred = torch.max(output, 1)
            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward(retain_graph=True)

            # grad_val = gradient_block[2*i+1].cpu().detach().numpy().squeeze()
            heatmap_lst = []
            grad_val = gradient_block[2*i+1].cpu().detach().numpy().squeeze()
            fmap = feature_block[2*i].cpu().detach().numpy().squeeze()
            heatmap1 = get_heatmap(fmap, grad_val)
            grad_val = gradient_block[2*i].cpu().detach().numpy().squeeze()
            fmap = feature_block[2*i+1].cpu().detach().numpy().squeeze()
            heatmap2 = get_heatmap(fmap, grad_val)
            # if i ==0:
            mask = CAM_mask(heatmap2, threshold, mode=mode)
            mask = mask.to(device)
            examples.append((pred.item(), perturbed_img.squeeze().detach().cpu().numpy(), heatmap1, heatmap2))

            perturbed_img = get_perturbed(perturbed_img, mask, eps)

    return examples

def visualize(examples, eps_list, num_iter, fname='analysis_fig.jpg'):
    print("===== Output analysis figure =====")
    count = 0
    plt.figure(figsize=(20, 11))
    row = len(examples[0]) - 1
    for i , eps in enumerate(eps_list):
        for idx, (pred, perturbed, heatmap1, heatmap2) in enumerate(examples):
            count += 1
            plt.subplot(row, num_iter, count)
            plt.xticks([], [])
            plt.yticks([], [])
            if idx == 0:
                plt.ylabel(f"Original", fontsize=30)
            if idx == 1:
                plt.ylabel(f"Epslion: {eps}", fontsize=30)
            plt.title(f"pred: {pred}", fontsize=40)
            plt.imshow(perturbed, cmap='gray')

            plt.subplot(row, num_iter, count + num_iter)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title(f"")
            plt.imshow(heatmap1, )

            plt.subplot(row, num_iter, count + num_iter + num_iter)
            # heatmap2 = cv2.cvtColor(heatmap2, cv2.COLOR_RGB2GRAY)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.title(f"")
            plt.imshow(heatmap2)

    plt.tight_layout()
    plt.savefig(fname)
    print("===== Finish =====")

if __name__ == '__main__':
    fixed_seed()

    dataset = get_json2array(exp_data)
    dataset = np.array(dataset).reshape(len(dataset), 1, 28, -1)
    dataset = (dataset+1) / 2.0
    sample = torch.from_numpy(dataset[0]).float()
    target = np.array([6])
    target = torch.from_numpy(target).long()
    dataset = TensorDataset(
        sample,
        target
    )
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    # os._exit(0)
    model = Net().to(device)
    model.load_state_dict(torch.load(resumed_model, map_location="cpu"))
    model.eval()

    feature_block = []
    gradient_block = []
    # TODO: other layer
    model.conv1.register_forward_hook(forward_hook)
    model.conv1.register_full_backward_hook(backward_hook)
    model.conv2.register_forward_hook(forward_hook)
    model.conv2.register_full_backward_hook(backward_hook)

    # TODO: Integrating FGSM & Grad CAM
    eps_list = [0.1, ]
    num_iter = 5
    threshold = 100
    mode = 'rgb'
    for eps in eps_list:
        examples = iterative_FGSM(data_loader, model, num_iter=num_iter, eps=eps, threshold=threshold, mode=mode)
        visualize(examples, eps_list, num_iter)