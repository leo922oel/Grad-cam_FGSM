import os 
import sys

import cv2
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

from plot import Comp_Full_Iter, show_iter, show_mask, box


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

def generate_heatmap(fmap, grads, eps=1e-8) -> np.ndarray:
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

def get_heatmap(heatmap_lst, num_block):
    for _ in range(num_block):
        grad_val = gradient_block.pop().cpu().detach().numpy().squeeze()
        fmap = feature_block.pop(0).cpu().detach().numpy().squeeze()
        heatmap_lst.append(generate_heatmap(fmap, grad_val))


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

def calculate_perturbed(img1, img2, mode: str=['mse', 'ssi', 'euc']):
    if mode == "mse":
        loss = F.mse_loss(img1, img2).item()
    elif mode == "ssi":
        img1_np = img1.cpu().squeeze().detach().numpy()
        img2_np = img2.cpu().squeeze().detach().numpy()
        loss, _ = ssim(img1_np, img2_np, full=True, data_range=img1_np.max() - img1_np.min())
        loss = 1 / loss
    elif mode == "euc":
        loss = torch.norm(img1 - img2).item()
    else:
        raise ValueError(f"Mode error: {mode}")

    return loss

def get_perturbed(img, mask, eps):
    data_grad = img.grad.sign()
    perturbed_img = img + eps * data_grad * mask if mask is not None else img + eps * data_grad
    m = eps * data_grad * mask if mask is not None else eps * data_grad
    perturbed_img = torch.clamp(perturbed_img, 0, 1)
    return perturbed_img.detach(), m.detach()

def CAM_mask(heatmap, threshold, mode: str=['rgb', 'gray']):
    if mode == 'rgb':
        r = heatmap[:, :, 0]
        r = (r == 255)
        g = heatmap[:, :, 1]
        g = (g <= threshold)
        b = heatmap[:, :, 2]
        b = (b == 0)
        mask = (r & g & b).astype(np.int8)
    elif mode == 'gray':
        mask = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
        mask = (mask >= threshold).astype(np.int8)
    elif mode == 'only_r':
        r = heatmap[:, :, 0]
        mask = (r >= threshold).astype(np.int8)
    else: ValueError('Mode of CAM mask error.')

    mask = torch.from_numpy(mask).long()
    return mask

def Full_FGSM(dataloader, model, num_block=2, eps=0.00001, noise_mode='mse'):
    correct = 0
    adversarial = 0
    examples = []
    noise_lst = []
    # for idx, (img, target) in tqdm(enumerate(dataloader), ):
    for (img, target) in tqdm(dataloader, desc="ADV"):
        # print("idx:", idx)
        original_img, target = img.to(device), target.to(device)
        original_img = original_img.view(1, 1, 28, -1)
        original_img.requires_grad = True
        heatmap_lst = []

        output = model(original_img)
        _, original_pred = torch.max(output, 1)
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        get_heatmap(heatmap_lst, num_block)
        examples.append((original_pred.item(), original_img.squeeze().detach().cpu().numpy(), heatmap_lst[0], heatmap_lst[1]))
        heatmap_lst.clear()

        # Full FGSM
        full_perturbed_img, m = get_perturbed(original_img, None, eps)
        output = model(full_perturbed_img)
        _, pred = torch.max(output, 1)
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        get_heatmap(heatmap_lst, num_block)
        examples.append((pred.item(), full_perturbed_img.squeeze().detach().cpu().numpy(), heatmap_lst[0], heatmap_lst[1], m))
        heatmap_lst.clear()
        correct += int(original_pred == target)
        
        full_FGSM_noise = calculate_perturbed(original_img, full_perturbed_img, noise_mode)
        if (original_pred == target) and (original_pred != pred):
            noise_lst.append((1-full_FGSM_noise)/2)
            adversarial += 1

    print("adversarial success rate: ", adversarial / correct)
    # asr.append(adversarial / correct)
    
    return noise_lst


def iterative_FGSM(dataloader, model, num_block=2, eps=0.00001, threshold=255, mask_mode='only_r', noise_mode='mse', mask_layer=0, iter_feet=1):
    correct = 0
    adversarial = 0
    examples = []
    iter_lst = []
    noise_lst = []
    asr = []
    for (img, target) in tqdm(dataloader, desc="ADV"):
        original_img, target = img.to(device), target.to(device)
        original_img = original_img.view(1, 1, 28, -1)
        original_img.requires_grad = True
        heatmap_lst = []

        output = model(original_img)
        _, original_pred = torch.max(output, 1)
        correct += int(original_pred == target)
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        get_heatmap(heatmap_lst, num_block)
        examples.append((original_pred.item(), original_img.squeeze().detach().cpu().numpy(), heatmap_lst[0], heatmap_lst[1], np.zeros([28, 28])))
        heatmap_lst.clear()

        # Full FGSM
        full_perturbed_img, m = get_perturbed(original_img, None, eps)
        full_perturbed_img.requires_grad = True
        output = model(full_perturbed_img)
        _, pred = torch.max(output, 1)
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        get_heatmap(heatmap_lst, num_block)
        examples.append((pred.item(), full_perturbed_img.squeeze().detach().cpu().numpy(), heatmap_lst[0], heatmap_lst[1], m.squeeze().detach().cpu().numpy()))
        heatmap_lst.clear()
        
        full_FGSM_noise = calculate_perturbed(original_img, full_perturbed_img, noise_mode)
        # print("Full FGSM noise: ", full_FGSM_noise)

        tmp_noise = 0
        num_iter = -1
        perturbed_img = original_img
        i_noise = []
        while full_FGSM_noise >  tmp_noise and (original_pred == target) and (original_pred != pred) and num_iter <= 100:
            num_iter += 1
            iter_noise = tmp_noise
            heatmap_lst.clear()
            perturbed_img.requires_grad = True
            output = model(perturbed_img)
            _, pred = torch.max(output, 1)
            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            get_heatmap(heatmap_lst, num_block)
            if num_iter > 0:
                i_noise.append(iter_noise)
                examples.append((pred.item(), perturbed_img.squeeze().detach().cpu().numpy(), heatmap_lst[0], heatmap_lst[1], mask.squeeze().detach().cpu().numpy()))

            mask = CAM_mask(heatmap_lst[mask_layer], threshold, mode=mask_mode)
            mask = mask.to(device)
            perturbed_img, m = get_perturbed(perturbed_img, mask, eps/iter_feet)
            tmp_noise = calculate_perturbed(original_img, perturbed_img, noise_mode)

        # print("iterative FGSM noise: ", iter_noise)
        # print("iter: ", num_iter-1)
        iter_lst.append(num_iter)
        if (original_pred == target) and (original_pred != pred):
            noise_lst.append((1-1/i_noise[-1])/2)
            adversarial += 1

    print("adversarial success rate: ", adversarial / correct)
    asr.append(adversarial / correct)
    # print("adversarial success rate:", adv_correct / len(dataloader))

    return examples, iter_lst, noise_lst, asr

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
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    valid_data = datasets.MNIST(
        root='./mnist',
        train=False,
        transform=transform,
        download=True,
    )
    valid_dataloader = DataLoader(
        dataset=valid_data,
        batch_size=1,
        num_workers=4,
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
    eps_list = [0.05, 0.1, 0.15]
    num_block = 2
    threshold = 0
    # mode: 'rgb', 'gray', 'only_r'
    mask_mode = 'only_r'
    mask_layer = 0
    # mode: 'mse', 'ssi', 'euc'
    noise_mode = 'ssi'
    iter_feet = 40

    noise_lst = []
    asr_lst = []
    for eps in eps_list:
        print(f"eps: ", eps)
        # print(f"dataset size: ", len(valid_dataloader))
        # noise = Full_FGSM(valid_dataloader, model, num_block=num_block, eps=eps, noise_mode=noise_mode)
        examples, iter_lst, noise, asr = iterative_FGSM(valid_dataloader, model, num_block=num_block, eps=eps, threshold=threshold, mask_mode=mask_mode, noise_mode=noise_mode, mask_layer=mask_layer, iter_feet=iter_feet)
        noise_lst.append(noise)
        # asr_lst.append(asr)
        # show_mask(examples,)
        # Comp_Full_Iter(examples, eps_list, iter_lst[0])
        # tmp_exp = examples.copy()
        # tmp_exp.pop(1)
        # show_iter(tmp_exp, eps_list)
        # print(f"iter: {np.mean(iter_lst)} \u00B1 {np.std(iter_lst)}")
        print(f"SSIMD: {np.mean(noise)} \u00B1 {np.std(noise)}")
    # box(eps_list, noise_lst, asr_lst)