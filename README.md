# Grad-cam FSGM

> 2023 Fall Trustworthy AI | Final project

## Adversarial Attacks and Interpretability Analysis  

This code provides an in-depth analysis of the combination of Fast Gradient Sign Method (FGSM) and Grad-CAM on a pre-trained Convolutional Neural Network (CNN) model for image classification. The primary goal is to visualize the impact of adversarial perturbations on the model's predictions and understand the regions of the input image that contribute to those predictions.

[Proposal Link](https://docs.google.com/presentation/d/1bRs6vrKyGtrYpHoATOGL1PEBbU3CKrGA4j_3ci92NhI/edit#slide=id.g29d0d875cfb_0_5)

[Final Project Presentation](https://docs.google.com/presentation/d/13yvNk-oImNNaP5vGvdlAyMPOwiBBXOx253MMO42EWRI/edit?usp=sharing)

## Setup

Make sure to install the required dependencies:

> Python 3.x  
PyTorch  
NumPy  
OpenCV  
Matplotlib  
tqdm  
skimage  

```bash
pip install -r requirements.txt
```

## Usage
Download Pre-trained Model:
Download the pre-trained model file (mnist_model.pth) and place it in the same directory as the code.

## Download Dataset:
Prepare the dataset file (ex. 71_data.json) and place it in the 'data' directory within the same directory as the code.  
Besides, you can use MNIST datasets.
## Run the Code:
Execute the code by running the script. This can be done using the following command:

```bash
python main.py
```

## Parameters
#### eps_list
    List of epsilon values for adversarial perturbations.
#### num_block
    Number of gradient map blocks.
#### num_iter
    Number of iterations for iterative FGSM.
#### threshold
    Threshold for creating the Grad-cam mask.
#### mask_mode
    Color mode for Grad-cam mask ('rgb', 'gray', 'only_r').
#### mask_layer
    Which layer of gradient map block will use to generate mask.
#### noise_mode
    Noise calculation mode for imgaes ('mse', 'ssi', 'euc').
#### feet_lst
    modify eps for iterative FGSM. Note that 1 for Grad-cam FGSM.
#### demo_idx
    Which sample will use to generate figure.
#### do_Full_FGSM
    Whether to execute FGSM.

## References
[www.shcas.net/jsjyup/2022/7](https://www.shcas.net/jsjyup/pdf/2022/7/%E5%9F%BA%E4%BA%8EGrad-CAM%E7%9A%84Mask-FGSM%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%E6%94%BB%E5%87%BB.pdf)

[面向深度學習可解釋性的對抗攻擊算法](http://www.joca.cn/CN/abstract/abstract24691.shtml)
