# iterativeFGSM_Grad-Cam

> 2023 Fall Trustworthy AI | Final project

This code provides an analysis of the combination of Fast Gradient Sign Method (FGSM) and Grad-CAM on a pre-trained convolutional neural network (CNN) model for image classification. The goal is to visualize the effect of adversarial perturbations on the model's predictions and understand the regions of the input image that contribute to the predictions.

[Proposal Link](https://docs.google.com/presentation/d/1bRs6vrKyGtrYpHoATOGL1PEBbU3CKrGA4j_3ci92NhI/edit#slide=id.g29d0d875cfb_0_5)

## Prerequisites

Make sure you have the following dependencies installed:

Python 3.x
PyTorch
NumPy
OpenCV
Matplotlib
tqdm

```bash
pip install -r requirements.txt
```

## Usage
Download Pre-trained Model:
Download the pre-trained model file (mnist_model.pth) and place it in the same directory as the code.

## Download Dataset:
Prepare the dataset file (71_data.json) and place it in the 'data' directory within the same directory as the code.

## Run the Code:
Execute the code by running the script. This can be done using the following command:

```bash
python main.py
```

## View Analysis Figure:
After running the code, an analysis figure (analysis_fig.jpg) will be generated, showcasing the original images, adversarial images, and Grad-CAM heatmaps. The figure will be saved in the same directory as the code.

## Parameters
* eps_list: List of epsilon values for adversarial perturbations.
* num_iter: Number of iterations for iterative FGSM.
* threshold: Threshold for creating the CAM mask.
* mode: Color mode for CAM mask ('rgb' or 'gray').

### Ref
[www.shcas.net/jsjyup/2022/7](https://www.shcas.net/jsjyup/pdf/2022/7/%E5%9F%BA%E4%BA%8EGrad-CAM%E7%9A%84Mask-FGSM%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%E6%94%BB%E5%87%BB.pdf)

[面向深度學習可解釋性的對抗攻擊算法](http://www.joca.cn/CN/abstract/abstract24691.shtml)
