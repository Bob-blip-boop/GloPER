import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as mpatches
from matplotlib import cm
import seaborn as sns
from collections import Counter
import cv2 as cv


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

def plot_og_img(image_o):
    plt.imshow(cv.cvtColor(image_o, cv.COLOR_BGR2RGB)) #* Convert BGR to RGB for correct display
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

def plt_processed_img(image_tensor):
    image_np = image_tensor.cpu().detach()
    image_np = image_np/image_np.max()
    image_pil = TF.to_pil_image(image_np)
    return image_pil

def plot_pattern(pattern):
    masked_images_np = pattern.permute(1, 2, 0).cpu().detach().numpy()
    # masked_images_np = np.clip(masked_images_np, 0, 1)
    plt.imshow(masked_images_np)
    plt.axis('off')
    plt.show()

def plot_binary(pattern):
    masked_images_np = pattern.squeeze(0).cpu().detach().numpy()
    plt.imshow(masked_images_np, cmap='grey')
    plt.axis('off')
    plt.show()

def plot_mask(mask, epoch, ls):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.imshow(mask, cmap='gray')
    ax.axis('off')
    plt.title(f'Epoch {epoch+1} Pattern Loss: {ls:.3f}')
    return fig

def plot_combined_patterns(patterns):
    combined_image = np.zeros_like(patterns[0])
    for i, pattern in enumerate(patterns):
        color = (np.array(matplotlib.colormaps['Set1'].colors) * 255).astype(np.uint8)[1]
        colored_pattern = np.zeros_like(pattern)
        mask = cv.cvtColor(pattern, cv.COLOR_BGR2GRAY) > 0  #* Assuming the pattern is non-zero in areas of interest
        colored_pattern[mask] = color
        #* p = cv.cvtColor(pattern, cv.COLOR_BGR2RGB)
        combined_image += colored_pattern

    plt.imshow(combined_image)
    plt.show()
    return combined_image

def plot_recon_mask(recon_mask, epoch, ls, save = False):
    image_np = recon_mask.permute(1, 2, 0).cpu().detach().numpy()
    image_np = image_np/image_np.max()
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_np)
    ax.axis('off')
    plt.title(f'Epoch {epoch+1} Pattern Loss: {ls:.3f}')
    return fig

def plot_recon_norm(class_mask_pixels_grey_normalized, epoch, ls, save = False):
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.imshow((class_mask_pixels_grey_normalized).squeeze().cpu().detach().numpy(), cmap='gray')
    ax.axis('off')
    plt.title(f'Epoch {epoch+1} Pattern Loss: {ls:.3f}')
    return fig

def get_optimizer(params, optimizer_name, lr, extra_params=None, **kwargs):
    """
    Create an optimizer for the model and optionally include extra parameters.

    Args:
        model (nn.Module): The main model.
        optimizer_name (str): Name of the optimizer (e.g., "SGD", "Adam", "AdamW", "RMSprop").
        lr (float): Learning rate.
        extra_params (iterable, optional): Additional parameters from other modules (e.g., TrainableFilter).
        **kwargs: Additional optimizer-specific arguments (e.g., momentum, weight_decay, betas).

    Returns:
        torch.optim.Optimizer: The selected optimizer.
    """
    # params = list(model.parameters())  # Get model parameters

    # Select optimizer
    if optimizer_name == "SGD":
        return torch.optim.SGD(params, lr=lr, **kwargs)
    elif optimizer_name == "Adam":
        return torch.optim.Adam(params, lr=lr, **kwargs)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, **kwargs)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")



