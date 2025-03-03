import os
import sys
import torch
import math
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import CyclicLR

import wandb
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import cv2
from rembg import remove
from matplotlib.colors import ListedColormap
import random

from utils import *
from models import *
from img_prep import *
# from clip_utils import *


def gumbel_softmax_sample(logits, temperature):
    """
    Draw a sample from the Gumbel-Softmax distribution.

    Parameters:
    - logits: Tensor of logits (unnormalized log probabilities)
    - temperature: Temperature parameter for controlling smoothness

    Returns:
    - A Tensor sampled from the Gumbel-Softmax distribution
    """
    #* Sample from the Gumbel distribution
    gumbel_noise = -torch.empty_like(logits).exponential_().log()
    logits = logits + gumbel_noise

    #* Apply softmax
    return F.softmax(logits / temperature, dim=1)

def gumbel_softmax(logits, temperature = 0.1, hard=True):
    """
    Gumbel-Softmax function with optional hard sampling.

    Parameters:
    - logits: Tensor of logits (unnormalized log probabilities)
    - temperature: Temperature parameter for controlling smoothness
    - hard: If True, perform hard sampling (returns one-hot vector)

    Returns:
    - A Tensor sampled from the Gumbel-Softmax distribution
    """
    y_soft = gumbel_softmax_sample(logits, temperature)

    if hard:
        #* Get the indices of the maximum value
        y_hard = torch.zeros_like(y_soft)
        y_hard.scatter_(1, y_soft.argmax(dim=1, keepdim=True), 1.0)
        #* Straight-through estimator
        return y_hard - y_soft.detach() + y_soft, y_soft
    else:
        return y_soft

def get_self_entropy_loss(output):
    #* Apply softmax to get the class probabilities for each pixel
    probs = F.softmax(output, dim=1) 
    #* Compute the pixel-wise entropy: -sum(p * log(p)) for each class
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)  #* Shape: (batch_size, width, height)
    #* Take the mean entropy across all pixels and batches
    self_entropy_loss = entropy.mean()
    return entropy, self_entropy_loss

def clean_shadow_mask(binary_mask, min_area=50):
    binary_mask = binary_mask.cpu().squeeze().numpy().astype(np.uint8)
    binary_mask = (binary_mask > 0).astype(np.uint8)  # Ensure binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == i] = 1
    # closed_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    return cleaned_mask

def plot_components_mask(labeled_mask):
    num_classes = labeled_mask.max() + 1  # Number of labels including 0
    cmap = plt.cm.get_cmap("tab20", num_classes)  # Choose a base colormap
    colors = cmap(np.arange(num_classes))        # Get colors
    
    # Set color for label 0 to black
    colors[0] = [0, 0, 0, 1]  # RGBA for black
    
    # Shuffle the remaining colors (excluding the black color for label 0)
    colors_to_shuffle = colors[1:]  # Exclude the first color (black)
    np.random.shuffle(colors_to_shuffle)  # Shuffle in place
    colors[1:] = colors_to_shuffle  # Replace with shuffled colors

    # Create a custom colormap with the shuffled colors
    custom_cmap = ListedColormap(colors)

    # Plot using the custom colormap
    fig, ax = plt.subplots(figsize=(10, 10))  # Adjust figure size for large images
    plt.imshow(labeled_mask, cmap=custom_cmap)
    plt.title("Labeled Mask with Zeros as Black")
    plt.axis('off')  # Turn off axis for better visualization
    # plt.show()
    return fig

# def get_normalized_weights(alpha, beta):
#     # alpha, beta are shape [1], 1D tensors
#     weights = torch.softmax(torch.cat([alpha, beta], dim=0), dim=0)
#     return weights[0], weights[1]  # These are ephemeral (non-Parameter) values

def get_normalized_weights(a, b, c, d, e):
    # alpha, beta are shape [1], 1D tensors
    weights = torch.softmax(torch.cat([a, b, c, d, e], dim=0), dim=0)
    return weights[0], weights[1], weights[2], weights[3], weights[4]  # These are ephemeral (non-Parameter) values

def get_fur_bg_logits(input_image, rgb_logits):
    centroid_output = rgb_logits.permute(0, 3, 4, 1, 2)  # [batch, H', W', 2, 3]

    # Compute intensity of each color (sum of RGB channels)
    color_intensity = centroid_output.sum(dim=-1)  # Shape: [batch, H', W', 2]

    # Sort by intensity to ensure consistency (lower intensity → fur)
    sorted_indices = color_intensity.argsort(dim=-1)  # Shape: [batch, H', W', 2]
    sorted_centroids = torch.gather(centroid_output, dim=3, index=sorted_indices.unsqueeze(-1).expand_as(centroid_output))

    # Assign consistently:
    fur_color = sorted_centroids[:, :, :, 0, :]  # Always the darker one
    bg_color = sorted_centroids[:, :, :, 1, :]  # Always the lighter one

    #* Ensure `fur_color` and `bg_color` have shape [batch, channels, H, W] before upsampling
    fur_color = fur_color.permute(0, 3, 1, 2)  # Shape: [1, 3, 256, 256]
    bg_color = bg_color.permute(0, 3, 1, 2)  # Shape: [1, 3, 256, 256] 

    #* Upsample to match input resolution (1024x1024)
    fur_color_upsampled = F.interpolate(fur_color, size=(1024, 1024), mode='nearest')  # [1, 3, 1024, 1024]
    bg_color_upsampled = F.interpolate(bg_color, size=(1024, 1024), mode='nearest')  # [1, 3, 1024, 1024] ✅ Now correct

    #* Compute pixel-wise distances
    pixel_dist_fur = torch.norm(input_image - fur_color_upsampled, dim=1)  # [1, 1024, 1024]
    pixel_dist_bg = torch.norm(input_image - bg_color_upsampled, dim=1)  # [1, 1024, 1024]

    fur_confidence = torch.exp(-pixel_dist_fur)
    bg_confidence = torch.exp(-pixel_dist_bg)

    # Normalize confidence (soft selection)
    total_confidence = fur_confidence + bg_confidence
    fur_weight = fur_confidence / total_confidence
    bg_weight = bg_confidence / total_confidence

    return fur_weight, bg_weight

def get_combined(input_image, class_mask_pixels_1, class_mask_pixels_2):
    # 1) Compute pixel-wise distances
    pixel_dist_1 = torch.norm(input_image - class_mask_pixels, dim=1)   # [1, H, W]
    pixel_dist_2 = torch.norm(input_image - class_mask_pixels_2, dim=1) # [1, H, W]

    # 2) Convert distances to confidence (larger distance => lower confidence)
    confidence_1 = torch.exp(-pixel_dist_1)  # [1, H, W]
    confidence_2 = torch.exp(-pixel_dist_2)  # [1, H, W]
    confidence_sum = confidence_1 + confidence_2

    # Avoid division by zero
    confidence_sum = torch.clamp(confidence_sum, min=1e-8)

    # 3) Compute normalized weights
    weight_1 = confidence_1 / confidence_sum
    weight_2 = confidence_2 / confidence_sum

    # 4) Expand for broadcasting over RGB dimension
    weight_1_expanded = weight_1.unsqueeze(1)  # [1, 1, H, W]
    weight_2_expanded = weight_2.unsqueeze(1)  # [1, 1, H, W]

    # 5) Blend the reconstructions
    final_reconstruction = (
        weight_1_expanded * class_mask_pixels 
    + weight_2_expanded * class_mask_pixels_2
    )
    return final_reconstruction

wandb.init(project="Pattern-Extract-local", config={
    "learning_rate": 0.001,
    "batch_size": 1,
    "optimizer": "AdamW",
    "epochs": 2000,
    "seed": 41,
    "num_classes": 2,
    "kernal_size": 11
    # Add more hyperparameters as needed
})
config = wandb.config
gpu_index = 0
set_seed(config.seed)
device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")

learning_rate = config.learning_rate
k_s = config.kernal_size

# input_image = torch.randint(0, 256, (1, 3, 512, 512), dtype = torch.float).to(device) 
# image_o = Image.open('simple2.png')
# image_o = Image.open('11.png')
# image_o = Image.open('363.png')
# image_o = Image.open('294.png')
# image_o = Image.open('tiger.jpg')
# image_o = Image.open('tiger_2.jpg')
# image_o = Image.open('stoat3.jpg')
# image_o = Image.open('tiger_bg.png')
# image_o = Image.open('turtle.jpg')
# image_o = Image.open('star.jpg')
# image_o = Image.open('cat.jpg')
image_o = Image.open('cow1_2.png')
image_o = Image.open('cat_3.JPG')


# image_o = Image.open('nyala_3.jpg')
# image_o = Image.open('giraf.jpg')
image_o = Image.open('tiger_3.jpg')
# image_o = Image.open('panda.jpg')
# wandb.log({"original_image": wandb.Image(image_o)})


# image_o = image_o.convert('HSV')
pxl_size = (1024, 1024)
preprocess = transform_prep_rgb(pxl_size)
bg_mask = get_bg_mask(image_o, pxl_size)
image_tensor = preprocess(image_o)
processed_img = plt_processed_img(image_tensor)
wandb.log({"original_image": wandb.Image(processed_img)})

input_image = image_tensor.unsqueeze(0).to(device)
bg_mask = bg_mask.unsqueeze(0).to(device)


local_size = 4
# local_size = 16
factor = 2
unet_model = UNetColorCentroidLocalall(config.num_classes, local_size = local_size, image_size = pxl_size).to(device)  
# clip_model, preprocess_clip = load_clip_to_device(device, model_name= "ViT-B/32")
w_0 = nn.Parameter(torch.ones(1, device=device)) 
w_1 = nn.Parameter(torch.ones(1, device=device))
w_2 = nn.Parameter(torch.ones(1, device=device)) 
w_3 = nn.Parameter(torch.ones(1, device=device))
w_4 = nn.Parameter(torch.ones(1, device=device)) 

optimizer = get_optimizer(
    list(unet_model.parameters()) + [w_0, w_1, w_2, w_3, w_4], 
    config.optimizer, 
    lr=config.learning_rate, 
)
# optimizer = torch.optim.Adam([alpha, beta], lr=1e-3)
kernal_filters = [8, 16, 32, 64, 128]

for epoch in tqdm(range(config.epochs)):
    unet_model.train() 
    optimizer.zero_grad()

    pre_masked_logits, centroid_rgb_logits,  centroid_rgb_logits_1,  centroid_rgb_logits_2, centroid_rgb_logits_3, centroid_rgb_logits_4 = unet_model(input_image)

    # #* get self entropy loss
    logits = pre_masked_logits * bg_mask.unsqueeze(1)
    entropy, self_entropy_loss = get_self_entropy_loss(logits)

    #* Sample from the Gumbel-Softmax distribution and get 1-hot encoding of the labels
    pre_masked_gumbel_softmax_hard, pre_masked_gumbel_softmax_soft = gumbel_softmax(pre_masked_logits, hard=True)
    gumbel_softmax_hard = pre_masked_gumbel_softmax_hard * bg_mask.unsqueeze(1)
    gumbel_mask_expanded = gumbel_softmax_hard.unsqueeze(2)  # Shape: [1, 2, 1, 512, 512]
    
    centroid_rgb_upsampled = centroid_rgb_logits.repeat_interleave(local_size, dim=3).repeat_interleave(local_size, dim=4) 
    class_mask_pixels = (gumbel_mask_expanded * centroid_rgb_upsampled).sum(dim=1)
    class_mask_pixels_masked = class_mask_pixels * bg_mask.unsqueeze(1)
    color_sim_loss = (torch.norm(input_image - class_mask_pixels, p=2, dim=1) * bg_mask.unsqueeze(1)).mean()
    
    centroid_rgb_upsampled_1 = centroid_rgb_logits_1.repeat_interleave(int(kernal_filters[1]), dim=3).repeat_interleave(int(kernal_filters[1]), dim=4) 
    class_mask_pixels_1 = (gumbel_mask_expanded * centroid_rgb_upsampled_1).sum(dim=1)
    class_mask_pixels_masked_1 = class_mask_pixels_1 * bg_mask.unsqueeze(1)
    color_sim_loss_1 = (torch.norm(input_image - class_mask_pixels_1, p=2, dim=1) * bg_mask.unsqueeze(1)).mean()
    
    centroid_rgb_upsampled_2 = centroid_rgb_logits_2.repeat_interleave(int(kernal_filters[2]), dim=3).repeat_interleave(int(kernal_filters[2]), dim=4) 
    class_mask_pixels_2 = (gumbel_mask_expanded * centroid_rgb_upsampled_2).sum(dim=1)
    class_mask_pixels_masked_2 = class_mask_pixels_2 * bg_mask.unsqueeze(1)
    color_sim_loss_2 = (torch.norm(input_image - class_mask_pixels_2, p=2, dim=1) * bg_mask.unsqueeze(1)).mean()
    
    centroid_rgb_upsampled_3 = centroid_rgb_logits_3.repeat_interleave(int(kernal_filters[3]), dim=3).repeat_interleave(int(kernal_filters[3]), dim=4) 
    class_mask_pixels_3 = (gumbel_mask_expanded * centroid_rgb_upsampled_3).sum(dim=1)
    class_mask_pixels_masked_3 = class_mask_pixels_3 * bg_mask.unsqueeze(1)
    color_sim_loss_3 = (torch.norm(input_image - class_mask_pixels_3, p=2, dim=1) * bg_mask.unsqueeze(1)).mean()
    
    centroid_rgb_upsampled_4 = centroid_rgb_logits_4.repeat_interleave(int(kernal_filters[4]), dim=3).repeat_interleave(int(kernal_filters[4]), dim=4) 
    class_mask_pixels_4 = (gumbel_mask_expanded * centroid_rgb_upsampled_4).sum(dim=1)
    class_mask_pixels_masked_4 = class_mask_pixels_4 * bg_mask.unsqueeze(1)
    color_sim_loss_4 = (torch.norm(input_image - class_mask_pixels_4, p=2, dim=1) * bg_mask.unsqueeze(1)).mean()
    
    #! TO Finish with added layers
    w_f0, w_f1, w_f2, w_f3, w_f4 = get_normalized_weights(w_0, w_1, w_2, w_3, w_4)

    fur_weights_smol, bg_weights_smol = get_fur_bg_logits(input_image, centroid_rgb_logits)
    fur_weights_big, bg_weights_big = get_fur_bg_logits(input_image, centroid_rgb_logits_1)

    # fur_weights_1, bg_weights_1 = get_fur_bg_logits(input_image, centroid_rgb_logits)
    # fur_weights_2, bg_weights_2 = get_fur_bg_logits(input_image, centroid_rgb_logits_1)
    # fur_weights_3, bg_weights_3 = get_fur_bg_logits(input_image, centroid_rgb_logits_2)
    # fur_weights_4, bg_weights_4 = get_fur_bg_logits(input_image, centroid_rgb_logits_3)
    # fur_weights_5, bg_weights_5 = get_fur_bg_logits(input_image, centroid_rgb_logits_4)

    fur_conf_final = 0 * fur_weights_smol + 1 * fur_weights_big  # [1, H, W]
    bg_conf_final  = 0 * bg_weights_smol + 1 * bg_weights_big    # [1, H, W]

    fur_conf_final = fur_weights_smol
    bg_conf_final  = bg_weights_smol

    # fur_conf_final = w_f0 * fur_weights_1 + w_f1 * fur_weights_2 + w_f2 * fur_weights_3 + w_f3 * fur_weights_4 + w_f4 * fur_weights_5
    # bg_conf_final  = w_f0 * bg_weights_1 + w_f1 * bg_weights_2 + w_f2 * bg_weights_3 + w_f3 * bg_weights_4 + w_f4 * bg_weights_5

    # fur_conf_final = fur_weights_1
    # bg_conf_final  = bg_weights_1

    fur_conf_final_exp = fur_conf_final.unsqueeze(1).repeat(1, 3, 1, 1)             # [1, 3, 1024, 1024]
    bg_conf_final_exp = bg_conf_final.unsqueeze(1).repeat(1, 3, 1, 1)   

    fur_conf_final_exp_ls = (torch.norm(input_image - fur_conf_final_exp, p=2, dim=1) * bg_mask.unsqueeze(1)).mean()
    bg_conf_final_exp_ls = (torch.norm(input_image - bg_conf_final_exp, p=2, dim=1) * bg_mask.unsqueeze(1)).mean()
    
    final_recon_loss = torch.min(
        fur_conf_final_exp_ls,
        bg_conf_final_exp_ls
    )

    # total_loss = color_sim_loss + color_sim_loss_1 + color_sim_loss_2 + color_sim_loss_3 + color_sim_loss_4 + final_recon_loss + 0.1 * self_entropy_loss
    # total_loss = color_sim_loss + color_sim_loss_1 + color_sim_loss_2 + color_sim_loss_3 + color_sim_loss_4 
    total_loss = color_sim_loss + color_sim_loss_1
    # total_loss = color_sim_loss
    # total_loss = color_sim_loss_1
    total_loss.backward()
    optimizer.step()


    wandb.log({
        "total_loss": total_loss.item(),
        "batch_c_sim_loss": color_sim_loss.item(),
        "batch_c_sim_loss_2": color_sim_loss_2.item(),
        "final_recon_loss": final_recon_loss.item(),
        "w_f0": w_f0.item(),
        "w_f1": w_f1.item(),
        "w_f2": w_f2.item(),
        "w_f3": w_f3.item(),
        "w_f4": w_f4.item(),
    })

    if (epoch + 1) % 200 == 0:
        fig = plot_recon_mask(class_mask_pixels_masked[0], epoch, color_sim_loss, save = False)
        wandb.log({"noisy_masks": wandb.Image(fig)})
        plt.close(fig)

        fig = plot_recon_mask(class_mask_pixels_masked_1[0], epoch, color_sim_loss_1, save = False)
        wandb.log({"noisy_masks_1": wandb.Image(fig)})
        plt.close(fig)

        fig = plot_recon_mask(class_mask_pixels_masked_2[0], epoch, color_sim_loss_2, save = False)
        wandb.log({"noisy_masks_2": wandb.Image(fig)})
        plt.close(fig)

        fig = plot_recon_mask(class_mask_pixels_masked_3[0], epoch, color_sim_loss_3, save = False)
        wandb.log({"noisy_masks_3": wandb.Image(fig)})
        plt.close(fig)

        fig = plot_recon_mask(class_mask_pixels_masked_4[0], epoch, color_sim_loss_4, save = False)
        wandb.log({"noisy_masks_4": wandb.Image(fig)})
        plt.close(fig)

        eps = 1e-8    
        total_conf = fur_conf_final + bg_conf_final + eps
        fur = fur_conf_final / total_conf
        bg  = bg_conf_final / total_conf
        binary_fur_mask = (fur > 0.51).float() * bg_mask
        fig = plot_recon_norm(binary_fur_mask, epoch, final_recon_loss, save = False)
        wandb.log({"final": wandb.Image(fig)})
        plt.close(fig)

wandb.finish()














