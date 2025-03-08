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

def binary_mask_gloper(
    image_o, pxl_size = (1024, 1024), 
    kernal_filters = [4, 8, 16, 32, 64, 128], 
    epochs = 1000, 
    learning_rate = 0.001, 
    num_classes = 2, 
    t = 0.5,
    reverse = False,
    optimizer_name = "AdamW"
):
    preprocess = transform_prep_rgb(pxl_size)
    bg_mask = get_bg_mask(image_o, pxl_size)
    image_tensor = preprocess(image_o)
    input_image = image_tensor.unsqueeze(0).to(device)
    bg_mask = bg_mask.unsqueeze(0).to(device)

    binary_masks = []
    # for local_size in tqdm(kernal_filters):
    for local_size in kernal_filters:
        set_seed(41)
        unet_model = UNetColorCentroidLocal(num_classes, local_size = local_size, image_size = pxl_size, kernal_sizes = kernal_filters).to(device)  
        optimizer = get_optimizer(
            list(unet_model.parameters()), 
            optimizer_name, 
            lr=learning_rate, 
        )

        for epoch in range(epochs):
            unet_model.train() 
            optimizer.zero_grad()

            pre_masked_logits, centroid_rgb_logits, centroid_rgb_logits_1 = unet_model(input_image)

            #* Sample from the Gumbel-Softmax distribution and get 1-hot encoding of the labels
            pre_masked_gumbel_softmax_hard, pre_masked_gumbel_softmax_soft = gumbel_softmax(pre_masked_logits, hard=True)
            gumbel_softmax_hard = pre_masked_gumbel_softmax_hard
            gumbel_mask_expanded = gumbel_softmax_hard.unsqueeze(2)  # Shape: [1, 2, 1, 512, 512]
            
            centroid_rgb_upsampled = centroid_rgb_logits.repeat_interleave(local_size, dim=3).repeat_interleave(local_size, dim=4) 
            class_mask_pixels = (gumbel_mask_expanded * centroid_rgb_upsampled).sum(dim=1)
            class_mask_pixels_masked = class_mask_pixels * bg_mask.unsqueeze(1) 
            color_sim_loss = (torch.norm(input_image - class_mask_pixels, p=2, dim=1) * bg_mask.unsqueeze(1)).mean()
            
            centroid_rgb_upsampled_1 = centroid_rgb_logits_1.repeat_interleave(local_size*2, dim=3).repeat_interleave(local_size*2, dim=4) 
            class_mask_pixels_1 = (gumbel_mask_expanded * centroid_rgb_upsampled_1).sum(dim=1)
            class_mask_pixels_masked_1 = class_mask_pixels_1 * bg_mask.unsqueeze(1)
            color_sim_loss_1 = (torch.norm(input_image - class_mask_pixels_1, p=2, dim=1) * bg_mask.unsqueeze(1)).mean()
            
            fur_weights_1, bg_weights_1 = get_fur_bg_logits(input_image, centroid_rgb_logits)

            total_loss = color_sim_loss + color_sim_loss_1
            total_loss.backward()
            optimizer.step()

        eps = 1e-8    
        total_conf_1 = fur_weights_1 + bg_weights_1 + eps
        fur_1 = fur_weights_1 / total_conf_1
        bg_1  = bg_weights_1 / total_conf_1
        binary_fur_mask = (fur_1 > t).float()
        if reverse:
            binary_fur_mask = cv2.bitwise_not(binary_fur_mask)
        binary_masks.append(binary_fur_mask)

    return binary_masks

def process_all_images(base_dir):
    """
    Process all images in each animal folder under `base_dir` using the specified segmentation method.
    
    Parameters:
        base_dir (str): The root directory containing subfolders for different animals.
        method (str): Segmentation method ("kmeans", "watershed", "sam", "clipseg", "dino").
    """
    
    # Valid image extensions
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    animals = ["Cat", "Cows", "Giraffe", "Nyala", "Pandas", "Seal", "SeaStar", "Tiger", "Turtle", "Zebra"]

    # Iterate through all subdirectories (each representing an animal type)
    for animal_folder in os.listdir(base_dir):
        # print(f"Processing: {animal_folder}")
        animal_path = os.path.join(base_dir, animal_folder)

        # Ensure it's a directory
        if not os.path.isdir(animal_path) or animal_folder not in animals:
            continue

        # Define paths for Original images and output masks
        input_dir = os.path.join(animal_path, "Original")
        output_dir = os.path.join(animal_path, method)  # Save masks in method-specific folder
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing images in: {input_dir} using {method}")
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
        
        # Process each image in the "Original" folder
        for filename in tqdm(image_files, desc=f"Processing {animal_folder}", unit="image"):
            # Full path to image
            image_path = os.path.join(input_dir, filename)
            image_o = Image.open(image_path)


            ks = [4, 8, 16, 32, 64, 128, 256]
            t = 0.5
            binary_masks = binary_mask_gloper(image_o, kernal_filters = ks, t = t)

            # Extract base filename
            base_name, _ = os.path.splitext(filename)

            for i, mask in enumerate(binary_masks):
                mask = mask.squeeze().cpu().detach().numpy()
                out_filename = f"{base_name}_{ks[i]}.png"
                out_path = os.path.join(output_dir, out_filename)
                mask_255 = (mask * 255).astype(np.uint8)
                cv.imwrite(out_path, mask_255)


set_seed(41)
device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

base_dir = r""
process_all_images(base_dir, method="gloper")



