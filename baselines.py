import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from rembg import remove
import torch
from PIL import Image
from sklearn.decomposition import PCA
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from img_prep import *


def get_kmeans_masks(image, bg_mask, k = 2, show_plot = False):

    foreground_mask  = bg_mask.astype(dtype = bool)
    foreground_pixels = image[foreground_mask]  # Select non-background pixels
    foreground_pixels = np.float32(foreground_pixels)  # Convert to float32 for k-means

    # Apply k-means clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    ret, label, center = cv.kmeans(foreground_pixels, k, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)

    # Reshape label array back to image shape
    label_image = np.full(bg_mask.shape, -1, dtype=np.int32)
    label_image[foreground_mask] = label.flatten()

    # Generate and plot binary masks
    binary_masks = np.stack([(label_image == i).astype(np.uint8) * bg_mask for i in range(k)], axis=0)
    
    if show_plot:
        plt.figure(figsize=(10, 5))
        for i in range(k):
            plt.subplot(1, k, i + 1)
            plt.imshow(binary_masks[i], cmap='gray')
            plt.title(f'Cluster {i}')
            plt.axis('off')
        plt.show()

    return binary_masks

def get_watershed_masks(image, bg_mask, show_plot=False):
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    # Apply GaussianBlur to smooth the image
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Otsu's thresholding for segmentation
    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Distance transform for foreground estimation
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0) 

    # Convert foreground to uint8 and apply bg_mask
    sure_fg = np.uint8(sure_fg) * bg_mask
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1  # Ensure background is labeled distinctly
    markers[unknown == 255] = 0  # Mark unknown regions as 0

    # Debugging: Visualize markers before watershed
    if show_plot:
        plt.imshow(markers, cmap="jet")
        plt.title("Markers Before Watershed")
        plt.colorbar()
        plt.show()

    # Apply watershed
    markers = cv.watershed(image, markers)

    # Create binary pattern mask
    pattern_mask = np.zeros_like(gray, dtype=np.uint8)
    pattern_mask[markers > 1] = 255  # Extracted patterns
    pattern_mask = pattern_mask * bg_mask  # Apply background mask

    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        axes[0].imshow(sure_fg, cmap="gray")
        axes[0].set_title("Sure Foreground")
        axes[0].axis("off")

        axes[1].imshow(markers, cmap="jet")
        axes[1].set_title("Markers After Watershed")
        axes[1].axis("off")

        axes[2].imshow(pattern_mask, cmap="gray")
        axes[2].set_title("Extracted Pattern Mask")
        axes[2].axis("off")

        plt.show()

    return pattern_mask

def binary_mask_sam(image, bg_mask, mask_generator, show_plot = False):
    binary_mask_3ch = cv.merge([bg_mask] * 3) 
    masked_image = image * binary_mask_3ch
    masks = mask_generator.generate(masked_image)

    # Sort masks by area (descending order)
    sorted_masks = sorted(masks, key=lambda x: np.sum(x["segmentation"]), reverse=True)
    # Extract the largest mask (keep separate)
    largest_mask = sorted_masks[0]["segmentation"]
    second_largest_mask = sorted_masks[1]["segmentation"]

    # Combine all other masks
    combined_mask = np.zeros_like(largest_mask, dtype=np.uint8)

    for mask_data in sorted_masks[1:]:  # Skip the largest mask
        combined_mask = np.logical_or(combined_mask, mask_data["segmentation"])

    # Convert boolean mask to uint8 (0 and 255)
    binary_mask = (combined_mask * 255).astype(np.uint8)
    combined_mask_0 = ~(binary_mask.astype(bool))  * bg_mask
    combined_mask_0 = combined_mask_0.astype(np.uint8) * 255
    combined_mask_1 = binary_mask * bg_mask
    largest_mask = (largest_mask * 255).astype(np.uint8) * bg_mask
    second_largest_mask = (second_largest_mask * 255).astype(np.uint8) * bg_mask
    combined_masks = [combined_mask_0, combined_mask_1, largest_mask, second_largest_mask]

    if show_plot:
        plt.figure(figsize=(8, 8))
        for i, mask in enumerate(combined_masks):
            plt.subplot(1, 3, i + 1)
            plt.imshow(mask, cmap="gray")
            plt.title(f"Combined Mask {i}")
            plt.axis("off")
        plt.show()
        
    return combined_masks

def binary_mask_clipseg(image, prompt, bg_mask, processor, model, show_plot = False):
    # Preprocess the image and text prompt
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the segmentation mask (convert logits to probabilities)
    mask = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

    # Normalize the mask for visualization
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    binary_mask = (mask > 0.3).astype(np.uint8)
    resized_mask = Image.fromarray(binary_mask * 255).resize(image.shape[:2], Image.NEAREST)

    resized_mask = resized_mask * bg_mask

    # Display original image and segmentation mask
    if show_plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(resized_mask, cmap="gray")
        plt.title(f"Segmentation for '{prompt[0]}'")
        plt.axis("off")
        plt.show()
        
    return binary_mask

def binary_mask_dino(image_path, dinov2_model, show_plot = False):
    image_o = Image.open(image_path)
    pxl_size = (1008, 1008)
    bg_mask = get_bg_mask(image_o, (1024,1024))
    preprocess = transform_prep_rgb(pxl_size)
    image_tensor = preprocess(image_o).unsqueeze(0).to(device)
    with torch.no_grad():
        features = dinov2_model.get_intermediate_layers(image_tensor, n=4)  # Get last 4 layers

    # Average across multiple layers for richer high-res features
    features = torch.stack(features).mean(dim=0)  
    b, num_patches, d = features.shape  # (1, 2704, 1536)
    h = w = int(num_patches ** 0.5)  # Assume square patch grid (52x52 for 728x728 input)
    feature_map = features.squeeze(0).view(h, w, d)  # Reshape to (52, 52, 1536)
    h, w, d = feature_map.shape  # feature_map is (52, 52, 1536)
    features_flat = feature_map.reshape(-1, d).cpu().numpy()  # Shape (2704, 1536)
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_flat)  # Shape (2704, 3)
    features_pca = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min())
    pca_image = features_pca.reshape(h, w, 3)
    pca_image_resized = cv.resize(pca_image, (1024,1024), interpolation=cv.INTER_CUBIC)
    bg_mask = bg_mask.cpu().numpy()  # Convert to NumPy if it's a PyTorch tensor
    bg_mask = (bg_mask > 0).astype(np.uint8)  # Convert to binary (0 or 1)
    pca_image_np = (pca_image_resized * 255).astype(np.uint8)  # Convert PCA image to [0,255] format
    # masked_pca_image = pca_image_np * bg_mask[:, :, np.newaxis]  # Apply mask on all RGB channels
    masked_pca_gray = cv.cvtColor(pca_image_np, cv.COLOR_RGB2GRAY)
    _, binary_mask = cv.threshold(masked_pca_gray, 100, 255, cv.THRESH_BINARY)

    binary_mask_0 = binary_mask * bg_mask
    binary_mask_1 = ~(binary_mask.astype(bool))
    binary_mask_1 = binary_mask_1 * bg_mask
    binary_masks = [binary_mask_0, (binary_mask_1 * 255).astype(np.uint8)]

    if show_plot:
        plt.figure(figsize=(8, 8))
        for i, mask in enumerate(binary_masks):
            plt.subplot(1, 2, i + 1)
            plt.imshow(mask, cmap="gray")
            plt.axis("off")
            plt.title(f"Binary Mask {i}")
        plt.show()
                
    return binary_masks

def process_all_images(base_dir, method="kmeans"):
    """
    Process all images in each animal folder under `base_dir` using the specified segmentation method.
    
    Parameters:
        base_dir (str): The root directory containing subfolders for different animals.
        method (str): Segmentation method ("kmeans", "watershed", "sam", "clipseg", "dino").
    """
    
    # Valid image extensions
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if method == "dino":
        dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
    elif method == "clipseg":
        clip_seg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
        processor_clip_seg = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    elif method == "sam":
        model_type = "vit_h"  
        sam_checkpoint = f"sam_checkpoints/sam_vit_h_4b8939.pth"  # Ensure you have the correct checkpoint file
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=64,
            crop_n_layers=2,
        )

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
            
            # Read and resize image
            image = cv.imread(image_path)
            image = cv.resize(image, (1024, 1024), interpolation=cv.INTER_LINEAR)
            
            # Remove background
            bg_removed = remove(image)
            alpha_channel = bg_removed[:, :, 3]
            bg_mask = (alpha_channel != 0).astype(np.uint8)

            # Apply the selected segmentation method
            if method == "kmeans":
                binary_masks = get_kmeans_masks(image, bg_mask, k=2, show_plot=False)

            elif method == "watershed":
                binary_masks = [get_watershed_masks(image, bg_mask, show_plot=False)]

            elif method == "sam":
                binary_masks = binary_mask_sam(image, bg_mask, mask_generator, show_plot=False)

            elif method == "clipseg":
                prompt = [f"Pattern of {animal_folder}"]
                binary_masks = [binary_mask_clipseg(image, prompt, bg_mask, processor_clip_seg, clip_seg, show_plot=False)]
            
            elif method == "dino":
                binary_masks = binary_mask_dino(image_path, dinov2_model, show_plot=False)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Extract base filename
            base_name, _ = os.path.splitext(filename)

            # Save each mask
            for i, mask in enumerate(binary_masks):
                out_filename = f"{base_name}_{i}.png"
                out_path = os.path.join(output_dir, out_filename)

                if method == "watershed" or method == "sam" or method == "dino":
                    cv.imwrite(out_path, mask)
                else:
                    mask_255 = (mask * 255).astype(np.uint8)
                    cv.imwrite(out_path, mask_255)

            # print(f"Processed and saved {method} masks for: {filename} in {output_dir}")

device = "cuda" if torch.cuda.is_available() else "cpu"

base_dir = r""
#* Methods: "kmeans", "watershed", "sam", "clipseg", "dino"
methods = ["kmeans", "watershed", "sam", "clipseg", "dino"]
for method in methods:
    process_all_images(base_dir, method=method)

