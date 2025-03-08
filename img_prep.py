import cv2 as cv
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image, ImageEnhance

from rembg import remove

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def convert_image_to_rgb(image):
    return image.convert("RGB")
    # return image..convert('L')

def convert_image_to_cmyk(image):
    return image.convert("CMYK")

def resize_to_divisible(image, divisor=16):
    width, height = image.size
    new_width = (width // divisor) * divisor
    new_height = (height // divisor) * divisor
    return image.resize((new_width, new_height), Image.BICUBIC)

def enhance_contrast(image, factor=3):
    # Increase contrast using PIL's ImageEnhance
    """
    Transforms an image to a tensor of size n_px x n_px

    Args:
        n_px (int): The size of the output image
        divisor (int, optional): The divisor of the output image size. Defaults to 16.

    Returns:
        Compose: A composed transform of resizing to n_px with BICUBIC interpolation,
                 resizing to the nearest multiple of divisor,
                 converting to RGB, and converting to a tensor
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def transform_prep_rgb(n_px, divisor=16, contrast_factor=1.5):
    return Compose([
        # Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        Resize(n_px),
        # lambda image: resize_to_divisible(image, divisor),  # Resize with divisor
        convert_image_to_rgb,  # Convert to RGB
        # lambda image: enhance_contrast(image, factor=contrast_factor),  # Adjust contrast
        ToTensor(),  # Convert to tensor
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

def transform_prep_cmyk(n_px, divisor=16, contrast_factor=1.5):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        lambda image: resize_to_divisible(image, divisor),  # Resize with divisor
        convert_image_to_cmyk,  # Convert to RGB
        lambda image: enhance_contrast(image, factor=contrast_factor),  # Adjust contrast
        ToTensor(),  # Convert to tensor
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

def get_transform(n_px, divisor=16, contrast_factor=1.5):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        lambda image: resize_to_divisible(image, divisor),  # Resize with divisor
        ToTensor(),  # Convert to tensor
    ])

def get_bg_mask(img_input, n_px):
    output = remove(img_input)
    preprocess = get_transform(n_px)
    image_tensor = preprocess(output)
    alpha_channel = image_tensor[3, :, :]  # Shape: [H, W]
    return (alpha_channel != 0).float()


def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale.

    Args:
        image (torch.Tensor): Input image of shape (3, H, W) or (B, 3, H, W) for batch processing.

    Returns:
        torch.Tensor: Grayscale image of shape (1, H, W) or (B, 1, H, W).
    """
    if image.dim() == 3:  # Single image, (3, H, W)
        if image.shape[0] != 3:
            raise ValueError("Input image must have 3 channels (RGB).")
        grayscale = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        grayscale = grayscale.unsqueeze(0)  # Add channel dimension to make it (1, H, W)
    elif image.dim() == 4:  # Batch of images, (B, 3, H, W)
        if image.shape[1] != 3:
            raise ValueError("Input images must have 3 channels (RGB).")
        grayscale = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        grayscale = grayscale.unsqueeze(1)  # Add channel dimension to make it (B, 1, H, W)
    else:
        raise ValueError("Input image must have shape (3, H, W) or (B, 3, H, W).")
    
    return grayscale * 255





