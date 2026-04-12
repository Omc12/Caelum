import fiftyone as fo
import fiftyone.zoo as foz
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from torchvision import transforms

# Prevent OpenCV from hoarding CPU threads when used with PyTorch DataLoader
cv2.setNumThreads(0)

def download_open_images_sky(num_samples=5000):
    """Downloads sky images from Open Images V7."""
    print(f"Downloading {num_samples} Sky images. This might take a while...")
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["classifications"],
        classes=["Tree", "Skyscraper"],
        max_samples=num_samples,
        dataset_name="sky-dataset"
    )
    # Extract the file paths of the downloaded images
    filepaths = [sample.filepath for sample in dataset]
    return filepaths

class SkyEnhancementDataset(Dataset):
    def __init__(self, filepaths, image_size=256):
        self.filepaths = filepaths
        self.image_size = image_size
        # PyTorch requires tensors and specific normalizations
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Converts 0-255 to 0.0-1.0
        ])

    def degrade_image(self, image):
        """Mathematically ruins the perfect sky to create our input data."""
        # Convert to HSV to easily drop saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.3  # Drop saturation by 70%
        hsv[:, :, 2] *= 0.8  # Drop brightness slightly
        
        dull_hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        dull_rgb = cv2.cvtColor(dull_hsv, cv2.COLOR_HSV2RGB)
        
        # Add a slight gray haze
        haze = np.full_like(dull_rgb, 128)
        final_dull = cv2.addWeighted(dull_rgb, 0.85, haze, 0.15, 0)
        return final_dull

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Load the perfect image (Ground Truth / Target)
        img_path = self.filepaths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV loads BGR, we need RGB
    
        # Resize to fit our CNN (256x256)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Create the degraded version (Input)
        dull_image = self.degrade_image(image)

        #-----HSV LOGIC-----
        dull_float = dull_image.astype(np.float32)/255.0
        perfect_float = image.astype(np.float32)/255.0

        dull_hsv = cv2.cvtColor(dull_float, cv2.COLOR_RGB2HSV)
        perfect_hsv = cv2.cvtColor(perfect_float, cv2.COLOR_RGB2HSV)

        dull_hsv[:, :, 0] /= 360.0
        perfect_hsv[:, :, 0] /= 360.0
        
        # Convert both to PyTorch Tensors
        x_dull = torch.from_numpy(dull_hsv).permute(2, 0, 1)
        y_perfect = torch.from_numpy(perfect_hsv).permute(2, 0, 1)
        
        return x_dull, y_perfect