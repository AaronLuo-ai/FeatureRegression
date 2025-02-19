import torchvision.transforms as transforms
import torch


# Define the transform
image_transform = transforms.Compose([
    transforms.Resize((128, 128)),     # Resize to 128x128
    transforms.ConvertImageDtype(torch.float32),  # Ensure float32 dtype
    transforms.Normalize(mean=[0.0], std=[1.0])  # 0-1 normalization (identity transform)
])
