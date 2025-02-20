import pandas as pd
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import nrrd
import matplotlib.pyplot as plt
from utils.transform import *
import sys

sys.path.append("..")
from model_param.customized_model import CustomResNet34Encoder


class RegressionDataset(Dataset):
    def __init__(
        self,
        encoder,
        root_dir="C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple",
        batch_path="C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple\\batch.csv",
        response_dir=r"C:\Users\aaron.l\Documents\db_20241213.xlsx",
        phase="train",
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.batch_path = batch_path
        self.response_dir = response_dir
        self.phase = phase
        self.transform = transform
        self.encoder = encoder

        # Correct zip usage
        df = pd.read_excel(response_dir)
        df["patient_info"] = list(zip(df["cnda_subject_label"], df["Tumor Response"]))
        new_df = df[["patient_info"]]
        # Keep only patient_info column
        new_df = new_df.drop_duplicates(subset=["patient_info"], keep="first")
        new_df["patient_info"] = new_df["patient_info"].apply(
            lambda x: (x[0], "Complete response") if x[1] == "Partial response" else x
        )

        # Same as the dataset for training the U-Net
        csv = pd.read_csv(self.batch_path)
        num_lines = len(csv)
        separation_index = int(0.75 * num_lines)
        if self.phase == "train":
            self.image_files = csv["Image"].tolist()[:separation_index]
        else:
            self.image_files = csv["Image"].tolist()[separation_index + 1 :]

        self.data = []
        print(len(self.image_files))

        for image_file in self.image_files:
            patient_id = image_file.split("_MR")[0]
            match = new_df[new_df["patient_info"].apply(lambda x: x[0] == patient_id)]
            if not match.empty:
                response = (
                    1 if match.iloc[0]["patient_info"][1] == "Complete response" else 0
                )
                image_path = self.root_dir / image_file

                # Read the NRRD file once
                image, _ = nrrd.read(image_path)  # Shape: (10, 320, 320)

                # Process each slice and store them individually
                for slice_idx in range(image.shape[0]):
                    slice_array = image[slice_idx, :, :]  # Shape: (320, 320)
                    slice_tensor = torch.from_numpy(
                        slice_array.astype(np.float32)
                    ).unsqueeze(
                        0
                    )  # Shape: (1, 320, 320)

                    self.data.append((slice_tensor, response))

    def __getitem__(self, index):
        slice_tensor, response = self.data[index]
        if self.transform:
            slice_tensor = self.transform(slice_tensor)
        # print("shape of slice tensor: ", slice_tensor.shape)
        # Features are extracted during DataLoader iteration, not in __init__
        features_tensor = self.encoder(
            slice_tensor.unsqueeze(0)
        )  # Add batch dimension: (1, 1, 128, 128)
        features_tensor = torch.cat([f.view(-1) for f in features_tensor], dim=0)
        # print("features_tensor.shape: ",features_tensor.shape)
        # print("features_tensor type: ",type(features_tensor))
        return features_tensor, torch.tensor(response, dtype=torch.float32)

    def __len__(self):
        return len(self.data)


def main():
    model_path = Path(
        "C:\\Users\\aaron.l\\Documents\\FeatureRegression\\model_param\\encoder.pth"
    )
    model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1)
    new_encoder = model.encoder
    new_encoder.load_state_dict(torch.load(model_path, weights_only=True))
    customized_encoder = CustomResNet34Encoder(new_encoder)
    customized_encoder.eval()
    batch_size = 5
    dataset = RegressionDataset(
        encoder=customized_encoder, phase="test", transform=image_transform
    )
    DataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    for batch_idx, (features, targets) in enumerate(DataLoader):
        print(f"\n Batch {batch_idx + 1}")
        print(
            "Features shape:", features.shape
        )  # Shape of features (batch_size, channels, H, W) or encoder output
        print("Targets:", targets)  # Binary targets (0 or 1)

        # # Visualize the first image in the batch (if features are still images)
        # plt.figure(figsize=(4, 4))
        # plt.imshow(features[0].squeeze().cpu().numpy(), cmap="gray")
        # plt.title(f"Target: {targets[0].item()}")
        # plt.axis("off")
        # plt.show()

        # Total number of batches
    print("\nTotal number of batches:", len(DataLoader))


if __name__ == "__main__":
    main()
