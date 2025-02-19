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


class MRIDataset(Dataset):
    def __init__(
        self, encoder, root_dir = "C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple",
            batch_path = "C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple\\batch.csv",
            response_dir = r"C:\Users\aaron.l\Documents\db_20241213.xlsx",
            target_size=(128, 128), phase="train", transform=None
    ):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.batch_path = batch_path
        self.transform = transform
        self.response_dir = response_dir
        self.phase = phase
        self.encoder = encoder # Model already set to evaluation mode
        # Correct zip usage
        df = pd.read_excel(response_dir)
        df['patient_info'] = list(zip(df['cnda_subject_label'], df['Tumor Response']))
        new_df = df[['patient_info']]
        # Keep only patient_info column
        new_df = new_df.drop_duplicates(subset=['patient_info'], keep='first')
        new_df['patient_info'] = new_df['patient_info'].apply(
            lambda x: (x[0], "Complete response") if x[1] == "Partial response" else x)

        # Fix np.unique() issue
        print("Number of unique elements:", new_df['patient_info'].nunique())
        print("Number of columns:", len(new_df))
        print(new_df)

        # Same as the dataset for training the U-Net
        csv = pd.read_csv(self.batch_path)
        num_lines = len(csv)
        separation_index = int(0.75 * num_lines)
        if self.phase == "train":
            self.image_files = csv["Image"].tolist()[:separation_index]
            self.mask_files = csv["Mask"].tolist()[:separation_index]
        else:
            self.image_files = csv["Image"].tolist()[separation_index + 1 :]
            self.mask_files = csv["Mask"].tolist()[separation_index + 1 :]

        self.images: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.responses: list[np.ndarray] = []
        print(len(self.image_files))
        for index in range(len(self.image_files)):
            ids = self.image_files[index].split("_MR")[0]
            exists = new_df['patient_info'].apply(lambda x: x[0] == ids).any()
            if exists:
                # Extract the first matching tuple
                matching_tuple = new_df.loc[new_df['patient_info'].apply(lambda x: x[0] == ids), 'patient_info'].values[0]

                # Extract the second element of the tuple
                second_element = matching_tuple[1]
                images, _ = nrrd.read(self.root_dir / self.image_files[index])
                masks, _ = nrrd.read(self.root_dir / self.mask_files[index])

                self.images.extend(images.astype(np.float32))
                self.masks.extend(masks.astype(np.float32))
                self.responses.append(1 if second_element == 'Complete response' else 0)

    def __getitem__(self, index):
        feature = self.feature[index]
        response = self.responses[index]
        if self.transform:
            features = self.transform(feature)
        return feature, response

    def __len__(self):
        return len(self.images)

def main():
    file_path = Path(r"C:\Users\aaron.l\Documents\db_20241213.xlsx")
    df = pd.read_excel(file_path)

    # Correct zip usage
    df['patient_info'] = list(zip(df['cnda_subject_label'], df['Tumor Response']))
    new_df = df[['patient_info']]
    # Keep only patient_info column
    new_df = new_df.drop_duplicates(subset=['patient_info'], keep='first')
    new_df['patient_info'] = new_df['patient_info'].apply(lambda x: (x[0], "Complete response") if x[1] == "Partial response" else x)

    # Fix np.unique() issue
    print("Number of unique elements:", new_df['patient_info'].nunique())
    print("Number of columns:", len(new_df))
    print(new_df)

    mri_path = Path("C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple")
    batch_path = Path(
        "C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple\\batch.csv"
    )
    model_path = Path("C:\\Users\\aaron.l\\Documents\\FeatureRegression\\model_param\\encoder.pth")
    model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1)
    # new_encoder = model.encoder
    # for index, row in df.iterrows():
    #     patient_id = row['patient_info'][0]  # Extract first element of tuple
    #     print(patient_id)
    # new_encoder.load_state_dict(torch.load(model_path, weights_only=True))
    # new_encoder.eval()  # Set model to evaluation mode
    #
    # for index, row in new_df.iterrows():
    #     patient_id = row['patient_info'][0]  # Extract first element of tuple
    #     # print("type of patient id:", type(row['patient_info']))
    #     # print("type of patient id:", type(patient_id))
    #     print("type of the second element:", type(row[1]), "value: ", row[1])
    #     print(patient_id)
    second_elements = new_df['patient_info'].apply(lambda x: x[1]).tolist()
    print(second_elements)


if __name__ == "__main__":
    main()
