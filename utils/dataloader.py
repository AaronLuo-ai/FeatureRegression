import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


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
    model = MyModel()  # Initialize model
    model.load_state_dict(torch.load("model.pth"))  # Load saved weights
    model.eval()  # Set model to evaluation mode

if __name__ == "__main__":
    main()
