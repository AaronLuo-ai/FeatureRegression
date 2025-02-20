# FeatureRegression
This repository uses features from the U-Net to predict complete responses

1. Data Preparation:
[x] Simplify the dataloader file by packing elements into transform.py as a Transformation


2. DataLoader:
[x] __getitem()__: return (x = extracted tumor region of the image using a mask, y = response) 

3. Linear Regression:
[ ] There are too many features in the bottleneck of the smp.ResNet34. To limit it down, we performed a MLP feature extraction
[ ] Limit it down to 512 features. Perform linear regression on 512 features