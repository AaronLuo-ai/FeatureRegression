import wandb
from pathlib import Path
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.dataloader import RegressionDataset
from utils.transform import image_transform
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import torch
from pl_bolts.models.regression import LinearRegression
import sys
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
sys.path.append("..")
from model_param.customized_model import CustomResNet34Encoder


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)

    model_path = Path(
        "C:\\Users\\aaron.l\\Documents\\FeatureRegression\\model_param\\encoder.pth"
    )
    model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1)
    new_encoder = model.encoder
    new_encoder.load_state_dict(torch.load(model_path))
    new_encoder.eval()
    customized_encoder = CustomResNet34Encoder(new_encoder)
    for param in customized_encoder.parameters():
        param.requires_grad = False

    input_dim = 512
    batch_size = 32
    num_workers = 4
    max_epochs = 50
    lr = 1e-4

    # Dataset and Dataloader
    RegressionTransformTrain = image_transform
    RegressionTransformTest = image_transform

    TrainDataset = RegressionDataset(
        encoder=customized_encoder, transform=RegressionTransformTrain, phase="train"
    )
    TestDataset = RegressionDataset(
        encoder=customized_encoder, transform=RegressionTransformTest, phase="test"
    )

    TrainDataLoader = DataLoader(
        TrainDataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    TestDataLoader = DataLoader(
        TestDataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=3e-3, momentum=0.9, weight_decay=0.0001
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01)
    # Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current device:", device)

    # Initialize Lightning Model
    regression_model = LinearRegression(input_dim=input_dim)

    # Logging with WandB
    run_name = f"linear_regression_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_batch={batch_size}"
    wandb_logger = WandbLogger(
        log_model=False, project="MRI-Feature-Regression", name=run_name
    )
    wandb_logger.watch(regression_model, log="all", log_freq=100, log_graph=False)

    # Callbacks
    # Callbacks (place here before Trainer)
    # early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    # Trainer (pass callbacks here)
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=max_epochs,
        # callbacks=[early_stopping],  # Pass callbacks here
        min_epochs=10,
        num_sanity_val_steps=0,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        check_val_every_n_epoch=1  # Ensure validation happens every epoch
    )

    # Train the model
    trainer.fit(regression_model, TrainDataLoader, TestDataLoader)

    # Test the model
    trainer.test(regression_model, TestDataLoader)

    # Finish WandB
    wandb_logger.experiment.unwatch(regression_model)
    wandb.finish()


if __name__ == "__main__":
    main()
