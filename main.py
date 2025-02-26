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
import sys
from utils.RegressionLightening import BinaryClassification

from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR

sys.path.append("..")
from model_param.customized_model import CustomResNet34Encoder


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)

    # Lab Compputer
    model_path = Path(
        "C:\\Users\\aaron.l\\Documents\\FeatureRegression\\model_param\\encoder.pth"
    )

    # model location for my own computer (IGNORE THIS)
    # model_path = Path(
    #         "/Users/luozisheng/Documents/Zhu_lab/FeatureRegression/model_param/encoder.pth"
    #     )

    unet = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1)
    new_encoder = unet.encoder
    new_encoder.load_state_dict(torch.load(model_path))
    new_encoder.eval()
    customized_encoder = CustomResNet34Encoder(new_encoder)
    for param in customized_encoder.parameters():
        param.requires_grad = False

    input_dim = 512
    batch_size = 32
    num_workers = 4
    max_epochs = 200
    min_epochs = 1
    lr = 1e-4
    check_val_every_n_epoch = 3

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Initialize Model
    model = BinaryClassification(input_dim=input_dim, lr=lr)

    # Initialize Callbacks
    early_stopping = EarlyStopping(monitor="validation/loss", patience=40, mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor="validation/loss",
        dirpath="checkpoints/",
        filename="best-checkpoint-{epoch:02d}-{validation/loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    # Initialize WandB Logger
    run_name = f"linear_regression_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_batch={batch_size}"
    wandb_logger = WandbLogger(
        log_model=False, project="MRI-Feature-Regression", name=run_name
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        callbacks=[early_stopping, checkpoint_callback],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator="gpu",
    )

    # Train and Validate
    trainer.fit(model, TrainDataLoader, TestDataLoader)
    trainer.validate(model, TestDataLoader)

    # Finish WandB
    wandb_logger.experiment.unwatch(model)
    wandb.finish()


if __name__ == "__main__":
    main()
