import random
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics import MeanAbsoluteError, R2Score, MeanSquaredError
import wandb


class LinearRegression(pl.LightningModule):
    def __init__(self, input_dim=512, optimizer_type="Adam", lr=1e-4):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # Single output
        self.loss_fn = nn.MSELoss()  # Main loss function
        self.mae = MeanAbsoluteError()  # Mean Absolute Error
        self.mse = MeanSquaredError()   # Mean Squared Error
        self.r2 = R2Score()             # R2 Score

        self.save_hyperparameters()  # Saves optimizer and hyperparameters

        # Save predictions and targets each epoch
        self.training_step_outputs = []
        self.training_step_targets = []
        self.val_step_outputs = []
        self.val_step_targets = []

    def forward(self, x):
        return self.linear(x).squeeze()

    # ===================================
    #             TRAINING STEP
    # ===================================
    def training_step(self, batch, batch_idx):
        features, targets = batch
        predictions = self(features)
        loss = self.loss_fn(predictions, targets)

        # Save predictions and targets for epoch-level metrics
        self.training_step_outputs.append(predictions.detach())
        self.training_step_targets.append(targets.detach())

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train_lr", current_lr, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    # ==============================
    #     TRAINING EPOCH END
    # ==============================
    def on_train_epoch_end(self):
        predictions = torch.cat(self.training_step_outputs)
        targets = torch.cat(self.training_step_targets)

        mse = self.mse(predictions, targets)
        mae = self.mae(predictions, targets)
        r2 = self.r2(predictions, targets)

        self.log("train_epoch_mse", mse, on_epoch=True, prog_bar=True)
        self.log("train_epoch_mae", mae, on_epoch=True, prog_bar=True)
        self.log("train_epoch_r2", r2, on_epoch=True, prog_bar=True)

        # Save example predictions to WandB
        rand_indices = random.sample(range(len(targets)), min(5, len(targets)))
        table = wandb.Table(columns=["Target", "Prediction"])
        for idx in rand_indices:
            table.add_data(float(targets[idx]), float(predictions[idx]))
        wandb.log({f"train_predictions_epoch_{self.current_epoch}": table})

        # Clear the lists
        self.training_step_outputs.clear()
        self.training_step_targets.clear()

    # ==============================
    #        VALIDATION STEP
    # ==============================
    def validation_step(self, batch, batch_idx):
        features, targets = batch
        predictions = self(features)
        loss = self.loss_fn(predictions, targets)

        # Save predictions and targets for epoch-level metrics
        self.val_step_outputs.append(predictions.detach())
        self.val_step_targets.append(targets.detach())

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # ==============================
    #     VALIDATION EPOCH END
    # ==============================
    def on_validation_epoch_end(self):
        predictions = torch.cat(self.val_step_outputs)
        targets = torch.cat(self.val_step_targets)

        mse = self.mse(predictions, targets)
        mae = self.mae(predictions, targets)
        r2 = self.r2(predictions, targets)

        self.log("val_epoch_mse", mse, on_epoch=True, prog_bar=True)
        self.log("val_epoch_mae", mae, on_epoch=True, prog_bar=True)
        self.log("val_epoch_r2", r2, on_epoch=True, prog_bar=True)

        # Save example predictions to WandB
        rand_indices = random.sample(range(len(targets)), min(5, len(targets)))
        table = wandb.Table(columns=["Target", "Prediction"])
        for idx in rand_indices:
            table.add_data(float(targets[idx]), float(predictions[idx]))
        wandb.log({f"val_predictions_epoch_{self.current_epoch}": table})

        # Clear the lists
        self.val_step_outputs.clear()
        self.val_step_targets.clear()

    # ==============================
    #         TEST STEP
    # ==============================
    def test_step(self, batch, batch_idx):
        features, targets = batch
        predictions = self(features)
        loss = self.loss_fn(predictions, targets)

        mse = self.mse(predictions, targets)
        mae = self.mae(predictions, targets)
        r2 = self.r2(predictions, targets)

        self.log("test_loss", loss)
        self.log("test_mse", mse)
        self.log("test_mae", mae)
        self.log("test_r2", r2)

        return loss

    # ==============================
    #     CONFIGURE OPTIMIZERS
    # ==============================
    def configure_optimizers(self):
        if self.hparams.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        else:
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.lr, momentum=0.9
            )
        return optimizer
