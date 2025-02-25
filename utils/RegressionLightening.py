import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import R2Score


class LinearRegression(pl.LightningModule):
    def __init__(self, input_dim=512, lr=1e-3, weight_decay=1e-5, tolerance=0.5):
        super().__init__()
        # Define model components
        self.linear = nn.Linear(input_dim, 1)  # Single output
        self.loss_fn = nn.MSELoss()            # Mean Squared Error for regression
        self.r2 = R2Score()                    # R2 Score for regression
        self.tolerance = tolerance             # Tolerance for accuracy

        # Save hyperparameters
        self.save_hyperparameters()
        self.lr = lr

    def forward(self, x):
        return self.linear(x).squeeze()

    # Custom Accuracy Function for Regression
    def regression_accuracy(self, predictions, targets):
        return torch.mean((torch.abs(predictions - targets) <= self.tolerance).float())

    # ===================================
    #             TRAINING STEP
    # ===================================
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        r2_score = self.r2(scores, y)
        accuracy = self.regression_accuracy(scores, y)

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/r2_score', r2_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ==============================
    #        VALIDATION STEP
    # ==============================
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        r2_score = self.r2(scores, y)
        accuracy = self.regression_accuracy(scores, y)

        self.log('validation/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('validation/r2_score', r2_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log('validation/accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ==============================
    #         TEST STEP
    # ==============================
    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        r2_score = self.r2(scores, y)
        accuracy = self.regression_accuracy(scores, y)

        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/r2_score', r2_score, on_step=False, on_epoch=True)
        self.log('test/accuracy', accuracy, on_step=False, on_epoch=True)
        return loss

    # ==============================
    #         COMMON STEP
    # ==============================
    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    # ==============================
    #    CONFIGURE OPTIMIZERS
    # ==============================
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
