import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy


class BinaryClassification(pl.LightningModule):
    def __init__(self, input_dim=512, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        # Define model components
        self.linear = nn.Linear(
            input_dim, 1
        )  # No sigmoid since BCEWithLogitsLoss handles it
        self.loss_fn = nn.BCEWithLogitsLoss()  # More stable than BCELoss
        self.accuracy = BinaryAccuracy()  # Accuracy metric using TorchMetrics

        # Save hyperparameters
        self.save_hyperparameters()
        self.lr = lr

    # ==============================
    #           FORWARD
    # ==============================
    def forward(self, x):
        return self.linear(x).squeeze()

    # ==============================
    #        TRAINING STEP
    # ==============================
    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        acc = self.accuracy(
            torch.sigmoid(scores), y.int()
        )  # Use torch.sigmoid before accuracy

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ==============================
    #       VALIDATION STEP
    # ==============================
    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        acc = self.accuracy(torch.sigmoid(scores), y.int())

        self.log("validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "validation/accuracy", acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    # ==============================
    #          TEST STEP
    # ==============================
    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        acc = self.accuracy(torch.sigmoid(scores), y.int())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/accuracy", acc, on_step=False, on_epoch=True)
        return loss

    # ==============================
    #         COMMON STEP
    # ==============================
    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(
            scores, y.float()
        )  # Ensure labels are float for BCEWithLogitsLoss
        return loss, scores, y

    # ==============================
    #    CONFIGURE OPTIMIZERS
    # ==============================
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
