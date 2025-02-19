import random
from typing import Any
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import logger
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger


class Segmentation(pl.LightningModule):
    def __init__(self, model, optimizer, loss_fn, metrics, scheduler, **kwargs):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metrics
        self.scheduler = scheduler
        self.save_hyperparameters()  # Automatically logs all passed arguments
        self.training_step_inputs = []
        self.training_step_outputs = (
            []
        )  # save outputs in each batch to compute metric overall epoch
        self.training_step_targets = (
            []
        )  # save targets in each batch to compute metric overall epoch
        self.val_step_inputs = []
        self.val_step_outputs = (
            []
        )  # save outputs in each batch to compute metric overall epoch
        self.val_step_targets = (
            []
        )  # save targets in each batch to compute metric overall epoch

    # def on_fit_start(self):
    #     if self.logger:
    #         print("Logging hyperparameters to WandB...")
    #         print("Hyperparameters:", self.hparams)
    #         self.logger.log_hyperparams({
    #             **self.hparams,
    #             "additional_param": "custom_value",
    #         })

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        masks = torch.clamp(masks, 0, 1)
        predictions = self.forward(images)
        loss = self.loss_fn(predictions, masks)

        # Log the learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "train/lr",
            current_lr,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        # Rest of the logging logic
        predicted_masks = predictions.clone()
        predicted_masks[predicted_masks > 0.5] = 1
        predicted_masks[predicted_masks <= 0.5] = 0
        predicted_masks = predicted_masks.clamp(0, 1)
        self.training_step_outputs.extend(predicted_masks)
        self.training_step_targets.extend(masks)
        self.training_step_inputs.extend(images)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self, *arg, **kwargs):
        predicted_masks = torch.stack(self.training_step_outputs)
        masks = torch.stack(self.training_step_targets)
        images = torch.stack(self.training_step_inputs)
        rand_num = random.sample(range(masks.shape[0]), 5)  # Use range(masks.shape[0])

        dice_score = self.metric(
            preds=predicted_masks.long(), target=masks.long()
        )  # Error
        self.log(
            "train/dice_score", dice_score, on_step=False, on_epoch=True, prog_bar=True
        )

        images_np = images[rand_num].cpu().detach().numpy().squeeze()
        predicted_masks_np = predicted_masks[rand_num].cpu().detach().numpy().squeeze()
        masks_np = masks[rand_num].cpu().detach().numpy().squeeze()

        table = wandb.Table(
            columns=["Image ID", "Training Overlay Image", "Training Mask", "Predicted"]
        )
        class_labels = {1: "tumor"}
        for index in range(images_np.shape[0]):
            # img = np.transpose(images_np[index], (1, 2, 0))  # From [C, H, W] -> [H, W, C]
            img = images_np[index]
            mask = masks_np[index]
            pred = predicted_masks_np[index]
            overlay_img = wandb.Image(
                img,
                masks={
                    "ground_truth": {"mask_data": mask, "class_labels": class_labels},
                    "predictions": {"mask_data": pred, "class_labels": class_labels},
                },
            )
            table.add_data(
                index,
                overlay_img,
                wandb.Image(masks_np[index].squeeze()),
                wandb.Image(predicted_masks_np[index].squeeze()),
            )
        wandb.log({f"training_epoch_{self.current_epoch}": table})

        self.training_step_outputs.clear()
        self.training_step_targets.clear()
        self.training_step_inputs.clear()

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        masks = torch.clamp(masks, 0, 1)
        predictions = self.forward(images)
        loss = self.loss_fn(predictions, masks)

        # Log the learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "validation/lr",
            current_lr,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        # Rest of the logging logic
        predicted_masks = predictions.clone()
        predicted_masks[predicted_masks > 0.5] = 1
        predicted_masks[predicted_masks <= 0.5] = 0
        predicted_masks = predicted_masks.clamp(0, 1)
        self.val_step_outputs.extend(predicted_masks)
        self.val_step_targets.extend(masks)
        self.val_step_inputs.extend(images)

        self.log("validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self, *arg, **kwargs) -> None:
        predicted_masks = torch.stack(self.val_step_outputs)
        masks = torch.stack(self.val_step_targets)
        images = torch.stack(self.val_step_inputs)
        rand_num = random.sample(range(masks.shape[0]), 5)  # Use range(masks.shape[0])
        # print("predicted_masks.shape: ", predicted_masks.shape, "masks.shape: ", masks.shape)
        dice_score = self.metric(preds=predicted_masks.long(), target=masks.long())
        self.log(
            "validation/dice_score",
            dice_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        images_np = images[rand_num].cpu().detach().numpy().squeeze()
        predicted_masks_np = predicted_masks[rand_num].cpu().detach().numpy().squeeze()
        masks_np = masks[rand_num].cpu().detach().numpy().squeeze()

        table = wandb.Table(
            columns=[
                "Image ID",
                "Validation Overlay Image",
                "Validation Mask",
                "Predicted",
            ]
        )
        class_labels = {1: "tumor"}
        # print("images_np.shape: ", images_np.shape, ",masks_np.shape: ", masks_np.shape, ",predicted_masks_np.shape: ", predicted_masks_np.shape)
        for index in range(images_np.shape[0]):
            # img = np.transpose(images_np[index], (1, 2, 0))  # From [C, H, W] -> [H, W, C]
            img = images_np[index]
            mask = masks_np[index]
            pred = predicted_masks_np[index]
            overlay_img = wandb.Image(
                img,
                masks={
                    "ground_truth": {"mask_data": mask, "class_labels": class_labels},
                    "predictions": {"mask_data": pred, "class_labels": class_labels},
                },
            )
            table.add_data(
                index,
                overlay_img,
                wandb.Image(mask.squeeze()),
                wandb.Image(pred.squeeze()),
            )
        wandb.log({f"validation_epoch_{self.current_epoch}": table})

        self.val_step_outputs.clear()
        self.val_step_targets.clear()
        self.val_step_inputs.clear()

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }
