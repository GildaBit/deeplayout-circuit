import torch
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from timm.scheduler.cosine_lr import CosineLRScheduler
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from metrics import BaseMetric
from losses import WingLoss
from model.gnn_model.circuitformer_deeplayout import CircuitFormerDeepLayout


class MInterfaceDeeplayout(pl.LightningModule):
    """
    LightningModule for the new DeepLayout-style pipeline.
    """
    def __init__(
        self,
        lr=3e-4,
        loss="mse",
        loss_weight=1.0,
        coord_loss_weight=0.1,
        label_weight=1.0,
        batch_size=4,
        weight_decay=1e-4,
        lr_scheduler="cosine",
        lr_decay_steps=10,
        lr_decay_rate=0.5,
        min_lr=1e-6,
        warmup_lr=1e-6,
        warmup_epochs=5,
        max_epochs=50,
        max_test_visualizations=6,
        resnet_ckpt_path=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.metric = BaseMetric()
        self.model = CircuitFormerDeepLayout(resnet_ckpt_path=resnet_ckpt_path)
        self.model_name = "CircuitFormerDeepLayout"
        self.configure_loss()

        self.max_test_visualizations = max_test_visualizations
        self.logged_test_visualizations = 0

    def forward(self, batch):
        return self.model(batch)

    def compute_coord_mask_loss(self, aux):
        """
        Auxiliary MSE over masked cells only.
        """
        coord_pred = aux["coord_pred"]
        coord_target = aux["coord_target"]
        cell_mask = aux["cell_mask"]

        if cell_mask is None or cell_mask.numel() == 0 or not cell_mask.any():
            return coord_pred.new_tensor(0.0)

        pred = coord_pred[cell_mask]
        target = coord_target[cell_mask]
        return F.mse_loss(pred, target)

    def training_step(self, batch, batch_idx):
        output, aux = self(batch)

    
        if not torch.isfinite(output).all():
            raise RuntimeError("Non-finite output detected in training_step")
    
        label = batch["label"]
        weight = batch["weight"]

        if batch_idx == 0:
            print("output min/max/mean:", output.min().item(), output.max().item(), output.mean().item())
            print("label min/max/mean:", label.min().item(), label.max().item(), label.mean().item())
    
        if output.shape[-2:] != label.shape[-2:]:
            output = torch.nn.functional.interpolate(
                output,
                size=label.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
    
        main_loss = self.hparams.loss_weight * torch.mean(((output - label) ** 2) * weight)
        coord_loss = self.compute_coord_mask_loss(aux)
        loss = main_loss + self.hparams.coord_loss_weight * coord_loss
    
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected: {loss}")
    
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log("loss_main", main_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=self.hparams.batch_size)
        self.log("loss_coord", coord_loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        output, aux = self(batch)
        label = batch["label"]
    
        if output.shape[-2:] != label.shape[-2:]:
            output = torch.nn.functional.interpolate(
                output,
                size=label.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
    
        if batch_idx == 0:
            print("label min/max:", label.min().item(), label.max().item())
            print("output min/max:", output.min().item(), output.max().item())
            print("label mean:", label.mean().item())
            print("output mean:", output.mean().item())
    
        for i in range(output.shape[0]):
            output_ = output[i].squeeze()
            label_ = label[i].squeeze()
            self.metric.update(label_.detach().cpu(), output_.detach().cpu())

    def test_step(self, batch, batch_idx):
        output, aux = self(batch)
        label = batch["label"]

        if output.shape[-2:] != label.shape[-2:]:
            output = torch.nn.functional.interpolate(
                output,
                size=label.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if batch_idx == 0:
            print("label min/max:", label.min().item(), label.max().item())
            print("output min/max:", output.min().item(), output.max().item())
            print("label_weight:", self.hparams.label_weight)
        
        for i in range(output.shape[0]):
            output_ = output[i].squeeze()
            label_ = label[i].squeeze()
        
            self.metric.update(label_.detach().cpu(), output_.detach().cpu())

        try:
            import wandb

            if hasattr(self.logger, "experiment"):
                log_dict = {}

                for i in range(output.shape[0]):
                    if self.logged_test_visualizations >= self.max_test_visualizations:
                        break

                    pred_arr = output[i].detach().cpu().squeeze().numpy()
                    gt_arr = label[i].detach().cpu().squeeze().numpy()
                    err_arr = np.abs(pred_arr - gt_arr)

                    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

                    im0 = axs[0].imshow(gt_arr, cmap="viridis", vmin=0.0, vmax=1.0)
                    axs[0].set_title("Ground Truth")
                    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

                    # clip for visualization only
                    pred_arr = np.clip(pred_arr, 0.0, 1.0)
                    
                    im1 = axs[1].imshow(pred_arr, cmap="viridis", vmin=0.0, vmax=1.0)
                    axs[1].set_title("Prediction")
                    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

                    im2 = axs[2].imshow(err_arr, cmap="viridis", vmin=0.0, vmax=1.0)
                    axs[2].set_title("Absolute Error")
                    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

                    for ax in axs:
                        ax.set_xticks([])
                        ax.set_yticks([])

                    plt.tight_layout()

                    idx = self.logged_test_visualizations
                    log_dict[f"test/images/comparison_{idx}"] = wandb.Image(fig, caption=f"Sample {idx}")

                    plt.close(fig)
                    self.logged_test_visualizations += 1

                if log_dict:
                    self.logger.experiment.log(log_dict)

        except Exception as e:
            print(f"[WARN] Visualization logging failed at batch {batch_idx}: {e}")

    def on_validation_epoch_end(self):
        pearson, spearman, kendall = self.metric.compute()
    
        self.log("val/pearson", pearson, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val/spearman", spearman, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("val/kendall", kendall, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, sync_dist=True)
    
        print("val pearson:", pearson, "spearman:", spearman, "kendall:", kendall)
        self.metric.reset()

    def on_test_epoch_end(self):
        pearson, spearman, kendall = self.metric.compute()
        self.log("test/pearson", pearson, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test/spearman", spearman, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("test/kendall", kendall, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size, sync_dist=True)
        print("pearson:", pearson, "spearman:", spearman, "kendall:", kendall)
        self.metric.reset()
        self.logged_test_visualizations = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_scheduler is None:
            return optimizer

        if self.hparams.lr_scheduler == "step":
            scheduler = lrs.StepLR(
                optimizer,
                step_size=self.hparams.lr_decay_steps,
                gamma=self.hparams.lr_decay_rate,
            )
        elif self.hparams.lr_scheduler == "cosine":
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=self.hparams.max_epochs,
                lr_min=self.hparams.min_lr,
                warmup_lr_init=self.hparams.warmup_lr,
                warmup_t=self.hparams.warmup_epochs,
                cycle_limit=1,
                t_in_epochs=True,
            )
        else:
            raise ValueError("Invalid lr_scheduler type!")

        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, metric=None):
        scheduler.step(epoch=self.current_epoch + 1)

    def configure_loss(self):
        loss = self.hparams.loss
        if loss == "mse":
            self.loss_function = F.mse_loss
        elif loss == "l1":
            self.loss_function = F.l1_loss
        elif loss == "bce":
            self.loss_function = F.binary_cross_entropy
        elif loss == "wingloss":
            self.loss_function = WingLoss()
        else:
            raise ValueError("Invalid Loss Type!")