import inspect
import torch
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from timm.scheduler.cosine_lr import CosineLRScheduler
import pytorch_lightning as pl
from metrics import BaseMetric
from model.circuitformer import CircuitFormer
from losses import WingLoss
import numpy as np
import matplotlib.pyplot as plt


class MInterface(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.metric = BaseMetric()
        self.load_model()
        self.configure_loss()
        self.model_name = "CircuitFormer"

        # save cfg into Lightning hparams so logger can show config keys nicely
        try:
            from omegaconf import OmegaConf
            self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
            self.hparams["model_name"] = self.model_name
        except Exception:
            self.save_hyperparameters()

        # controls for visualization logging
        self.max_test_visualizations = getattr(cfg, "max_test_visualizations", 6)
        self.logged_test_visualizations = 0
    

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        x1, y1, x2, y2, offset, label, weight = batch
        output = self([x1, y1, x2, y2, offset])
        loss = self.cfg.loss_weight * torch.mean(((output - label) ** 2) * weight)
        # loss = self.cfg.loss_weight * self.loss_function(output, label)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.cfg.batch_size)
        return loss


    def validation_step(self, batch, batch_idx):
        x1, y1, x2, y2, offset, label, weight = batch
        output = self([x1, y1, x2, y2, offset]) / self.cfg.label_weight
        for i in range(output.shape[0]):
            output_ = output[i,].squeeze()
            label_ = label[i,].squeeze()
            self.metric.update(label_.detach().cpu(),output_.detach().cpu())

    def test_step(self, batch, batch_idx):    
        x1, y1, x2, y2, offset, label, weight = batch
        output = self([x1, y1, x2, y2, offset]) / self.cfg.label_weight

        if batch_idx == 0:
            print("label min/max:", label.min().item(), label.max().item())
            print("output min/max:", output.min().item(), output.max().item())
            print("label_weight:", self.cfg.label_weight)
       
        for i in range(output.shape[0]):
            output_ = output[i].squeeze()
            label_ = label[i].squeeze()
            self.metric.update(label_.detach().cpu(),output_.detach().cpu())

        # Log a small number of visual examples during test
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

                    pred_vmin = float(pred_arr.min())
                    pred_vmax = float(pred_arr.max())
                    if pred_vmax <= pred_vmin:
                        pred_vmax = pred_vmin + 1e-8
                    
                    err_vmax = float(err_arr.max())
                    if err_vmax <= 0:
                        err_vmax = 1e-8
                    
                    im1 = axs[1].imshow(pred_arr, cmap="viridis", vmin=pred_vmin, vmax=pred_vmax)
                    axs[1].set_title("Prediction")
                    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
                    
                    im2 = axs[2].imshow(err_arr, cmap="viridis", vmin=0.0, vmax=err_vmax)
                    axs[2].set_title("Absolute Error")
                    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
                    
                    for ax in axs:
                        ax.set_xticks([])
                        ax.set_yticks([])
                    
                    plt.tight_layout()

                    idx = self.logged_test_visualizations
                    log_dict[f"test/images/comparison_{idx}"] = wandb.Image(
                        fig, caption=f"Sample {idx}"
                    )

                    plt.close(fig)
                    self.logged_test_visualizations += 1

                if log_dict:
                    self.logger.experiment.log(log_dict)

        except Exception as e:
            print(f"[WARN] Visualization logging failed at batch {batch_idx}: {e}")

    def on_test_epoch_end(self):
        # Make the Progress Bar leave there
        pearson, spearman, kendall = self.metric.compute()
        self.log('test/pearson', pearson, on_step=False, on_epoch=True, batch_size=self.cfg.batch_size, sync_dist=True)
        self.log('test/spearman', spearman, on_step=False, on_epoch=True, batch_size=self.cfg.batch_size, sync_dist=True)
        self.log('test/kendall', kendall, on_step=False, on_epoch=True, batch_size=self.cfg.batch_size, sync_dist=True)
        print("pearson:", pearson, "spearman:", spearman, "kendall:", kendall)
        self.metric.reset()
        self.logged_test_visualizations = 0

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        pearson, spearman, kendall = self.metric.compute()
        self.log('val/pearson', pearson, on_step=False, on_epoch=True, batch_size=self.cfg.batch_size, sync_dist=True)
        self.log('val/spearman', spearman, on_step=False, on_epoch=True, batch_size=self.cfg.batch_size, sync_dist=True)
        self.log('val/kendall', kendall, on_step=False, on_epoch=True, batch_size=self.cfg.batch_size, sync_dist=True)
        print("pearson:", pearson, "spearman:", spearman, "kendall:", kendall)
        self.metric.reset()



    def configure_optimizers(self):
        weight_decay = getattr(self.cfg, "weight_decay", 0) or 0

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=weight_decay
        )

        if self.cfg.lr_scheduler is None:
            return optimizer

        if self.cfg.lr_scheduler == "step":
            scheduler = lrs.StepLR(
                optimizer,
                step_size=self.cfg.lr_decay_steps,
                gamma=self.cfg.lr_decay_rate,
            )
        elif self.cfg.lr_scheduler == "cosine":
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=self.cfg.max_epochs,
                lr_min=self.cfg.min_lr,
                warmup_lr_init=self.cfg.warmup_lr,
                warmup_t=self.cfg.warmup_epochs,
                cycle_limit=1,
                t_in_epochs=True,
            )
        else:
            raise ValueError("Invalid lr_scheduler type!")

        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, metric=None):
        # timm's scheduler need the epoch value, so have to overwrite lr_scheduler_step
        scheduler.step(epoch=self.current_epoch + 1)

    def configure_loss(self):
        loss = self.cfg.loss
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'wingloss':
            self.loss_function = WingLoss()
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        self.model = CircuitFormer()
        print("Using model: CircuitFormer")

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
