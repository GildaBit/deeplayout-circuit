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
import matplotlib.cm as cm


class MInterface(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.metric = BaseMetric()
        self.load_model()
        self.configure_loss()
        self.cnt=0

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
            output_ = output[i,].squeeze(1)
            label_ = label[i,].squeeze(1)
            self.metric.update(label_.detach().cpu(),output_.detach().cpu())

    def test_step(self, batch, batch_idx):
        
        # Helper for coloring visualizations
        def to_colormap(img, cmap_name="inferno"):
            img = img.astype(np.float32)
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) / (mx - mn)
            cmap = cm.get_cmap(cmap_name)
            colored = cmap(img)[..., :3]  # drop alpha channel
            return (colored * 255).astype(np.uint8)
        
        x1, y1, x2, y2, offset, label, weight = batch
        output = self([x1, y1, x2, y2, offset]) / self.cfg.label_weight
       
        for i in range(output.shape[0]):
            output_ = output[i,].squeeze(1)
            label_ = label[i,].squeeze(1)
            self.metric.update(label_.detach().cpu(),output_.detach().cpu())

        # Log a small number of visual examples during test
        if batch_idx < 3:
            try:
                import wandb
    
                # Use the first sample in the batch
                pred_img = output[0].detach().cpu().squeeze().numpy()
                gt_img = label[0].detach().cpu().squeeze().numpy()
                err_img = np.abs(pred_img - gt_img)

                # inferno good fot hotspot style intensity maps, viridis good for error maps
                pred_img = to_colormap(pred_img, "inferno")
                gt_img = to_colormap(gt_img, "inferno")
                err_img = to_colormap(err_img, "viridis")
    
                if hasattr(self.logger, "experiment"):
                    self.logger.experiment.log({
                        f"test/prediction_{batch_idx}": wandb.Image(pred_img, caption=f"Prediction batch {batch_idx}"),
                        f"test/ground_truth_{batch_idx}": wandb.Image(gt_img, caption=f"Ground truth batch {batch_idx}"),
                        f"test/error_map_{batch_idx}": wandb.Image(err_img, caption=f"Error map batch {batch_idx}"),
                    })
            except Exception as e:
                print(f"[WARN] Visualization logging failed at batch {batch_idx}: {e}")

    def on_test_epoch_end(self):
        # Make the Progress Bar leave there
        pearson, spearman, kendall = self.metric.compute()
        self.log('pearson', pearson, on_step=False, on_epoch=True, batch_size=self.cfg.batch_size, sync_dist=True)
        self.log('spearman', spearman, on_step=False, on_epoch=True, batch_size=self.cfg.batch_size, sync_dist=True)
        self.log('kendall', kendall, on_step=False, on_epoch=True, batch_size=self.cfg.batch_size, sync_dist=True)
        print("pearson:", pearson, "spearman:", spearman, "kendall:", kendall)
        self.metric.reset()

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        pearson, spearman, kendall = self.metric.compute()
        self.log('pearson', pearson, on_step=False, on_epoch=True, batch_size=self.cfg.batch_size, sync_dist=True)
        self.log('spearman', spearman, on_step=False, on_epoch=True, batch_size=self.cfg.batch_size, sync_dist=True)
        self.log('kendall', kendall, on_step=False, on_epoch=True, batch_size=self.cfg.batch_size, sync_dist=True)
        print("pearson:", pearson, "spearman:", spearman, "kendall:", kendall)
        self.metric.reset()



    def configure_optimizers(self):
        if getattr(self.cfg, 'weight_decay'):
            weight_decay = self.cfg.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=weight_decay)

        if self.cfg.lr_scheduler is None:
            return optimizer
        else:
            if self.cfg.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.cfg.lr_decay_steps,
                                       gamma=self.cfg.lr_decay_rate)
            elif self.cfg.lr_scheduler == 'cosine':
                # scheduler = lrs.CosineAnnealingLR(optimizer,
                #                                   T_max=self.cfg.lr_decay_steps,
                #                                   eta_min=self.cfg.lr_decay_min_lr)
                scheduler = CosineLRScheduler(optimizer,
                                              t_initial = self.cfg.max_epochs,
                                              lr_min=self.cfg.min_lr,
                                              warmup_lr_init=self.cfg.warmup_lr,
                                              warmup_t=self.cfg.warmup_epochs,
                                              cycle_limit=1,
                                              t_in_epochs=True,
                                                )
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
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
        Model = CircuitFormer()
        print(Model)
        self.model = Model

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
