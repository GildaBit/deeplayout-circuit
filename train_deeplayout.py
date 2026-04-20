import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data.gnn_data.data_interface_deeplayout import DInterfaceDeeplayout
from model_interface_deeplayout import MInterfaceDeeplayout
from pytorch_lightning.loggers import WandbLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLayout-style hybrid model on CircuitNet.")

    # data paths
    parser.add_argument("--data_root", type=str, required=True, help="Path to geometry/placement samples.")
    parser.add_argument("--label_root", type=str, required=True, help="Path to dense label files.")
    parser.add_argument("--graph_root", type=str, required=True, help="Path to graph feature files.")
    parser.add_argument("--graph_ext", type=str, default=".npy", help="Graph file extension.")
    parser.add_argument("--save_dir", type=str, default="./experiments/deeplayout_run", help="Checkpoint/log output dir.")

    # training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--devices", type=str, default="1", help="Use '1' for single GPU or CPU fallback.")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--seed", type=int, default=42)

    # optimization
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--loss_weight", type=float, default=1.0)
    parser.add_argument("--coord_loss_weight", type=float, default=0.1)
    parser.add_argument("--label_weight", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--lr_decay_steps", type=int, default=10)
    parser.add_argument("--lr_decay_rate", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    # misc
    parser.add_argument("--resnet_ckpt_path", type=str, default=None, help="Optional resnet18 encoder checkpoint.")
    parser.add_argument("--max_test_visualizations", type=int, default=6)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint path for resume/test.")

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    datamodule = DInterfaceDeeplayout(
        data_root=args.data_root,
        label_root=args.label_root,
        graph_root=args.graph_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        graph_ext=args.graph_ext,
        loop=1,
    )

    lr_scheduler = None if args.lr_scheduler == "none" else args.lr_scheduler

    model = MInterfaceDeeplayout(
        lr=args.lr,
        loss=args.loss,
        loss_weight=args.loss_weight,
        coord_loss_weight=args.coord_loss_weight,
        label_weight=args.label_weight,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        lr_scheduler=lr_scheduler,
        lr_decay_steps=args.lr_decay_steps,
        lr_decay_rate=args.lr_decay_rate,
        min_lr=args.min_lr,
        warmup_lr=args.warmup_lr,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        max_test_visualizations=args.max_test_visualizations,
        resnet_ckpt_path=args.resnet_ckpt_path,
    )

    # save a checkpoint every 10 epochs, plus last checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_dir / "checkpoints"),
        filename="epoch{epoch:03d}",
        every_n_epochs=10,
        save_top_k=-1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = int(args.devices)
    else:
        accelerator = "cpu"
        devices = 1

    wandb_logger = WandbLogger(
        project="deeplayout",
        name=save_dir.name,
        save_dir=str(save_dir),
        job_type=args.mode,
    )
    wandb_logger.experiment.config.update(vars(args))
    
    trainer = pl.Trainer(
        default_root_dir=str(save_dir),
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10,
        sync_batchnorm=False,
        accumulate_grad_batches=4,
    )

    if args.mode == "train":
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    else:
        if args.ckpt_path is None:
            raise ValueError("--ckpt_path is required in test mode.")
        trainer.test(model, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()