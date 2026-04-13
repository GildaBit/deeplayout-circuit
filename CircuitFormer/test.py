import pytorch_lightning as pl
import hydra
import torch

from model import MInterface
from data import DInterface
from utils import setup_config
from pathlib import Path

@hydra.main(config_path='config', config_name='config')
def main(cfg):
    callbacks = setup_config(cfg)
    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    data_module = DInterface(cfg.data)
    model = MInterface(cfg.model)

    # AUTO DETECT GPU SETUP
    if cfg.trainer.accelerator in ["gpu", "auto"]:
        gpu_count = torch.cuda.device_count()
    else:
        gpu_count = 0

    if gpu_count > 1:
        cfg.trainer.sync_batchnorm = True
        cfg.trainer.strategy = "ddp"
    else:
        cfg.trainer.sync_batchnorm = False

    print(f"[INFO] GPUs detected: {gpu_count}, sync_batchnorm={cfg.trainer.sync_batchnorm}")

    trainer = pl.Trainer(**cfg.trainer,
                         **callbacks,)
    trainer.test(model, data_module, ckpt_path=cfg.experiment.ckpt_path)

if __name__ == '__main__':
    main()
