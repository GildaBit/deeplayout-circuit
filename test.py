import pytorch_lightning as pl
import hydra
import torch
torch.set_float32_matmul_precision("high")

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

    # force a clean single-GPU test run
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.devices = 1
    else:
        gpu_count = 0
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1

    cfg.trainer.sync_batchnorm = False

    # only set this if the key exists in config
    if "strategy" in cfg.trainer:
        cfg.trainer.strategy = "auto"

    print(f"[INFO] GPUs detected: {gpu_count}")
    print(f"[INFO] Trainer accelerator: {cfg.trainer.accelerator}")
    print(f"[INFO] Trainer devices: {cfg.trainer.devices}")
    print(f"[INFO] Trainer strategy: {cfg.trainer.get('strategy', 'not-set')}")
    print(f"[INFO] sync_batchnorm: {cfg.trainer.sync_batchnorm}")

    trainer = pl.Trainer(**cfg.trainer,
                         **callbacks,)
    trainer.test(model, data_module, ckpt_path=cfg.experiment.ckpt_path)

if __name__ == '__main__':
    main()
