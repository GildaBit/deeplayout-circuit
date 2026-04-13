import torch
import pytorch_lightning as pl
import hydra
from pathlib import Path

from model import MInterface
from data import DInterface
from utils import setup_config

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    callbacks = setup_config(cfg)
    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=True)

    data_module = DInterface(cfg.data)
    model = MInterface(cfg.model)

    # safer default training setup
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        cfg.trainer.accelerator = "gpu"

        # start simple: 1 GPU unless you explicitly change config later
        if cfg.trainer.devices == "auto":
            cfg.trainer.devices = 1
    else:
        gpu_count = 0
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1
        
    cfg.trainer.sync_batchnorm = False
    if "strategy" in cfg.trainer:
        cfg.trainer.strategy = "auto"

    print(f"[INFO] GPUs detected: {gpu_count}")
    print(f"[INFO] Trainer accelerator: {cfg.trainer.accelerator}")
    print(f"[INFO] Trainer devices: {cfg.trainer.devices}")

    trainer = pl.Trainer(**cfg.trainer, **callbacks)

    ckpt_path = cfg.experiment.get("ckpt_path", None)
    
    if ckpt_path:
        print(f"[INFO] Initializing model weights from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("[INFO] Loaded pretrained weights only; starting fresh optimizer/scheduler state")
        print(f"[INFO] Missing keys: {len(missing)}")
        print(f"[INFO] Unexpected keys: {len(unexpected)}")
    else:
        print("[INFO] Training from scratch")
    
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()