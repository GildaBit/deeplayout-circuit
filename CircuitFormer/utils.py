import os
from pathlib2 import Path
from omegaconf import OmegaConf, DictConfig
from collections.abc import Callable
from typing import Optional
import pytorch_lightning as pl

def load_model_path(root=None, version=None, v_num=None, best=False):
    """ When best = True, return the best model's path in a directory 
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the 
        first three args.
    Args: 
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """
    def sort_by_epoch(path):
        name = path.stem
        epoch=int(name.split('-')[1].split('=')[1])
        return epoch
    
    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('lightning_logs', version, 'checkpoints'))
        else:
            return str(Path('lightning_logs', f'version_{v_num}', 'checkpoints'))

    if root==version==v_num==None:
        return None

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files=[i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
    return res

def load_model_path_by_args(args):
    return load_model_path(root=args.load_dir, version=args.load_ver, v_num=args.load_v_num)

def setup_config(cfg: DictConfig, override: Optional[Callable] = None):
    OmegaConf.set_struct(cfg, False)

    if override is not None:
        override(cfg)

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)

    save_dir = Path(cfg.experiment.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert Hydra config to a plain Python dict
    # (W&B cannot handle OmegaConf objects directly)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    loggers = []
    use_wandb = False

    # Check if W&B is available and user is already authenticated
    try:
        import wandb  # only used to test availability

        # Detect existing login via environment or .netrc
        api_key_in_env = bool(os.environ.get("WANDB_API_KEY"))
        api_key_in_netrc = Path.home().joinpath(".netrc").exists()

        if api_key_in_env or api_key_in_netrc:
            use_wandb = True
    except Exception:
        # If wandb is not installed or fails to import
        use_wandb = False

    if use_wandb:
        try:
            # Use W&B for experiment tracking if available
            loggers.append(
                pl.loggers.WandbLogger(
                    project=cfg.experiment.project,
                    save_dir=cfg.experiment.save_dir,
                    config=cfg_dict,  # must be plain dict
                )
            )
            print("[INFO] Using WandbLogger")
        except Exception as e:
            # If W&B initialization fails, fall back safely
            print(f"[WARN] W&B setup failed: {e}")
            loggers.append(
                pl.loggers.CSVLogger(
                    save_dir=cfg.experiment.save_dir,
                    name="csv_logs",
                )
            )
            print("[INFO] Falling back to CSVLogger")
    else:
        # Default fallback when W&B is not configured
        loggers.append(
            pl.loggers.CSVLogger(
                save_dir=cfg.experiment.save_dir,
                name="csv_logs",
            )
        )
        print("[INFO] W&B not configured, using CSVLogger")

    # Standard callbacks for training
    callbacks = {
        "logger": loggers,
        "callbacks": [
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint(
                dirpath=cfg.experiment.save_dir,
                filename="{epoch}-{pearson:.4f}",
                monitor="pearson",
                save_last=True,
                mode="max",
            ),
        ],
    }
    print(cfg)
    return callbacks