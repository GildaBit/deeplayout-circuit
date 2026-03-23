import sys
from pathlib import Path

# Add repo root to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from pathlib import Path
import yaml

# for being able to easily reference root paths for dataset and files
def load_paths(config_path="configs/paths.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    circuitnet_root = Path(cfg["circuitnet_root"]).expanduser()
    output_root = Path(cfg["output_root"]).expanduser()

    return {
        "circuitnet_root": circuitnet_root,
        "output_root": output_root
    }