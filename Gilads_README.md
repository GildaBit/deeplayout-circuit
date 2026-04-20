# CircuitFormer Setup Guide (CircuitNet-N28)

This document describes the full process required to run CircuitFormer on CircuitNet-N28, including dataset setup, environment configuration, required fixes, and visualization.

---

## 1. Dataset Preparation

### Dataset Location
`/home/jovyan/shared/ctrl-nxp/CircuitNet-N28/`

### Required Components
- Graph features: `graph_features/instance_placement_gcell.tar.gz`
- Labels: `training_set/congestion/label/`

### Extract Graph Features

```bash
mkdir -p /tmp/circuitnet_work

tar -xzf /home/jovyan/shared/ctrl-nxp/CircuitNet-N28/graph_features/instance_placement_gcell.tar.gz -C /tmp/circuitnet_work
```


Rename extracted folder:

```bash
mv /tmp/circuitnet_work/instance_placement_gcell /tmp/circuitnet_work/instance_placement
```


---

## 2. Train/Test Split (Paper-Aligned)

### Source (Training)
- RISCY-a  
- RISCY-b  
- RISCY-FPU-a  
- RISCY-FPU-b  

### Target (Testing)
- zero-riscy-a  
- zero-riscy-b  

### Generated Split Files


ckt_splits/
├── pretrain.txt
├── val_100.txt
├── test_100.txt
├── finetune_5.txt
├── finetune_10.txt
├── finetune_20.txt


Copy into repo:

```bash
cp ckt_splits/pretrain.txt data/train.txt
cp ckt_splits/val_100.txt data/val.txt
cp ckt_splits/test_100.txt data/test.txt
```

---

## 3. Environment Setup

### Install Requirements

```bash
pip install -r requirements.txt
```


### Additional Required Packages

```bash
# IMPORTANT NOTE INSTALL DIFF VERSION OF TORCH SCATTER IF GPU ACCESS
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html --force-reinstall
pip install spconv
pip install --upgrade wandb
pip install --upgrade typing_extensions
```


---

## 4. Repository Fixes

### Fix Dataset Loading

```python
data_path = os.path.join(self.data_root, self.data_list[data_idx])
data_dict = np.load(data_path, allow_pickle=True).item()
data = np.array(list(data_dict.values()))
```


### Fix Split Parsing

```python
self.data_list = [line.strip() for line in f if line.strip()]
```


### Fix Hardcoded Paths

```python
os.path.join(os.path.dirname(file), "train.txt")
```


### Fix Checkpoint Path

```python
ckpt_path = os.path.abspath(
    os.path.join(os.path.dirname(file), '..', 'ckpts', 'resnet18.pth')
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(ckpt_path, map_location=device)
```


### Dynamic GPU + BatchNorm Handling

```python
import torch

gpu_count = torch.cuda.device_count()

if gpu_count > 1:
cfg.trainer.sync_batchnorm = True
cfg.trainer.strategy = "ddp"
else:
cfg.trainer.sync_batchnorm = False
```

### Fix WandB Config Serialization

```python
from omegaconf import OmegaConf
config = OmegaConf.to_container(cfg, resolve=True)
```


### WandB Safe Fallback
- Use WandB if logged in
- Otherwise fallback to CSVLogger
- Prevents crashes for users without API keys

---

## 5. Checkpoints Setup

Create directory:

```bash
mkdir -p ckpts
cd ckpts
```


### ResNet Backbone

```bash
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
ln -s resnet18-f37072fd.pth resnet18.pth
```


### CircuitFormer Model
Download from README (Google Drive or Baidu) and place:

```bash
ckpts/circuitformer.ckpt
```


---

## 6. Trainer Configuration


trainer:
accelerator: auto
devices: auto
precision: 32
sync_batchnorm: False


---

## 7. Running the Model

From repo root:

```bash
python test.py data.data_root=/home/jovyan/shared/ctrl-nxp/CircuitNet-N28/micron_instance_placement/instance_placement/ data.label_root=/home/jovyan/shared/ctrl-nxp/CircuitNet-N28/training_set/congestion/label/
```


---

## 8. Visualization (WandB)

### Add Visualization in `test_step`

Logs:
- prediction
- ground truth
- error map

### Colormap Usage
- inferno → predictions
- inferno → ground truth
- viridis → error

### Fix Visualization Bug

```python
pred_img = to_colormap(pred_img, "inferno")
gt_img = to_colormap(gt_img, "inferno")
err_img = to_colormap(err_img, "viridis")
```


---

## Final Status

- Dataset loading: ✔  
- Graph features: ✔  
- Model initialization: ✔  
- Checkpoints loaded: ✔  
- Inference completed: ✔  
- Metrics computed: ✔  
- Visualization working: ✔  

Example results:

Pearson ≈ 0.51
Spearman ≈ 0.47
Kendall ≈ 0.35