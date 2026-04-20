import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from model.voxelset.voxset import VoxSeT
from model.gnn_model.graph_encoder_deeplayout import GraphEncoderDeepLayout
from model.gnn_model.graph_rasterizer_deeplayout import rasterize_cell_features
from model.gnn_model.graph_masking_deeplayout import create_layout_masked_cell_features

def _break_up_pc(pc):
    """
    pc = [x1, y1, x2, y2, offset]

    Builds the same box-derived features used by the old VoxSeT path:
        [cx, cy, x1, y1, x2, y2, w, h, area]
    """
    for i in range(len(pc)):
        pc[i] = torch.unsqueeze(pc[i], dim=-1)

    x = (pc[0] + pc[2]) / 2
    y = (pc[1] + pc[3]) / 2
    width = pc[2] - pc[0]
    height = pc[3] - pc[1]
    area = width * height

    features = torch.concat([x, y, pc[0], pc[1], pc[2], pc[3], width, height, area], dim=-1)
    offset = pc[4]
    return features, offset


def _build_voxset_input(points, offset, grid_size=256):
    """
    Convert concatenated point features into VoxSeT input format.

    Output per point:
        [batch_idx, cx, cy, x1, y1, x2, y2, w, h, area]
    """
    points_list = []

    for i in range(len(offset)):
        if i == 0:
            pts = points[:offset[i]].clone()
        else:
            pts = points[offset[i - 1]:offset[i]].clone()

        # normalize center coordinates for the spatial path
        pts[:, 0] = pts[:, 0] / pts[:, 0].max().clamp(min=1e-6) * (grid_size - 1)
        pts[:, 1] = pts[:, 1] / pts[:, 1].max().clamp(min=1e-6) * (grid_size - 1)

        batch_index = i * torch.ones_like(pts[:, 0])
        points_list.append(torch.cat([batch_index[:, None], pts], dim=-1))

    return torch.cat(points_list, dim=0)


class HybridEncoderDeepLayout(nn.Module):
    """
    Hybrid encoder:
      1) existing geometry branch (VoxSeT)
      2) new graph branch (heterogeneous GNN + masking)
      3) fuse both branches into a dense feature map
    """
    def __init__(
        self,
        grid_size=256,
        graph_hidden_dim=32,
        graph_out_dim=32,
        mask_ratio=0.5,
        mask_bin_size=16,
        enable_masking=True,
    ):
        super().__init__()

        self.grid_size = grid_size
        self.mask_ratio = mask_ratio
        self.mask_bin_size = mask_bin_size
        self.enable_masking = enable_masking

        self.layout_encoder = VoxSeT(grid_size=grid_size)

        self.graph_encoder = GraphEncoderDeepLayout(
            cell_in_dim=9,
            net_in_dim=3,
            hidden_dim=graph_hidden_dim,
            out_dim=graph_out_dim,
            num_layers=2,
            dropout=0.0,
        )

        # auxiliary head for masked coordinate reconstruction
        self.coord_head = nn.Sequential(
            nn.Linear(graph_out_dim, graph_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(graph_out_dim, 2),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(64 + graph_out_dim, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, batch):
        """
        batch format:
        {
          "geom": [x1, y1, x2, y2, offset],
          "graph": {
              "cell_feat": ...
              "net_feat": ...
              "cell_to_net_edge_index": ...
              "net_to_cell_edge_index": ...
              "cell_to_cell_edge_index": ...
              "cell_xy_grid": ...
              "batch_index": ...
              "coord_target": ...
          }
        }

        Returns:
            fused feature map for main decoder
            aux dict for masked coordinate loss
        """
        geom = batch["geom"]
        graph = batch["graph"]

        # ----- geometry branch -----
        points, offset = _break_up_pc(geom)
        vox_input = _build_voxset_input(points, offset, self.grid_size)
        layout_feat = self.layout_encoder(vox_input)

        # ----- graph masking -----
        if self.enable_masking and self.training:
            masked_cell_feat, cell_mask, grid_ids = create_layout_masked_cell_features(
                cell_feat=graph["cell_feat"],
                cell_xy_grid=graph["cell_xy_grid"],
                grid_size=self.grid_size,
                mask_bin_size=self.mask_bin_size,
                mask_ratio=self.mask_ratio,
            )
        else:
            masked_cell_feat = graph["cell_feat"]
            cell_mask = torch.zeros(
                graph["cell_feat"].shape[0],
                dtype=torch.bool,
                device=graph["cell_feat"].device,
            )
            grid_ids = torch.zeros(
                graph["cell_feat"].shape[0],
                dtype=torch.long,
                device=graph["cell_feat"].device,
            )

        # ----- graph branch -----
        cell_emb = self.graph_encoder(
            masked_cell_feat,
            graph["net_feat"],
            graph["cell_to_net_edge_index"],
            graph["net_to_cell_edge_index"],
            graph.get("cell_to_cell_edge_index", None),
        )

        coord_pred = self.coord_head(cell_emb)

        graph_plane = rasterize_cell_features(
            graph["cell_xy_grid"],
            cell_emb,
            graph["batch_index"],
            grid_size=self.grid_size,
        )

        if graph_plane.shape[-2:] != layout_feat.shape[-2:]:
            graph_plane = F.interpolate(
                graph_plane,
                size=layout_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fused = self.fuse(torch.cat([layout_feat, graph_plane], dim=1))

        aux = {
            "coord_pred": coord_pred,
            "coord_target": graph["coord_target"],
            "cell_mask": cell_mask,
            "grid_ids": grid_ids,
        }
        return fused, aux


class CircuitFormerDeepLayout(nn.Module):
    """
    Main model:
        HybridEncoderDeepLayout -> Unet++
    """
    def __init__(self, resnet_ckpt_path=None):
        super().__init__()

        self.encoder = HybridEncoderDeepLayout(
            grid_size=256,
            graph_hidden_dim=64,
            graph_out_dim=64,
            mask_ratio=0.5,
            mask_bin_size=16,
            enable_masking=True,
        )

        self.decoder = smp.UnetPlusPlus(
            encoder_name="resnet18",
            encoder_depth=5,
            decoder_use_batchnorm=True,
            decoder_channels=(512, 256, 128, 64, 64),
            decoder_attention_type=None,
            in_channels=64,
            classes=1,
            activation="sigmoid",
            aux_params=None,
        )

        if resnet_ckpt_path is not None and os.path.exists(resnet_ckpt_path):
            ckpt = torch.load(resnet_ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            ckpt.pop("conv1.weight", None)
            self.decoder.encoder.load_state_dict(ckpt, strict=False)
        elif resnet_ckpt_path is not None:
            print(f"[WARN] Decoder backbone checkpoint not found: {resnet_ckpt_path}. Continuing without it.")

    def forward(self, batch):
        fused, aux = self.encoder(batch)
        output = self.decoder(fused)
        return output, aux