import torch


def rasterize_cell_features(
    cell_xy_grid: torch.Tensor,
    cell_feat: torch.Tensor,
    batch_index: torch.Tensor,
    grid_size: int = 256,
) -> torch.Tensor:
    """
    Bilinearly splat per-cell embeddings onto a dense 2D grid.

    Output:
        [B, C, grid_size, grid_size]
    """
    assert cell_xy_grid.dim() == 2 and cell_xy_grid.size(1) == 2
    assert cell_feat.dim() == 2
    assert batch_index.dim() == 1

    device = cell_feat.device
    dtype = cell_feat.dtype

    B = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 1
    N, C = cell_feat.shape

    if N == 0:
        return torch.zeros(B, C, grid_size, grid_size, device=device, dtype=dtype)

    xy = cell_xy_grid.clamp(0, grid_size - 1)

    x = xy[:, 0]
    y = xy[:, 1]

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = (x0 + 1).clamp(max=grid_size - 1)
    y1 = (y0 + 1).clamp(max=grid_size - 1)

    wx1 = x - x0.float()
    wy1 = y - y0.float()
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    w00 = wx0 * wy0
    w01 = wx0 * wy1
    w10 = wx1 * wy0
    w11 = wx1 * wy1

    # 4 neighbors for each point
    b_all = torch.cat([batch_index, batch_index, batch_index, batch_index], dim=0)
    x_all = torch.cat([x0, x0, x1, x1], dim=0)
    y_all = torch.cat([y0, y1, y0, y1], dim=0)
    w_all = torch.cat([w00, w01, w10, w11], dim=0)

    # Flatten [B, H, W] -> [B*H*W]
    flat_idx = b_all * (grid_size * grid_size) + y_all * grid_size + x_all

    out_flat = torch.zeros(B * grid_size * grid_size, C, device=device, dtype=dtype)
    cnt_flat = torch.zeros(B * grid_size * grid_size, 1, device=device, dtype=dtype)

    feat_all = torch.cat([
        cell_feat * w00.unsqueeze(1),
        cell_feat * w01.unsqueeze(1),
        cell_feat * w10.unsqueeze(1),
        cell_feat * w11.unsqueeze(1),
    ], dim=0)

    cnt_all = w_all.unsqueeze(1)

    out_flat.index_add_(0, flat_idx, feat_all)
    cnt_flat.index_add_(0, flat_idx, cnt_all)

    out_flat = out_flat / cnt_flat.clamp(min=1e-6)

    out = out_flat.view(B, grid_size, grid_size, C).permute(0, 3, 1, 2).contiguous()
    return out