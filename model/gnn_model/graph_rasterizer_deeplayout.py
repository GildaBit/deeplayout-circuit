import torch


def rasterize_cell_features(
    cell_xy_grid: torch.Tensor,
    cell_feat: torch.Tensor,
    batch_index: torch.Tensor,
    grid_size: int = 256,
) -> torch.Tensor:
    """
    Splat per-cell embeddings onto a dense 2D grid.

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

    out = torch.zeros(B, C, grid_size, grid_size, device=device, dtype=dtype)
    cnt = torch.zeros(B, 1, grid_size, grid_size, device=device, dtype=dtype)

    xy = cell_xy_grid.round().long()
    x = xy[:, 0].clamp(0, grid_size - 1)
    y = xy[:, 1].clamp(0, grid_size - 1)

    for i in range(N):
        b = int(batch_index[i].item())
        out[b, :, y[i], x[i]] += cell_feat[i]
        cnt[b, 0, y[i], x[i]] += 1.0

    out = out / cnt.clamp(min=1.0)
    return out