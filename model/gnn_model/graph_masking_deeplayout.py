import torch


def compute_grid_ids(cell_xy_grid: torch.Tensor, grid_size: int = 256, mask_bin_size: int = 16) -> torch.Tensor:
    """
    Group nearby cells into coarse spatial masking regions.
    """
    xy = cell_xy_grid.round().long().clamp(0, grid_size - 1)

    gx = xy[:, 0] // mask_bin_size
    gy = xy[:, 1] // mask_bin_size

    bins_per_axis = max(grid_size // mask_bin_size, 1)
    return gy * bins_per_axis + gx


def sample_masked_regions(grid_ids: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
    """
    Randomly choose a fraction of non-empty regions to mask.
    """
    unique_regions = torch.unique(grid_ids)
    if unique_regions.numel() == 0:
        return unique_regions

    num_regions = unique_regions.numel()
    num_to_mask = max(1, int(round(num_regions * mask_ratio)))
    num_to_mask = min(num_to_mask, num_regions)

    perm = torch.randperm(num_regions, device=grid_ids.device)
    return unique_regions[perm[:num_to_mask]]


def build_cell_mask_from_regions(grid_ids: torch.Tensor, masked_region_ids: torch.Tensor) -> torch.Tensor:
    """
    Convert masked region ids into a bool mask over cells.
    """
    if masked_region_ids.numel() == 0:
        return torch.zeros_like(grid_ids, dtype=torch.bool)

    return (grid_ids[:, None] == masked_region_ids[None, :]).any(dim=1)


def apply_feature_mask_to_cells(
    cell_feat: torch.Tensor,
    cell_mask: torch.Tensor,
    coord_dims=(0, 1, 2, 3, 4, 5),
) -> torch.Tensor:
    """
    Zero coordinate-related features for masked cells, while keeping
    size-related features visible.

    Expected feature order:
        [cx, cy, x1, y1, x2, y2, w, h, area]
    """
    masked_cell_feat = cell_feat.clone()

    if cell_mask.any():
        row_idx = torch.where(cell_mask)[0]
        col_idx = torch.tensor(coord_dims, device=cell_feat.device)
        masked_cell_feat[row_idx[:, None], col_idx[None, :]] = 0.0

    return masked_cell_feat


def create_layout_masked_cell_features(
    cell_feat: torch.Tensor,
    cell_xy_grid: torch.Tensor,
    grid_size: int = 256,
    mask_bin_size: int = 16,
    mask_ratio: float = 0.5,
):
    """
    Full masking pipeline for the graph branch.
    """
    grid_ids = compute_grid_ids(
        cell_xy_grid=cell_xy_grid,
        grid_size=grid_size,
        mask_bin_size=mask_bin_size,
    )

    masked_region_ids = sample_masked_regions(
        grid_ids=grid_ids,
        mask_ratio=mask_ratio,
    )

    cell_mask = build_cell_mask_from_regions(
        grid_ids=grid_ids,
        masked_region_ids=masked_region_ids,
    )

    masked_cell_feat = apply_feature_mask_to_cells(
        cell_feat=cell_feat,
        cell_mask=cell_mask,
    )

    return masked_cell_feat, cell_mask, grid_ids