import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def aggregate_mean(src_feat: torch.Tensor, edge_index: torch.Tensor, num_dst: int) -> torch.Tensor:
    """
    Mean aggregation from source nodes to destination nodes.
    edge_index[0] = source, edge_index[1] = destination
    """
    device = src_feat.device
    dtype = src_feat.dtype

    out = torch.zeros(num_dst, src_feat.size(1), device=device, dtype=dtype)
    deg = torch.zeros(num_dst, 1, device=device, dtype=dtype)

    if edge_index.numel() == 0:
        return out

    src_idx = edge_index[0]
    dst_idx = edge_index[1]

    out.index_add_(0, dst_idx, src_feat[src_idx])

    ones = torch.ones(dst_idx.numel(), 1, device=device, dtype=dtype)
    deg.index_add_(0, dst_idx, ones)

    return out / deg.clamp(min=1.0)


class CellNetMessagePassing(nn.Module):
    """
    One hetero message-passing block:
      cells -> nets
      nets  -> cells
      optional cells -> cells
    """
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.cell_to_net_update = MLP(hidden_dim * 2, hidden_dim, hidden_dim)
        self.net_to_cell_update = MLP(hidden_dim * 2, hidden_dim, hidden_dim)
        self.cell_to_cell_update = MLP(hidden_dim * 2, hidden_dim, hidden_dim)

        self.cell_norm = nn.LayerNorm(hidden_dim)
        self.net_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        cell_h: torch.Tensor,
        net_h: torch.Tensor,
        cell_to_net_edge_index: torch.Tensor,
        net_to_cell_edge_index: torch.Tensor,
        cell_to_cell_edge_index: torch.Tensor | None = None,
    ):
        msg_c2n = aggregate_mean(cell_h, cell_to_net_edge_index, num_dst=net_h.size(0))
        net_h = self.net_norm(net_h + self.cell_to_net_update(torch.cat([net_h, msg_c2n], dim=-1)))

        msg_n2c = aggregate_mean(net_h, net_to_cell_edge_index, num_dst=cell_h.size(0))
        cell_h = cell_h + self.net_to_cell_update(torch.cat([cell_h, msg_n2c], dim=-1))

        if cell_to_cell_edge_index is not None and cell_to_cell_edge_index.numel() > 0:
            msg_c2c = aggregate_mean(cell_h, cell_to_cell_edge_index, num_dst=cell_h.size(0))
            cell_h = cell_h + self.cell_to_cell_update(torch.cat([cell_h, msg_c2c], dim=-1))

        cell_h = self.cell_norm(cell_h)
        return cell_h, net_h


class GraphEncoderDeepLayout(nn.Module):
    """
    Heterogeneous graph encoder.

    Cell input:
        [cx, cy, x1, y1, x2, y2, w, h, area]

    Net input:
        [degree, span_x, span_y]
    """
    def __init__(
        self,
        cell_in_dim: int = 9,
        net_in_dim: int = 3,
        hidden_dim: int = 32,
        out_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.cell_proj = MLP(cell_in_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.net_proj = MLP(net_in_dim, hidden_dim, hidden_dim, dropout=dropout)

        self.layers = nn.ModuleList([
            CellNetMessagePassing(hidden_dim) for _ in range(num_layers)
        ])

        self.out = MLP(hidden_dim, hidden_dim, out_dim, dropout=dropout)

    def forward(
        self,
        cell_x: torch.Tensor,
        net_x: torch.Tensor,
        cell_to_net_edge_index: torch.Tensor,
        net_to_cell_edge_index: torch.Tensor,
        cell_to_cell_edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cell_h = self.cell_proj(cell_x)
        net_h = self.net_proj(net_x)

        for layer in self.layers:
            cell_h, net_h = layer(
                cell_h,
                net_h,
                cell_to_net_edge_index,
                net_to_cell_edge_index,
                cell_to_cell_edge_index,
            )

        return self.out(cell_h)