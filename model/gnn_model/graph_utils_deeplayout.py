import numpy as np
import torch


def build_cell_features(x1, y1, x2, y2):
    """
    Per-cell geometry features.

    Feature order:
        [cx, cy, x1, y1, x2, y2, w, h, area]
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    area = w * h

    cell_feat = torch.stack([cx, cy, x1, y1, x2, y2, w, h, area], dim=-1)
    cell_xy = torch.stack([cx, cy], dim=-1)
    return cell_feat, cell_xy


def normalize_xy_to_grid(cell_xy: torch.Tensor, batch_index: torch.Tensor, grid_size: int = 256):
    """
    Normalize cell centers independently per sample into [0, grid_size - 1].

    batch_index matters after batching, because all cells from multiple samples
    are concatenated together.
    """
    out = torch.zeros_like(cell_xy)
    num_batches = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 1

    for b in range(num_batches):
        mask = batch_index == b
        xy = cell_xy[mask]
        if xy.numel() == 0:
            continue

        x = xy[:, 0]
        y = xy[:, 1]

        x_max = x.max().clamp(min=1e-6)
        y_max = y.max().clamp(min=1e-6)

        out[mask, 0] = x / x_max * (grid_size - 1)
        out[mask, 1] = y / y_max * (grid_size - 1)

    return out


def build_bipartite_edges(nets, device=None):
    """
    Convert:
        nets = [[cell ids for net0], [cell ids for net1], ...]

    into:
        cell_to_net_edge_index: [2, E]
        net_to_cell_edge_index: [2, E]
    """
    cell_src = []
    net_dst = []

    for net_idx, cell_ids in enumerate(nets):
        for c in cell_ids:
            cell_src.append(c)
            net_dst.append(net_idx)

    if len(cell_src) == 0:
        cell_to_net = torch.zeros(2, 0, dtype=torch.long, device=device)
        net_to_cell = torch.zeros(2, 0, dtype=torch.long, device=device)
        return cell_to_net, net_to_cell

    cell_src = torch.tensor(cell_src, dtype=torch.long, device=device)
    net_dst = torch.tensor(net_dst, dtype=torch.long, device=device)

    cell_to_net = torch.stack([cell_src, net_dst], dim=0)
    net_to_cell = torch.stack([net_dst, cell_src], dim=0)
    return cell_to_net, net_to_cell


def build_net_features(cell_xy: torch.Tensor, nets, device=None):
    """
    Net features:
        [degree, span_x, span_y]
    """
    feats = []
    for cell_ids in nets:
        if len(cell_ids) == 0:
            feats.append([0.0, 0.0, 0.0])
            continue

        pts = cell_xy[cell_ids]
        xs = pts[:, 0]
        ys = pts[:, 1]

        degree = float(len(cell_ids))
        span_x = float(xs.max() - xs.min()) if xs.numel() > 0 else 0.0
        span_y = float(ys.max() - ys.min()) if ys.numel() > 0 else 0.0
        feats.append([degree, span_x, span_y])

    if len(feats) == 0:
        return torch.zeros(0, 3, dtype=cell_xy.dtype, device=device if device is not None else cell_xy.device)

    return torch.tensor(feats, dtype=cell_xy.dtype, device=device if device is not None else cell_xy.device)


def build_cell_to_cell_edges_from_nets(nets, device=None):
    """
    Optional cell-cell edges by clique expansion inside each net.
    This is a simple and readable approximation.
    """
    src = []
    dst = []

    for cell_ids in nets:
        uniq = list(dict.fromkeys(cell_ids))
        for i in range(len(uniq)):
            for j in range(len(uniq)):
                if i == j:
                    continue
                src.append(uniq[i])
                dst.append(uniq[j])

    if len(src) == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    return torch.tensor([src, dst], dtype=torch.long, device=device)


def pin_maps_to_nets(pin_to_cell, pin_to_net, num_cells=None):
    """
    CircuitNet graph features can be built from pin->cell and pin->net mappings.
    This helper converts them into the simple nets list format. :contentReference[oaicite:3]{index=3}
    """
    pin_to_cell = np.asarray(pin_to_cell).astype(np.int64)
    pin_to_net = np.asarray(pin_to_net).astype(np.int64)

    net_to_cells = {}
    for c, n in zip(pin_to_cell, pin_to_net):
        if n not in net_to_cells:
            net_to_cells[n] = []
        net_to_cells[n].append(int(c))

    nets = []
    for net_id in sorted(net_to_cells.keys()):
        cell_ids = sorted(set(net_to_cells[net_id]))
        if num_cells is not None:
            cell_ids = [c for c in cell_ids if 0 <= c < num_cells]
        if len(cell_ids) > 0:
            nets.append(cell_ids)

    return nets


def edge_index_cell_net_to_nets(edge_index_cell_net, num_cells=None):
    """
    Support graph files stored directly as [2, E] cell-net incidence edges.
    """
    edge_index_cell_net = np.asarray(edge_index_cell_net).astype(np.int64)
    assert edge_index_cell_net.ndim == 2 and edge_index_cell_net.shape[0] == 2

    cell_ids = edge_index_cell_net[0]
    net_ids = edge_index_cell_net[1]

    net_to_cells = {}
    for c, n in zip(cell_ids, net_ids):
        if n not in net_to_cells:
            net_to_cells[n] = []
        net_to_cells[n].append(int(c))

    nets = []
    for net_id in sorted(net_to_cells.keys()):
        uniq = sorted(set(net_to_cells[net_id]))
        if num_cells is not None:
            uniq = [c for c in uniq if 0 <= c < num_cells]
        if len(uniq) > 0:
            nets.append(uniq)

    return nets


def extract_nets_from_graph_dict(graph_dict, num_cells: int):
    """
    Try several likely key patterns because graph file naming can vary.

    Supported patterns:
      1) graph_dict["nets"]
      2) pin-to-cell + pin-to-net mapping
      3) edge_index_cell_net style bipartite edges
    """
    candidate_nets_keys = [
        "nets", "net_to_cell", "net2cell"
    ]
    candidate_pin_to_cell_keys = [
        "pin_to_cell", "pin2cell", "pin_cell", "p2c"
    ]
    candidate_pin_to_net_keys = [
        "pin_to_net", "pin2net", "pin_net", "p2n"
    ]
    candidate_edge_keys = [
        "edge_index_cell_net", "cell_net_edge_index", "edges_cell_net"
    ]

    for key in candidate_nets_keys:
        if key in graph_dict:
            raw_nets = graph_dict[key]
            nets = []
            for arr in raw_nets:
                arr = list(map(int, np.asarray(arr).reshape(-1).tolist()))
                arr = [c for c in sorted(set(arr)) if 0 <= c < num_cells]
                if len(arr) > 0:
                    nets.append(arr)
            if len(nets) > 0:
                return nets

    pin_to_cell = None
    pin_to_net = None

    for key in candidate_pin_to_cell_keys:
        if key in graph_dict:
            pin_to_cell = graph_dict[key]
            break

    for key in candidate_pin_to_net_keys:
        if key in graph_dict:
            pin_to_net = graph_dict[key]
            break

    if pin_to_cell is not None and pin_to_net is not None:
        return pin_maps_to_nets(pin_to_cell, pin_to_net, num_cells=num_cells)

    for key in candidate_edge_keys:
        if key in graph_dict:
            return edge_index_cell_net_to_nets(graph_dict[key], num_cells=num_cells)

    raise KeyError(
        f"Could not find graph connectivity in graph bundle. "
        f"Available keys: {sorted(graph_dict.keys())}"
    )


def build_single_sample_graph(x1, y1, x2, y2, nets, grid_size=256):
    """
    Builds one sample's graph tensors.
    """
    device = x1.device

    cell_feat, cell_xy = build_cell_features(x1, y1, x2, y2)

    # all zeros here because this is one sample before batching
    batch_index = torch.zeros(cell_feat.size(0), dtype=torch.long, device=device)

    # grid-space positions are used for both masking and rasterization
    cell_xy_grid = normalize_xy_to_grid(cell_xy, batch_index, grid_size=grid_size)

    net_feat = build_net_features(cell_xy, nets, device=device)
    c2n, n2c = build_bipartite_edges(nets, device=device)
    c2c = build_cell_to_cell_edges_from_nets(nets, device=device)

    return {
        "cell_feat": cell_feat,
        "net_feat": net_feat,
        "cell_to_net_edge_index": c2n,
        "net_to_cell_edge_index": n2c,
        "cell_to_cell_edge_index": c2c,
        "cell_xy_grid": cell_xy_grid,
        "batch_index": batch_index,
        # target for masked center-coordinate reconstruction
        "coord_target": cell_xy_grid.clone(),
    }


def collate_graph_batch(samples):
    """
    Batch samples with geometry, graph tensors, and dense 2D labels.

    Each sample must look like:
        {
            "geom": [x1, y1, x2, y2, None],
            "graph": <dict from build_single_sample_graph>,
            "label": [H, W] or [1, H, W],
            "weight": [H, W] or [1, H, W],
        }
    """
    x1_all, y1_all, x2_all, y2_all = [], [], [], []
    offsets = []

    cell_feat_all = []
    net_feat_all = []
    cell_xy_grid_all = []
    batch_index_all = []
    coord_target_all = []

    c2n_all = []
    n2c_all = []
    c2c_all = []

    labels = []
    weights = []

    cell_base = 0
    net_base = 0
    running_cells = 0

    device = samples[0]["label"].device

    for b, sample in enumerate(samples):
        x1, y1, x2, y2, _ = sample["geom"]
        g = sample["graph"]

        assert g["cell_feat"].shape[0] == g["cell_xy_grid"].shape[0], \
            f"cell_feat/cell_xy_grid mismatch: {g['cell_feat'].shape[0]} vs {g['cell_xy_grid'].shape[0]}"
        assert g["cell_feat"].shape[0] == g["coord_target"].shape[0], \
            f"cell_feat/coord_target mismatch: {g['cell_feat'].shape[0]} vs {g['coord_target'].shape[0]}"

        geom_n_cells = x1.numel()
        graph_n_cells = g["cell_feat"].shape[0]
        n_nets = g["net_feat"].shape[0]

        x1_all.append(x1)
        y1_all.append(y1)
        x2_all.append(x2)
        y2_all.append(y2)

        running_cells += geom_n_cells
        offsets.append(running_cells)

        cell_feat_all.append(g["cell_feat"])
        net_feat_all.append(g["net_feat"])
        cell_xy_grid_all.append(g["cell_xy_grid"])
        coord_target_all.append(g["coord_target"])

        batch_index_all.append(torch.full((graph_n_cells,), b, dtype=torch.long, device=x1.device))

        if g["cell_to_net_edge_index"].numel() > 0:
            edge = g["cell_to_net_edge_index"].clone()
            edge[0] += cell_base
            edge[1] += net_base
            c2n_all.append(edge)

        if g["net_to_cell_edge_index"].numel() > 0:
            edge = g["net_to_cell_edge_index"].clone()
            edge[0] += net_base
            edge[1] += cell_base
            n2c_all.append(edge)

        if g["cell_to_cell_edge_index"].numel() > 0:
            edge = g["cell_to_cell_edge_index"].clone()
            edge += cell_base
            c2c_all.append(edge)

        label = sample["label"]
        weight = sample["weight"]

        if label.dim() == 2:
            label = label.unsqueeze(0)
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)

        labels.append(label)
        weights.append(weight)

        cell_base += graph_n_cells
        net_base += n_nets

    return {
        "geom": [
            torch.cat(x1_all, dim=0),
            torch.cat(y1_all, dim=0),
            torch.cat(x2_all, dim=0),
            torch.cat(y2_all, dim=0),
            torch.tensor(offsets, dtype=torch.long, device=device),
        ],
        "graph": {
            "cell_feat": torch.cat(cell_feat_all, dim=0),
            "net_feat": torch.cat(net_feat_all, dim=0) if len(net_feat_all) > 0 else torch.zeros(0, 3, device=device),
            "cell_xy_grid": torch.cat(cell_xy_grid_all, dim=0),
            "batch_index": torch.cat(batch_index_all, dim=0),
            "coord_target": torch.cat(coord_target_all, dim=0),
            "cell_to_net_edge_index": torch.cat(c2n_all, dim=1) if len(c2n_all) > 0 else torch.zeros(2, 0, dtype=torch.long, device=device),
            "net_to_cell_edge_index": torch.cat(n2c_all, dim=1) if len(n2c_all) > 0 else torch.zeros(2, 0, dtype=torch.long, device=device),
            "cell_to_cell_edge_index": torch.cat(c2c_all, dim=1) if len(c2c_all) > 0 else torch.zeros(2, 0, dtype=torch.long, device=device),
        },
        "label": torch.stack(labels, dim=0),
        "weight": torch.stack(weights, dim=0),
    }