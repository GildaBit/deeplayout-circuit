import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import convolve1d
from pathlib import Path
import torch.nn.functional as F

from model.gnn_model.graph_utils_deeplayout import (
    build_single_sample_graph,
    extract_nets_from_graph_dict,
)


# kept from your original file so weighting behavior stays familiar
TRAIN_BUCKET_NUM = [2581492, 17149, 41509, 43174, 30339, 34103, 36373, 16562, 26444, 25157, 16077, 19504, 24324, 14210, 14945, 50272, 20398, 133562, 21247, 58371, 107704, 118764, 246384, 81568, 197408, 164580, 367299, 338230, 323874, 614891, 498301, 705948, 866064, 925084, 916282, 1265832, 1251761, 1650988, 1318899, 2342261, 1623725, 1784712, 1851044, 2101910, 2573112, 2532386, 3081342, 2345318, 2810366, 2697930, 3035275, 2756727, 3036616, 2769940, 3439666, 2650078, 2395155, 2587796, 3772054, 3690807, 3535069, 3617550, 3145361, 3602836, 3603838, 3955619, 3368344, 3641182, 3238609, 4168025, 3639133, 3940118, 4113411, 3858332, 4029462, 3758855, 3944048, 3604445, 4008148, 4373663, 4132053, 4745153, 3872252, 4193883, 4302939, 4146266, 3854443, 4302867, 3659075, 4283141, 4227904, 4131651, 4026522, 3813998, 3802860, 4166017, 4754549, 3756769, 4198892, 4852302, 3336808, 4453865, 4117645, 4110658, 3869876, 3858507, 4097277, 4001008, 4271764, 4525024, 3807901, 4153564, 4025843, 4011729, 4620492, 4597749, 4871241, 4135272, 5493407, 4831656, 4286181, 4540263, 4233062, 4941352, 4506741, 3974884, 4318617, 4202539, 3792502, 4444073, 4475465, 4849234, 4505929, 3718776, 4119914, 5023962, 4006894, 4138024, 3595549, 3190685, 3643390, 3522463, 3415693, 3270879, 3265063, 3265904, 3059192, 3006037, 2689766, 3540200, 3200022, 3066402, 2857296, 2400664, 3089188, 2681121, 2920757, 2859823, 2453946, 2584223, 2481059, 1993527, 2080652, 1857478, 2100471, 2227679, 1843444, 1570269, 1545903, 2594179, 2648206, 1505328, 1659068, 1555914, 1251633, 1319450, 1275957, 1096012, 1406018, 1017790, 1131564, 1103892, 1203477, 974773, 774545, 1041176, 1069225, 989386, 956256, 752702, 814265, 713137, 830536, 717171, 673416, 641348, 677930, 610276, 703582, 662650, 595399, 663509, 644617, 536808, 551355, 723515, 525907, 444384, 437081, 429924, 477362, 419360, 450082, 455833, 353762, 480223, 339048, 331852, 325773, 361584, 313321, 388309, 303634, 296717, 292218, 287859, 282119, 278385, 315556, 267902, 263248, 257821, 254147, 249724, 244866, 241026, 237560, 233389, 230312, 228318, 273194, 265990, 216499, 213006, 207992, 204900, 201361, 198349, 195512, 193625, 190216, 237447, 183548, 180730, 178054, 214243, 172899, 169969, 167370, 164359, 161912, 159353, 157049, 153626, 152019, 149567, 147619, 145016, 142688, 140056, 138479, 136086, 134878, 132312, 130043, 127826, 126196, 124814, 123025, 121080, 118452, 117164, 116179, 113965, 111744, 110891, 109231, 107428, 105662, 104450, 103051, 101024, 100012, 98834, 96960, 95889, 94063, 92341, 91653, 89477, 88864, 87385, 86131, 84877, 83501, 82666, 81436, 80088, 79293, 78435, 77271, 75670, 74870, 74422, 72608, 72398, 71148, 70102, 69500, 67939, 67259, 66603, 65836, 64456, 63613, 62659, 62176, 61303, 60065, 59872, 58680, 58186, 57295, 56459, 55465, 55249, 54811, 54035, 52726, 51761, 51874, 50654, 49941, 49984, 48736, 48240, 47756, 46873, 46581, 45450, 45186, 44582, 44606, 43828, 43533, 42620, 42353, 41451, 40751, 40255, 40255, 39527, 39031, 38681, 38135, 37659, 36847, 36598, 36124, 35842, 35158, 35488, 34611, 34102, 33709, 33551, 32905, 32944, 32264, 32104, 31496, 30900, 30839, 30495, 29886, 29925, 29382, 29046, 28641, 28036, 27936, 27787, 27403, 27221, 26666, 26233, 26341, 25649, 25518, 25232, 24915, 24538, 24241, 24251, 23932, 23657, 23175, 23181, 22830, 22343, 22320, 22144, 21921, 21464, 21310, 21091, 20820, 20549, 20334, 20087, 19873, 19756, 19291, 19034, 19043, 18930, 18577, 18581, 18326, 17915, 17654, 17696, 17451, 17024, 17148, 17019, 16633, 16765, 16216, 16445, 15999, 16047, 15680, 15375, 15369, 15455, 15231, 14891, 14742, 14296, 14569, 14165, 14253, 13960, 13828, 13733, 13530, 13531, 13700, 13508, 12986, 12899, 12817, 12704, 12495, 12453, 12248, 12188, 11964, 12006, 11639, 11705, 11621, 11543, 11278, 11159, 11023, 11042, 10800, 10732, 10649, 10477, 10439, 10247, 10305, 10058, 9990, 9990, 9808, 9664, 9631, 9601, 9425, 9457, 9321, 9092, 8967, 9066, 8807, 8886, 8812, 8661, 8397, 8673, 8336, 8332, 8376, 8139, 8054, 7918, 7909, 7918, 7810, 7598, 7742, 7513, 7581, 7426, 7368, 7240, 7040, 7170, 7080, 6951, 6794, 6884, 6763, 6591, 6760, 6675, 6498, 6539, 6323, 6241, 6112, 6242, 6109, 6043, 6065, 6115, 5817, 5871, 5795, 5666, 5789, 5666, 5648, 5521, 5443, 5446, 5456, 5387, 5188, 5182, 5123, 5222, 5107, 5211, 5041, 4839, 4864, 4856, 4686, 4766, 4775, 4578, 4664, 4485, 4501, 4577, 4499, 4442, 4427, 4384, 4271, 4183, 4221, 4057, 4049, 4160, 4008, 4008, 4038, 3945, 3947, 3982, 3839, 3789, 3770, 3727, 3775, 3749, 3615, 3661, 3663, 3475, 3503, 3519, 3519, 3423, 3396, 3233, 3313, 3220, 3230, 3187, 3193, 3242, 3139, 3196, 3089, 3033, 3093, 3035, 2903, 2881, 2904, 2891, 2913, 2942, 2796, 2713, 2755, 2700, 2770, 2710, 2647, 2607, 2693, 2535, 2495, 2499, 2436, 2508, 2511, 2464, 2406, 2403, 2357, 2311, 2419, 2362, 2329, 2298, 2348, 2223, 2242, 2112, 2157, 2124, 2112, 2170, 2110, 2071, 2080, 2112, 2081, 2046, 1983, 1988, 1949, 1967, 1853, 1894, 1969, 1875, 1883, 1845, 1815, 1790, 1796, 1807, 1660, 1625, 1786, 1706, 1679, 1582, 1685, 1652, 1574, 1667, 1563, 1554, 1520, 1518, 1493, 1430, 1489, 1480, 1493, 1475, 1429, 1481, 1452, 1485, 1517, 1382, 1357, 1363, 1421, 1311, 1303, 1340, 1297, 1310, 1305, 1317, 1252, 1257, 1237, 1270, 1298, 1212, 1252, 1252, 1214, 1137, 1198, 1097, 1148, 1085, 1116, 1142, 1077, 1076, 1064, 1059, 1035, 1032, 1036, 976, 978, 1022, 996, 1028, 912, 955, 973, 980, 978, 963, 970, 913, 975, 929, 871, 881, 907, 854, 880, 904, 840, 834, 890, 811, 831, 756, 808, 814, 771, 781, 767, 744, 785, 737, 755, 751, 719, 752, 733, 711, 698, 720, 713, 715, 684, 664, 659, 685, 692, 675, 641, 632, 643, 615, 635, 654, 612, 629, 590, 634, 606, 619, 605, 567, 562, 589, 518, 564, 525, 514, 552, 567, 537, 515, 498, 535, 551, 510, 507, 487, 447, 496, 462, 484, 473, 459, 489, 450, 424, 462, 467, 447, 449, 412, 449, 408, 445, 392, 461, 415, 434, 417, 428, 416, 367, 413, 396, 371, 375, 428, 362, 387, 332, 344, 369, 364, 383, 346, 342, 338, 371, 345, 350, 348, 317, 330, 322, 316, 309, 320, 316, 292, 323, 274, 321, 300, 292, 253, 275, 290, 290, 273, 315, 310, 247, 266, 265, 288, 276, 254, 258, 271, 245, 255, 242, 260, 225, 268, 245, 233, 216, 205, 213, 254, 220, 233, 253, 245, 212, 231, 216, 223, 203, 247, 208, 207, 225, 188, 199, 191, 197, 202, 182, 174, 186, 160, 167, 188, 177, 196, 177, 194, 181, 169, 168, 190, 169, 161, 147, 153, 162, 158, 180, 129, 160, 180, 160, 146, 151, 137, 157, 181, 144, 160, 137, 167, 140, 137, 157, 132, 134, 133, 123, 147, 137, 120, 112, 123, 103, 124, 121, 108, 100, 114, 126, 106, 100, 125, 126, 101, 113, 109, 102, 106, 95, 111, 121, 115, 93, 124, 120, 89, 96, 92, 83, 94, 79, 102, 88, 91, 87, 99, 93, 105, 102, 92, 82, 8434]


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


class CircuitnetDeeplayout(Dataset):
    """
    Graph-aware CircuitNet dataset.

    Required args:
        split
        data_root
        label_root
        graph_root

    graph_root should contain graph files with the same relative path stem as data_root,
    for example:
        data_root/design/foo/bar.npy
        graph_root/design/foo/bar.npy

    Supported graph file styles:
        - dict with "nets"
        - dict with pin_to_cell + pin_to_net
        - dict with edge_index_cell_net
    """
    def __init__(self, split='train', data_root='trainval', label_root='trainval', graph_root=None, graph_ext='.npy', loop=1):
        super().__init__()

        if graph_root is None:
            raise ValueError("graph_root must be provided for the graph-aware DeepLayout pipeline.")

        self.split, self.loop = split, loop
        self.data_root = data_root
        self.label_root = label_root
        self.graph_root = graph_root
        self.graph_ext = graph_ext

        # Build a basename -> full path index for graph files once.
        # This is more robust than assuming graph files live directly under graph_root
        # with the same relative path as placement files.
        self.graph_index = self._build_graph_index()
        print(f"Indexed {len(self.graph_index)} graph files from {self.graph_root}")

        # train.txt / val.txt / test.txt live one directory up in data/
        base_dir = os.path.dirname(__file__)      # path to data/gnn_data
        data_dir = os.path.dirname(base_dir)      # path to data/
        
        if split['split'] == 'train':
            self.is_train = True
            with open(os.path.join(data_dir, "train.txt"), "r") as f:
                self.data_list = [line.strip() for line in f if line.strip()]
        elif split['split'] == 'val':
            self.is_train = False
            with open(os.path.join(data_dir, "val.txt"), "r") as f:
                self.data_list = [line.strip() for line in f if line.strip()]
        elif split['split'] == 'test':
            self.is_train = False
            with open(os.path.join(data_dir, "test.txt"), "r") as f:
                self.data_list = [line.strip() for line in f if line.strip()]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.data_idx = np.arange(len(self.data_list))
        print(f"Totally {len(self.data_idx)} labels in {split} set.")

        self.bucket_weights = self._get_bucket_weights()

    def get_bin_idx(self, x):
        return min(int(x * np.float32(1000)), 1000)

    def _get_bucket_weights(self):
        value_lst = TRAIN_BUCKET_NUM
        lds_kernel = 'gaussian'
        lds_ks = 5
        lds_sigma = 2
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)

        value_lst = np.sqrt(value_lst)
        smoothed_value = convolve1d(np.asarray(value_lst), weights=lds_kernel_window, mode='reflect')
        smoothed_value = list(smoothed_value)
        scaling = np.sum(TRAIN_BUCKET_NUM) / (np.sum(np.array(TRAIN_BUCKET_NUM) / (np.array(smoothed_value) + 1e-7)) + 1e-7)
        bucket_weights = [np.float32(scaling / smoothed_value[bucket]) for bucket in range(len(TRAIN_BUCKET_NUM))]
        return bucket_weights

    def _get_weights(self, label):
        sp = label.shape
        if self.bucket_weights is not None:
            label_flat = label.view(-1).cpu().numpy()
            assert label_flat.dtype == np.float32
            weights = np.array(list(map(lambda v: self.bucket_weights[min(self.get_bin_idx(v), 99)], label_flat)))
            weights = torch.tensor(weights, dtype=torch.float32).view(*sp)
        else:
            weights = torch.tensor([np.float32(1.)], dtype=torch.float32).repeat(*sp)
        return weights

    def _build_graph_index(self):
        """
        Recursively index graph files by normalized key.
    
        Example:
            zero-riscy-a-2-c20_node_attr.npy
        is indexed under:
            zero-riscy-a-2-c20
    
        This avoids relying on exact basename matches.
        """
        graph_index = {}
    
        for path in Path(self.graph_root).rglob("*"):
            if not path.is_file():
                continue
    
            stem = path.stem  # e.g. zero-riscy-a-2-c20_node_attr
    
            # strip known graph-file suffixes
            normalized = stem
            for suffix in [
                "_node_attr",
                "_edge_index",
                "_edge_attr",
                "_net_attr",
                "_pin_attr",
                "_pin_to_net",
                "_pin_to_cell",
                "_cell_to_net",
            ]:
                if normalized.endswith(suffix):
                    normalized = normalized[: -len(suffix)]
                    break
    
            if normalized not in graph_index:
                graph_index[normalized] = []
            graph_index[normalized].append(str(path))
    
        return graph_index

    def _sample_to_graph_key(self, rel_path):
        """
        Convert a per-sample filename into the shared graph key.
    
        Example:
            8313-zero-riscy-a-2-c20-u0.7-m2-p8-f0.npy
        becomes:
            zero-riscy-a-2-c20
    
        Rule:
            drop the leading numeric sample id
            keep tokens until and including the first token that starts with 'c'
        """
        basename = os.path.basename(rel_path)
        stem, _ = os.path.splitext(basename)
    
        parts = stem.split("-")
        if len(parts) < 3:
            raise ValueError(f"Unexpected sample filename format: {rel_path}")
    
        # skip the leading sample id
        kept = []
        for token in parts[1:]:
            kept.append(token)
            if token.startswith("c"):
                break
    
        if len(kept) == 0:
            raise ValueError(f"Could not extract graph key from sample filename: {rel_path}")
    
        return "-".join(kept)
        
    def _get_graph_files(self, rel_path):
        """
        Resolve all graph files associated with one sample's shared graph key.
        """
        graph_key = self._sample_to_graph_key(rel_path)
    
        if graph_key in self.graph_index:
            return self.graph_index[graph_key]
    
        raise FileNotFoundError(
            f"Could not resolve graph files for rel_path='{rel_path}'. "
            f"Extracted graph_key='{graph_key}'. Available example keys: "
            f"{list(self.graph_index.keys())[:10]}"
        )

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        rel_path = self.data_list[data_idx]

        data_path = self._resolve_existing_file(self.data_root, rel_path)
        label_path = self._resolve_existing_file(self.label_root, rel_path)

        # ----- geometry -----
        data_dict = np.load(data_path, allow_pickle=True).item()
        data = np.array(list(data_dict.values()))

        y1, x1, y2, x2 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        x1 = torch.FloatTensor(x1)
        y1 = torch.FloatTensor(y1)
        x2 = torch.FloatTensor(x2)
        y2 = torch.FloatTensor(y2)

        # ----- dense label -----
        label = np.load(label_path)
        label = torch.tensor(label, dtype=torch.float32)
        
        # Congestion labels may come as [H, W, 1]
        if label.dim() == 3 and label.shape[-1] == 1:
            label = label.squeeze(-1)
        
        # ensure shape [1, H, W]
        if label.dim() == 2:
            label = label.unsqueeze(0)
        
        # keep a fixed 256x256 target shape
        if tuple(label.shape[-2:]) != (256, 256):
            label = F.interpolate(
                label.unsqueeze(0),
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        
        weight = self._get_weights(label)
        
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        
        if tuple(weight.shape[-2:]) != (256, 256):
            weight = F.interpolate(
                weight.unsqueeze(0),
                size=(256, 256),
                mode="nearest",
            ).squeeze(0)

        # ----- graph connectivity -----
        graph_path = self._get_graph_path(rel_path)
        graph_npz = np.load(graph_path, allow_pickle=True)
        
        cell_feat = np.asarray(graph_npz["cell_feat"], dtype=np.float32)
        net_feat = np.asarray(graph_npz["net_feat"], dtype=np.float32)
        cell_to_net_edge_index = np.asarray(graph_npz["cell_to_net_edge_index"], dtype=np.int64)
        net_to_cell_edge_index = np.asarray(graph_npz["net_to_cell_edge_index"], dtype=np.int64)
        cell_to_cell_edge_index = np.asarray(graph_npz["cell_to_cell_edge_index"], dtype=np.int64)
        
        if cell_feat.ndim == 1:
            cell_feat = cell_feat.reshape(1, -1)
        
        if net_feat.ndim == 1:
            if net_feat.size == 0:
                net_feat = net_feat.reshape(0, 3)
            else:
                net_feat = net_feat.reshape(1, -1)
        
        def sanitize_edge_index(edge_index, num_src, num_dst):
            edge_index = np.asarray(edge_index, dtype=np.int64)
        
            if edge_index.size == 0:
                return np.zeros((2, 0), dtype=np.int64)
        
            edge_index = edge_index.reshape(2, -1)
        
            src = edge_index[0]
            dst = edge_index[1]
        
            mask = (
                (src >= 0) & (src < num_src) &
                (dst >= 0) & (dst < num_dst)
            )
        
            edge_index = edge_index[:, mask]
            return edge_index
        
        num_cells = cell_feat.shape[0]
        num_nets = net_feat.shape[0]
        
        cell_to_net_edge_index = sanitize_edge_index(cell_to_net_edge_index, num_cells, num_nets)
        net_to_cell_edge_index = sanitize_edge_index(net_to_cell_edge_index, num_nets, num_cells)
        cell_to_cell_edge_index = sanitize_edge_index(cell_to_cell_edge_index, num_cells, num_cells)

#        cell_feat, net_feat, cell_to_net_edge_index, net_to_cell_edge_index, cell_to_cell_edge_index = self.cap_graph_size(
#            cell_feat,
#            net_feat,
#            cell_to_net_edge_index,
#            net_to_cell_edge_index,
#            cell_to_cell_edge_index,
#            max_cells=12000,
#        )

        num_cells = cell_feat.shape[0]
        num_nets = net_feat.shape[0]
        
        graph = {
            "cell_feat": torch.tensor(cell_feat, dtype=torch.float32),
            "net_feat": torch.tensor(net_feat, dtype=torch.float32),
            "cell_to_net_edge_index": torch.tensor(cell_to_net_edge_index, dtype=torch.long),
            "net_to_cell_edge_index": torch.tensor(net_to_cell_edge_index, dtype=torch.long),
            "cell_to_cell_edge_index": torch.tensor(cell_to_cell_edge_index, dtype=torch.long),
        }

        if graph["cell_feat"].dim() != 2:
            raise ValueError(
                f"Processed graph cell_feat must be 2D, got shape {tuple(graph['cell_feat'].shape)} "
                f"for sample {rel_path} from {graph_path}"
            )
        
        # Build the per-sample dynamic pieces the model still expects
        num_cells = graph["cell_feat"].shape[0]
        batch_index = torch.zeros(num_cells, dtype=torch.long)
        
        # coord_target and cell_xy_grid are based on the first two feature dims [cx, cy]
        cell_xy = graph["cell_feat"][:, :2]
        
        # normalize to grid space for masking + rasterization
        from model.gnn_model.graph_utils_deeplayout import normalize_xy_to_grid
        cell_xy_grid = normalize_xy_to_grid(cell_xy, batch_index, grid_size=256)
        
        graph["cell_xy_grid"] = cell_xy_grid
        graph["batch_index"] = batch_index
        graph["coord_target"] = cell_xy_grid.clone()

        return {
            "geom": [x1, y1, x2, y2, None],
            "graph": graph,
            "label": label,
            "weight": weight,
        }

    def _sample_to_graph_key(self, rel_path):
        """
        Convert sample filename to shared graph key.
    
        Example:
            8313-zero-riscy-a-2-c20-u0.7-m2-p8-f0.npy
        becomes:
            zero-riscy-a-2-c20
        """
        basename = os.path.basename(rel_path)
        stem, _ = os.path.splitext(basename)
    
        parts = stem.split("-")
        if len(parts) < 3:
            raise ValueError(f"Unexpected sample filename format: {rel_path}")
    
        kept = []
        for token in parts[1:]:
            kept.append(token)
            if token.startswith("c"):
                break
    
        return "-".join(kept)
    
    
    def _get_graph_path(self, rel_path):
        base = os.path.basename(rel_path)
        stem, _ = os.path.splitext(base)
        graph_path = os.path.join(self.graph_root, stem + ".graph.npz")
    
        if not os.path.exists(graph_path):
            raise FileNotFoundError(
                f"Processed graph file not found for sample '{rel_path}'. "
                f"Expected: {graph_path}"
            )
    
        return graph_path
    
    def _resolve_existing_file(self, root, rel_path):
        """
        Resolve a sample path under a dataset root.
    
        Tries:
          1) root / rel_path
          2) root / rel_path without '.npy'
        """
        cand1 = os.path.join(root, rel_path)
        if os.path.exists(cand1):
            return cand1
    
        stem, ext = os.path.splitext(rel_path)
        cand2 = os.path.join(root, stem)
        if os.path.exists(cand2):
            return cand2
    
        raise FileNotFoundError(
            f"Could not resolve file for rel_path='{rel_path}' under root='{root}'. "
            f"Tried:\n  {cand1}\n  {cand2}"
        )

    @staticmethod
    def cap_graph_size(
        cell_feat,
        net_feat,
        cell_to_net_edge_index,
        net_to_cell_edge_index,
        cell_to_cell_edge_index,
        max_cells=12000,
    ):
        num_cells = cell_feat.shape[0]
        if num_cells <= max_cells:
            return (
                cell_feat,
                net_feat,
                cell_to_net_edge_index,
                net_to_cell_edge_index,
                cell_to_cell_edge_index,
            )

        keep_cells = np.arange(max_cells, dtype=np.int64)

        old_to_new_cell = -np.ones(num_cells, dtype=np.int64)
        old_to_new_cell[keep_cells] = np.arange(len(keep_cells), dtype=np.int64)

        # Cap cell features
        cell_feat = cell_feat[keep_cells]

        # Filter cell->net edges by kept cells
        if cell_to_net_edge_index.size > 0:
            c_src = cell_to_net_edge_index[0]
            n_dst = cell_to_net_edge_index[1]

            mask = np.isin(c_src, keep_cells)
            c_src = c_src[mask]
            n_dst = n_dst[mask]

            c_src = old_to_new_cell[c_src]
            cell_to_net_edge_index = np.stack([c_src, n_dst], axis=0)
        else:
            cell_to_net_edge_index = np.zeros((2, 0), dtype=np.int64)

        # Keep only nets touched by kept cells and densify net ids
        if cell_to_net_edge_index.size > 0:
            used_nets = np.unique(cell_to_net_edge_index[1])

            old_to_new_net = -np.ones(net_feat.shape[0], dtype=np.int64)
            old_to_new_net[used_nets] = np.arange(len(used_nets), dtype=np.int64)

            net_feat = net_feat[used_nets]
            cell_to_net_edge_index[1] = old_to_new_net[cell_to_net_edge_index[1]]

            net_to_cell_edge_index = np.stack(
                [cell_to_net_edge_index[1], cell_to_net_edge_index[0]],
                axis=0,
            )
        else:
            net_feat = net_feat[:0]
            net_to_cell_edge_index = np.zeros((2, 0), dtype=np.int64)

        # Filter cell->cell edges by kept cells and remap
        if cell_to_cell_edge_index.size > 0:
            src = cell_to_cell_edge_index[0]
            dst = cell_to_cell_edge_index[1]

            mask = np.isin(src, keep_cells) & np.isin(dst, keep_cells)
            src = src[mask]
            dst = dst[mask]

            src = old_to_new_cell[src]
            dst = old_to_new_cell[dst]

            cell_to_cell_edge_index = np.stack([src, dst], axis=0)
        else:
            cell_to_cell_edge_index = np.zeros((2, 0), dtype=np.int64)

        return (
            cell_feat,
            net_feat,
            cell_to_net_edge_index,
            net_to_cell_edge_index,
            cell_to_cell_edge_index,
        )
    
    def __len__(self):
        return len(self.data_idx) * self.loop