import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from src.loaders import load_elliptic
from src.utils import normalize_adj
from torch import Tensor
import torch.nn.functional as F

class Perturber(BaseTransform):
    def __init__(self, feature_noise: float = 0.1, structure_noise: float = 0.1):
        super().__init__()
        self.feature_noise = feature_noise
        self.structure_noise = structure_noise

    def forward(self, data: Data) -> Data:
        data = data.clone()
        if self.feature_noise > 0:
            noise = torch.normal(0, self.feature_noise, size=data.x.shape, device=data.x.device)
            data.x = data.x + noise
        if self.structure_noise > 0:
            edge_index = data.edge_index
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.structure_noise
            data.edge_index = edge_index[:, mask]
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = data.edge_attr[mask]
        return data

    def augment(self, data: Data) -> Data:
        """Return a Data whose x is [orig.x | perturbed.x]."""
        orig = data.clone()
        pert = self.forward(data)
        out = orig.clone()
        out.x = torch.cat([orig.x, pert.x], dim=1)
        # keep original edges/edge_attr
        out.edge_index = orig.edge_index
        if hasattr(orig, 'edge_attr'):
            out.edge_attr = orig.edge_attr
        return out


class Interpolator(BaseTransform):
    def __init__(self, interpolation_rate: float = 0.2):
        super().__init__()
        self.interpolation_rate = interpolation_rate

    def forward(self, data: Data) -> Data:
        data = data.clone()
        x = data.x
        adj = to_dense_adj(data.edge_index, max_num_nodes=x.size(0))[0]
        x_in = x.clone()
        row_norm = normalize_adj(adj, dim=1)
        col_norm = normalize_adj(adj, dim=0)
        out_nb = row_norm @ x_in
        in_nb = col_norm.t() @ x_in
        combo = (out_nb + in_nb) * 0.5
        data.x = (1 - self.interpolation_rate) * x_in + self.interpolation_rate * combo
        return data

    def augment(self, data: Data) -> Data:
        """Return a Data whose x is [orig.x | interpolated.x]."""
        orig = data.clone()
        interp = self.forward(data)
        out = orig.clone()
        out.x = torch.cat([orig.x, interp.x], dim=1)
        out.edge_index = orig.edge_index
        if hasattr(orig, 'edge_attr'):
            out.edge_attr = orig.edge_attr
        return out


class Aggregator(BaseTransform):
    def __init__(self, method="max"):
        super().__init__()
        if method not in ["mean", "max"]:
            raise ValueError("Method must be either 'mean' or 'max'.")
        self.method = method

    def forward(self, data: Data) -> Data:
        data = data.clone()
        x = data.x
        adj = to_dense_adj(data.edge_index, max_num_nodes=x.size(0))[0]
        if self.method == "max":
            data.x = self._max_agg(x, adj)
        else:
            data.x = self._mean_agg(x, adj)
        return data

    def _max_agg(self, x: Tensor, adj: Tensor) -> Tensor:
        temp = 0.1
        w = F.softmax(adj / temp, dim=1)
        return w @ x

    def _mean_agg(self, x: Tensor, adj: Tensor) -> Tensor:
        deg = adj.sum(1).clamp(min=1e-10)
        inv = torch.diag(1.0 / deg)
        return (inv @ adj) @ x

    def augment(self, data: Data) -> Data:
        """Return a Data whose x is [orig.x | aggregated.x]."""
        orig = data.clone()
        agg = self.forward(data)
        out = orig.clone()
        out.x = torch.cat([orig.x, agg.x], dim=1)
        out.edge_index = orig.edge_index
        if hasattr(orig, 'edge_attr'):
            out.edge_attr = orig.edge_attr
        return out


if __name__ == "__main__":
    from torch_geometric.transforms import Compose
    data = load_elliptic(use_temporal=True, t=5, summarize=False)
    p = Perturber(0.2, 0.5)
    aug_p = p.augment(data)
    print("Perturb-augment x.shape:", aug_p.x.shape)

    i = Interpolator(0.5)
    aug_i = i.augment(data)
    print("Interp-augment x.shape:", aug_i.x.shape)

    a = Aggregator("mean")
    aug_a = a.augment(data)
    print("Agg-augment x.shape:", aug_a.x.shape)
    # Load a sample data object
    data = load_elliptic(use_temporal=True, t=5, summarize=False)

    # Define a composed transform that augments at each step
    transform = Compose([
        lambda d: Perturber(0.2, 0.5).augment(d),
        lambda d: Interpolator(0.5).augment(d),
        lambda d: Aggregator("mean").augment(d),
    ])

    # Apply the composed transform
    augmented_data = transform(data)

    print("Final augmented x.shape:", augmented_data.x.shape)