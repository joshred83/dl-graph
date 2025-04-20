import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from src.loaders import load_elliptic
from src.utils import normalize_adj

class Perturber(BaseTransform):
    """
    Applies feature and structure perturbations to a PyG Data object.
    Adds Gaussian noise to node features and randomly drops edges in the adjacency matrix.

    Args:
        feature_noise (float): Standard deviation of Gaussian noise added to features.
        structure_noise (float): Probability of dropping each edge (edge dropout).
    """
    def __init__(self, feature_noise: float = 0.1, structure_noise: float = 0.1):
        super().__init__()
        self.feature_noise = feature_noise
        self.structure_noise = structure_noise

    def forward(self, data: Data) -> Data:
        data = data.clone()
        
        # Noisy feature perturbation
        if self.feature_noise > 0 :
            noise = torch.normal(0, self.feature_noise, size=data.x.shape, device=data.x.device)
            data.x = data.x + noise

        # Structure perturbation (edge dropout)
        if self.structure_noise > 0 :
            edge_index = data.edge_index
            num_edges = edge_index.size(1)
            # Only apply perturbation to existing edges (where adj_copy > 0)
            # This ensures we only drop existing edges, not add new ones in random directions
            mask = torch.rand(num_edges, device=edge_index.device) > self.structure_noise
            data.edge_index = edge_index[:, mask]

            # Optionally, drop edge attributes if present
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = data.edge_attr[mask]

        return data
    
class Interpolator(BaseTransform):
    """
    Interpolates node features with their neighbors' features using the adjacency matrix.
    Implements the interpolation strategy from GraphMix (https://arxiv.org/pdf/1909.11715).

    Args:
        interpolation_rate (float): Weight for neighbor features in interpolation (0=no mix, 1=all neighbor).
    """
    def __init__(self, interpolation_rate: float = 0.2):
        super().__init__()
        self.interpolation_rate = interpolation_rate

    def forward(self, data: Data) -> Data:
        data = data.clone()
        x = data.x
        # Convert edge_index to dense adjacency matrix
        adj = to_dense_adj(data.edge_index, max_num_nodes=x.size(0))[0]

        x_in = x.clone()
        
        # Outgoing neighbors (rows)
        row_normalized_adj = normalize_adj(adj, dim=1)
        col_normalized_adj = normalize_adj(adj, dim=0)
        
        out_neighbor_features = torch.mm(row_normalized_adj, x_in)
        in_neighbor_features = torch.mm(col_normalized_adj.t(), x_in)

        # Combine both directions
        combined_neighbor_features = (out_neighbor_features + in_neighbor_features) / 2.0
        data.x = (1 - self.interpolation_rate) * x_in + self.interpolation_rate * combined_neighbor_features

        return data
    


if __name__ == "__main__":
    print("Testing Perturber and Interpolator...")

    # Get a timeslice (e.g., t=5) from the loaded dataset
    data = load_elliptic(use_temporal=True, t=5, summarize=False)
    # Pass the timeslice Data object directly to test
    # Use the provided data object (e.g., a timeslice from load_elliptic)
    print("Original x:\n", data.x)
    print("Original edge_index:\n", data.edge_index)

    # Test Perturber
    perturber = Perturber(feature_noise=0.2, structure_noise=0.5)
    perturbed = perturber(data)
    print("\nPerturbed x:\n", perturbed.x)
    print("Perturbed edge_index:\n", perturbed.edge_index)
    perturber = Perturber(feature_noise=0.0, structure_noise=0.0)
    perturbed = perturber(data)
    print("\nPerturbed x:\n", perturbed.x)
    print("Perturbed edge_index:\n", perturbed.edge_index)
    # Test Interpolator
    interpolator = Interpolator(interpolation_rate=0.5)
    interpolated = interpolator(data)
    print("\nInterpolated x:\n", interpolated.x)