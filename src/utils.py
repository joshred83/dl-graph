from torch_geometric.datasets import EllipticBitcoinDataset, EllipticBitcoinTemporalDataset
from torch_geometric.loader import NeighborLoader, ClusterLoader
import torch
import torch_geometric as pyg
from torch_geometric import transforms

def load_elliptic(
    root=None, 
    force_reload=False,
    use_aggregated=False,
    use_temporal=False,
    t=None
):
    """
    Load the EllipticBitcoinDataset.

    Paper: 
        Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional
        Networks for Financial Forensics" <https://arxiv.org/abs/1908.02591>`_
        paper.

    Content:
        This anonymized dataset is a transaction graph collected from the
        Bitcoin blockchain. A node in the graph represents a transaction,
        and an edge can be viewed as a flow of Bitcoins between transactions.
        Each node has 166 features and is labeled as being created by a
        "licit", "illicit", or "unknown" entity.

    Nodes and Edges:
        - 203,769 nodes and 234,355 edges.
        - 2% (4,545) of the nodes are labeled class1 (illicit).
        - 21% (42,019) are labeled class2 (licit).
        - The remaining transactions are not labeled with regard to licit
          versus illicit.

    Features:
        - 166 features per node.
        - Each node has a time step (1 to 49), representing when a
          transaction was broadcasted.
        - Each time step contains a single connected component; there are
          no edges connecting different time steps.
        - The first 94 features represent local transaction information
          (e.g., time step, number of inputs/outputs, transaction fee,
          output volume, aggregated figures).
        - The remaining 72 features are aggregated features, obtained using
          transaction information one-hop backward/forward from the center
          node (e.g., max, min, std, correlation coefficients).
    
    Adapted from the description at
    https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
    
    Args:
        root (str): Directory to store the dataset.
        force_reload (bool): Whether to reload the dataset.

    Returns:
        data: The loaded data object (a graph).
    """
    if use_temporal and t is None:
        raise ValueError("Temporal data requires a time step (t) to be specified.")
    # default root to the data directory if not specified
    if root is None:
        root = "../data/elliptic"

    if use_temporal:
        dataset = EllipticBitcoinTemporalDataset(root=root, force_reload=force_reload, t=t)
    else:
        dataset = EllipticBitcoinDataset(root=root, force_reload=force_reload)
    data = dataset[0]

    
    if use_aggregated:
        # data already contains aggregated features
        pass
    else:
        # Remove the aggregated features, correcting for the temporal case
        # where the first feature is the time step
        data.x = data.x[:, 0:93]
                        


    summarize_graph(data)
    return data

def summarize_graph(data):
    """
    Prints a summary of the given graph data object, including the number
    of nodes, edges, features per node, label distribution, and time steps
    if available.

    Args:
        data: A graph data object, typically from PyTorch Geometric,
            expected to have attributes such as 'num_nodes', 'num_edges',
            'x' (node features), 'y' (node labels), and optionally
            'time_step'.

    Outputs:
        Prints the following information to the console:
            - Number of nodes
            - Number of edges
            - Number of features per node
            - Distribution of node labels (if present)
            - Unique time steps (if present)
    """

    # Check if the data object has the required attributes and print them

    num_nodes = getattr(data, 'num_nodes', None)
    num_edges = getattr(data, 'num_edges', None)
    num_features = data.x.size(1) if hasattr(data, 'x') and data.x is not None else None

    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"Features per node: {num_features}")

    # Check if the data object has labels and time steps
    # and print their distributions
    if hasattr(data, 'y') and data.y is not None:
        unique_labels, counts = torch.unique(data.y, return_counts=True)
        print("Label distribution:")
        for label, count in zip(unique_labels.tolist(), counts.tolist()):
            print(f"  Class {label}: {count} nodes")
    if hasattr(data, 'time_step') and data.time_step is not None:
        time_steps = torch.unique(data.time_step).tolist()
        print(f"Time steps: {time_steps}")

    print("Directed:", data.is_directed())
    print("Temporal:", hasattr(data, 'time_step'))
    print("Bipartite:", data.is_bipartite())


if __name__ == "__main__":
    # Example usage
    data = load_elliptic(use_temporal=True, t=2)
    
    print(data)
    data = load_elliptic(use_temporal=True, t=2, use_aggregated=True)
    
    print()

