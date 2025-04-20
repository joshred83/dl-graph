import torch


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


def _merge_defaults(kwargs, defaults):  
    """
    Merges default arguments with user-specified arguments.
    Args:
        kwargs (dict): User-specified arguments.
        defaults (dict): Default arguments.
    Returns:
        dict: Merged arguments.
    """
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value
    return kwargs

def normalize_adj(adj: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Normalize the adjacency matrix of a graph.
    Args:
        adj (torch.Tensor): The adjacency matrix of the graph.
        dim (int): The dimension along which to normalize.              

    Returns:    
        torch.Tensor: The normalized adjacency matrix.
    """

        # Validate input
    if not isinstance(adj, torch.Tensor) or adj.dim() != 2:
        raise TypeError("adj must be a 2D dense torch.Tensor. Did you use torch_geometric.utils.to_dense_adj?")
    
    if adj.size(0) != adj.size(1):
        raise ValueError("adj must be a square matrix")
    degree = torch.sum(adj, dim=dim)
    degree = torch.clamp(degree, min=1e-10)
    inv_degree = 1.0 / degree
    inv_degree = torch.diag(inv_degree)
    if dim == 1:
        # Row normalization: D^{-1} A
        return torch.mm(inv_degree, adj)
    elif dim == 0:
        # Column normalization: A D^{-1}
        return torch.mm(adj, inv_degree)
    else:
        raise ValueError("dim must be 0 (column) or 1 (row)")

if __name__ == "__main__":
    from src.loaders import load_elliptic
    from torch_geometric.utils import to_dense_adj
    """    
    Example usage of the utility functions.
    - `summarize_graph`: Prints a summary of the graph data object.
    - `normalize_adj`: Normalizes the adjacency matrix of a graph (dimension sums to 1).
    
    Demonstrations use a time slice from the Elliptic Bitcoin dataset.
    """

    # Example usage
    data = load_elliptic(use_temporal=True, t=2, summarize=True)
    

    # Create a small adjacency matrix
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]

    # Row-normalize (each row sums to 1)
    row_normalized = normalize_adj(adj, dim=1)
    print("Original adjacency:\n", adj)
    print("Row-normalized adjacency:\n", row_normalized)
    print()

    # Column-normalize (each column sums to 1)
    col_normalized = normalize_adj(adj, dim=0)
    print("Original adjacency:\n", adj)
    print("Column-normalized adjacency:\n", col_normalized)
    print()
