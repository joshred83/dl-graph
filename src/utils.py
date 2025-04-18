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


if __name__ == "__main__":
    # Example usage
    from loaders import load_elliptic
    data = load_elliptic(use_temporal=True, t=2)
    
    print(data)
    data = load_elliptic(use_temporal=True, t=2, use_aggregated=True)
    
    print()

