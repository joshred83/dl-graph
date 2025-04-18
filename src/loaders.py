from torch_geometric.datasets import EllipticBitcoinDataset, EllipticBitcoinTemporalDataset
from torch_geometric.loader import NeighborLoader, ClusterLoader, ClusterData
import torch
import torch_geometric as pyg
from torch_geometric import transforms
from utils import summarize_graph
import inspect
import warnings
from torch_geometric.loader import ClusterData, ClusterLoader
import argparse

def load_elliptic(
    root=None, 
    force_reload=False,
    use_aggregated=False,
    use_temporal=False,
    t=None,
    summarize=False

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

    if summarize:                    
        summarize_graph(data)
    return data

def make_loader(data, loader_type='neighbor', **kwargs):
    """
    Create a data loader for the given graph data object.

    Args:
        data: A graph data object, typically from PyTorch Geometric.
        loader_type (str): The type of loader to create ('neighbor' or 'cluster').
        kwargs (dict): Additional arguments for the loader.
            - For NeighborLoader: batch_size, shuffle, num_neighbors, input_nodes
            - For ClusterLoader: batch_size, shuffle
    Returns:
        loader: A data loader for the graph data object.
    """

    if loader_type == 'cluster':

        return cluster_loader(data, **kwargs)


    elif loader_type == 'neighbor':

        return neighbor_loader(data, **kwargs)


def neighbor_loader(data, **kwargs):
    """
    data   : a PyG Data graph
    kwargs : dict containing any args for NeighborLoader

    Returns:
    #    loader: A data loader for the graph data object.
    """

    # Some defaults. They can be overridden by kwargs. 
    defaults = dict(batch_size=2048, shuffle=True, num_neighbors=[10, 10], input_nodes=None)

    # make sure we have the defaults, but override them if specified in wargs
    kwargs = _merge_defaults(kwargs, defaults)

    # if any kwargs are left over, warn about them...and that's it.

    if kwargs:
        warnings.warn(f"neighbor_loader: ignoring unexpected args {list(kwargs)}")

    # return the loader
    return NeighborLoader(data, **kwargs)


def cluster_loader(data, **kwargs):
    """
    data   : a PyG Data graph
    kwargs : dict containing any args for ClusterData or ClusterLoader

    Returns:
       loader: A data loader for the graph data object.
    """
    # 1) grab everything
    kwargs = {} if kwargs is None else dict(kwargs)

    # splits up kwargs assigning them to the relevant signature
    # could pose a problem if the same kwarg is used in both signatures

    cd_sig = inspect.signature(ClusterData.__init__)
    cl_sig = inspect.signature(ClusterLoader.__init__)
    cd_params = set(cd_sig.parameters) - {"self"}        # drop 'self'
    cl_params = set(cl_sig.parameters) - {"self", "clustered"}  # drop 'self' & first positional

    cluster_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in cd_params}
    loader_kwargs  = {k: kwargs.pop(k) for k in list(kwargs) if k in cl_params}

    # Some defaults. They can be overridden by kwargs. 
    cluster_defaults = dict(num_parts=1500, recursive=False, save_dir="../data/elliptic")
    loader_defaults  = dict(batch_size=20, shuffle=True, num_workers=12)

    # make sure we have the defaults, but override them if specified in wargs
    cluster_kwargs = _merge_defaults(cluster_kwargs, cluster_defaults)
    loader_kwargs  = _merge_defaults(loader_kwargs,  loader_defaults)

    # if any kwargs are left over, warn about them...and that's it.

    if kwargs:
        warnings.warn(f"cluster_loader: ignoring unexpected args {list(kwargs)}")

    # partition into clusters
    clustered = ClusterData(data, **cluster_kwargs)
    # return the loader
    return ClusterLoader(clustered, **loader_kwargs)

def _merge_defaults(kwargs, defaults):
    """
    Allows for defaults to be set on kwarg-type arguments. 
    Similar to functools.partial. 
    Merge defaults into kwargs, without overwriting existing keys.
    If kwargs is None, start from an empty dict.
    """
    if kwargs is None:
        kwargs = {}
    for k, v in defaults.items():
        kwargs.setdefault(k, v)
    return kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data loader with cluster or neighbor loader.")
    parser.add_argument("--cluster", action="store_true", help="Use cluster loader if set to True.")
    args = parser.parse_args()

    
    
    # Example usage
    data = load_elliptic(use_temporal=True, t=2, summarize=True)

    if args.cluster:
        print("Using Cluster Loader:")
        loader = make_loader(data, loader_type='cluster', batch_size=20,
                            num_parts=100, recursive=False,
                            shuffle=True,
                            num_workers=0)
    else:
        print("Using Neighbor Loader:")

        # Note: The batch size and other parameters can be adjusted as needed.
        loader = make_loader(data, loader_type='neighbor', batch_size=1024, shuffle=True,)

    for batch in loader:
        print(batch)
        break


    print()

