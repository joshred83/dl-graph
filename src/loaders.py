from torch_geometric.datasets import EllipticBitcoinDataset, EllipticBitcoinTemporalDataset
from torch_geometric.loader import NeighborLoader, ClusterLoader, ClusterData, DataLoader



import inspect
import warnings
import argparse
from src.utils import summarize_graph, _merge_defaults
from src.local.elliptic import EllipticBitcoinDataset as LocalEllipticBitcoinDataset
from src.local.elliptic_temporal import EllipticBitcoinTemporalDataset as LocalEllipticBitcoinTemporalDataset
def load_elliptic(
    root=None, 
    force_reload=False,
    use_aggregated=False,
    use_temporal=False,
    t=None,
    summarize=False,
    local=False

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
    if local:
        # Use the local version of the dataset
        EBD = LocalEllipticBitcoinDataset
        EBDT = LocalEllipticBitcoinTemporalDataset
    else:
        # Use the PyG version of the dataset
        EBD = EllipticBitcoinDataset
        EBDT = EllipticBitcoinTemporalDataset
    if use_temporal and t is None:
        raise ValueError("Temporal data requires a time step (t) to be specified.")
    # default root to the data directory if not specified
    if root is None:
        root = "../data/elliptic"

    if use_temporal:
        dataset = EBDT(root=root, force_reload=force_reload, t=t)
    else:
        dataset = EBD(root=root, force_reload=force_reload)

    data = dataset[0]

    if use_aggregated:
        # data already contains aggregated features, no action needed
        pass
    else:
        # Remove the aggregated features
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

    else:   
        raise ValueError(f"Unknown loader type: {loader_type}.")

def neighbor_loader(data, **kwargs):
    """
    data   : a PyG Data graph
    kwargs : dict containing any args for NeighborLoader

    Returns:
    #    loader: A data loader for the graph data object.
    """

    # Some defaults. They can be overridden by kwargs. 
    defaults = dict(batch_size=2048, shuffle=False, num_neighbors=[10, 10], input_nodes=None)

    # make sure we have the defaults, but override them if specified in kwargs
    kwargs = _merge_defaults(kwargs, defaults)

    # if any kwargs are left over, warn about them...and that's it.
    actual_kwargs = (set(inspect.signature(NeighborLoader.__init__).parameters) |
                    set(inspect.signature(DataLoader.__init__).parameters))
    actual_kwargs -= {"self", "data", "kwargs"}  # drop 'self' & first positional
    wrong_kwargs = set(kwargs.keys()) - actual_kwargs
    for k in wrong_kwargs:
        warnings.warn(f"neighbor_loader: ignoring unexpected args: ** {k}:{kwargs[k]} **" )
        kwargs.pop(k)

    # return the loader
    return NeighborLoader(data, **kwargs)


def cluster_loader(data, _raise=True, **kwargs):
    """
    data   : a PyG Data graph
    kwargs : dict containing any args for ClusterData or ClusterLoader

    Returns:
       loader: A data loader for the graph data object.
    """
    if _raise:
        raise NotImplementedError("This algorithm throws SegFaults. If you want to try anyway, set _raise=False") 
    # grab everything
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
    loader_defaults  = dict(batch_size=20, shuffle=False, num_workers=12)

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


if __name__ == "__main__":
    from warnings import warn

    # Minimal tests for Local datasets
    load_elliptic_test_cases = [
        ("force_reload", {"force_reload": True}),
        ("static default", {}),
        ("static local", {"local": True}),
        ("temporal t=1", {"use_temporal": True, "t": 1}),
        ("temporal t=32 local", {"use_temporal": True, "t": 32, "local": True}),
        ("summarize", {"summarize": True}),
        ("summarize local", {"summarize": True, "local": True}),
        ("force_reload local", {"force_reload": True, "local": True}),
        ("static default", {}),
        ("static local", {"local": True}),
        ("temporal t=17", {"use_temporal": True, "t": 17}),
        ("temporal t=49 local", {"use_temporal": True, "t": 49, "local": True}),
        ("summarize", {"summarize": True}),
        ("summarize local", {"summarize": True, "local": True}),
    ]
    results = []
    for desc, kwargs in load_elliptic_test_cases:
        print(f"Testing: {desc}")
        
        try:
            data = load_elliptic(**kwargs)
            results.append(f"    Success: {desc}\n"
                         + f"      nodes={data.num_nodes}, edges={data.num_edges}")
        except Exception as e:
            results.append(f"    Failed: {desc}\n"
                         + f"      error={e}")
    print("\n".join(results))

    t_data = load_elliptic(root="../data/elliptic", force_reload=False,local=True, use_temporal=True, t=1)

    loader = make_loader(
        t_data,
        loader_type='neighbor',
        batch_size=1024,
        shuffle=True,
        test_warning="foo")
    

    [print(i) for i in loader]
