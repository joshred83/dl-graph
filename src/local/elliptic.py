# src/datasets/elliptic_with_fit.py

import torch
from torch_geometric.datasets import EllipticBitcoinDataset as _BaseElliptic
from torch_geometric.data import Data


class EllipticBitcoinDataset(_BaseElliptic):
    """
    Subclass of the Torch‑Geometric EllipticBitcoinDataset
    with an added `fit()` helper to train a GNN end‑to‑end.
    """

    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        force_reload: bool = False,
    ):
        super().__init__(root, transform, pre_transform, force_reload)
        # self.data is now loaded and available

    def process(self) -> None:
        import pandas as pd

        feat_df = pd.read_csv(self.raw_paths[0], header=None)
        edge_df = pd.read_csv(self.raw_paths[1])
        class_df = pd.read_csv(self.raw_paths[2])

        columns = {0: 'txId', 1: 'time_step'}
        feat_df = feat_df.rename(columns=columns)

        feat_df, edge_df, class_df = self._process_df(
            feat_df,
            edge_df,
            class_df,
        )

        x = torch.from_numpy(feat_df.loc[:, 2:].values).to(torch.float)

        # There exists 3 different classes in the dataset:
        # 0=licit,  1=illicit, 2=unknown
        mapping = {'unknown': 2, '1': 1, '2': 0}
        class_df['class'] = class_df['class'].map(mapping)
        y = torch.from_numpy(class_df['class'].values)

        mapping = {idx: i for i, idx in enumerate(feat_df['txId'].values)}
        edge_df['txId1'] = edge_df['txId1'].map(mapping)
        edge_df['txId2'] = edge_df['txId2'].map(mapping)
        edge_index = torch.from_numpy(edge_df.values).t().contiguous()

        # Timestamp based split:
        # train_mask: 1 - 34 time_step, test_mask: 35-49 time_step
        time_step = torch.from_numpy(feat_df['time_step'].values)
        train_mask = (time_step < 35) 
        test_mask = (time_step >= 35) 

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    test_mask=test_mask, time=time_step)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

if __name__ == "__main__":
    # setup so that it runs as a module from the root directory
    # from root: python -m src.local.elliptic
    try:
        dataset = EllipticBitcoinDataset(root="data/elliptic", force_reload=True)
        data = dataset[0]
        print(f"Loaded data: nodes={data.num_nodes}, edges={data.num_edges}, features={data.num_features}")
        print(f"Train mask sum: {data.train_mask.sum().item()}, Test mask sum: {data.test_mask.sum().item()}")
        # Test for time attribute
        if hasattr(data, "time"):
            print(f"Time attribute exists. Shape: {data.time.shape}")
            time_values = data.time.unique(return_counts=True)

            time_values = {int(k): int(v) for k, v in zip(*time_values)}
            print(f"Time value distribution:") 
            print(time_values)
        else:
            print("Time attribute does NOT exist!")
    except Exception as e:
        print(f"Error: {e}")