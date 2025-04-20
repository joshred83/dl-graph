import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from src.dominant import (
    DOMINANTAugmented,
)  # note we're using the custom DOMINANTAugmented class here
from tqdm import tqdm  #
import datetime
import os
import json
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from typing import Dict, Tuple
import numpy as np
from src.loaders import load_elliptic, make_loader
from src.traditional_models import train_traditional_classifier
import argparse
import yaml
import warnings
from pygod.detector import DOMINANT
from src.loaders import neighbor_loader

def load_dataset(root=None, 
        force_reload=False,
        use_aggregated=True,
        use_temporal=False,
        t=None,
        summarize=False):
    """
    Load the EllipticBitcoinDataset

    Args:
        root (str): Directory to store the dataset
        force_reload (bool): Whether to reload the dataset

    Returns:
        data: The loaded dataset
    """
    #dataset = EllipticBitcoinDataset(root=root, force_reload=force_reload)
    #data = dataset[0]
    data = load_elliptic(root=root, 
        force_reload=force_reload,
        use_aggregated=use_aggregated,
        use_temporal=use_temporal,
        t=t,
        summarize=summarize)
    return data

def create_model(config=None) -> Tuple[DOMINANT, torch.device]:
    """Create the DOMINANT model and move it to the appropriate device.
    Args:
        data: The graph data object.
        config: Configuration dictionary for the model.
        - forces save_emb to True
    Returns:
        model: The DOMINANT model.
        device: The device (CPU or GPU) on which the model is located.
    """

    # Create the model
    model = DOMINANT(
        batch_size=512,
        epoch=50,
        verbose=2
    )

    model.save_emb = True
    
    # suggested example params for testing
    #mymodel = DOMINANT(num_neigh=[10,10], num_layers=2,hid_dim=32, verbose=3, batch_size=512, epoch=25, save_emb=True)

    return model

def _create_loader(
    data: EllipticBitcoinDataset,
    num_workers=4,
    batch_size=8192,
    num_neighbors=[10, 10],
    use_train_mask=True,
) -> NeighborLoader:
    """
    Create a NeighborLoader for the dataset

    Need to test vs. neighbor_loader

    Args:
        data: The dataset
        batch_size (int): The batch size
        num_neighbors (list): The number of neighbors to sample
        use_train_mask (bool): Whether to use the train mask

    Returns:
        loader: The NeighborLoader
    """
    input_nodes = data.train_mask if use_train_mask else data.test_mask
    # data already implements the train/test split

    # Create a NeighborLoader for the dataset
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        input_nodes=input_nodes,
    )

    return loader

def get_data_from_loader(loader:NeighborLoader, device='cpu') -> Data:
    mydata = next(iter(loader))
    mydata = mydata.to(device)
    pyg_data = Data(
        x=mydata.x,
        edge_index=mydata.edge_index,
        y=mydata.y if hasattr(mydata, 'y') else None
    )
    return pyg_data


def train_model(
        model:DOMINANT,
        data:Data,
        device='cpu',
        output_directory="./outputs",
        save_embeddings=True,
        timestamp:str=None) -> DOMINANT:
    print("training model...")
    os.makedirs(output_directory, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    data = data   

    # Train the DOMINANT model
    with torch.set_grad_enabled(True):  # Explicitly control gradient tracking
        model.fit(data)
    print("model training complete!")
    del data

    if save_embeddings:
        embeddings = model.emb.detach().cpu().numpy()
        output_path = os.path.join(output_directory, f'embeddings_{timestamp}.npy')
        np.save(output_path, embeddings)
        print(f"embeddings saved to {output_path}")

        labels = model.label_.detach().cpu().numpy()
        labels_path = os.path.join(output_directory, f"labels_{timestamp}.npy")
        np.save(labels_path, labels)
        print(f"labels saved to {labels_path}")

    return model

def calculate_metrics(model:DOMINANT, pyg_data:Data, output_directory="./outputs", output_append="train", timestamp:str=None, ):
    os.makedirs(output_directory, exist_ok=True)

    predicted_labels, predicted_probs = model.predict(pyg_data, return_prob=True)

    # Calculate metrics
    if hasattr(pyg_data, 'y'):
        mask = pyg_data.y != 2
        labels = pyg_data.y.cpu().numpy()[mask]
        predicted_labels = predicted_labels.cpu().numpy()[mask]
        predicted_probs = predicted_probs.cpu().numpy()[mask]

        accuracy = accuracy_score(labels, predicted_labels)
        f1 = f1_score(labels, predicted_labels)
        precision = precision_score(labels, predicted_labels)
        recall = recall_score(labels, predicted_labels)
        class_report = classification_report(labels, predicted_labels)
        class_report_str = classification_report(labels, predicted_labels, output_dict=True)

        roc_auc = roc_auc_score(labels, predicted_probs)


        print(f'ROC AUC: {roc_auc:.4f}')
        print(f"Classification Report:\n {class_report}")


        metrics = {
        "roc_auc": float(roc_auc),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "classification_report": class_report_str,
        }
        metrics_path = os.path.join(output_directory, f"metrics_{timestamp}_{output_append}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        print(f"classification report saved to {metrics_path}")

        return metrics


def test_model(
        model:DOMINANT,
        data:Data,
        device='cpu',
        output_directory="./outputs",
        timestamp:str=None) -> DOMINANT:
    print("testing model...")
    os.makedirs(output_directory, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    data = data.to(device)

    metrics = calculate_metrics(model, data, output_directory=output_directory, output_append="test", timestamp=timestamp)    
    return metrics

def train_test_transfer_learning(
        model:DOMINANT,
        data:Data,
        config:Dict[str, any] =None,
        timestamp:str =None,):
    
    _, _, embeddings = model.predict(data, return_prob=True, return_emb=True)
    #embeddings = model.emb.detach().cpu().numpy()
    embeddings = embeddings.detach().cpu().numpy()
    #print(embeddings.shape)
    labels = data.y.detach().cpu().numpy()
    #print(labels.shape)

    classifier_results = {}
    for classifier_type in config.get("classifiers", []):
        if classifier_type in ["rf", "mlp"]:
            print(f"\nTraining {classifier_type.upper()} classifier...")
            classifier_results[classifier_type] = train_traditional_classifier(
                embeddings,
                labels,
                classifier_type=classifier_type,
                output_directory=config["save_dir"],
                timestamp=timestamp,
            )
        else:
            print(f" Unknown classifier type '{classifier_type}'. Skipping")  

def main(config=None):

    if config is None:
        config = {
            "classifiers": ["rf", "mlp"],
            "save_dir": "saved_models"
        }

    dataset = load_dataset()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    print(f"Timestamp: {timestamp}")

    loader = make_loader(data=dataset, loader_type='neighbor', batch_size=8196)

    train_data = get_data_from_loader(loader)

    mymodel = create_model(config=config)

    trained_model = train_model(mymodel, train_data, output_directory=config["save_dir"], timestamp=timestamp)

    test_data = get_data_from_loader(loader)

    test_model(model=trained_model, data=test_data, output_directory=config["save_dir"], timestamp=timestamp)

    train_test_transfer_learning(
        model=trained_model, 
        data=train_data, 

        config=config,
        timestamp=timestamp
    )

if __name__ == "__main__":
    main()
