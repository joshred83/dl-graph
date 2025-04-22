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
from src.backbone import HybridGCNGATBackbone, GATBackbone, GCNSkipBackbone
from torch_geometric.nn import GCN
from src.transforms import Interpolator, Perturber
from torch_geometric.transforms import Compose



def load_dataset(root=None, 
        force_reload=False,
        use_aggregated=True,
        use_temporal=False,
        t=None,
        summarize=False, mask="train"):
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
    
    if mask is None:
        return data
    
    """input_nodes = data.train_mask if use_train_mask else data.test_mask
    # data already implements the train/test split

    # Create a NeighborLoader for the dataset
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        input_nodes=input_nodes,
    )"""

    assert (mask == 'train' or mask == 'test'), "if mask is not None, must be 'train' or 'test'"
    if mask == "train":
        input_nodes= data.train_mask
    elif mask == "test":
        input_nodes= data.test_mask
    return data, input_nodes

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

    model_params = {
        'hid_dim': config.get('hid_dim', 64),
        'num_layers': config.get('num_layers', 4),
        'dropout': config.get('dropout', 0.0),
        'weight_decay': config.get('weight_decay', 0.0),
        'act': config.get('act', F.relu),
        'sigmoid_s': config.get('sigmoid_s', False),
        'contamination': config.get('contamination', 0.1),
        'lr': config.get('lr', 0.001),
        'epoch': config.get('epoch', 25),
        'gpu': config.get('gpu', -1),
        'batch_size': config.get('batch_size', 0),
        'num_neigh': config.get('num_neigh', -1),
        'weight': config.get('weight', 0.5),
        'verbose': config.get('verbose', 1),
    }

    backbone = config.get("backbone", "gcn")
    assert backbone in [
            'gcn',
            'gat',
            'hybrid',
        ], "Backbone must be one of ['gcn', 'gat', 'hybrid']"

    match backbone:
        case 'gcn':
            model_params['backbone'] = GCN
        case 'gat':
            model_params['backbone'] = GATBackbone
        case 'hybrid':
            model_params['backbone'] = HybridGCNGATBackbone
        case 'gcn_skip':
            model_params['backbone'] = GCNSkipBackbone

    print(f"creating model with {model_params}")

    # Create the model
    model = DOMINANT(
        **model_params
    )

    model.save_emb = True
    
    # suggested example params for testing
    #mymodel = DOMINANT(num_neigh=[10,10], num_layers=2,hid_dim=32, verbose=3, batch_size=512, epoch=25, save_emb=True)

    return model

def _create_loader(
    data: EllipticBitcoinDataset,
    num_workers=4,
    batch_size=8196,
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
    print("data loaded from loader:")
    print(pyg_data)
    return pyg_data


def train_model(
        model:DOMINANT,
        data:Data,
        device='cpu',
        output_directory="./outputs",
        save_embeddings=False,
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

def transform_data(data:Data, perturb:bool=False, interpolate:bool=False) -> Data:
    if not (perturb or interpolate):
        return data
    
    if (perturb or interpolate):
        pipeline = []

        if perturb:
            print("Using perturbation")
            p = Perturber()
            pipeline.append(p)
        if interpolate:
            print("Using interpolation")
            i = Interpolator(interpolation_rate=0.1)
            pipeline.append(i)

        transform = Compose(pipeline)
    
    return transform(data)

def main(config_path=None):

    config = load_config(config_path)

    dataset, input_nodes = load_dataset(
        mask="train",
        use_aggregated=config["data"]["use_aggregated"],
        use_temporal=config["data"]["use_temporal"]
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    print(f"Timestamp: {timestamp}")

    loader = make_loader(
            data=dataset, 
            loader_type='neighbor', 
            batch_size=config["data"]["batch_size"], 
            input_nodes=input_nodes,
        )
    
    train_data = get_data_from_loader(loader)

    train_data = transform_data(train_data, perturb=config['transform']['perturb'], 
                                interpolate=config['transform']['interpolate'])

    mymodel = create_model(config=config["model"])

    trained_model = train_model(
            mymodel, 
            train_data, 
            output_directory=config["training"]["save_dir"], 
            save_embeddings=config["training"]["save_embeddings"],
            timestamp=timestamp
        )
    
    dataset, input_nodes = load_dataset(
        mask="test",
        use_aggregated=config["data"]["use_aggregated"],
        use_temporal=config["data"]["use_temporal"]
    )

    loader = make_loader(
            data=dataset, 
            loader_type='neighbor', 
            batch_size=config["data"]["batch_size"], 
            input_nodes=input_nodes,

        )    
    
    test_data = get_data_from_loader(loader)

    test_metrics = test_model(
            model=trained_model, 
            data=test_data, 
            output_directory=config["training"]["save_dir"], 
            timestamp=timestamp
        )
    
    train_test_transfer_learning(
        model=trained_model, 
        data=test_data, 
        config={
            "classifiers": config["classifiers"],
            "save_dir": config["training"]["save_dir"]
        },
        timestamp=timestamp
    )


def load_config(config_path=None):
    """
    Load configuration from a YAML file and merge it with default config
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        dict: Merged configuration dictionary
    """
    # Default configuration
    default_config = {
        "model": {
            "hid_dim": 64,
            "num_layers": 3,
            "dropout": 0.0,
            "weight_decay": 0.0,
            "contamination": 0.1,
            "backbone": "gcn",
            "lr": 0.004,
            "epoch": 25,
            "gpu": -1,  
            "batch_size": 2056,
            "num_neigh": 10,
            "weight": 0.5,
            "verbose": 2,
        },
        "data": {
            "use_aggregated": False,
            "use_temporal": False,
            "batch_size": 2056*16,
        },
        "training": {
            "save_embeddings": True,
            "save_dir": "./saved_models",
        },
        "classifiers": ["rf", "mlp"],
        "transform":{
            "perturb": True,
            "interpolate": False,
        }
    }
    
    # Load configuration from YAML if provided
    if config_path is not None:
        try:
            with open(config_path, 'r') as file:
                yaml_config = yaml.safe_load(file)
                
            # Update default config with values from YAML
            def update_nested_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        update_nested_dict(d[k], v)
                    else:
                        d[k] = v
                        
            update_nested_dict(default_config, yaml_config)
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            warnings.warn(f"Error loading config from {config_path}: {e}. Using default configuration.")
    
    print("Configuration:", json.dumps(default_config, indent=2, default=str))
    return default_config



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate DOMINANT model')
    parser.add_argument('--config', type=str, default=None, help='Path to the YAML configuration file')
    args = parser.parse_args()
    
    main(config_path=args.config)
