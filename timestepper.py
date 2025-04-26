import torch
import numpy as np
import os
import json
import datetime
from tqdm import tqdm
from src.loaders import load_elliptic, make_loader
from torch_geometric.transforms import Compose
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from src import transforms
from pygod.detector import DOMINANT
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
import warnings
import argparse
import yaml

from src.backbone import HybridGCNGATBackbone, GATBackbone, GCNSkipBackbone, GraphSAGEBackbone
from torch_geometric.nn import GCN

def backbone_map(backbone_name):
    """
    Maps the backbone name to the corresponding class.
    """
    if backbone_name == "gcn":
        return GCN
    elif backbone_name == "gcn_skip":
        return GCNSkipBackbone
    elif backbone_name == "graphsage":
        return GraphSAGEBackbone
    elif backbone_name == "gat":
        return GATBackbone
    elif backbone_name == "hybrid_gcn_gat":
        return HybridGCNGATBackbone
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
def tx(data, transforms=None):
    data = Compose(transforms)(data)
    return data

def train_and_eval_all_timesteps(config, transforms=None):
    # Setup
    output_dir = config["training"]["save_dir"]
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = {}
    all_losses = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the range of timesteps to train on
    timesteps = list(range(1, 50))

    for train_t in timesteps:
        print(f"\n=== Training on timestep {train_t} ===")
        # Load training data for this timestep
        train_data = load_elliptic(
            use_temporal=True,
            t=train_t,
            local=config["data"].get("local", False),
            use_aggregated=config["data"].get("use_aggregated", False),
            force_reload=config["data"].get("force_reload", False),
        )
        # train_loader = make_loader(
        #                     train_data,
        #                     loader_type='neighbor',
        #                     batch_size=config["data"]["batch_size"],
        #                     shuffle=False,
        #                     )

        # Create and train model
        backbone = backbone_map(config["model"]["backbone"])


        model = DOMINANT(
            hid_dim=config["model"].get("hid_dim", 64),
            num_layers=config["model"].get("num_layers", 3),
            dropout=config["model"].get("dropout", 0.0),
            weight_decay=config["model"].get("weight_decay", 0.0),
            contamination=config["model"].get("contamination", 0.1),
            lr=config["model"].get("lr", 0.004),
            epoch=config["model"].get("epoch", 25),
            gpu=config["model"].get("gpu", 0),
            batch_size=config["model"].get("batch_size", 10000),
            num_neigh=config["model"].get("num_neigh", 10),
            weight=config["model"].get("weight", 0.5),
            verbose=config["model"].get("verbose", 2),
            backbone=backbone,
        )

        # torch.manual_seed(42)
        # np.random.seed(42)
        # for batch in train_loader:
        #             # Optionally apply transforms here if needed
        #     batch = batch.to(device)
        #     if transforms:
        #         batch = tx(batch, transforms)

        model.fit(data=train_data)
        # Optionally save embeddings, etc.

        # Track training loss if available
        if hasattr(model, "losses_"):
            all_losses.setdefault(train_t, {})["train"] = model.losses_

        # Now evaluate on all remaining timesteps (including train_t for reference)
        for test_t in timesteps:
            print(f"  Predicting on timestep {test_t}...")
            test_data = load_elliptic(
                use_temporal=True,
                t=test_t,
                local=config["data"].get("local", False),
                use_aggregated=config["data"].get("use_aggregated", False),
                force_reload=False,
            )

            all_labels, all_preds, all_probs = [], [], []

            with torch.no_grad():
     
                # for batch in test_loader:
                    test_data = test_data.to(device)
                    if transforms:
                        test_data = tx(test_data, transforms)
                    lbs, probs = model.predict(test_data, return_prob=True)
                    mask = (test_data.y != 2).cpu()
                    all_labels.append(test_data.y[mask].cpu())
                    all_preds.append(lbs[mask.cpu()].cpu())
                    all_probs.append(probs[mask].cpu())
            pred_labels = torch.cat(all_preds).numpy()
            pred_probs = torch.cat(all_probs).numpy()
            labels = torch.cat(all_labels).numpy()

            # Metrics
            metrics = {
                "roc_auc": float(roc_auc_score(labels, pred_probs)),
                "accuracy": float(accuracy_score(labels, pred_labels)),
                "f1": float(f1_score(labels, pred_labels)),
                "precision": float(precision_score(labels, pred_labels)),
                "recall": float(recall_score(labels, pred_labels)),
                "classification_report": classification_report(labels, pred_labels, output_dict=True),
            }
            all_metrics.setdefault(train_t, {})[test_t] = metrics

            # Optionally track test loss if available
            if hasattr(model, "losses_"):
                all_losses.setdefault(train_t, {})[test_t] = model.losses_

       # Save model if needed
        # Save after each train_t
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
        metrics_path = os.path.join(output_dir, f"metrics_trainT_{train_t}_{timestamp}.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics[train_t], f)
        print(f"Saved metrics for train_t={train_t} to {metrics_path}")
    

    # Save all metrics and losses at the end
    with open(os.path.join(output_dir, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f)
    with open(os.path.join(output_dir, "all_losses.json"), "w") as f:
        json.dump(all_losses, f)
    print("All metrics and losses saved.")
    return all_metrics, all_losses

def load_config(config_path=None):
    # (Same as your previous load_config)
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
            "gpu": 0,
            "batch_size": 2056*4,
            "num_neigh": 10,
            "weight": 0.5,
            "verbose": 2,
        },
        "data": {
            "use_aggregated": False,
            "use_temporal": True,
            "local": True,
            "batch_size": 2056*4,
        },
        "training": {
            "save_embeddings": True,
            "save_dir": "./saved_models",
        },
    }
    if config_path is not None:
        try:
            with open(config_path, 'r') as file:
                yaml_config = yaml.safe_load(file)
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
    parser = argparse.ArgumentParser(description='Train DOMINANT on each timestep and evaluate on all others')
    parser.add_argument('--config', type=str, default=None, help='Path to the YAML configuration file')
    args = parser.parse_args()
    config = load_config(args.config)
    train_and_eval_all_timesteps(config)