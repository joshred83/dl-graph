import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import datetime
import os
import json
from typing import Dict
from tqdm import tqdm

import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCN, GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.data import Data
from src.backbone import *
from sklearn.utils.class_weight import compute_class_weight


import argparse
import yaml

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from src.losses.focal_loss import FocalLoss, reweight
from main import load_dataset, create_loader, load_model_for_transfer_learning, train_traditional_classifier
from src.transforms import Interpolator, Perturber
from torch_geometric.transforms import Compose


def train_model(
    model,
    loader: NeighborLoader,
    learning_rate=0.001,
    device="cpu",
    num_epochs=5,
    output_directory="./outputs",
    timestamp: str = None,
    loss_type: str = "focal", # 'focal' or 'ce' or 'weighted_ce'
    per_cls_weights: torch.Tensor = None,
    gamma: float = 1.0,
    beta: float = 0.0,
) -> Dict[str, list]:
    """
    Train the model

    Args:
        model: The model to train
        loader: Data loader for training data
        learning_rate (float): Learning rate for the optimizer
        device: Device to train on
        num_epochs (int): Number of epochs to train
        save_dir (str): Directory to save model and logs

    Returns:
        dict: Training metrics including loss histories
    """
    os.makedirs(output_directory, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training histories
    train_histories = {
        'loss': []
    }
    
    # Test histories (will be populated during evaluation if performed during training)
    test_histories = {
        'loss': []
    }
    
    # Classification metrics histories
    metrics_histories = {
        'auc': [],
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': []
    }
    print(f"training with {loss_type} loss on {model.__class__.__name__} model")

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        all_pred_labels = []
        all_true_labels = []
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_total_loss = 0.0

        for i, batch in enumerate(loader):
            batch = batch.to(device)

            x = batch.x
            y = batch.y
            edge_index = batch.edge_index
            batch_size = getattr(batch, "batch_size", x.size(0))

            optimizer.zero_grad()
            out = model(x, edge_index) # use full structure to allow message passing
            # filter with train_mask to only compute loss with known labels 
            train_out, train_labels = out[batch.train_mask], batch.y[batch.train_mask] #[:, :2]
            
            # check unique labels
            # print(f"unique labels in batch (torch.unique(batch.y)): {torch.unique(batch.y)}")
            # print(f"unique labels filtered (torch.unique(batch.y[batch.train_mask])): {torch.unique(train_labels)}")

            per_cls_weights = reweight(torch.bincount(train_labels).tolist(), beta=beta)

            if loss_type == 'focal':
                if len(per_cls_weights) < 2:
                    print(f"Warning: skiping batch with insufficient classes, {per_cls_weights}")
                    continue
                # print(f"training with focal, weights {torch.unique(train_labels, return_counts=True)}, gamma {gamma}, beta {beta}")
                criterion = FocalLoss(weight=per_cls_weights, gamma=gamma)
                # loss = criterion(
                #     train_out, train_labels)

            elif loss_type == 'weighted_ce':

                # print(f"unique labels in batch (torch.unique(batch.y)): {torch.unique(batch.y)}, type {batch.y.dtype}, shape {batch.y.shape}")
                # print(f"unique labels filtered (torch.unique(batch.y[batch.train_mask])): {torch.unique(train_labels)}, type {train_labels.dtype}, shape {train_labels.shape}")
                # print("testing with ce")
                # print(f"training with ce, weights {torch.unique(train_labels)}")
                weight = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.array([0, 1]),
                                        y = train_labels.cpu().numpy()                                                    
                                    )
                criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float).to(device))
            
            else:
                criterion = torch.nn.CrossEntropyLoss()
            
            loss = criterion(train_out, train_labels)
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(), max_norm=1.0
            # )  # gradient clipping - eric
            optimizer.step()
            train_total_loss += loss.item()

            pred_labels = torch.argmax(out, dim=1)[batch.train_mask]
            true_labels = batch.y[batch.train_mask]

            all_pred_labels.extend(pred_labels.cpu().numpy())
            all_true_labels.extend(true_labels.cpu().numpy())

        batch_count = len(loader)  # Total number of batches
        avg_loss = train_total_loss / batch_count
        print(
            f"Avg Batch Loss: {avg_loss:.3e}, "
        )

        # Record training epoch metrics
        train_histories['loss'].append(avg_loss)

        train_auc = roc_auc_score(all_true_labels, all_pred_labels)
        train_accuracy = accuracy_score(all_true_labels, all_pred_labels)
        train_f1 = f1_score(all_true_labels, all_pred_labels)
        train_precision = precision_score(all_true_labels, all_pred_labels)
        train_recall = recall_score(all_true_labels, all_pred_labels)

        # print(f"training metrics: auc {train_auc:.3f}, accuracy {train_accuracy:.3f}, f1 {train_f1:.3f}, precision {train_precision:.3f}, recall {train_recall:.3f}")

        metrics_histories['auc'].append(train_auc)
        metrics_histories['accuracy'].append(train_accuracy)
        metrics_histories['f1'].append(train_f1)
        metrics_histories['precision'].append(train_precision)
        metrics_histories['recall'].append(train_recall)

    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    model_path = os.path.join(output_directory, f"dominant_model_{timestamp}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    metrics = {
        "loss_history": train_histories['loss']
    }


    metrics_path = os.path.join(output_directory, f"train_metrics_{timestamp}.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in metrics.items()}, f
        )  # apparently json doesn't like torch tensors
    print(f"Metrics saved to {metrics_path}")

    plot_epoch_metrics(
        metrics_histories,
        output_directory=output_directory,
        timestamp=timestamp,
    )
    print(f"Metrics plot saved to {output_directory}/metrics_plot_{timestamp}.png")

    return model, metrics

def plot_epoch_metrics(
    metrics_histories: Dict[str, list],
    output_directory="./outputs",
    timestamp: str = None,
) -> None:
    """Plot classification metrics over epochs."""
    plt.figure(figsize=(12, 8))
    for metric, values in metrics_histories.items():
        plt.plot(range(1, len(values) + 1), values, label=metric.capitalize())
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title("Classification Metrics Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f"metrics_plot_{timestamp}.png"))
    plt.close()

    return None

def plot_loss(
    loss_history,
    # attr_loss_history,
    # struct_loss_history,
    output_directory="./outputs",
    timestamp: str = None,
) -> None:
    """
    Plot the training loss

    Args:
        loss_history: The loss history
        attr_loss_history: The attribute loss history
        struct_loss_history: The structure loss history
        output_directory (str): Directory to save the plot

    Returns:
        None
    """
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, "o-", label="Total Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Loss History")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f"loss_plot_{timestamp}.png"))
    plt.close()

    return None

def test_model(
    model: GCNSkipBackbone,
    data,
    device,
    batch_size=2048,
    num_neighbors=[10, 10],
    output_directory="./outputs",
    threshold=0.5,
    timestamp: str = None,
    loss_type: str = "focal",  # 'focal' or 'ce' or 'weighted_ce
    gamma: float = 1.0,
    beta: float = 0.0,
) -> Dict[str, float]:
    """
    Test the DOMINANT model on the test dataset

    Args:
        model: The trained model
        data: The dataset
        device: Device to test on
        batch_size: Batch size for testing
        num_neighbors: Number of neighbors to sample
        output_directory: Directory to save results
        threshold: Threshold for anomaly detection

    Returns:
        dict: Test metrics
    """
    os.makedirs(output_directory, exist_ok=True)

    # print(f"Model input dimension: {model.shared_encoder.convs[0].in_channels}")
    # print(f"Using aggregation: {model.use_aggregation}")
    # if model.use_aggregation:
    #     print(
    #         f"Aggregation methods - Mean: {model.aggregation_mean}, Max: {model.aggregation_max}"
    #     )
    #     expected_dim = data.num_node_features
    #     if model.aggregation_mean:
    #         expected_dim += data.num_node_features
    #     if model.aggregation_max:
    #         expected_dim += data.num_node_features
    #     print(f"Expected input dimension after aggregation: {expected_dim}")

    # Create loader for test data
    loader = create_loader(
        data, batch_size=batch_size, num_neighbors=num_neighbors, use_train_mask=False
    )

    model.eval()

    all_losses = []
    # all_scores = []
    all_labels = []
    all_pred = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            batch = batch.to(device)
            x = batch.x
            y = batch.y
            edge_index = batch.edge_index
            batch_size = getattr(batch, "batch_size", x.size(0))

            # Forward pass through the model
            out = model(x, edge_index)

            # Get predictions
            probs = torch.softmax(out, dim=1)
            scores = probs[:, 1]  # Assuming binary classification

            test_mask = getattr(batch, "test_mask", torch.ones_like(y, dtype=torch.bool))
            if test_mask.sum() == 0:
                continue

            y_true = y[test_mask]
            # print(f"Unique labels in test_mask: {torch.unique(y_true)}")
            y_pred_logits = out[test_mask]#[:, :2]
            # y_scores = scores[test_mask]
            test_pred = torch.argmax(y_pred_logits, dim=1)


            # test_out, test_labels = out[batch.test_mask][:, :2], batch.y[batch.test_mask]

            # Handle class imbalance (optional in test)
            if loss_type == 'focal':
                per_cls_weights = reweight(torch.bincount(y_true).tolist(), beta=beta)
                if len(per_cls_weights) < 2:
                    print(f"Warning: skiping batch with insufficient classes: {per_cls_weights}")
                    continue
                # print(f"testing with focal, weights {per_cls_weights}, gamma {gamma}, beta {beta}")
                criterion = FocalLoss(weight=per_cls_weights.to(device), gamma=gamma)
            elif loss_type == 'weighted_ce':
                # print("testing with ce")
                print(f"unique labels in test_mask: {torch.unique(y_true)}")
                weight = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.array([0, 1]),
                                        y = y_true.cpu().numpy()                                                    
                                    )
                criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float).to(device))
            
            else:
                criterion = torch.nn.CrossEntropyLoss()

            loss = criterion(y_pred_logits, y_true)
            all_losses.append(loss.item())
            all_labels.append(y_true.cpu().numpy())
            # all_scores.append(y_scores.cpu().numpy())
            all_pred.append(test_pred.cpu().numpy())

    # Convert lists to numpy arrays
    # all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    all_pred = np.concatenate(all_pred)

    # print(f"test all_scores, {all_scores}")

    # Filter out unknown labels (class 2)
    mask = all_labels != 2
    # scores = all_scores[mask]
    labels = all_labels[mask]
    preds = all_pred[mask]

    """The nodes are then ranked according to their anomaly scores in
    descending order, and the top-k nodes are recognized as anoma-
    lies - page 7 in https://arxiv.org/pdf/2106.07178"""

    # sort_indices = np.argsort(scores)[::-1] # descending order
    # sorted_scores = scores[sort_indices]
    # #print(sorted_scores[:50])
    # sorted_labels = labels[sort_indices]
    # #print(sorted_labels[:50])

    # # threshold = np.mean(sorted_labels == 1) 
    # # print(f"calculated threshold: {threshold:.3f}")

    # # num_samples = len(sorted_scores)
    # # num_class_1 = int(num_samples * threshold)
    # predictions = np.zeros_like(sorted_labels)
    # # predictions[0:num_class_1] = 1  # Mark top-k as anomalies

    # accuracy = accuracy_score(labels, scores > threshold)
    # f1 = f1_score(labels, scores > threshold)
    # precision = precision_score(labels, scores > threshold)
    # recall = recall_score(labels, scores > threshold)
    # classification_report_output = classification_report(
    #     labels, scores > threshold)
    # print(confusion_matrix(labels, scores > threshold))
    # confusion_matrix_output = confusion_matrix(
    #     labels, scores > threshold)
    # print(confusion_matrix_output)
    auc = roc_auc_score(labels, preds)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    classification_report_output = classification_report(
        labels, preds)
    confusion_matrix_output = confusion_matrix(
        labels, preds)
    # print(confusion_matrix_output)

    # classification_report_str = classification_report(
    #     labels, (scores > threshold).astype(int), output_dict=True
    # )

    classification_report_str = classification_report(
        labels, preds, output_dict=True
    )
    avg_loss = np.mean(all_losses)

    metrics = {
        "roc_auc": float(auc),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        # "loss": float(avg_loss),
        "classification_report": classification_report_str,
    }

    # Print metrics
    print("\nTest Metrics:")
    #print(f"AUC: {metrics['auc']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    # print(f"Loss: {metrics['loss']:.3e}")
    print("\nClassification Report:")
    print(classification_report_output)

    # Save metrics
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    metrics_path = os.path.join(output_directory, f"test_metrics_{timestamp}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")

    print(metrics)

    return metrics

def train_test_transfer_learning(
    model,  
    data,
    device,
    config: Dict[str, any] = None,
    timestamp: str = None,
) -> Dict[str, float]:
    
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).detach().cpu().numpy()
    
    labels = data.y.detach().cpu().numpy()
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

def main(config=None):
    """
    Main function to run the training and testing pipeline

    Args:
        config (dict, optional): Configuration parameters
    """
    if config is None:
        config = {
            # Model parameters
            "hidden_dim": 64,
            "dropout": 0.1,
            # Training parameters
            "batch_size": 2048,
            "num_neighbors": [10, 10],
            "learning_rate": 0.001,
            "num_epochs": 2,
            # Paths for output
            "data_root": "data/elliptic",
            "save_dir": "./saved_models",
            # threshold for our testing
            "threshold": 0.9,
            # transfer learning options
            "transfer_learning": False,
            "load_model_path": None,
            # focal loss
            "loss_type": "focal",
            "gamma": 1.0,
            "num_layers":2,
            "transfer_learning": True,
            "classifiers": ["rf", "mlp"],
            "backbone": "gcn_skip",
            "transform": {
                "perturb": False,
                "interpolate": False,
            },
        }

    data = load_dataset(root=config["data_root"])
    input_nodes= data.train_mask
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    print(f"Timestamp: {timestamp}")


    if config["load_model_path"] is None:

        backbone = config.get("backbone", "gcn")

        assert backbone in [
                'gcn',
                'gat',
                'hybrid',
                'gcn_skip',
                'graphsage'
            ], "Backbone must be one of ['gcn', 'gat', 'hybrid', 'gcn_skip', 'graphsage']"

        match backbone:
            case 'gcn':
                mymodel = GCN
            case 'gat':
                mymodel = GATBackbone
            case 'hybrid':
                mymodel = HybridGCNGATBackbone
            case 'gcn_skip':
                mymodel = GCNSkipBackbone
            case 'graphsage':
                mymodel= GraphSAGEBackbone

        model = mymodel(in_channels=data.num_features,
                    hidden_channels=config["hidden_dim"],
                    num_layers=config["num_layers"],
                    out_channels=2,
                    dropout=config["dropout"],
                    )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        data = transform_data(data, perturb=config['transform']['perturb'], 
                                interpolate=config['transform']['interpolate'])

        train_loader = create_loader(
            data,
            batch_size=config["batch_size"],
            num_neighbors=config["num_neighbors"],
            use_train_mask=True,
        )

        # print("Train loader input nodes:")
        # print(train_loader.input_nodes.unique())


        model, training_metrics = train_model(
            model,
            train_loader,
            learning_rate=config["learning_rate"],
            device=device,
            num_epochs=config["num_epochs"],
            output_directory=config["save_dir"],
            timestamp=timestamp,
            loss_type=config["loss_type"],
            gamma=config["gamma"],
            #beta=config["beta"],
        )
        plot_loss(
            training_metrics["loss_history"],
            output_directory=config["save_dir"],
            timestamp=timestamp,
        )

        test_metrics = test_model(
            model,
            data,
            device,
            batch_size=config["batch_size"],
            num_neighbors=config["num_neighbors"],
            output_directory=config["save_dir"],
            threshold=config["threshold"],
            timestamp=timestamp,
            loss_type=config["loss_type"],
            gamma=config["gamma"],
            #beta=config["beta"],
        )
    elif config["load_model_path"] is not None:
        print(f"Loading pre-trained model from {config['load_model_path']}")
        model, device = load_model_for_transfer_learning(
            model_path=config["load_model_path"],
            data=data,
            config=config,)



    if config["transfer_learning"]:
        # Transfer learning logic here
        train_test_transfer_learning(
            model,
            data,
            device,
            config=config,
            timestamp=timestamp,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate DOMINANT model on Elliptic Bitcoin dataset')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    args = parser.parse_args()
    
    config = None

    if args.config:
        try:
            with open(args.config, 'r') as file:
                config = yaml.safe_load(file)
            print(f"Loaded configuration from {args.config}")
            print(f"Configuration: {config}")
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            print("Using default configuration instead.")
    
    main(config=config)