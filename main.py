import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_dense_adj
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

# from pygod.models import DOMINANT



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


def create_model(data, config=None) -> Tuple[DOMINANTAugmented, torch.device]:
    """create the DOMINANT model with the specified configuration
    Args:
        data: The dataset
        config: Configuration for the model
        - if no config is provided, a default configuration will be used
    Returns:
        model: The created model
        device: The device to use (CPU or GPU)"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config is None:
        config = {
            "num_layers": 3,
            "hidden_dim": 64,
            "num_heads": 4,
            "dropout": 0.2,
            "backbone": "gcn",
            "use_interpolation": False,
            "use_perturbation": False,
            "interpolation_rate": 0.1,
            "feature_noise": 0.05,
            "structure_noise": 0.05,
            "use_adaptive_alpha": False,
            "start_alpha": 0.5,
            "end_alpha": 0.5,
            "use_aggregation": True,
            "aggregation_mean": False,
            "aggregation_max": False,
        }



    model = DOMINANTAugmented(
        num_layers=config["num_layers"],
        hid_dim=config["hidden_dim"],
        dropout=config["dropout"],
        backbone=config["backbone"],
        use_interpolation=config["use_interpolation"],
        use_perturbation=config["use_perturbation"],
        interpolation_rate=config["interpolation_rate"],
        feature_noise=config["feature_noise"],
        structure_noise=config["structure_noise"],
        use_adaptive_alpha=config["use_adaptive_alpha"],
        alpha=config["start_alpha"],
        use_aggregation=config["use_aggregation"],
        aggregation_mean=config["aggregation_mean"],
        aggregation_max=config["aggregation_max"],
        end_alpha=config["end_alpha"],
        in_dim=data.num_node_features,
    ).to(device)

    return model, device


def create_loader(
    data: EllipticBitcoinDataset,
    num_workers=4,
    batch_size=2048,
    num_neighbors=[10, 10],
    use_train_mask=True,
) -> NeighborLoader:
    """
    Create a NeighborLoader for the dataset

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


def train_model(
    model: DOMINANTAugmented,
    loader: NeighborLoader,
    learning_rate=0.001,
    device="cpu",
    num_epochs=10,
    output_directory="./outputs",
    timestamp: str = None,
) -> Dict[str, list]:
    """
    Train the DOMINANT model

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
        'loss': [],
        'attr_loss': [],
        'struct_loss': [],
        'alpha': []
    }
    
    # Test histories (will be populated during evaluation if performed during training)
    test_histories = {
        'loss': [],
        'attr_loss': [],
        'struct_loss': []
    }
    
    # Classification metrics histories
    metrics_histories = {
        'auc': [],
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': []
    }

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_total_loss = 0
        train_total_attr_loss = 0
        train_total_struct_loss = 0


        if model.use_adaptive_alpha:
            current_alpha = model.update_alpha(epoch, num_epochs)
            print(f"Current alpha: {current_alpha:.4f}")
        else:
            current_alpha = model.current_alpha

        for i, batch in enumerate(loader):
            batch = batch.to(device)
            x = batch.x
            edge_index = batch.edge_index
            batch_size = getattr(batch, "batch_size", x.size(0))

            optimizer.zero_grad()
            x_hat, s_hat = model(
                x, edge_index, apply_augmentation=False
            )  # this is working fine currently but we may need to change if we don't use augmentation
            s = to_dense_adj(edge_index)[0].to(device)

            if model.use_aggregation:
                loss_matrix = model.compute_loss(
                    x[:batch_size],  # Original features
                    x_hat[:batch_size],  # Reconstructed features
                    s[:batch_size, :batch_size],  # Original structure
                    s_hat[:batch_size, :batch_size],  # Reconstructed structure
                )
                # essentially we want to ensure consistent dimensionality across our original and reconstructed matrices
                # for features and for structure
                # the :batchsize limits makes sure we're calculating loss on the same subset of nodes

            loss_matrix = model.compute_loss(
                x[:batch_size],
                x_hat[:batch_size],
                s[:batch_size, :],
                s_hat[:batch_size],
            )

            loss = torch.mean(
                loss_matrix
            )  # batchwise mean for calculating the gradient

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # gradient clipping - eric
            optimizer.step()

            train_total_loss += loss.item()  # but it gets summed for record keeping

            with torch.no_grad():  # since we're just updating the loss scorekeeping. we might want to graph this later in reporting.
                # Attribute loss
                attr_matrix = model.loss_func(
                    x[:batch_size],
                    x_hat[:batch_size],
                    s[:batch_size, :],
                    s_hat[:batch_size],
                    weight=1.0,  #  from the docstring: `Balancing weight... between 0 and 1 inclusive between node feature and graph structure.`
                    # logically, it's set to 1.0 here because we want to calculate the attribute loss
                    # below, we will set it to 0 to calculate the structure loss
                    pos_weight_a=model.pos_weight_a,
                    pos_weight_s=model.pos_weight_s,
                    bce_s=model.bce_s,
                )
                attr_loss = torch.mean(attr_matrix).item()
                train_total_attr_loss += attr_loss

                # Structure loss
                struct_matrix = model.loss_func(
                    x[:batch_size],
                    x_hat[:batch_size],
                    s[:batch_size, :],
                    s_hat[:batch_size],
                    weight=0.0,
                    pos_weight_a=model.pos_weight_a,
                    pos_weight_s=model.pos_weight_s,
                    bce_s=model.bce_s,
                )
                struct_loss = torch.mean(struct_matrix).item()
                train_total_struct_loss += struct_loss

        batch_count = len(loader)  # Total number of batches
        avg_loss = train_total_loss / batch_count
        avg_attr_loss = train_total_attr_loss / batch_count
        avg_struct_loss = train_total_struct_loss / batch_count
        print(
            f"Avg Batch Loss: {avg_loss:.3e}, "
            f"Avg Batch Attribute Loss: {avg_attr_loss:.3e}, "
            f"Avg Batch Structure Loss: {avg_struct_loss:.3e}"
        )

        # Record training epoch metrics
        train_histories['loss'].append(avg_loss)
        train_histories['attr_loss'].append(avg_attr_loss)  
        train_histories['struct_loss'].append(avg_struct_loss)  
        train_histories['alpha'].append(current_alpha)  

        evaluate_metrics = {
            "auc": 0.0,
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
        # Evaluate the model on the training set
        # Note: This is a placeholder for actual evaluation logic
        # You may want to implement a proper evaluation function
        # evaluate_metrics = evaluate_model(model, data, device)
        # Store evaluation metrics


    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    model_path = os.path.join(output_directory, f"dominant_model_{timestamp}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    metrics = {
        "loss_history": train_histories['loss'],
        "attr_loss_history": train_histories['attr_loss'],
        "struct_loss_history": train_histories['struct_loss'],
        "alpha_history": train_histories['alpha'],
    }

    metrics_path = os.path.join(output_directory, f"train_metrics_{timestamp}.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in metrics.items()}, f
        )  # apparently json doesn't like torch tensors
    print(f"Metrics saved to {metrics_path}")

    return model, metrics

def evaluate_model(
    model: DOMINANTAugmented,
    data,
    device,
    batch_size=2048,
    num_neighbors=[10, 10],
    output_directory="./outputs",
    threshold=0.5,
    timestamp: str = None,
) -> Dict[str, float]:
    """
    Evaluate the DOMINANT model on the test dataset

    Args:
        model: The trained model
        data: The dataset
        device: Device to evaluate on
        batch_size: Batch size for evaluation
        num_neighbors: Number of neighbors to sample
        output_directory: Directory to save results
        threshold: Threshold for anomaly detection

    Returns:
        dict: Evaluation metrics
    """
    # Placeholder for evaluation logic
    # You may want to implement a proper evaluation function
    # For now, we will just return dummy metrics
    return {
        "auc": 0.0,
        "accuracy": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }

def test_model(
    model: DOMINANTAugmented,
    data,
    device,
    batch_size=2048,
    num_neighbors=[10, 10],
    output_directory="./outputs",
    threshold=0.5,
    timestamp: str = None,
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

    print(f"Model input dimension: {model.shared_encoder.convs[0].in_channels}")
    print(f"Data feature dimension: {data.num_node_features}")
    print(f"Using aggregation: {model.use_aggregation}")
    if model.use_aggregation:
        print(
            f"Aggregation methods - Mean: {model.aggregation_mean}, Max: {model.aggregation_max}"
        )
        expected_dim = data.num_node_features
        if model.aggregation_mean:
            expected_dim += data.num_node_features
        if model.aggregation_max:
            expected_dim += data.num_node_features
        print(f"Expected input dimension after aggregation: {expected_dim}")

    # Create loader for test data
    loader = create_loader(
        data, batch_size=batch_size, num_neighbors=num_neighbors, use_train_mask=False
    )

    model.eval()

    all_losses = []
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            batch = batch.to(device)
            x = batch.x
            edge_index = batch.edge_index
            batch_size = getattr(batch, "batch_size", x.size(0))

            # Forward pass through the model
            x_hat, s_hat = model(x, edge_index, apply_augmentation=True)
            s = to_dense_adj(edge_index)[0].to(device)

            # Compute loss (use the same approach as in training)
            if model.use_aggregation:
                loss_matrix = model.compute_loss(
                    x[:batch_size],
                    x_hat[:batch_size],
                    s[:batch_size, :batch_size],
                    s_hat[:batch_size, :batch_size],
                )
            else:
                loss_matrix = model.compute_loss(
                    x[:batch_size],
                    x_hat[:batch_size],
                    s[:batch_size, :],
                    s_hat[:batch_size],
                )
            print(loss_matrix[:10])
            # Get batch metrics
            batch_loss = torch.mean(loss_matrix).item()
            batch_scores = loss_matrix.cpu().numpy()
            batch_labels = batch.y[:batch_size].cpu().numpy()

            # Store results
            all_losses.append(batch_loss)
            all_scores.append(batch_scores)
            all_labels.append(batch_labels)

    # Convert lists to numpy arrays
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    # Filter out unknown labels (class 2)
    mask = all_labels != 2
    scores = all_scores[mask]
    labels = all_labels[mask]

    """The nodes are then ranked according to their anomaly scores in
    descending order, and the top-k nodes are recognized as anoma-
    lies - page 7 in https://arxiv.org/pdf/2106.07178"""

    sort_indices = np.argsort(scores)[::-1] # descending order
    sorted_scores = scores[sort_indices]
    #print(sorted_scores[:50])
    sorted_labels = labels[sort_indices]
    #print(sorted_labels[:50])

    threshold = np.mean(sorted_labels == 1) 
    print(f"calculated threshold: {threshold:.3f}")

    num_samples = len(sorted_scores)
    num_class_1 = int(num_samples * threshold)
    predictions = np.zeros_like(sorted_labels)
    predictions[0:num_class_1] = 1  # Mark top-k as anomalies
    #predictions = 1-predictions

    #print(predictions[:50])
    #print(sorted_labels[:50])

    # Convert labels: 0=licit (normal), 1=illicit (anomaly)
    # Since in the dataset 0=licit and 1=illicit, we need to invert for evaluation
    #labels = 1 - labels  # Now 1=illicit (anomaly), 0=licit (normal)

    # Calculate evaluation metrics
    #predictions = (scores >= threshold).astype(int)
    #try:
    #    auc = roc_auc_score(labels, scores)
    #except Exception as e:
    #    print(f"Error calculating AUC: {e}")
    #    auc = 0.0

    accuracy = accuracy_score(sorted_labels, predictions)
    f1 = f1_score(sorted_labels, predictions)
    precision = precision_score(sorted_labels, predictions)
    recall = recall_score(sorted_labels, predictions)
    classification_report_output = classification_report(sorted_labels, predictions)
    classification_report_str = classification_report(
        sorted_labels, predictions, output_dict=True
    )
    avg_loss = np.mean(all_losses)

    metrics = {
        #"auc": float(auc),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "loss": float(avg_loss),
        "classification_report": classification_report_str,
    }

    # Print metrics
    print("\nTest Metrics:")
    #print(f"AUC: {metrics['auc']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"Loss: {metrics['loss']:.3e}")
    print("\nClassification Report:")
    print(classification_report_output)

    # Save metrics
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    metrics_path = os.path.join(output_directory, f"test_metrics_{timestamp}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_path}")

    return metrics


def plot_loss(
    loss_history,
    attr_loss_history,
    struct_loss_history,
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

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(attr_loss_history) + 1),
        attr_loss_history,
        "o-",
        label="Attribute Loss",
    )
    plt.plot(
        range(1, len(struct_loss_history) + 1),
        struct_loss_history,
        "o-",
        label="Structure Loss",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Attribute and Structure Loss History")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_directory, f"attr_struct_loss_plot_{timestamp}.png")
    )
    plt.close()

def extract_embeddings(
    model: DOMINANTAugmented,
    data,
    device,
    batch_size=2048,
    num_neighbors=[10, 10],
    save=True,
    output_directory="./outputs",
    timestamp: str = None,
) -> Dict[str, np.ndarray]:
    """
    Extract node embeddings from the trained DOMINANT model

    Args:
        model: The trained model
        data: The dataset
        device: Device to use
        batch_size: Batch size for processing
        num_neighbors: Number of neighbors to sample
        output_directory: Directory to save results
        timestamp: Timestamp for file naming

    Returns:
        dict: Node embeddings and labels
    """
    os.makedirs(output_directory, exist_ok=True)

    # Create loader for all data (both train and test)
    loader = create_loader(
        data, batch_size=batch_size, num_neighbors=num_neighbors, use_train_mask=None
    )

    model.eval()

    all_embeddings = []
    all_node_ids = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            batch = batch.to(device)
            x = batch.x
            edge_index = batch.edge_index
            batch_size = getattr(batch, "batch_size", x.size(0))

            # Forward pass to get embeddings (but don't need reconstructions)
            _, _ = model(x, edge_index, apply_augmentation=False)
            # The embeddings are stored in model.emb after forward pass
            batch_embeddings = model.emb[:batch_size].cpu().numpy()
            batch_labels = batch.y[:batch_size].cpu().numpy()
            batch_ids = batch.index[:batch_size].cpu().numpy() if hasattr(batch, 'index') else np.arange(batch_size)

            all_embeddings.append(batch_embeddings)
            all_node_ids.append(batch_ids)
            all_labels.append(batch_labels)

    # Concatenate results
    embeddings = np.vstack(all_embeddings)
    node_ids = np.concatenate(all_node_ids)
    labels = np.concatenate(all_labels)

    # Save embeddings
    if save:
        embeddings_file = os.path.join(output_directory, f"node_embeddings_{timestamp}.npz")
        np.savez(embeddings_file, embeddings=embeddings, node_ids=node_ids, labels=labels)
        print(f"Embeddings saved to {embeddings_file}")

    return {
        "embeddings": embeddings,
        "node_ids": node_ids,
        "labels": labels
    }

def train_test_transfer_learning(
    model: DOMINANTAugmented,  
    data,
    device,
    config: Dict[str, any] = None,
    timestamp: str = None,
) -> Dict[str, float]:
    
    embeddings_data = extract_embeddings(
        model,
        data,
        device,
        batch_size=config["batch_size"],
        num_neighbors=config["num_neighbors"],
        output_directory=config["save_dir"],
        timestamp=timestamp,
    )
    classifier_results = {}
    for classifier_type in config.get("classifiers", []):
        if classifier_type in ["rf", "mlp"]:
            print(f"\nTraining {classifier_type.upper()} classifier...")
            classifier_results[classifier_type] = train_traditional_classifier(
                embeddings_data["embeddings"],
                embeddings_data["labels"],
                classifier_type=classifier_type,
                output_directory=config["save_dir"],
                timestamp=timestamp,
            )
        else:
            print(f" Unknown classifier type '{classifier_type}'. Skipping")

def load_model_for_transfer_learning(
    model_path: str,
    data,
    device=None,
    config=None,
) -> Dict[str, any]:
    """
    Load a saved model and use it for transfer learning
    
    Args:
        model_path: Path to the saved model
        data: The dataset
        device: Device to use (if None, will use CUDA if available)
        config: Model configuration
        
    Returns:
        dict: Results from transfer learning
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a new model with the same architecture
    model, _ = create_model(data, config)
    
    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"Loaded model from {model_path}")

    return model, device

def main(config=None):
    """
    Main function to run the training and testing pipeline

    Args:
        config (dict, optional): Configuration parameters
    """
    if config is None:
        config = {
            # Model parameters
            "num_layers": 3,
            "hidden_dim": 64,
            "num_heads": 4,
            "dropout": 0.1,
            "apply_augmentation": False,
            "use_interpolation": False,
            "use_perturbation": True,
            "interpolation_rate": 0.1,
            "feature_noise": 0.05,
            "structure_noise": 0.05,
            "use_adaptive_alpha": True,
            "start_alpha": 0.6,
            "end_alpha": 0.5,
            "use_aggregation": False,
            "aggregation_mean": False,
            "aggregation_max": False,
            # Training parameters
            "batch_size": 2048,
            "num_neighbors": [10, 10],
            "learning_rate": 0.001,
            "num_epochs": 2,
            # Paths for output
            "data_root": "data/elliptic",
            "save_dir": "./saved_models",
            # threshold for our testing
            "threshold": 0.5,
            # transfer learning options
            "transfer_learning": True,
            "classifiers": ["rf", "mlp"],
            "load_model_path": None,
        }

    data = load_dataset(root=config["data_root"])
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    print(f"Timestamp: {timestamp}")

    if config["load_model_path"] is None:

        model, device = create_model(data, config)

        train_loader = create_loader(
            data,
            batch_size=config["batch_size"],
            num_neighbors=config["num_neighbors"],
            use_train_mask=True,
        )

        model, training_metrics = train_model(
            model,
            train_loader,
            learning_rate=config["learning_rate"],
            device=device,
            num_epochs=config["num_epochs"],
            output_directory=config["save_dir"],
            timestamp=timestamp,
        )
        plot_loss(
            training_metrics["loss_history"],
            training_metrics["attr_loss_history"],
            training_metrics["struct_loss_history"],
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
    
    main(config)
