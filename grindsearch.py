import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import datetime
import os
import json
import itertools
from typing import Dict, List, Optional
from tqdm import tqdm
import argparse
import yaml

# Import functions from the main script
from mainGCN import load_dataset, create_loader, load_model_for_transfer_learning, train_traditional_classifier, train_model, test_model, transform_data
from src.backbone import GATBackbone, HybridGCNGATBackbone, GCNSkipBackbone, GraphSAGEBackbone

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

import argparse
import yaml


def train_test_transfer_learning(
    model,  
    data,
    device,
    config: Dict[str, any] = None,
    timestamp: str = None,
    output_directory: str = "./outputs",
    return_metrics: bool = False,
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Train traditional ML models (RF, MLP) on embeddings from trained GNN.
    Saves and visualizes results.
    
    Args:
        model: Trained GNN model
        data: Dataset
        device: Training device
        config: Configuration parameters
        timestamp: Timestamp for file naming
        output_directory: Directory to save results
        return_metrics: Whether to return metrics dictionary
        
    Returns:
        Dictionary of metrics by classifier type if return_metrics=True
    """
    print("\nExtracting embeddings for transfer learning...")
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).detach().cpu().numpy()
    
    labels = data.y.detach().cpu().numpy()
    classifier_results = {}
    
    for classifier_type in config.get("classifiers", []):
        if classifier_type in ["rf", "mlp"]:
            print(f"\nTraining {classifier_type.upper()} classifier...")
            results = train_traditional_classifier(
                embeddings,
                labels,
                classifier_type=classifier_type,
                output_directory=output_directory,
                timestamp=timestamp,
            )
            classifier_results[classifier_type] = results
        else:
            print(f" Unknown classifier type '{classifier_type}'. Skipping")
    
    # Save consolidated results
    transfer_learning_results_path = os.path.join(output_directory, f"transfer_learning_results_{timestamp}.json")
    with open(transfer_learning_results_path, "w") as f:
        json.dump(classifier_results, f, indent=2)
    
    if return_metrics:
        return classifier_results
    return None

def plot_transfer_learning_results(
    results: Dict[str, Dict[str, float]],
    output_directory: str = "./outputs",
    experiment_name: str = "transfer_learning",
) -> None:
    """
    Plot transfer learning results comparing classifiers.
    
    Args:
        results: Dictionary of results by classifier type
        output_directory: Directory to save plots
        experiment_name: Name for the plot files
    """
    if not results:
        return
    
    # Extract metrics
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    classifier_types = list(results.keys())
    
    # Create comparison bar chart
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
    
    for i, metric in enumerate(metrics):
        values = [results[clf].get(metric, 0) for clf in classifier_types]
        ax = axes[i] if len(metrics) > 1 else axes
        
        bars = ax.bar(classifier_types, values)
        ax.set_title(f"{metric.upper()} by Classifier Type")
        ax.set_ylim(0, 1)  # Assuming metrics are between 0 and 1
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f"{experiment_name}_transfer_learning_comparison.png"))
    plt.close()
    
    # Create radar chart for comparing classifiers across metrics
    if len(classifier_types) > 1:
        # Number of variables
        N = len(metrics)
        
        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], metrics, size=10)
        
        # Draw the y-axis labels (0-100)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], size=8)
        plt.ylim(0, 1)
        
        # Plot each classifier
        for clf in classifier_types:
            values = [results[clf].get(metric, 0) for metric in metrics]
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=clf.upper())
            ax.fill(angles, values, alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("Transfer Learning Performance Comparison")
        
        plt.savefig(os.path.join(output_directory, f"{experiment_name}_transfer_learning_radar.png"))
        plt.close()
    
    return None

def plot_param_effect(results_df, param, metric="test_f1", output_directory="./grid_search_results"):
    """
    Plot the effect of a parameter on a specific metric.
    """
    plt.figure(figsize=(10, 6))
    
    # Group by parameter and calculate mean and std of the metric
    grouped = results_df.groupby(param)[metric].agg(['mean', 'std']).reset_index()
    
    plt.errorbar(grouped[param], grouped['mean'], yerr=grouped['std'], marker='o', linestyle='-')
    plt.xlabel(param)
    plt.ylabel(metric)
    plt.title(f"Effect of {param} on {metric}")
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(output_directory, f"{param}_{metric}_effect.png"))
    plt.close()

def visualize_results(results_df, output_directory="./grid_search_results"):
    """
    Create visualizations for grid search results.
    """
    os.makedirs(output_directory, exist_ok=True)
    
    # Identify metric columns and parameter columns
    metric_prefixes = ['train_', 'test_', 'rf_', 'mlp_']
    exclude_cols = ['trial', 'timestamp', 'error']
    
    # Get all columns that are metrics (not parameters)
    metric_cols = []
    for col in results_df.columns:
        for prefix in metric_prefixes:
            if col.startswith(prefix):
                metric_cols.append(col)
                break
    
    # Parameter columns are those that are not metrics or excluded
    param_cols = [col for col in results_df.columns 
                 if col not in metric_cols and col not in exclude_cols]
    
    # Heatmap for 2D parameter interactions
    if len(param_cols) >= 2:
        for i, param1 in enumerate(param_cols):
            for param2 in param_cols[i+1:]:
                if results_df[param1].nunique() > 1 and results_df[param2].nunique() > 1:
                    plt.figure(figsize=(10, 8))
                    
                    # Create pivot table for the heatmap
                    pivot = results_df.pivot_table(
                        values='test_f1', 
                        index=param1,
                        columns=param2,
                        aggfunc='mean'
                    )
                    
                    # Plot heatmap
                    plt.imshow(pivot, cmap='viridis', aspect='auto')
                    plt.colorbar(label='F1 Score')
                    
                    # Set ticks and labels
                    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
                    plt.yticks(range(len(pivot.index)), pivot.index)
                    
                    plt.xlabel(param2)
                    plt.ylabel(param1)
                    plt.title(f'F1 Score by {param1} and {param2}')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_directory, f'heatmap_{param1}_{param2}.png'))
                    plt.close()
    
    # Bar chart comparing backbones
    if 'backbone' in results_df.columns and results_df['backbone'].nunique() > 1:
        plt.figure(figsize=(12, 6))
        backbone_perf = results_df.groupby('backbone')['test_f1'].mean().sort_values(ascending=False)
        
        plt.bar(backbone_perf.index, backbone_perf.values)
        plt.ylabel('F1 Score')
        plt.title('Performance by Backbone Type')
        plt.ylim(0, 1)  # Assuming F1 is between 0 and 1
        
        # Add value labels on top of bars
        for i, v in enumerate(backbone_perf.values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, 'backbone_comparison.png'))
        plt.close()
        
        # If transfer learning results exist, create comparison plots
        if 'rf_f1' in results_df.columns and 'mlp_f1' in results_df.columns:
            # Compare GNN vs RF vs MLP for each backbone
            plt.figure(figsize=(15, 8))
            
            # Get unique backbones
            backbones = results_df['backbone'].unique()
            x = np.arange(len(backbones))
            width = 0.25  # Width of bars
            
            # Calculate mean F1 scores by backbone for each model
            gnn_scores = [results_df[results_df['backbone'] == b]['test_f1'].mean() for b in backbones]
            rf_scores = [results_df[results_df['backbone'] == b]['rf_f1'].mean() for b in backbones]
            mlp_scores = [results_df[results_df['backbone'] == b]['mlp_f1'].mean() for b in backbones]
            
            # Create grouped bar chart
            plt.bar(x - width, gnn_scores, width, label='GNN Direct')
            plt.bar(x, rf_scores, width, label='RF on Embeddings')
            plt.bar(x + width, mlp_scores, width, label='MLP on Embeddings')
            
            plt.xlabel('Backbone Architecture')
            plt.ylabel('F1 Score')
            plt.title('Performance Comparison: GNN vs Transfer Learning Models')
            plt.xticks(x, backbones)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_directory, 'gnn_vs_transfer_learning_by_backbone.png'))
            plt.close()
    
    # Create summary table of top configurations
    top_configs = results_df.sort_values('test_f1', ascending=False).head(5)
    top_configs.to_csv(os.path.join(output_directory, 'top_configurations.csv'), index=False)
    
    # Feature importance plots for parameters
    for metric in ['test_f1', 'rf_f1', 'mlp_f1']:
        if metric in results_df.columns:
            # For each parameter, plot its effect on the metric
            for param in param_cols:
                if results_df[param].nunique() > 1:  # Only if parameter has multiple values
                    plot_param_effect(results_df, param, metric, output_directory)

def grid_search(
    data,
    param_grid: Dict[str, List],
    base_config: Dict,
    output_directory: str = "./grid_search_results",
    num_trials: int = 1,
    run_transfer_learning: bool = True
) -> pd.DataFrame:
    """
    Perform grid search over specified parameters for GNN models.
    
    Args:
        data: The dataset
        param_grid: Dictionary with parameter names as keys and lists of parameter values
        base_config: Base configuration to use for parameters not in the grid
        output_directory: Directory to save results
        num_trials: Number of times to run each configuration (for statistical significance)
        run_transfer_learning: Whether to run transfer learning evaluations
        
    Returns:
        DataFrame containing results of all configurations
    """
    os.makedirs(output_directory, exist_ok=True)
    
    # Generate all combinations of parameters
    param_names = param_grid.keys()
    param_combinations = list(itertools.product(*(param_grid[param] for param in param_names)))
    
    # Prepare results dataframe
    results = []
    
    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Track total combinations and progress
    total_combinations = len(param_combinations)
    print(f"Total parameter combinations to evaluate: {total_combinations}")
    
    for combo_idx, param_values in enumerate(param_combinations):
        print(f"\n[{combo_idx+1}/{total_combinations}] Evaluating parameter combination:")
        
        # Create configuration for this combination
        config = base_config.copy()
        combo_params = dict(zip(param_names, param_values))
        for param, value in combo_params.items():
            if param == "perturb":
                if "transform" not in config:
                    config["transform"] = {}
                config["transform"]["perturb"] = value
            else:
                config[param] = value
                
        # Print current configuration
        for param, value in combo_params.items():
            print(f"  {param}: {value}")
        
        # Run trials for this configuration
        for trial in range(num_trials):
            if num_trials > 1:
                print(f"\nTrial {trial+1}/{num_trials}")
            
            # Create timestamp for this run
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
            
            # Create experiment name based on parameters
            experiment_name = f"{combo_params['backbone']}_h{combo_params['hidden_dim']}_d{combo_params['dropout']}_l{combo_params['num_layers']}"
            experiment_dir = os.path.join(output_directory, experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Save configuration
            config_path = os.path.join(experiment_dir, f"config_{timestamp}.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            
            # Transform data if needed
            transformed_data = transform_data(
                data, 
                perturb=config["transform"]["perturb"], 
                interpolate=config["transform"].get("interpolate", False)
            )
            
            # Initialize model based on backbone
            backbone_type = config["backbone"]
            backbone_classes = {
                'gcn': GCN,
                'gat': GATBackbone,
                'hybrid': HybridGCNGATBackbone,
                'gcn_skip': GCNSkipBackbone,
                'graphsage': GraphSAGEBackbone
            }
            
            model_class = backbone_classes[backbone_type]
            model = model_class(
                in_channels=data.num_features,
                hidden_channels=config["hidden_dim"],
                num_layers=config["num_layers"],
                out_channels=2,
                dropout=config["dropout"],
            )
            model = model.to(device)
            
            # Create data loader
            train_loader = create_loader(
                transformed_data,
                batch_size=config["batch_size"],
                num_neighbors=config["num_neighbors"],
                use_train_mask=True,
            )
            
            # Train the model
            try:
                print(f"\nTraining {backbone_type} model with {config['hidden_dim']} hidden units, "
                      f"{config['num_layers']} layers, dropout {config['dropout']}...")
                
                model, training_metrics = train_model(
                    model,
                    train_loader,
                    learning_rate=config["learning_rate"],
                    device=device,
                    num_epochs=config["num_epochs"],
                    output_directory=experiment_dir,
                    timestamp=timestamp,
                    loss_type=config["loss_type"],
                    gamma=config["gamma"],
                )
                
                # Test the model
                print("\nEvaluating model on test set...")
                test_metrics = test_model(
                    model,
                    transformed_data,
                    device,
                    batch_size=config["batch_size"],
                    num_neighbors=config["num_neighbors"],
                    output_directory=experiment_dir,
                    threshold=config.get("threshold", 0.5),
                    timestamp=timestamp,
                    loss_type=config["loss_type"],
                    gamma=config["gamma"],
                )
                
                # Run transfer learning if enabled
                transfer_learning_metrics = {}
                if run_transfer_learning:
                    print("\nRunning transfer learning with RF and MLP classifiers...")
                    # Configure transfer learning options
                    tl_config = config.copy()
                    tl_config["transfer_learning"] = True
                    tl_config["classifiers"] = ["rf", "mlp"]
                    
                    # Run transfer learning
                    tl_results = train_test_transfer_learning(
                        model,
                        transformed_data,
                        device,
                        config=tl_config,
                        timestamp=timestamp,
                        output_directory=experiment_dir,
                        return_metrics=True
                    )
                    
                    # Store transfer learning metrics
                    transfer_learning_metrics = tl_results
                    
                    # Plot transfer learning results
                    plot_transfer_learning_results(
                        tl_results, 
                        output_directory=experiment_dir,
                        experiment_name=f"{backbone_type}_{config['hidden_dim']}_{config['num_layers']}_{timestamp}"
                    )
                
                # Record results
                result = {
                    "trial": trial,
                    "timestamp": timestamp,
                    **combo_params,
                    "train_loss": training_metrics["loss_history"][-1],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_f1": test_metrics["f1"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_roc_auc": test_metrics.get("roc_auc", None),
                }
                
                # Add transfer learning metrics if available
                if transfer_learning_metrics:
                    for classifier_type, metrics in transfer_learning_metrics.items():
                        for metric_name, value in metrics.items():
                            result[f"{classifier_type}_{metric_name}"] = value
                
                results.append(result)
                
                # Save intermediate results
                results_df = pd.DataFrame(results)
                results_df.to_csv(os.path.join(output_directory, "grid_search_results.csv"), index=False)
                
            except Exception as e:
                print(f"Error during training/testing: {e}")
                # Record failure
                result = {
                    "trial": trial,
                    "timestamp": timestamp,
                    **combo_params,
                    "error": str(e),
                }
                results.append(result)
                
    # Create final dataframe
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_path = os.path.join(output_directory, "grid_search_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Grid search results saved to {results_path}")
    
    # Generate summary of best results
    if not results_df.empty and "test_f1" in results_df.columns:
        # Find best configuration based on F1 score
        best_idx = results_df["test_f1"].idxmax()
        best_config = results_df.loc[best_idx]
        
        print("\nBest configuration (GNN):")
        for param in param_names:
            print(f"  {param}: {best_config[param]}")
        print(f"  F1 Score: {best_config['test_f1']}")
        print(f"  Accuracy: {best_config['test_accuracy']}")
        print(f"  Precision: {best_config['test_precision']}")
        print(f"  Recall: {best_config['test_recall']}")
        
        # Find best transfer learning configurations if available
        if 'rf_f1' in results_df.columns:
            rf_best_idx = results_df["rf_f1"].idxmax()
            rf_best_config = results_df.loc[rf_best_idx]
            
            print("\nBest configuration (RF Transfer Learning):")
            for param in param_names:
                print(f"  {param}: {rf_best_config[param]}")
            print(f"  F1 Score: {rf_best_config['rf_f1']}")
            print(f"  Accuracy: {rf_best_config['rf_accuracy']}")
            print(f"  Precision: {rf_best_config['rf_precision']}")
            print(f"  Recall: {rf_best_config['rf_recall']}")
            
        if 'mlp_f1' in results_df.columns:
            mlp_best_idx = results_df["mlp_f1"].idxmax()
            mlp_best_config = results_df.loc[mlp_best_idx]
            
            print("\nBest configuration (MLP Transfer Learning):")
            for param in param_names:
                print(f"  {param}: {mlp_best_config[param]}")
            print(f"  F1 Score: {mlp_best_config['mlp_f1']}")
            print(f"  Accuracy: {mlp_best_config['mlp_accuracy']}")
            print(f"  Precision: {mlp_best_config['mlp_precision']}")
            print(f"  Recall: {mlp_best_config['mlp_recall']}")
        
        # Generate plots for best parameters
        # E.g., effect of hidden_dim on F1 score
        for param in param_names:
            if len(param_grid[param]) > 1:
                plot_param_effect(results_df, param, metric="test_f1", output_directory=output_directory)
    
    return results_df

def main():
    """
    Main function to run the grid search.
    """
    parser = argparse.ArgumentParser(description='Run grid search for GNN models on Elliptic Bitcoin dataset')
    parser.add_argument('--output_dir', type=str, default='./grid_search_results', help='Directory to save results')
    parser.add_argument('--data_root', type=str, default='data/elliptic', help='Path to dataset')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of trials per configuration')
    parser.add_argument('--no_transfer_learning', action='store_true', help='Disable transfer learning evaluation')
    parser.add_argument('--reduced', action='store_true', 
                        help='Run with reduced parameter grid for faster execution')
    args = parser.parse_args()
    
    # Define parameter grid
    if args.reduced:
        # Reduced parameter grid for quicker testing
        param_grid = {
            'backbone': ['gcn', 'gat', 'hybrid','graphsage'],  # Only two backbone types  
            'hidden_dim': [128],               # Only one hidden dimension
            'dropout': [0.3],                 # Only one dropout rate 
            'learning_rate': [0.01],         # Only one learning rate 
            'gamma': [1.0],                   # Only one gamma value
            'num_layers': [2],                # Only one layer count
            'batch_size': [2048],             # Only one batch size
            'loss_type': ['focal', 'weighted_ce', 'ce'],
            'perturb': [False]                # Only one perturbation setting
        }
        print("Running with reduced parameter grid for faster execution")
    else:
        # Full parameter grid
        param_grid = {
            'backbone': ['gcn', 'gat', 'hybrid', 'gcn_skip', 'graphsage'],
            'hidden_dim': [64, 128],
            'dropout': [0.1, 0.3],
            'learning_rate': [0.001, 0.01],
            'gamma': [1.0, 2.0],
            'num_layers': [2, 3],
            'batch_size': [2048],
            'perturb': [False, True]
        }
    
    # Define base configuration
    base_config = {
        'num_epochs': 100,
        'num_neighbors': [10, 10],
        'save_dir': args.output_dir,
        'data_root': args.data_root,
        'threshold': 0.5,
        'transfer_learning': False,
        'load_model_path': None,
        'transform': {
            'interpolate': False,
        }
    }

    print("Base configuration:")
    for key, value in base_config.items():
        print(f"  {key}: {value}")
    
    # Load dataset
    print("Loading dataset...")
    data = load_dataset(root=args.data_root)
    
    # Run grid search
    print("Starting grid search...")
    results = grid_search(
        data,
        param_grid,
        base_config,
        output_directory=args.output_dir,
        num_trials=args.num_trials,
        run_transfer_learning=not args.no_transfer_learning
    )
    
    # Visualize results
    print("Creating visualizations...")
    visualize_results(results, output_directory=args.output_dir)
    
    # Create summary visualizations for transfer learning results if applicable
    if not args.no_transfer_learning and 'rf_f1' in results.columns:
        print("Creating transfer learning summary visualizations...")
        
        # Compare GNN vs Transfer Learning performance
        plt.figure(figsize=(12, 8))
        
        # Extract backbone types
        backbones = results['backbone'].unique()
        
        # Prepare data for grouped bar chart
        x = np.arange(len(backbones))
        width = 0.2  # width of bars
        
        # Calculate average F1 scores by backbone for each model type
        gnn_scores = []
        rf_scores = []
        mlp_scores = []
        
        for backbone in backbones:
            backbone_results = results[results['backbone'] == backbone]
            gnn_scores.append(backbone_results['test_f1'].mean())
            rf_scores.append(backbone_results['rf_f1'].mean())
            mlp_scores.append(backbone_results['mlp_f1'].mean())
        
        # Create grouped bar chart
        plt.bar(x - width, gnn_scores, width, label='GNN')
        plt.bar(x, rf_scores, width, label='RF')
        plt.bar(x + width, mlp_scores, width, label='MLP')
        
        plt.xlabel('Backbone Architecture')
        plt.ylabel('F1 Score')
        plt.title('Performance Comparison: GNN vs Transfer Learning')
        plt.xticks(x, backbones)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'gnn_vs_transfer_learning.png'))
        plt.close()
    
    print("Grid search completed!")

if __name__ == "__main__":
    main()