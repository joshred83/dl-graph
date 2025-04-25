from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import json
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict
import os

def train_traditional_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    classifier_type: str = "rf",
    test_size: float = 0.2,
    output_directory: str = "./outputs",
    timestamp: str = None,
) -> Dict[str, any]:
    """
    Train a traditional ML classifier on the embeddings

    Args:
        embeddings: Node embeddings from DOMINANT
        labels: Node labels (ground truth)
        classifier_type: Type of classifier ("rf" for Random Forest, "mlp" for MLP)
        test_size: Fraction of data to use for testing
        output_directory: Directory to save results
        timestamp: Timestamp for file naming

    Returns:
        dict: Trained model and evaluation metrics
    """
    os.makedirs(output_directory, exist_ok=True)

    # Filter out unknown labels (class 2)
    mask = labels != 2
    filtered_embeddings = embeddings[mask]
    filtered_labels = labels[mask]

    # Convert labels: 0=licit (normal), 1=illicit (anomaly)
    # Keep original binary classification (0=licit, 1=illicit)
    # This is different from anomaly detection where we might invert

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        filtered_embeddings, filtered_labels, test_size=test_size, random_state=42, stratify=filtered_labels
    )

    # Select and train classifier
    if classifier_type.lower() == "rf":
        classifier = RandomForestClassifier(
            n_estimators=100, max_depth=None, n_jobs=-1, random_state=42
        )
        model_name = "RandomForest"
    elif classifier_type.lower() == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=300, random_state=42, early_stopping=True
        )
        model_name = "MLP"
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    # Train model
    print(f"Training {model_name} classifier...")
    classifier.fit(X_train, y_train)

    # Evaluate model
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1] if hasattr(classifier, "predict_proba") else None

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),   
    }
    
    if y_proba is not None:
        metrics["auc"] = roc_auc_score(y_test, y_proba)

    # Print metrics
    print(f"\n{model_name} Classifier Metrics:")
    for metric_name, metric_value in metrics.items():
        if metric_name != "classification_report":
            print(f"{metric_name}: {metric_value:.3f}")
        else:
            print(f"{metric_name}:\n{json.dumps(metric_value, indent=2)}")

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and metrics
    model_file = os.path.join(output_directory, f"{model_name.lower()}_model_{timestamp}.pkl")
    import pickle
    with open(model_file, "wb") as f:
        pickle.dump(classifier, f)
    print(f"Model saved to {model_file}")

    metrics_file = os.path.join(output_directory, f"{model_name.lower()}_metrics_{timestamp}.json")
    with open(metrics_file, "w") as f:
        json.dump({k: v for k, v in metrics.items()}, f)
    print(f"Metrics saved to {metrics_file}")

    return {
        "model": classifier,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }