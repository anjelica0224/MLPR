# utils/visualization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(train_losses, val_losses, val_metrics, metric_name='Accuracy'):
    """
    Plot training and validation curves
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        val_metrics (list): Validation metrics (accuracy, F1, etc.)
        metric_name (str): Name of the metric for plot label
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Metric subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_metrics, 'g-', label=f'Validation {metric_name}')
    plt.title(f'Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_curves_{metric_name.lower()}.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plot confusion matrix
    
    Args:
        y_true (numpy.ndarray): Ground truth labels
        y_pred (numpy.ndarray): Predicted labels
        class_names (list): List of class names
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def visualize_embeddings(embeddings, labels, method='tsne', title='Feature Embeddings'):
    """
    Visualize embeddings using dimensionality reduction
    
    Args:
        embeddings (numpy.ndarray): Feature embeddings
        labels (numpy.ndarray): Labels for coloring
        method (str): Dimensionality reduction method ('tsne' or 'pca')
        title (str): Plot title
    """
    # Convert labels to strings for better plotting
    if labels.dtype != np.object_:
        labels = labels.astype(str)
    
    # Reduce dimensionality
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 2))
        reduced_embeddings = reducer.fit_transform(embeddings)
        method_name = 't-SNE'
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        method_name = 'PCA'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create DataFrame for easy plotting
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels
    })
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='x', y='y',
        hue='label',
        data=df,
        palette='tab10',
        alpha=0.7,
        s=50
    )
    plt.title(f'{title} ({method_name})')
    plt.xlabel(f'{method_name} Dimension 1')
    plt.ylabel(f'{method_name} Dimension 2')
    plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'embeddings_{method.lower()}_{title.lower().replace(" ", "_")}.png')
    plt.close()