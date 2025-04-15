import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_embeddings(text_path, video_path, labels_path, model=None, layer="raw"):
    """
    Visualize embeddings at different stages of the model
    
    Parameters:
    - text_path: path to text features CSV
    - video_path: path to video features CSV
    - labels_path: path to emotion labels CSV
    - model: trained FusionClassifier model (optional)
    - layer: which embeddings to visualize:
      * "raw": original embeddings 
      * "projected": after projection layers
      * "combined": after fusion
    """
    # Load data
    text_features = pd.read_csv(text_path)
    video_features = pd.read_csv(video_path)
    labels = pd.read_csv(labels_path)
    
    # Get feature columns
    text_cols = text_features.columns[1:] if text_features.columns[0] == 'id' else text_features.columns
    video_cols = video_features.columns[1:] if video_features.columns[0] == 'id' else video_features.columns
    
    # Extract features
    X_text = text_features[text_cols].values
    X_video = video_features[video_cols].values
    
    # Get labels
    if 'emotion' in labels.columns:
        y = labels['emotion'].values
    else:
        y = labels.iloc[:, 0].values
        
    # Choose what to visualize
    if layer == "raw":
        # Combine raw features
        combined_features = np.hstack([X_text, X_video])
        title = "Raw Feature Embeddings"
    
    elif layer == "projected" and model is not None:
        # Get projected features
        model.eval()
        with torch.no_grad():
            text_tensor = torch.FloatTensor(X_text)
            video_tensor = torch.FloatTensor(X_video)
            
            text_projected = model.text_encoder(text_tensor).numpy()
            video_projected = model.video_encoder(video_tensor).numpy()
            
            combined_features = np.hstack([text_projected, video_projected])
        title = "Projected Feature Embeddings"
    
    elif layer == "combined" and model is not None:
        # Get features just before classification
        model.eval()
        with torch.no_grad():
            text_tensor = torch.FloatTensor(X_text)
            video_tensor = torch.FloatTensor(X_video)
            
            text_projected = model.text_encoder(text_tensor)
            video_projected = model.video_encoder(video_tensor)
            combined_features = torch.cat([text_projected, video_projected], dim=1).numpy()
        title = "Combined Feature Embeddings"
    
    else:
        raise ValueError("Invalid layer or model not provided")
    
    # Reduce dimensionality for visualization
    print(f"Reducing dimensionality using t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_features = tsne.fit_transform(combined_features)
    
    # Create DataFrame for visualization
    df_plot = pd.DataFrame(reduced_features, columns=['x', 'y'])
    df_plot['emotion'] = y
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="x", y="y", 
        hue="emotion", 
        data=df_plot,
        palette="tab10",
        alpha=0.7,
        s=50
    )
    plt.title(f"{title} (t-SNE)")
    plt.savefig(f"embeddings_{layer}.png")
    plt.close()
    
    # Also try PCA
    pca = PCA(n_components=2, random_state=42)
    reduced_features_pca = pca.fit_transform(combined_features)
    
    df_plot_pca = pd.DataFrame(reduced_features_pca, columns=['x', 'y'])
    df_plot_pca['emotion'] = y
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="x", y="y", 
        hue="emotion", 
        data=df_plot_pca,
        palette="tab10",
        alpha=0.7,
        s=50
    )
    plt.title(f"{title} (PCA)")
    plt.savefig(f"embeddings_{layer}_pca.png")
    plt.close()

# Example usage
if __name__ == "__main__":
    # Paths to data
    text_features_path = "data/text/features_advanced.csv"
    video_features_path = "data/video/embeddings.csv"
    labels_path = "data/text/y_labels.csv"
    
    # Visualize raw embeddings
    visualize_embeddings(text_features_path, video_features_path, labels_path)
    
    # Load a trained model (if available)
    try:
        from multimodal_emotion_recognition import FusionClassifier
        
        # Load data to get dimensions
        text_features = pd.read_csv(text_features_path)
        video_features = pd.read_csv(video_features_path)
        labels = pd.read_csv(labels_path)
        
        text_cols = text_features.columns[1:] if text_features.columns[0] == 'id' else text_features.columns
        video_cols = video_features.columns[1:] if video_features.columns[0] == 'id' else video_features.columns
        
        text_dim = len(text_cols)
        video_dim = len(video_cols)
        num_classes = len(labels['emotion'].unique()) if 'emotion' in labels.columns else len(labels.iloc[:, 0].unique())
        
        # Create model with correct dimensions
        model = FusionClassifier(
            text_dim=text_dim,
            video_dim=video_dim,
            projection_dim=64,
            num_classes=num_classes
        )
        
        # Load weights if model was trained
        try:
            model.load_state_dict(torch.load("emotion_recognition_model.pth"))
            print("Loaded trained model weights")
            
            # Visualize projected and combined embeddings
            visualize_embeddings(text_features_path, video_features_path, labels_path, model, "projected")
            visualize_embeddings(text_features_path, video_features_path, labels_path, model, "combined")
        except FileNotFoundError:
            print("No trained model found. Only raw embeddings will be visualized.")
    except ImportError:
        print("Model module not found. Only raw embeddings will be visualized.")