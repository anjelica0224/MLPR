import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define model architecture
class TextEncoder(nn.Module):
    def __init__(self, input_dim=100, output_dim=64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        return self.projection(x)

class VideoEncoder(nn.Module):
    def __init__(self, input_dim=512, output_dim=64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        return self.projection(x)

class FusionClassifier(nn.Module):
    def __init__(self, text_dim=100, video_dim=512, projection_dim=64, num_classes=6):
        super().__init__()
        self.text_encoder = TextEncoder(text_dim, projection_dim)
        self.video_encoder = VideoEncoder(video_dim, projection_dim)
        
        # Combined dimensions after projection and concatenation
        combined_dim = projection_dim * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(combined_dim // 2, num_classes)
        )
    
    def forward(self, text_features, video_features):
        text_encoded = self.text_encoder(text_features)
        video_encoded = self.video_encoder(video_features)
        
        # Concatenate along the feature dimension
        combined = torch.cat([text_encoded, video_encoded], dim=1)
        
        # Final classification
        return self.classifier(combined)

# Data loading and preprocessing
def load_and_prepare_data(text_path, video_path, labels_path):
    # Load the data
    text_features = pd.read_csv(text_path)
    video_features = pd.read_csv(video_path)
    labels = pd.read_csv(labels_path)
    
    # Process text features
    # Check if the first column is unnamed (likely an index)
    if text_features.columns[0].startswith('Unnamed') or text_features.columns[0] == '':
        # Use the first column as index
        text_features = text_features.rename(columns={text_features.columns[0]: 'index'})
        text_features['index'] = text_features.index
    else:
        # Create a new index column
        text_features['index'] = text_features.index
    
    # Process video features - add an index column if not present
    if video_features.columns[0].startswith('Unnamed') or video_features.columns[0] == '':
        video_features = video_features.rename(columns={video_features.columns[0]: 'index'})
        video_features['index'] = video_features.index
    else:
        video_features['index'] = video_features.index
    
    # Process labels
    if labels.columns[0].startswith('Unnamed') or labels.columns[0] == '':
        labels = labels.rename(columns={labels.columns[0]: 'index'})
        labels['index'] = labels.index
    else:
        labels['index'] = labels.index
    
    # Remove any emotion column from features if present
    if 'emotion' in text_features.columns:
        text_features = text_features.drop(columns=['emotion'])
    
    # Get feature columns (excluding index and any possible emotion column)
    text_cols = [col for col in text_features.columns if col != 'index' and col != 'emotion']
    video_cols = [col for col in video_features.columns if col != 'index' and col != 'emotion']
    
    # Merge datasets to ensure we only use samples present in all datasets
    # First merge text features with labels
    merged_data = pd.merge(text_features, labels, on='index', how='inner')
    # Then merge with video features
    merged_data = pd.merge(merged_data, video_features, on='index', how='inner')
    
    # Extract aligned features and labels
    X_text = merged_data[text_cols].values
    X_video = merged_data[video_cols].values
    
    # Get labels (either from 'emotion' column or the second column in labels)
    if 'emotion' in merged_data.columns:
        y = merged_data['emotion'].values
    else:
        label_col = [col for col in labels.columns if col != 'index'][0]
        y = merged_data[label_col].values
    
    # Convert string labels to integers if needed
    label_map = {label: idx for idx, label in enumerate(np.unique(y))}
    y_encoded = np.array([label_map[label] for label in y])
    
    print(f"Aligned data shapes - Text: {X_text.shape}, Video: {X_video.shape}, Labels: {y_encoded.shape}")
    
    return X_text, X_video, y_encoded, label_map

# Training function
def train_model(model, X_text, X_video, y, epochs=50, batch_size=32, lr=0.001):
    # Split data
    X_text_train, X_text_val, X_video_train, X_video_val, y_train, y_val = train_test_split(
        X_text, X_video, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_text_train = torch.FloatTensor(X_text_train)
    X_video_train = torch.FloatTensor(X_video_train)
    y_train = torch.LongTensor(y_train)
    
    X_text_val = torch.FloatTensor(X_text_val)
    X_video_val = torch.FloatTensor(X_video_val)
    y_val = torch.LongTensor(y_val)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_text_train, X_video_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_text_val, X_video_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for text_batch, video_batch, label_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(text_batch, video_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for text_batch, video_batch, label_batch in val_loader:
                outputs = model(text_batch, video_batch)
                loss = criterion(outputs, label_batch)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += label_batch.size(0)
                correct += (predicted == label_batch).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        accuracy = correct / total
        val_accuracies.append(accuracy)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}')
    
    # Plot the training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    
    return model, val_accuracies[-1]

# Evaluation function
def evaluate_model(model, X_text, X_video, y, label_map):
    # Convert to tensors
    X_text = torch.FloatTensor(X_text)
    X_video = torch.FloatTensor(X_video)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(X_text, X_video)
        _, predicted = torch.max(outputs, 1)
    
    # Convert predictions to numpy
    predicted = predicted.numpy()
    
    # Invert label mapping for readability
    inv_label_map = {v: k for k, v in label_map.items()}
    y_true = np.array([inv_label_map[label] for label in y])
    y_pred = np.array([inv_label_map[label] for label in predicted])
    
    # Print classification report
    print(classification_report(y, predicted))
    
    # Plot confusion matrix
    cm = confusion_matrix(y, predicted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(inv_label_map.values()),
                yticklabels=list(inv_label_map.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    return y_pred

# Main execution
if __name__ == "__main__":
    # Set paths
    text_features_path = "data/text/features_advanced.csv"
    video_features_path = "data/video/embeddings.csv"
    labels_path = "data/text/y_labels.csv"
    
    # Load data
    X_text, X_video, y, label_map = load_and_prepare_data(
        text_features_path, video_features_path, labels_path
    )
    
    # Get dimensions
    text_dim = X_text.shape[1]  # 100 as per your information
    video_dim = X_video.shape[1]  # ResNet feature dimension
    num_classes = len(np.unique(y))
    
    # Create model
    model = FusionClassifier(
        text_dim=text_dim,
        video_dim=video_dim,
        projection_dim=64,
        num_classes=num_classes
    )
    
    # Train model
    print("Training model...")
    model, accuracy = train_model(model, X_text, X_video, y, epochs=30, batch_size=16)
    print(f"Final validation accuracy: {accuracy:.4f}")
    
    # Evaluate
    print("Evaluating model...")
    predictions = evaluate_model(model, X_text, X_video, y, label_map)
    
    # Save the model
    torch.save(model.state_dict(), "emotion_recognition_model.pth")
    print("Model saved successfully!")
    
    # Optional: Perform ablation studies
    # Train text-only model
    text_only_model = FusionClassifier(
        text_dim=text_dim,
        video_dim=1,  # Dummy dimension
        projection_dim=64,
        num_classes=num_classes
    )
    
    dummy_video = np.zeros((X_text.shape[0], 1))
    _, text_accuracy = train_model(
        text_only_model, X_text, dummy_video, y, 
        epochs=30, batch_size=16, lr=0.001
    )
    print(f"Text-only model accuracy: {text_accuracy:.4f}")
    
    # Train video-only model
    video_only_model = FusionClassifier(
        text_dim=1,  # Dummy dimension
        video_dim=video_dim,
        projection_dim=64,
        num_classes=num_classes
    )
    
    dummy_text = np.zeros((X_video.shape[0], 1))
    _, video_accuracy = train_model(
        video_only_model, dummy_text, X_video, y, 
        epochs=30, batch_size=16, lr=0.001
    )
    print(f"Video-only model accuracy: {video_accuracy:.4f}")
    
    # Compare performances
    print("\nModel Performance Comparison:")
    print(f"Full model accuracy: {accuracy:.4f}")
    print(f"Text-only model accuracy: {text_accuracy:.4f}")
    print(f"Video-only model accuracy: {video_accuracy:.4f}")