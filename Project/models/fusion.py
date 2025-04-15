# models/fusion.py
import torch
import torch.nn as nn
from .encoders import TextEncoder, VideoEncoder

class EmotionClassifier(nn.Module):
    """Multimodal fusion model for emotion recognition"""
    
    def __init__(self, text_dim, video_dim, projection_dim, num_classes, dropout_rate=0.5):
        """
        Args:
            text_dim (int): Text feature dimension
            video_dim (int): Video feature dimension
            projection_dim (int): Projection dimension for each modality
            num_classes (int): Number of emotion classes
            dropout_rate (float): Dropout rate for classifier
        """
        super(EmotionClassifier, self).__init__()
        
        # Encoders
        self.text_encoder = TextEncoder(text_dim, projection_dim)
        self.video_encoder = VideoEncoder(video_dim, projection_dim)
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim, num_classes)
        )
    
    def forward(self, text_features, video_features):
        """
        Forward pass
        
        Args:
            text_features (torch.Tensor): Text features [batch_size, text_dim]
            video_features (torch.Tensor): Video features [batch_size, video_dim]
            
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
        """
        # Encode individual modalities
        text_encoded = self.text_encoder(text_features)
        video_encoded = self.video_encoder(video_features)
        
        # Concatenate features (simple fusion)
        fused_features = torch.cat([text_encoded, video_encoded], dim=1)
        
        # Classification
        logits = self.classifier(fused_features)
        return logits
    
    def get_embeddings(self, text_features, video_features, layer="combined"):
        """
        Get embeddings at different stages
        
        Args:
            text_features (torch.Tensor): Text features
            video_features (torch.Tensor): Video features
            layer (str): Which embeddings to return:
                - "text_projected": Text features after projection
                - "video_projected": Video features after projection
                - "combined": Combined features after fusion
                
        Returns:
            torch.Tensor: Requested embeddings
        """
        if layer == "text_projected":
            return self.text_encoder(text_features)
        elif layer == "video_projected":
            return self.video_encoder(video_features)
        elif layer == "combined":
            text_encoded = self.text_encoder(text_features)
            video_encoded = self.video_encoder(video_features)
            return torch.cat([text_encoded, video_encoded], dim=1)
        else:
            raise ValueError(f"Unknown layer: {layer}")

    def get_attention_weights(self):
        """
        For future implementation: return attention weights
        if attention-based fusion is used
        """
        return None  # Placeholder for future extension


# Alternative fusion methods (for future use)
class AttentionFusion(nn.Module):
    """
    Attention-based fusion (more sophisticated than simple concatenation)
    This is provided as a template for future enhancement
    """
    def __init__(self, dim):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, text_features, video_features):
        # Stack features for attention
        features = torch.stack([text_features, video_features], dim=1)  # [batch, 2, dim]
        
        # Compute attention weights
        combined = torch.cat([text_features, video_features], dim=1)  # [batch, dim*2]
        weights = self.attention(combined).unsqueeze(-1)  # [batch, 2, 1]
        
        # Apply attention
        weighted_features = features * weights
        fused = weighted_features.sum(dim=1)  # [batch, dim]
        
        return fused