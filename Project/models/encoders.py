# models/encoders.py
import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    """Encoder for text features"""
    
    def __init__(self, input_dim, projection_dim, dropout_rate=0.3):
        """
        Args:
            input_dim (int): Input feature dimension (FastText embedding size, default 100)
            projection_dim (int): Output feature dimension after projection
            dropout_rate (float): Dropout rate
        """
        super(TextEncoder, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Text features [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Projected features [batch_size, projection_dim]
        """
        return self.projection(x)


class VideoEncoder(nn.Module):
    """Encoder for video features"""
    
    def __init__(self, input_dim, projection_dim, dropout_rate=0.3):
        """
        Args:
            input_dim (int): Input feature dimension (ResNet feature size)
            projection_dim (int): Output feature dimension after projection
            dropout_rate (float): Dropout rate
        """
        super(VideoEncoder, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Video features [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Projected features [batch_size, projection_dim]
        """
        return self.projection(x)