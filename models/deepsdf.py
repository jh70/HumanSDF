import torch
import torch.nn as nn

class DeepSDF(nn.Module):
    """Conditional SDF generation network"""
    def __init__(self, latent_size, hidden_dim=512):
        super(DeepSDF, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(3 + latent_size, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),  # Residual connection start
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),  # Residual connection end
            nn.Linear(hidden_dim, 1)  # Output layer, no activation
        ])
        self.activation = nn.ReLU()

    def forward(self, coords, latent):
        # Ensure latent has the right dimension
        latent = latent.unsqueeze(1).expand(-1, coords.shape[1], -1)
        x = torch.cat([coords, latent], dim=-1)
        residual = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply ReLU for all except last layer
                if i == 3:  # Save activation as residual at layer 4
                    x = self.activation(x)
                    residual = x
                elif i == 6:  # Add residual and activate at layer 7
                    x += residual
                    x = self.activation(x)
                else:
                    x = self.activation(x)
        return x
