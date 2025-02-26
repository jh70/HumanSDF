import torch
import torch.nn as nn

class PointCloudVAE(nn.Module):
    """Conditional VAE with separate encoders for point cloud and attributes."""
    def __init__(self, attr_dim=4, latent_dim=256, num_points=2048):
        super(PointCloudVAE, self).__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim
        
        # Encoder for Point Cloud (3 channels)
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # Output: [B, 256, 1]
        )
        
        # Encoder for Attributes (4 channels)
        self.attr_encoder = nn.Sequential(
            nn.Conv1d(attr_dim, 64, 1),  # Input channels = attr_dim
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # Output: [B, 256, 1]
        )
        
        # 修改潜变量生成层，分别为点云和属性生成独立的潜变量
        self.fc_mu_pc = nn.Linear(256, latent_dim)
        self.fc_logvar_pc = nn.Linear(256, latent_dim)
        self.fc_mu_attr = nn.Linear(256, latent_dim)
        self.fc_logvar_attr = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 256, 512),  # Latent + attribute features
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * num_points)  # Reconstruct point cloud
        )

    def encode(self, pointcloud, attributes):
        # 编码点云
        point_features = self.point_encoder(pointcloud.permute(0, 2, 1)).flatten(1)
        mu_pc = self.fc_mu_pc(point_features)
        logvar_pc = self.fc_logvar_pc(point_features)

        # 编码属性
        attr_features = self.attr_encoder(attributes.unsqueeze(-1)).flatten(1)
        mu_attr = self.fc_mu_attr(attr_features)
        logvar_attr = self.fc_logvar_attr(attr_features)

        return (mu_pc, logvar_pc), (mu_attr, logvar_attr)
    
    def decode(self, z):
        # 重构并调整形状为[B, num_points, 3]
        recon = self.decoder(z)
        return recon.view(-1, 3, self.num_points).permute(0, 2, 1)
    
    def reparameterize(self, mu_pc, logvar_pc, mu_attr, logvar_attr):
        z_pc = mu_pc + torch.randn_like(mu_pc) * torch.exp(0.5 * logvar_pc)
        z_attr = mu_attr + torch.randn_like(mu_attr) * torch.exp(0.5 * logvar_attr)
        return torch.cat([z_pc, z_attr], dim=1)

    def forward(self, pointcloud, attributes):
        (mu_pc, logvar_pc), (mu_attr, logvar_attr) = self.encode(pointcloud, attributes)
        z = self.reparameterize(mu_pc, logvar_pc, mu_attr, logvar_attr)
        recon = self.decode(z)
        return recon, (mu_pc, logvar_pc), (mu_attr, logvar_attr)
