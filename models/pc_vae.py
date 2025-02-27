import torch
import torch.nn as nn
import torch.nn.init as init

class PointCloudVAE(nn.Module):
    def __init__(self, attr_dim=4, latent_dim=256, num_points=8192):
        super(PointCloudVAE, self).__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim
        
        # 点云编码器（添加BatchNorm）
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # 属性编码器（添加BatchNorm）
        self.attr_encoder = nn.Sequential(
            nn.Conv1d(attr_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # 潜变量生成层
        self.fc_mu_pc = nn.Linear(256, latent_dim)
        self.fc_logvar_pc = nn.Linear(256, latent_dim)
        self.fc_mu_attr = nn.Linear(256, latent_dim)
        self.fc_logvar_attr = nn.Linear(256, latent_dim)
        
        # 初始化logvar层的权重和偏置
        for layer in [self.fc_logvar_pc, self.fc_logvar_attr]:
            init.normal_(layer.weight, mean=0.0, std=0.01)
            init.constant_(layer.bias, 0.0)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),  # 拼接后的维度是latent_dim*2
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * num_points)
        )

    def encode(self, pointcloud, attributes):
        # 确保输入数据已归一化
        point_features = self.point_encoder(pointcloud.permute(0, 2, 1)).flatten(1)
        mu_pc = self.fc_mu_pc(point_features)
        logvar_pc = self.fc_logvar_pc(point_features)

        # 处理属性输入：[B, attr_dim] -> [B, attr_dim, 1]
        attr_features = self.attr_encoder(attributes.unsqueeze(-1)).flatten(1)
        mu_attr = self.fc_mu_attr(attr_features)
        logvar_attr = self.fc_logvar_attr(attr_features)

        return (mu_pc, logvar_pc), (mu_attr, logvar_attr)
    
    def decode(self, z):
        recon = self.decoder(z)
        return recon.view(-1, 3, self.num_points).permute(0, 2, 1)
    
    def reparameterize(self, mu_pc, logvar_pc, mu_attr, logvar_attr):
        std_pc = torch.exp(0.5 * logvar_pc)
        eps_pc = torch.randn_like(std_pc)
        z_pc = mu_pc + eps_pc * std_pc

        std_attr = torch.exp(0.5 * logvar_attr)
        eps_attr = torch.randn_like(std_attr)
        z_attr = mu_attr + eps_attr * std_attr

        return torch.cat([z_pc, z_attr], dim=1)

    def forward(self, pointcloud, attributes):
        (mu_pc, logvar_pc), (mu_attr, logvar_attr) = self.encode(pointcloud, attributes)
        z = self.reparameterize(mu_pc, logvar_pc, mu_attr, logvar_attr)
        recon = self.decode(z)
        return recon, (mu_pc, logvar_pc), (mu_attr, logvar_attr)
