import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.pc_vae import PointCloudVAE
from utils.data_utils import MatDataProcessor, PCNDataset
from utils.losses import chamfer_loss, vae_kl_loss
import yaml

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 加载配置
    config = load_config('configs/train_config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化数据处理器和数据集
    processor = MatDataProcessor()  # 移除json_path参数
    dataset = PCNDataset(
        mat_dir=config['data']['mat_dir'],
        processor=processor,
        num_samples=int(config['model']['num_points'])
    )
    
    # 数据加载器改进
    dataloader = DataLoader(
        dataset,
        batch_size=int(config['train']['batch_size']),
        shuffle=True,
        num_workers=os.cpu_count()//2,
        pin_memory=True
    )

    # 模型初始化（根据配置参数）
    model = PointCloudVAE(
        num_points=int(config['model']['num_points']),
        latent_dim=int(config['model']['latent_dim']),
        attr_dim=4      # 4个属性维度
    ).to(device)
    
    # 优化器配置
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['train']['lr']),
        weight_decay=float(config['train']['weight_decay'])
    )

    # 训练循环改进
    best_loss = float('inf')
    for epoch in range(int(config['train']['epochs'])):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (points, attrs, sdf_values) in enumerate(dataloader):
            points = points.to(device)  # [B, N, 3]
            attrs = attrs.to(device)    # [B, 4]

            # 前向传播
            recon, (mu_pc, logvar_pc), (mu_attr, logvar_attr) = model(points, attrs)
            
            # 损失计算改进（添加权重参数）
            cd_loss = chamfer_loss(recon, points)
            kl_loss = vae_kl_loss(mu_pc, logvar_pc) * float(config['train']['beta_pc']) + \
                      vae_kl_loss(mu_attr, logvar_attr) * float(config['train']['beta_attr'])
            total_loss = cd_loss + kl_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # 统计损失
            epoch_loss += total_loss.item()
            
            # 每100个batch打印进度
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1} [{batch_idx * len(points)}/{len(dataset)}]'
                      f'\tLoss: {total_loss.item():.4f}')

        # 保存最佳模型
        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'checkpoints/best_model_pc_vae.pth')

        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    main()
