import yaml
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.deepsdf import DeepSDF
from models.pc_vae import PointCloudVAE
from utils.data_utils import PCNDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import os

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 加载配置文件
    config = load_config('configs/train_config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化预训练的VAE模型
    vae = PointCloudVAE(
        num_points=int(config['model']['num_points']),
        latent_dim=int(config['model']['latent_dim']),
        attr_dim=4      # 4个属性维度
    ).to(device)
    vae.load_state_dict(torch.load(f'checkpoints/best_model_pc_vae.pth'))
    vae.eval()  # 冻结 VAE 的权重

    # 初始化 DeepSDF 模型
    deepsdf = DeepSDF(latent_size=512).to(device)  # latent_dim * 2

    # 数据加载
    dataset = PCNDataset(
        mat_dir=config['data']['mat_dir'],
        processor=processor,
        num_samples=int(config['model']['num_points'])
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config['train']['batch_size']),
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        pin_memory=True
    )

    # 优化器
    optimizer = optim.Adam(deepsdf.parameters(), lr=float(config['train']['lr']), weight_decay=float(config['train']['weight_decay']))

    # 训练 DeepSDF
    best_loss = float('inf')
    for epoch in range(int(config['train']['epochs'])):
        deepsdf.train()
        epoch_loss = 0.0
        for batch_idx, (coords, attrs, sdf_values) in enumerate(dataloader):
            coords = coords.to(device)  # 坐标数据 [B, N, 3]
            sdf_values = sdf_values.to(device)  # 对应的 SDF 值 [B, N]

            # 使用 VAE 编码器获取潜在向量
            with torch.no_grad():  # 冻结 VAE 编码器
                _, (mu_pc, _), (mu_attr, _) = vae.encode(coords, None)  # 假设不需要属性数据
                z = torch.cat([mu_pc, mu_attr], dim=1)  # 合并潜在向量

            # 预测 SDF 值
            pred_sdf = deepsdf(coords, z)

            # 计算损失（均方误差）
            loss = torch.nn.functional.mse_loss(pred_sdf, sdf_values)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失
            epoch_loss += loss.item()

            # 每100个batch打印进度
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1} [{batch_idx * len(coords)}/{len(dataset)}] Loss: {loss.item():.4f}')

        # 保存最佳模型
        avg_loss = epoch_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(deepsdf.state_dict(), 'checkpoints/best_model_deepsdf.pth')

        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    main()
