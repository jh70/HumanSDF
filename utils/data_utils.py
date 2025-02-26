import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat  # 修正导入方式
from scipy.spatial import cKDTree
import os

class MatDataProcessor:
    def __init__(self):
        """不再需要json文件，属性直接通过.mat文件计算"""
        pass

    def get_attributes(self, filename):
        """从.mat文件计算四个特征维度"""
        # 加载点云数据（假设数据存储在'pointcloud'字段中）
        mat_data = loadmat(filename)
        
        if 'points' not in mat_data:
            raise KeyError(f"pointcloud key not found in {filename}")
        
        points = mat_data['points']  # 假设点云数据是Nx3的数组
        
        # 计算四个属性
        num_points = points.shape[0]  # 点的数量
        x_len = np.ptp(points[:, 0])  # x轴极差（max-min）
        y_len = np.ptp(points[:, 1])  # y轴极差
        z_len = np.ptp(points[:, 2])  # z轴极差
        
        return np.array([num_points, x_len, y_len, z_len], dtype=np.float32)

    def calculate_sdf(self, points):
        """计算点云的SDF值"""
        # 创建KDTree用于快速最近邻查找
        tree = cKDTree(points)
        
        # 计算每个点到最近其他点的距离
        distances, _ = tree.query(points, k=2)  # k=2，因为每个点自己也会被找到，取第二近的点
        
        # 获取每个点的SDF值，距离值作为SDF值
        # 如果点云是表面点，距离为正值；如果是内部点，距离为负值
        sdf_values = distances[:, 1]  # 距离数组的第二列即为每个点的最小外部距离

        # 返回计算出的SDF值
        return sdf_values

    def get_sdf_values(self, filename):
        """从.mat文件提取点云并计算SDF值"""
        # 加载.mat文件
        mat_data = loadmat(filename)
        
        if 'points' not in mat_data:
            raise KeyError(f"pointcloud key not found in {filename}")
        
        points = mat_data['points']  # (N, 3)数组
        
        # 计算SDF值
        sdf_values = self.calculate_sdf(points)
        
        return np.array(sdf_values, dtype=np.float32)

    def normalize_points(self, points):
        """标准化点云"""
        if points.shape[1] != 3:
            raise ValueError("Points should have 3 dimensions (N, 3).")
        centroid = np.mean(points, axis=0)
        points -= centroid
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points /= max_dist
        return points

class PCNDataset(Dataset):
    def __init__(self, mat_dir, processor, num_samples=2048):
        self.mat_files = [
            os.path.join(mat_dir, f) 
            for f in os.listdir(mat_dir) 
            if f.endswith('.mat')
        ]
        self.processor = processor
        self.num_samples = num_samples

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        # 加载.mat文件
        mat_path = self.mat_files[idx]
        mat_data = loadmat(mat_path)
        points = mat_data['points']  # (N, 3)数组
        
        # 点云标准化（可选）
        if hasattr(self.processor, 'normalize_points'):
            points = self.processor.normalize_points(points)
        
        # 统一采样点数
        points = self._sample_points(points)
        
        # 获取属性（传入完整文件路径）
        attrs = self.processor.get_attributes(mat_path)
        
        # 获取SDF值
        sdf_values = self.processor.get_sdf_values(mat_path)
        
        # 转换为Tensor
        points = torch.FloatTensor(points)  # (num_samples, 3)
        attrs = torch.FloatTensor(attrs)    # (4,)
        sdf_values = torch.FloatTensor(sdf_values)  # (num_samples,)
        
        return points, attrs, sdf_values

    def _sample_points(self, points):
        """统一采样到固定点数"""
        num_points = points.shape[0]
        
        # 点不足时循环填充
        if num_points < self.num_samples:
            indices = np.random.choice(num_points, self.num_samples, replace=True)
        # 点过多时随机下采样
        else:
            indices = np.random.choice(num_points, self.num_samples, replace=False)
            
        return points[indices]
