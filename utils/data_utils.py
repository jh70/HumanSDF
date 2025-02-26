import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from scipy.spatial import cKDTree
import os

class MatDataProcessor:
	def __init__(self, eps=1e-6):
		self.eps = eps  # Small epsilon to prevent zero distances

	def get_attributes(self, points):
		"""Calculate features from points array"""
		num_points = points.shape[0]
		x_len = np.ptp(points[:, 0])  # Peak-to-peak (max - min)
		y_len = np.ptp(points[:, 1])
		z_len = np.ptp(points[:, 2])
		return np.array([num_points, x_len, y_len, z_len], dtype=np.float32)

	def calculate_sdf(self, points):
		"""Calculate SDF with numerical safeguards"""
		# Input validation
		assert not np.isnan(points).any(), "NaN values in points"
		
		tree = cKDTree(points)
		distances, _ = tree.query(points, k=2)
		
		# Add epsilon to prevent zero distances
		sdf_values = distances[:, 1] + self.eps
		return sdf_values.astype(np.float32)

	def normalize_points(self, points):
		"""Normalize points to [-1, 1] range"""
		centroid = np.mean(points, axis=0)
		points -= centroid
		max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
		points /= max_dist
		return points

class PCNDataset(Dataset):
	def __init__(self, mat_dir, processor, num_samples=8192):
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
		# Load original points
		mat_path = self.mat_files[idx]
		mat_data = loadmat(mat_path)
		original_points = mat_data['points']
		
		# 1. Sample points first
		sampled_points = self.sample_points(original_points)
		
		# 2. Normalize sampled points
		if hasattr(self.processor, 'normalize_points'):
			sampled_points = self.processor.normalize_points(sampled_points)
		
		# 3. Compute attributes AFTER sampling/normalization
		attrs = self.processor.get_attributes(sampled_points)
		
		# 4. Compute SDF on processed points
		sdf_values = self.processor.calculate_sdf(sampled_points)
		
		# Convert to tensors
		points_tensor = torch.FloatTensor(sampled_points)
		attrs_tensor = torch.FloatTensor(attrs)
		sdf_tensor = torch.FloatTensor(sdf_values)

		# Final validation
		assert not torch.isnan(points_tensor).any(), "NaN in points tensor"
		assert not torch.isnan(sdf_tensor).any(), "NaN in SDF tensor"
		
		return points_tensor, attrs_tensor, sdf_tensor

	def sample_points(self, points):
		"""Uniform sampling with numerical safety"""
		num_points = points.shape[0]
		
		if num_points == 0:
			raise ValueError("Empty point cloud")
		
		# Random indices with proper seeding
		rng = np.random.default_rng(seed=int(torch.initial_seed()) % (1 << 32))
		
		if num_points < self.num_samples:
			indices = rng.choice(num_points, self.num_samples, replace=True)
		else:
			indices = rng.choice(num_points, self.num_samples, replace=False)
			
		return points[indices]
