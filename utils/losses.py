import torch

def chamfer_loss(pred, target):
    """Chamfer Distance Loss"""
    dist = torch.cdist(pred, target)
    min_dist_pred_to_target = torch.min(dist, dim=2)[0]
    min_dist_target_to_pred = torch.min(dist, dim=1)[0]
    return (min_dist_pred_to_target.mean(dim=1) + min_dist_target_to_pred.mean(dim=1)).mean()

def vae_kl_loss(mu, logvar):
    """VAE KL Divergence Loss"""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
