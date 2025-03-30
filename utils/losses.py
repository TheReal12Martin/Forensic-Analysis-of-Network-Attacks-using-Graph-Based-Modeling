import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class FocalLoss(nn.Module):
    def __init__(self, alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def weighted_loss(pred, target, weights=Config.CLASS_WEIGHTS, device='cpu'):
    weights = torch.tensor(weights, device=device)
    return F.cross_entropy(pred, target, weight=weights)