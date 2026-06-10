import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        _, c, _, _ = inputs.shape
        log_prob = F.log_softmax(inputs, dim=1)
        prob = torch.exp(log_prob)
        targets_one_hot = F.one_hot(targets, num_classes=c).permute(0, 3, 1, 2).float()
        pt = (prob * targets_one_hot).sum(dim=1)
        alpha_t = self.alpha.to(inputs.device)[targets] if self.alpha is not None else 1.0
        loss = -alpha_t * (1 - pt) ** self.gamma * (log_prob * targets_one_hot).sum(dim=1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(cfg):
    if cfg.get("loss_function", "focal_loss") == "focal_loss":
        return FocalLoss(alpha=cfg.get("focal_alpha", [1, 1, 1]), gamma=cfg.get("focal_gamma", 2))
    return nn.CrossEntropyLoss()
