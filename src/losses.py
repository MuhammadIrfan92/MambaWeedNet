import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import math



def kl_divergence_feature_maps(f1, f2, reduction="mean"):
    log_p = F.log_softmax(f1, dim=1)
    log_q = F.log_softmax(f2, dim=1)
    p = log_p.exp()               # or torch.softmax(f1, dim=1)

    kl = p * (log_p - log_q)      # (B, C, H, W)
    kl = kl.sum(dim=1)            # sum over channels

    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    else:
        return kl


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


class LLMBlockOnly(nn.Module):
    def __init__(self, channel, layer,  h=16, w=16): # remove thse h and w later
  
        super(LLMBlockOnly, self).__init__()

        # Encoder
        self.h_t = 32
        self.w_t = 64

        enc1 = math.isqrt(channel)
        assert enc1 * enc1 == channel, f"{channel} is not a perfect square"

        self.sin, self.cos = self.generate_2d_sin_cos_positional_encoding(enc1, enc1)
        self.llm = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").model.layers[layer].to('cuda')
        
        for param in self.llm.parameters():
            param.requires_grad = False

    
    def generate_2d_sin_cos_positional_encoding(self, height, width):
        token = height * width
        # 2D coordinates and Calculate sin and cos encodings
        sin_y = torch.sin(torch.arange(token).view(1, token, 1))  # Shape (H*W, 1)
        cos_x = torch.cos(torch.arange(token).view(1, token, 1))  # Shape (H*W, 1)
        return sin_y.cuda(), cos_x.cuda()


    def resize_feature_map_to_2048_old(self, x, h_out=32, w_out=64):
        """
        x: Tensor of shape (B, C, H, W)
        returns: Tensor of shape (B, C, llm_block_input_dim)
        """

        B, C, H, W = x.shape
        assert H == 36 and W == 36, "Expected input spatial size (36, 36)"
        # 1. Spatial resize
        x = F.interpolate(
            x,
            size=(h_out, w_out),   
            mode="bilinear",
            align_corners=False
        )  
        # 2. Flatten spatial dimensions into token dimension
        x = x.flatten(2)  # (B, C, 2048)

        return x
    
    def resize_feature_map_to_2048(self,x):
        """
        x: Tensor of shape (B, C, H, W), arbitrary H and W
        returns: Tensor of shape (B, C, llm_block_input_dim)
        """
        # 1. Spatial resampling to fixed grid
        x = F.interpolate(
            x,
            size=(self.h_t, self.w_t),      
            mode="bilinear",
            align_corners=False
        ) 
        # 2. Flatten spatial dimensions into token dimension
        return x.flatten(2)  # (B, C, 2048)
    
    def restore_feature_map_from_2048(self, x, h_out=36, w_out=36):
        """
        x: Tensor of shape (B, C, llm_block_input_dim)
        returns: Tensor of shape (B, C, H, W)
        """
        B, C, N = x.shape
        # 1. Restore 2D spatial structure
        x = x.view(B, C, self.h_t, self.w_t) 
        # 2. Resize back to original spatial resolution
        x = F.interpolate(
            x,
            size=(h_out, w_out),     
            mode="bilinear",
            align_corners=False
        )  
        return x

    def forward(self, x):
        B, C, H, W = x.shape # (B, C, H, W)
        x = self.resize_feature_map_to_2048(x)
        x_llm = self.llm(hidden_states=x, position_embeddings=[self.sin, self.cos])[0]
        x = self.restore_feature_map_from_2048(x_llm, H, W).reshape(B, C, H, W)
        return x

def aux_weight_cosine(epoch: int, max_epochs: int, lambda0: float = 1.0, lambda_min: float = 0.0):
    """Cosine-decay weight for auxiliary loss."""
    t = min(max(epoch, 0), max_epochs)
    return lambda_min + 0.5 * (lambda0 - lambda_min) * (1 + math.cos(math.pi * t / max_epochs))

def build_loss(cfg):
    if cfg.get("loss_function", "focal_loss") == "focal_loss":
        return FocalLoss(alpha=cfg.get("focal_alpha", [1, 1, 1]), gamma=cfg.get("focal_gamma", 2))
    return nn.CrossEntropyLoss()
