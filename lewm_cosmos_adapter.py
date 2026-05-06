import torch
import torch.nn as nn
import torch.nn.functional as F


class CosmosToLeWMAdapter(nn.Module):
    """
    把 Cosmos Reason2 的 2048 维特征映射到 LeWM 的 192 维 latent。

    输入:
        x: [B, T, 2048]

    输出:
        y: [B, T, 192]
    """

    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=192):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def alignment_loss(pred, target, cos_weight=0.1):
    """
    pred:
        Adapter 输出，shape [B, T, 192]

    target:
        LeWM encoder 输出，shape [B, T, 192]
    """

    assert pred.shape == target.shape, f"pred {pred.shape}, target {target.shape}"

    mse_loss = F.mse_loss(pred, target)

    cos_loss = 1.0 - F.cosine_similarity(
        pred,
        target,
        dim=-1,
    ).mean()

    total_loss = mse_loss + cos_weight * cos_loss

    return total_loss, {
        "mse_loss": mse_loss.item(),
        "cos_loss": cos_loss.item(),
        "total_loss": total_loss.item(),
    }