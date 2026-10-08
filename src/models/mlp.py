import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLP_coarse_fine(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.2, embed_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_coarse  = nn.Linear(hidden, embed_dim)
        self.head_fine    = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        h = self.backbone(x)
        zc = F.normalize(self.head_coarse(h), p=2, dim=1)
        zf = F.normalize(self.head_fine(h),   p=2, dim=1)
        return zc, zf


class MLP_fine(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.2, embed_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_fine   = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        h = self.backbone(x)
        zf = F.normalize(self.head_fine(h),   p=2, dim=1)
        return zf

class MLP_dual_backbone(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.2, embed_dim=64):
        super().__init__()
        # coarse branch
        self.backbone_c = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_coarse = nn.Linear(hidden, embed_dim)

        # fine branch
        self.backbone_f = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_fine = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        hc = self.backbone_c(x)
        hf = self.backbone_f(x)
        zc = F.normalize(self.head_coarse(hc), p=2, dim=1)
        zf = F.normalize(self.head_fine(hf),   p=2, dim=1)
        return zc, zf