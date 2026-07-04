import torch
import torch.nn as nn
import torch.nn.functional as F

class JointAttentionDistillLoss(nn.Module):
    def __init__(self, weight_spatial=1.0, weight_channel=1.0, use_cosine=True):
        super().__init__()
        self.weight_spatial = weight_spatial
        self.weight_channel = weight_channel
        self.use_cosine = use_cosine
        self.mse = nn.MSELoss()

    def forward(self, feats_new, feats_old):
        loss_all = []
        for f_new, f_old in zip(feats_new, feats_old):
            # 空间注意力
            att_new_s = f_new.abs().mean(dim=1)   # (B,H,W)
            att_old_s = f_old.abs().mean(dim=1)
            att_new_s = F.normalize(att_new_s.flatten(1), p=2, dim=1, eps=1e-6)
            att_old_s = F.normalize(att_old_s.flatten(1), p=2, dim=1, eps=1e-6)

            if self.use_cosine:
                loss_spatial = 1 - F.cosine_similarity(att_new_s, att_old_s.detach(), dim=1).mean()
            else:
                loss_spatial = self.mse(att_new_s, att_old_s.detach())

            # 通道注意力
            att_new_c = f_new.abs().mean(dim=[2,3])  # (B,C)
            att_old_c = f_old.abs().mean(dim=[2,3])
            att_new_c = F.normalize(att_new_c, p=2, dim=1, eps=1e-6)
            att_old_c = F.normalize(att_old_c, p=2, dim=1, eps=1e-6)

            if self.use_cosine:
                loss_channel = 1 - F.cosine_similarity(att_new_c, att_old_c.detach(), dim=1).mean()
            else:
                loss_channel = self.mse(att_new_c, att_old_c.detach())

            loss_all.append(self.weight_spatial * loss_spatial +
                            self.weight_channel * loss_channel)

        return sum(loss_all)
