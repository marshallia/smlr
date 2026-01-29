# src/losses.py
import torch
import torch.nn as nn

class ClipContrastiveLoss(nn.Module):
    def __init__(self, init_temp=0.07, learnable=True):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0/init_temp))) if learnable else None
        self.ce = nn.CrossEntropyLoss()

    def forward(self, img_feats, txt_feats, logit_scale_external=None):
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        scale = torch.exp(self.logit_scale) if self.logit_scale is not None else torch.exp(logit_scale_external)
        logits = scale * img_feats @ txt_feats.t()
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = (self.ce(logits, labels) + self.ce(logits.t(), labels)) * 0.5
        return loss, scale
