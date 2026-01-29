# src/utils.py
import torch, random, numpy as np, os

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_optimizers(model, lr_backbone, lr_head, weight_decay):
    params_backbone, params_head = [], []
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if "visual" in n or "transformer" in n or "text" in n:
            params_backbone.append(p)
        else:
            params_head.append(p)
    opt = torch.optim.AdamW([
        {"params": params_backbone, "lr": lr_backbone},
        {"params": params_head, "lr": lr_head if params_head else lr_backbone}
    ], weight_decay=weight_decay)
    return opt

def build_scheduler(optimizer, total_steps, warmup_steps=1000, mode="cosine"):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        if mode == "cosine":
            return 0.5 * (1 + np.cos(np.pi * progress))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
