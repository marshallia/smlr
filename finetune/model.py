# src/model.py
import torch
import torch.nn as nn
from torchvision import models

def load_clip(name="ViT-B-32", device="cuda", use_open_clip=True, num_classes=None, train_mode="contrastive"):
    if use_open_clip:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained="laion2b_s34b_b79k")
        tokenizer = open_clip.get_tokenizer(name)
    else:
        import clip
        model, preprocess = clip.load(name, device=device)
        tokenizer = clip.tokenize
    model.to(device)

    classifier = None 
    if train_mode == "supervised" and num_classes is not None: 
        # CLIP vision encoder outputs features before projection 
        feat_dim = model.visual.output_dim 
        classifier = nn.Linear(feat_dim, num_classes).to(device) 
    return model, preprocess, tokenizer, classifier


def set_trainable(model, freeze_vision=False, freeze_text=False, freeze_projection=False): 
    """ 
    Configure which parts of CLIP are trainable.
    - freeze_vision: if True, vision encoder is frozen
    - freeze_text: if True, text encoder is frozen
    - freeze_projection: if True, projection head is frozen 
    """
    # Vision encoder 
    if hasattr(model, "visual"): 
        for p in model.visual.parameters(): 
            p.requires_grad = not freeze_vision 
            
    # Text encoder (transformer or text module depending on implementation) 
    if hasattr(model, "transformer"): 
        for p in model.transformer.parameters(): 
            p.requires_grad = not freeze_text 
    elif hasattr(model, "text"): 
        for p in model.text.parameters(): 
                p.requires_grad = not freeze_text 
                
    # Projection head (varies by implementation) 
    if hasattr(model.visual, "proj"): 
        proj = model.visual.proj 
        if isinstance(proj, nn.Parameter): 
            proj.requires_grad = not freeze_projection 
        else: 
            for p in proj.parameters():
                p.requires_grad = not freeze_projection
        
    # Ensure logit scale is present and trainable 
    if not hasattr(model, "logit_scale"): 
        model.logit_scale = nn.Parameter( 
            torch.ones([]) * torch.log(torch.tensor(1/0.07)) 
            ) 
    
    return model
