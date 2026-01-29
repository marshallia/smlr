# src/prompts.py
import torch

def build_prompt_bank(labels, templates):
    bank = []
    for lbl in labels:
        for t in templates:
            bank.append(t.format(label=lbl.replace("_"," ")))
    return bank

def encode_prompt_bank(tokenize, text_encoder, prompts, device):
    with torch.no_grad():
        tokens = tokenize(prompts).to(device)
        text_feats = text_encoder(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats
