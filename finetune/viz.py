import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE 
import umap
from finetune.train import build_prompt_bank, encode_prompt_bank

def reduce_embeddings(X, method="umap", **kwargs):
    """
    Reduce high-dimensional embeddings to 2D for visualization.
    Supported methods: "umap", "tsne", "pca".
    """
    if method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42, **kwargs)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, **kwargs)
    elif method == "pca":
        reducer = PCA(n_components=2, **kwargs)
    else:
        raise ValueError(f"Unsupported reduction method: {method}")
    
    Z = reducer.fit_transform(X)
    return Z

def get_features(cfg, loader, device, model):
    imgs, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="Visualization data"):
            # Handle dict or tuple batches
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                y = batch["label"]
            else:
                images, y = batch
                images = images.to(device)

            # Encode images
            img_feats = model.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            if cfg["train_mode"] == "contrastive":
                # Use precomputed text features
                prompt_bank = cfg.get("prompt_bank")
                text_feats = cfg.get("text_feats")
                per_class = len(prompt_bank) // len(cfg["labels"])

                class_txt = []
                for i in range(len(cfg["labels"])):
                    feats = text_feats[i*per_class:(i+1)*per_class]
                    class_txt.append(feats.mean(dim=0))
                class_txt = torch.stack(class_txt, dim=0).to(device).to(img_feats.dtype)

                # Normalize
                class_txt = class_txt / class_txt.norm(dim=-1, keepdim=True)

                # Similarity logits
                logits = img_feats @ class_txt.T
                preds = logits.argmax(dim=-1).cpu().numpy()

            else:  # supervised mode
                logits = classifier(img_feats)
                preds = logits.argmax(dim=-1).cpu().numpy()

            imgs.append(img_feats.cpu().numpy())
            labels.append(preds)
    return imgs, labels

def visualize_embeddings(model, loader, classifier, cfg, device, mode="validation", tokenizer=None, class_map=None, method="umap", out_path="embeddings1.png"):
    """
    Visualize embeddings for contrastive or supervised training modes.
    - Contrastive: image features projected into text space, clustered by class prompts.
    - Supervised: image features passed through classifier head, clustered by predicted labels.
    """

    if classifier is not None:
        classifier.eval()
    
    if mode == "inference":
        labels = [k for k,_ in sorted(class_map.items(), key=lambda x:x[1])]
        cfg["labels"] = labels

        prompt_bank = build_prompt_bank(labels, [cfg["prompts"]["template"]] + cfg["prompts"]["augment_templates"])
        cfg["prompt_bank"] = prompt_bank
        text_feats = encode_prompt_bank(tokenizer, model.encode_text, prompt_bank, device) 
        cfg["text_feats"] = text_feats

    imgs, labels = get_features(cfg, loader, device, model)

    # Concatenate all batches
    X = np.concatenate(imgs, axis=0)
    y = np.concatenate(labels, axis=0)

    # Dimensionality reduction (UMAP, t-SNE, PCA, etc.)
    Z = reduce_embeddings(X, method=method)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    for c in np.unique(y):
        idx = y == c
        plt.scatter(Z[idx, 0], Z[idx, 1], s=6, label=f"class {c}", alpha=0.7)
    plt.legend(markerscale=3, frameon=False)
    plt.title("Image embedding clusters")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    # plt.show()

def vis_multi_modal(model, loader, classifier, cfg, device, mode="validation", tokenizer=None, class_map=None, method="umap", out_path="multi_modal_embb.png"):
    if classifier is not None:
        classifier.eval()
    
    if mode == "inference":
        labels = [k for k,_ in sorted(class_map.items(), key=lambda x:x[1])]
        cfg["labels"] = labels

        prompt_bank = build_prompt_bank(labels, [cfg["prompts"]["template"]] + cfg["prompts"]["augment_templates"])
        cfg["prompt_bank"] = prompt_bank
        text_feats = encode_prompt_bank(tokenizer, model.encode_text, prompt_bank, device) 
        cfg["text_feats"] = text_feats

    imgs, labels = get_features(cfg, loader, device, model)

    all_embeds = np.vstack([imgs, text_emb1])
    modalities = (["image"] * len(img_emb) + ["text1"] * len(text_emb1)+ ["text2"] * len(text_emb2))

    all_classes = data1['class_name'].to_list()+data1['class_name'].to_list()+data2['class_name'].to_list()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30) 
    tsne_result = tsne.fit_transform(all_embeds)
    plt.figure(figsize=(8,6)) 
    for modality, marker in zip(["image", "text1", "text2"], ["o", "s","^"]): 
        idx = [i for i, m in enumerate(modalities) if m == modality] 
        plt.scatter(tsne_result[idx,0], 
                    tsne_result[idx,1], 
                    c=[class_colors[c] for c in np.array(all_classes)[idx]], 
                    label=modality, alpha=0.7, marker=marker) 
    plt.legend() 
    plt.title("t-SNE visualization by modality and class") 
    plt.show()
