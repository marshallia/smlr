# src/train.py
import os, torch
import shutil
from tqdm import tqdm
from torch import amp
from torch.amp import GradScaler
from .model import load_clip, set_trainable
from .dataset import make_loaders
from .prompt import build_prompt_bank, encode_prompt_bank
from .losses import ClipContrastiveLoss
from .utils import set_seed, build_optimizers, build_scheduler

def train(cfg):
    set_seed(cfg["seed"])
    device = cfg["device"]

    model, preprocess, tokenize, classifier = load_clip(cfg["model"]["name"], device, cfg["model"]["use_open_clip"],num_classes=cfg.get("num_classes"), train_mode=cfg["train_mode"])
    """
    Freeze text encoder, train vision + projection only:
    model = set_trainable(model, freeze_vision=False, freeze_text=True, freeze_projection=False)
    Train everything (full fine‑tune):
    model = set_trainable(model, freeze_vision=False, freeze_text=False, freeze_projection=False)
    Freeze both encoders, train only projection:
    model = set_trainable(model, freeze_vision=True, freeze_text=True, freeze_projection=False)
    """
    model = set_trainable(model, cfg["model"]["freeze_vision"], cfg["model"]["freeze_text"],cfg["model"]["freeze_projection"])
    train_loader, val_loader, class_map = make_loaders(cfg['data']['csv_mode'],
        cfg["data"]["train_root"], cfg["data"]["val_root"], preprocess,
        cfg["train"]["batch_size"], cfg["train"]["num_workers"], 
        cfg["data"]["train_csv"], cfg["data"]["val_csv"]
    )

    labels = [k for k,_ in sorted(class_map.items(), key=lambda x:x[1])]
    cfg["labels"] = labels
    # Contrastive mode setup 
    if cfg["train_mode"] == "contrastive":
        prompt_bank = build_prompt_bank(labels, [cfg["prompts"]["template"]] + cfg["prompts"]["augment_templates"])
        cfg["prompt_bank"] = prompt_bank
        text_feats = encode_prompt_bank(tokenize, model.encode_text, prompt_bank, device) 
        cfg["text_feats"] = text_feats

        loss_fn = ClipContrastiveLoss(cfg["loss"]["temperature_init"], cfg["loss"]["temperature_learnable"]) 
    else: 
        # Supervised mode 
        loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = build_optimizers(model, cfg["train"]["lr_backbone"], cfg["train"]["lr_head"], cfg["train"]["weight_decay"])
    total_steps = cfg["train"]["epochs"] * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps, cfg["train"]["warmup_steps"], cfg["train"]["scheduler"])
    scaler = GradScaler(enabled=cfg["train"]["amp"])

    os.makedirs(cfg["logging"]["out_dir"], exist_ok=True)
    step = 0
    best_acc = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for images, y in tqdm(train_loader, total= len(train_loader), desc="Training data"):
            images, y = images.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            # choose a prompt per sample (stochastic prompt augmentation)
            # map class index to a slice of prompt_bank
            # simple approach: average text features per class across templates
            with amp.autocast("cuda", enabled=cfg["train"]["amp"]): 
                img_feats = model.encode_image(images)
                if cfg["train_mode"] == "contrastive": # average text features per class
                
                    with torch.no_grad():
                        # build class-wise averaged text features
                        class_txt = []
                        per_class = len(prompt_bank) // len(labels)
                        for i in range(len(labels)):
                            feats = text_feats[i*per_class:(i+1)*per_class]
                            class_txt.append(feats.mean(dim=0))
                        class_txt = torch.stack(class_txt, dim=0)  # [C, D]
                        txt_batch = class_txt[y].to(img_feats.dtype)  # [B, D]
                    loss, scale = loss_fn(img_feats, txt_batch)
                else: 
                    # supervised mode
                    logits = classifier(img_feats)
                    loss = loss_fn(logits, y)
                    scale = None

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

        # periodic eval + checkpoint
        # if (epoch + 1) % cfg["logging"]["eval_every"] == 0:
        val_loss, val_acc = evaluate(model, val_loader, classifier, cfg, device)
        if cfg["train_mode"] == "contrastive": 
            print(f"Epoch {epoch+1}: train_loss={loss.item():.4f} " 
                  f"val_loss={val_loss if val_loss is not None else 0:.4f} " 
                  f"val_acc={val_acc:.4f} " f"logit_scale={scale.item():.3f}") 
        else: 
            # supervised mode 
            print(f"Epoch {epoch+1}: train_loss={loss.item():.4f} " 
                  f"val_loss={val_loss:.4f} " f"val_acc={val_acc:.4f}")
        if val_acc > best_acc: 
            best_acc = val_acc 
            torch.save({ 
                "epoch": epoch+1, 
                "model": model.state_dict(), 
                "optimizer": optimizer.state_dict(), 
                "scheduler": scheduler.state_dict(), 
                "best_acc": best_acc 
                }, os.path.join(cfg["logging"]["out_dir"], "best_model.pt")) 
            print(f"New best accuracy {best_acc:.4f}, model saved.")
    shutil.copyfile(os.path.join("configs/fine_tune_config.yaml"),
                    os.path.join(cfg["logging"]["out_dir"], "fine_tune_config.yaml"))

def evaluate(model, val_loader, classifier, cfg, device, mode="validation", tokenizer=None, class_map=None):
    """
    Evaluate CLIP model on validation set.
    Returns (val_loss, val_acc) for consistency.
    - Contrastive mode: val_loss=None, val_acc computed via text prompts.
    - Supervised mode: val_loss=avg CE loss, val_acc computed via classifier.
    """
    model.eval()
    if classifier is not None:
        classifier.eval()

    correct, total = 0, 0
    total_loss = 0.0

    if mode == "inference":
        labels = [k for k,_ in sorted(class_map.items(), key=lambda x:x[1])]
        cfg["labels"] = labels

        prompt_bank = build_prompt_bank(labels, [cfg["prompts"]["template"]] + cfg["prompts"]["augment_templates"])
        cfg["prompt_bank"] = prompt_bank
        text_feats = encode_prompt_bank(tokenizer, model.encode_text, prompt_bank, device) 
        cfg["text_feats"] = text_feats

    with torch.no_grad():
        for images, y in val_loader:
            images, y = images.to(device), y.to(device)
            img_feats = model.encode_image(images)

            if cfg["train_mode"] == "contrastive":
                # Precomputed text features stored in cfg
                prompt_bank = cfg.get("prompt_bank")
                text_feats = cfg.get("text_feats")
                per_class = len(prompt_bank) // len(cfg["labels"])

                class_txt = []
                for i in range(len(cfg["labels"])):
                    feats = text_feats[i*per_class:(i+1)*per_class]
                    class_txt.append(feats.mean(dim=0))
                class_txt = torch.stack(class_txt, dim=0).to(device).to(img_feats.dtype)

                # Normalize
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                class_txt = class_txt / class_txt.norm(dim=-1, keepdim=True)

                logits = img_feats @ class_txt.T
                preds = logits.argmax(dim=-1)

                correct += (preds == y).sum().item()
                total += y.size(0)

            else:  # supervised mode
                logits = classifier(img_feats)
                preds = logits.argmax(dim=-1)

                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits, y)
                total_loss += loss.item()

                correct += (preds == y).sum().item()
                total += y.size(0)
    print(correct, total)
    val_acc = correct / total if total > 0 else 0.0
    val_loss = total_loss / max(1, len(val_loader)) if cfg["train_mode"] == "supervised" else None

    return val_loss, val_acc
