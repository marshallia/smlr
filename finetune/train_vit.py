import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from utils import set_seed
from PIL import Image
import os
import json
import pandas as pd
from tqdm import tqdm
import yaml

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None, class_to_idx = None,):
        df = pd.read_csv(csv_file, sep=";")
        self.image_paths = df['filepath'].to_list()
        self.annotations = df['classname'].to_list()
        self.transform = transform

        if class_to_idx is None:
            classes = sorted(set(self.annotations))
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        
        self.labels = [self.class_to_idx[i] for i in self.annotations]
        

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        lbl = self.labels[index]
        return image, lbl
    
def train_val(cfg, train_csv, eval_csv, test_csv):
    device = cfg['device']
    set_seed(cfg["seed"])
    num_classes =  cfg['num_classes']

    transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

    train_ds = ImageDataset(train_csv, transform)
    val_ds = ImageDataset(eval_csv, transform)
    test_ds = ImageDataset(test_csv, transform)

    train_loader = DataLoader(train_ds, batch_size = cfg['train']['batch_size'], shuffle = True, num_workers = cfg['train']['num_workers'], pin_memory = True)
    val_loader = DataLoader(val_ds, batch_size = cfg['train']['batch_size'], shuffle = False, num_workers = cfg['train']['num_workers'], pin_memory = True)
    test_loader = DataLoader(test_ds, batch_size = cfg['train']['batch_size'], shuffle = False, num_workers = cfg['train']['num_workers'], pin_memory = True)

    model = models.vit_b_32(weights = models.ViT_B_32_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr =cfg['train']['lr_backbone'])
    criterion = nn.CrossEntropyLoss()

    warmup_epochs = 3
    def warmup(epoch):
        return float(epoch+1) / warmup_epochs if epoch < warmup_epochs else 1.0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)        
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train']['epochs']-warmup_epochs)

    train_losses = []
    val_losses = []
    val_accuracies = []
    best_acc = 0.0
    for epoch in range(cfg['train']['epochs']):
        model.train()
        total_loss  = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        avg_train_loss = total_loss/len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        correct, total_val, total_val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                correct+= (preds==labels).sum().item()
                total_val_loss +=loss.item()
                total_val+=labels.size(0)
        avg_val_loss = total_val_loss/len(val_loader)
        val_accuracy = correct/total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            lr_scheduler.step()
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | " f"Train Loss: {avg_train_loss:.4f} | " f"Val Loss: {avg_val_loss:.4f} | " f"Val Acc: {val_accuracy:.2f}")
        
        if avg_val_loss > best_acc: 
            best_acc = avg_val_loss 
            torch.save({ 
                "epoch": epoch+1, 
                "model": model.state_dict(), 
                "optimizer": optimizer.state_dict(), 
                "scheduler": lr_scheduler.state_dict(), 
                "best_acc": best_acc 
                }, os.path.join("runs","test_vit", "best_model.pt")) 
            print(f"New best accuracy {best_acc:.4f}, model saved.")

    torch.save({ 
        "epoch": epoch+1, 
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(), 
        "scheduler": lr_scheduler.state_dict(), 
        "best_acc": best_acc 
        }, os.path.join("runs","test_vit", "last_epoch_model.pt")) 
    print(f"New best accuracy {best_acc:.4f}, model saved.")


    print(f"Evaluation on Target dataset")  
    model.eval()
    correct, total_val, total_val_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            correct+= (preds==labels).sum().item()
            total_val_loss +=loss.item()
            total_val+=labels.size(0)
    avg_val_loss = total_val_loss/len(val_loader)
    val_accuracy = correct/total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Final Evaluation" f"Train Loss: {avg_train_loss:.4f} | " f"Val Loss: {avg_val_loss:.4f} | " f"Val Acc: {val_accuracy:.2f}")

    

if __name__=='__main__':
    with open("runs/supervised/fine_tune_config.yaml") as f:
        cfg = yaml.safe_load(f)
    train_val(cfg, "data/train/source_train.csv","data/val/source_val.csv", "data/val/target_val.csv")