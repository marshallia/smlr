# src/datasets.py
import os, csv, random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np

class ImageFolderWithLabels(Dataset):
    def __init__(self, csv_mode, root, preprocess, csv_file=None, class_map=None):
        self.root = root
        self.preprocess = preprocess
        self.paths, self.labels = [], []
        if csv_mode:
            data = pd.read_csv(os.path.join(self.root, csv_file), sep=";")
            self.labels = data["classname"].to_list()
            self.paths = [x.strip() for x in data["filepath"].to_list()]
            classes = sorted([d for d in np.unique(self.labels)])
            self.class_to_idx = {c:i for i,c in enumerate(classes)} if class_map is None else class_map
            
        else:
            classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
            self.class_to_idx = {c:i for i,c in enumerate(classes)} if class_map is None else class_map
            for c in self.class_to_idx:
                cdir = os.path.join(root, c)
                for f in os.listdir(cdir):
                    if f.lower().endswith((".jpg",".png",".jpeg",".bmp",".webp")):
                        self.paths.append(os.path.join(cdir, f))
                        self.labels.append(self.class_to_idx[c])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        num_label = self.class_to_idx[self.labels[idx]]
        return self.preprocess(img), num_label

def make_loaders(csv_mode, train_root, val_root, preprocess, batch_size, num_workers, train=None,validation=None):
    train_ds = ImageFolderWithLabels(csv_mode,train_root, preprocess, csv_file=train)
    val_ds   = ImageFolderWithLabels(csv_mode,val_root, preprocess, csv_file=validation,class_map=train_ds.class_to_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.class_to_idx
