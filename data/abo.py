import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as T


def get_transforms(size=64):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def build_abo_loaders(cfg):
    root = os.path.join(cfg.data_dir, "abo_small")
    full = datasets.ImageFolder(root=root, transform=get_transforms(cfg.img_size))
    n = len(full)
    n_train = int(n * cfg.train_split)
    n_val = n - n_train
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
