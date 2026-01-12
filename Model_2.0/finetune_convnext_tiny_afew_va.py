# finetune_convnext_tiny_afew_va.py
"""
Finetune ConvNeXt-Tiny for Valence/Arousal regression on AFEW-VA frames.
- Frame-level training
- VA normalized to [-1, 1] by /10
- Head + last ConvNeXt stage unfrozen
"""

import os, random, math, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
import cv2


# ------------------ Utils ------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


# ------------------ Dataset ------------------

class AFEWVAFrameDataset(Dataset):
    def __init__(self, dataframe, train=False, gaussian_std=0.02, hflip=False):
        self.df = dataframe.reset_index(drop=True)
        self.train = train

        self.pre = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor()
        ])

        self.hflip = T.RandomHorizontalFlip(p=0.5) if (train and hflip) else None
        self.gaussian = AddGaussianNoise(std=gaussian_std) if train else None

        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = cv2.imread(row["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = self.pre(img)
        if self.hflip: x = self.hflip(x)
        if self.gaussian: x = self.gaussian(x)
        x = (x - self.mean) / self.std

        # Normalize VA → [-1, 1]
        y = torch.tensor([
            row["valence"] / 10.0,
            row["arousal"] / 10.0
        ], dtype=torch.float32)

        return x, y


# ------------------ Metrics ------------------

def compute_metrics(preds, targets):
    mse = nn.functional.mse_loss(preds, targets).item()
    mae = nn.functional.l1_loss(preds, targets).item()
    rmse = math.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}


# ------------------ Model ------------------

def build_model(weights_path, device, dropout_p=0.2):
    model = torchvision.models.convnext_tiny(weights=None)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm(in_features),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(dropout_p),
        nn.Linear(512, 2)
    )

    model.to(device)
    return model


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


# ------------------ Train / Val ------------------

def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn):
    model.train()
    preds, targs = [], []
    total_loss = 0.0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = loss_fn(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        preds.append(out.detach().cpu())
        targs.append(y.detach().cpu())

    preds = torch.cat(preds)
    targs = torch.cat(targs)

    return total_loss / len(loader.dataset), compute_metrics(preds, targs)


def validate_one_epoch(model, loader, device, loss_fn):
    model.eval()
    preds, targs = [], []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Val", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)

            total_loss += loss.item() * x.size(0)
            preds.append(out.cpu())
            targs.append(y.cpu())

    preds = torch.cat(preds)
    targs = torch.cat(targs)

    return total_loss / len(loader.dataset), compute_metrics(preds, targs), preds, targs


# ------------------ Main ------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv_path)

    # Use video-level splits from 'split' column
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    train_ds = AFEWVAFrameDataset(train_df, train=True,
                                  gaussian_std=args.gaussian_std,
                                  hflip=args.hflip)
    val_ds   = AFEWVAFrameDataset(val_df)
    test_ds  = AFEWVAFrameDataset(test_df)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.weights_path, device, args.dropout)

    set_requires_grad(model.features, False)
    set_requires_grad(model.features[-1], True)

    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": args.head_lr},
        {"params": model.features[-1].parameters(), "lr": args.backbone_lr}
    ], weight_decay=args.weight_decay)

    loss_fn = nn.SmoothL1Loss(beta=args.smoothl1_beta)
    scaler = torch.cuda.amp.GradScaler()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs-args.warmup_epochs, eta_min=args.min_lr
    )

    best_mse = float("inf")
    patience = 0

    for epoch in range(1, args.epochs+1):
        if epoch > args.warmup_epochs:
            scheduler.step(epoch - args.warmup_epochs)

        train_loss, train_m = train_one_epoch(
            model, train_loader, optimizer, scaler, device, loss_fn
        )
        val_loss, val_m, _, _ = validate_one_epoch(
            model, val_loader, device, loss_fn
        )

        print(f"Epoch {epoch:02d} | Train MSE {train_m['mse']:.4f} | Val MSE {val_m['mse']:.4f}")

        if val_m["mse"] < best_mse - args.min_delta:
            best_mse = val_m["mse"]
            patience = 0
            torch.save(model.state_dict(), args.checkpoint_path)
        else:
            patience += 1
            if patience >= args.patience:
                break

    model.load_state_dict(torch.load(args.checkpoint_path))
    _, test_m, preds, _ = validate_one_epoch(model, test_loader, device, loss_fn)

    preds = preds * 10.0  # rescale
    print("Final Test MSE (raw VA):", test_m["mse"] * 100)


# ------------------ Args ------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--weights_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--head_lr", type=float, default=5e-3)
    p.add_argument("--backbone_lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--gaussian_std", type=float, default=0.02)
    p.add_argument("--hflip", action="store_true")
    p.add_argument("--smoothl1_beta", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--min_delta", type=float, default=1e-5)
    p.add_argument("--checkpoint_path", type=str, default="best_afew_va.pt")
    args = p.parse_args()

    main(args)
