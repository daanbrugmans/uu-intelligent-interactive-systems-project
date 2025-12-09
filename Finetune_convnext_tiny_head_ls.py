# finetune_convnext_tiny_va_head_laststage.py
"""
Finetune ConvNeXt-Tiny for Valence/Arousal regression on EMOTIC crops.
- Train head + last stage from start for faster improvements.
- SmoothL1Loss (Huber) with beta.
- Cosine LR scheduler with linear warmup.
- Gaussian noise + horizontal flip.
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


class EmoticVA(Dataset):
    def __init__(self, dataframe, train=False, gaussian_std=0.05, horizontal_flip=False):
        self.df = dataframe.reset_index(drop=True)
        self.train = train
        self.horizontal_flip = horizontal_flip
        base_transforms = [
            T.ToPILImage(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor()
        ]
        self.pre_tensor = T.Compose(base_transforms)
        self.gaussian = AddGaussianNoise(std=gaussian_std) if train else None
        self.hflip = T.RandomHorizontalFlip(p=0.5) if (train and horizontal_flip) else None
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.load(row['img_path'])
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        tensor = self.pre_tensor(img)
        if self.hflip: tensor = self.hflip(tensor)
        if self.gaussian: tensor = self.gaussian(tensor)
        tensor = (tensor - self.mean) / self.std
        target = torch.tensor([(row['Valence'] - 1.0) / 9.0, (row['Arousal'] - 1.0) / 9.0], dtype=torch.float32)
        return tensor, target


def compute_metrics(preds, targets):
    mse = nn.functional.mse_loss(preds, targets).item()
    mae = nn.functional.l1_loss(preds, targets).item()
    rmse = math.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}


def build_model(weights_path, device, dropout_p=0.2):
    model = torchvision.models.convnext_tiny(weights=None)
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm(in_features, eps=1e-6),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(dropout_p),
        nn.Linear(512, 2)
    )
    model.to(device)
    return model


def set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad

def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn):
    model.train()
    running_loss = 0.0
    ns = 0
    preds, targs = [], []
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
        ns += imgs.size(0)
        preds.append(outputs.detach().cpu())
        targs.append(targets.detach().cpu())
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    preds, targs = torch.cat(preds), torch.cat(targs)
    return running_loss/ns, compute_metrics(preds, targs)

def validate_one_epoch(model, loader, device, loss_fn):
    model.eval()
    running_loss = 0.0
    ns = 0
    preds, targs = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item() * imgs.size(0)
            ns += imgs.size(0)
            preds.append(outputs.detach().cpu())
            targs.append(targets.detach().cpu())
    preds, targs = torch.cat(preds), torch.cat(targs)
    return running_loss/ns, compute_metrics(preds, targs), preds, targs


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    base = Path(args.base_path)
    annot_path = base / "annots_arrs"
    crop_path = base / "img_arrs"

    df_all = pd.concat([pd.read_csv(annot_path / f) for f in ["annot_arrs_train.csv","annot_arrs_val.csv","annot_arrs_test.csv"]])
    df_all = df_all[['Crop_name','Valence','Arousal']].copy()
    df_all['img_path'] = df_all['Crop_name'].apply(lambda x: str(crop_path / x))
    total = len(df_all)
    train_cnt = int(0.7*total); val_cnt=int(0.15*total)
    df_shuf = df_all.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    train_df, val_df, test_df = df_shuf[:train_cnt], df_shuf[train_cnt:train_cnt+val_cnt], df_shuf[train_cnt+val_cnt:]
    print(f"train:{len(train_df)}, val:{len(val_df)}, test:{len(test_df)}")

    train_ds = EmoticVA(train_df, train=True, gaussian_std=args.gaussian_std, horizontal_flip=args.hflip)
    val_ds   = EmoticVA(val_df, train=False)
    test_ds  = EmoticVA(test_df, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.weights_path, device, dropout_p=args.dropout)
    print("Model loaded and head replaced.")

    set_requires_grad(model.features, False)
    set_requires_grad(model.features[-1], True)
    print("Unfroze last stage + head")

    optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": args.head_lr},
        {"params": model.features[-1].parameters(), "lr": args.backbone_lr}
    ], weight_decay=args.weight_decay)

    loss_fn = nn.SmoothL1Loss(beta=args.smoothl1_beta)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-args.warmup_epochs, eta_min=args.min_lr)
    base_lrs = [g['lr'] for g in optimizer.param_groups]

    best_val_mse = float('inf'); pat_count = 0

    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_head_ls.pt"
    history = []

    for epoch in range(1, args.epochs+1):
        start = datetime.now()
        # Linear warmup
        if epoch <= args.warmup_epochs:
            warmup_scale = epoch / max(1, args.warmup_epochs)
            for i, g in enumerate(optimizer.param_groups): g['lr'] = base_lrs[i]*warmup_scale
        else: scheduler.step(epoch - args.warmup_epochs)

        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn)
        val_loss, val_metrics, val_preds, val_targs = validate_one_epoch(model, val_loader, device, loss_fn)
        epoch_time = (datetime.now() - start).total_seconds()

        print(f"Epoch {epoch:02d}/{args.epochs} - {epoch_time:.1f}s | Train MSE: {train_metrics['mse']:.4f} | Val MSE: {val_metrics['mse']:.4f}")
        print(f"LR groups: {[round(g['lr'],8) for g in optimizer.param_groups]}")

        # Early stopping
        if val_metrics['mse'] + args.min_delta < best_val_mse:
            best_val_mse = val_metrics['mse']; pat_count=0
            torch.save({"epoch":epoch,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scaler_state":scaler.state_dict()}, best_ckpt_path)
            print(f"  --> New best model saved (val_mse={best_val_mse:.6f})")
        else:
            pat_count += 1
            print(f"  No improvement. patience: {pat_count}/{args.patience}")
        if pat_count >= args.patience: break

        history.append({"epoch":epoch,"train_loss":train_loss,"train_mse":train_metrics["mse"],"val_loss":val_loss,"val_mse":val_metrics["mse"]})

    # Final test
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print("Loaded best checkpoint for final evaluation.")

    test_loss, test_metrics, test_preds_scaled, test_targs_scaled = validate_one_epoch(model, test_loader, device, loss_fn)
    def rescale(x): return x*9.0+1.0
    test_preds, test_targs = rescale(test_preds_scaled), rescale(test_targs_scaled)
    mse_orig = nn.functional.mse_loss(test_preds, test_targs).item()
    mae_orig = nn.functional.l1_loss(test_preds, test_targs).item()
    rmse_orig = math.sqrt(mse_orig)
    print(f"Final test results (1..10) | MSE:{mse_orig:.4f} | MAE:{mae_orig:.4f} | RMSE:{rmse_orig:.4f}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--head_lr", type=float, default=5e-3)
    parser.add_argument("--backbone_lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--gaussian_std", type=float, default=0.02)
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--smoothl1_beta", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=8) 
    parser.add_argument("--min_delta", type=float, default=1e-5, help="Minimum improvement to reset patience")
    args = parser.parse_args()
    main(args)

