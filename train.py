import os
import csv
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn

from data_contract import get_dataloaders, NUM_CLASSES, IGNORE_INDEX, IMAGE_SIZE, CLASS_NAMES
from models import build_model


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_iou_from_confmat(confmat):
    intersection = np.diag(confmat)
    union = confmat.sum(axis=1) + confmat.sum(axis=0) - intersection
    valid = union > 0
    iou = np.where(valid, intersection / np.maximum(union, 1), np.nan)
    return iou, float(np.nanmean(iou))


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=NUM_CLASSES):
    model.eval()
    total_loss = 0.0
    confmat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = criterion(logits, masks)
        total_loss += loss.item() * imgs.size(0)

        preds = logits.argmax(dim=1).cpu().numpy()
        targets = masks.cpu().numpy()
        valid = targets != IGNORE_INDEX

        t_valid = targets[valid]
        p_valid = preds[valid]
        hist = np.bincount(num_classes * t_valid + p_valid, minlength=num_classes ** 2)
        confmat += hist.reshape(num_classes, num_classes)

    per_class_iou, miou = compute_iou_from_confmat(confmat)
    return total_loss / len(loader.dataset), per_class_iou, miou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="baseline", choices=["baseline", "skip_fusion"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("/kaggle/working/logs", exist_ok=True)
    os.makedirs("/kaggle/working/checkpoints", exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, _ = get_dataloaders(batch_size=args.batch_size, seed=args.seed)
    model = build_model(arch=args.arch, num_classes=NUM_CLASSES).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config_record = {
        "dataset": "CelebAMask-HQ (19-class facial semantic segmentation)",
        "arch": args.arch,
        "num_classes": NUM_CLASSES,
        "seed": args.seed,
        "resolution": list(IMAGE_SIZE),
        "augmentation": "RandomHorizontalFlip(p=0.5, with L/R facial part swap) + RandomResizedCrop(scale=(0.80,1.0), ratio=(0.95,1.05))",
        "optimizer": f"AdamW(lr={args.lr}, weight_decay={args.weight_decay})",
        "scheduler": f"CosineAnnealingLR(T_max={args.epochs})",
        "loss": f"CrossEntropyLoss(ignore_index={IGNORE_INDEX})",
        "stopping_rule": f"Fixed budget of {args.epochs} epochs; checkpoint selected by highest validation mIoU",
        "batch_size": args.batch_size,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    with open(f"/kaggle/working/logs/{args.arch}_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    print(f"=== Training {args.arch} | Classes: {NUM_CLASSES} | Params: {total_params:,} | Seed: {args.seed} ===")

    log_path = f"/kaggle/working/logs/{args.arch}_train.csv"
    class_cols = [f"iou_{CLASS_NAMES[c]}" for c in range(NUM_CLASSES)]
    best_miou = -1.0
    best_epoch = -1

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "val_loss", "val_miou", "epoch_sec"] + class_cols)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            running_loss = 0.0

            for imgs, masks, _ in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)

            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            train_loss = running_loss / len(train_loader.dataset)
            val_loss, val_ious, val_miou = validate(model, val_loader, criterion, device)
            elapsed = time.time() - t0

            writer.writerow(
                [epoch, f"{current_lr:.6f}", f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_miou:.4f}", f"{elapsed:.1f}"]
                + [f"{v:.4f}" if not np.isnan(v) else "nan" for v in val_ious]
            )
            f.flush()

            marker = ""
            if val_miou > best_miou:
                best_miou = val_miou
                best_epoch = epoch
                torch.save(model.state_dict(), f"/kaggle/working/checkpoints/{args.arch}_best.pt")
                marker = "  <-- best"

            # Print overall mIoU + representative large (skin, hair) and fine (l_eye, u_lip) classes
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val mIoU: {val_miou:.4f} (Skin: {val_ious[1]:.3f}, Hair: {val_ious[17]:.3f}, "
                f"L-Eye: {val_ious[4]:.3f}, U-Lip: {val_ious[12]:.3f}){marker}"
            )

    print(f"Finished {args.arch}. Best Val mIoU: {best_miou:.4f} at Epoch {best_epoch}")


if __name__ == "__main__":
    main()
