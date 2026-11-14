import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from data_contract import (
    CelebAMaskHQDataset,
    SPLIT_FILE,
    NUM_CLASSES,
    CLASS_NAMES,
    PALETTE,
    denormalize,
    colorize_mask,
)
from models import build_model


def load_train_log(csv_path):
    epochs, train_loss, val_loss, val_miou = [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_miou.append(float(row["val_miou"]))
    return epochs, train_loss, val_loss, val_miou


def plot_training_curves(out_path="/kaggle/working/figures/training_curves.png"):
    ep_b, tr_b, va_b, miou_b = load_train_log("/kaggle/working/logs/baseline_train.csv")
    ep_s, tr_s, va_s, miou_s = load_train_log("/kaggle/working/logs/skip_fusion_train.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].plot(ep_b, tr_b, "--", color="#1f77b4", label="Baseline Train Loss")
    axes[0].plot(ep_b, va_b, "-", color="#1f77b4", linewidth=2, label="Baseline Val Loss")
    axes[0].plot(ep_s, tr_s, "--", color="#d62728", label="Skip Fusion Train Loss")
    axes[0].plot(ep_s, va_s, "-", color="#d62728", linewidth=2, label="Skip Fusion Val Loss")
    axes[0].set_title("Cross-Entropy Loss (19 Facial Classes)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(ep_b, miou_b, "o-", color="#1f77b4", linewidth=2, markersize=5, label="Baseline Val mIoU")
    axes[1].plot(ep_s, miou_s, "s-", color="#d62728", linewidth=2, markersize=5, label="Skip Fusion Val mIoU")
    axes[1].set_title("Validation Mean IoU (19-Class mIoU)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_per_class_iou(out_path="/kaggle/working/figures/per_class_iou_comparison.png"):
    with open("/kaggle/working/logs/test_eval_results.json", "r") as f:
        res = json.load(f)

    b_ious = res["models"]["baseline"]["per_class_iou"]
    s_ious = res["models"]["skip_fusion"]["per_class_iou"]

    names = [CLASS_NAMES[c] for c in range(NUM_CLASSES)]
    deltas = [s_ious[n] - b_ious[n] for n in names]
    order = np.argsort(deltas)  # ascending so largest gains appear at the top

    sorted_names = [names[i] for i in order]
    sorted_b = [b_ious[names[i]] for i in order]
    sorted_s = [s_ious[names[i]] for i in order]
    sorted_d = [deltas[i] for i in order]

    y = np.arange(len(sorted_names))
    height = 0.38

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.barh(y - height / 2, sorted_b, height, color="#4c78a8", label="Baseline (Test mIoU = 0.6531)")
    ax.barh(y + height / 2, sorted_s, height, color="#e45756", label="Skip Fusion (Test mIoU = 0.7068)")

    for idx, (s_val, d_val) in enumerate(zip(sorted_s, sorted_d)):
        ax.text(s_val + 0.012, y[idx] + height / 2, f"{d_val:+.3f}", va="center", fontsize=8.5, fontweight="bold", color="#222222")

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_names, fontsize=9.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Held-Out Test IoU (N = 1,500)")
    ax.set_title("Per-Class Test IoU & Skip Fusion Gain Across All 19 CelebAMask-HQ Classes")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved {out_path}")


@torch.no_grad()
def render_comparison_grid(sample_ids, title, out_path, models_dict, id_to_pos, test_ds, b_scores, s_scores, device):
    n = len(sample_ids)
    fig, axes = plt.subplots(n, 4, figsize=(13.5, 3.35 * n + 0.9))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, img_id in enumerate(sample_ids):
        ds_idx = id_to_pos[img_id]
        img_tensor, mask_tensor, _ = test_ds[ds_idx]
        inp = img_tensor.unsqueeze(0).to(device)

        pred_b = models_dict["baseline"](inp).argmax(dim=1).squeeze(0).cpu().numpy()
        pred_s = models_dict["skip_fusion"](inp).argmax(dim=1).squeeze(0).cpu().numpy()

        rgb = denormalize(img_tensor)
        gt_color = colorize_mask(mask_tensor.numpy())
        b_overlay = (0.45 * rgb + 0.55 * (colorize_mask(pred_b) / 255.0)).clip(0, 1)
        s_overlay = (0.45 * rgb + 0.55 * (colorize_mask(pred_s) / 255.0)).clip(0, 1)

        b_iou = b_scores[img_id]
        s_iou = s_scores[img_id]
        delta = s_iou - b_iou

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"Source ID: {img_id}", fontsize=9.5)

        axes[row, 1].imshow(gt_color)
        axes[row, 1].set_title("Ground Truth (19-Class)", fontsize=9.5)

        axes[row, 2].imshow(b_overlay)
        axes[row, 2].set_title(f"Baseline Prediction\nFG mIoU: {b_iou:.4f}", fontsize=9.5)

        axes[row, 3].imshow(s_overlay)
        axes[row, 3].set_title(f"Skip Fusion Prediction\nFG mIoU: {s_iou:.4f} ({delta:+.4f})", fontsize=9.5)

        for col in range(4):
            axes[row, col].axis("off")

    legend_patches = [
        mpatches.Patch(color=PALETTE[c] / 255.0, label=f"{c}:{CLASS_NAMES[c]}")
        for c in range(NUM_CLASSES)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=7,
        fontsize=7.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs("/kaggle/working/figures", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plot_training_curves()
    plot_per_class_iou()

    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)
    with open("/kaggle/working/logs/test_per_image_ious.json", "r") as f:
        scores = json.load(f)

    test_ds = CelebAMaskHQDataset(splits["test"], is_train=False)
    id_to_pos = {str(hq_idx): i for i, hq_idx in enumerate(splits["test"])}

    b_scores = scores["baseline"]
    s_scores = scores["skip_fusion"]
    test_ids = list(s_scores.keys())

    # Top 4 largest foreground mIoU improvements (skip_fusion - baseline)
    top_improve_ids = sorted(test_ids, key=lambda k: s_scores[k] - b_scores[k], reverse=True)[:4]
    # Lowest 4 foreground mIoU cases under skip_fusion (failure analysis)
    worst_fail_ids = sorted(test_ids, key=lambda k: s_scores[k])[:4]

    print("Top improvement test IDs:", top_improve_ids)
    print("Lowest FG mIoU (failure) test IDs:", worst_fail_ids)

    models_dict = {}
    for arch in ["baseline", "skip_fusion"]:
        m = build_model(arch=arch, num_classes=NUM_CLASSES).to(device)
        m.load_state_dict(torch.load(f"/kaggle/working/checkpoints/{arch}_best.pt", map_location=device))
        m.eval()
        models_dict[arch] = m

    render_comparison_grid(
        top_improve_ids,
        "Test Set Qualitative Wins (Largest Foreground mIoU Gains from Skip Fusion)",
        "/kaggle/working/figures/test_improvements.png",
        models_dict,
        id_to_pos,
        test_ds,
        b_scores,
        s_scores,
        device,
    )

    render_comparison_grid(
        worst_fail_ids,
        "Test Set Failure Cases (Lowest Foreground mIoU Samples)",
        "/kaggle/working/figures/test_failures.png",
        models_dict,
        id_to_pos,
        test_ds,
        b_scores,
        s_scores,
        device,
    )


if __name__ == "__main__":
    main()
