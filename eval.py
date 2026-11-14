import os
import json
import time
import numpy as np
import torch

from data_contract import get_dataloaders, NUM_CLASSES, IGNORE_INDEX, IMAGE_SIZE, CLASS_NAMES
from models import build_model


def compute_iou_from_confmat(confmat):
    intersection = np.diag(confmat)
    union = confmat.sum(axis=1) + confmat.sum(axis=0) - intersection
    valid = union > 0
    iou = np.where(valid, intersection / np.maximum(union, 1), np.nan)
    return iou, float(np.nanmean(iou))


def compute_single_image_fg_miou(pred, target, num_classes=NUM_CLASSES):
    """Mean IoU over foreground classes (1..18) present in GT or prediction for a single image."""
    valid = target != IGNORE_INDEX
    ious = []
    for c in range(1, num_classes):
        p = (pred == c) & valid
        t = (target == c) & valid
        union = np.logical_or(p, t).sum()
        if union > 0:
            inter = np.logical_and(p, t).sum()
            ious.append(float(inter / union))
    return float(np.mean(ious)) if ious else 1.0


@torch.no_grad()
def evaluate_test_set(model, test_loader, device, num_classes=NUM_CLASSES):
    model.eval()
    confmat = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_image_fg_miou = {}

    for imgs, masks, img_ids in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().numpy()
        targets = masks.numpy()

        valid = targets != IGNORE_INDEX
        t_valid = targets[valid]
        p_valid = preds[valid]
        hist = np.bincount(num_classes * t_valid + p_valid, minlength=num_classes ** 2)
        confmat += hist.reshape(num_classes, num_classes)

        for pred_i, target_i, img_id in zip(preds, targets, img_ids):
            per_image_fg_miou[img_id] = compute_single_image_fg_miou(pred_i, target_i, num_classes)

    per_class_iou, miou = compute_iou_from_confmat(confmat)
    return {
        "per_class_iou": [None if np.isnan(v) else round(float(v), 4) for v in per_class_iou],
        "miou": float(miou),
        "confmat": confmat.tolist(),
        "per_image_fg_miou": per_image_fg_miou,
    }


@torch.no_grad()
def benchmark_latency(model, device, input_size=(1, 3, *IMAGE_SIZE), warmup=50, runs=300):
    model.eval()
    dummy = torch.randn(*input_size, device=device)

    for _ in range(warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr))
    return {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "precision": "FP32",
        "batch_size": input_size[0],
        "input_shape": list(input_size),
        "warmup_iters": warmup,
        "timed_iters": runs,
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(std_ms, 3),
        "fps": round(1000.0 / mean_ms, 1),
    }


def main():
    os.makedirs("/kaggle/working/logs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, splits = get_dataloaders(batch_size=16, seed=42)

    archs = ["baseline", "skip_fusion"]
    summary = {
        "dataset": "CelebAMask-HQ",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "test_split_count": len(splits["test"]),
        "models": {},
    }
    per_image_scores = {}

    for arch in archs:
        ckpt_path = f"/kaggle/working/checkpoints/{arch}_best.pt"
        model = build_model(arch=arch, num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        total_params = sum(p.numel() for p in model.parameters())
        metrics = evaluate_test_set(model, test_loader, device)
        latency = benchmark_latency(model, device)

        per_image_scores[arch] = metrics.pop("per_image_fg_miou")
        summary["models"][arch] = {
            "total_params": total_params,
            "test_miou": round(metrics["miou"], 4),
            "per_class_iou": {CLASS_NAMES[c]: metrics["per_class_iou"][c] for c in range(NUM_CLASSES)},
            "latency": latency,
        }

    with open("/kaggle/working/logs/test_eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("/kaggle/working/logs/test_per_image_ious.json", "w") as f:
        json.dump(per_image_scores, f)

    b = summary["models"]["baseline"]
    s = summary["models"]["skip_fusion"]
    lat_cond = b["latency"]

    print(f"=== Held-Out Test Set Evaluation: CelebAMask-HQ (N = {summary['test_split_count']} images, {NUM_CLASSES} classes) ===")
    print(f"Latency Conditions: {lat_cond['device']} | {lat_cond['precision']} | "
          f"Batch={lat_cond['batch_size']} | Input={lat_cond['input_shape']} | "
          f"Warmup={lat_cond['warmup_iters']}, Timed={lat_cond['timed_iters']}\n")

    header = f"{'Model':<14} | {'Params':>10} | {'Test mIoU':>10} | {'Latency (ms)':>16} | {'FPS':>7}"
    print(header)
    print("-" * len(header))
    for name, m in [("baseline", b), ("skip_fusion", s)]:
        lat_str = f"{m['latency']['mean_ms']:.2f} +/- {m['latency']['std_ms']:.2f}"
        print(f"{name:<14} | {m['total_params']:>10,} | {m['test_miou']:>10.4f} | {lat_str:>16} | {m['latency']['fps']:>7.1f}")

    print(f"\nDelta (skip_fusion - baseline): "
          f"dParams = +{s['total_params'] - b['total_params']:,} | "
          f"dmIoU = {s['test_miou'] - b['test_miou']:+.4f} | "
          f"dLatency = {s['latency']['mean_ms'] - b['latency']['mean_ms']:+.2f} ms\n")

    print("=== Per-Class Test IoU Breakdown (19 Classes) ===")
    c_header = f"{'ID':>2} | {'Class Name':<12} | {'Baseline IoU':>12} | {'Skip Fusion':>12} | {'Delta':>9}"
    print(c_header)
    print("-" * len(c_header))
    for c in range(NUM_CLASSES):
        cname = CLASS_NAMES[c]
        b_val = b["per_class_iou"][cname]
        s_val = s["per_class_iou"][cname]
        if b_val is not None and s_val is not None:
            print(f"{c:>2} | {cname:<12} | {b_val:>12.4f} | {s_val:>12.4f} | {s_val - b_val:>+9.4f}")
        else:
            print(f"{c:>2} | {cname:<12} | {'N/A':>12} | {'N/A':>12} | {'N/A':>9}")


if __name__ == "__main__":
    main()
