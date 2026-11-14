import os
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# ==========================================
# DATA CONTRACT SPECIFICATION (CelebAMask-HQ)
# ==========================================
DATASET_DIR = "/kaggle/input/datasets/ipythonx/celebamaskhq/CelebAMask-HQ"
CACHE_DIR = "/kaggle/working/cache_256"
SPLIT_FILE = "/kaggle/working/splits.json"
IMAGE_SIZE = (256, 256)
NUM_CLASSES = 19
IGNORE_INDEX = 255

# Deterministic subset sizes sampled from official CelebA-HQ partitions (seed=42)
SPLIT_COUNTS = {"train": 5000, "val": 800, "test": 1500}

# Canonical 19-class facial parsing order (back-to-front compositing priority)
PART_NAMES = [
    "skin",    # 1
    "l_brow",  # 2
    "r_brow",  # 3
    "l_eye",   # 4
    "r_eye",   # 5
    "eye_g",   # 6  (eyeglasses)
    "l_ear",   # 7
    "r_ear",   # 8
    "ear_r",   # 9  (earring)
    "nose",    # 10
    "mouth",   # 11
    "u_lip",   # 12
    "l_lip",   # 13
    "neck",    # 14
    "neck_l",  # 15 (necklace)
    "cloth",   # 16
    "hair",    # 17
    "hat",     # 18
]
CLASS_NAMES = {0: "background", **{i + 1: name for i, name in enumerate(PART_NAMES)}}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Distinct 19-color RGB palette for visualization
PALETTE = np.array([
    [20, 25, 40],     # 0: background
    [255, 204, 170],  # 1: skin
    [110, 60, 30],    # 2: l_brow
    [140, 80, 40],    # 3: r_brow
    [0, 180, 255],    # 4: l_eye
    [0, 130, 220],    # 5: r_eye
    [255, 230, 0],    # 6: eye_g
    [235, 160, 130],  # 7: l_ear
    [215, 140, 110],  # 8: r_ear
    [255, 140, 0],    # 9: ear_r
    [245, 130, 130],  # 10: nose
    [180, 30, 60],    # 11: mouth
    [235, 50, 90],    # 12: u_lip
    [210, 40, 120],   # 13: l_lip
    [200, 155, 125],  # 14: neck
    [255, 215, 90],   # 15: neck_l
    [50, 175, 120],   # 16: cloth
    [135, 75, 195],   # 17: hair
    [255, 95, 50],    # 18: hat
], dtype=np.uint8)


def create_or_load_splits(dataset_dir=DATASET_DIR, split_file=SPLIT_FILE, seed=42):
    """Deterministically sample fixed-size train/val/test subsets from the official
    CelebA-HQ-to-CelebA-mapping.txt partitions (zero cross-partition overlap).
    """
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            return json.load(f)

    map_path = os.path.join(dataset_dir, "CelebA-HQ-to-CelebA-mapping.txt")
    full_train, full_val, full_test = [], [], []

    with open(map_path, "r") as f:
        _ = f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            hq_idx = int(parts[0])
            orig_idx = int(parts[1])
            if orig_idx < 162770:
                full_train.append(hq_idx)
            elif orig_idx < 182637:
                full_val.append(hq_idx)
            else:
                full_test.append(hq_idx)

    rng = random.Random(seed)
    train_ids = sorted(rng.sample(sorted(full_train), SPLIT_COUNTS["train"]))
    val_ids = sorted(rng.sample(sorted(full_val), SPLIT_COUNTS["val"]))
    test_ids = sorted(rng.sample(sorted(full_test), SPLIT_COUNTS["test"]))

    splits = {
        "protocol": f"Deterministic fixed-seed (seed={seed}) subset of official CelebAMask-HQ train/val/test partitions",
        "seed": seed,
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
    with open(split_file, "w") as f:
        json.dump(splits, f, indent=2)
    return splits


def _build_mask_file_index():
    """Index all mask files across subfolders 0..14 in 15 os.listdir calls."""
    part_to_id = {name: i + 1 for i, name in enumerate(PART_NAMES)}
    mask_root = os.path.join(DATASET_DIR, "CelebAMask-HQ-mask-anno")
    file_index = defaultdict(list)

    for sub in os.listdir(mask_root):
        sub_path = os.path.join(mask_root, sub)
        if not os.path.isdir(sub_path):
            continue
        for fname in os.listdir(sub_path):
            if not fname.endswith(".png"):
                continue
            idx_str, part_ext = fname.split("_", 1)
            part_name = part_ext[:-4]
            if part_name in part_to_id:
                file_index[int(idx_str)].append((part_to_id[part_name], os.path.join(sub_path, fname)))

    for hq_idx in file_index:
        file_index[hq_idx].sort(key=lambda x: x[0])
    return file_index


def prepare_cache(target_ids):
    img_dir = os.path.join(CACHE_DIR, "images")
    mask_dir = os.path.join(CACHE_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    existing_masks = set(os.listdir(mask_dir))
    todo_ids = [i for i in target_ids if f"{i}.png" not in existing_masks]
    if not todo_ids:
        return

    print(f"Indexing mask files and caching {len(todo_ids)} target images at 256x256...")
    file_index = _build_mask_file_index()

    def process_one(hq_idx):
        img_out = os.path.join(img_dir, f"{hq_idx}.jpg")
        mask_out = os.path.join(mask_dir, f"{hq_idx}.png")

        src_img = os.path.join(DATASET_DIR, "CelebA-HQ-img", f"{hq_idx}.jpg")
        img_bgr = cv2.imread(src_img, cv2.IMREAD_COLOR)
        img_256 = cv2.resize(img_bgr, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(img_out, img_256, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        merged = np.zeros((512, 512), dtype=np.uint8)
        for class_id, part_path in file_index[hq_idx]:
            part_arr = cv2.imread(part_path, cv2.IMREAD_GRAYSCALE)
            if part_arr is not None:
                merged[part_arr > 127] = class_id

        mask_256 = cv2.resize(merged, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(mask_out, mask_256)

    with ThreadPoolExecutor(max_workers=12) as ex:
        list(tqdm(ex.map(process_one, todo_ids), total=len(todo_ids), desc="Caching 256x256"))
    print("Cache ready.")


class CelebAMaskHQDataset(Dataset):
    def __init__(self, image_ids, cache_dir=CACHE_DIR, is_train=False, image_size=IMAGE_SIZE):
        self.image_ids = image_ids
        self.cache_dir = cache_dir
        self.is_train = is_train
        self.image_size = image_size

    def __len__(self):
        return len(self.image_ids)

    def sync_transform(self, img, mask):
        if self.is_train:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask_np = np.array(TF.hflip(mask), dtype=np.uint8)
                swapped = mask_np.copy()
                for left_id, right_id in [(2, 3), (4, 5), (7, 8)]:
                    swapped[mask_np == left_id] = right_id
                    swapped[mask_np == right_id] = left_id
                mask = Image.fromarray(swapped)

            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=(0.80, 1.0), ratio=(0.95, 1.05)
            )
            img = TF.resized_crop(img, i, j, h, w, self.image_size, InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, self.image_size, InterpolationMode.NEAREST)

        img_tensor = TF.normalize(TF.to_tensor(img), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img_tensor, mask_tensor

    def __getitem__(self, idx):
        hq_idx = self.image_ids[idx]
        img_path = os.path.join(self.cache_dir, "images", f"{hq_idx}.jpg")
        mask_path = os.path.join(self.cache_dir, "masks", f"{hq_idx}.png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        img_tensor, mask_tensor = self.sync_transform(img, mask)
        return img_tensor, mask_tensor, str(hq_idx)


def get_dataloaders(batch_size=16, num_workers=2, seed=42):
    splits = create_or_load_splits(seed=seed)
    all_target_ids = splits["train"] + splits["val"] + splits["test"]
    prepare_cache(all_target_ids)

    train_ds = CelebAMaskHQDataset(splits["train"], is_train=True)
    val_ds = CelebAMaskHQDataset(splits["val"], is_train=False)
    test_ds = CelebAMaskHQDataset(splits["test"], is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, splits


def denormalize(img_tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (img_tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def colorize_mask(mask_np):
    vis = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    valid = (mask_np >= 0) & (mask_np < NUM_CLASSES)
    vis[valid] = PALETTE[mask_np[valid]]
    return vis


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    os.makedirs("/kaggle/working/figures", exist_ok=True)
    train_loader, val_loader, test_loader, splits = get_dataloaders(batch_size=4, seed=42)
    imgs, masks, ids = next(iter(train_loader))

    print(f"Splits -> Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    print(f"Batch image shape: {imgs.shape}, mask shape: {masks.shape}, unique class IDs in batch: {torch.unique(masks).tolist()}")

    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    for row in range(4):
        rgb = denormalize(imgs[row])
        m = masks[row].numpy()
        m_color = colorize_mask(m)
        overlay = (0.55 * rgb + 0.45 * (m_color / 255.0)).clip(0, 1)

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"ID {ids[row]} (Augmented Input)")
        axes[row, 1].imshow(m_color)
        axes[row, 1].set_title("Merged 19-Class Mask")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Alignment Overlay")
        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("/kaggle/working/figures/data_contract_check.png", dpi=150)
    print("Saved /kaggle/working/figures/data_contract_check.png")
