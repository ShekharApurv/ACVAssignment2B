# 19-Class Facial Semantic Segmentation on CelebAMask-HQ

A reproducible PyTorch semantic segmentation pipeline evaluating the impact of multi-scale skip connections (**Skip Fusion**) against a plain encoder-decoder **Baseline** on a deterministic 7,300-image subset of **CelebAMask-HQ**.

---

## 1. Project Overview & Key Findings

High-resolution facial parsing requires classifying both large, smooth regions (`skin`, `hair`, `background`) and fine, localized anatomical structures (`l_brow`, `r_brow`, `l_eye`, `r_eye`, `u_lip`, `l_lip`, `l_ear`, `r_ear`). A standard stride-32 encoder-decoder loses high-frequency spatial detail during downsampling, blurring small facial boundaries.

To isolate the effect of multi-scale spatial feature reuse under a strictly controlled experimental budget, we compare two variants of `FaceSegNet` (ImageNet-pretrained `ResNet-18` backbone):

* **Baseline (`baseline`):** Plain 4-stage bilinear upsampling + $3 \times 3$ convolution decoder (`12,754,099` parameters).
* **Controlled Variant (`skip_fusion`):** Identical architecture with $1 \times 1$ `Conv-BN` skip projections from encoder stages `layer3` (stride 16), `layer2` (stride 8), and `layer1` (stride 4) added element-wise to the corresponding decoder stages (`12,841,011` parameters).

### Summary Results (Held-Out Test Set, $N = 1,500$)

| Model | Parameters | Best Val mIoU | Held-Out Test mIoU | FP32 Latency (ms/img) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 12,754,099 | 0.6461 (Ep 12) | 0.6531 | $6.09 \pm 0.06$ | 164.2 |
| **Skip Fusion** | 12,841,011 | **0.6932** (Ep 12) | **0.7068** | $6.29 \pm 0.12$ | 159.1 |
| **Delta ($\Delta$)** | **+86,912 (+0.68%)** | **+0.0471** | **+0.0537 (+8.22% rel.)** | **+0.20 ms** | **-5.1 FPS** |

*Latency measured on an NVIDIA Tesla T4 GPU (`FP32`, `batch_size=1`, input shape `[1, 3, 256, 256]`, 50 warmup iterations followed by 300 `torch.cuda.synchronize()`-timed forward passes).*

---

## 2. Repository & Submission Structure

```text
/23115013_acv_assignment_2b/
├── data_contract.py           # Official split mapping, 19-class mask compositing, 256x256 cache & transforms
├── models.py                  # FaceSegNet architecture (baseline vs. skip_fusion)
├── train.py                   # Deterministic training loop, validation mIoU tracking & checkpointing
├── eval.py                    # Held-out test evaluation (19-class IoU) & FP32 latency benchmarking
├── visualize.py               # Training curves, per-class IoU chart & qualitative/failure panels
├── generate_report_pdf.py     # Compiles project_report.pdf with embedded figures
├── splits.json                # Deterministic train (5,000) / val (800) / test (1,500) image ID manifest
├── env_versions.txt           # Hardware & Python package versions
├── logs/
│   ├── baseline_config.json
│   ├── baseline_train.csv
│   ├── skip_fusion_config.json
│   ├── skip_fusion_train.csv
│   ├── test_eval_results.json
│   └── test_per_image_ious.json
└── figures/
    ├── data_contract_check.png
    ├── training_curves.png
    ├── per_class_iou_comparison.png
    ├── test_improvements.png
    └── test_failures.png
```

---

## 3. Data Contract & Split Protocol (`data_contract.py`)

* **Source Dataset:** CelebAMask-HQ (`ipythonx/celebamaskhq` on Kaggle), containing 30,000 facial portraits and 18 binary per-part masks ($512 \times 512$) per image.
* **Leakage-Free Split Partitioning (`splits.json`):** Images are mapped back to the official non-overlapping CelebA partitions via `CelebA-HQ-to-CelebA-mapping.txt` (`orig_idx < 162770` $\rightarrow$ train pool; `162770 <= orig_idx < 182637` $\rightarrow$ val pool; `orig_idx >= 182637` $\rightarrow$ test pool) and deterministically subsampled with `seed=42`:
  * **Train:** `5,000` images
  * **Validation:** `800` images
  * **Held-Out Test:** `1,500` images
* **19-Class Label Compositing:** Individual part masks are composited in canonical back-to-front order onto a single integer label map (`0`–`18`, `ignore_index=255`) and resized to $256 \times 256$ (`INTER_LINEAR` for RGB images, `INTER_NEAREST` for masks):
  * `0`: `background`, `1`: `skin`, `2`: `l_brow`, `3`: `r_brow`, `4`: `l_eye`, `5`: `r_eye`, `6`: `eye_g`, `7`: `l_ear`, `8`: `r_ear`, `9`: `ear_r`, `10`: `nose`, `11`: `mouth`, `12`: `u_lip`, `13`: `l_lip`, `14`: `neck`, `15`: `neck_l`, `16`: `cloth`, `17`: `hair`, `18`: `hat`.
* **Synchronized Transforms:**
  * *Train:* `RandomHorizontalFlip(p=0.5)` with anatomical left/right class swapping (`l_brow` $\leftrightarrow$ `r_brow`, `l_eye` $\leftrightarrow$ `r_eye`, `l_ear` $\leftrightarrow$ `r_ear`), `RandomResizedCrop((256, 256), scale=(0.80, 1.0), ratio=(0.95, 1.05))`, and ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
  * *Validation / Test:* Deterministic $256 \times 256$ input with ImageNet normalization only.

---

## 4. Reproducing the Experiment

### Environment Requirements
* Python 3.10+
* PyTorch 2.x, Torchvision, OpenCV (`opencv-python`), NumPy, Pillow, Matplotlib, `tqdm` (and `fpdf2` for PDF compilation)
* Dataset mounted at `/kaggle/input/datasets/ipythonx/celebamaskhq/CelebAMask-HQ`

### Local Setup (venv + requirements.txt)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

# 2. Upgrade pip and install the project dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This repository includes a project-level `requirements.txt` so the environment can be recreated consistently across machines.

### Step-by-Step Execution

```bash
# 1. Build 256x256 cache, lock splits.json, and verify mask alignment
python data_contract.py

# 2. Train Baseline model (12 epochs, seed=42)
python train.py --arch baseline --epochs 12 --batch_size 16 --seed 42

# 3. Train Skip Fusion model under the identical budget (12 epochs, seed=42)
python train.py --arch skip_fusion --epochs 12 --batch_size 16 --seed 42

# 4. Evaluate both best checkpoints on the 1,500 held-out test images & benchmark latency
python eval.py

# 5. Generate training curves, per-class IoU chart, and qualitative/failure panels
python visualize.py

# 6. Compile the formatted PDF report with embedded figures
python generate_report_pdf.py
```

---

## 5. Per-Class Held-Out Test IoU Breakdown (19 Classes)

| ID | Class Name | Baseline IoU | Skip Fusion IoU | Delta ($\Delta$) | ID | Class Name | Baseline IoU | Skip Fusion IoU | Delta ($\Delta$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **0** | `background` | 0.9062 | **0.9178** | +0.0116 | **10** | `nose` | 0.8502 | **0.8685** | +0.0183 |
| **1** | `skin` | 0.8982 | **0.9186** | +0.0204 | **11** | `mouth` | 0.6866 | **0.7715** | +0.0849 |
| **2** | `l_brow` | 0.6240 | **0.7200** | +0.0960 | **12** | `u_lip` | 0.6262 | **0.7350** | **+0.1088** |
| **3** | `r_brow` | 0.6073 | **0.7123** | **+0.1050** | **13** | `l_lip` | 0.7223 | **0.7872** | +0.0649 |
| **4** | `l_eye` | 0.6843 | **0.7541** | +0.0698 | **14** | `neck` | 0.7878 | **0.8143** | +0.0265 |
| **5** | `r_eye` | 0.6742 | **0.7598** | +0.0856 | **15** | `neck_l` | 0.0000 | 0.0000 | +0.0000 |
| **6** | `eye_g` | 0.7463 | **0.7933** | +0.0470 | **16** | `cloth` | 0.7038 | **0.7368** | +0.0330 |
| **7** | `l_ear` | 0.6527 | **0.7502** | +0.0975 | **17** | `hair` | 0.8896 | **0.9053** | +0.0157 |
| **8** | `r_ear` | 0.6268 | **0.7325** | **+0.1057** | **18** | `hat` | 0.7226 | **0.7437** | +0.0211 |
| **9** | `ear_r` | 0.0000 | **0.0077** | +0.0077 | | **Mean (mIoU)** | **0.6531** | **0.7068** | **+0.0537** |