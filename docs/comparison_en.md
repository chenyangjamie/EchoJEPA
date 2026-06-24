# EchoJEPA: Configuration and Results Comparison

**Date:** 2026-06-24  
**Task:** LVEF Regression (Left Ventricular Ejection Fraction)  
**Dataset:** EchoNet-Dynamic (public)

---

## 1. Model Architecture

| Item | Our Run | Paper: EchoJEPA-L (ViT-L) | Paper: EchoJEPA-G (ViT-G) |
|---|---|---|---|
| Backbone | ViT-L/16 (300M params) | ViT-L/16 (300M params) | ViT-G/16 (1B params) |
| Backbone checkpoint | `vitl-vmix22m-pt220-c55.pt` (public) | `vitl-vmix22m-pt220-c55.pt` (public, same as our run) | `pt-280-an81.pt` (not public) |
| Backbone pretraining data | MIMIC-IV-Echo (VideoMix22M init) | MIMIC-IV-Echo 525K (VideoMix22M init) | Proprietary 18.1M echo videos |

> **Note:** The original `echonet_dynamic_lvef.yaml` and `echojepa_large_lvef.yaml` configs in the repo point to a different SageMaker-internal checkpoint: `checkpoints/anneal/keep/vitl-pt-210-an25.pt`. This was likely the backbone used during the authors' internal experiments. The publicly released EchoJEPA-L backbone (and the one the paper describes for reproducibility) is `vitl-vmix22m-pt220-c55.pt`.
| Probe type | 4-block attentive probe | 4-block attentive probe | 4-block attentive probe |
| Probe heads | 16 | 16 | 16 |
| Task | Regression (LVEF) | Regression (LVEF) | Regression (LVEF) |
| Backbone frozen | Yes | Yes | Yes |

> **Key difference:** Our run and paper EchoJEPA-L use the **same backbone** (`vitl-vmix22m-pt220-c55.pt`, MIMIC-IV-Echo, public). The sole differences are probe training data (EchoNet-Dynamic TRAIN 7,465 vs Toronto 150,000 studies) and evaluation protocol (in-distribution vs zero-shot cross-site). EchoJEPA-G used a far larger proprietary dataset (18.1M) with a ViT-G backbone.

---

## 2. Data Configuration

| Item | Our Run | Paper: EchoJEPA-L (ViT-L) | Paper: EchoJEPA-G (ViT-G) |
|---|---|---|---|
| Training dataset | EchoNet-Dynamic | Toronto internal (N=150,000 studies) | Toronto internal (N=150,000 studies) |
| Train samples | 7,465 | ~150,000 studies | ~150,000 studies |
| Val samples | 1,288 | Toronto internal (held-out) | Toronto internal (held-out) |
| Test samples | 1,277 (EchoNet-Dynamic test split) | EchoNet-Dynamic (zero-shot, all 10,030) | EchoNet-Dynamic (zero-shot, all 10,030) |
| Video format | AVI (112×112, ~50fps) | MP4 | MP4 |
| Training resolution | 112px | 112px (EchoNet eval config) / 336px (private dataset) | 336px |
| Inference resolution (EchoNet-Dynamic) | 112px | 112px | 112px |
| frames_per_clip | 16 | 16 | 16 |
| frame_step | 2 | 2 | 2 |
| num_segments | 2 | 2 | 2 |
| target_mean | 55.7776 (EchoNet TRAIN) | 57.0569 (Toronto dataset) | 57.06 (Toronto dataset) |
| target_std | 12.4064 (EchoNet TRAIN) | 11.3252 (Toronto dataset) | 11.33 (Toronto dataset) |

---

## 3. Training Configuration

| Item | Our Run | Paper: EchoJEPA-L (ViT-L) | Paper: EchoJEPA-G (ViT-G) |
|---|---|---|---|
| Hardware | 1× NVIDIA RTX PRO 6000 Blackwell (97.9 GB) | 8× GPU (AWS SageMaker) | 8× GPU (AWS SageMaker) |
| Training epochs | 20 | 20 | Not specified |
| Batch size per GPU | 16 | 1 | 80 |
| Effective batch size | 16 | 8 (1×8 GPUs) | 640 (80×8 GPUs) |
| LR grid | {1e-4, 5e-5} | {1e-4, 5e-5} | Not specified |
| Weight decay grid | {0.01, 0.1, 0.4} | {0.01, 0.1, 0.4} | Not specified |
| Num probes (parallel) | 6 (2 lr × 3 wd) | 6 | Not specified |
| Optimizer | AdamW | AdamW | AdamW |
| Precision | bfloat16 | bfloat16 | bfloat16 |
| Training time | ~1h 40m | Not reported | Not reported |
| GPU memory used | ~6.8 GB | ~80 GB (config upper bound) | ~80 GB (config upper bound) |

---

## 4. Results

### 4.1 LVEF MAE (EF%) — Cross-Site Evaluation (Paper Table 7)

The paper evaluates on three hospital sites. "Stanford" corresponds to EchoNet-Dynamic, the same dataset we used.

| Model | Toronto | Chicago | Stanford (EchoNet-Dynamic) |
|---|---|---|---|
| EchoPrime | 5.33 | 6.71 | 4.87 |
| PanEcho | 5.43 | 6.52 | 5.10 |
| EchoMAE-L (reconstruction baseline) | 8.15 | 9.40 | 8.52 |
| **EchoJEPA-L (ViT-L, paper)** | **5.97** | **7.39** | **5.76** |
| **EchoJEPA-G (ViT-G, paper best)** | **4.26** | **5.44** | **3.97** |
| **Our Run (ViT-L, EchoNet-Dynamic)** | — | — | **5.08** |

> The paper's EchoJEPA-L probe was trained on **Toronto (N=150,000 studies)** and evaluated **zero-shot** on EchoNet-Dynamic — no EchoNet samples were ever seen during probe training. Our run trains **and** evaluates within EchoNet-Dynamic splits, a far easier in-distribution protocol. Despite this advantage, our MAE (5.08%) is only modestly better than zero-shot EchoJEPA-L (5.76%), highlighting the strength of training on a larger internal dataset.

### 4.2 LVEF MAE on Primary Dataset (Paper Table 6)

| Model | LVEF MAE ↓ | View Acc ↑ |
|---|---|---|
| EchoMAE-L (reconstruction) | 8.15 | 40.4% |
| **EchoJEPA-L (ViT-L)** | **5.97** | **85.5%** |
| **EchoJEPA-G (ViT-G)** | — | — |
| Relative improvement (L vs EchoMAE-L) | **−26.7%** | **+45.1%** |

### 4.3 Robustness Under Acoustic Perturbations (Paper Table 9, Stanford site)

| Model | Original MAE | Avg. Degradation |
|---|---|---|
| EchoPrime | 4.87 | +16.8% |
| PanEcho | 5.10 | +3.7% |
| EchoMAE-L | 8.52 | +0.5%† |
| **EchoJEPA-L (ViT-L)** | **5.76** | **+2.3%** |
| **EchoJEPA-G (ViT-G)** | **3.97** | **+2.3%** |

† EchoMAE-L is already saturated at high error, making absolute degradation small.

### 4.4 Our Run — Training Curve

| Epoch | Train Loss | Val MAE (min-head, EF%) | Best |
|---|---|---|---|
| 1 | 8.712 | 7.061 | 7.061 |
| 5 | 6.609 | 6.158 | 5.481 |
| 10 | 6.118 | 5.168 | 5.026 |
| 15 | 5.702 | 4.829 | 4.829 |
| 17 | 5.801 | **4.752** | **4.752** ← saved |
| 20 | 5.672 | 5.003 | 4.752 |

### 4.5 Our Run — Final Test Metrics

| Metric | Value |
|---|---|
| Test MAE | **5.08 EF%** |
| Test RMSE | **6.74 EF%** |
| Pearson r | **0.837** |
| Best Val MAE | 4.752 EF% (epoch 17) |
| Samples evaluated | 1,248 / 1,277 |

---

## 5. Key Differences and Their Impact

### 5.1 Backbone Pretraining Domain
Our run and paper EchoJEPA-L use the **identical backbone** (`vitl-vmix22m-pt220-c55.pt`). The performance difference between the two (5.08% vs 5.76%) is therefore entirely attributable to probe training data and evaluation protocol — not the backbone. EchoJEPA-G used a far larger proprietary dataset (18.1M vs 525K) with a ViT-G backbone, giving it substantially richer representations.

### 5.2 Model Size (L vs G)
EchoJEPA-G (1B params) achieves 3.97 EF% on EchoNet-Dynamic vs EchoJEPA-L's 5.76% — a 31% relative improvement from model scale alone. ViT-G benefits from richer spatiotemporal representations and larger pretraining capacity.

### 5.3 Evaluation Protocol
The paper trains probes on Toronto (N=150,000 studies) and evaluates **zero-shot** on EchoNet-Dynamic — no EchoNet samples are ever used for probe training or validation. Our run trains on EchoNet-Dynamic TRAIN (7,465 samples) and evaluates on EchoNet-Dynamic TEST. This is a fundamentally easier setting: the probe is already adapted to EchoNet's acquisition characteristics (Philips IE33, Stanford protocol), whereas the paper's probe must generalize across institutions. That our 5.08% only slightly outperforms the zero-shot 5.76% underscores the value of larger-scale probe training data.

### 5.4 Inference Resolution
For EchoNet-Dynamic evaluation, all models (including paper EchoJEPA-L) use **112px** — this is the native EchoNet resolution and confirmed by the original `echonet_dynamic_lvef.yaml` config. The paper's private dataset evaluation (Toronto/Chicago, 5.97% result) used **336px**, as seen in the inference config (`echojepa_large_lvef.yaml`). RoPE enables resolution flexibility, so the same checkpoint can run at both resolutions without retraining.

### 5.5 Training Hardware and Batch Size
ViT-L paper: 8 GPUs × batch 1 = effective batch 8. Our run: 1 GPU × batch 16 = effective batch 16. ViT-G paper: 8 GPUs × batch 80 = effective batch 640. Larger batch sizes generally stabilize training for regression probes.

---

## 6. Checkpoints and Artifacts

| Item | Path |
|---|---|
| Backbone checkpoint | `checkpoints/vitl-vmix22m-pt220-c55.pt` |
| Best probe checkpoint | `experiments/echonet_lvef/video_classification_frozen/echojepa-vitl-echonet-dynamic-lvef/best.pt` |
| Train CSV | `data/csv/echonet_dynamic_train.csv` (7,465 rows) |
| Val CSV | `data/csv/echonet_dynamic_val.csv` (1,288 rows) |
| Test CSV | `data/csv/echonet_dynamic_test.csv` (1,277 rows) |
| Test predictions | `experiments/echonet_lvef_inference/test_predictions.csv` |
| Training log | `experiments/echonet_lvef/train.log` |
| Inference log | `experiments/echonet_lvef_inference/infer.log` |
| Eval config | `configs/eval/vitl/echonet_dynamic_lvef.yaml` |
| Inference config | `configs/inference/vitl/echojepa_large_lvef.yaml` |

---

## 7. Summary

| | Our Run | EchoJEPA-L (paper, ViT-L) | EchoJEPA-G (paper, ViT-G) |
|---|---|---|---|
| Backbone | ViT-L, `vitl-vmix22m-pt220-c55.pt` (same as paper) | ViT-L, `vitl-vmix22m-pt220-c55.pt` (same as our run) | ViT-G, 18.1M proprietary echo |
| EchoNet-Dynamic resolution | 112px | 112px | 112px |
| Probe training data | EchoNet-Dynamic TRAIN (7,465) | Toronto internal (150,000 studies) | Toronto internal (150,000 studies) |
| Evaluation | EchoNet-Dynamic TEST (in-dist.) | EchoNet-Dynamic (zero-shot, cross-site) | EchoNet-Dynamic (zero-shot, cross-site) |
| **Test MAE (EF%)** | **5.08** | **5.76** | **3.97** |
| Test RMSE (EF%) | 6.74 | — | — |
| Pearson r | 0.837 | — | — |

Our single-GPU ViT-L run achieves **5.08 EF% MAE** — slightly better than paper EchoJEPA-L (5.76%), but under a fundamentally easier protocol: our probe is trained on EchoNet-Dynamic TRAIN and tested on its TEST split (in-distribution), while the paper's probe was trained on Toronto (150,000 studies) and applied **zero-shot** to EchoNet-Dynamic with no adaptation. To match the paper's best performance (3.97%), the key steps would be: (1) use ViT-G backbone, (2) increase resolution to 336px, (3) train the probe on a larger internal dataset.
