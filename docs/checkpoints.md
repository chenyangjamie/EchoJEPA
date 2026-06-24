# EchoJEPA Checkpoints Summary

**Location:** `/raid2/nextgen/new_projects/EchoJEPA/checkpoints/`

---

## Downloaded Checkpoints (5 total)

All 5 checkpoints are EchoJEPA versions — all pretrained on **MIMIC-IV-Echo (525K echocardiogram clips)**.  
The original V-JEPA 2 natural-video-only weights (`vitl.pt`, `vitg-384.pt`, etc.) are separate and were NOT downloaded.

| Checkpoint | Architecture | Initialization | Echo Pretraining Data | Training Completeness |
|---|---|---|---|---|
| `vitl-vmix22m-pt220-c55.pt` | ViT-L (300M) | V-JEPA 2 (VideoMix22M natural video) | MIMIC-IV-Echo | Complete — 220 pretrain + 55 cooldown epochs |
| `vitl-scratch-pt-210-c25.pt` | ViT-L (300M) | Random (scratch) | MIMIC-IV-Echo | Complete — 210 pretrain + 25 cooldown epochs |
| `vjepa 2.1/vjepa21_vitl_mimic_pt100.pt` | ViT-L (300M, V-JEPA 2.1) | Unknown | MIMIC-IV-Echo | Mid-training snapshot — 100 epochs, no cooldown |
| `vjepa 2.1/vjepa21_vitl_mimic_pt117.pt` | ViT-L (300M, V-JEPA 2.1) | Unknown | MIMIC-IV-Echo | Mid-training snapshot — 117 epochs, no cooldown |
| `vjepa 2.1/vjepa2_1_vitb_mimic_pt169_c60.pt` | ViT-B (80M, V-JEPA 2.1) | Unknown | MIMIC-IV-Echo | Complete — 169 pretrain + 60 cooldown epochs |

---

## Notes

**Naming convention:** `pt` = pretraining epochs, `c` = cooldown/annealing epochs.  
Cooldown applies a linear LR decay to final_lr ≈ 1e-6, improving convergence and resolution scaling.

**V-JEPA 2 vs V-JEPA 2.1 architecture difference:**
- **V-JEPA 2** — masked latent prediction on a subset of tokens; better for functional tasks (LVEF, RVSP)
- **V-JEPA 2.1** — dense predictive supervision over *all* tokens + deep self-supervision across encoder layers; better for spatial/dense tasks (segmentation, tracking, attention visualization)

**Recommended checkpoint per task (from README):**
- LVEF regression, RVSP regression, view classification → `vitl-vmix22m-pt220-c55.pt`
- Segmentation, tracking, interpretability → V-JEPA 2.1 checkpoints

---

## Relation to Paper

| Paper Model | Params | Echo Pretraining Data | Backbone Checkpoint | Public? |
|---|---|---|---|---|
| EchoJEPA-G (best) | 1B (ViT-G) | Proprietary 18.1M echo videos | `pt-280-an81.pt` | **No** |
| EchoJEPA-L (paper, public release) | 300M (ViT-L) | MIMIC-IV-Echo 525K | `vitl-vmix22m-pt220-c55.pt` | **Yes** ← we use this |
| — | 300M (ViT-L) | MIMIC-IV-Echo 525K | `vitl-scratch-pt-210-c25.pt` | **Yes** |

`vitl-vmix22m-pt220-c55.pt` is the paper-reported public EchoJEPA-L backbone (MIMIC-IV-Echo, V-JEPA 2 init), used both for the paper's reproducibility release and for our run.

> **Note:** The original repo configs (`echonet_dynamic_lvef.yaml`, `echojepa_large_lvef.yaml`) reference a SageMaker-internal checkpoint `checkpoints/anneal/keep/vitl-pt-210-an25.pt`, which was likely the backbone used during the authors' internal experiments. Its exact training data is unknown (name lacks the "mimic" prefix present in all other MIMIC checkpoints). It is not publicly available.

**Paper-reported LVEF MAE on EchoNet-Dynamic (Stanford site, cross-site evaluation):**
- EchoJEPA-L (ViT-L): **5.76 EF%**
- EchoJEPA-G (ViT-G): **3.97 EF%**

**Our run** using `vitl-vmix22m-pt220-c55.pt` on EchoNet-Dynamic (in-distribution evaluation): **5.08 EF% MAE**

---

## Probe Checkpoints

Pre-trained probe checkpoints are **not publicly available** — all referenced paths in the inference configs point to private SageMaker EFS paths (`/home/sagemaker-user/user-default-efs/...`).

Our trained probe (EchoNet-Dynamic LVEF regression):
- Path: `experiments/echonet_lvef/video_classification_frozen/echojepa-vitl-echonet-dynamic-lvef/best.pt`
- Best val MAE: 4.752 EF% (epoch 17 of 20)
- Test MAE: 5.08 EF% (1,248 / 1,277 samples)
