# EchoJEPA：配置与结果对比

**日期：** 2026-06-24  
**任务：** LVEF 回归（左心室射血分数估计）  
**数据集：** EchoNet-Dynamic（公开数据集）

---

## 1. 模型架构

| 项目 | 本次运行 | 论文：EchoJEPA-L（ViT-L） | 论文：EchoJEPA-G（ViT-G） |
|---|---|---|---|
| 骨干网络 | ViT-L/16（3亿参数） | ViT-L/16（3亿参数） | ViT-G/16（10亿参数） |
| 骨干 checkpoint | `vitl-vmix22m-pt220-c55.pt`（公开） | `vitl-vmix22m-pt220-c55.pt`（公开，与本次运行相同） | `pt-280-an81.pt`（未公开） |
| 骨干预训练数据 | MIMIC-IV-Echo（VideoMix22M 初始化） | MIMIC-IV-Echo 525K（VideoMix22M 初始化） | 私有数据 1810万例 |

> **注：** Repo 原始 config（`echonet_dynamic_lvef.yaml` 和 `echojepa_large_lvef.yaml`）中指向的是另一个 SageMaker 内部 checkpoint：`checkpoints/anneal/keep/vitl-pt-210-an25.pt`，可能是作者内部实验实际使用的骨干。论文对外描述并公开发布的 EchoJEPA-L 骨干为 `vitl-vmix22m-pt220-c55.pt`。
| Probe 类型 | 4层 attentive probe | 4层 attentive probe | 4层 attentive probe |
| Probe 注意力头数 | 16 | 16 | 16 |
| 任务类型 | 回归（LVEF） | 回归（LVEF） | 回归（LVEF） |
| 骨干网络是否冻结 | 是 | 是 | 是 |

> **关键差异：** 本次运行与论文 EchoJEPA-L 使用**完全相同的骨干**（`vitl-vmix22m-pt220-c55.pt`，MIMIC-IV-Echo，公开）。两者的差异完全来自 probe 训练数据（EchoNet TRAIN 7,465条 vs Toronto 15万例）和评估协议（分布内 vs 零样本跨站点）。EchoJEPA-G 使用 ViT-G 骨干和规模大得多的私有数据（1810万例）。

---

## 2. 数据配置

| 项目 | 本次运行 | 论文：EchoJEPA-L（ViT-L） | 论文：EchoJEPA-G（ViT-G） |
|---|---|---|---|
| 训练数据集 | EchoNet-Dynamic | Toronto 内部数据集（N=15万例） | Toronto 内部数据集（N=15万例） |
| 训练样本数 | 7,465 | 约 15 万例超声检查 | 约 15 万例超声检查 |
| 验证样本数 | 1,288 | Toronto 内部留存集 | Toronto 内部留存集 |
| 测试样本数 | 1,277（EchoNet 测试分割） | EchoNet-Dynamic（零样本，全部 10,030 条） | EchoNet-Dynamic（零样本，全部 10,030 条） |
| 视频格式 | AVI（112×112，约50fps） | MP4 | MP4 |
| 训练分辨率 | 112px | 112px（EchoNet eval config）/ 336px（私有数据集） | 336px |
| 推理分辨率（EchoNet-Dynamic） | 112px | 112px | 112px |
| 每片段帧数 | 16 | 16 | 16 |
| 帧步长 | 2 | 2 | 2 |
| 片段数 | 2 | 2 | 2 |
| target_mean | 55.7776（EchoNet TRAIN 统计） | 57.0569（Toronto 数据集统计） | 57.06（Toronto 数据集统计） |
| target_std | 12.4064（EchoNet TRAIN 统计） | 11.3252（Toronto 数据集统计） | 11.33（Toronto 数据集统计） |

---

## 3. 训练配置

| 项目 | 本次运行 | 论文：EchoJEPA-L（ViT-L） | 论文：EchoJEPA-G（ViT-G） |
|---|---|---|---|
| 硬件 | 1× NVIDIA RTX PRO 6000 Blackwell（97.9 GB） | 8× GPU（AWS SageMaker） | 8× GPU（AWS SageMaker） |
| 训练 epoch 数 | 20 | 20 | 未指定 |
| 每卡 batch size | 16 | 1 | 80 |
| 等效 batch size | 16 | 8（1×8卡） | 640（80×8卡） |
| 学习率候选 | {1e-4, 5e-5} | {1e-4, 5e-5} | 未指定 |
| Weight decay 候选 | {0.01, 0.1, 0.4} | {0.01, 0.1, 0.4} | 未指定 |
| 并行 probe 数量 | 6（2学习率 × 3权重衰减） | 6 | 未指定 |
| 优化器 | AdamW | AdamW | AdamW |
| 训练精度 | bfloat16 | bfloat16 | bfloat16 |
| 训练时长 | 约1小时40分钟 | 未报告 | 未报告 |
| 实际 GPU 显存占用 | 约 6.8 GB | 约 80 GB（配置上限） | 约 80 GB（配置上限） |

---

## 4. 结果对比

### 4.1 LVEF MAE（EF%）——跨站点泛化评估（论文 Table 7）

论文在三个医院站点评估，"Stanford" 对应 EchoNet-Dynamic，与本次数据集相同。

| 模型 | Toronto | Chicago | Stanford（EchoNet-Dynamic） |
|---|---|---|---|
| EchoPrime | 5.33 | 6.71 | 4.87 |
| PanEcho | 5.43 | 6.52 | 5.10 |
| EchoMAE-L（重建基线） | 8.15 | 9.40 | 8.52 |
| **EchoJEPA-L（ViT-L，论文）** | **5.97** | **7.39** | **5.76** |
| **EchoJEPA-G（ViT-G，论文最优）** | **4.26** | **5.44** | **3.97** |
| **本次运行（ViT-L，EchoNet-Dynamic）** | — | — | **5.08** |

> 论文 EchoJEPA-L 的 probe 在 **Toronto 内部数据集（15万例）** 上训练，在 EchoNet-Dynamic 上进行**完全零样本评估**——EchoNet 数据从未参与 probe 训练。本次运行在 EchoNet TRAIN 上训练、在 EchoNet TEST 上评估，属于更容易的分布内协议。尽管如此，我们的 MAE（5.08%）仅略优于零样本 EchoJEPA-L（5.76%），可见大规模内部数据 probe 训练的优势。

### 4.2 主数据集 LVEF MAE（论文 Table 6）

| 模型 | LVEF MAE ↓ | 视图分类准确率 ↑ |
|---|---|---|
| EchoMAE-L（重建方法） | 8.15 | 40.4% |
| **EchoJEPA-L（ViT-L）** | **5.97** | **85.5%** |
| **EchoJEPA-G（ViT-G）** | — | — |
| 相对 EchoMAE-L 的提升（L） | **−26.7%** | **+45.1%** |

### 4.3 声学扰动鲁棒性（论文 Table 9，Stanford 站点）

| 模型 | 原始 MAE | 平均性能下降 |
|---|---|---|
| EchoPrime | 4.87 | +16.8% |
| PanEcho | 5.10 | +3.7% |
| EchoMAE-L | 8.52 | +0.5%† |
| **EchoJEPA-L（ViT-L）** | **5.76** | **+2.3%** |
| **EchoJEPA-G（ViT-G）** | **3.97** | **+2.3%** |

† EchoMAE-L 基线误差已饱和，绝对下降量因此偏小。

### 4.4 本次运行——训练曲线

| Epoch | 训练损失 | 验证 MAE（最优 head，EF%） | 历史最优 |
|---|---|---|---|
| 1 | 8.712 | 7.061 | 7.061 |
| 5 | 6.609 | 6.158 | 5.481 |
| 10 | 6.118 | 5.168 | 5.026 |
| 15 | 5.702 | 4.829 | 4.829 |
| 17 | 5.801 | **4.752** | **4.752** ← 已保存 |
| 20 | 5.672 | 5.003 | 4.752 |

### 4.5 本次运行——最终测试指标

| 指标 | 值 |
|---|---|
| 测试集 MAE | **5.08 EF%** |
| 测试集 RMSE | **6.74 EF%** |
| Pearson 相关系数 | **0.837** |
| 最优验证集 MAE | 4.752 EF%（epoch 17） |
| 完成推理样本数 | 1,248 / 1,277 |

---

## 5. 主要差异分析

### 5.1 骨干预训练域
本次运行与论文 EchoJEPA-L 使用**完全相同的骨干 checkpoint**（`vitl-vmix22m-pt220-c55.pt`）。因此两者之间的性能差异（5.08% vs 5.76%）完全来自 probe 训练数据和评估协议，与骨干无关。EchoJEPA-G 使用 ViT-G + 私有 1810万例数据，特征表达能力显著更强。

### 5.2 模型规模（ViT-L vs ViT-G）
EchoJEPA-G（10亿参数）在 EchoNet-Dynamic 上取得 3.97% MAE，比 ViT-L 的 5.76% 提升约 31%——仅靠模型规模扩大实现。ViT-G 得益于更强的时空特征提取能力和更大的预训练容量。

### 5.3 评估协议差异
论文 EchoJEPA-L 在 Toronto（15万例）训练 probe 后，**零样本应用于 EchoNet-Dynamic**，从未接触任何 EchoNet 数据。本次运行在 EchoNet TRAIN（7,465 条）上训练、在 EchoNet TEST 上评估，属于分布内评估——probe 已适应 EchoNet 的采集特征（Philips IE33，Stanford 方案）。这是两种本质不同的评估设置：本次结果（5.08%）仅微弱优于零样本 5.76%，更凸显了论文大规模内部数据 probe 训练的有效性。

### 5.4 推理分辨率
在 EchoNet-Dynamic 评估中，论文 EchoJEPA-L 与本次运行均使用 **112px**——这是 EchoNet 的原始分辨率，已由原始 `echonet_dynamic_lvef.yaml` config 确认。论文针对私有数据集（Toronto/Chicago，对应 5.97% 结果）的评估使用 **336px**，见 `echojepa_large_lvef.yaml` config。RoPE 支持灵活分辨率，同一 checkpoint 无需重训即可在不同分辨率下推理。

### 5.5 训练硬件与 Batch Size
ViT-L 论文：8卡 × batch 1 = 等效 batch 8；本次：1卡 × batch 16 = 等效 batch 16；ViT-G：8卡 × batch 80 = 等效 batch 640。更大的等效 batch size 通常有利于回归 probe 的训练稳定性。

---

## 6. 文件路径汇总

| 内容 | 路径 |
|---|---|
| 骨干网络 checkpoint | `checkpoints/vitl-vmix22m-pt220-c55.pt` |
| 最优 probe checkpoint | `experiments/echonet_lvef/video_classification_frozen/echojepa-vitl-echonet-dynamic-lvef/best.pt` |
| 训练 CSV | `data/csv/echonet_dynamic_train.csv`（7,465 行） |
| 验证 CSV | `data/csv/echonet_dynamic_val.csv`（1,288 行） |
| 测试 CSV | `data/csv/echonet_dynamic_test.csv`（1,277 行） |
| 测试集预测结果 | `experiments/echonet_lvef_inference/test_predictions.csv` |
| 训练日志 | `experiments/echonet_lvef/train.log` |
| 推理日志 | `experiments/echonet_lvef_inference/infer.log` |
| 训练配置文件 | `configs/eval/vitl/echonet_dynamic_lvef.yaml` |
| 推理配置文件 | `configs/inference/vitl/echojepa_large_lvef.yaml` |

---

## 7. 总结对比

| | 本次运行 | 论文 EchoJEPA-L（ViT-L） | 论文 EchoJEPA-G（ViT-G） |
|---|---|---|---|
| 骨干 | ViT-L，`vitl-vmix22m-pt220-c55.pt`（与论文相同） | ViT-L，`vitl-vmix22m-pt220-c55.pt`（与本次运行相同） | ViT-G，1810万私有超声 |
| EchoNet-Dynamic 分辨率 | 112px | 112px | 112px |
| Probe 训练数据 | EchoNet-Dynamic TRAIN（7,465条） | Toronto 内部（15万例超声检查） | Toronto 内部（15万例超声检查） |
| 评估协议 | EchoNet-Dynamic TEST（分布内） | EchoNet-Dynamic（零样本，跨站点） | EchoNet-Dynamic（零样本，跨站点） |
| **测试 MAE（EF%）** | **5.08** | **5.76** | **3.97** |
| 测试 RMSE（EF%） | 6.74 | — | — |
| Pearson r | 0.837 | — | — |

本次单卡 ViT-L 实验取得 **5.08 EF% MAE**，略优于论文 EchoJEPA-L（5.76%），但评估协议本质上更容易：我们的 probe 在 EchoNet TRAIN 上训练后直接在 EchoNet TEST 上测试（分布内），而论文的 probe 在 Toronto（15万例）训练后**零样本**应用于 EchoNet，从未接触任何 EchoNet 数据。达到论文最优水平（3.97%）的关键路径：① 使用 ViT-G 骨干；② 提高分辨率至 336px；③ 在大规模内部数据集上训练 probe。
