# RT-DETR Baseline 测试指南

## 📋 概述

本指南说明如何在ESVT代码库中运行RT-DETR baseline测试。根据论文，**baseline = RTDETR = 只使用ResNet + HybridEncoder（不含时序模块） + RTDETRv2 Decoder**。

## 🎯 Baseline架构

Baseline模型组成（来自论文Table II）：
- ✅ Backbone: ResNet18/34/50
- ✅ Encoder: HybridEncoder（包含FPN + PAN + Transformer，但**不使用**ConvLSTM）
- ✅ Decoder: RTDETRv2 Decoder
- ❌ **不包含**：BFPN（双向特征融合）
- ❌ **不包含**：ConvLSTM（时序模块）
- ❌ **不包含**：RevNorm（可逆归一化）

## 🚀 快速开始

### 1. 训练Baseline模型

```bash
cd /Users/zwj/Documents/毕设/ESVT

# 使用ResNet18 baseline训练（对应论文中24.11M参数的baseline）
python train_baseline.py \
    --backbone resnet18 \
    --baseline_mode True \
    --streaming_type none \
    --dataset_path /Users/zwj/Documents/毕设/EMRS-BAIDU \
    --output_dir outputs/baseline_rtdetr_r18/ \
    --batch_size 16 \
    --epoches 72 \
    --device cuda

# 使用ResNet50 baseline训练
python train_baseline.py \
    --backbone resnet50 \
    --baseline_mode True \
    --streaming_type none \
    --dataset_path /Users/zwj/Documents/毕设/EMRS-BAIDU \
    --output_dir outputs/baseline_rtdetr_r50/ \
    --batch_size 8 \
    --epoches 72 \
    --device cuda
```

### 2. 评估Baseline模型

```bash
# 评估训练好的baseline模型
python train_baseline.py \
    --backbone resnet18 \
    --baseline_mode True \
    --streaming_type none \
    --dataset_path /Users/zwj/Documents/毕设/EMRS-BAIDU \
    --test_only True \
    --resume outputs/baseline_rtdetr_r18/checkpoint_best.pth \
    --device cuda
```

## 📊 预期性能（来自论文Table II）

| Model | BFPN | ConvLSTM | RevNorm | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 | Params(M) |
|-------|------|----------|---------|--------------|---------|----------|-----------|
| **Baseline (RT-DETR)** | ❌ | ❌ | ❌ | **33.9%** | **51.0%** | **35.9%** | **24.11** |
| ESVT (Full) | ✅ | ✅ | ✅ | 37.2% | 54.1% | 40.1% | 30.28 |

提升：**+3.3%** mAP@0.5:0.95, **+3.1%** mAP@0.5, **+4.2%** mAP@0.75

## 🔧 关键配置参数

### Baseline模式关键参数：

```python
# 在train_baseline.py中
--baseline_mode True          # 🔥 启用baseline模式，禁用所有时序模块
--streaming_type none         # 🔥 不使用streaming模块（lstm/stc）
--backbone resnet18           # 使用ResNet18（论文baseline使用）
--transformer_scale hybrid_transformer_L  # 使用L scale（hidden_dim=256）
```

### 与ESVT完整模型的对比：

| 参数 | Baseline | ESVT Full |
|------|----------|-----------|
| `baseline_mode` | `True` | `False` |
| `streaming_type` | `none` | `lstm` or `stc` |
| 采样策略 | `RandomSampler` | `StreamingSampler` |
| 时序融合 | ❌ 禁用 | ✅ 启用 |

## 📁 代码修改说明

### 1. `train_baseline.py`
- 新增 `--baseline_mode` 参数
- Baseline模式使用 `RandomSampler` 代替 `StreamingSampler`
- 输出目录默认为 `outputs/baseline_rtdetr/`

### 2. `models/ESVT/encoder/hybrid_encoder.py`
- 新增 `baseline_mode` 参数
- 当 `baseline_mode=True` 或 `streaming_type='none'` 时：
  - `self.stm = None`（不初始化ConvLSTM）
  - `forward()` 中跳过时序融合逻辑

### 3. `models/ESVT/__init__.py`
- 在 `build_ESVT()` 中检测 `baseline_mode`
- 自动将 `streaming_type` 设置为 `'none'`

## 🔍 验证Baseline配置

运行以下命令验证模型是否正确配置为baseline模式：

```python
import argparse
import sys
sys.path.append('/Users/zwj/Documents/毕设/ESVT')
from models import build_model

# 创建args
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ESVT')
parser.add_argument('--model_type', default='event')
parser.add_argument('--backbone', default='resnet18')
parser.add_argument('--backbone_pretrained', default=False)
parser.add_argument('--transformer_scale', default='hybrid_transformer_L')
parser.add_argument('--streaming_type', default='none')
parser.add_argument('--baseline_mode', default=True)
parser.add_argument('--dataset', default='UAV-EOD')
args = parser.parse_args([])

# 构建模型
model = build_model(args)

# 检查encoder是否禁用了streaming模块
print(f"Baseline mode: {model.encoder.baseline_mode}")
print(f"Streaming module (should be None): {model.encoder.stm}")
print(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
```

预期输出：
```
Baseline mode: True
Streaming module (should be None): None
Total params: ~24.11M
```

## 🆚 Baseline vs ESVT 对比

### 架构对比：

```
Baseline (RT-DETR):
Input → Backbone (ResNet) → HybridEncoder (FPN+PAN+Transformer) → Decoder → Output
                                          ↑
                                    无时序融合

ESVT (Full):
Input → Backbone (ResNet) → SBFPN (ConvLSTM + BFPN + RevNorm) → Decoder → Output
                                    ↑
                              有时序融合 + 双向特征金字塔
```

### 训练策略对比：

| 策略 | Baseline | ESVT |
|------|----------|------|
| 采样方式 | 随机采样（Random） | 流式采样（Streaming） |
| 帧间关联 | ❌ 独立检测 | ✅ 时序关联 |
| 长期遮挡处理 | ❌ 无法处理 | ✅ 可以处理 |
| 失焦恢复 | ❌ 无法处理 | ✅ 可以处理 |

## 📝 注意事项

1. **数据集路径**：确保 `--dataset_path` 指向正确的数据集目录
   ```
   /Users/zwj/Documents/毕设/EMRS-BAIDU/
   ├── train/
   ├── val/
   └── annotations/
   ```

2. **Batch Size**：
   - ResNet18: 建议 batch_size=16
   - ResNet50: 建议 batch_size=8（显存较大时可增加）

3. **训练时长**：
   - 论文使用72 epochs
   - Learning rate drop at epoch 70

4. **对比实验**：
   - 先训练baseline获得基准性能
   - 再训练完整ESVT模型
   - 对比两者性能差异验证改进有效性

## 📊 结果分析

训练完成后，对比以下指标：

```bash
# Baseline结果（预期）
mAP@0.5:0.95: ~33.9%
mAP@0.5: ~51.0%
mAP@0.75: ~35.9%

# ESVT结果（预期）
mAP@0.5:0.95: ~37.2%
mAP@0.5: ~54.1%
mAP@0.75: ~40.1%

# 提升
Δ mAP@0.5:0.95: +3.3%
Δ mAP@0.5: +3.1%
Δ mAP@0.75: +4.2%
```

## 🐛 常见问题

### Q1: 如何确认运行的是baseline而不是完整ESVT？
A: 检查训练日志开头是否显示：
```
🎯 Running RT-DETR Baseline Mode
Baseline Mode: True
Streaming Type: none
```

### Q2: Baseline和ESVT可以使用相同的数据集吗？
A: 可以。但注意：
- Baseline使用 `RandomSampler`（随机采样）
- ESVT使用 `StreamingSampler`（流式采样）

### Q3: 参数量不匹配怎么办？
A: 论文baseline参数量为24.11M，如果差异较大，检查：
- Backbone是否为ResNet18
- `hidden_dim` 是否为256（L scale）
- `streaming_type` 是否为 `'none'`

## 📧 联系方式

如有问题，请检查：
1. 论文Table II中的baseline配置
2. `train_baseline.py` 中的默认参数
3. 训练日志中的模型配置信息
