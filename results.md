# ESVT 实验结果汇总

## 实验配置

- backbone: ResNet-18
- decoder: RTDETRv2
- epochs: 72
- dataset: RSEOD (UAV-EOD)

---

## 结果对比

| 指标 | none (baseline) | lstm | lstm_true (残差) |
|------|:-:|:-:|:-:|
| **训练脚本** | train_baseline.py | train.py | train.py |
| **Sampler** | RandomSampler | StreamingSampler | StreamingSampler |
| **ConvLSTM** | ✗ | ✓ (输出丢弃) | ✓ (残差融合) |
| AP@0.50:0.95 | 0.451 | 0.413 | **0.429** |
| AP@0.50 | 0.576 | 0.534 | **0.550** |
| AP@0.75 | 0.495 | 0.466 | **0.496** |
| AP small | 0.221 | 0.185 | **0.232** |
| AP medium | 0.801 | 0.747 | **0.758** |
| AP large | 0.906 | **0.919** | 0.828 |
| AR@100 | 0.532 | 0.517 | **0.492** |
| AR small | 0.404 | 0.409 | **0.383** |
| AR medium | 0.872 | 0.855 | **0.824** |
| AR large | 0.938 | **0.945** | 0.858 |
| train_loss (ep71) | - | 13.441 | **12.125** |
| train_loss_vfl | - | 0.402 | **0.394** |
| train_loss_bbox | - | 0.116 | **0.101** |
| train_loss_giou | - | 0.442 | **0.405** |

> AR 数据来自 `test_coco_eval_bbox` 数组索引 6-11。

---

## 重要说明：对比不公平

`none` 与 `lstm`/`lstm_true` 使用了不同的训练 sampler，**结果不可直接比较**：

- `none` 使用 **RandomSampler**：完全随机采样，数据多样性高，有利于检测性能
- `lstm`/`lstm_true` 使用 **StreamingSampler**：按时序顺序采样，相邻 batch 高度相关，数据多样性低

StreamingSampler 是为保证 ConvLSTM 时序状态连续性而设计的，但同时降低了训练数据多样性。

**待做：用 StreamingSampler 重跑 `none`，建立公平 baseline。**

---

## Bug 修复记录

### 1. 原始代码 ConvLSTM 输出被丢弃 (hybrid_encoder.py)

```python
# 原始代码 (bug): LSTM 运行但输出被覆盖
stm_feats, status = list(proj_feats), list(status)

# 修复后 (lstm_true): 残差融合
stm_feats = [p + s for p, s in zip(proj_feats, stm_feats)]
```

### 2. 原始代码 status 未初始化 (engine.py)

```python
# 原始代码: 第一个 iteration 访问未定义的 status → NameError
pre_status = [(state[0].detach(), state[1].detach()) for state in status]

# 修复后
status = None  # 在循环外初始化
if indexes[-1][-1] % 100 == 0 or status is None:
    pre_status = None
```

### 3. lstm_true 直接替换特征导致 AP=0

```python
# 问题: LSTM 随机初始化输出完全替换 backbone 特征 → 模型无法检测
stm_feats = list(stm_feats)

# 修复: 残差连接，训练初期等价于 none，逐渐学习时序贡献
stm_feats = [p + s for p, s in zip(proj_feats, stm_feats)]
```

---

## 待做实验

- [ ] `none` + StreamingSampler（公平 baseline）
- [ ] `lstm_true` 可学习缩放因子（`proj + tanh(scale) * stm`）
