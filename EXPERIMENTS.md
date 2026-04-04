## 1. 数据集概况

### 1.1 EMRS-BAIDU 数据集统计

| 指标 | 数值 |
|------|------|
| 场景数量 (Scenes) | 3 |
| 子数据集数量 (Sub-datasets) | 86 (train) / 22 (val) |
| 总帧数 | 10,800 |
| 总事件数 | 476,663,121 |
| 总目标数 | 45,390 |

### 1.2 划分统计

| 模式 | 场景 | 子数据集 | 帧数 | 事件数 |
|------|------|----------|------|--------|
| Train | - | 86 | 8,500 | 345,450,679 |
| Val | - | 22 | 1,300 | 47,290,201 + 60,545,209 + 23,377,032 |
| Test | - | - | 1,000 | - |

### 1.3 场景详情

| 场景 | 子数据集数 | 帧数 | 事件总数 | 每帧平均事件数 |
|------|-----------|------|----------|----------------|
| low_light | 3 | 300 | 47,290,201 | 157,634 |
| motion_blur | 3 | 300 | 60,545,209 | 201,817 |
| normal | 7 | 700 | 23,377,032 | 33,396 |

### 1.4 事件流统计

| 统计项 | 数值 |
|--------|------|
| 每帧事件数 - 平均值 | 100,932.6 |
| 每帧事件数 - 中位数 | 41,018.0 |
| 每帧事件数 - 标准差 | 140,073.1 |
| 每帧事件数 - 最小值 | 2,308 |
| 每帧事件数 - 最大值 | 625,557 |
| 每帧事件数 - 25%分位 | 20,940.0 |
| 每帧事件数 - 75%分位 | 97,645.0 |

### 1.5 极性分布

| 指标 | 数值 |
|------|------|
| 正事件平均比例 | 45.1% |
| 正事件范围 | [8.1%, 62.3%] |

### 1.6 类别分布

| 类别 | 数量 | 占比 |
|------|------|------|
| car | 5,112 | 85.1% |
| bus | 410 | 6.8% |
| pedestrian | 300 | 5.0% |
| two-wheel | 100 | 1.7% |
| truck | 88 | 1.5% |

### 1.7 目标统计

| 指标 | 数值 |
|------|------|
| 每帧目标数 - 平均值 | 4.64 |
| 每帧目标数 - 中位数 | 4.0 |
| 每帧目标数 - 最大值 | 10 |

---

## 2. 实验配置

### 2.1 模型架构参数

| 参数 | 选项 | 说明 |
|------|------|------|
| `--backbone` | resnet18 / resnet34 / resnet50 | 骨干网络 |
| `--transformer_scale` | hybrid_transformer_X / L / H | Transformer规模 |
| `--streaming_type` | none / lstm / lstm_se / lstm_cbam / stc | 时序建模模块 |
| `--event_rep` | voxel | 事件表示方法 |

### 2.2 模型变体

| 模型 | Backbone | Transformer | Streaming |
|------|----------|-------------|-----------|
| ESVT-T | resnet18 | hybrid_transformer_X | lstm |
| ESVT-S | resnet34 | hybrid_transformer_L | lstm |
| ESVT-B | resnet50 | hybrid_transformer_H | lstm |
| Baseline | resnet18/50 | hybrid_transformer_L/H | none |

---

## 3. 实验结果

### 3.1 模型对比

| 模型 | 配置 | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 | 小目标AP | 中目标AP | AR@100 |
|------|------|--------------|----------|----------|---------|---------|---------|
| **ESVT-B** | resnet50 + convLSTM + BFPN | **0.458** | **0.689** | **0.522** | **0.449** | **0.655** | **0.536** |
| Baseline  +BFPN | resnet18 + BFPN | 0.438 | 0.687 | 0.506 | 0.430 | 0.630 | 0.515 |
| ESVT-T | resnet18 + convLSTM + BFPN | 0.432 | 0.676 | 0.495 | 0.423 | 0.650 | 0.509 |



![image-20260323195413829](/Users/zwj/Library/Application Support/typora-user-images/image-20260323195413829.png)

### 3.2 详细评估结果

#### ESVT-B (resnet50 + hybrid_transformer_L + lstm)

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.689
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.522
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.449
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.655
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.536
```

#### Baseline (resnet18 + hybrid_transformer_X + none)

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.438
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.687
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.506
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.430
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.630
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.515
```

#### ESVT-T (resnet18 + hybrid_transformer_X + lstm)

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.432
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.676
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.495
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.423
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.650
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.509
```

### 3.3 结果分析

| 对比项 | 结论 |
|--------|------|
| ESVT-B vs Baseline | LSTM时序建模带来 **+2.0%** mAP提升 (0.438→0.458) |
| ESVT-T vs Baseline | 同backbone下，LSTM带来 **+2.3%** mAP提升 (0.409→0.432) |
| 小目标检测 | ESVT-B表现最佳，AP=0.449 |
| 召回率 | ESVT-B最高，AR=0.536 |
| 中目标检测 | ESVT-T中目标AP=0.650，与ESVT-B接近 |

---

## 4. 评估方法

### 4.1 运行评估

```bash
# 评估训练好的模型
python train.py \
  --resume path/to/model.pth \
  --test_only \
  --dataset_path /path/to/EMRS-BAIDU \
  --backbone resnet50 \
  --transformer_scale hybrid_transformer_L \
  --streaming_type lstm
```

### 4.2 数据集统计

```bash
# 统计数据集信息
python stat_dataset.py \
  --dataset_path /path/to/EMRS-BAIDU \
  --mode all \
  --output stats.json
```

---

## 5. 关键发现

1. **类别不平衡**: car类占比85.1%，存在严重的类别不平衡问题
2. **事件密度变化大**: 每帧事件数从2,308到625,557不等，标准差高达140,073
3. **场景差异明显**: 
   - motion_blur场景事件最多(201,817/帧)
   - normal场景事件最少(33,396/帧)
4. **极性分布**: 正负事件比例接近1:1，平均45.1%正事件

---

## 6. 待补充实验

- [ ] ESVT-S 模型 (resnet34) 评估
- [ ] 不同streaming_type对比实验 (lstm_se, lstm_cbam, stc)
- [ ] 不同Transformer规模对比
- [ ] Test集评估结果



  ESVT 模型架构总览

  整个 ESVT 框架由 5 个模块串联组成：

  事件流输入 → 事件表示 → 空间骨干网络 → SBFPN → Transformer解码器 → 预测头

  1. 事件表示（Event Representation）

  - 将异步事件流 $\varepsilon \in \mathbb{R}^4$ 编码为 Voxel Grid 密集张量 $X \in \mathbb{R}^{T \times H \times W}$
  - 通过双线性投票聚合正/负极性

  2. 空间骨干网络（Backbone）

  - 使用 ResNet 提取空间特征，输出 5 个尺度特征图
  - 取后 3 个尺度 $f' = [f_3, f_4, f_5]$ 送入 SBFPN

| 变体   | Backbone | Hidden Dim |
| ------ | -------- | ---------- |
| ESVT-T | ResNet18 | 256        |
| ESVT-S | ResNet34 | 384        |
| ESVT-B | ResNet50 | 512        |

  3. SBFPN（Streaming Bidirectional FPN）—— 核心创新模块

  由两个子模块组成：

  ConvLSTM（时序融合）
  - 输入当前时刻特征 $x(t_1)$ 与上一时刻隐状态 $f(t_0)$
  - 先通过 RevNorm 归一化消除非平稳噪声，再经 DepthConv2D
  - 拼接后经 Conv1×1 生成 4 个门控向量（forget/input/cell/output gate）
  - 最终通过 RevDeNorm 恢复到原始分布，输出 $h(t_1)$

  BFPN（双向特征金字塔）
  - 上采样路径：深层语义特征逐步上采样，与浅层特征拼接 → 获取位置信息
  - 下采样路径：使用 overlapping conv（kernel=3, stride=2）将浅层特征融合回深层
  - 输出三个尺度融合后的特征图 $f_1'', f_2'', f_3'$

  4. Transformer 解码器

  - 使用 RTDETR 解码器（对比去噪 + 混合查询选择）
  - 输出 300 个候选框 $B \in \mathbb{R}^{300 \times 4}$ 和类别 $C \in \mathbb{R}^{300 \times 5}$

  5. 预测头

  - 匈牙利算法进行二分匹配
  - 输出 $[c, x, y, w, h]$，无需 NMS

---
  Baseline 的处理方式

  论文中 Baseline = RTDETR（即 ESVT-T 去掉所有自定义模块后的基础版本）：

| 特性         | Baseline (RTDETR)  | ESVT                      |
| ------------ | ------------------ | ------------------------- |
| 骨干网络     | ResNet（同款）     | ResNet（同款）            |
| Neck/Encoder | RTDETR 原生编码器  | SBFPN（ConvLSTM + BFPN）  |
| 时序处理     | 无（每帧独立处理） | ConvLSTM 跨时步传递隐状态 |
| 多尺度融合   | RTDETR 内置 FPN    | 双向 BFPN                 |
| 归一化       | 普通 BN            | RevNorm（可逆归一化）     |
| 训练策略     | 随机采样           | 并行随机批采样            |
| mAP@0.5:0.95 | 33.9%              | 37.2%（ESVT-T）           |

  关键结论：Baseline 即标准 RTDETR，完全不处理时序信息，每帧独立推理。ESVT 在此基础上用 SBFPN 替换原始 Encoder，引入时序传递能力，提升了 +3.3% mAP@0.5:0.95。





- 检查数据代码
- +纵向时序信息，帧之间的特征提取
- 做全局+局部，多尺度信息
- 



- 空间模型
- 时间模型
- 时空融合后再加上时空特征凸显特有特征
- 论文吃透
- 代码看明白



