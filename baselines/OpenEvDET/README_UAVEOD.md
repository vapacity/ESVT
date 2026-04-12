# OpenEvDET/MvHeatDET on UAV-EOD Dataset

## 改造完成

已将 OpenEvDET (CVPR 2025) 的 MvHeatDET 方法适配到 UAV-EOD 数据集。

---

## 改动文件

### 新增文件
1. `src/data/uaveod/uaveod_dataset.py` - UAV-EOD 数据集类
2. `src/data/uaveod/__init__.py` - 模块初始化
3. `configs/dataset/UAVEOD_detection.yml` - 数据集配置
4. `configs/evheat/MvHeatDET_UAVEOD.yml` - 主配置文件
5. `train_uaveod.sh` - 训练脚本

### 修改文件
1. `src/data/__init__.py` - 添加 uaveod 模块导入

---

## 核心功能

### 1. 数据加载
- **输入格式**: 原始事件流 (NPY) + APS frames (PNG)
- **事件表示**: 实时转换为 voxel grid (3 bins, bilinear interpolation)
- **支持模式**:
  - `use_aps=False`: 使用 voxel grid 作为输入 (默认)
  - `use_aps=True`: 使用 APS frames 作为输入

### 2. 数据集结构支持
支持 UAV-EOD 的嵌套结构：
```
train/
  images/scene1/subdataset1/*.png
  events/scene1/subdataset1/*.npy
  labels/scene1/subdataset1/*.json
```

### 3. 类别映射
UAV-EOD 5类 → MvHeatDET:
```python
0: 'car'
1: 'two-wheel'
2: 'pedestrian'
3: 'bus'
4: 'truck'
```

---

## 使用步骤

### Step 1: 修改数据路径

编辑 `configs/dataset/UAVEOD_detection.yml`，修改以下路径：

```yaml
train_dataloader:
  dataset:
    img_folder: /your/path/to/UAV-EOD/train/images
    event_folder: /your/path/to/UAV-EOD/train/events
    ann_folder: /your/path/to/UAV-EOD/train/labels

val_dataloader:
  dataset:
    img_folder: /your/path/to/UAV-EOD/val/images
    event_folder: /your/path/to/UAV-EOD/val/events
    ann_folder: /your/path/to/UAV-EOD/val/labels
```

### Step 2: 训练

```bash
cd /Users/zwj/Documents/毕设/OpenEvDET/MvHeatDET
bash train_uaveod.sh
```

或者手动运行：

```bash
python tools/train.py \
    -c configs/evheat/MvHeatDET_UAVEOD.yml \
    --use-amp \
    --seed 0
```

### Step 3: 调整超参数（可选）

编辑 `configs/evheat/include/dataloader.yml`:
- `batch_size`: 默认 8，根据显存调整
- `num_workers`: 默认 4，根据 CPU 核心数调整

编辑 `configs/evheat/include/optimizer.yml`:
- 学习率、weight decay 等

---

## 关键参数

### 数据集参数 (UAVEOD_detection.yml)
```yaml
num_bins: 3           # Voxel grid 时间分辨率
use_aps: False        # False=voxel grid, True=APS frames
```

### 模型参数 (mvheatdet.yml)
```yaml
num_classes: 5        # UAV-EOD 类别数
img_size: 640         # 输入尺寸
num_queries: 100      # DETR queries
```

---

## 与 ESVT 的对比

| 特性 | ESVT | MvHeatDET |
|------|------|-----------|
| Backbone | ResNet + HybridEncoder | MvHeat_DET (Swin-like) |
| 时序模块 | LSTM/HAST | 无（单帧） |
| 频域处理 | 无 | DCT/FFT/Haar MoE |
| Decoder | RT-DETR v2 | RT-DETR |
| 输入 | Voxel grid | Voxel grid / APS |

---

## 预期性能

根据 CVPR 2025 论文，MvHeatDET 在 EvDET200K 上达到 SOTA。

在 UAV-EOD 上的预期：
- **优势**: 更强的 backbone，频域多尺度特征
- **劣势**: 无时序建模（单帧检测）

---

## 调试

如果遇到问题，检查：

1. **数据路径**: 确保 `img_folder`, `event_folder`, `ann_folder` 正确
2. **类别数**: 确认 `num_classes=5`
3. **数据加载**: 运行时会打印 `[UAVEODDetection] Loaded N samples`
4. **显存**: batch_size=8 约需 12-16GB 显存

---

## 下一步

训练完成后，在 `temp/experiment_results_4.11.md` 中添加 MvHeatDET 的结果，与 HAST-highonly 对比。
