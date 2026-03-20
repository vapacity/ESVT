# 集成指南 - 如何使用增强版 ConvLSTM

## 📝 修改 `models/ESVT/__init__.py`

### Step 1: 添加导入

在文件顶部（第4行之后）添加增强版 Encoder 的导入：

```python
from models.ESVT.esvt import ESVT
from models.ESVT.backbone.hgnetv2 import HGNetv2
from models.ESVT.backbone.resnet import ResNet
from models.ESVT.encoder.hybrid_encoder import HybridEncoder
from models.ESVT.encoder.hybrid_encoder_enhanced import HybridEncoderEnhanced  # ← 新增这一行
from models.ESVT.decoder.rtdetrv2_decoder import RTDETRTransformerv2
from models.ESVT.criterion.rtdetrv2_criterion import RTDETRCriterionv2
from models.ESVT.criterion.matcher import HungarianMatcher
from models.ESVT.postprocessor.rtdetr_postprocessor import RTDETRPostProcessor
```

### Step 2: 修改 `build_ESVT()` 函数

将原来的函数（第 11-34 行）替换为：

```python
def build_ESVT(args):
    # 🔥 Check baseline mode
    baseline_mode = getattr(args, 'baseline_mode', False)
    streaming_type = args.streaming_type if not baseline_mode else 'none'

    # 🔥 新增: 根据 streaming_type 自动选择 Encoder
    # 如果是增强版变体 (lstm_se, lstm_cbam等), 使用 HybridEncoderEnhanced
    # 否则使用原版 HybridEncoder
    use_enhanced = streaming_type.startswith('lstm_') and streaming_type != 'lstm'
    EncoderClass = HybridEncoderEnhanced if use_enhanced else HybridEncoder

    if args.model_type == 'event':

        if args.backbone[:-1] == 'hgnetv2':
            return ESVT(
                backbone=HGNetv2(name=args.backbone[-1], pretrained=args.backbone_pretrained),
                encoder=EncoderClass(  # ← 改这里: 使用动态选择的 Encoder
                    name=args.transformer_scale[-1],
                    streaming_type=streaming_type,
                    baseline_mode=baseline_mode
                ),
                decoder=RTDETRTransformerv2(name=args.transformer_scale[-1], dataset=args.dataset)
            )

        elif args.backbone[:-2] == 'resnet':
            return ESVT(
                backbone=ResNet(name=args.backbone[-2:], pretrained=args.backbone_pretrained),
                encoder=EncoderClass(  # ← 改这里: 使用动态选择的 Encoder
                    backbone_name=args.backbone[-2:],
                    name=args.transformer_scale[-1],
                    streaming_type=streaming_type,
                    baseline_mode=baseline_mode
                ),
                decoder=RTDETRTransformerv2(name=args.transformer_scale[-1], dataset=args.dataset)
            )

    # TODO 还未实现其他模态, 稍后实现
    elif args.model_type == 'image':
        pass
    elif args.model_type == 'multimodel':
        pass
```

---

## 🚀 使用方法

### 方法 1: 修改代码后使用（推荐）

完成上述修改后，直接在训练命令中指定 `streaming_type`:

```bash
# Baseline (使用原版 HybridEncoder)
python train.py --streaming_type lstm

# SE 注意力 (自动使用 HybridEncoderEnhanced)
python train.py --streaming_type lstm_se

# CBAM 注意力
python train.py --streaming_type lstm_cbam

# ECA 注意力
python train.py --streaming_type lstm_eca

# Spatial 注意力
python train.py --streaming_type lstm_spatial

# SE 应用在 cell 输出后
python train.py --streaming_type lstm_se_after

# CBAM 应用在两个位置
python train.py --streaming_type lstm_cbam_both
```

### 方法 2: 不修改代码，临时测试

如果你不想修改 `models/ESVT/__init__.py`，可以创建一个测试脚本：

```python
# test_enhanced_esvt.py
import torch
from models.ESVT.esvt import ESVT
from models.ESVT.backbone.resnet import ResNet
from models.ESVT.encoder.hybrid_encoder_enhanced import HybridEncoderEnhanced
from models.ESVT.decoder.rtdetrv2_decoder import RTDETRTransformerv2

# 构建增强版模型
model = ESVT(
    backbone=ResNet(name='18', pretrained=False),
    encoder=HybridEncoderEnhanced(
        backbone_name='18',
        name='L',
        streaming_type='lstm_cbam',  # 使用 CBAM
        baseline_mode=False
    ),
    decoder=RTDETRTransformerv2(name='L', dataset='UAV-EOD')
)

# 测试前向传播
x = torch.randn(2, 5, 346, 260)  # batch=2, channels=5, H=346, W=260
output, targets, status = model(x)
print(f"Output: {output}")
print(f"Status: {status}")
```

---

## 📊 验证修改是否生效

运行以下代码验证：

```python
import argparse
from models.ESVT import build_ESVT

# 创建参数
args = argparse.Namespace(
    model='ESVT',
    model_type='event',
    backbone='resnet18',
    backbone_pretrained=False,
    transformer_scale='hybrid_transformer_L',
    streaming_type='lstm_cbam',  # 测试 CBAM
    baseline_mode=False,
    dataset='UAV-EOD'
)

# 构建模型
model = build_ESVT(args)

# 检查 encoder 类型
print(f"Encoder 类型: {type(model.encoder).__name__}")
# 应该输出: HybridEncoderEnhanced

# 检查 streaming module 类型
print(f"STM 类型: {type(model.encoder.stm).__name__}")
# 应该输出: DWSConvLSTM2d_Enhanced

# 检查注意力配置
if hasattr(model.encoder.stm, 'attention_type'):
    print(f"注意力类型: {model.encoder.stm.attention_type}")
    print(f"注意力位置: {model.encoder.stm.attention_position}")
```

**预期输出**:
```
Encoder 类型: HybridEncoderEnhanced
STM 类型: DWSConvLSTM2d_Enhanced
注意力类型: cbam
注意力位置: before_gates
```

---

## 🔄 切换不同的配置

### 配置对照表

| streaming_type | Encoder | ConvLSTM | 注意力类型 |
|----------------|---------|----------|-----------|
| `lstm` | HybridEncoder (原版) | DWSConvLSTM2d | 无 |
| `lstm_se` | HybridEncoderEnhanced | DWSConvLSTM2d_Enhanced | SE |
| `lstm_cbam` | HybridEncoderEnhanced | DWSConvLSTM2d_Enhanced | CBAM |
| `lstm_eca` | HybridEncoderEnhanced | DWSConvLSTM2d_Enhanced | ECA |
| `lstm_spatial` | HybridEncoderEnhanced | DWSConvLSTM2d_Enhanced | Spatial |
| `lstm_se_after` | HybridEncoderEnhanced | DWSConvLSTM2d_Enhanced | SE (after cell) |
| `lstm_cbam_both` | HybridEncoderEnhanced | DWSConvLSTM2d_Enhanced | CBAM (both) |

---

## ⚠️ 常见问题

### Q1: 修改后原来的实验还能跑吗？

**A**: 能！只要 `streaming_type='lstm'`（或 `'none'`），就会使用原版的 HybridEncoder，完全不受影响。

### Q2: 如何确认使用的是哪个版本？

**A**: 在训练开始时，模型会打印信息：
```python
# 在 build_ESVT() 中添加 debug 信息
print(f"[DEBUG] Using {EncoderClass.__name__} with streaming_type='{streaming_type}'")
```

### Q3: 如果不修改 `models/ESVT/__init__.py` 可以吗？

**A**: 可以，但你需要在每个训练脚本中手动构建模型（参见方法2）。推荐还是修改 `__init__.py`，一劳永逸。

---

## 📦 完整的修改后的 `models/ESVT/__init__.py`

```python
from models.ESVT.esvt import ESVT
from models.ESVT.backbone.hgnetv2 import HGNetv2
from models.ESVT.backbone.resnet import ResNet
from models.ESVT.encoder.hybrid_encoder import HybridEncoder
from models.ESVT.encoder.hybrid_encoder_enhanced import HybridEncoderEnhanced  # ← 新增
from models.ESVT.decoder.rtdetrv2_decoder import RTDETRTransformerv2
from models.ESVT.criterion.rtdetrv2_criterion import RTDETRCriterionv2
from models.ESVT.criterion.matcher import HungarianMatcher
from models.ESVT.postprocessor.rtdetr_postprocessor import RTDETRPostProcessor


def build_ESVT(args):
    # 🔥 Check baseline mode
    baseline_mode = getattr(args, 'baseline_mode', False)
    streaming_type = args.streaming_type if not baseline_mode else 'none'

    # 🔥 新增: 根据 streaming_type 自动选择 Encoder
    use_enhanced = streaming_type.startswith('lstm_') and streaming_type != 'lstm'
    EncoderClass = HybridEncoderEnhanced if use_enhanced else HybridEncoder

    if args.model_type == 'event':

        if args.backbone[:-1] == 'hgnetv2':
            return ESVT(
                backbone=HGNetv2(name=args.backbone[-1], pretrained=args.backbone_pretrained),
                encoder=EncoderClass(
                    name=args.transformer_scale[-1],
                    streaming_type=streaming_type,
                    baseline_mode=baseline_mode
                ),
                decoder=RTDETRTransformerv2(name=args.transformer_scale[-1], dataset=args.dataset)
            )

        elif args.backbone[:-2] == 'resnet':
            return ESVT(
                backbone=ResNet(name=args.backbone[-2:], pretrained=args.backbone_pretrained),
                encoder=EncoderClass(
                    backbone_name=args.backbone[-2:],
                    name=args.transformer_scale[-1],
                    streaming_type=streaming_type,
                    baseline_mode=baseline_mode
                ),
                decoder=RTDETRTransformerv2(name=args.transformer_scale[-1], dataset=args.dataset)
            )

    # TODO 还未实现其他模态, 稍后实现
    elif args.model_type == 'image':
        pass
    elif args.model_type == 'multimodel':
        pass


def build_ESVT_criterion(args):
    return RTDETRCriterionv2(
        matcher=HungarianMatcher(weight_dict=args.matcher_weight_dict, use_focal_loss=args.use_focal_loss),
        weight_dict=args.criterion_weight_dict,
        losses=args.criterion_losses,
        dataset=args.dataset
    )


def build_ESVT_postprocessor(args):
    return RTDETRPostProcessor(dataset=args.dataset,
                               use_focal_loss=args.use_focal_loss,
                               num_top_queries=args.num_top_queries)
```

---

## ✅ 总结

1. **只需修改一个文件**: `models/ESVT/__init__.py`
2. **添加 1 行导入** + **改 3 行代码**（EncoderClass 选择）
3. **完全向后兼容**: 原有实验不受影响
4. **自动切换**: 根据 `streaming_type` 参数自动选择版本

修改完成后，你就可以在训练命令中自由切换不同的注意力机制了！🎉
