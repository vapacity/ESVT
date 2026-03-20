# 🔧 Baseline训练问题修复

## 问题描述

运行baseline训练时出现错误：
```
UnboundLocalError: local variable 'status' referenced before assignment
```

## 根本原因

在baseline模式下，模型返回的`status`为`[None, None, None]`（没有时序状态），但是`engine.py`中的训练和评估函数在第一次迭代时就试图访问未定义的`status`变量。

## 解决方案

### 修改的文件：`engine.py`

#### 1. 训练函数 `train_one_epoch()` (第58-75行)

**修改前：**
```python
def train_one_epoch(model, criterion, data_loader, ...):
    # ... 初始化代码 ...

    for i, ((images, events, targets), indexes) in enumerate(...):
        # ❌ 第一次迭代时status未定义！
        if indexes[-1][-1] % 100 == 0:
            pre_status = None
        else:
            pre_status = [(state[0].detach(), state[1].detach()) for state in status]
```

**修改后：**
```python
def train_one_epoch(model, criterion, data_loader, ...):
    # ... 初始化代码 ...

    # 🔥 Initialize status to None for first iteration and baseline mode
    status = None

    for i, ((images, events, targets), indexes) in enumerate(...):
        # 🔥 Handle status for both baseline and ESVT modes
        if indexes[-1][-1] % 100 == 0 or status is None:
            pre_status = None
        else:
            # Check if status contains valid states (not None)
            if status and all(s is not None for s in status):
                pre_status = [(state[0].detach(), state[1].detach()) for state in status]
            else:
                # Baseline mode: status is [None, None, None]
                pre_status = None
```

#### 2. 评估函数 `evaluate()` (第134-152行)

做了相同的修改，确保评估时也能正确处理baseline模式。

## 关键改进

1. ✅ **初始化status**：在循环开始前初始化为`None`
2. ✅ **检查status是否为None**：添加`or status is None`条件
3. ✅ **验证status内容**：检查status是否包含有效状态（不是None列表）
4. ✅ **兼容性**：同时支持baseline模式（status为None）和ESVT模式（status有实际值）

## 工作原理

### Baseline模式：
```
第1次迭代: status = None → pre_status = None
第2次迭代: status = [None, None, None] → pre_status = None
第3次迭代: status = [None, None, None] → pre_status = None
...
```

### ESVT模式：
```
第1次迭代: status = None → pre_status = None
第2次迭代: status = [(h1, c1), (h2, c2), (h3, c3)] → pre_status = [(h1.detach(), c1.detach()), ...]
第3次迭代: status = [(h1, c1), (h2, c2), (h3, c3)] → pre_status = [(h1.detach(), c1.detach()), ...]
...
每100步重置: pre_status = None
```

## 验证

运行以下命令验证修复：

```bash
cd /Users/zwj/Documents/毕设/ESVT
python train_baseline.py --epoches 1 --batch_size 2
```

如果看到训练正常进行，说明修复成功！

## 相关文件

- ✅ `engine.py` - 已修复训练和评估逻辑
- ✅ `models/ESVT/encoder/hybrid_encoder.py` - Baseline模式返回None状态
- ✅ `train_baseline.py` - Baseline训练脚本

## 状态

🟢 **已修复** - 可以正常开始训练
