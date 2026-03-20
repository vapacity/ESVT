"""
增强版 ConvLSTM - Enhanced DWSConvLSTM2d with Attention Mechanisms

基于原始 DWSConvLSTM2d (lstm.py) 的改进版本
主要改进:
1. 在门控计算前/后加入注意力机制
2. 支持多种注意力类型 (SE, CBAM, ECA, Spatial)
3. 可配置的注意力位置 (before_gates, after_gates, both)
4. 保持与原版的接口兼容性

用于消融实验对比不同注意力机制的效果

Author: Enhanced for ESVT ablation study
Date: 2026-03-20
"""

from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms.functional as F

# 导入原始的 RevNorm (保持一致性)
from models.ESVT.lstm.lstm import RevNorm

# 导入注意力模块
from models.ESVT.lstm.attention_modules import (
    ChannelAttention,
    SpatialAttention,
    CBAM,
    ECAAttention,
    LSTMGateAttention
)


class DWSConvLSTM2d_Enhanced(nn.Module):
    """
    增强版深度可分离 ConvLSTM

    与原版的区别:
    1. 新增 attention_type 参数: 选择注意力类型
    2. 新增 attention_position 参数: 选择注意力应用位置
    3. 新增 use_attention 参数: 控制是否启用注意力 (方便消融实验)

    参数:
        dim: 通道数
        dws_conv: 是否使用深度可分离卷积
        dws_conv_only_hidden: 只对隐藏状态做卷积
        dws_conv_kernel_size: 卷积核大小
        cell_update_dropout: Dropout 概率
        attention_type: 注意力类型 ('none', 'se', 'cbam', 'eca', 'spatial')
        attention_position: 注意力位置 ('before_gates', 'after_cell', 'both')
        attention_reduction: 通道注意力的降维比例
    """
    def __init__(self,
                 dim: int = 256,
                 dws_conv: bool = True,
                 dws_conv_only_hidden: bool = True,
                 dws_conv_kernel_size: int = 3,
                 cell_update_dropout: float = 0.,
                 attention_type: str = 'cbam',
                 attention_position: str = 'before_gates',
                 attention_reduction: int = 16):
        super().__init__()

        self.dim = dim
        xh_dim = dim * 2
        gates_dim = dim * 4
        conv3x3_dws_dim = dim if dws_conv_only_hidden else xh_dim

        # ===== 原始 ConvLSTM 组件 =====
        self.conv3x3_dws = nn.Conv2d(
            in_channels=conv3x3_dws_dim,
            out_channels=conv3x3_dws_dim,
            kernel_size=dws_conv_kernel_size,
            padding=dws_conv_kernel_size // 2,
            groups=conv3x3_dws_dim
        ) if dws_conv else nn.Identity()

        self.conv1x1 = nn.Conv2d(
            in_channels=xh_dim,
            out_channels=gates_dim,
            kernel_size=1
        )

        self.conv_only_hidden = dws_conv_only_hidden
        self.cell_update_dropout = nn.Dropout(p=cell_update_dropout)
        self.revnorm = RevNorm(num_features=dim, affine=True, subtract_last=False)

        # ===== 新增: 注意力机制 =====
        self.attention_type = attention_type
        self.attention_position = attention_position
        self.use_attention = attention_type != 'none'

        if self.use_attention:
            # 注意力应用在 concat 后的特征上
            if attention_position in ['before_gates', 'both']:
                self.attention_before = self._build_attention(
                    xh_dim, attention_type, attention_reduction
                )
            else:
                self.attention_before = None

            # 注意力应用在 cell 输出上
            if attention_position in ['after_cell', 'both']:
                self.attention_after = self._build_attention(
                    dim, attention_type, attention_reduction
                )
            else:
                self.attention_after = None

    def _build_attention(self, channels, attn_type, reduction):
        """构建注意力模块"""
        if attn_type == 'se' or attn_type == 'channel':
            return ChannelAttention(channels, reduction)
        elif attn_type == 'cbam':
            return CBAM(channels, reduction)
        elif attn_type == 'eca':
            return ECAAttention(channels)
        elif attn_type == 'spatial':
            return SpatialAttention()
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")

    def forward(self, x, hc_previous) -> Tuple[Tensor, Tensor]:
        """
        前向传播

        Args:
            x: 当前时间步输入 [B, C, H, W]
            hc_previous: 上一时间步状态 (h_t-1, c_t-1) 或 None

        Returns:
            h_t1: 当前隐藏状态 [B, C, H, W]
            c_t1: 当前记忆单元 [B, C, H, W]
        """
        # ===== 初始化检查 (与原版相同) =====
        if hc_previous is None:
            hidden = x
            cell = x
            hc_previous = (hidden, cell)

        h_t0, c_t0 = hc_previous
        B, C, H, W = x.shape
        c_t0, h_t0 = F.resize(c_t0, [H, W]), F.resize(h_t0, [H, W])

        # ===== RevNorm 处理 h_t0 (与原版相同) =====
        B, C, H, W = h_t0.shape
        h_t0 = h_t0.flatten(2)
        h_t0 = h_t0.permute(0, 2, 1)
        h_t0 = self.revnorm(h_t0, 'norm')
        h_t0 = h_t0.permute(0, 2, 1)
        h_t0 = h_t0.reshape(B, C, H, W).contiguous()

        # ===== 深度可分离卷积 (与原版相同) =====
        if self.conv_only_hidden:
            h_t0 = self.conv3x3_dws(h_t0)

        # ===== 拼接 (与原版相同) =====
        xh = torch.cat((x, h_t0), dim=1)

        if not self.conv_only_hidden:
            xh = self.conv3x3_dws(xh)

        # ===== 🔥 新增: 门控前注意力 =====
        if self.use_attention and self.attention_before is not None:
            xh = self.attention_before(xh)

        # ===== 门控计算 (与原版相同) =====
        mix = self.conv1x1(xh)
        cell_input, gates = torch.tensor_split(mix, [self.dim], dim=1)
        gates = torch.sigmoid(gates)
        forget_gate, input_gate, output_gate = torch.tensor_split(gates, 3, dim=1)

        cell_input = self.cell_update_dropout(torch.tanh(cell_input))

        # ===== LSTM Cell 更新 (与原版相同) =====
        c_t1 = forget_gate * c_t0 + input_gate * cell_input
        h_t1 = output_gate * torch.tanh(c_t1)

        # ===== 🔥 新增: Cell 输出后注意力 =====
        if self.use_attention and self.attention_after is not None:
            h_t1 = self.attention_after(h_t1)

        # ===== RevDeNorm 恢复分布 (与原版相同) =====
        B, C, H, W = h_t1.shape
        h_t1 = h_t1.flatten(2)
        h_t1 = h_t1.permute(0, 2, 1)
        h_t1 = self.revnorm(h_t1, 'denorm')
        h_t1 = h_t1.permute(0, 2, 1)
        h_t1 = h_t1.reshape(B, C, H, W).contiguous()

        return h_t1, c_t1


# ============================================================
# 预定义的消融实验配置
# ============================================================

def build_convlstm_variant(variant_name: str, dim: int = 256):
    """
    构建不同的 ConvLSTM 变体 (用于消融实验)

    Args:
        variant_name: 变体名称
        dim: 通道数

    Returns:
        DWSConvLSTM2d_Enhanced 实例

    可用变体:
        'baseline': 原始 ConvLSTM (无注意力)
        'se': 加入 SE (通道注意力)
        'cbam': 加入 CBAM (通道+空间注意力)
        'eca': 加入 ECA (高效通道注意力)
        'spatial': 只加入空间注意力
        'se_after': SE 应用在 cell 输出后
        'cbam_both': CBAM 应用在两个位置
    """
    configs = {
        'baseline': {
            'attention_type': 'none',
            'attention_position': 'before_gates',
        },
        'se': {
            'attention_type': 'se',
            'attention_position': 'before_gates',
        },
        'cbam': {
            'attention_type': 'cbam',
            'attention_position': 'before_gates',
        },
        'eca': {
            'attention_type': 'eca',
            'attention_position': 'before_gates',
        },
        'spatial': {
            'attention_type': 'spatial',
            'attention_position': 'before_gates',
        },
        'se_after': {
            'attention_type': 'se',
            'attention_position': 'after_cell',
        },
        'cbam_both': {
            'attention_type': 'cbam',
            'attention_position': 'both',
        },
    }

    if variant_name not in configs:
        raise ValueError(
            f"Unknown variant: {variant_name}. "
            f"Available: {list(configs.keys())}"
        )

    config = configs[variant_name]
    return DWSConvLSTM2d_Enhanced(
        dim=dim,
        **config
    )


# ============================================================
# 测试代码
# ============================================================

if __name__ == '__main__':
    """
    测试增强版 ConvLSTM 的功能
    """
    print("="*80)
    print("Testing Enhanced ConvLSTM Variants")
    print("="*80)

    # 测试输入
    batch_size = 2
    channels = 256
    height, width = 32, 32
    x = torch.randn(batch_size, channels, height, width)

    print(f"\nInput shape: {x.shape}")

    # 测试所有变体
    variants = ['baseline', 'se', 'cbam', 'eca', 'spatial', 'se_after', 'cbam_both']

    for variant_name in variants:
        print(f"\n{'='*80}")
        print(f"Testing variant: {variant_name}")
        print(f"{'='*80}")

        # 构建模型
        model = build_convlstm_variant(variant_name, dim=channels)

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # 前向传播测试
        try:
            # 第一个时间步 (无历史状态)
            h1, c1 = model(x, None)
            print(f"✅ Time step 1: h shape = {h1.shape}, c shape = {c1.shape}")

            # 第二个时间步 (有历史状态)
            x2 = torch.randn(batch_size, channels, height, width)
            h2, c2 = model(x2, (h1, c1))
            print(f"✅ Time step 2: h shape = {h2.shape}, c shape = {c2.shape}")

            # 验证输出形状
            assert h2.shape == x.shape, "Output shape mismatch!"
            assert c2.shape == x.shape, "Cell shape mismatch!"

            print(f"✅ Forward pass successful!")

        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("All variants tested! ✅")
    print("="*80)

    # 参数量对比
    print("\n" + "="*80)
    print("Parameter Comparison")
    print("="*80)
    print(f"{'Variant':<15} {'Parameters':>15} {'Increase':>15}")
    print("-"*80)

    baseline_params = sum(p.numel() for p in build_convlstm_variant('baseline').parameters())
    print(f"{'baseline':<15} {baseline_params:>15,} {'-':>15}")

    for variant_name in variants[1:]:  # 跳过 baseline
        model = build_convlstm_variant(variant_name)
        params = sum(p.numel() for p in model.parameters())
        increase = params - baseline_params
        increase_pct = (increase / baseline_params) * 100
        print(f"{variant_name:<15} {params:>15,} {f'+{increase:,} ({increase_pct:.2f}%)':>15}")

    print("="*80)
