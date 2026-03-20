"""
注意力模块库 - Attention Modules Library
用于增强 ConvLSTM 的注意力机制

包含以下模块:
1. ChannelAttention (SE-like): 通道注意力
2. SpatialAttention: 空间注意力
3. CBAM: 通道+空间注意力 (Convolutional Block Attention Module)
4. ECA: 高效通道注意力 (Efficient Channel Attention)

Author: Enhanced for ESVT ablation study
Date: 2026-03-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    通道注意力模块 (Squeeze-and-Excitation 风格)

    论文参考:
    - Hu et al. "Squeeze-and-Excitation Networks" CVPR 2018
    - 借鉴 HGNetv2 的 aggregation_squeeze/excitation 思想

    工作流程:
    1. Global Average Pooling: 压缩空间维度 [B,C,H,W] -> [B,C,1,1]
    2. FC + ReLU: 降维学习通道关系 [B,C] -> [B,C/r]
    3. FC + Sigmoid: 升维生成注意力权重 [B,C/r] -> [B,C]
    4. 重标定: 原始特征 * 注意力权重
    """
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: 输入通道数
            reduction: 降维比例 (论文推荐 16)
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] (带通道注意力的特征)
        """
        b, c, _, _ = x.size()
        # Squeeze: [B,C,H,W] -> [B,C,1,1] -> [B,C]
        y = self.avg_pool(x).view(b, c)
        # Excitation: [B,C] -> [B,C]
        y = self.fc(y).view(b, c, 1, 1)
        # Scale + 残差: 保证训练初期稳定性
        return x + x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    空间注意力模块

    论文参考: Woo et al. "CBAM: Convolutional Block Attention Module" ECCV 2018

    工作流程:
    1. 通道维度的 max/avg pooling: [B,C,H,W] -> [B,2,H,W]
    2. 7×7 卷积: 学习空间关系 [B,2,H,W] -> [B,1,H,W]
    3. Sigmoid: 生成空间注意力图
    4. 重标定: 原始特征 * 空间权重
    """
    def __init__(self, kernel_size=7):
        """
        Args:
            kernel_size: 卷积核大小 (论文推荐 7)
        """
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] (带空间注意力的特征)
        """
        # 在通道维度上做 max 和 avg pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B,1,H,W]
        # 拼接: [B,2,H,W]
        concat = torch.cat([avg_out, max_out], dim=1)
        # 7×7 卷积 + sigmoid
        attention = self.sigmoid(self.conv(concat))  # [B,1,H,W]
        # Scale + 残差: 保证训练初期稳定性
        return x + x * attention


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    结合通道注意力和空间注意力，带残差连接保证训练稳定性

    论文: Woo et al. "CBAM: Convolutional Block Attention Module" ECCV 2018

    顺序: 通道注意力 -> 空间注意力 (论文证明这个顺序更好)

    注意: 加入残差连接 (x + attention(x))，避免训练初期注意力权重随机
    导致特征大幅压缩，引发梯度不稳定
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        """
        Args:
            channels: 输入通道数
            reduction: 通道注意力的降维比例
            kernel_size: 空间注意力的卷积核大小
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] (带 CBAM 注意力的特征)
        """
        # 残差连接: 注意力作为增量叠加到原始特征上
        x = x + self.channel_attention(x)  # 通道注意力 + 残差
        x = x + self.spatial_attention(x)  # 空间注意力 + 残差
        return x


class ECAAttention(nn.Module):
    """
    ECA: Efficient Channel Attention
    高效通道注意力 (比 SE 更轻量)

    论文: Wang et al. "ECA-Net: Efficient Channel Attention for Deep CNNs" CVPR 2020

    核心思想:
    - 不使用 FC 层降维 (避免损失信息)
    - 使用 1D 卷积学习通道间的局部依赖
    - 参数量极少,计算高效
    """
    def __init__(self, channels, k_size=3):
        """
        Args:
            channels: 输入通道数
            k_size: 1D 卷积核大小 (论文建议自适应计算)
        """
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D 卷积 (在通道维度上)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] (带 ECA 注意力的特征)
        """
        b, c, _, _ = x.size()
        # Global Average Pooling: [B,C,H,W] -> [B,C,1,1]
        y = self.avg_pool(x)
        # 1D 卷积: [B,C,1,1] -> [B,1,C] -> [B,1,C] -> [B,C,1,1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Sigmoid 生成权重
        y = self.sigmoid(y)
        # Scale + 残差: 保证训练初期稳定性
        return x + x * y.expand_as(x)


class DualAttention(nn.Module):
    """
    双路注意力: 并行的通道注意力和空间注意力
    (与 CBAM 的区别: CBAM 是串行,这个是并行)

    适用场景: 需要同时强调通道和空间特征的场合
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(DualAttention, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        # 并行处理后残差融合
        channel_out = self.channel_attention(x)
        spatial_out = self.spatial_attention(x)
        return x + channel_out + spatial_out


# ============================================================
# 用于 ConvLSTM 的专用注意力模块
# ============================================================

class LSTMGateAttention(nn.Module):
    """
    LSTM 门控增强注意力

    核心思想:
    - 在 LSTM 的门控计算前,对输入特征施加注意力
    - 帮助 LSTM 更好地聚焦重要信息

    用法: 在 ConvLSTM 的 forward 中,对 xh (concat 后的特征) 应用此模块
    """
    def __init__(self, channels, attention_type='cbam', reduction=16):
        """
        Args:
            channels: 输入通道数
            attention_type: 注意力类型 ('se', 'cbam', 'eca', 'spatial')
            reduction: 降维比例
        """
        super(LSTMGateAttention, self).__init__()

        if attention_type == 'se' or attention_type == 'channel':
            self.attention = ChannelAttention(channels, reduction)
        elif attention_type == 'cbam':
            self.attention = CBAM(channels, reduction)
        elif attention_type == 'eca':
            self.attention = ECAAttention(channels)
        elif attention_type == 'spatial':
            self.attention = SpatialAttention()
        elif attention_type == 'dual':
            self.attention = DualAttention(channels, reduction)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.attention_type = attention_type

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - ConvLSTM 中 concat([x_t, h_t-1]) 后的特征
        Returns:
            out: [B, C, H, W] - 带注意力增强的特征
        """
        return self.attention(x)


if __name__ == '__main__':
    """
    测试代码: 验证各个注意力模块的功能
    """
    print("="*80)
    print("Testing Attention Modules")
    print("="*80)

    # 测试输入
    batch_size = 2
    channels = 256
    height, width = 32, 32
    x = torch.randn(batch_size, channels, height, width)

    print(f"\nInput shape: {x.shape}")

    # 1. 测试 ChannelAttention
    print("\n[1] Testing ChannelAttention (SE-like)...")
    ca = ChannelAttention(channels)
    out_ca = ca(x)
    print(f"   Output shape: {out_ca.shape}")
    print(f"   Parameters: {sum(p.numel() for p in ca.parameters()):,}")

    # 2. 测试 SpatialAttention
    print("\n[2] Testing SpatialAttention...")
    sa = SpatialAttention()
    out_sa = sa(x)
    print(f"   Output shape: {out_sa.shape}")
    print(f"   Parameters: {sum(p.numel() for p in sa.parameters()):,}")

    # 3. 测试 CBAM
    print("\n[3] Testing CBAM...")
    cbam = CBAM(channels)
    out_cbam = cbam(x)
    print(f"   Output shape: {out_cbam.shape}")
    print(f"   Parameters: {sum(p.numel() for p in cbam.parameters()):,}")

    # 4. 测试 ECA
    print("\n[4] Testing ECA...")
    eca = ECAAttention(channels)
    out_eca = eca(x)
    print(f"   Output shape: {out_eca.shape}")
    print(f"   Parameters: {sum(p.numel() for p in eca.parameters()):,}")

    # 5. 测试 LSTMGateAttention
    print("\n[5] Testing LSTMGateAttention (for ConvLSTM)...")
    lstm_attn = LSTMGateAttention(channels, attention_type='cbam')
    out_lstm = lstm_attn(x)
    print(f"   Output shape: {out_lstm.shape}")
    print(f"   Parameters: {sum(p.numel() for p in lstm_attn.parameters()):,}")

    print("\n" + "="*80)
    print("All tests passed! ✅")
    print("="*80)
