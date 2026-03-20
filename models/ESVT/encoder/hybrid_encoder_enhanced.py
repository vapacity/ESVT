"""
增强版 Hybrid Encoder - 支持增强版 ConvLSTM

基于原始 HybridEncoder 的改进版本
主要改进:
1. 支持增强版 ConvLSTM (带注意力机制)
2. 通过 lstm_variant 参数选择不同的注意力配置
3. 保持与原版的接口兼容性
4. 不修改其他部分 (Transformer, FPN, PAN)

用法:
    from models.ESVT.encoder.hybrid_encoder_enhanced import HybridEncoderEnhanced

    # 使用 CBAM 注意力的 ConvLSTM
    encoder = HybridEncoderEnhanced(
        streaming_type='lstm_cbam',
        ...
    )

Author: Enhanced for ESVT ablation study
Date: 2026-03-20
"""

import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ESVT.utils import get_activation
from models.ESVT.lstm.lstm import DWSConvLSTM2d  # 原始版本
from models.ESVT.lstm.lstm_enhanced import DWSConvLSTM2d_Enhanced, build_convlstm_variant  # 增强版本

# 导入原始 encoder 的其他组件 (保持不变)
from models.ESVT.encoder.hybrid_encoder import (
    ConvNormLayer,
    RepVggBlock,
    CSPRepLayer,
    TransformerEncoderLayer,
    TransformerEncoder
)


__all__ = ['HybridEncoderEnhanced']


class HybridEncoderEnhanced(nn.Module):
    """
    增强版 Hybrid Encoder

    与原版 HybridEncoder 的唯一区别:
    - streaming_type 支持更多选项,用于选择不同的 ConvLSTM 变体

    streaming_type 选项:
        'none': 无时序模块 (baseline)
        'lstm': 原始 ConvLSTM
        'lstm_se': ConvLSTM + SE 注意力
        'lstm_cbam': ConvLSTM + CBAM 注意力
        'lstm_eca': ConvLSTM + ECA 注意力
        'lstm_spatial': ConvLSTM + 空间注意力
        'lstm_se_after': ConvLSTM + SE (应用在 cell 输出后)
        'lstm_cbam_both': ConvLSTM + CBAM (应用在两个位置)
    """
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 backbone_name=None,
                 name='L',
                 streaming_type='lstm_cbam',  # 🔥 默认使用 CBAM 增强版
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='v2',
                 baseline_mode=False):
        super().__init__()

        if name == 'X':
            hidden_dim = 384
            dim_feedforward = 2048
        elif name == 'H':
            hidden_dim = 512
            dim_feedforward = 2048
            num_encoder_layers = 2
        else:
            assert name in ['L', 'X', 'H']

        if backbone_name == '18' or backbone_name == '34':
            in_channels = [128, 256, 512]

        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.baseline_mode = baseline_mode
        self.streaming_type = streaming_type  # 🔥 保存配置用于记录

        # channel projection (与原版完全相同)
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim))
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                raise AttributeError()

            self.input_proj.append(proj)

        # ===== 🔥 LSTM 模块选择 (修改部分) =====
        self.stm = self._build_streaming_module(streaming_type, hidden_dim, baseline_mode)

        # encoder transformer (与原版完全相同)
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn (与原版完全相同)
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan (与原版完全相同)
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _build_streaming_module(self, streaming_type, hidden_dim, baseline_mode):
        """
        构建时序流模块

        Args:
            streaming_type: 流类型
            hidden_dim: 隐藏层维度
            baseline_mode: 是否为 baseline 模式

        Returns:
            streaming module 或 None
        """
        if baseline_mode or streaming_type == 'none':
            return None

        # 原始 ConvLSTM
        if streaming_type == 'lstm':
            return DWSConvLSTM2d(dim=hidden_dim)

        # 增强版 ConvLSTM (带注意力)
        lstm_variants = {
            'lstm_se': 'se',
            'lstm_cbam': 'cbam',
            'lstm_eca': 'eca',
            'lstm_spatial': 'spatial',
            'lstm_se_after': 'se_after',
            'lstm_cbam_both': 'cbam_both',
        }

        if streaming_type in lstm_variants:
            variant_name = lstm_variants[streaming_type]
            return build_convlstm_variant(variant_name, dim=hidden_dim)

        # 其他类型 (如果将来有 stc 等)
        raise ValueError(f"Unknown streaming_type: {streaming_type}")

    def _reset_parameters(self):
        """与原版完全相同"""
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """与原版完全相同"""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats, pre_status=None):
        """
        前向传播 (与原版完全相同的逻辑)

        Args:
            feats: backbone 输出的特征列表
            pre_status: 上一时间步的状态

        Returns:
            outs: 输出特征列表
            status: 当前时间步的状态
        """
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # lstm
        if self.baseline_mode or self.stm is None:
            stm_feats = proj_feats
            status = [None] * len(feats)
        else:
            if pre_status is None:
                pre_status = [None] * len(feats)
            stm_status = [self.stm(proj_feat, pre_state) for proj_feat, pre_state in zip(proj_feats, pre_status)]
            stm_feats, status = zip(*[(state[0], state) for state in stm_status])
            stm_feats, status = list(stm_feats), list(status)  # 🔥 修正: 使用 stm_feats 而不是 proj_feats

        proj_feats = stm_feats

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs, status


if __name__ == '__main__':
    """
    测试增强版 Encoder
    """
    print("="*80)
    print("Testing Enhanced Hybrid Encoder")
    print("="*80)

    # 模拟 backbone 输出
    batch_size = 2
    feats = [
        torch.randn(batch_size, 512, 64, 64),   # scale 1
        torch.randn(batch_size, 1024, 32, 32),  # scale 2
        torch.randn(batch_size, 2048, 16, 16),  # scale 3
    ]

    print(f"\nInput features:")
    for i, feat in enumerate(feats):
        print(f"  Scale {i+1}: {feat.shape}")

    # 测试不同的 streaming_type
    streaming_types = ['none', 'lstm', 'lstm_se', 'lstm_cbam', 'lstm_eca']

    for st in streaming_types:
        print(f"\n{'='*80}")
        print(f"Testing streaming_type: {st}")
        print(f"{'='*80}")

        try:
            encoder = HybridEncoderEnhanced(
                streaming_type=st,
                backbone_name='50',
                name='L'
            )

            # 计算参数量
            total_params = sum(p.numel() for p in encoder.parameters())
            print(f"Total parameters: {total_params:,}")

            # 前向传播
            outs, status = encoder(feats)

            print(f"✅ Output features:")
            for i, out in enumerate(outs):
                print(f"  Scale {i+1}: {out.shape}")
            print(f"Status: {['None' if s is None else f'({s[0].shape}, {s[1].shape})' for s in status]}")

        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("All tests completed! ✅")
    print("="*80)
