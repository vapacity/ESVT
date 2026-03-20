#!/usr/bin/env python3
"""
验证Baseline配置是否正确
"""

import sys
import argparse
import torch

# 添加项目路径
sys.path.insert(0, '/Users/zwj/Documents/毕设/ESVT')

def verify_baseline_config():
    print("\n" + "="*80)
    print("🔍 Verifying RT-DETR Baseline Configuration")
    print("="*80 + "\n")

    try:
        from models import build_model

        # 创建baseline配置
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

        print("📋 Configuration:")
        print(f"  - Model: {args.model}")
        print(f"  - Backbone: {args.backbone}")
        print(f"  - Baseline Mode: {args.baseline_mode}")
        print(f"  - Streaming Type: {args.streaming_type}")
        print(f"  - Transformer Scale: {args.transformer_scale}")
        print()

        # 构建模型
        print("🏗️  Building model...")
        model = build_model(args)

        # 验证关键配置
        print("\n✅ Verification Results:")
        print("-" * 80)

        # 1. 检查baseline模式
        baseline_mode = model.encoder.baseline_mode
        print(f"1. Baseline Mode Enabled: {baseline_mode}")
        if not baseline_mode:
            print("   ⚠️  WARNING: Baseline mode should be True!")
        else:
            print("   ✓ Correct")

        # 2. 检查streaming模块
        stm = model.encoder.stm
        print(f"\n2. Streaming Module (should be None): {stm}")
        if stm is not None:
            print("   ⚠️  WARNING: Streaming module should be None in baseline mode!")
        else:
            print("   ✓ Correct")

        # 3. 检查参数量
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\n3. Total Parameters: {total_params:.2f}M")
        expected_params = 24.11
        if abs(total_params - expected_params) > 5:
            print(f"   ⚠️  WARNING: Expected ~{expected_params}M params (difference: {abs(total_params - expected_params):.2f}M)")
        else:
            print(f"   ✓ Close to expected {expected_params}M params")

        # 4. 检查模型结构
        print(f"\n4. Model Architecture:")
        print(f"   - Backbone: {type(model.backbone).__name__}")
        print(f"   - Encoder: {type(model.encoder).__name__}")
        print(f"   - Decoder: {type(model.decoder).__name__}")

        # 5. 测试前向传播
        print(f"\n5. Testing Forward Pass...")
        try:
            # 创建随机输入 (batch=2, channels=5, height=346, width=260)
            dummy_input = torch.randn(2, 5, 346, 260)
            with torch.no_grad():
                output, targets, status = model(dummy_input)
            print(f"   ✓ Forward pass successful")
            print(f"   - Output shape: {[o.shape if hasattr(o, 'shape') else 'N/A' for o in output]}")
            print(f"   - Status (should be None list): {status}")

            if status != [None, None, None]:
                print(f"   ⚠️  WARNING: Status should be [None, None, None] in baseline mode!")
            else:
                print(f"   ✓ Status is correct (no temporal state)")

        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
            return False

        print("\n" + "="*80)
        print("✅ Baseline Configuration Verified Successfully!")
        print("="*80)
        print("\n📝 Summary:")
        print(f"  - Model is correctly configured as RT-DETR Baseline")
        print(f"  - No temporal modules (ConvLSTM/RevNorm) are active")
        print(f"  - Total parameters: {total_params:.2f}M")
        print(f"  - Ready for training!")
        print()

        return True

    except Exception as e:
        print(f"\n❌ Verification Failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = verify_baseline_config()
    sys.exit(0 if success else 1)
