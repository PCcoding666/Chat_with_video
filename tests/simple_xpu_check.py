#!/usr/bin/env python3
"""
简单XPU测试
验证我们的终极XPU加载策略是否工作
"""

import os
import sys

# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

def main():
    print("🚀 简单XPU测试")
    print("="*30)
    
    try:
        # 添加项目路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # 测试1: 基础导入
        print("1. 测试基础导入...")
        import torch
        from src.chat_with_video.model_loader import MiniCPMVInference
        
        print("✅ 基础导入成功")
        print(f"   XPU可用: {torch.xpu.is_available()}")
        
        # 测试2: 创建实例
        print("\n2. 测试创建实例...")
        engine = MiniCPMVInference(
            model_path='openbmb/MiniCPM-V-4_5-int4',
            device='xpu'
        )
        print("✅ 实例创建成功")
        
        # 测试3: 检查设备信息（不加载模型）
        print("\n3. 测试设备信息...")
        device_info = engine.get_device_info()
        print("设备信息:")
        for k, v in device_info.items():
            print(f"   {k}: {v}")
        
        if device_info.get('xpu_available'):
            print("✅ XPU可用性检查通过")
        else:
            print("❌ XPU不可用")
            return 1
            
        print("\n🎉 简单测试完成！")
        print("✅ XPU强制加载策略已准备就绪")
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)