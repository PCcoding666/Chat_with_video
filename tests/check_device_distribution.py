#!/usr/bin/env python3
"""
设备分布检查脚本
检查模型实际加载在哪个设备上
"""

import os
import torch

# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

def check_current_service_device():
    """检查当前运行服务的模型设备分布"""
    try:
        print("🔍 检查当前服务的模型设备分布...")
        
        # 添加项目路径
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        from src.chat_with_video.video_chat_service import VideoChatService
        
        # 创建服务
        service = VideoChatService()
        
        # 初始化服务
        if service.initialize():
            model = service.inference_engine.model
            
            print("\n📊 模型设备分布详情:")
            
            device_summary = {}
            total_params = 0
            
            for name, param in model.named_parameters():
                device = str(param.device)
                param_count = param.numel()
                total_params += param_count
                
                if device not in device_summary:
                    device_summary[device] = {'count': 0, 'params': 0, 'layers': []}
                
                device_summary[device]['count'] += 1
                device_summary[device]['params'] += param_count
                device_summary[device]['layers'].append(name)
            
            print(f"总参数量: {total_params:,}")
            print(f"设备分布:")
            
            for device, info in device_summary.items():
                percentage = (info['params'] / total_params) * 100
                print(f"  {device}:")
                print(f"    - 层数: {info['count']}")
                print(f"    - 参数量: {info['params']:,} ({percentage:.1f}%)")
                print(f"    - 主要层: {info['layers'][:3]}{'...' if len(info['layers']) > 3 else ''}")
            
            # 检查主要层的设备分布
            print("\n🧩 关键层设备分布:")
            key_layers = ['embed', 'attention', 'mlp', 'lm_head', 'vision']
            
            for name, param in model.named_parameters():
                for key in key_layers:
                    if key in name.lower():
                        print(f"  {name}: {param.device}")
                        break
            
            # 显示XPU使用情况
            if torch.xpu.is_available():
                print(f"\n💾 Intel XPU状态:")
                print(f"  设备数量: {torch.xpu.device_count()}")
                print(f"  当前设备: {torch.xpu.current_device()}")
                
                try:
                    allocated = torch.xpu.memory_allocated() / 1024**3
                    print(f"  显存使用: {allocated:.2f} GB")
                except:
                    print("  显存使用: 无法获取")
            
            return True
            
        else:
            print("❌ 服务初始化失败")
            return False
            
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_xpu_tensor_operations():
    """测试XPU张量操作"""
    print("\n🧪 测试Intel XPU张量操作:")
    
    try:
        if not torch.xpu.is_available():
            print("  ❌ Intel XPU不可用")
            return False
        
        # 测试基本操作
        print("  ✅ XPU可用")
        
        # 创建XPU张量
        x = torch.randn(1000, 1000).to('xpu')
        y = torch.randn(1000, 1000).to('xpu')
        
        # 矩阵运算
        result = torch.mm(x, y)
        
        print(f"  ✅ 矩阵运算测试成功，结果形状: {result.shape}")
        print(f"  ✅ 结果设备: {result.device}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ XPU测试失败: {e}")
        return False

def main():
    """主检查函数"""
    print("="*60)
    print("🔍 Intel XPU 设备分布诊断")
    print("="*60)
    
    # 1. 测试XPU基础功能
    xpu_ok = test_xpu_tensor_operations()
    
    if not xpu_ok:
        print("\n❌ Intel XPU基础测试失败，请检查驱动和环境")
        return 1
    
    # 2. 检查当前服务的设备分布
    service_ok = check_current_service_device()
    
    if not service_ok:
        print("\n❌ 服务设备检查失败")
        return 1
    
    print("\n" + "="*60)
    print("✅ 设备检查完成")
    print("="*60)
    
    print("\n📋 诊断建议:")
    print("  1. 如果模型主要在CPU上，需要修复model_loader.py")
    print("  2. 如果模型在XPU上但性能差，可能需要优化配置")
    print("  3. 检查Intel oneAPI运行时库是否正确安装")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)