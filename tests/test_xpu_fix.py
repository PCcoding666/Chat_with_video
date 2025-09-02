#!/usr/bin/env python3
"""
测试Intel XPU模型加载修复
验证模型是否正确加载到Intel GPU上
"""

import os
import torch

# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

def test_xpu_model_loading():
    """测试XPU模型加载"""
    try:
        print("🔧 测试Intel XPU模型加载修复...")
        
        # 导入修复后的模型加载器
        from src.chat_with_video.model_loader import MiniCPMVInference
        
        # 创建推理引擎
        print("📦 创建推理引擎...")
        inference_engine = MiniCPMVInference(device='xpu')
        
        # 初始化模型
        print("🚀 初始化模型...")
        inference_engine.initialize()
        
        # 检查设备信息
        device_info = inference_engine.get_device_info()
        print(f"📊 设备信息:")
        for key, value in device_info.items():
            print(f"  {key}: {value}")
        
        # 详细检查模型设备分布
        print("\n🔍 详细设备分布检查:")
        model = inference_engine.model
        
        device_summary = {}
        total_params = 0
        xpu_params = 0
        
        for name, param in model.named_parameters():
            device = str(param.device)
            param_count = param.numel()
            total_params += param_count
            
            if 'xpu' in device.lower():
                xpu_params += param_count
            
            if device not in device_summary:
                device_summary[device] = {'count': 0, 'params': 0, 'layers': []}
            
            device_summary[device]['count'] += 1
            device_summary[device]['params'] += param_count
            device_summary[device]['layers'].append(name)
        
        print(f"📈 总参数量: {total_params:,}")
        print(f"🎯 XPU参数量: {xpu_params:,} ({(xpu_params/total_params)*100:.1f}%)")
        
        for device, info in device_summary.items():
            percentage = (info['params'] / total_params) * 100
            print(f"  {device}:")
            print(f"    - 层数: {info['count']}")
            print(f"    - 参数量: {info['params']:,} ({percentage:.1f}%)")
            if len(info['layers']) <= 3:
                print(f"    - 层名: {info['layers']}")
            else:
                print(f"    - 主要层: {info['layers'][:3]}... (+{len(info['layers'])-3}更多)")
        
        # 测试推理
        print("\n🧪 测试推理功能...")
        try:
            test_msgs = [
                {'role': 'user', 'content': ['测试Intel XPU推理']}
            ]
            
            response = inference_engine.chat(
                msgs=test_msgs,
                max_new_tokens=20,
                temperature=0.7
            )
            
            print(f"✅ 推理测试成功!")
            print(f"📝 回答: {response}")
            
        except Exception as e:
            print(f"⚠️ 推理测试失败: {e}")
        
        # 验证结果
        if xpu_params > 0:
            print(f"\n🎉 修复成功! 模型已加载到Intel XPU")
            print(f"💯 XPU参数比例: {(xpu_params/total_params)*100:.1f}%")
            
            # 显示XPU使用情况
            if hasattr(torch.xpu, 'memory_allocated'):
                allocated = torch.xpu.memory_allocated() / 1024**3
                print(f"💾 XPU显存使用: {allocated:.2f} GB")
            
            return True
        else:
            print(f"\n❌ 修复失败! 模型仍在CPU上")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("🔧 Intel XPU 模型加载修复测试")
    print("="*60)
    
    # 基础XPU检查
    if not torch.xpu.is_available():
        print("❌ Intel XPU不可用")
        return 1
    
    print(f"✅ Intel XPU可用，设备数量: {torch.xpu.device_count()}")
    
    # 测试修复
    success = test_xpu_model_loading()
    
    if success:
        print("\n" + "="*60)
        print("🎉 XPU模型加载修复测试通过!")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("❌ XPU模型加载修复测试失败!")
        print("="*60)
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)