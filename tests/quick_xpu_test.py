#!/usr/bin/env python3
"""
快速XPU验证测试
快速验证XPU加载是否成功
"""

import os
import sys
import time

# 设置环境变量 - 在导入任何PyTorch相关库之前
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

def main():
    print("🚀 快速XPU验证测试")
    print("="*50)
    
    try:
        # 添加项目路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        print("1. 导入模块...")
        from src.chat_with_video.model_loader import MiniCPMVInference
        
        print("✅ 模块导入成功")
        
        print("\n2. 创建推理引擎...")
        inference_engine = MiniCPMVInference(
            model_path='openbmb/MiniCPM-V-4_5-int4',
            device='xpu'
        )
        
        print("✅ 推理引擎创建成功")
        
        print("\n3. 初始化模型 (强制XPU加载)...")
        start_time = time.time()
        inference_engine.initialize()
        init_time = time.time() - start_time
        
        print(f"✅ 模型初始化成功！耗时: {init_time:.2f}秒")
        
        print("\n4. 验证设备信息...")
        device_info = inference_engine.get_device_info()
        
        print("设备信息:")
        for key, value in device_info.items():
            print(f"  {key}: {value}")
        
        # 检查是否在XPU上
        if device_info.get('device') == 'xpu':
            print("\n🎉 成功！模型确实在XPU上运行")
            print("✅ Intel XPU显存查询限制已彻底绕过")
        else:
            print(f"\n❌ 失败！模型在 {device_info.get('device')} 上运行，不是XPU")
            return 1
        
        print("\n5. 测试简单推理...")
        test_msgs = [
            {'role': 'user', 'content': ['Hello, XPU!']}
        ]
        
        response = inference_engine.chat(
            msgs=test_msgs,
            max_new_tokens=20,
            do_sample=False
        )
        
        print(f"✅ 推理测试成功！回答: {response}")
        
        print("\n" + "="*50)
        print("🎉 快速验证测试通过！")
        print("✅ XPU强制加载策略工作正常")
        print("✅ Intel Arc GPU显存查询问题已解决")
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    input("\n按Enter键退出...")
    sys.exit(exit_code)