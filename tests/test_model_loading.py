#!/usr/bin/env python3
"""
模型加载测试
测试模型在XPU上的加载和初始化
"""

import sys
import time
import torch
from .test_utils import setup_test_environment, setup_project_path, print_separator

# 设置测试环境
setup_test_environment()
setup_project_path()

def test_model_loading():
    """测试模型加载"""
    print_separator("🚀 模型加载测试")
    
    try:
        from src.chat_with_video.model_loader import MiniCPMVInference
        
        # 创建推理引擎实例
        print("创建MiniCPMVInference实例...")
        start_time = time.time()
        
        inference_engine = MiniCPMVInference(
            model_path='openbmb/MiniCPM-V-4_5-int4',
            device='xpu'
        )
        
        print("✅ 实例创建成功")
        
        # 测试初始化
        print("\n初始化模型...")
        init_start = time.time()
        inference_engine.initialize()
        init_time = time.time() - init_start
        
        print(f"✅ 模型初始化成功！耗时: {init_time:.2f}秒")
        
        # 验证设备信息
        device_info = inference_engine.get_device_info()
        print(f"\n📊 设备信息:")
        for key, value in device_info.items():
            print(f"  {key}: {value}")
        
        # 验证模型确实在XPU上
        if device_info.get('device') == 'xpu':
            print("✅ 模型确实在XPU上运行")
            return True
        else:
            print(f"❌ 模型在 {device_info.get('device')} 上运行，不是XPU")
            return False
            
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_inference():
    """测试简单推理"""
    print_separator("🧠 简单推理测试")
    
    try:
        from src.chat_with_video.model_loader import MiniCPMVInference
        
        # 创建推理引擎实例（复用已初始化的）
        inference_engine = MiniCPMVInference(
            model_path='openbmb/MiniCPM-V-4_5-int4',
            device='xpu'
        )
        
        # 初始化模型
        inference_engine.initialize()
        
        # 测试简单推理
        print("测试简单推理...")
        test_msgs = [
            {'role': 'user', 'content': ['Hello, XPU!']}
        ]
        
        inference_start = time.time()
        response = inference_engine.chat(
            msgs=test_msgs,
            max_new_tokens=20,
            do_sample=False
        )
        inference_time = time.time() - inference_start
        
        print(f"✅ 推理测试成功！")
        print(f"   推理耗时: {inference_time:.2f}秒")
        print(f"   回答: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ 简单推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print_separator("🧪 模型加载测试套件")
    
    success_count = 0
    total_tests = 2
    
    # 测试1: 模型加载
    if test_model_loading():
        success_count += 1
    
    # 测试2: 简单推理
    if test_simple_inference():
        success_count += 1
    
    # 总结
    print_separator("📊 测试结果总结")
    print(f"通过测试: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 所有模型测试通过！")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)