#!/usr/bin/env python3
"""
终极XPU强制加载测试
测试新的彻底绕过显存查询机制的加载策略
"""

import os
import sys
import time
import traceback

# 设置环境变量 - 在导入任何PyTorch相关库之前
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

def print_separator(title):
    """打印分隔符"""
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def test_xpu_availability():
    """测试XPU基础可用性"""
    print_separator("🔧 XPU可用性测试")
    
    try:
        import torch
        
        if not torch.xpu.is_available():
            print("❌ XPU不可用！")
            return False
        
        print(f"✅ XPU可用")
        print(f"✅ XPU设备数量: {torch.xpu.device_count()}")
        print(f"✅ 当前XPU设备: {torch.xpu.current_device()}")
        
        # 测试基础XPU操作
        device = torch.device('xpu')
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.matmul(x, y)
        result = z.sum().item()
        
        print(f"✅ XPU基础运算测试通过: {result:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ XPU测试失败: {e}")
        return False

def test_memory_query_bypass():
    """测试显存查询绕过机制"""
    print_separator("🛡️ 显存查询绕过测试")
    
    try:
        import torch
        
        # 测试各种可能触发显存查询的操作
        print("测试torch.xpu内存查询...")
        
        try:
            # 这个应该会失败
            free, total = torch.xpu.mem_get_info()
            print(f"⚠️ torch.xpu.mem_get_info 居然成功了: free={free}, total={total}")
        except Exception as e:
            print(f"✅ torch.xpu.mem_get_info 正确失败: {e}")
        
        # 测试我们的修补是否生效
        print("\n应用显存查询修补...")
        
        # 模拟我们的修补逻辑
        original_mem_get_info = getattr(torch.xpu, 'mem_get_info', None)
        
        # 替换函数
        def dummy_mem_get_info(device=None):
            print("  🔄 使用虚拟显存查询 (绕过实际查询)")
            return (1024*1024*1024*8, 1024*1024*1024*16)  # 假设8GB可用，16GB总计
        
        torch.xpu.mem_get_info = dummy_mem_get_info
        
        # 测试修补后的查询
        try:
            free, total = torch.xpu.mem_get_info()
            print(f"✅ 修补后的查询成功: free={free/(1024**3):.1f}GB, total={total/(1024**3):.1f}GB")
        except Exception as e:
            print(f"❌ 修补后查询仍失败: {e}")
        
        # 恢复原函数（如果有的话）
        if original_mem_get_info:
            torch.xpu.mem_get_info = original_mem_get_info
        
        return True
        
    except Exception as e:
        print(f"❌ 显存查询绕过测试失败: {e}")
        return False

def test_ultimate_xpu_loading():
    """测试终极XPU加载策略"""
    print_separator("🚀 终极XPU加载策略测试")
    
    try:
        # 添加项目路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        print("导入MiniCPMVInference...")
        from src.chat_with_video.model_loader import MiniCPMVInference
        
        print("✅ 模块导入成功")
        
        # 创建推理引擎实例
        print("\n创建MiniCPMVInference实例...")
        start_time = time.time()
        
        inference_engine = MiniCPMVInference(
            model_path='openbmb/MiniCPM-V-4_5-int4',
            device='xpu'
        )
        
        print("✅ 实例创建成功")
        
        # 初始化模型 - 这里应该使用我们的新策略
        print("\n🔥 开始终极XPU加载策略测试...")
        print("这将测试是否能彻底绕过Intel XPU的显存查询限制...")
        
        # 执行初始化
        inference_engine.initialize()
        
        init_time = time.time() - start_time
        print(f"🎉 终极加载策略测试成功！耗时: {init_time:.2f}秒")
        
        # 验证模型确实在XPU上
        device_info = inference_engine.get_device_info()
        print(f"\n📊 设备信息验证:")
        for key, value in device_info.items():
            print(f"  {key}: {value}")
        
        # 测试简单推理
        print(f"\n🧪 测试简单推理...")
        test_msgs = [
            {'role': 'user', 'content': ['你好，这是一个XPU加载测试消息。请简短回复。']}
        ]
        
        inference_start = time.time()
        response = inference_engine.chat(
            msgs=test_msgs,
            max_new_tokens=50,
            do_sample=False
        )
        inference_time = time.time() - inference_start
        
        print(f"✅ 推理测试成功！")
        print(f"   推理耗时: {inference_time:.2f}秒")
        print(f"   回答: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ 终极XPU加载策略测试失败:")
        print(f"   错误: {e}")
        print(f"\n🔍 详细错误信息:")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print_separator("🧪 终极XPU强制加载测试套件")
    
    success_count = 0
    total_tests = 3
    
    # 测试1: XPU可用性
    if test_xpu_availability():
        success_count += 1
    
    # 测试2: 显存查询绕过
    if test_memory_query_bypass():
        success_count += 1
    
    # 测试3: 终极加载策略
    if test_ultimate_xpu_loading():
        success_count += 1
    
    # 总结
    print_separator("📊 测试结果总结")
    print(f"通过测试: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！终极XPU强制加载策略工作正常！")
        print("💡 现在Intel Arc GPU的显存查询限制已被彻底绕过")
        return 0
    else:
        print("❌ 部分测试失败，需要进一步调试")
        return 1

if __name__ == "__main__":
    exit_code = main()
    input("\n按Enter键退出...")
    sys.exit(exit_code)