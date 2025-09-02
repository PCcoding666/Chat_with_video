#!/usr/bin/env python3
"""
XPU基础功能测试
测试XPU的基础功能和环境配置
"""

import sys
import torch
from .test_utils import setup_test_environment, setup_project_path, print_separator, check_xpu_availability, get_xpu_info

# 设置测试环境
setup_test_environment()
setup_project_path()

def test_xpu_basic():
    """测试XPU基础功能"""
    print_separator("🔧 XPU基础功能测试")
    
    # 检查XPU可用性
    if not check_xpu_availability():
        print("❌ XPU不可用")
        return False
    
    print("✅ XPU可用")
    
    # 获取XPU信息
    xpu_info = get_xpu_info()
    print(f"✅ XPU设备数量: {xpu_info['device_count']}")
    print(f"✅ 当前XPU设备: {xpu_info['current_device']}")
    
    # 测试基础XPU操作
    try:
        device = torch.device('xpu')
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.matmul(x, y)
        result = z.sum().item()
        print(f"✅ XPU基础运算测试通过: {result:.2f}")
        return True
    except Exception as e:
        print(f"❌ XPU基础运算测试失败: {e}")
        return False

def test_environment_variables():
    """测试环境变量设置"""
    print_separator("⚙️ 环境变量测试")
    
    required_env_vars = [
        'DISABLE_TRITON',
        'USE_TRITON', 
        'TRITON_DISABLE',
        'DISABLE_FLASH_ATTN',
        'PYTORCH_DISABLE_TRITON'
    ]
    
    all_set = True
    for var in required_env_vars:
        value = os.environ.get(var, None)
        if value is None:
            print(f"❌ 环境变量 {var} 未设置")
            all_set = False
        else:
            print(f"✅ 环境变量 {var} = {value}")
    
    return all_set

def test_project_imports():
    """测试项目模块导入"""
    print_separator("📦 项目模块导入测试")
    
    try:
        from src.chat_with_video.model_loader import MiniCPMVInference
        print("✅ model_loader 模块导入成功")
        
        from src.chat_with_video.video_chat_service import VideoChatService
        print("✅ video_chat_service 模块导入成功")
        
        from src.chat_with_video.gradio_app import VideoChatGradioApp
        print("✅ gradio_app 模块导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 项目模块导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print_separator("🧪 XPU基础功能测试套件")
    
    success_count = 0
    total_tests = 3
    
    # 测试1: XPU基础功能
    if test_xpu_basic():
        success_count += 1
    
    # 测试2: 环境变量
    if test_environment_variables():
        success_count += 1
    
    # 测试3: 项目模块导入
    if test_project_imports():
        success_count += 1
    
    # 总结
    print_separator("📊 测试结果总结")
    print(f"通过测试: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 所有基础测试通过！")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)