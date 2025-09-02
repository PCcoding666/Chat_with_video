#!/usr/bin/env python3
"""
XPU测试配置和工具
提供统一的测试环境配置和工具函数
"""

import os
import sys

# 统一的环境变量设置
def setup_test_environment():
    """设置测试环境变量"""
    env_vars = {
        'DISABLE_TRITON': '1',
        'USE_TRITON': '0',
        'TRITON_DISABLE': '1',
        'DISABLE_FLASH_ATTN': '1',
        'PYTORCH_DISABLE_TRITON': '1'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value

# 统一的项目路径设置
def setup_project_path():
    """设置项目路径"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# 打印分隔符
def print_separator(title, width=60):
    """打印分隔符"""
    print("\n" + "="*width)
    print(f" {title} ")
    print("="*width)

# 检查XPU可用性
def check_xpu_availability():
    """检查XPU可用性"""
    try:
        import torch
        return torch.xpu.is_available()
    except:
        return False

# 获取XPU设备信息
def get_xpu_info():
    """获取XPU设备信息"""
    try:
        import torch
        if torch.xpu.is_available():
            return {
                'available': True,
                'device_count': torch.xpu.device_count(),
                'current_device': torch.xpu.current_device()
            }
        else:
            return {'available': False}
    except Exception as e:
        return {'available': False, 'error': str(e)}