#!/usr/bin/env python3
"""
Intel XPU 环境设置脚本
尝试解决 PyTorch XPU DLL 依赖问题
"""

import os
import sys
import subprocess

def setup_intel_environment():
    """设置Intel XPU所需的环境变量"""
    print("🔧 设置Intel XPU环境变量...")
    
    # 设置Intel GPU相关环境变量
    env_vars = {
        # PyTorch XPU相关
        'DISABLE_TRITON': '1',
        'USE_TRITON': '0', 
        'TRITON_DISABLE': '1',
        'DISABLE_FLASH_ATTN': '1',
        'PYTORCH_DISABLE_TRITON': '1',
        
        # Intel GPU驱动程序路径
        'INTEL_GRAPHICS_PATH': r'C:\Program Files\Intel\Intel Graphics Software',
        'INTEL_GPU_DRIVER_PATH': r'C:\WINDOWS\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_1ffd93357d60fb67',
        
        # Level Zero 相关
        'ZE_ENABLE_LAYERS': '1',
        'ZE_LOADER_DEBUG': '1',
        
        # OpenCL 相关  
        'INTEL_OPENCL_ICD': r'C:\WINDOWS\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_1ffd93357d60fb67\Intel_OpenCL_ICD64.dll',
        
        # SYCL 相关
        'SYCL_CACHE_PERSISTENT': '1',
        'SYCL_DEVICE_ALLOWLIST': 'level_zero:gpu',
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  ✓ {key} = {value}")
    
    # 添加Intel GPU运行时路径到PATH
    intel_paths = [
        r'C:\Program Files\Intel\Intel Graphics Software',
        r'C:\WINDOWS\System32\DriverStore\FileRepository\iigd_dch.inf_amd64_1ffd93357d60fb67',
        r'C:\WINDOWS\System32',
        r'C:\WINDOWS\SysWOW64',
    ]
    
    current_path = os.environ.get('PATH', '')
    for path in intel_paths:
        if os.path.exists(path) and path not in current_path:
            os.environ['PATH'] = path + ';' + os.environ['PATH']
            print(f"  ✓ 添加到PATH: {path}")

def test_pytorch_import():
    """测试PyTorch XPU导入"""
    print("\n🧪 测试PyTorch XPU导入...")
    
    try:
        print("1. 导入torch...")
        import torch
        print(f"  ✓ PyTorch版本: {torch.__version__}")
        
        print("2. 检查XPU可用性...")
        if hasattr(torch, 'xpu'):
            print("  ✓ torch.xpu模块存在")
            
            try:
                is_available = torch.xpu.is_available()
                print(f"  ✓ XPU可用: {is_available}")
                
                if is_available:
                    device_count = torch.xpu.device_count()
                    print(f"  ✓ XPU设备数量: {device_count}")
                    
                    # 测试基本操作
                    print("3. 测试XPU基本操作...")
                    x = torch.tensor([1.0, 2.0, 3.0]).to('xpu')
                    result = x.sum()
                    print(f"  ✓ XPU张量计算成功: {result.item()}")
                    
                    return True
                else:
                    print("  ❌ XPU不可用")
                    return False
                    
            except Exception as e:
                print(f"  ❌ XPU操作失败: {e}")
                return False
        else:
            print("  ❌ torch.xpu模块不存在")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 Intel XPU环境设置和测试")
    print("=" * 50)
    
    # 设置环境
    setup_intel_environment()
    
    # 测试导入
    success = test_pytorch_import()
    
    if success:
        print("\n🎉 Intel XPU环境设置成功！")
        print("现在可以尝试运行你的应用了。")
    else:
        print("\n❌ Intel XPU环境设置失败！")
        print("可能需要安装Intel oneAPI运行时。")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)