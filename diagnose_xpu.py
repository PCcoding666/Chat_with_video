#!/usr/bin/env python3
"""
Intel XPU PyTorch 诊断工具
"""

import os
import sys
import subprocess

# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

def check_system_info():
    """检查系统信息"""
    print("=== 系统信息检查 ===")
    print(f"Python版本: {sys.version}")
    print(f"平台: {sys.platform}")
    
    # 检查GPU信息
    try:
        result = subprocess.run(
            ['powershell', '-Command', 'Get-WmiObject Win32_VideoController | Select-Object Name, DriverVersion'],
            capture_output=True, text=True
        )
        print("GPU信息:")
        print(result.stdout)
    except Exception as e:
        print(f"无法获取GPU信息: {e}")


def check_torch_installation():
    """检查PyTorch安装"""
    print("\n=== PyTorch安装检查 ===")
    
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        
        # 检查是否有XPU支持
        if hasattr(torch, 'xpu'):
            print("✓ 发现XPU模块")
            
            try:
                # 尝试检查XPU设备
                if torch.xpu.is_available():
                    print(f"✓ XPU可用，设备数量: {torch.xpu.device_count()}")
                    
                    # 获取设备信息
                    for i in range(torch.xpu.device_count()):
                        device_name = torch.xpu.get_device_name(i)
                        print(f"  设备 {i}: {device_name}")
                else:
                    print("❌ XPU不可用")
            except Exception as e:
                print(f"❌ XPU检查失败: {e}")
        else:
            print("❌ 没有XPU模块")
            
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
    except Exception as e:
        print(f"❌ PyTorch检查失败: {e}")


def check_dependencies():
    """检查依赖库"""
    print("\n=== 依赖库检查 ===")
    
    required_libs = [
        'numpy', 'PIL', 'cv2', 'transformers', 
        'gradio', 'accelerate', 'decord'
    ]
    
    for lib in required_libs:
        try:
            if lib == 'PIL':
                import PIL
                print(f"✓ {lib}: {PIL.__version__}")
            elif lib == 'cv2':
                import cv2
                print(f"✓ {lib}: {cv2.__version__}")
            else:
                module = __import__(lib)
                version = getattr(module, '__version__', 'unknown')
                print(f"✓ {lib}: {version}")
        except ImportError:
            print(f"❌ {lib}: 未安装")
        except Exception as e:
            print(f"⚠️ {lib}: {e}")


def check_dll_files():
    """检查关键DLL文件"""
    print("\n=== DLL文件检查 ===")
    
    # 检查虚拟环境中的torch库
    try:
        import torch
        torch_path = torch.__file__
        torch_dir = os.path.dirname(torch_path)
        lib_dir = os.path.join(torch_dir, 'lib')
        
        print(f"PyTorch安装路径: {torch_dir}")
        print(f"库文件目录: {lib_dir}")
        
        if os.path.exists(lib_dir):
            dll_files = [f for f in os.listdir(lib_dir) if f.endswith('.dll')]
            print(f"发现 {len(dll_files)} 个DLL文件:")
            
            critical_dlls = ['c10_xpu.dll', 'torch_xpu.dll', 'intel_xpu_backend.dll']
            
            for dll in critical_dlls:
                dll_path = os.path.join(lib_dir, dll)
                if os.path.exists(dll_path):
                    print(f"  ✓ {dll}")
                else:
                    print(f"  ❌ {dll} (缺失)")
            
            # 显示所有XPU相关的DLL
            xpu_dlls = [f for f in dll_files if 'xpu' in f.lower()]
            if xpu_dlls:
                print("XPU相关DLL:")
                for dll in xpu_dlls:
                    print(f"  - {dll}")
            else:
                print("❌ 没有找到XPU相关DLL")
        else:
            print(f"❌ 库目录不存在: {lib_dir}")
            
    except Exception as e:
        print(f"❌ DLL检查失败: {e}")


def check_environment_vars():
    """检查环境变量"""
    print("\n=== 环境变量检查 ===")
    
    important_vars = [
        'DISABLE_TRITON', 'USE_TRITON', 'TRITON_DISABLE',
        'DISABLE_FLASH_ATTN', 'PYTORCH_DISABLE_TRITON',
        'INTEL_GPU_PATH', 'ONEAPI_ROOT', 'PATH'
    ]
    
    for var in important_vars:
        value = os.environ.get(var, 'Not Set')
        if var == 'PATH':
            # 只显示Intel相关的PATH
            paths = value.split(';') if value != 'Not Set' else []
            intel_paths = [p for p in paths if 'intel' in p.lower()]
            if intel_paths:
                print(f"{var} (Intel相关):")
                for path in intel_paths:
                    print(f"  - {path}")
            else:
                print(f"{var}: 没有Intel相关路径")
        else:
            print(f"{var}: {value}")


def main():
    """主函数"""
    print("Intel XPU PyTorch 诊断工具")
    print("=" * 50)
    
    check_system_info()
    check_environment_vars()
    check_dependencies()
    check_dll_files()
    check_torch_installation()
    
    print("\n=== 诊断完成 ===")
    print("请检查上述信息，如有问题请根据错误信息进行修复。")


if __name__ == "__main__":
    main()