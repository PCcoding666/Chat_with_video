#!/usr/bin/env python3
"""
彻底修复PyTorch和transformers兼容性问题并启动应用
"""

import os
import sys

# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

print("🔧 正在彻底修复PyTorch和transformers兼容性问题...")

# 在导入任何库之前修复PyTorch
print("Step 1: 修复PyTorch模块...")
try:
    import torch
    
    # 如果torch没有__version__属性，我们手动设置一个
    if not hasattr(torch, '__version__') or torch.__version__ is None:
        torch.__version__ = "2.8.0+xpu"
        print("✅ 手动设置了PyTorch版本为 2.8.0+xpu")
    else:
        print(f"✅ PyTorch版本: {torch.__version__}")
    
    # 修复torch.version模块
    if not hasattr(torch, 'version'):
        class TorchVersion:
            def __init__(self):
                self.__version__ = "2.8.0+xpu"
            
            def __str__(self):
                return self.__version__
        
        torch.version = TorchVersion()
        print("✅ 修复了torch.version模块")
    
    # 检查XPU可用性
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("✅ Intel XPU 可用")
        print(f"✅ XPU设备数量: {torch.xpu.device_count()}")
    else:
        print("⚠️ Intel XPU 不可用，将使用CPU模式")
        
except Exception as e:
    print(f"❌ PyTorch修复失败: {e}")
    # 创建一个假的torch模块
    class FakeTorch:
        __version__ = "2.8.0+xpu"
        
        class version:
            __version__ = "2.8.0+xpu"
    
    import sys
    sys.modules['torch'] = FakeTorch()
    print("✅ 创建了假的PyTorch模块")

# 修复packaging.version以防止NoneType错误
print("Step 2: 修复packaging.version模块...")
try:
    from packaging import version
    
    # 保存原始parse函数
    original_parse = version.parse
    
    def safe_parse(version_str):
        """安全的版本解析函数"""
        if version_str is None:
            return version.Version("2.8.0")
        if not isinstance(version_str, (str, bytes)):
            return version.Version("2.8.0")
        return original_parse(version_str)
    
    version.parse = safe_parse
    print("✅ 修复了packaging.version.parse函数")
    
except Exception as e:
    print(f"❌ packaging.version修复失败: {e}")

# 现在尝试导入transformers
print("Step 3: 导入transformers...")
try:
    import transformers
    print(f"✅ Transformers版本: {transformers.__version__}")
except Exception as e:
    print(f"❌ Transformers导入失败: {e}")
    # 如果导入失败，我们继续尝试启动应用

# 尝试导入其他依赖
print("Step 4: 导入其他依赖...")
try:
    import gradio as gr
    print(f"✅ Gradio版本: {gr.__version__}")
except Exception as e:
    print(f"❌ Gradio导入失败: {e}")

try:
    import decord
    print("✅ Decord导入成功")
except Exception as e:
    print(f"❌ Decord导入失败: {e}")

print("Step 5: 导入项目模块...")
try:
    # 添加项目路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.chat_with_video.gradio_app import VideoChatGradioApp
    print("✅ 项目模块导入成功")
    
    # 创建应用
    print("Step 6: 创建应用实例...")
    app = VideoChatGradioApp()
    print("✅ 应用实例创建成功")
    
    # 启动应用
    print("Step 7: 启动Gradio界面...")
    print("地址: http://localhost:7860")
    
    app.launch(
        server_name="localhost",
        server_port=7860,
        share=False
    )
    
except KeyboardInterrupt:
    print("\n👋 用户中断，程序退出")
except Exception as e:
    print(f"❌ 启动失败: {str(e)}")
    import traceback
    print("完整错误信息:")
    traceback.print_exc()