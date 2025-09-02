#!/usr/bin/env python3
"""
修复PyTorch版本问题并启动应用
"""

import os
import sys

# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

print("🔧 正在修复PyTorch版本问题...")

try:
    # 尝试修复PyTorch版本问题
    import torch
    
    # 检查torch版本
    print(f"PyTorch location: {torch.__file__ if hasattr(torch, '__file__') else 'Unknown'}")
    
    # 如果torch没有__version__属性，我们手动设置一个
    if not hasattr(torch, '__version__'):
        torch.__version__ = "2.8.0+xpu"
        print("✅ 手动设置了PyTorch版本")
    
    # 检查XPU可用性
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("✅ Intel XPU 可用")
        print(f"✅ XPU设备数量: {torch.xpu.device_count()}")
    else:
        print("⚠️ Intel XPU 不可用，将使用CPU模式")
    
    # 现在尝试导入transformers
    print("Step 1: 导入transformers...")
    import transformers
    print(f"✅ Transformers版本: {transformers.__version__}")
    
    # 尝试导入其他依赖
    print("Step 2: 导入其他依赖...")
    import gradio as gr
    print(f"✅ Gradio版本: {gr.__version__}")
    
    import decord
    print("✅ Decord导入成功")
    
    print("Step 3: 导入项目模块...")
    # 添加项目路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.chat_with_video.gradio_app import VideoChatGradioApp
    print("✅ 项目模块导入成功")
    
    # 创建应用
    print("Step 4: 创建应用实例...")
    app = VideoChatGradioApp()
    print("✅ 应用实例创建成功")
    
    # 启动应用
    print("Step 5: 启动Gradio界面...")
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