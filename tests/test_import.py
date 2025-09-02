#!/usr/bin/env python3
"""
简单的导入测试脚本
"""

import os
# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

print("=== 开始导入测试 ===")

try:
    print("1. 导入基础模块...")
    import sys
    import argparse
    import time
    from pathlib import Path
    from typing import List, Optional
    print("   ✓ 基础模块导入成功")
except Exception as e:
    print(f"   ❌ 基础模块导入失败: {e}")
    exit(1)

try:
    print("2. 导入PyTorch...")
    import torch
    print(f"   ✓ PyTorch导入成功: {torch.__version__}")
    print(f"   ✓ XPU可用: {torch.xpu.is_available()}")
except Exception as e:
    print(f"   ❌ PyTorch导入失败: {e}")
    exit(1)

try:
    print("3. 导入gradio...")
    import gradio as gr
    print(f"   ✓ Gradio导入成功: {gr.__version__}")
except Exception as e:
    print(f"   ❌ Gradio导入失败: {e}")
    exit(1)

try:
    print("4. 导入项目模块...")
    from src.chat_with_video.gradio_app import VideoChatGradioApp
    print("   ✓ gradio_app导入成功")
except Exception as e:
    print(f"   ❌ gradio_app导入失败: {e}")
    print(f"   错误详情: {type(e).__name__}: {e}")
    
    # 尝试导入依赖模块
    try:
        print("   尝试导入video_chat_service...")
        from src.chat_with_video.video_chat_service import VideoChatService
        print("   ✓ video_chat_service导入成功")
    except Exception as e2:
        print(f"   ❌ video_chat_service导入失败: {e2}")
    
    exit(1)

print("\n=== 所有导入测试通过 ===")
print("尝试创建应用实例...")

try:
    app = VideoChatGradioApp()
    print("✓ 应用实例创建成功")
    
    print("尝试打印横幅...")
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                   MiniCPM-V 视频聊天 Demo                   ║
║                     Intel Arc GPU 版本                      ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    
except Exception as e:
    print(f"❌ 应用创建失败: {e}")
    exit(1)

print("\n🎉 测试完成，应用ready!")