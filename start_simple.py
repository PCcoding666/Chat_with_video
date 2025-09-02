#!/usr/bin/env python3
"""
简化的启动脚本
"""

import os
# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

print("🌐 正在启动视频聊天Web界面...")

try:
    print("Step 1: 导入模块...")
    from src.chat_with_video.gradio_app import VideoChatGradioApp
    print("✓ 模块导入成功")
    
    # 创建应用
    print("Step 2: 创建应用实例...")
    app = VideoChatGradioApp()
    print("✓ 应用实例创建成功")
    
    # 启动应用
    print("Step 3: 启动Gradio界面...")
    print("地址: http://0.0.0.0:7860")
    
    print("正在启动服务器...")
    app.launch(
        server_name="0.0.0.0",
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