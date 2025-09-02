#!/usr/bin/env python3
"""
简单的Gradio测试
"""

import os
# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

import gradio as gr

def simple_chat(message):
    """简单的聊天函数"""
    return f"您说: {message}\n\nAI回复: 这是一个测试回复。当前系统已经准备就绪，等待集成视频聊天功能！"

def create_interface():
    """创建简单的界面"""
    
    with gr.Blocks(title="视频聊天测试界面") as interface:
        gr.Markdown("# MiniCPM-V 视频聊天 Demo - Intel GPU版本")
        gr.Markdown("## 🚀 系统状态测试")
        
        with gr.Row():
            with gr.Column():
                message_input = gr.Textbox(
                    label="测试消息",
                    placeholder="输入测试消息...",
                    lines=2
                )
                submit_btn = gr.Button("发送", variant="primary")
            
            with gr.Column():
                output = gr.Textbox(
                    label="响应",
                    lines=5,
                    interactive=False
                )
        
        submit_btn.click(
            fn=simple_chat,
            inputs=[message_input],
            outputs=[output]
        )
        
        # 添加说明
        gr.Markdown("""
        ### 📝 说明
        这是一个简化的测试界面，用于验证Gradio是否正常工作。
        
        ### 🔧 下一步
        - ✅ Gradio界面正常
        - ⏳ 集成视频聊天功能
        - ⏳ Intel XPU模型推理
        """)
    
    return interface

if __name__ == "__main__":
    print("🌐 启动简单Gradio测试界面...")
    
    try:
        # 创建界面
        interface = create_interface()
        
        # 启动服务
        print("正在启动Gradio服务...")
        print("地址: http://localhost:7860")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
        
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")
        import traceback
        traceback.print_exc()