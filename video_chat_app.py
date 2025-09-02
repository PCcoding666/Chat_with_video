#!/usr/bin/env python3
"""
完整的MiniCPM-V视频聊天Gradio应用
使用完全延迟加载策略
"""

import os
import gradio as gr
import time
from typing import Optional, Any, Dict, List, Tuple

# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

class LazyVideoChatApp:
    """延迟加载的视频聊天应用"""
    
    def __init__(self):
        self.service = None
        self.current_video_data = None
        
    def initialize_service(self, max_frames: int = 180, max_packing: int = 3) -> str:
        """初始化服务（延迟加载）"""
        try:
            # 只有在需要时才导入
            from src.chat_with_video.video_chat_service import VideoChatService
            
            print("正在初始化视频聊天服务...")
            self.service = VideoChatService(
                model_path='openbmb/MiniCPM-V-4_5',
                device='xpu',
                max_frames=max_frames,
                max_packing=max_packing
            )
            
            # 初始化服务
            success = self.service.initialize()
            
            if success:
                info = self.service.get_system_info()
                return f"""✅ 服务初始化成功！

🔧 系统配置:
- 模型: {info.get('model_path', 'N/A')}
- 设备: {info.get('device', 'N/A')} 
- XPU可用: {info.get('xpu_available', False)}
- 设备数量: {info.get('device_count', 0)}

📹 3D重采样器配置:
- 最大帧数: {info['video_encoder_config']['max_frames']}
- 最大打包数: {info['video_encoder_config']['max_packing']}

🚀 系统已就绪！"""
            else:
                return "❌ 服务初始化失败"
                
        except Exception as e:
            import traceback
            return f"❌ 初始化失败: {str(e)}\n\n详细信息:\n{traceback.format_exc()}"
    
    def process_video(self, video_file, fps: int = 5, force_packing: Optional[int] = None) -> str:
        """处理视频"""
        if not self.service:
            return "❌ 请先初始化服务"
        
        if not video_file:
            return "❌ 请上传视频文件"
        
        try:
            start_time = time.time()
            
            frames, temporal_ids = self.service.process_video(
                video_path=video_file,
                choose_fps=fps,
                force_packing=force_packing if force_packing and force_packing > 0 else None
            )
            
            self.current_video_data = (frames, temporal_ids)
            process_time = time.time() - start_time
            
            return f"""✅ 视频处理完成！

📊 处理结果:
- 提取帧数: {len(frames)}
- 时序组数: {len(temporal_ids)} 
- 处理耗时: {process_time:.2f}秒

💬 现在可以开始聊天了！"""
            
        except Exception as e:
            return f"❌ 视频处理失败: {str(e)}"
    
    def chat_with_video(self, question: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """与视频聊天"""
        if not self.service:
            return "❌ 请先初始化服务"
            
        if not self.current_video_data:
            return "❌ 请先处理视频"
            
        if not question.strip():
            return "❌ 请输入问题"
        
        try:
            frames, temporal_ids = self.current_video_data
            start_time = time.time()
            
            # 调用服务进行推理
            answer = self.service.chat_with_frames(
                frames=frames,
                temporal_ids=temporal_ids,
                question=question,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            
            inference_time = time.time() - start_time
            return f"""🤖 AI回答:
{answer}

⏱️ 推理耗时: {inference_time:.2f}秒"""
            
        except Exception as e:
            return f"❌ 聊天失败: {str(e)}"


def create_app():
    """创建Gradio应用"""
    app_instance = LazyVideoChatApp()
    
    with gr.Blocks(title="MiniCPM-V 视频聊天", theme=gr.themes.Default()) as interface:
        gr.Markdown("# 🎥 MiniCPM-V 视频聊天 Demo - Intel GPU版本")
        
        with gr.Tab("📋 系统控制"):
            gr.Markdown("## 🚀 系统初始化")
            
            with gr.Row():
                with gr.Column(scale=1):
                    max_frames = gr.Slider(50, 300, value=180, step=10, label="最大帧数")
                    max_packing = gr.Slider(1, 6, value=3, step=1, label="最大打包数")
                    init_btn = gr.Button("🔧 初始化服务", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    init_status = gr.Textbox(
                        label="初始化状态",
                        lines=15,
                        interactive=False,
                        value="点击'初始化服务'开始..."
                    )
            
            init_btn.click(
                fn=app_instance.initialize_service,
                inputs=[max_frames, max_packing],
                outputs=[init_status]
            )
        
        with gr.Tab("📹 视频处理"):
            gr.Markdown("## 📤 视频上传与处理")
            
            with gr.Row():
                with gr.Column(scale=1):
                    video_file = gr.File(
                        label="选择视频文件",
                        file_types=["video"]
                    )
                    fps_slider = gr.Slider(1, 10, value=5, step=1, label="采样帧率 (FPS)")
                    force_packing_num = gr.Number(
                        label="强制打包数量 (0=自动)",
                        value=0,
                        minimum=0,
                        maximum=6
                    )
                    process_btn = gr.Button("🎬 处理视频", variant="primary")
                
                with gr.Column(scale=2):
                    process_status = gr.Textbox(
                        label="处理状态",
                        lines=10,
                        interactive=False
                    )
            
            process_btn.click(
                fn=app_instance.process_video,
                inputs=[video_file, fps_slider, force_packing_num],
                outputs=[process_status]
            )
        
        with gr.Tab("💬 视频聊天"):
            gr.Markdown("## 🤖 与视频进行智能对话")
            
            with gr.Row():
                with gr.Column(scale=1):
                    question_input = gr.Textbox(
                        label="输入问题",
                        placeholder="请描述这个视频的内容...",
                        lines=3
                    )
                    
                    with gr.Row():
                        max_tokens = gr.Slider(512, 4096, value=2048, step=128, label="最大生成长度")
                        temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="创造性")
                    
                    chat_btn = gr.Button("💭 开始聊天", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    chat_output = gr.Textbox(
                        label="AI回答",
                        lines=15,
                        interactive=False
                    )
            
            chat_btn.click(
                fn=app_instance.chat_with_video,
                inputs=[question_input, max_tokens, temperature],
                outputs=[chat_output]
            )
        
        with gr.Tab("ℹ️ 说明"):
            gr.Markdown("""
            ## 📖 使用说明
            
            ### 步骤1: 系统初始化
            1. 在"系统控制"标签页中调整参数
            2. 点击"初始化服务"按钮
            3. 等待模型加载完成（首次使用需要下载约8GB模型）
            
            ### 步骤2: 视频处理  
            1. 在"视频处理"标签页上传视频文件
            2. 调整采样帧率和打包参数
            3. 点击"处理视频"按钮
            
            ### 步骤3: 开始聊天
            1. 在"视频聊天"标签页输入问题
            2. 调整生成参数
            3. 点击"开始聊天"按钮
            
            ## 🎯 支持的功能
            - 视频内容描述
            - 细节提取
            - 情绪分析  
            - 人物识别
            - 场景理解
            
            ## ⚡ Intel Arc GPU加速
            - 使用Intel Arc 130V GPU进行推理
            - 支持3D重采样器优化
            - bfloat16精度优化显存使用
            """)
    
    return interface


if __name__ == "__main__":
    print("🌐 启动MiniCPM-V视频聊天应用...")
    
    try:
        interface = create_app()
        
        print("✅ 界面创建成功")
        print("🚀 正在启动Gradio服务...")
        print("📍 访问地址: http://localhost:7861")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")
        import traceback
        traceback.print_exc()