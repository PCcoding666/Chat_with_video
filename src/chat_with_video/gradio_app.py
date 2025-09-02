"""
Gradio Web界面 - MiniCPM-V视频聊天
支持视频上传、实时聊天和3D重采样器参数调节
基于Intel XPU的高效推理
"""

import os
import gradio as gr
import time
from typing import Optional, List, Tuple, Any

# 延迟导入以避免在应用启动时就开始加载模型
# from .video_chat_service import VideoChatService


class VideoChatGradioApp:
    """Gradio视频聊天应用"""
    
    def __init__(self, 
                 model_path: str = 'openbmb/MiniCPM-V-4_5-int4',
                 device: str = 'xpu'):
        """
        初始化Gradio应用
        
        Args:
            model_path: 模型路径 (INT4量化版本)
            device: 设备类型
        """
        self.model_path = model_path
        self.device = device
        self.service: Optional[VideoChatService] = None
        self.current_video_data: Optional[Tuple] = None  # 缓存当前视频的帧和时序ID
        
        print(f"Gradio视频聊天应用初始化:")
        print(f"  - 模型: {model_path}")
        print(f"  - 设备: {device}")
    
    def initialize_service(self, 
                          max_frames: int = 180, 
                          max_packing: int = 3, 
                          time_scale: float = 0.1) -> str:
        """
        初始化视频聊天服务
        
        Args:
            max_frames: 最大帧数
            max_packing: 最大打包数
            time_scale: 时间缩放因子
            
        Returns:
            初始化状态信息
        """
        try:
            # 延迟导入VideoChatService
            from .video_chat_service import VideoChatService
            
            # 创建服务实例
            self.service = VideoChatService(
                model_path=self.model_path,
                device=self.device,
                max_frames=max_frames,
                max_packing=max_packing,
                time_scale=time_scale
            )
            
            # 初始化服务
            success = self.service.initialize()
            
            if success:
                info = self.service.get_system_info()
                status_text = f"""
✅ 服务初始化成功！

🔧 系统配置:
- 模型: {info.get('model_path', 'N/A')}
- 设备: {info.get('device', 'N/A')}
- XPU可用: {info.get('xpu_available', False)}
- 设备数量: {info.get('device_count', 0)}

📹 3D重采样器配置:
- 最大帧数: {info['video_encoder_config']['max_frames']}
- 最大打包数: {info['video_encoder_config']['max_packing']}
- 时间缩放: {info['video_encoder_config']['time_scale']}

🚀 系统已就绪，可以开始视频聊天！
                """
                return status_text.strip()
            else:
                return "❌ 服务初始化失败，请检查设备和模型配置"
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"❌ 初始化错误: {str(e)}\n\n详细错误:\n{error_details}"
    
    def process_video_upload(self, video_file, fps: int, force_packing: Optional[int]) -> str:
        """
        处理视频上传
        
        Args:
            video_file: 上传的视频文件
            fps: 采样帧率
            force_packing: 强制打包数量
            
        Returns:
            处理状态信息
        """
        if not self.service:
            return "❌ 服务未初始化，请先点击'初始化服务'按钮"
        
        if not video_file:
            return "❌ 请先上传视频文件"
        
        try:
            start_time = time.time()
            
            # 处理视频
            frames, temporal_ids = self.service.process_video(
                video_path=video_file,
                choose_fps=fps,
                force_packing=force_packing if force_packing > 0 else None
            )
            
            # 缓存视频数据
            self.current_video_data = (frames, temporal_ids)
            
            process_time = time.time() - start_time
            
            status_text = f"""
✅ 视频处理完成！

📊 处理结果:
- 提取帧数: {len(frames)}
- 时序组数: {len(temporal_ids)}
- 处理耗时: {process_time:.2f}秒

🎯 3D重采样器统计:
- 采样帧率: {fps} FPS
- 打包模式: {"强制 " + str(force_packing) if force_packing and force_packing > 0 else "自动"}

💬 现在可以开始与视频聊天了！
            """
            
            return status_text.strip()
            
        except Exception as e:
            return f"❌ 视频处理失败: {str(e)}"
    
    def chat_with_video(self, 
                       question: str, 
                       max_tokens: int, 
                       temperature: float, 
                       top_p: float) -> str:
        """
        与视频进行聊天
        
        Args:
            question: 用户问题
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: Top-p参数
            
        Returns:
            模型回答
        """
        if not self.service:
            return "❌ 服务未初始化，请先点击'初始化服务'按钮"
        
        if not self.current_video_data:
            return "❌ 请先上传并处理视频"
        
        if not question.strip():
            return "❌ 请输入您的问题"
        
        try:
            frames, temporal_ids = self.current_video_data
            
            start_time = time.time()
            
            # 使用缓存的视频数据进行聊天
            answer = self.service.chat_with_frames(
                frames=frames,
                temporal_ids=temporal_ids,
                question=question,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            inference_time = time.time() - start_time
            
            result_text = f"""
🤖 AI回答:
{answer}

⏱️ 推理耗时: {inference_time:.2f}秒
            """
            
            return result_text.strip()
            
        except Exception as e:
            return f"❌ 聊天失败: {str(e)}"
    
    def get_video_info(self, video_file) -> str:
        """
        获取视频信息
        
        Args:
            video_file: 视频文件
            
        Returns:
            视频信息文本
        """
        if not self.service:
            return "❌ 服务未初始化"
        
        if not video_file:
            return "❌ 请先上传视频"
        
        try:
            info = self.service.video_encoder.get_video_info(video_file)
            
            info_text = f"""
📹 视频信息:
- 帧率: {info.get('fps', 'N/A'):.2f} FPS
- 时长: {info.get('duration', 'N/A'):.2f} 秒
- 总帧数: {info.get('total_frames', 'N/A')}
- 分辨率: {info.get('width', 'N/A')} x {info.get('height', 'N/A')}
            """
            
            return info_text.strip()
            
        except Exception as e:
            return f"❌ 获取视频信息失败: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """
        创建Gradio界面
        
        Returns:
            Gradio Blocks界面
        """
        with gr.Blocks(
            title="MiniCPM-V 视频聊天 - Intel XPU版",
            theme=gr.themes.Soft(),
            css=""".gradio-container {max-width: 1200px; margin: auto;}"""
        ) as interface:
            
            gr.Markdown("""
            # 🎥 MiniCPM-V 视频聊天 - Intel XPU版
            
            基于MiniCPM-V-4.5多模态大语言模型的视频聊天系统  
            ✨ 特色功能: 3D重采样器 | Intel GPU加速 | 实时推理
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 🔧 系统配置")
                    
                    # 3D重采样器参数
                    with gr.Group():
                        gr.Markdown("### 📊 3D重采样器参数")
                        max_frames = gr.Slider(
                            minimum=60, maximum=300, value=180, step=20,
                            label="最大帧数", info="打包后接收的最大帧数"
                        )
                        max_packing = gr.Slider(
                            minimum=1, maximum=6, value=3, step=1,
                            label="最大打包数", info="视频帧3D压缩的最大打包数量"
                        )
                        time_scale = gr.Slider(
                            minimum=0.05, maximum=0.5, value=0.1, step=0.05,
                            label="时间缩放", info="时序ID计算的时间缩放因子"
                        )
                    
                    # 初始化按钮
                    init_btn = gr.Button("🚀 初始化服务", variant="primary")
                    init_status = gr.Textbox(
                        label="初始化状态", 
                        placeholder="点击上方按钮初始化服务...",
                        max_lines=15
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("## 📹 视频处理")
                    
                    # 视频上传
                    video_upload = gr.File(
                        label="上传视频文件",
                        file_types=["video"],
                        type="filepath"
                    )
                    
                    # 视频信息
                    video_info_btn = gr.Button("📊 获取视频信息")
                    video_info = gr.Textbox(
                        label="视频信息",
                        placeholder="上传视频后点击获取信息...",
                        max_lines=6
                    )
                    
                    # 视频处理参数
                    with gr.Row():
                        fps = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="采样帧率 (FPS)", info="从视频中提取帧的频率"
                        )
                        force_packing = gr.Slider(
                            minimum=0, maximum=6, value=0, step=1,
                            label="强制打包数", info="0表示自动，1-6强制指定"
                        )
                    
                    # 处理按钮
                    process_btn = gr.Button("🔄 处理视频", variant="secondary")
                    process_status = gr.Textbox(
                        label="处理状态",
                        placeholder="上传视频后点击处理...",
                        max_lines=10
                    )
            
            gr.Markdown("## 💬 视频聊天")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # 聊天输入
                    question = gr.Textbox(
                        label="您的问题",
                        placeholder="请输入您想了解视频内容的问题...",
                        lines=3
                    )
                    
                    # 生成参数
                    with gr.Row():
                        max_tokens = gr.Slider(
                            minimum=256, maximum=4096, value=2048, step=256,
                            label="最大生成长度", info="生成回答的最大token数"
                        )
                        temperature = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                            label="创造性 (Temperature)", info="控制回答的创造性"
                        )
                        top_p = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.8, step=0.1,
                            label="多样性 (Top-p)", info="控制回答的多样性"
                        )
                    
                    # 聊天按钮
                    chat_btn = gr.Button("🗣️ 开始聊天", variant="primary")
                
                with gr.Column(scale=3):
                    # 聊天结果
                    chat_result = gr.Textbox(
                        label="AI回答",
                        placeholder="处理视频并提问后，AI回答将显示在这里...",
                        lines=15
                    )
            
            # 示例问题
            gr.Markdown("### 💡 示例问题")
            example_questions = [
                "描述这个视频的主要内容",
                "视频中有多少个人？",
                "视频的背景音乐是什么风格？",
                "分析视频中人物的情绪",
                "视频传达了什么信息？"
            ]
            
            for question_text in example_questions:
                example_btn = gr.Button(f"📝 {question_text}", size="sm")
                example_btn.click(
                    lambda q=question_text: q,
                    outputs=question
                )
            
            # 绑定事件
            init_btn.click(
                self.initialize_service,
                inputs=[max_frames, max_packing, time_scale],
                outputs=init_status
            )
            
            video_info_btn.click(
                self.get_video_info,
                inputs=video_upload,
                outputs=video_info
            )
            
            process_btn.click(
                self.process_video_upload,
                inputs=[video_upload, fps, force_packing],
                outputs=process_status
            )
            
            chat_btn.click(
                self.chat_with_video,
                inputs=[question, max_tokens, temperature, top_p],
                outputs=chat_result
            )
            
            # 回车键提交
            question.submit(
                self.chat_with_video,
                inputs=[question, max_tokens, temperature, top_p],
                outputs=chat_result
            )
        
        return interface
    
    def launch(self, 
              server_name: str = "0.0.0.0", 
              server_port: int = 7860, 
              share: bool = False):
        """
        启动Gradio应用
        
        Args:
            server_name: 服务器地址
            server_port: 端口号
            share: 是否创建公共链接
        """
        interface = self.create_interface()
        
        print(f"🚀 启动Gradio应用...")
        print(f"   地址: http://{server_name}:{server_port}")
        
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            show_error=True
        )


def main():
    """主函数"""
    try:
        # 创建Gradio应用
        app = VideoChatGradioApp()
        
        # 启动应用
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
        
    except Exception as e:
        print(f"应用启动失败: {str(e)}")


if __name__ == "__main__":
    main()