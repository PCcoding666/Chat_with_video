"""
视频聊天交互接口
整合视频编码器和MiniCPM-V模型，提供完整的视频对话功能
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .video_encoder import VideoEncoder
from .model_loader import MiniCPMVInference


class VideoChatInterface:
    """视频聊天交互接口"""
    
    def __init__(self, model_path: str = 'openbmb/MiniCPM-V-4_5', device: str = 'xpu'):
        """
        初始化视频聊天接口
        
        Args:
            model_path: MiniCPM-V模型路径
            device: 设备类型，默认为'xpu'（Intel GPU）
        """
        print("初始化视频聊天接口...")
        
        # 初始化视频编码器
        self.video_encoder = VideoEncoder()
        print("✓ 视频编码器初始化完成")
        
        # 初始化模型推理引擎
        self.inference_engine = MiniCPMVInference(model_path, device)
        print("✓ 模型推理引擎初始化完成")
        
        # 预热模型
        self.inference_engine.warm_up()
        print("✓ 模型预热完成")
        
        print("视频聊天接口初始化完成!")
    
    def validate_video_file(self, video_path: str) -> bool:
        """
        验证视频文件是否有效
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            bool: 文件是否有效
        """
        if not os.path.exists(video_path):
            print(f"错误: 视频文件不存在: {video_path}")
            return False
        
        # 检查文件扩展名
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
        file_ext = Path(video_path).suffix.lower()
        
        if file_ext not in valid_extensions:
            print(f"警告: 不常见的视频格式: {file_ext}")
        
        # 获取文件大小
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        print(f"视频文件大小: {file_size:.2f} MB")
        
        if file_size > 1000:  # 1GB
            print("警告: 视频文件较大，处理可能需要更长时间")
        
        return True
    
    def get_video_preview(self, video_path: str) -> Dict[str, Any]:
        """
        获取视频预览信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含视频信息的字典
        """
        try:
            video_info = self.video_encoder.get_video_info(video_path)
            
            if video_info:
                preview = {
                    'duration': f"{video_info['duration']:.2f}秒",
                    'fps': f"{video_info['fps']:.2f}",
                    'total_frames': video_info['total_frames'],
                    'resolution': f"{video_info['width']}x{video_info['height']}",
                    'estimated_sample_frames': min(180, int(video_info['duration'] * 3))  # 估算采样帧数
                }
                
                print("视频预览信息:")
                for key, value in preview.items():
                    print(f"  {key}: {value}")
                
                return preview
            else:
                return {}
                
        except Exception as e:
            print(f"获取视频预览失败: {str(e)}")
            return {}
    
    def chat_with_video(self, video_path: str, question: str, 
                       fps: int = 5, force_packing: Optional[int] = None,
                       max_new_tokens: int = 2048, temperature: float = 0.7) -> str:
        """
        视频对话主接口
        
        Args:
            video_path: 视频文件路径
            question: 用户问题
            fps: 采样帧率，默认5帧/秒
            force_packing: 强制3D打包数量
            max_new_tokens: 最大生成token数
            temperature: 生成温度
            
        Returns:
            模型的回答
        """
        try:
            print(f"\n{'='*50}")
            print(f"开始处理视频对话")
            print(f"视频: {video_path}")
            print(f"问题: {question}")
            print(f"{'='*50}")
            
            # 1. 验证视频文件
            if not self.validate_video_file(video_path):
                return "错误: 无法访问视频文件"
            
            # 2. 获取视频预览信息
            preview = self.get_video_preview(video_path)
            
            # 3. 视频编码
            print("\n步骤1: 正在处理视频...")
            start_time = time.time()
            
            frames, temporal_ids = self.video_encoder.encode_video(
                video_path, 
                choose_fps=fps, 
                force_packing=force_packing
            )
            
            encoding_time = time.time() - start_time
            print(f"视频处理完成，耗时: {encoding_time:.2f}秒")
            print(f"采样帧数: {len(frames)}")
            print(f"时序组数: {len(temporal_ids)}")
            
            # 4. 构建消息
            print("\n步骤2: 构建对话消息...")
            msgs = [
                {'role': 'user', 'content': frames + [question]}
            ]
            
            # 5. 模型推理
            print("\n步骤3: 正在生成回答...")
            start_time = time.time()
            
            answer = self.inference_engine.chat(
                msgs=msgs,
                use_image_id=False,
                max_slice_nums=1,
                temporal_ids=temporal_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            inference_time = time.time() - start_time
            print(f"推理完成，耗时: {inference_time:.2f}秒")
            
            # 6. 清理缓存
            self.inference_engine.clear_cache()
            
            print(f"\n{'='*50}")
            print("对话完成!")
            print(f"总处理时间: {encoding_time + inference_time:.2f}秒")
            print(f"{'='*50}\n")
            
            return answer
            
        except Exception as e:
            error_msg = f"视频对话处理失败: {str(e)}"
            print(f"错误: {error_msg}")
            return error_msg
    
    def batch_chat_with_video(self, video_path: str, questions: List[str],
                             fps: int = 5, force_packing: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        批量视频对话，复用视频编码结果
        
        Args:
            video_path: 视频文件路径
            questions: 问题列表
            fps: 采样帧率
            force_packing: 强制3D打包数量
            
        Returns:
            (问题, 答案) 元组列表
        """
        try:
            print(f"\n开始批量处理 {len(questions)} 个问题...")
            
            # 验证视频文件
            if not self.validate_video_file(video_path):
                return [("错误", "无法访问视频文件")] * len(questions)
            
            # 一次性编码视频
            print("正在编码视频...")
            frames, temporal_ids = self.video_encoder.encode_video(
                video_path, choose_fps=fps, force_packing=force_packing
            )
            
            results = []
            
            for i, question in enumerate(questions, 1):
                print(f"\n处理问题 {i}/{len(questions)}: {question}")
                
                try:
                    # 构建消息
                    msgs = [{'role': 'user', 'content': frames + [question]}]
                    
                    # 推理
                    answer = self.inference_engine.chat(
                        msgs=msgs,
                        use_image_id=False,
                        max_slice_nums=1,
                        temporal_ids=temporal_ids
                    )
                    
                    results.append((question, answer))
                    print(f"✓ 问题 {i} 处理完成")
                    
                except Exception as e:
                    error_msg = f"处理失败: {str(e)}"
                    results.append((question, error_msg))
                    print(f"✗ 问题 {i} 处理失败: {error_msg}")
                
                # 清理缓存
                self.inference_engine.clear_cache()
            
            print(f"\n批量处理完成! 成功: {len([r for r in results if not r[1].startswith('处理失败')])}/{len(questions)}")
            return results
            
        except Exception as e:
            error_msg = f"批量处理失败: {str(e)}"
            print(f"错误: {error_msg}")
            return [(q, error_msg) for q in questions]
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统和设备信息"""
        device_info = self.inference_engine.get_device_info()
        
        system_info = {
            'video_encoder': {
                'max_frames': self.video_encoder.MAX_NUM_FRAMES,
                'max_packing': self.video_encoder.MAX_NUM_PACKING,
                'time_scale': self.video_encoder.TIME_SCALE
            },
            'model_info': {
                'model_path': self.inference_engine.model_path,
                'device': self.inference_engine.device
            },
            'device_info': device_info
        }
        
        return system_info
    
    def interactive_chat(self):
        """交互式聊天模式"""
        print("\n" + "="*60)
        print("🎥 MiniCPM-V 视频聊天 - Intel GPU版本")
        print("="*60)
        
        # 显示系统信息
        system_info = self.get_system_info()
        print(f"📱 设备: {system_info['device_info']['device']}")
        print(f"🔧 XPU可用: {system_info['device_info']['xpu_available']}")
        print(f"💾 设备数量: {system_info['device_info']['device_count']}")
        
        print("\n💡 使用说明:")
        print("  1. 输入视频文件路径")
        print("  2. 输入您的问题")
        print("  3. 等待模型处理和回答")
        print("  4. 输入 'quit' 退出程序")
        print("-"*60)
        
        while True:
            try:
                # 获取视频路径
                video_path = input("\n🎬 请输入视频文件路径 (或 'quit' 退出): ").strip()
                
                if video_path.lower() == 'quit':
                    print("👋 再见!")
                    break
                
                if not video_path:
                    continue
                
                # 验证视频文件
                if not self.validate_video_file(video_path):
                    continue
                
                # 获取视频预览
                self.get_video_preview(video_path)
                
                # 持续对话模式
                while True:
                    question = input("\n❓ 请输入您的问题 (或 'new' 选择新视频, 'quit' 退出): ").strip()
                    
                    if question.lower() == 'quit':
                        print("👋 再见!")
                        return
                    
                    if question.lower() == 'new':
                        break
                    
                    if not question:
                        continue
                    
                    # 处理对话
                    answer = self.chat_with_video(video_path, question)
                    
                    print("\n🤖 回答:")
                    print("-" * 40)
                    print(answer)
                    print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，退出程序")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {str(e)}")
                continue


if __name__ == "__main__":
    # 测试视频聊天接口
    try:
        print("测试视频聊天接口...")
        
        chat_interface = VideoChatInterface()
        
        # 显示系统信息
        system_info = chat_interface.get_system_info()
        print("系统信息:", system_info)
        
        # 启动交互式聊天
        chat_interface.interactive_chat()
        
    except Exception as e:
        print(f"测试失败: {str(e)}")