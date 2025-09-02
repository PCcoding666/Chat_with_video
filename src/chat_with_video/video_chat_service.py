"""
视频聊天服务 - 统一服务接口
整合模型加载、3D重采样器和聊天功能
专为Intel XPU优化的MiniCPM-V视频聊天服务
"""

import os
import time
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image

from .model_loader import MiniCPMVInference
from .video_encoder import VideoEncoder


class VideoChatService:
    """视频聊天服务 - 统一接口"""
    
    def __init__(self, 
                 model_path: str = 'openbmb/MiniCPM-V-4_5-int4',
                 device: str = 'xpu',
                 max_frames: int = 180,
                 max_packing: int = 3,
                 time_scale: float = 0.1):
        """
        初始化视频聊天服务
        
        Args:
            model_path: 模型路径，默认MiniCPM-V-4.5-int4量化版本
            device: 设备类型，默认为xpu（Intel GPU）
            max_frames: 最大帧数限制
            max_packing: 最大打包数量（1-6）
            time_scale: 时间缩放因子
        """
        self.model_path = model_path
        self.device = device
        
        # 初始化组件
        self.inference_engine: Optional[MiniCPMVInference] = None
        self.video_encoder = VideoEncoder(
            max_frames=max_frames,
            max_packing=max_packing,
            time_scale=time_scale
        )
        
        self._initialized = False
        
        print(f"视频聊天服务已配置:")
        print(f"  - 模型: {model_path}")
        print(f"  - 设备: {device}")
        print(f"  - 3D重采样器参数: max_frames={max_frames}, max_packing={max_packing}")
    
    def initialize(self) -> bool:
        """
        初始化服务（延迟加载）
        
        Returns:
            bool: 初始化是否成功
        """
        if self._initialized:
            return True
        
        try:
            print("正在初始化视频聊天服务...")
            
            # 初始化推理引擎
            self.inference_engine = MiniCPMVInference(
                model_path=self.model_path,
                device=self.device
            )
            
            # 初始化模型
            self.inference_engine.initialize()
            
            # 可选：预热模型
            print("正在预热模型...")
            self.inference_engine.warm_up()
            
            self._initialized = True
            print("视频聊天服务初始化完成!")
            return True
            
        except Exception as e:
            print(f"服务初始化失败: {str(e)}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统状态信息
        
        Returns:
            系统信息字典
        """
        info = {
            'initialized': self._initialized,
            'model_path': self.model_path,
            'device': self.device,
            'video_encoder_config': {
                'max_frames': self.video_encoder.MAX_NUM_FRAMES,
                'max_packing': self.video_encoder.MAX_NUM_PACKING,
                'time_scale': self.video_encoder.TIME_SCALE
            }
        }
        
        if self.inference_engine:
            device_info = self.inference_engine.get_device_info()
            info.update(device_info)
        
        return info
    
    def process_video(self, 
                     video_path: str, 
                     choose_fps: int = 3,
                     force_packing: Optional[int] = None) -> Tuple[List[Image.Image], List[List[int]]]:
        """
        处理视频文件，提取帧和时序ID
        
        Args:
            video_path: 视频文件路径
            choose_fps: 采样帧率
            force_packing: 强制打包数量
            
        Returns:
            Tuple[frames, temporal_ids]: PIL图像帧列表和时序ID分组
        """
        try:
            print(f"开始处理视频: {video_path}")
            
            # 检查文件是否存在
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
            # 获取视频信息
            video_info = self.video_encoder.get_video_info(video_path)
            print(f"视频信息: {video_info}")
            
            # 使用3D重采样器编码视频
            frames, temporal_ids = self.video_encoder.encode_video(
                video_path=video_path,
                choose_fps=choose_fps,
                force_packing=force_packing
            )
            
            print(f"视频处理完成: {len(frames)}帧, {len(temporal_ids)}个时序组")
            return frames, temporal_ids
            
        except Exception as e:
            print(f"视频处理失败: {str(e)}")
            raise
    
    def chat_with_video(self,
                       video_path: str,
                       question: str,
                       choose_fps: int = 5,
                       force_packing: Optional[int] = None,
                       max_new_tokens: int = 2048,
                       temperature: float = 0.7,
                       top_p: float = 0.8) -> str:
        """
        与视频进行聊天对话
        
        Args:
            video_path: 视频文件路径
            question: 用户问题
            choose_fps: 视频采样帧率
            force_packing: 强制打包数量
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: Top-p采样参数
            
        Returns:
            模型回答
        """
        # 确保服务已初始化
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("服务初始化失败")
        
        try:
            start_time = time.time()
            
            # 处理视频
            print(f"处理视频: {video_path}")
            frames, temporal_ids = self.process_video(
                video_path=video_path,
                choose_fps=choose_fps,
                force_packing=force_packing
            )
            
            process_time = time.time() - start_time
            print(f"视频处理耗时: {process_time:.2f}秒")
            
            # 构建消息
            msgs = [
                {'role': 'user', 'content': frames + [question]}
            ]
            
            print(f"开始推理，问题: {question}")
            inference_start = time.time()
            
            # 调用模型推理
            answer = self.inference_engine.chat(
                msgs=msgs,
                use_image_id=False,
                max_slice_nums=1,
                temporal_ids=temporal_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            inference_time = time.time() - inference_start
            total_time = time.time() - start_time
            
            print(f"推理耗时: {inference_time:.2f}秒")
            print(f"总耗时: {total_time:.2f}秒")
            print(f"回答: {answer}")
            
            return answer
            
        except Exception as e:
            print(f"视频聊天失败: {str(e)}")
            raise
    
    def chat_with_frames(self,
                        frames: List[Image.Image],
                        temporal_ids: List[List[int]],
                        question: str,
                        max_new_tokens: int = 2048,
                        temperature: float = 0.7,
                        top_p: float = 0.8) -> str:
        """
        使用已处理的帧和时序ID进行聊天
        
        Args:
            frames: PIL图像帧列表
            temporal_ids: 时序ID分组列表
            question: 用户问题
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: Top-p采样参数
            
        Returns:
            模型回答
        """
        # 确保服务已初始化
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("服务初始化失败")
        
        try:
            # 构建消息
            msgs = [
                {'role': 'user', 'content': frames + [question]}
            ]
            
            print(f"使用预处理帧进行推理，问题: {question}")
            
            # 调用模型推理
            answer = self.inference_engine.chat(
                msgs=msgs,
                use_image_id=False,
                max_slice_nums=1,
                temporal_ids=temporal_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            return answer
            
        except Exception as e:
            print(f"聊天失败: {str(e)}")
            raise
    
    def clear_cache(self):
        """清理缓存"""
        if self.inference_engine:
            self.inference_engine.clear_cache()
    
    def shutdown(self):
        """关闭服务"""
        try:
            self.clear_cache()
            print("视频聊天服务已关闭")
        except Exception as e:
            print(f"关闭服务时出错: {str(e)}")


if __name__ == "__main__":
    # 测试代码
    try:
        print("测试视频聊天服务...")
        
        # 创建服务实例
        service = VideoChatService()
        
        # 显示系统信息
        info = service.get_system_info()
        print("系统信息:", info)
        
        # 注意：需要提供真实的视频文件路径进行测试
        test_video = "test_video.mp4"
        test_question = "描述这个视频的内容"
        
        if os.path.exists(test_video):
            # 测试视频聊天
            answer = service.chat_with_video(
                video_path=test_video,
                question=test_question,
                choose_fps=5
            )
            print(f"测试结果: {answer}")
        else:
            print(f"测试视频 {test_video} 不存在，跳过聊天测试")
        
        print("服务测试完成!")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")