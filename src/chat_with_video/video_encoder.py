"""
视频编码模块 - 支持3D重采样器的完整实现
3D重采样器通过引入temporal_ids将多帧压缩为64个token
用于MiniCPM-V模型的视频处理
"""

import math
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional


class VideoEncoder:
    """视频帧采样和3D重采样器 - 完整实现"""
    
    def __init__(self, max_frames: int = 180, max_packing: int = 3, time_scale: float = 0.1):
        """
        初始化视频编码器
        
        Args:
            max_frames: 打包后接收的最大帧数，实际最大有效帧数为 MAX_NUM_FRAMES * MAX_NUM_PACKING
            max_packing: 最大打包数量，有效范围1-6，用于视频帧的3D压缩
            time_scale: 时间缩放因子，用于时序ID计算
        """
        self.MAX_NUM_FRAMES = max_frames
        self.MAX_NUM_PACKING = max_packing
        self.TIME_SCALE = time_scale
        
        print(f"3D重采样器已初始化:")
        print(f"  - 最大帧数: {max_frames}")
        print(f"  - 最大打包数: {max_packing}")
        print(f"  - 时间缩放: {time_scale}")
    
    def uniform_sample(self, frame_list: List, target_count: int) -> List:
        """
        均匀采样帧
        
        Args:
            frame_list: 原始帧列表
            target_count: 目标采样数量
            
        Returns:
            采样后的帧列表
        """
        gap = len(frame_list) / target_count
        indices = [int(i * gap + gap / 2) for i in range(target_count)]
        return [frame_list[i] for i in indices]
    
    def map_to_nearest_scale(self, values: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        将值映射到最近的缩放值
        
        Args:
            values: 输入值数组
            scale: 缩放数组
            
        Returns:
            映射后的值数组
        """
        tree = cKDTree(np.asarray(scale)[:, None])
        _, indices = tree.query(np.asarray(values)[:, None])
        return np.asarray(scale)[indices]
    
    def group_array(self, arr: List, size: int) -> List[List]:
        """
        将数组按指定大小分组
        
        Args:
            arr: 输入数组
            size: 分组大小
            
        Returns:
            分组后的数组列表
        """
        return [arr[i:i+size] for i in range(0, len(arr), size)]
    
    def encode_video(self, video_path: str, choose_fps: int = 3, 
                    force_packing: Optional[int] = None) -> Tuple[List[Image.Image], List[List[int]]]:
        """
        将视频编码为帧序列和temporal_ids，实现3D重采样器功能
        3D重采样器通过将多帧组织为两个对应序列：
        - frames: List[Image] - PIL图像帧列表
        - temporal_ids: List[List[Int]] - 时序ID分组列表
        
        Args:
            video_path: 视频文件路径
            choose_fps: 采样帧率，控制从视频中提取帧的频率
            force_packing: 强制打包数量（可选），可以强制启用3D打包
            
        Returns:
            Tuple[frames, temporal_ids]: 
                - frames: PIL图像帧列表
                - temporal_ids: 时序ID分组列表，用于3D重采样器
        """
        try:
            # 使用decord读取视频
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            video_duration = len(vr) / fps
            
            print(f"视频路径: {video_path}")
            print(f"视频时长: {video_duration:.2f}秒")
            print(f"原始FPS: {fps:.2f}")
            print(f"总帧数: {len(vr)}")
            
            # 根据视频时长和采样帧率动态计算打包参数
            if choose_fps * int(video_duration) <= self.MAX_NUM_FRAMES:
                # 短视频，不需要打包
                packing_nums = 1
                choose_frames = round(min(choose_fps, round(fps)) * min(self.MAX_NUM_FRAMES, video_duration))
            else:
                # 长视频，需要计算打包数量
                packing_nums = math.ceil(video_duration * choose_fps / self.MAX_NUM_FRAMES)
                if packing_nums <= self.MAX_NUM_PACKING:
                    choose_frames = round(video_duration * choose_fps)
                else:
                    choose_frames = round(self.MAX_NUM_FRAMES * self.MAX_NUM_PACKING)
                    packing_nums = self.MAX_NUM_PACKING
            
            # 如果强制指定打包数量
            if force_packing:
                packing_nums = min(force_packing, self.MAX_NUM_PACKING)
                print(f"强制打包数量: {packing_nums}")
            
            print(f"选择帧数: {choose_frames}")
            print(f"打包数量: {packing_nums}")
            
            # 均匀采样帧索引
            frame_idx = [i for i in range(0, len(vr))]
            frame_idx = np.array(self.uniform_sample(frame_idx, choose_frames))
            
            print(f"获取视频帧={len(frame_idx)}, 打包数={packing_nums}")
            
            # 获取视频帧数据
            frames = vr.get_batch(frame_idx).asnumpy()
            
            # 计算时序ID，这是3D重采样器的关键部分
            frame_idx_ts = frame_idx / fps  # 将帧索引转换为时间戳
            scale = np.arange(0, video_duration, self.TIME_SCALE)  # 创建时间刻度
            
            # 将帧时间戳映射到最近的刻度值
            frame_ts_id = self.map_to_nearest_scale(frame_idx_ts, scale) / self.TIME_SCALE
            frame_ts_id = frame_ts_id.astype(np.int32)
            
            # 验证数据一致性
            assert len(frames) == len(frame_ts_id), f"帧数({len(frames)})与时序ID数量({len(frame_ts_id)})不匹配"
            
            # 转换为PIL图像格式
            frames_pil = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
            
            # 将时序ID按打包数量分组，这是3D重采样器的核心功能
            frame_ts_id_group = self.group_array(frame_ts_id.tolist(), packing_nums)
            
            print(f"3D重采样器处理完成:")
            print(f"  - 总帧数: {len(frames_pil)}")
            print(f"  - 时序组数: {len(frame_ts_id_group)}")
            print(f"  - 每组帧数: {packing_nums}")
            
            # 显示第一个时序组的示例
            if frame_ts_id_group:
                print(f"  - 第一个时序组: {frame_ts_id_group[0]}")
            
            return frames_pil, frame_ts_id_group
            
        except Exception as e:
            print(f"视频编码错误: {str(e)}")
            raise
    
    def get_video_info(self, video_path: str) -> dict:
        """
        获取视频基本信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含视频信息的字典
        """
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            duration = len(vr) / fps
            
            return {
                'fps': fps,
                'duration': duration,
                'total_frames': len(vr),
                'width': vr[0].shape[1],
                'height': vr[0].shape[0]
            }
        except Exception as e:
            print(f"获取视频信息错误: {str(e)}")
            return {}


if __name__ == "__main__":
    # 测试代码
    encoder = VideoEncoder()
    
    # 这里需要一个测试视频文件
    test_video = "test_video.mp4"  # 替换为实际的视频文件路径
    
    try:
        # 获取视频信息
        info = encoder.get_video_info(test_video)
        print("视频信息:", info)
        
        # 编码视频
        frames, temporal_ids = encoder.encode_video(test_video, choose_fps=5)
        
        print(f"编码结果: {len(frames)} 帧, {len(temporal_ids)} 个时序组")
        print(f"第一个时序组: {temporal_ids[0] if temporal_ids else '无'}")
        
    except FileNotFoundError:
        print(f"测试视频文件 {test_video} 不存在，请提供有效的视频文件进行测试")
    except Exception as e:
        print(f"测试失败: {str(e)}")