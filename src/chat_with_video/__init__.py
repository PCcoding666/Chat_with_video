"""
Chat with Video - MiniCPM-V 视频聊天项目
Intel GPU版本
"""

from .video_encoder import VideoEncoder
from .model_loader import MiniCPMVInference
from .video_chat_interface import VideoChatInterface

__version__ = "0.1.0"
__author__ = "PCcoding666"
__email__ = "e1143754@u.nus.edu"

__all__ = [
    "VideoEncoder",
    "MiniCPMVInference", 
    "VideoChatInterface",
]