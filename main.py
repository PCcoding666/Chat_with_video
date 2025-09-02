#!/usr/bin/env python3
"""
MiniCPM-V 视频聊天 Demo - Intel GPU版本

这是一个完整的视频聊天演示程序，使用MiniCPM-V-4.5模型在Intel Arc GPU上进行推理。
支持3D重采样器、视频理解、多轮对话和批量处理功能。

使用方法:
  python main.py                    # 交互式模式
  python main.py --video path.mp4   # 指定视频文件
  python main.py --batch            # 批量处理模式
  python main.py --web              # Web界面模式 (Gradio)
  python main.py --test             # 运行测试
"""

# 修复Triton DLL加载问题 - 必须在任何其他导入之前设置
import os
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# 注释掉有问题的导入，改为延迟导入
# from src.chat_with_video.video_chat_interface import VideoChatInterface
from src.chat_with_video.gradio_app import VideoChatGradioApp


def print_banner():
    """打印程序横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                   MiniCPM-V 视频聊天 Demo                   ║
║                     Intel Arc GPU 版本                      ║
╠══════════════════════════════════════════════════════════════╣
║  🎥 支持多种视频格式 (MP4, AVI, MOV, MKV 等)              ║
║  🚀 Intel Arc GPU 硬件加速                                 ║
║  🤖 MiniCPM-V-4.5 多模态大语言模型                         ║
║  🔆 3D重采样器支持，压缩多帧为64个token               ║
║  💻 Web界面支持，Gradio驱动的直观交互界面               ║
║  💬 支持中英文视频内容理解                                  ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_interactive_mode(video_path: Optional[str] = None):
    """运行交互式模式"""
    try:
        print("\n🚀 正在初始化视频聊天系统...")
        
        # 延迟导入以避免早期加载问题
        from src.chat_with_video.video_chat_interface import VideoChatInterface
        chat_interface = VideoChatInterface()
        
        if video_path:
            # 单个视频模式
            print(f"\n📁 指定视频文件: {video_path}")
            
            if not chat_interface.validate_video_file(video_path):
                print("❌ 视频文件无效")
                return
            
            # 获取视频预览
            chat_interface.get_video_preview(video_path)
            
            # 持续对话
            while True:
                question = input("\n❓ 请输入您的问题 (或 'quit' 退出): ").strip()
                
                if question.lower() == 'quit':
                    break
                
                if not question:
                    continue
                
                answer = chat_interface.chat_with_video(video_path, question)
                
                print("\n🤖 回答:")
                print("-" * 40)
                print(answer)
                print("-" * 40)
        else:
            # 完全交互式模式
            chat_interface.interactive_chat()
            
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        return 1
    
    return 0


def run_web_mode(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    """运行Web界面模式"""
    try:
        print("\n🌐 正在启动Web界面模式...")
        print(f"   地址: http://{host}:{port}")
        
        # 创建Gradio应用
        app = VideoChatGradioApp()
        
        # 启动应用
        app.launch(
            server_name=host,
            server_port=port,
            share=share
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
        return 0
    except Exception as e:
        print(f"\n❌ Web界面启动失败: {str(e)}")
        return 1
    
    return 0


def run_batch_mode():
    """运行批量处理模式"""
    try:
        print("\n🚀 正在初始化批量处理系统...")
        
        # 延迟导入
        from src.chat_with_video.video_chat_interface import VideoChatInterface
        chat_interface = VideoChatInterface()
        
        # 获取视频文件
        video_path = input("\n📁 请输入视频文件路径: ").strip()
        
        if not chat_interface.validate_video_file(video_path):
            print("❌ 视频文件无效")
            return 1
        
        # 获取问题列表
        print("\n📝 请输入问题列表 (每行一个问题，空行结束):")
        questions = []
        
        while True:
            question = input(f"问题 {len(questions) + 1}: ").strip()
            if not question:
                break
            questions.append(question)
        
        if not questions:
            print("❌ 没有输入任何问题")
            return 1
        
        print(f"\n📊 开始批量处理 {len(questions)} 个问题...")
        
        # 批量处理
        results = chat_interface.batch_chat_with_video(video_path, questions)
        
        # 显示结果
        print("\n" + "="*60)
        print("📋 批量处理结果")
        print("="*60)
        
        for i, (question, answer) in enumerate(results, 1):
            print(f"\n问题 {i}: {question}")
            print(f"回答: {answer}")
            print("-" * 40)
        
        # 保存结果到文件
        output_file = f"batch_results_{Path(video_path).stem}.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"视频文件: {video_path}\n")
                f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                
                for i, (question, answer) in enumerate(results, 1):
                    f.write(f"问题 {i}: {question}\n")
                    f.write(f"回答: {answer}\n")
                    f.write("-" * 40 + "\n\n")
            
            print(f"\n💾 结果已保存到: {output_file}")
            
        except Exception as e:
            print(f"\n⚠️ 保存结果失败: {str(e)}")
        
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
        return 1
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        return 1
    
    return 0


def run_test_mode():
    """运行测试模式"""
    try:
        print("\n🧪 正在运行系统测试...")
        
        # 1. 测试基础组件
        print("\n1️⃣ 测试基础组件...")
        
        # 测试视频编码器
        try:
            from src.chat_with_video.video_encoder import VideoEncoder
            encoder = VideoEncoder()
            print("  ✓ 视频编码器初始化成功")
        except Exception as e:
            print(f"  ❌ 视频编码器初始化失败: {str(e)}")
            return 1
        
        # 测试模型加载器
        try:
            # 延迟导入以避免问题
            from src.chat_with_video.model_loader import MiniCPMVInference
            
            # 检查XPU可用性
            import torch
            if not torch.xpu.is_available():
                print("  ❌ Intel XPU 不可用")
                return 1
            
            print("  ✓ Intel XPU 可用")
            print(f"  ✓ XPU设备数量: {torch.xpu.device_count()}")
            
        except Exception as e:
            print(f"  ❌ XPU检查失败: {str(e)}")
            return 1
        
        # 2. 测试依赖库
        print("\n2️⃣ 测试依赖库...")
        
        required_packages = [
            'torch', 'torchvision', 'transformers', 
            'decord', 'scipy', 'numpy', 'PIL'
        ]
        
        for package in required_packages:
            try:
                if package == 'PIL':
                    import PIL
                else:
                    __import__(package)
                print(f"  ✓ {package} 导入成功")
            except ImportError as e:
                print(f"  ❌ {package} 导入失败: {str(e)}")
                return 1
        
        # 3. 测试模型加载（如果用户同意）
        print("\n3️⃣ 测试模型加载...")
        
        load_model = input("是否测试模型加载？这将下载大约8GB的模型文件 (y/N): ").strip().lower()
        
        if load_model == 'y':
            try:
                print("  正在加载模型（这可能需要几分钟）...")
                inference_engine = MiniCPMVInference()
                print("  ✓ 模型加载成功")
                
                # 显示设备信息
                device_info = inference_engine.get_device_info()
                print(f"  ✓ 当前设备: {device_info['device']}")
                
                if 'memory_allocated_gb' in device_info:
                    print(f"  ✓ 显存使用: {device_info['memory_allocated_gb']:.2f} GB")
                
                # 测试简单推理
                try:
                    print("  正在测试推理...")
                    test_msgs = [{'role': 'user', 'content': ['测试消息']}]
                    response = inference_engine.chat(test_msgs, max_new_tokens=10)
                    print("  ✓ 推理测试成功")
                except Exception as e:
                    print(f"  ⚠️ 推理测试失败: {str(e)}")
                
            except Exception as e:
                print(f"  ❌ 模型加载失败: {str(e)}")
                return 1
        else:
            print("  ⏭️ 跳过模型加载测试")
        
        print("\n" + "="*50)
        print("🎉 所有测试通过！系统准备就绪")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n\n👋 用户中断测试")
        return 1
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        return 1
    
    return 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MiniCPM-V 视频聊天 Demo - Intel GPU版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py                          # 交互式模式
  python main.py --video demo.mp4         # 指定视频文件
  python main.py --batch                  # 批量处理模式
  python main.py --web                    # Web界面模式 (Gradio)
  python main.py --web --port 8080        # 指定端口的Web模式
  python main.py --test                   # 运行系统测试

支持的视频格式: MP4, AVI, MOV, MKV, FLV, WMV, WEBM
        """
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        help='指定要处理的视频文件路径'
    )
    
    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='启动批量处理模式'
    )
    
    parser.add_argument(
        '--web', '-w',
        action='store_true',
        help='启动Web界面模式 (Gradio)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Web服务器地址 (默认: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=7860,
        help='Web服务器端口 (默认: 7860)'
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='创建公共分享链接 (仅Web模式)'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='运行系统测试'
    )
    
    parser.add_argument(
        '--no-banner',
        action='store_true',
        help='不显示程序横幅'
    )
    
    args = parser.parse_args()
    
    # 显示横幅
    if not args.no_banner:
        print_banner()
    
    # 根据参数选择运行模式
    if args.test:
        return run_test_mode()
    elif args.batch:
        return run_batch_mode()
    elif args.web:
        return run_web_mode(
            host=args.host,
            port=args.port,
            share=args.share
        )
    else:
        return run_interactive_mode(args.video)


if __name__ == "__main__":
    import time
    
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 程序异常退出: {str(e)}")
        sys.exit(1)
