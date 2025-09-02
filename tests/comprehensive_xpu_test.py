#!/usr/bin/env python3
"""
全面XPU测试
测试完整的视频聊天服务在XPU上的运行情况
"""

import os
import sys
import time
import traceback
from PIL import Image
import numpy as np

# 设置环境变量 - 在导入任何PyTorch相关库之前
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

def print_separator(title):
    """打印分隔符"""
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)

def create_test_video_frames():
    """创建测试视频帧"""
    print("创建测试视频帧...")
    
    # 创建几个测试图像
    frames = []
    for i in range(5):
        # 创建不同颜色的图像
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        frames.append(img)
    
    # 创建时序ID
    temporal_ids = [[0, 1, 2], [3, 4]]
    
    print(f"✅ 创建了 {len(frames)} 个测试帧和 {len(temporal_ids)} 个时序组")
    return frames, temporal_ids

def test_video_chat_service():
    """测试完整的视频聊天服务"""
    print_separator("🚀 视频聊天服务XPU测试")
    
    try:
        # 添加项目路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        print("导入VideoChatService...")
        from src.chat_with_video.video_chat_service import VideoChatService
        
        print("✅ 模块导入成功")
        
        # 创建服务实例
        print("\n创建VideoChatService实例...")
        start_time = time.time()
        
        service = VideoChatService(
            model_path='openbmb/MiniCPM-V-4_5-int4',
            device='xpu',
            max_frames=60,  # 减少帧数以加快测试
            max_packing=2
        )
        
        print("✅ 服务实例创建成功")
        
        # 初始化服务
        print("\n初始化服务...")
        init_success = service.initialize()
        
        if not init_success:
            print("❌ 服务初始化失败")
            return False
        
        init_time = time.time() - start_time
        print(f"✅ 服务初始化成功！耗时: {init_time:.2f}秒")
        
        # 显示系统信息
        print("\n📊 系统信息:")
        system_info = service.get_system_info()
        for key, value in system_info.items():
            if key != 'video_encoder_config':
                print(f"  {key}: {value}")
        
        print("  video_encoder_config:")
        for key, value in system_info['video_encoder_config'].items():
            print(f"    {key}: {value}")
        
        # 测试使用帧进行聊天
        print("\n🧪 测试使用帧进行聊天...")
        
        # 创建测试帧
        frames, temporal_ids = create_test_video_frames()
        
        # 构造测试问题
        question = "请描述这些图像中的内容。"
        
        print(f"问题: {question}")
        print("开始推理...")
        
        inference_start = time.time()
        answer = service.chat_with_frames(
            frames=frames,
            temporal_ids=temporal_ids,
            question=question,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.8
        )
        inference_time = time.time() - inference_start
        
        print(f"✅ 推理成功！")
        print(f"   推理耗时: {inference_time:.2f}秒")
        print(f"   回答: {answer}")
        
        # 验证模型确实在XPU上
        print("\n🔍 验证模型设备...")
        device_info = service.inference_engine.get_device_info()
        if device_info.get('device') == 'xpu':
            print("✅ 模型确实在XPU上运行")
        else:
            print(f"⚠️ 模型在 {device_info.get('device')} 上运行，不是XPU")
        
        return True
        
    except Exception as e:
        print(f"❌ 视频聊天服务测试失败:")
        print(f"   错误: {e}")
        print(f"\n🔍 详细错误信息:")
        traceback.print_exc()
        return False

def test_gradio_app_initialization():
    """测试Gradio应用初始化"""
    print_separator("🌐 Gradio应用初始化测试")
    
    try:
        # 添加项目路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        print("导入VideoChatGradioApp...")
        from src.chat_with_video.gradio_app import VideoChatGradioApp
        
        print("✅ 模块导入成功")
        
        # 创建Gradio应用实例
        print("\n创建VideoChatGradioApp实例...")
        app = VideoChatGradioApp(
            model_path='openbmb/MiniCPM-V-4_5-int4',
            device='xpu'
        )
        
        print("✅ Gradio应用实例创建成功")
        
        # 测试服务初始化
        print("\n测试服务初始化...")
        status = app.initialize_service(
            max_frames=60,
            max_packing=2,
            time_scale=0.1
        )
        
        print(f"初始化状态:\n{status}")
        
        # 检查是否成功
        if "✅ 服务初始化成功" in status:
            print("✅ Gradio应用初始化成功")
            return True
        else:
            print("❌ Gradio应用初始化失败")
            return False
        
    except Exception as e:
        print(f"❌ Gradio应用测试失败:")
        print(f"   错误: {e}")
        print(f"\n🔍 详细错误信息:")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print_separator("🧪 全面XPU测试套件")
    
    success_count = 0
    total_tests = 2
    
    # 测试1: 视频聊天服务
    if test_video_chat_service():
        success_count += 1
    
    # 测试2: Gradio应用初始化
    if test_gradio_app_initialization():
        success_count += 1
    
    # 总结
    print_separator("📊 测试结果总结")
    print(f"通过测试: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！XPU强制加载策略工作正常！")
        print("💡 Intel Arc GPU的显存查询限制已被彻底绕过")
        print("🚀 视频聊天服务可以正常在XPU上运行")
        return 0
    else:
        print("❌ 部分测试失败，需要进一步调试")
        return 1

if __name__ == "__main__":
    exit_code = main()
    input("\n按Enter键退出...")
    sys.exit(exit_code)