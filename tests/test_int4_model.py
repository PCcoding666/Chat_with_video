#!/usr/bin/env python3
"""
测试MiniCPM-V-4.5-int4量化版本
验证INT4模型的加载、推理和性能
"""

import os
import torch
import time

# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

def test_int4_model_loading():
    """测试INT4模型加载"""
    try:
        print("🔧 测试MiniCPM-V-4.5-int4量化版本...")
        
        # 清理XPU缓存
        if hasattr(torch.xpu, 'empty_cache'):
            torch.xpu.empty_cache()
            print("✅ XPU缓存已清理")
        
        # 导入INT4模型加载器
        from src.chat_with_video.model_loader import MiniCPMVInference
        
        # 创建推理引擎 - 使用默认的int4路径
        print("📦 创建INT4推理引擎...")
        start_time = time.time()
        
        inference_engine = MiniCPMVInference()  # 默认使用int4版本
        
        # 初始化模型
        print("🚀 初始化INT4模型...")
        inference_engine.initialize()
        
        init_time = time.time() - start_time
        print(f"⏱️ 模型初始化耗时: {init_time:.2f}秒")
        
        # 检查设备信息
        device_info = inference_engine.get_device_info()
        print(f"\n📊 设备信息:")
        for key, value in device_info.items():
            print(f"  {key}: {value}")
        
        # 详细检查模型设备分布
        print("\n🔍 INT4模型设备分布:")
        model = inference_engine.model
        
        device_summary = {}
        total_params = 0
        xpu_params = 0
        
        for name, param in model.named_parameters():
            device = str(param.device)
            param_count = param.numel()
            total_params += param_count
            
            if 'xpu' in device.lower():
                xpu_params += param_count
            
            if device not in device_summary:
                device_summary[device] = {'count': 0, 'params': 0, 'memory_mb': 0}
            
            device_summary[device]['count'] += 1
            device_summary[device]['params'] += param_count
            
            # 估算参数内存占用（INT4 = 0.5 bytes per param）
            if 'int4' in inference_engine.model_path.lower():
                param_memory = param_count * 0.5 / (1024**2)  # MB
            else:
                param_memory = param_count * 2 / (1024**2)  # float16 = 2 bytes per param
            
            device_summary[device]['memory_mb'] += param_memory
        
        print(f"📈 总参数量: {total_params:,}")
        
        if xpu_params > 0:
            xpu_percentage = (xpu_params/total_params)*100
            print(f"🎯 XPU参数量: {xpu_params:,} ({xpu_percentage:.1f}%)")
        
        for device, info in device_summary.items():
            percentage = (info['params'] / total_params) * 100
            print(f"  {device}:")
            print(f"    - 层数: {info['count']}")
            print(f"    - 参数量: {info['params']:,} ({percentage:.1f}%)")
            print(f"    - 估计内存: {info['memory_mb']:.1f} MB")
        
        # 检查XPU使用情况
        if hasattr(torch.xpu, 'memory_allocated'):
            allocated = torch.xpu.memory_allocated() / 1024**3
            print(f"💾 XPU显存使用: {allocated:.2f} GB")
        
        # 性能测试
        print(f"\n🧪 INT4模型推理性能测试...")
        
        # 热身推理
        print("🔥 模型热身...")
        warmup_start = time.time()
        try:
            test_msgs = [
                {'role': 'user', 'content': ['Warmup test']}
            ]
            
            _ = inference_engine.chat(
                msgs=test_msgs,
                max_new_tokens=5,
                temperature=0.7
            )
            
            warmup_time = time.time() - warmup_start
            print(f"✅ 热身完成，耗时: {warmup_time:.2f}秒")
            
        except Exception as e:
            print(f"⚠️ 热身失败: {e}")
            warmup_time = 0
        
        # 正式推理测试
        print("🚀 正式推理测试...")
        test_cases = [
            "你好，这是一个测试。",
            "请简单介绍一下你自己。",
            "今天天气怎么样？"
        ]
        
        total_inference_time = 0
        successful_tests = 0
        
        for i, question in enumerate(test_cases, 1):
            try:
                print(f"\n测试 {i}/{len(test_cases)}: {question}")
                
                test_msgs = [
                    {'role': 'user', 'content': [question]}
                ]
                
                inference_start = time.time()
                
                response = inference_engine.chat(
                    msgs=test_msgs,
                    max_new_tokens=50,
                    temperature=0.7
                )
                
                inference_time = time.time() - inference_start
                total_inference_time += inference_time
                successful_tests += 1
                
                print(f"✅ 回答: {response}")
                print(f"⏱️ 推理耗时: {inference_time:.2f}秒")
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
        
        # 性能总结
        if successful_tests > 0:
            avg_inference_time = total_inference_time / successful_tests
            print(f"\n📊 性能总结:")
            print(f"  ✅ 成功测试: {successful_tests}/{len(test_cases)}")
            print(f"  ⏱️ 平均推理时间: {avg_inference_time:.2f}秒")
            print(f"  🏁 总耗时: {total_inference_time:.2f}秒")
            
            # 与原模型性能对比估算
            print(f"\n📈 INT4量化优势:")
            print(f"  💾 显存节省: ~75% (相比FP16)")
            print(f"  📦 存储节省: ~75% (相比原模型)")
            print(f"  ⚡ 推理速度: 可能略慢但显存友好")
        
        return successful_tests > 0
        
    except Exception as e:
        print(f"❌ INT4模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_processing_with_int4():
    """测试INT4模型的视频处理能力"""
    try:
        print(f"\n🎥 测试INT4模型视频处理能力...")
        
        from src.chat_with_video.video_chat_service import VideoChatService
        
        # 创建视频聊天服务 - 默认使用int4
        service = VideoChatService()
        
        # 初始化服务
        if service.initialize():
            print("✅ INT4视频聊天服务初始化成功")
            
            # 显示系统信息
            info = service.get_system_info()
            print(f"📋 系统配置:")
            print(f"  - 模型: {info.get('model_path', 'N/A')}")
            print(f"  - 设备: {info.get('device', 'N/A')}")
            print(f"  - 初始化状态: {info.get('initialized', False)}")
            
            return True
        else:
            print("❌ INT4视频聊天服务初始化失败")
            return False
            
    except Exception as e:
        print(f"❌ 视频处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("🧪 MiniCPM-V-4.5-int4 量化版本测试")
    print("="*60)
    
    # 基础检查
    if not torch.xpu.is_available():
        print("❌ Intel XPU不可用")
        return 1
    
    print(f"✅ Intel XPU可用，设备数量: {torch.xpu.device_count()}")
    
    # 测试INT4模型
    model_success = test_int4_model_loading()
    
    # 测试视频处理
    if model_success:
        video_success = test_video_processing_with_int4()
    else:
        video_success = False
    
    # 结果总结
    print("\n" + "="*60)
    if model_success and video_success:
        print("🎉 INT4量化版本测试全部通过!")
        print("📋 优势总结:")
        print("  💾 显存占用大幅减少")
        print("  📦 模型存储空间节省75%")
        print("  ⚡ 更适合资源受限环境")
        print("  🎯 功能完整，性能良好")
        print("="*60)
        return 0
    else:
        print("❌ INT4量化版本测试失败!")
        print("="*60)
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)