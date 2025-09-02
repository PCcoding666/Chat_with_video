#!/usr/bin/env python3
"""
测试混合推理策略
针对Intel XPU显存不足的问题，使用CPU+XPU混合推理
"""

import os
import torch

# 设置环境变量
os.environ['DISABLE_TRITON'] = '1'
os.environ['USE_TRITON'] = '0'
os.environ['TRITON_DISABLE'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_DISABLE_TRITON'] = '1'

def test_mixed_inference_strategy():
    """测试混合推理策略"""
    try:
        print("🔧 测试CPU+XPU混合推理策略...")
        
        # 先清理XPU缓存
        if hasattr(torch.xpu, 'empty_cache'):
            torch.xpu.empty_cache()
            print("✅ XPU缓存已清理")
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        # 检查当前显存使用情况
        if hasattr(torch.xpu, 'memory_allocated'):
            before_alloc = torch.xpu.memory_allocated() / 1024**3
            print(f"📊 清理前XPU显存使用: {before_alloc:.2f} GB")
        
        # 导入模型加载器
        from src.chat_with_video.model_loader import MiniCPMVInference
        
        # 创建推理引擎，但还不初始化
        print("📦 创建推理引擎实例...")
        inference_engine = MiniCPMVInference(device='xpu')
        
        print("🔧 手动配置混合推理模式...")
        
        # 手动配置混合推理
        from transformers import AutoModel, AutoTokenizer
        
        # 加载模型到CPU，然后只移动部分层到XPU
        print("📥 在CPU上加载模型...")
        model = AutoModel.from_pretrained(
            "openbmb/MiniCPM-V-4_5",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"}  # 强制CPU
        )
        
        print("✅ 模型已加载到CPU")
        
        # 检查模型结构，选择性地移动层到XPU
        print("🔍 分析模型结构...")
        
        # 统计模型层信息
        total_layers = 0
        cpu_layers = 0
        xpu_layers = 0
        
        # 尝试移动部分层到XPU（比如embedding层和前几层transformer层）
        try:
            # 首先尝试移动vision相关组件到XPU
            if hasattr(model, 'vision_tower'):
                print("🎯 尝试移动vision_tower到XPU...")
                try:
                    model.vision_tower = model.vision_tower.to('xpu')
                    print("✅ vision_tower已移动到XPU")
                    xpu_layers += 1
                except Exception as e:
                    print(f"⚠️ vision_tower移动失败: {e}")
            
            # 尝试移动embedding层
            if hasattr(model, 'embed_tokens'):
                print("🎯 尝试移动embed_tokens到XPU...")
                try:
                    model.embed_tokens = model.embed_tokens.to('xpu')
                    print("✅ embed_tokens已移动到XPU")
                    xpu_layers += 1
                except Exception as e:
                    print(f"⚠️ embed_tokens移动失败: {e}")
            
            # 尝试移动前几层transformer层到XPU
            if hasattr(model, 'layers'):
                max_xpu_layers = min(4, len(model.layers))  # 最多移动4层到XPU
                print(f"🎯 尝试移动前{max_xpu_layers}层transformer到XPU...")
                
                for i in range(max_xpu_layers):
                    try:
                        model.layers[i] = model.layers[i].to('xpu')
                        print(f"✅ 第{i}层已移动到XPU")
                        xpu_layers += 1
                    except Exception as e:
                        print(f"⚠️ 第{i}层移动失败: {e}")
                        break  # 如果某层失败，停止移动后续层
                
                cpu_layers = len(model.layers) - xpu_layers
                
        except Exception as e:
            print(f"⚠️ 混合配置失败: {e}")
        
        # 设置为评估模式
        model = model.eval()
        
        # 分配给推理引擎
        inference_engine.model = model
        inference_engine._initialized = True
        
        # 显示最终配置
        print(f"\n📊 混合推理配置结果:")
        print(f"  🔵 XPU层数: {xpu_layers}")
        print(f"  🔴 CPU层数: {cpu_layers}")
        
        # 检查设备分布
        device_summary = {}
        total_params = 0
        
        for name, param in model.named_parameters():
            device = str(param.device)
            param_count = param.numel()
            total_params += param_count
            
            if device not in device_summary:
                device_summary[device] = {'count': 0, 'params': 0}
            
            device_summary[device]['count'] += 1
            device_summary[device]['params'] += param_count
        
        print(f"\n📈 设备参数分布:")
        print(f"  📊 总参数量: {total_params:,}")
        
        for device, info in device_summary.items():
            percentage = (info['params'] / total_params) * 100
            print(f"  {device}: {info['count']} 层, {info['params']:,} 参数 ({percentage:.1f}%)")
        
        # 检查XPU使用情况
        if hasattr(torch.xpu, 'memory_allocated'):
            after_alloc = torch.xpu.memory_allocated() / 1024**3
            print(f"💾 配置后XPU显存使用: {after_alloc:.2f} GB")
        
        # 测试推理
        print(f"\n🧪 测试混合推理...")
        try:
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "openbmb/MiniCPM-V-4_5",
                trust_remote_code=True
            )
            
            inference_engine.tokenizer = tokenizer
            
            # 简单推理测试
            test_msgs = [
                {'role': 'user', 'content': ['Hello, this is a test.']}
            ]
            
            response = inference_engine.chat(
                msgs=test_msgs,
                max_new_tokens=10,
                temperature=0.7
            )
            
            print(f"✅ 混合推理测试成功!")
            print(f"📝 回答: {response}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ 推理测试失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 混合推理策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("🧪 CPU+XPU 混合推理策略测试")
    print("="*60)
    
    success = test_mixed_inference_strategy()
    
    if success:
        print("\n" + "="*60)
        print("🎉 混合推理策略测试成功!")
        print("📋 建议: 使用CPU+XPU混合模式以优化显存使用")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("❌ 混合推理策略测试失败!")
        print("="*60)
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)